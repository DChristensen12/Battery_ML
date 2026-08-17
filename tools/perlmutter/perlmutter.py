#!/usr/bin/env python3
"""Batch-run SolvationNet scripts on NERSC Perlmutter via SLURM job arrays.

This tool adds campaign-level orchestration on top of Perlmutter/SLURM:
batch submission of 80+ boxes in one command, persistent tracking across
sessions, auto-retry of failed jobs from checkpoints, and pipeline chaining
(equilibrate → run_md → analyze) via SLURM dependencies.

Prerequisites:
    1. NERSC account with a compute allocation
    2. SSH keys via sshproxy.sh (https://docs.nersc.gov/connect/mfa/)
    3. Run `perlmutter.py setup` once to configure

Quickstart:
    # One-time setup
    perlmutter.py setup
    perlmutter.py sync --install

    # Submit all boxes for equilibration
    perlmutter.py campaign create ds3_uma_npt \\
        --workflow equilibrate \\
        --inputs "data/DS3_TO_RUN/*.xyz" \\
        -- --model uma --workflow npt --npt-steps 100000

    # Check progress
    perlmutter.py campaign status ds3_uma_npt

    # Retry failures
    perlmutter.py campaign retry ds3_uma_npt

    # Download results
    perlmutter.py pull ds3_uma_npt
"""

import argparse
import glob
import json
import os
import pathlib
import shlex
import subprocess
import sys
import textwrap

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / ".perlmutter.json"
TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent / "templates"

PERLMUTTER_HOST = "perlmutter.nersc.gov"

KNOWN_WORKFLOWS = {
    "equilibrate": {
        "template": "equilibrate_array.sbatch",
        "description": "NVT/NPT equilibration (electrolyte_toolkit)",
        "gpu": True,
    },
    "run_md": {
        "template": "run_md_array.sbatch",
        "description": "Production MD (electrolyte_toolkit)",
        "gpu": True,
    },
    "metadynamics": {
        "template": "metadynamics.sbatch",
        "description": "Well-tempered metadynamics (deep-shell)",
        "gpu": True,
    },
    "analyze": {
        "template": "analyze_array.sbatch",
        "description": "Trajectory analysis (CPU-only)",
        "gpu": False,
    },
}


# ── config ─────────────────────────────────────────────────────────────

def load_config():
    if not CONFIG_PATH.exists():
        print("Not configured. Run: perlmutter.py setup", file=sys.stderr)
        sys.exit(1)
    return json.loads(CONFIG_PATH.read_text())


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")


# ── SSH / rsync helpers ────────────────────────────────────────────────

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]


def ssh_target(cfg):
    return f"{cfg['username']}@{PERLMUTTER_HOST}"


def run_ssh(cfg, command, capture=False, check=True):
    cmd = ["ssh"] + SSH_OPTS + [ssh_target(cfg), command]
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return r.stdout.strip()
    return subprocess.run(cmd, check=check).returncode


def run_rsync(cfg, local_path, remote_path, download=False, extra_flags=None):
    target = ssh_target(cfg)
    flags = ["-avz", "--delete", "-e", f"ssh {' '.join(SSH_OPTS)}"]
    if extra_flags:
        flags.extend(extra_flags)
    if download:
        src, dst = f"{target}:{remote_path}", str(local_path)
    else:
        src, dst = str(local_path), f"{target}:{remote_path}"
    return subprocess.run(["rsync"] + flags + [src, dst], check=True).returncode


def run_scp(cfg, local_path, remote_path, download=False):
    target = ssh_target(cfg)
    if download:
        src, dst = f"{target}:{remote_path}", str(local_path)
    else:
        src, dst = str(local_path), f"{target}:{remote_path}"
    return subprocess.run(["scp"] + SSH_OPTS + ["-r", src, dst], check=True).returncode


def remote_dir(cfg):
    return f"{cfg['scratch']}/solvationnet"


# ── setup ──────────────────────────────────────────────────────────────

def cmd_setup(args):
    print("Perlmutter setup for SolvationNet")
    print("=" * 40)

    username = input("NERSC username: ").strip()
    project = input("Allocation/project (e.g. m1234): ").strip()

    print(f"\nTesting SSH to {username}@{PERLMUTTER_HOST}...")
    try:
        scratch = run_ssh({"username": username}, "echo $SCRATCH", capture=True)
    except subprocess.CalledProcessError:
        print(
            "\nSSH failed. Make sure your keys are set up.\n"
            "See: https://docs.nersc.gov/connect/mfa/\n"
            "Run sshproxy.sh to get a 24h SSH certificate.",
            file=sys.stderr,
        )
        return 1

    if not scratch:
        scratch = f"/pscratch/sd/{username[0]}/{username}"
        print(f"Could not detect $SCRATCH, using default: {scratch}")
    else:
        print(f"Detected $SCRATCH: {scratch}")

    conda_env = input("Conda env name on Perlmutter [solvationnet]: ").strip() or "solvationnet"

    cfg = {
        "username": username,
        "project": project,
        "scratch": scratch,
        "conda_env": conda_env,
    }
    save_config(cfg)

    rd = remote_dir(cfg)
    run_ssh(cfg, f"mkdir -p {rd}/inputs {rd}/checkpoints {rd}/projects {rd}/campaigns")
    print(f"\nConfig saved to {CONFIG_PATH.relative_to(REPO_ROOT)}")
    print(f"Remote workspace: {rd}")
    print("\nNext: perlmutter.py sync --install")
    return 0


# ── sync ───────────────────────────────────────────────────────────────

def cmd_sync(args):
    cfg = load_config()
    rd = remote_dir(cfg)

    print("Syncing toolkit code to Perlmutter...")
    run_ssh(cfg, f"mkdir -p {rd}/electrolyte_toolkit {rd}/deep-shell {rd}/templates")

    et_dir = str(REPO_ROOT / "electrolyte_toolkit") + "/"
    ds_dir = str(REPO_ROOT / "deep-shell") + "/"
    tmpl_dir = str(TEMPLATES_DIR) + "/"
    setup_script = pathlib.Path(__file__).resolve().parent / "setup_env.sh"

    run_rsync(cfg, et_dir, f"{rd}/electrolyte_toolkit/")
    run_rsync(cfg, ds_dir, f"{rd}/deep-shell/")
    run_rsync(cfg, tmpl_dir, f"{rd}/templates/", extra_flags=["--exclude=__pycache__"])
    if setup_script.exists():
        run_scp(cfg, setup_script, f"{rd}/setup_env.sh")

    print("Code synced.")
    if args.install:
        print("Running setup_env.sh (this may take a few minutes)...")
        run_ssh(cfg, f"bash {rd}/setup_env.sh {cfg['conda_env']}")
    return 0


# ── campaign create ────────────────────────────────────────────────────

def cmd_campaign_create(args):
    from campaign import Campaign

    cfg = load_config()
    name = args.name
    workflow = args.workflow

    if workflow not in KNOWN_WORKFLOWS:
        print(f"Unknown workflow: {workflow}. Available: {', '.join(KNOWN_WORKFLOWS)}", file=sys.stderr)
        return 1

    input_patterns = args.inputs
    input_files = []
    for pattern in input_patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"No files match: {pattern}", file=sys.stderr)
            return 1
        input_files.extend(matched)

    if not input_files:
        print("No input files found.", file=sys.stderr)
        return 1

    extra_args = args.extra_args

    slurm_args = {
        "qos": args.qos,
        "time": args.time,
        "gpus": args.gpus,
        "max_concurrent": args.max_concurrent,
    }

    print(f"Creating campaign '{name}': {workflow} on {len(input_files)} inputs")
    campaign = Campaign.create(name, workflow, input_files, extra_args, slurm_args)

    if args.dry_run:
        print(f"\n[DRY RUN] Would submit {len(input_files)} tasks")
        print(f"Workflow args: {extra_args}")
        print(f"SLURM: qos={args.qos} time={args.time} gpus={args.gpus} max_concurrent={args.max_concurrent}")
        print(f"\nFirst 5 inputs:")
        for f in input_files[:5]:
            print(f"  {pathlib.Path(f).name}")
        if len(input_files) > 5:
            print(f"  ... and {len(input_files) - 5} more")
        return 0

    rd = remote_dir(cfg)

    print("Syncing input files...")
    run_ssh(cfg, f"mkdir -p {rd}/inputs")
    for f in input_files:
        run_scp(cfg, f, f"{rd}/inputs/{pathlib.Path(f).name}")

    campaign_dir = f"{rd}/campaigns/{name}"
    run_ssh(cfg, f"mkdir -p {campaign_dir}")

    manifest_lines = "\n".join(pathlib.Path(f).name for f in input_files)
    manifest_path = f"{campaign_dir}/manifest.txt"
    run_ssh(cfg, f"cat > {manifest_path} << 'MANIFEST_EOF'\n{manifest_lines}\nMANIFEST_EOF")

    template_file = TEMPLATES_DIR / KNOWN_WORKFLOWS[workflow]["template"]
    template = template_file.read_text()
    script = template.format(
        CAMPAIGN=name,
        PROJECT=cfg["project"],
        CONDA_ENV=cfg["conda_env"],
        QOS=args.qos,
        TIME=args.time,
        GPU_COUNT=args.gpus,
        N_TASKS=len(input_files),
        MAX_CONCURRENT=args.max_concurrent,
        PROJECT_DIR=rd,
        CAMPAIGN_DIR=campaign_dir,
        MANIFEST_PATH=manifest_path,
        EXTRA_ARGS=extra_args,
    )

    script_path = f"{campaign_dir}/job.sbatch"
    run_ssh(cfg, f"cat > {script_path} << 'SBATCH_EOF'\n{script}\nSBATCH_EOF")

    print(f"Submitting SLURM job array (1-{len(input_files)}, max {args.max_concurrent} concurrent)...")
    output = run_ssh(cfg, f"sbatch {script_path}", capture=True)
    print(output)

    array_job_id = None
    for word in output.split():
        if word.isdigit():
            array_job_id = word
            break

    if array_job_id:
        campaign.record_submission(array_job_id)
        print(f"\nCampaign '{name}' submitted: job array {array_job_id}")
        print(f"Track with: perlmutter.py campaign status {name}")
    else:
        print("Warning: could not parse job ID from sbatch output.", file=sys.stderr)

    return 0


# ── campaign status ────────────────────────────────────────────────────

def cmd_campaign_status(args):
    from campaign import Campaign

    cfg = load_config()
    name = args.name

    if name:
        try:
            campaign = Campaign.load(name)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            return 1
        campaign.refresh_status(lambda cmd, **kw: run_ssh(cfg, cmd, **kw))
        print(campaign.format_dashboard())
    else:
        all_campaigns = Campaign.list_all()
        if not all_campaigns:
            print("No campaigns. Create one with: perlmutter.py campaign create <name> ...")
            return 0
        for cname, cdata in all_campaigns.items():
            c = Campaign(cname, cdata)
            c.refresh_status(lambda cmd, **kw: run_ssh(cfg, cmd, **kw))
            print(c.format_dashboard())
            print()
    return 0


# ── campaign retry ─────────────────────────────────────────────────────

def cmd_campaign_retry(args):
    from campaign import Campaign

    cfg = load_config()
    try:
        campaign = Campaign.load(args.name)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 1

    campaign.refresh_status(lambda cmd, **kw: run_ssh(cfg, cmd, **kw))
    retryable = campaign.retryable_tasks(include_nan=args.include_nan)

    if not retryable:
        print("No retryable failures.")
        return 0

    retry_indices = campaign.retryable_indices(include_nan=args.include_nan)
    print(f"Retrying {len(retryable)} failed tasks:")
    for t in retryable:
        print(f"  {t['input']:<45s} {t['status']} (attempt {t['submit_count'] + 1})")

    if args.dry_run:
        print("\n[DRY RUN] Would resubmit the above tasks.")
        return 0

    rd = remote_dir(cfg)
    campaign_dir = f"{rd}/campaigns/{args.name}"

    retry_manifest = "\n".join(t["input"] for t in retryable)
    retry_manifest_path = f"{campaign_dir}/retry_manifest.txt"
    run_ssh(cfg, f"cat > {retry_manifest_path} << 'MANIFEST_EOF'\n{retry_manifest}\nMANIFEST_EOF")

    workflow = campaign.workflow
    template_file = TEMPLATES_DIR / KNOWN_WORKFLOWS[workflow]["template"]
    template = template_file.read_text()

    sa = campaign.slurm_args
    time = args.time or sa.get("time", "04:00:00")
    qos = args.qos or sa.get("qos", "regular")
    gpus = sa.get("gpus", 1)
    max_concurrent = sa.get("max_concurrent", 16)

    script = template.format(
        CAMPAIGN=f"{args.name}_retry",
        PROJECT=cfg["project"],
        CONDA_ENV=cfg["conda_env"],
        QOS=qos,
        TIME=time,
        GPU_COUNT=gpus,
        N_TASKS=len(retryable),
        MAX_CONCURRENT=max_concurrent,
        PROJECT_DIR=rd,
        CAMPAIGN_DIR=campaign_dir,
        MANIFEST_PATH=retry_manifest_path,
        EXTRA_ARGS=campaign.workflow_args,
    )

    script_path = f"{campaign_dir}/retry_job.sbatch"
    run_ssh(cfg, f"cat > {script_path} << 'SBATCH_EOF'\n{script}\nSBATCH_EOF")

    output = run_ssh(cfg, f"sbatch {script_path}", capture=True)
    print(output)

    array_job_id = None
    for word in output.split():
        if word.isdigit():
            array_job_id = word
            break

    if array_job_id:
        campaign.record_submission(array_job_id, task_indices=retry_indices)
        print(f"Retry submitted: job array {array_job_id}")

    return 0


# ── campaign list ──────────────────────────────────────────────────────

def cmd_campaign_list(args):
    from campaign import Campaign

    all_campaigns = Campaign.list_all()
    if not all_campaigns:
        print("No campaigns.")
        return 0

    for name, data in all_campaigns.items():
        c = Campaign(name, data)
        counts = c.summary()
        total = len(c.tasks)
        done = counts.get("done", 0)
        failed = sum(counts.get(s, 0) for s in ("failed", "oom", "timeout", "nan"))
        print(f"  {name:<30s} {c.workflow:<15s} {done}/{total} done  {failed} failed  {data.get('created', '?')}")
    return 0


# ── campaign pipeline ──────────────────────────────────────────────────

def cmd_campaign_pipeline(args):
    from campaign import Campaign

    cfg = load_config()
    name = args.name

    input_patterns = args.inputs
    input_files = []
    for pattern in input_patterns:
        input_files.extend(sorted(glob.glob(pattern)))
    if not input_files:
        print("No input files found.", file=sys.stderr)
        return 1

    steps = []
    for step_str in args.steps:
        parts = step_str.split(":", 1)
        wf = parts[0]
        wf_args = parts[1] if len(parts) > 1 else ""
        if wf not in KNOWN_WORKFLOWS:
            print(f"Unknown workflow in pipeline: {wf}", file=sys.stderr)
            return 1
        steps.append({"workflow": wf, "args": wf_args})

    if not steps:
        print("No pipeline steps specified.", file=sys.stderr)
        return 1

    print(f"Pipeline '{name}': {len(steps)} steps, {len(input_files)} inputs")
    for i, s in enumerate(steps):
        print(f"  Step {i+1}: {s['workflow']} {s['args']}")

    if args.dry_run:
        print("\n[DRY RUN] Would submit the above pipeline.")
        return 0

    rd = remote_dir(cfg)

    print("Syncing input files...")
    run_ssh(cfg, f"mkdir -p {rd}/inputs")
    for f in input_files:
        run_scp(cfg, f, f"{rd}/inputs/{pathlib.Path(f).name}")

    manifest_lines = "\n".join(pathlib.Path(f).name for f in input_files)
    pipeline_dir = f"{rd}/campaigns/{name}"
    run_ssh(cfg, f"mkdir -p {pipeline_dir}")
    manifest_path = f"{pipeline_dir}/manifest.txt"
    run_ssh(cfg, f"cat > {manifest_path} << 'MANIFEST_EOF'\n{manifest_lines}\nMANIFEST_EOF")

    prev_job_id = None
    step_job_ids = []

    for i, step in enumerate(steps):
        wf = step["workflow"]
        wf_args = step["args"]
        template_file = TEMPLATES_DIR / KNOWN_WORKFLOWS[wf]["template"]
        template = template_file.read_text()

        step_name = f"{name}_step{i+1}_{wf}"
        script = template.format(
            CAMPAIGN=step_name,
            PROJECT=cfg["project"],
            CONDA_ENV=cfg["conda_env"],
            QOS=args.qos,
            TIME=args.time,
            GPU_COUNT=args.gpus,
            N_TASKS=len(input_files),
            MAX_CONCURRENT=args.max_concurrent,
            PROJECT_DIR=rd,
            CAMPAIGN_DIR=pipeline_dir,
            MANIFEST_PATH=manifest_path,
            EXTRA_ARGS=wf_args,
        )

        script_path = f"{pipeline_dir}/step{i+1}_{wf}.sbatch"
        run_ssh(cfg, f"cat > {script_path} << 'SBATCH_EOF'\n{script}\nSBATCH_EOF")

        sbatch_cmd = f"sbatch"
        if prev_job_id:
            sbatch_cmd += f" --dependency=aftercorr:{prev_job_id}"
        sbatch_cmd += f" {script_path}"

        output = run_ssh(cfg, sbatch_cmd, capture=True)
        print(f"  Step {i+1} ({wf}): {output}")

        job_id = None
        for word in output.split():
            if word.isdigit():
                job_id = word
                break
        step_job_ids.append(job_id)
        prev_job_id = job_id

    wf_args_combined = " | ".join(f"{s['workflow']}:{s['args']}" for s in steps)
    campaign = Campaign.create(name, "pipeline", input_files, wf_args_combined, {
        "qos": args.qos, "time": args.time, "gpus": args.gpus,
        "max_concurrent": args.max_concurrent,
    })
    campaign.data["pipeline_steps"] = steps
    campaign.data["pipeline_job_ids"] = step_job_ids
    if step_job_ids and step_job_ids[-1]:
        campaign.record_submission(step_job_ids[-1])
    campaign.save()

    print(f"\nPipeline '{name}' submitted.")
    return 0


# ── pull ───────────────────────────────────────────────────────────────

def cmd_pull(args):
    from campaign import Campaign

    cfg = load_config()
    try:
        campaign = Campaign.load(args.campaign)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 1

    campaign.refresh_status(lambda cmd, **kw: run_ssh(cfg, cmd, **kw))
    done = campaign.done_tasks()

    if not done:
        print("No completed tasks to pull.")
        return 0

    local_dest = pathlib.Path(args.output) if args.output else campaign.auto_output_dir()
    local_dest.mkdir(parents=True, exist_ok=True)

    rd = remote_dir(cfg)
    ckpt_dir = f"{rd}/checkpoints"

    print(f"Downloading {len(done)} completed results to {local_dest}")
    downloaded = 0
    for t in done:
        box_name = t["box_name"]
        remote_xyz = f"{ckpt_dir}/{box_name}/{box_name}.xyz"
        local_xyz = local_dest / f"{box_name}.xyz"
        if local_xyz.exists() and not args.force:
            continue
        try:
            run_scp(cfg, local_xyz, remote_xyz, download=True)
            downloaded += 1
        except subprocess.CalledProcessError:
            print(f"  Warning: could not download {box_name}.xyz", file=sys.stderr)

    if args.with_diagnostics:
        diag_dir = local_dest / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        for t in done:
            box_name = t["box_name"]
            remote_png = f"{ckpt_dir}/{box_name}/diagnostics.png"
            local_png = diag_dir / f"{box_name}_diagnostics.png"
            if local_png.exists() and not args.force:
                continue
            try:
                run_scp(cfg, local_png, remote_png, download=True)
            except subprocess.CalledProcessError:
                pass

    print(f"Downloaded {downloaded} new files. Total done: {len(done)}/{len(campaign.tasks)}")
    return 0


# ── tail ───────────────────────────────────────────────────────────────

def cmd_tail(args):
    cfg = load_config()
    rd = remote_dir(cfg)
    campaign_name = args.campaign
    task_id = args.task_id

    log_path = f"{rd}/campaigns/{campaign_name}/{campaign_name}_{task_id}.out"
    try:
        run_ssh(cfg, f"tail -n 50 -f {log_path}")
    except KeyboardInterrupt:
        pass
    return 0


# ── checkpoints ────────────────────────────────────────────────────────

def cmd_checkpoints(args):
    cfg = load_config()
    rd = remote_dir(cfg)
    ckpt_dir = f"{rd}/checkpoints"

    if args.campaign:
        from campaign import Campaign
        try:
            campaign = Campaign.load(args.campaign)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            return 1
        box_names = [t["box_name"] for t in campaign.tasks]
        filter_cmd = " ".join(shlex.quote(b) for b in box_names)
        run_ssh(cfg, f"cd {rd}/electrolyte_toolkit && python equilibrate.py --status --checkpoint-dir {ckpt_dir}")
    else:
        run_ssh(cfg, f"cd {rd}/electrolyte_toolkit && python equilibrate.py --status --checkpoint-dir {ckpt_dir}")
    return 0


# ── ssh / ls ───────────────────────────────────────────────────────────

def cmd_ssh(args):
    cfg = load_config()
    if args.command:
        return run_ssh(cfg, args.command)
    os.execvp("ssh", ["ssh"] + SSH_OPTS + [ssh_target(cfg)])


def cmd_ls(args):
    cfg = load_config()
    target = args.path or remote_dir(cfg)
    output = run_ssh(cfg, f"ls -lh {target}", capture=True)
    print(output)
    return 0


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-run SolvationNet on NERSC Perlmutter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # setup
    sub.add_parser("setup", help="configure NERSC credentials").set_defaults(func=cmd_setup)

    # sync
    p = sub.add_parser("sync", help="push toolkit code to Perlmutter (rsync)")
    p.add_argument("--install", action="store_true", help="also run setup_env.sh")
    p.set_defaults(func=cmd_sync)

    # campaign
    campaign_parser = sub.add_parser("campaign", help="batch job management")
    campaign_sub = campaign_parser.add_subparsers(dest="campaign_cmd", required=True)

    # campaign create
    p = campaign_sub.add_parser("create", help="create and submit a batch campaign")
    p.add_argument("name", help="campaign name (e.g. ds3_uma_npt)")
    p.add_argument("--workflow", required=True, choices=list(KNOWN_WORKFLOWS),
                    help="which script to run")
    p.add_argument("--inputs", nargs="+", required=True,
                    help="input file glob patterns (e.g. 'data/DS3_TO_RUN/*.xyz')")
    p.add_argument("--qos", default="regular", choices=["regular", "debug", "preempt"])
    p.add_argument("--time", default="04:00:00", help="wall time per task (default: 04:00:00)")
    p.add_argument("--gpus", type=int, default=1, help="GPUs per task (default: 1)")
    p.add_argument("--max-concurrent", type=int, default=16,
                    help="max simultaneous array tasks (default: 16)")
    p.add_argument("--dry-run", action="store_true", help="show what would be submitted")
    p.add_argument("--args", dest="extra_args", default="",
                    help="workflow args as a quoted string (e.g. '--model uma --workflow npt')")
    p.set_defaults(func=cmd_campaign_create)

    # campaign status
    p = campaign_sub.add_parser("status", help="show campaign dashboard")
    p.add_argument("name", nargs="?", help="campaign name (omit for all)")
    p.set_defaults(func=cmd_campaign_status)

    # campaign retry
    p = campaign_sub.add_parser("retry", help="resubmit failed tasks")
    p.add_argument("name", help="campaign name")
    p.add_argument("--include-nan", action="store_true",
                    help="also retry NaN failures (usually not useful)")
    p.add_argument("--time", default=None, help="override wall time for retries")
    p.add_argument("--qos", default=None, help="override QOS for retries")
    p.add_argument("--dry-run", action="store_true", help="show what would be retried")
    p.set_defaults(func=cmd_campaign_retry)

    # campaign list
    campaign_sub.add_parser("list", help="list all campaigns").set_defaults(func=cmd_campaign_list)

    # campaign pipeline
    p = campaign_sub.add_parser("pipeline", help="submit a multi-step pipeline")
    p.add_argument("name", help="pipeline name")
    p.add_argument("--inputs", nargs="+", required=True,
                    help="input file glob patterns")
    p.add_argument("--steps", nargs="+", required=True,
                    help="pipeline steps as 'workflow:args' (e.g. 'equilibrate:--model uma')")
    p.add_argument("--qos", default="regular", choices=["regular", "debug", "preempt"])
    p.add_argument("--time", default="04:00:00", help="wall time per step per task")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--max-concurrent", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_campaign_pipeline)

    # pull
    p = sub.add_parser("pull", help="download results from a campaign")
    p.add_argument("campaign", help="campaign name")
    p.add_argument("-o", "--output", help="local output directory (auto-derived if omitted)")
    p.add_argument("--with-diagnostics", action="store_true",
                    help="also download diagnostics.png files")
    p.add_argument("--force", action="store_true", help="overwrite existing local files")
    p.set_defaults(func=cmd_pull)

    # tail
    p = sub.add_parser("tail", help="tail a running task's SLURM output")
    p.add_argument("campaign", help="campaign name")
    p.add_argument("task_id", help="SLURM array task ID (1-based)")
    p.set_defaults(func=cmd_tail)

    # checkpoints
    p = sub.add_parser("checkpoints", help="show equilibration checkpoint status on remote")
    p.add_argument("campaign", nargs="?", help="filter to a specific campaign")
    p.set_defaults(func=cmd_checkpoints)

    # ssh
    p = sub.add_parser("ssh", help="run a command on Perlmutter or open a shell")
    p.add_argument("command", nargs="?")
    p.set_defaults(func=cmd_ssh)

    # ls
    p = sub.add_parser("ls", help="list files on Perlmutter")
    p.add_argument("path", nargs="?")
    p.set_defaults(func=cmd_ls)

    args = parser.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
