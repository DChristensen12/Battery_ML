# Perlmutter Tools

Batch-run SolvationNet scripts on [NERSC Perlmutter](https://docs.nersc.gov/systems/perlmutter/) via SLURM job arrays with campaign-level tracking, auto-retry, and pipeline chaining.

## Why this exists

You already know how to SSH into Perlmutter and write sbatch scripts. This tool handles the parts that are tedious to do manually when you're running 80+ electrolyte boxes:

- **Batch submission** — one command submits all your boxes as a SLURM job array
- **Campaign tracking** — persistent state that remembers which boxes are done, failed, or still running, even across terminal sessions
- **Auto-retry** — detects OOM, timeout, and NaN failures; resubmits only the failures from their last checkpoint
- **Pipeline chaining** — equilibrate → run_md → analyze as a single command with SLURM dependency chains
- **Result aggregation** — downloads completed results into the correct local directory structure

## Prerequisites

1. A **NERSC account** with an active compute allocation (e.g. `m1234`)
2. **SSH keys** set up via [sshproxy.sh](https://docs.nersc.gov/connect/mfa/). Run it every 24 hours to refresh your SSH certificate:
   ```bash
   ./sshproxy.sh -u your_username
   ```
3. **Python 3.10+** on your local machine (no extra pip packages needed — the tool uses only the standard library)
4. **rsync** on your local machine (pre-installed on macOS and Linux)

## Quickstart

From the repo root:

```bash
# 1. One-time: configure your NERSC credentials
python tools/perlmutter/perlmutter.py setup

# 2. One-time: push code and create the conda env on Perlmutter
python tools/perlmutter/perlmutter.py sync --install

# 3. Submit all 83 boxes for NPT equilibration with UMA
python tools/perlmutter/perlmutter.py campaign create ds3_uma_npt \
    --workflow equilibrate \
    --inputs "data/DS3_TO_RUN/*.xyz" \
    --args "--model uma --workflow npt --npt-steps 100000"

# 4. Check progress (run any time, even days later)
python tools/perlmutter/perlmutter.py campaign status ds3_uma_npt

# 5. Retry any OOM/timeout failures
python tools/perlmutter/perlmutter.py campaign retry ds3_uma_npt

# 6. Download completed results
python tools/perlmutter/perlmutter.py pull ds3_uma_npt
```

## Setup

### `setup`

Interactive prompt for your NERSC username, allocation code, and conda env name. Tests SSH connectivity, detects your `$SCRATCH` path, and creates the remote workspace directory.

```bash
python tools/perlmutter/perlmutter.py setup
```

This saves your config to `.perlmutter.json` at the repo root (gitignored — no credentials are committed).

### `sync`

Pushes the latest `electrolyte_toolkit/` and `deep-shell/` code to Perlmutter using rsync (single SSH connection, delta transfer — much faster than copying files one by one).

```bash
python tools/perlmutter/perlmutter.py sync            # push code only
python tools/perlmutter/perlmutter.py sync --install   # also create/update the conda env
```

Run `sync` any time you modify the toolkit scripts locally. The `--install` flag runs `setup_env.sh` on Perlmutter to create a conda environment with PyTorch (CUDA), ASE, orb-models, fairchem-core, and mace-torch.

## Campaigns

A **campaign** is a named batch run. The tool tracks every input file, its SLURM job ID, and its status (pending, submitted, running, done, failed, oom, timeout, nan). Campaign state lives in `.perlmutter_campaigns.json` (gitignored) and persists across terminal sessions.

### `campaign create`

Create and submit a batch campaign. Each input file becomes one task in a SLURM job array.

```bash
python tools/perlmutter/perlmutter.py campaign create <name> \
    --workflow <workflow> \
    --inputs <glob> [<glob> ...] \
    --args "<workflow arguments>" \
    [--qos regular|debug|preempt] \
    [--time HH:MM:SS] \
    [--gpus N] \
    [--max-concurrent N] \
    [--dry-run]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--workflow` | (required) | `equilibrate`, `run_md`, `metadynamics`, or `analyze` |
| `--inputs` | (required) | One or more file glob patterns |
| `--args` | `""` | Workflow-specific arguments as a quoted string |
| `--qos` | `regular` | SLURM QOS (`regular` up to 12h, `debug` up to 30min, `preempt`) |
| `--time` | `04:00:00` | Wall time per task |
| `--gpus` | `1` | GPUs per task |
| `--max-concurrent` | `16` | Max simultaneous array tasks (the `%` throttle on `--array`) |
| `--dry-run` | off | Show what would be submitted without actually submitting |

**Examples:**

```bash
# Equilibrate with UMA, NPT-only
python tools/perlmutter/perlmutter.py campaign create uma_npt \
    --workflow equilibrate \
    --inputs "data/DS3_TO_RUN/*.xyz" \
    --args "--model uma --workflow npt --npt-steps 100000"

# Equilibrate with OrbMol-v2, NVT then NPT
python tools/perlmutter/perlmutter.py campaign create orb_nvt_npt \
    --workflow equilibrate \
    --inputs "data/DS3_TO_RUN/*.xyz" \
    --args "--model orbmol_v2 --workflow nvt+npt"

# Submit just a few boxes for testing (debug queue, 30 min)
python tools/perlmutter/perlmutter.py campaign create test_run \
    --workflow equilibrate \
    --inputs "data/DS3_TO_RUN/NaPF6_EC_1M.xyz" "data/DS3_TO_RUN/NaPF6_PC_1M.xyz" \
    --args "--model uma --workflow npt" \
    --qos debug --time 00:30:00

# Dry run to preview before committing
python tools/perlmutter/perlmutter.py campaign create ds3_check \
    --workflow equilibrate \
    --inputs "data/DS3_TO_RUN/*.xyz" \
    --args "--model uma --workflow npt" \
    --dry-run
```

### `campaign status`

Shows a dashboard with progress, running/failed counts, and failure details.

```bash
python tools/perlmutter/perlmutter.py campaign status <name>   # one campaign
python tools/perlmutter/perlmutter.py campaign status          # all campaigns
```

Example output:

```
Campaign: ds3_uma_npt (equilibrate)
Args:     --model uma --workflow npt --npt-steps 100000
Created:  2026-08-17T12:00:00

Progress: ██████████████████░░░░░░░░░░░░░░ 56/83 (67%)
  Done: 56   Running: 12   Submitted: 0   Failed: 3   Pending: 12

Failed:
  NaFSI_DME_1M.xyz                                oom        tries=1  retryable
  NaClO4_water_17M.xyz                             timeout    tries=1  retryable
  NaPF6_EC-DEC-9-1_1M.xyz                          nan        tries=1  NOT retryable
```

This queries SLURM's `sacct` on Perlmutter in a single SSH call and classifies each task.

### `campaign retry`

Resubmits failed tasks. Only retries OOM, timeout, and generic failures by default — NaN failures are excluded because they indicate a physics problem, not a resource issue.

```bash
python tools/perlmutter/perlmutter.py campaign retry <name> \
    [--time HH:MM:SS] \
    [--qos regular|debug|preempt] \
    [--include-nan] \
    [--dry-run]
```

The retried tasks automatically resume from their last checkpoint (the sbatch template detects existing checkpoints and passes `--resume` to `equilibrate.py`). You can increase the wall time for tasks that timed out:

```bash
# Retry with more time
python tools/perlmutter/perlmutter.py campaign retry ds3_uma_npt --time 08:00:00

# Preview what would be retried
python tools/perlmutter/perlmutter.py campaign retry ds3_uma_npt --dry-run
```

### `campaign list`

One-line summary of every campaign.

```bash
python tools/perlmutter/perlmutter.py campaign list
```

```
  ds3_uma_npt                    equilibrate     56/83 done  3 failed  2026-08-17T12:00:00
  orb_nvt_npt                    equilibrate     79/83 done  0 failed  2026-08-10T09:30:00
```

### `campaign pipeline`

Submits a multi-step workflow where each step runs only after its corresponding task in the previous step succeeds. Uses SLURM's `--dependency=aftercorr:` for per-task chaining (not per-array — task 42 of step 2 waits only for task 42 of step 1, not all 83).

```bash
python tools/perlmutter/perlmutter.py campaign pipeline <name> \
    --inputs <glob> \
    --steps "workflow1:args" "workflow2:args" ["workflow3:args" ...] \
    [--qos regular] [--time 04:00:00] [--gpus 1] [--max-concurrent 16] \
    [--dry-run]
```

**Example:** Full pipeline from equilibration through analysis:

```bash
python tools/perlmutter/perlmutter.py campaign pipeline ds3_full \
    --inputs "data/DS3_TO_RUN/*.xyz" \
    --steps \
        "equilibrate:--model uma --workflow npt --npt-steps 100000" \
        "run_md:-p npt -T 300 -P 1.0 -n 100000 --model uma" \
        "analyze:--protocol npt" \
    --time 06:00:00
```

This submits three chained SLURM job arrays. Each box flows through equilibrate → run_md → analyze independently. If equilibration fails for one box, its downstream steps are automatically skipped.

## Retrieving Results

### `pull`

Downloads completed results from a campaign to your local machine.

```bash
python tools/perlmutter/perlmutter.py pull <campaign> \
    [-o /custom/output/dir] \
    [--with-diagnostics] \
    [--force]
```

If you don't specify `-o`, the output directory is auto-derived from the campaign's workflow and model to match the existing data layout:

| Workflow + Model | Local output directory |
|---|---|
| `npt` + `uma` | `data/na-electrolyte-solvation-boxes/NPT_FAIRChem-UMA/` |
| `nvt` + `orbmol_v2` | `data/na-electrolyte-solvation-boxes/NVT_OrbMolV2/` |
| `nvt+npt` + `uma` | `data/na-electrolyte-solvation-boxes/NVT-NPT_FAIRChem-UMA/` |

This means `pull` drops results exactly where the repo expects them — ready for the dataset README and HuggingFace upload.

```bash
# Download final .xyz structures
python tools/perlmutter/perlmutter.py pull ds3_uma_npt

# Also grab the diagnostics plots
python tools/perlmutter/perlmutter.py pull ds3_uma_npt --with-diagnostics

# Force re-download of everything
python tools/perlmutter/perlmutter.py pull ds3_uma_npt --force
```

## Monitoring

### `tail`

Live-tail the SLURM output of a specific task in a campaign. Press Ctrl+C to stop.

```bash
python tools/perlmutter/perlmutter.py tail <campaign> <task_id>
```

Task IDs are 1-based (matching the SLURM array index). To find which task ID corresponds to which box, check the manifest: task 1 is the first file listed, task 2 is the second, etc.

```bash
# Watch task 42 of the ds3 campaign
python tools/perlmutter/perlmutter.py tail ds3_uma_npt 42
```

### `checkpoints`

Runs `equilibrate.py --status` on Perlmutter to show the checkpoint-level view — box name, equilibration phase, step count, density, and NaN flags.

```bash
python tools/perlmutter/perlmutter.py checkpoints              # all boxes
python tools/perlmutter/perlmutter.py checkpoints ds3_uma_npt  # one campaign
```

## Utility Commands

### `ssh`

Open an interactive shell on Perlmutter or run a one-off command:

```bash
python tools/perlmutter/perlmutter.py ssh                     # interactive shell
python tools/perlmutter/perlmutter.py ssh "squeue -u $USER"   # one-off command
python tools/perlmutter/perlmutter.py ssh "du -sh $SCRATCH/solvationnet/"
```

### `ls`

List files on Perlmutter. Defaults to the SolvationNet workspace.

```bash
python tools/perlmutter/perlmutter.py ls                        # project root
python tools/perlmutter/perlmutter.py ls checkpoints/            # checkpoint dirs
python tools/perlmutter/perlmutter.py ls campaigns/ds3_uma_npt/  # campaign logs
```

Paths are relative to `$SCRATCH/solvationnet/`.

## Remote Directory Layout

Everything lives under `$SCRATCH/solvationnet/` on Perlmutter:

```
$SCRATCH/solvationnet/
    electrolyte_toolkit/     # Synced Python scripts (from sync)
    deep-shell/              # Synced metadynamics scripts (from sync)
    templates/               # Synced SLURM templates
    setup_env.sh             # Conda env setup script
    inputs/                  # Uploaded .xyz input files
    checkpoints/             # Equilibration checkpoints and outputs
        NaPF6_PC_1M/
            ckpt_npt_50000.xyz
            ckpt_npt_50000.npz
            NaPF6_PC_1M.xyz       # Final equilibrated structure
            diagnostics.png
        NaFSI_DME_1M/
            ...
    projects/                # run_md project layouts (for pipelines)
    campaigns/               # Manifests, sbatch scripts, SLURM logs
        ds3_uma_npt/
            manifest.txt
            job.sbatch
            ds3_uma_npt_1.out     # SLURM output for task 1
            ds3_uma_npt_1.err
            ...
```

Checkpoints are stored on `$SCRATCH` (Lustre parallel filesystem, high quota) rather than `$HOME` (NFS, 40 GB quota). The templates override `equilibrate.py`'s default `~/electrolyte_equilibration` to `$SCRATCH/solvationnet/checkpoints/`.

## How SLURM Job Arrays Work

When you create a campaign with 83 inputs, the tool:

1. Writes a **manifest** file listing one input filename per line
2. Generates an **sbatch script** with `#SBATCH --array=1-83%16` (83 tasks, max 16 running at once)
3. Each task reads its input from the manifest: `sed -n "${SLURM_ARRAY_TASK_ID}p" manifest.txt`
4. The `%16` throttle prevents queue-hogging and allows early failure detection

This is one SLURM submission, not 83 individual `sbatch` calls. The scheduler handles it much more efficiently, and you get a single job ID to track.

## Failure Handling

The tool classifies failures by type and handles each appropriately:

| Status | Cause | Auto-retryable? | What happens on retry |
|--------|-------|------------------|-----------------------|
| `done` | Completed successfully | n/a | n/a |
| `oom` | Out of memory (exit 137) | Yes | Resumes from last checkpoint |
| `timeout` | Exceeded wall time | Yes | Resumes from last checkpoint |
| `failed` | Other nonzero exit | Yes | Resumes from last checkpoint |
| `nan` | NaN in final positions (exit 42) | No | Skipped (physics problem, not resource problem) |

**NaN detection:** The sbatch template runs a Python post-check after `equilibrate.py` finishes. If the final structure contains NaN positions, the task exits with code 42. The campaign tracker marks this as `nan` and excludes it from auto-retry because retrying from the same checkpoint will hit the same NaN. These need manual investigation (try a different model, check the input geometry, etc.).

**Checkpoint-aware resume:** When a task is retried, the sbatch template checks whether a checkpoint directory already exists. If so, it passes `--resume BOX_NAME` to `equilibrate.py` instead of starting fresh. This means OOM and timeout retries pick up exactly where they left off — no wasted compute.

## Best Practices

### Start small

Test with 2-3 boxes on the debug queue before submitting 83:

```bash
python tools/perlmutter/perlmutter.py campaign create test_2boxes \
    --workflow equilibrate \
    --inputs "data/DS3_TO_RUN/NaPF6_EC_1M.xyz" "data/DS3_TO_RUN/NaPF6_PC_1M.xyz" \
    --args "--model uma --workflow npt --npt-steps 10000" \
    --qos debug --time 00:30:00
```

### Use `--dry-run`

Preview the campaign before committing GPU hours:

```bash
python tools/perlmutter/perlmutter.py campaign create ds3_check \
    --workflow equilibrate \
    --inputs "data/DS3_TO_RUN/*.xyz" \
    --args "--model uma --workflow npt" \
    --dry-run
```

### Sync before submitting

Always push the latest code before creating a campaign. If you fix a bug in `equilibrate.py`, that fix won't reach Perlmutter until you sync:

```bash
python tools/perlmutter/perlmutter.py sync
```

### Refresh your SSH keys

NERSC SSH certificates expire after 24 hours. If you get SSH errors, re-run `sshproxy.sh`:

```bash
./sshproxy.sh -u your_username
```

### Wall time guidelines

Equilibration time depends heavily on system size (atom count) and step count. Rough guidelines for a single A100:

| Atoms | 50k steps | 100k steps |
|-------|-----------|------------|
| ~500 | ~30 min | ~1 hr |
| ~1000 | ~1 hr | ~2 hr |
| ~2000 | ~2-3 hr | ~4-6 hr |
| ~5000+ | ~6-8 hr | ~12+ hr |

Start with `--time 04:00:00` for most boxes. If jobs time out, retry with `--time 08:00:00` — checkpoints mean you don't lose progress.

### Keep `--max-concurrent` reasonable

The default of 16 is a good balance. Going higher (e.g., 64) will drain your allocation faster and may affect your fair-share priority on Perlmutter. Going lower (e.g., 4) is fine if you want to be conservative or are sharing the allocation with others.

### Naming campaigns

Use descriptive names that encode the key parameters:

```
ds3_uma_npt           # Dataset 3, UMA model, NPT workflow
ds3_orb_nvt_npt       # Dataset 3, OrbMol-v2, NVT+NPT
la_metad_oh           # Lanthanum metadynamics, OH system
test_3boxes_debug     # Test run
```

## File Reference

| File | Purpose |
|------|---------|
| `perlmutter.py` | Main CLI — all commands route through here |
| `campaign.py` | Campaign state management, sacct parsing, dashboard formatting |
| `setup_env.sh` | Creates/updates the conda env on Perlmutter |
| `templates/equilibrate_array.sbatch` | SLURM job array template for batch equilibration |
| `templates/run_md_array.sbatch` | SLURM job array template for batch production MD |
| `templates/analyze_array.sbatch` | SLURM job array template for batch trajectory analysis (CPU-only) |
| `templates/metadynamics.sbatch` | Single-job template for metadynamics |

## Local Config Files (gitignored)

| File | Created by | Contents |
|------|-----------|----------|
| `.perlmutter.json` | `setup` | NERSC username, project code, scratch path, conda env name |
| `.perlmutter_campaigns.json` | `campaign create` | Campaign state: tasks, job IDs, statuses |

These live at the repo root and are gitignored. Each user runs `setup` to create their own. No `.env` file is needed.

## Troubleshooting

**"Not configured. Run: perlmutter.py setup"** — You haven't run `setup` yet, or the `.perlmutter.json` file was deleted. Run `setup` again.

**SSH errors** — Your sshproxy certificate probably expired. Re-run `sshproxy.sh -u your_username` and try again.

**"No files match: ..."** — The glob pattern didn't find any files. Make sure you're running from the repo root and the path is correct. Wrap globs in quotes so your shell doesn't expand them before Python sees them.

**All tasks show "submitted" but nothing is running** — The SLURM queue may be full. Run `perlmutter.py ssh "squeue -u $USER"` to check. Jobs in the `PD` (pending) state are waiting for resources.

**Many OOM failures** — The system may be too large for a single A100's 80 GB. Try reducing `--npt-steps` to get a shorter run, or contact the group about multi-GPU strategies.

**NaN failures** — The ML potential produced unstable forces. Common causes: bad initial geometry (atoms too close), unsupported chemistry for the chosen model, or insufficient equilibration before NPT. Try a different `--model`, or inspect the input box in the 3D viewer for clashes.

**Campaign state seems stale** — `campaign status` refreshes from SLURM each time you run it. If a job finished between your last status check and now, just run status again.
