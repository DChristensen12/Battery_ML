"""Campaign state management for batch SolvationNet runs on Perlmutter.

A campaign is a named batch run (e.g., "equilibrate 83 boxes with UMA/NPT").
State is persisted to .perlmutter_campaigns.json at the repo root so it
survives across terminal sessions.
"""

import json
import pathlib
import shlex
import subprocess
from datetime import datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CAMPAIGNS_FILE = REPO_ROOT / ".perlmutter_campaigns.json"

OUTPUT_DIR_MAP = {
    ("npt", "uma"): "NPT_FAIRChem-UMA",
    ("nvt", "uma"): "NVT_FAIRChem-UMA",
    ("nvt+npt", "uma"): "NVT-NPT_FAIRChem-UMA",
    ("npt", "orbmol_v2"): "NPT_OrbMolV2",
    ("nvt", "orbmol_v2"): "NVT_OrbMolV2",
    ("nvt+npt", "orbmol_v2"): "NVT-NPT_OrbMolV2",
}


def box_name_from_input(filename):
    """Derive the box name from an input filename, matching equilibrate.py's logic."""
    name = pathlib.Path(filename).stem
    if name.endswith("_opt"):
        name = name[:-4]
    return name


class Campaign:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data or {}

    @classmethod
    def create(cls, name, workflow, inputs, workflow_args, slurm_args):
        tasks = []
        for inp in inputs:
            fname = pathlib.Path(inp).name
            tasks.append({
                "input": fname,
                "box_name": box_name_from_input(fname),
                "slurm_job_id": None,
                "status": "pending",
                "exit_code": None,
                "submit_count": 0,
            })

        data = {
            "workflow": workflow,
            "workflow_args": workflow_args,
            "slurm_args": slurm_args,
            "created": datetime.now().isoformat(),
            "tasks": tasks,
            "array_job_ids": [],
            "pipeline_steps": None,
        }
        c = cls(name, data)
        c.save()
        return c

    @classmethod
    def load(cls, name):
        all_campaigns = _load_all()
        if name not in all_campaigns:
            raise KeyError(f"Campaign '{name}' not found. Use 'campaign list' to see available.")
        return cls(name, all_campaigns[name])

    @classmethod
    def list_all(cls):
        return _load_all()

    def save(self):
        all_campaigns = _load_all()
        all_campaigns[self.name] = self.data
        CAMPAIGNS_FILE.write_text(json.dumps(all_campaigns, indent=2) + "\n")

    @property
    def tasks(self):
        return self.data["tasks"]

    @property
    def workflow(self):
        return self.data["workflow"]

    @property
    def workflow_args(self):
        return self.data["workflow_args"]

    @property
    def slurm_args(self):
        return self.data["slurm_args"]

    def record_submission(self, array_job_id, task_indices=None):
        """Record a SLURM array submission. task_indices: which tasks were submitted (0-based)."""
        self.data["array_job_ids"].append(array_job_id)
        targets = task_indices if task_indices is not None else range(len(self.tasks))
        for i in targets:
            self.tasks[i]["slurm_job_id"] = f"{array_job_id}_{i + 1}"
            self.tasks[i]["status"] = "submitted"
            self.tasks[i]["submit_count"] += 1
        self.save()

    def refresh_status(self, ssh_func):
        """Query sacct for all array jobs in this campaign and update task statuses."""
        if not self.data["array_job_ids"]:
            return

        job_ids = ",".join(self.data["array_job_ids"])
        output = ssh_func(
            f"sacct -j {job_ids} --parsable2 --noheader "
            f"--format=JobID,State,ExitCode,Elapsed,MaxRSS",
            capture=True,
        )
        if not output:
            return

        sacct_map = {}
        for line in output.strip().split("\n"):
            parts = line.split("|")
            if len(parts) < 3:
                continue
            job_id = parts[0]
            if "." in job_id:
                continue
            state = parts[1]
            exit_code_str = parts[2]
            elapsed = parts[3] if len(parts) > 3 else ""
            maxrss = parts[4] if len(parts) > 4 else ""
            try:
                exit_code = int(exit_code_str.split(":")[0])
            except (ValueError, IndexError):
                exit_code = None
            sacct_map[job_id] = {
                "state": state, "exit_code": exit_code,
                "elapsed": elapsed, "maxrss": maxrss,
            }

        for task in self.tasks:
            jid = task.get("slurm_job_id")
            if not jid or jid not in sacct_map:
                continue
            info = sacct_map[jid]
            task["exit_code"] = info["exit_code"]
            task["elapsed"] = info.get("elapsed", "")
            task["status"] = _classify_status(info["state"], info["exit_code"])

        self.save()

    def summary(self):
        counts = {}
        for t in self.tasks:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
        return counts

    def failed_tasks(self):
        return [t for t in self.tasks if t["status"] in ("failed", "oom", "timeout", "nan")]

    def retryable_tasks(self, include_nan=False):
        retryable = {"failed", "oom", "timeout"}
        if include_nan:
            retryable.add("nan")
        return [t for t in self.tasks if t["status"] in retryable]

    def retryable_indices(self, include_nan=False):
        retryable = {"failed", "oom", "timeout"}
        if include_nan:
            retryable.add("nan")
        return [i for i, t in enumerate(self.tasks) if t["status"] in retryable]

    def done_tasks(self):
        return [t for t in self.tasks if t["status"] == "done"]

    def input_files(self):
        return [t["input"] for t in self.tasks]

    def format_dashboard(self):
        counts = self.summary()
        total = len(self.tasks)
        done = counts.get("done", 0)
        running = counts.get("running", 0)
        submitted = counts.get("submitted", 0)
        failed = sum(counts.get(s, 0) for s in ("failed", "oom", "timeout", "nan"))
        pending = counts.get("pending", 0)

        wf = self.workflow
        wf_args = self.workflow_args
        created = self.data.get("created", "?")

        bar_width = 30
        filled = int(bar_width * done / total) if total else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        lines = [
            f"Campaign: {self.name} ({wf})",
            f"Args:     {wf_args}",
            f"Created:  {created}",
            "",
            f"Progress: {bar} {done}/{total} ({100 * done // total if total else 0}%)",
            f"  Done: {done}   Running: {running}   Submitted: {submitted}   "
            f"Failed: {failed}   Pending: {pending}",
        ]

        failures = self.failed_tasks()
        if failures:
            lines.append("")
            lines.append("Failed:")
            for t in failures:
                retryable = "retryable" if t["status"] != "nan" else "NOT retryable"
                elapsed = t.get("elapsed", "")
                lines.append(
                    f"  {t['input']:<45s} {t['status']:<10s} "
                    f"tries={t['submit_count']}  {retryable}  {elapsed}"
                )

        return "\n".join(lines)

    def auto_output_dir(self):
        """Derive the local output directory from workflow + model, matching existing conventions."""
        wf_args = self.workflow_args
        model = "uma"
        workflow = "npt"
        for token in wf_args.split():
            if token in ("uma", "orbmol_v2"):
                model = token
            if token in ("npt", "nvt", "nvt+npt"):
                workflow = token

        subdir = OUTPUT_DIR_MAP.get((workflow, model))
        if subdir:
            return REPO_ROOT / "data" / "na-electrolyte-solvation-boxes" / subdir
        return REPO_ROOT / "data" / "perlmutter_results" / self.name


def _classify_status(slurm_state, exit_code):
    state = slurm_state.upper().strip()
    if state in ("PENDING",):
        return "submitted"
    if state in ("RUNNING", "REQUEUED"):
        return "running"
    if state == "TIMEOUT":
        return "timeout"
    if state in ("OUT_OF_MEMORY",) or exit_code == 137:
        return "oom"
    if state in ("COMPLETED",):
        if exit_code == 42:
            return "nan"
        if exit_code == 0:
            return "done"
        return "failed"
    if state in ("FAILED", "NODE_FAIL"):
        if exit_code == 42:
            return "nan"
        if exit_code == 137:
            return "oom"
        return "failed"
    if state in ("CANCELLED", "CANCELLED+"):
        return "failed"
    return "failed"


def _load_all():
    if not CAMPAIGNS_FILE.exists():
        return {}
    try:
        return json.loads(CAMPAIGNS_FILE.read_text())
    except (json.JSONDecodeError, IOError):
        return {}
