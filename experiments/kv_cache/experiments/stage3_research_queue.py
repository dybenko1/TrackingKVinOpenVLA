"""Persistent sequential Track B / Stage 3C / Stage 3D research queue."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
KV = REPO / "experiments/kv_cache"
CONTAINER = os.environ.get("OPENVLA_CONTAINER", "f5240fe761b5")
TRACKB_PID = 1683056
OUT = KV / "outputs/stage3_research_queue"
STATUS = OUT / "queue_status.json"
LOG = OUT / "queue.log"
STAGE_SCRIPT = "experiments/kv_cache/experiments/stage3cd_oracle_masked_matrix.py"
ANALYSIS_SCRIPT = "experiments/kv_cache/experiments/stage3_factorial_analysis.py"
STAGES = {
    "stage3c": ("stage3c_oracle_k_all_layers", "Stage 3C K-only layers 0–31"),
    "stage3d": ("stage3d_oracle_kv_first14", "Stage 3D K+V layers 0–13"),
}
TRACKB_MODULE = "experiments.kv_cache.experiments.track_b_segmentation.supervise"
lock_handle = None


def _now() -> float:
    return time.time()


def _log(text: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {text}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _read_status() -> dict[str, Any]:
    if STATUS.exists():
        try:
            return json.loads(STATUS.read_text())
        except (OSError, ValueError):
            pass
    return {"schema_version": 1, "jobs": {}}


def _save_status(status: dict[str, Any]) -> None:
    status["updated_at_unix"] = _now()
    _atomic_json(STATUS, status)


def _run(command: list[str], log_path: Path = LOG, cwd: Path = REPO) -> int:
    _log("$ " + " ".join(command))
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as handle:
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
    return process.wait()


def _container_python(command: str) -> int:
    return _run(["docker", "exec", CONTAINER, "bash", "-lc", f"cd /opt/openvla && {command}"])


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.strip()


def _push_paths(paths: list[Path], message: str) -> str:
    if _git("branch", "--show-current") != "main":
        raise RuntimeError("Research queue requires the existing main branch.")
    if _git("diff", "--cached", "--name-only"):
        raise RuntimeError("Git index is unexpectedly staged; refusing to mix unrelated files into a queue commit.")
    relative = [str(p.relative_to(REPO)) for p in paths if p.exists() and p.is_file()]
    if not relative:
        raise RuntimeError("No existing requested artifact paths to commit.")
    _git("add", "-f", *relative)
    _git("diff", "--cached", "--check")
    _git("commit", "-m", message)
    commit = _git("rev-parse", "HEAD")
    _run(["git", "push", "origin", "main"])
    return commit


def _write_job_report(stage: str, result_dir: Path, code_commit: str, status: str, error: str | None = None) -> None:
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    lines = [f"# {stage.upper()} oracle-masked projection-skip run", "", f"Status: **{status.upper()}**.", "",
             f"Implementation revision: `{code_commit}`.", "", "Configuration: LIBERO Spatial tasks 4/5/7, states 0/1/2, seed 7, horizon 220, 10 settling steps, R=5, visual tokens 1–256.",
             f"Layers/components: {summary.get('config', {}).get('layers', 'see configuration')}; {summary.get('config', {}).get('components', 'see configuration')}.",
             "Fresh baselines are referenced from Stage 3B results commit `c9b05be688ab0ad5d474665300b8504f4c077ebb` after metadata compatibility checks.",
             "Projection-row accounting is structural; latency was not measured."]
    if error:
        lines += ["", "Failure detail:", "", f"`{error}`"]
    if summary.get("status") == "complete":
        agg = summary.get("aggregate", {})
        lines += ["", f"Scenario success: Fresh {agg.get('fresh_successes')}/9; intervention {agg.get('true_skip_successes')}/9; among Fresh-success cases {agg.get('true_skip_successes_among_fresh_successes')}/{agg.get('fresh_success_scenarios')}.",
                  f"Total skipped projection rows: K={agg.get('total_skipped_k_rows', 0):,}; V={agg.get('total_skipped_v_rows', 0):,}."]
    (result_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    file_list = [result_dir / "REPORT.md", summary_path, result_dir / "validation.json"]
    file_list.extend(sorted(result_dir.glob("task*_state*/results.json")))
    manifest = [{"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in file_list if p.exists()]
    _atomic_json(result_dir / "manifest.json", manifest)


def _commit_stage_output(stage: str, result_dir: Path, code_commit: str) -> str:
    status_path = result_dir / "run_status.json"
    _atomic_json(status_path, {"status": "complete", "stage": stage, "implementation_commit": code_commit,
                              "scenario_count": 9, "latency_measured": False})
    paths = [result_dir / "summary.json", result_dir / "validation.json", result_dir / "REPORT.md",
             result_dir / "manifest.json", status_path]
    paths.extend(sorted(result_dir.glob("task*_state*/results.json")))
    return _push_paths(paths, f"results(kv-cache): publish {stage.upper()} oracle-masked ablation")


def _append_execution_excerpt(source_log: Path, result_dir: Path, stage: str) -> None:
    try:
        lines = source_log.read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    selected = [line for line in lines if "VALIDATION_PASS" in line or "SCENARIO_COMPLETE" in line
                or "Traceback" in line or "RuntimeError" in line or "AssertionError" in line]
    path = result_dir / "execution_excerpt.log"
    path.write_text("\n".join(selected[-100:]) + "\n", encoding="utf-8")


def _trackb_complete() -> bool:
    path = KV / "outputs/track_b_segmentation/run_status.json"
    if not path.exists():
        return False
    try:
        state = json.loads(path.read_text())
        return state.get("status") == "completed" and state.get("github_sync_status") == "pushed"
    except (OSError, ValueError):
        return False


def _run_trackb(code_commit: str) -> tuple[str, int]:
    path = KV / "outputs/track_b_segmentation/run_status.json"
    if _trackb_complete():
        _log("Track B existing output validation and GitHub sync already complete; reusing results.")
        return _git("rev-parse", "HEAD"), 0
    # The previous detached Track B instance exited after a one-key split typo;
    # no session/process is active, so this is a safe resume of its saved dataset.
    cmd = [sys.executable, "-m", TRACKB_MODULE, "--container", CONTAINER,
           "--stage3b-supervisor-pid", str(TRACKB_PID), "--code-commit", code_commit]
    rc = _run(cmd)
    if rc == 0 and _trackb_complete():
        state = json.loads(path.read_text())
        return state.get("results_commit", _git("rev-parse", "HEAD")), 0
    return "", rc or 1


def _run_stage(stage: str, result_dir: Path, code_commit: str) -> tuple[str, int]:
    existing = result_dir / "summary.json"
    if existing.exists():
        try:
            d = json.loads(existing.read_text())
            valid = d.get("status") == "complete" and len(d.get("scenarios", [])) == 9
            if valid and all((result_dir / f"task{t}_state{s}" / "results.json").is_file()
                             for t in (4,5,7) for s in (0,1,2)):
                _log(f"{stage.upper()} validated artifacts exist; will reuse them without rerunning scenarios.")
                return _commit_stage_output(stage, result_dir, code_commit), 0
        except (OSError, ValueError):
            pass
    validation_path = result_dir / "validation.json"
    if not validation_path.exists():
        rc = _container_python(f"python {STAGE_SCRIPT} --stage {stage} --validate-only")
        if rc != 0:
            return "", rc
    validation = json.loads(validation_path.read_text())
    if not validation.get("all_valid"):
        raise RuntimeError(f"{stage} validation artifact does not pass all checks.")
    rc = _container_python(f"python {STAGE_SCRIPT} --stage {stage} --matrix")
    if rc != 0:
        return "", rc
    d = json.loads(existing.read_text())
    assert d.get("status") == "complete" and len(d.get("scenarios", [])) == 9
    for task in (4,5,7):
        for state in (0,1,2):
            scenario = json.loads((result_dir / f"task{task}_state{state}" / "results.json").read_text())
            assert scenario["true_skip"]["all_valid"]
    _write_job_report(stage, result_dir, code_commit, "complete")
    return _commit_stage_output(stage, result_dir, code_commit), 0


def _gpu_idle_gate() -> dict[str, Any]:
    probe = "python - <<'PY'\nimport torch,json\na=torch.cuda.is_available()\nd={'cuda_available':bool(a)}\nif a:\n f,t=torch.cuda.mem_get_info(0); d.update(device=torch.cuda.get_device_name(0),free_bytes=int(f),total_bytes=int(t))\nprint(json.dumps(d))\nPY"
    result = subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", probe], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"CUDA probe failed: {result.stdout[-1000:]}")
    marker = next((line for line in result.stdout.splitlines() if line.startswith("{")), None)
    if not marker:
        raise RuntimeError("CUDA probe did not return JSON.")
    d = json.loads(marker)
    if not d.get("cuda_available") or d.get("free_bytes", 0) < 6 * 1024**3:
        raise RuntimeError(f"Insufficient CUDA resources: {d}")
    ps = subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", "ps -eo args"], text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    active = [line.strip() for line in ps.splitlines() if any(x in line for x in
              ("stage3cd_oracle_masked_matrix.py --stage", "track_b_segmentation.main run"))]
    if active:
        raise RuntimeError(f"A competing OpenVLA experiment process is active: {active}")
    return d


def _failure_commit(status: dict[str, Any], job: str, error: str, code_commit: str) -> None:
    _log(f"[{job.upper()}_FAILED] {error}")
    status["jobs"][job].update({"status": "failed", "error": error, "finished_at_unix": _now()})
    _save_status(status)
    diag = OUT / f"{job}_failure.md"
    diag.write_text(f"# {job} failed\n\nImplementation commit: `{code_commit}`\n\n`{error}`\n\nFull log: `{LOG.relative_to(REPO)}`\n", encoding="utf-8")
    try:
        commit = _push_paths([STATUS, diag], f"docs(experiments): record {job} queue failure")
        status["jobs"][job]["failure_status_commit"] = commit
        _save_status(status)
    except BaseException as exc:
        _log(f"[FAILURE_STATUS_PUSH_FAILED] {type(exc).__name__}: {exc}")


def _finish_job(status: dict[str, Any], job: str, result_commit: str, code_commit: str):
    status["jobs"][job].update({"status": "complete", "exit_code": 0,
                                "result_commit": result_commit, "finished_at_unix": _now()})
    _save_status(status)
    if job == "track_b":
        # Track B commits its own validated metrics/figures via its established supervisor.
        try:
            result = _push_paths([STATUS], "docs(experiments): record Track B queue completion")
            status["jobs"][job]["queue_status_commit"] = result
            _save_status(status)
        except BaseException as exc:
            _log(f"[QUEUE_STATUS_PUSH_FAILED] {type(exc).__name__}: {exc}")


def run_queue():
    global lock_handle
    OUT.mkdir(parents=True, exist_ok=True)
    lock_handle = (OUT / ".queue.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another stage3 research queue supervisor holds the lock; refusing a duplicate run.")
    branch = _git("branch", "--show-current")
    if branch != "main":
        raise RuntimeError(f"Expected main branch, found {branch}.")
    code_commit = _git("rev-parse", "HEAD")
    status = _read_status()
    status.setdefault("jobs", {})
    for job in ("track_b", "stage3c", "stage3d", "factorial"):
        status["jobs"].setdefault(job, {"status": "queued"})
    status.update({"status": "running", "branch": "main", "implementation_commit": code_commit,
                   "container": CONTAINER, "started_at_unix": status.get("started_at_unix", _now()),
                   "ssh_disconnect_persistence": "tmux session; survives SSH disconnect while host remains up",
                   "host_reboot_recovery": "not configured; requires manual restart with --resume"})
    _save_status(status)
    _log(f"[QUEUE_START] pid={os.getpid()} branch=main implementation={code_commit}; sequential jobs; R5 policies unchanged.")
    jobs = [
        ("track_b", lambda: _run_trackb(code_commit)),
        ("stage3c", lambda: (_gpu_idle_gate(), _run_stage("stage3c", KV / "outputs/stage3c_oracle_k_all_layers", code_commit))[1]),
        ("stage3d", lambda: (_gpu_idle_gate(), _run_stage("stage3d", KV / "outputs/stage3d_oracle_kv_first14", code_commit))[1]),
    ]
    for job, action in jobs:
        if status["jobs"][job].get("status") == "complete":
            _log(f"[QUEUE_RESUME] {job} already complete; skipping.")
            continue
        status["jobs"][job].update({"status": "running", "started_at_unix": _now(), "exit_code": None})
        _save_status(status)
        try:
            if job == "track_b" and _trackb_complete():
                result_commit, rc = _run_trackb(code_commit)
            elif job in ("stage3c", "stage3d"):
                _, (result_commit, rc) = (None, action())
            else:
                result_commit, rc = action()
            if rc != 0:
                raise RuntimeError(f"job exited with status {rc}")
            _finish_job(status, job, result_commit or _git("rev-parse", "HEAD"), code_commit)
            _log(f"[JOB_COMPLETE] {job}; results commit={result_commit or _git('rev-parse','HEAD')}")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            _failure_commit(status, job, error, code_commit)
            # Failures in one independent experiment do not block later cells.
            continue

    status["jobs"]["factorial"].update({"status": "running", "started_at_unix": _now()})
    _save_status(status)
    try:
        analysis_rc = _container_python(f"python {ANALYSIS_SCRIPT}")
        out = KV / "outputs/stage3_factorial_comparison"
        if analysis_rc != 0 or not (out / "summary.json").is_file() or not (out / "REPORT.md").is_file():
            raise RuntimeError(f"factorial analysis exited with status {analysis_rc} or missing outputs")
        analysis = json.loads((out / "summary.json").read_text())
        status["jobs"]["factorial"].update({"status": "complete" if analysis.get("status") == "complete" else "incomplete",
                                            "exit_code": 0, "finished_at_unix": _now()})
        status["status"] = "complete" if analysis.get("status") == "complete" else "incomplete"
        _save_status(status)
        paths = [out / "REPORT.md", out / "comparison.csv", out / "summary.json"]
        paths += sorted(out.glob("*.png"))
        result_commit = _push_paths(paths + [STATUS], "results(kv-cache): publish Stage 3 factorial comparison")
        status["jobs"]["factorial"]["result_commit"] = result_commit
        status["status"] = "complete" if analysis.get("status") == "complete" else "incomplete"
        _save_status(status)
        _push_paths([STATUS], "docs(experiments): record final research queue status")
        _log(f"[QUEUE_FINISHED] status={status['status']} aggregate commit={result_commit}")
    except BaseException as exc:
        status["jobs"]["factorial"].update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at_unix": _now()})
        status["status"] = "incomplete"
        _save_status(status)
        try:
            _push_paths([STATUS], "docs(experiments): record incomplete factorial queue")
        except BaseException as push_exc:
            _log(f"[FINAL_STATUS_PUSH_FAILED] {type(push_exc).__name__}: {push_exc}")
        _log(f"[FACTORIAL_FAILED] {type(exc).__name__}: {exc}")


def main():
    run_queue()


if __name__ == "__main__":
    main()
