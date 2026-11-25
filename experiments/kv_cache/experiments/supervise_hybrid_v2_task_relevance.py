#!/usr/bin/env python3
"""Persistent host supervisor for the detached hybrid policy experiment."""
from __future__ import annotations

import json
import os
import fcntl
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BRANCH = "thesis/trackb-dense-grounding-v2"
REMOTE = "git@github.com:dybenko1/TrackingKVinOpenVLA.git"
IMAGE = "openvla-talk2dino-siglip:20261116"
CONTAINER = "trackb-hybrid-v2-relevance-20261124"
BASE = REPO / "experiments/kv_cache/outputs/hybrid_v2_task_relevance"
OUT = BASE / "live_run_20261124"
STATUS = OUT / "SUPERVISOR_STATUS.json"
LOG = OUT / "logs/supervisor.log"
RROOT = Path("/opt/openvla/cache/worktrees/trackb-dense-grounding-v2")
COUT = RROOT / "experiments/kv_cache/outputs/hybrid_v2_task_relevance/live_run_20261124"


def write_status(phase: str, message: str):
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    try:
        data = json.loads(STATUS.read_text())
    except Exception:
        pass
    data.update(phase=phase, message=message, container=CONTAINER, updated_unix=time.time())
    tmp = STATUS.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATUS)


def log(message: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n")
        f.flush()
        os.fsync(f.fileno())


def check_output(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def launch() -> str:
    croot = str(COUT)
    command = (
        f"mkdir -p {croot}/logs {croot}/runtime/numba {croot}/runtime/matplotlib; "
        f"mkdir -p /tmp/hybrid-home; "
        f"python -u {RROOT}/experiments/kv_cache/experiments/hybrid_v2_task_relevance_runner.py "
        f"--mode run-all --output-dir {croot} > {croot}/logs/experiment.log 2>&1; "
        f"rc=$?; echo $rc > {croot}/logs/experiment.exit; exit $rc"
    )
    args = ["docker", "run", "-d", "--name", CONTAINER, "--restart=no", "--runtime", "nvidia",
            "--network", "host", "--group-add", "2011",
            "-e", "NVIDIA_DRIVER_CAPABILITIES=all", "-e", "MUJOCO_GL=egl", "-e", "PYOPENGL_PLATFORM=egl",
            "-e", "HOME=/tmp/hybrid-home", "-e", f"LIBERO_CONFIG_PATH={croot}/runtime/libero_config",
            "-e", f"NUMBA_CACHE_DIR={croot}/runtime/numba", "-e", f"MPLCONFIGDIR={croot}/runtime/matplotlib",
            "-e", f"PYTHONPATH={RROOT}:/opt/LIBERO", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
            "-v", "/home/oscar/projects/TrackingKVinOpenVLA/openvla:/opt/openvla",
            "-v", "/home/oscar/projects/TrackingKVinOpenVLA/LIBERO:/opt/LIBERO",
            "-v", "/home/oscar/projects/TrackingKVinOpenVLA/dusty-robosuite:/opt/robosuite",
            "-v", "/tmp:/tmp", "-w", str(RROOT), IMAGE, "bash", "-lc", command]
    return check_output(*args, capture_output=True).stdout.strip()


def publish():
    required = ["RUN_STATUS.json", "HYBRID_V2_TASK_RELEVANCE_REPORT.md", "summary.json",
                "per_scenario_metrics.csv", "comparison_references.json"]
    for name in required:
        if not (OUT / name).is_file():
            raise RuntimeError(f"Missing validated result: {OUT / name}")
    summary = json.loads((OUT / "summary.json").read_text())
    status = json.loads((OUT / "RUN_STATUS.json").read_text())
    visuals = list((OUT / "mask_visualizations").glob("*.png"))
    scenarios = list((OUT / "scenario_results").glob("task*_state*.json"))
    if summary.get("status") != "complete" or summary.get("scenario_count") != 9 or len(status.get("completed", [])) != 9:
        raise RuntimeError("Runner status does not show nine completed scenarios")
    if len(visuals) < 18 or len(scenarios) != 9:
        raise RuntimeError(f"Expected >=18 visuals and 9 scenario files; found {len(visuals)}, {len(scenarios)}")
    paths = [
        BASE / "RUN_STATUS.json", BASE / "configuration.json", BASE / "summary.json",
        BASE / "per_scenario_metrics.csv", BASE / "HYBRID_V2_TASK_RELEVANCE_REPORT.md",
        BASE / "VALIDATION_SMOKE_REPORT.md", BASE / "validation_smoke.json", BASE / "mask_visualizations",
        OUT / "RUN_STATUS.json", OUT / "configuration.json",
        OUT / "summary.json", OUT / "comparison_references.json", OUT / "per_scenario_metrics.csv",
        OUT / "HYBRID_V2_TASK_RELEVANCE_REPORT.md", OUT / "mask_visualizations",
    ]
    # Add only result summaries and visuals; full per-step trace JSON and logs remain local.
    check_output("git", "-C", str(REPO), "add", "-f",
                 *[str(Path(p).relative_to(REPO)) for p in paths])
    check_output("git", "-C", str(REPO), "diff", "--cached", "--check")
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "--cached", "--quiet"])
    if diff.returncode == 1:
        check_output("git", "-C", str(REPO), "commit", "-m", "Publish hybrid V2 task-relevance KV results")
    sha = check_output("git", "-C", str(REPO), "rev-parse", "HEAD", capture_output=True).stdout.strip()
    check_output("git", "-C", str(REPO), "push", REMOTE, BRANCH, capture_output=True)
    remote_sha = check_output("git", "ls-remote", REMOTE, f"refs/heads/{BRANCH}", capture_output=True).stdout.split()[0]
    if remote_sha != sha:
        raise RuntimeError(f"Remote SHA {remote_sha} differs from local {sha}")
    return sha


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "runtime").mkdir(exist_ok=True)
    lock = (OUT / "runtime/supervisor.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        write_status("already_running", "Another supervisor process holds the lock")
        return
    write_status("preflight", "Checking branch, assets, Docker runtime, and duplicate worker")
    if check_output("git", "-C", str(REPO), "branch", "--show-current", capture_output=True).stdout.strip() != BRANCH:
        raise RuntimeError(f"Wrong branch; expected {BRANCH}")
    if not (BASE / "validation_smoke.json").is_file():
        raise RuntimeError("Live R5 smoke validation evidence is missing")
    container_exists = subprocess.run(["docker", "inspect", CONTAINER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if container_exists:
        running = check_output("docker", "inspect", "-f", "{{.State.Running}}", CONTAINER, capture_output=True).stdout.strip()
        if running == "true":
            write_status("already_running", "Named container already running; duplicate refused")
            return
        check_output("docker", "rm", CONTAINER, capture_output=True)
    check_output("docker", "run", "--rm", "--runtime", "nvidia", IMAGE, "python", "-c",
                 "import torch,open_clip; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0),open_clip.__version__)",
                 capture_output=True, timeout=90)
    conf = OUT / "runtime/libero_config"
    conf.mkdir(parents=True, exist_ok=True)
    (conf / "config.yaml").write_text(
        "benchmark_root: /opt/LIBERO/libero/libero\n"
        "bddl_files: /opt/LIBERO/libero/libero/bddl_files\n"
        "init_states: /opt/LIBERO/libero/libero/init_files\n"
        "datasets: /opt/LIBERO/libero/datasets\n"
        "assets: /opt/LIBERO/libero/libero/assets\n")
    for attempt in range(1, 4):
        write_status("launching", f"Starting nine-scenario run, attempt {attempt}/3")
        cid = launch()
        log(f"Started detached Docker worker {cid}; attempt={attempt}")
        write_status("running", "Detached GPU worker executing sequential scenarios")
        while check_output("docker", "inspect", "-f", "{{.State.Running}}", CONTAINER,
                           capture_output=True).stdout.strip() == "true":
            time.sleep(20)
            done = 0
            try:
                done = len(json.loads((OUT / "RUN_STATUS.json").read_text()).get("completed", []))
            except Exception:
                pass
            write_status("running", f"Docker worker active; scenario completion {done}/9")
        code = int(check_output("docker", "inspect", "-f", "{{.State.ExitCode}}", CONTAINER,
                                capture_output=True).stdout.strip())
        log(f"Worker exit code={code}")
        if code == 0:
            data = json.loads((OUT / "RUN_STATUS.json").read_text())
            if data.get("status") == "complete" and len(data.get("completed", [])) == 9:
                break
        if attempt == 3:
            write_status("failed", "Three attempts exhausted; partial artifacts preserved")
            return
        check_output("docker", "rm", CONTAINER, capture_output=True)
        write_status("retrying", "Rerunning incomplete scenarios; completed scenario artifacts are reused")
    write_status("validating", "Checking all required reports, per-scenario results, and visual panels")
    sha = publish()
    write_status("complete", f"Validated and published; commit {sha} verified on GitHub")
    log(f"GitHub result commit {sha} verified")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("failed", f"{type(exc).__name__}: {exc}; outputs preserved")
        log(f"ERROR: {type(exc).__name__}: {exc}")
        raise
