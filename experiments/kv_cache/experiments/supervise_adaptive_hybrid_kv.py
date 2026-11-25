#!/usr/bin/env python3
"""Detached host supervisor for the adaptive hybrid KV experiment.

Run this script under nohup or tmux on the NVIDIA host. It owns one named
Docker worker, resumes at completed scenario boundaries, validates compact
outputs, then publishes only the source, validation summary, report, metrics,
and compact figures to the thesis branch. Raw traces remain local.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BRANCH = "thesis/trackb-dense-grounding-v2"
REMOTE = "git@github.com:dybenko1/TrackingKVinOpenVLA.git"
IMAGE = "openvla-talk2dino-siglip:20261116"
CONTAINER = "trackb-adaptive-hybrid-kv-20261125"
BASE = REPO / "experiments/kv_cache/outputs/adaptive_hybrid_kv_cache"
OUT = BASE / "primary_20261125"
LOG = OUT / "logs/supervisor.log"
STATUS = OUT / "SUPERVISOR_STATUS.json"
LOCK = OUT / "runtime/supervisor.lock"
V2_CKPT = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/best_projection.pt"
V2_SHA256 = "8232ec8f657da47645d73741b16560a1140bfea89cce6826948b9457ce4854d4"
SOURCE_FILES = [
    "experiments/kv_cache/experiments/adaptive_hybrid_kv_policy.py",
    "experiments/kv_cache/experiments/adaptive_hybrid_kv_runner.py",
    "experiments/kv_cache/experiments/test_adaptive_hybrid_kv_policy.py",
    "experiments/kv_cache/experiments/supervise_adaptive_hybrid_kv.py",
    "experiments/kv_cache/experiments/closed_loop_language_kv_reuse.py",
    "experiments/kv_cache/experiments/static_reuse_experiment.py",
    "experiments/kv_cache/experiments/stage4a_oracle_free_attention.py",
]


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=str)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def log(message: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n")
        f.flush(); os.fsync(f.fileno())


def run(*args, capture=False, check=True, timeout=None):
    return subprocess.run(args, check=check, text=True, capture_output=capture, timeout=timeout)


def set_status(phase, **fields):
    current = {}
    try: current = json.loads(STATUS.read_text())
    except (OSError, ValueError): pass
    current.update(status=phase, phase=phase, updated_unix=time.time(), container=CONTAINER, **fields)
    atomic_json(STATUS, current)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def preflight():
    branch = run("git", "-C", str(REPO), "branch", "--show-current", capture=True).stdout.strip()
    if branch != BRANCH: raise RuntimeError(f"Wrong branch: {branch!r}, expected {BRANCH!r}")
    if sha256(V2_CKPT) != V2_SHA256: raise RuntimeError("Frozen V2 epoch-40 checkpoint SHA-256 mismatch")
    dirty = run("git", "-C", str(REPO), "status", "--porcelain", capture=True).stdout.strip()
    if dirty: raise RuntimeError(f"Source worktree must be clean before detached run: {dirty}")
    remote_sha = run("git", "ls-remote", REMOTE, f"refs/heads/{BRANCH}", capture=True, timeout=30).stdout.split()[0]
    local_sha = run("git", "-C", str(REPO), "rev-parse", "HEAD", capture=True).stdout.strip()
    if remote_sha != local_sha: raise RuntimeError(f"Source commit is not published: local={local_sha} remote={remote_sha}")
    run("docker", "image", "inspect", IMAGE, capture=True)
    # Only inspect workload ownership; never stop or alter an unrelated container.
    active = run("docker", "ps", "--format", "{{.Names}} {{.Image}} {{.Status}}", capture=True).stdout.strip().splitlines()
    run("docker", "run", "--rm", "--runtime", "nvidia", IMAGE, "python", "-c",
        "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info(0))",
        capture=True, timeout=90)
    runtime = OUT / "runtime/libero_config"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "config.yaml").write_text(
        "benchmark_root: /opt/LIBERO/libero/libero\n"
        "bddl_files: /opt/LIBERO/libero/libero/bddl_files\n"
        "init_states: /opt/LIBERO/libero/libero/init_files\n"
        "datasets: /opt/LIBERO/libero/datasets\n"
        "assets: /opt/LIBERO/libero/libero/assets\n")
    usage = shutil.disk_usage(REPO)
    if usage.free < 5 * 1024**3: raise RuntimeError(f"Less than 5 GiB free for compact policy outputs: {usage.free}")
    # Prove noninteractive GitHub auth with a no-op branch push before launch.
    run("git", "-C", str(REPO), "push", "--dry-run", REMOTE, BRANCH, timeout=45, capture=True)
    return {"branch": branch, "source_commit": local_sha, "remote_sha": remote_sha,
            "v2_checkpoint_sha256": V2_SHA256, "free_disk_bytes": usage.free,
            "active_containers_read_only": active, "cuda_check": "passed inside validated Docker image"}


def launch_container(attempt: int):
    container_out = "/opt/openvla/experiments/kv_cache/outputs/adaptive_hybrid_kv_cache/primary_20261125"
    attempt_log = f"{container_out}/logs/experiment_attempt_{attempt:02d}.log"
    command = (
        f"mkdir -p {container_out}/runtime/numba {container_out}/runtime/matplotlib /tmp/adaptive-hybrid-home; "
        f"python -u /opt/openvla/experiments/kv_cache/experiments/adaptive_hybrid_kv_runner.py "
        f"--mode run-all --output-dir {container_out} "
        f"> {attempt_log} 2>&1; rc=$?; "
        f"echo $rc > {container_out}/logs/experiment_attempt_{attempt:02d}.exit; exit $rc"
    )
    args = ["docker", "run", "-d", "--name", CONTAINER, "--restart=no", "--runtime", "nvidia",
        "--network", "host", "--group-add", "2011", "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
        "-e", "MUJOCO_GL=egl", "-e", "PYOPENGL_PLATFORM=egl", "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1", "-e", "PYTHONPATH=/opt/openvla:/opt/LIBERO",
        "-e", "HOME=/tmp/adaptive-hybrid-home", "-e",
        f"LIBERO_CONFIG_PATH={container_out}/runtime/libero_config", "-e",
        f"NUMBA_CACHE_DIR={container_out}/runtime/numba", "-e", f"MPLCONFIGDIR={container_out}/runtime/matplotlib",
        "-v", f"{REPO}:/opt/openvla",
        "-v", "/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/models:/opt/openvla/cache/models",
        "-v", "/home/oscar/projects/TrackingKVinOpenVLA/LIBERO:/opt/LIBERO",
        "-v", "/home/oscar/projects/TrackingKVinOpenVLA/dusty-robosuite:/opt/robosuite", "-v", "/tmp:/tmp",
        "-w", "/opt/openvla", IMAGE, "bash", "-lc", command]
    return run(*args, capture=True).stdout.strip()


def validate_results():
    required = ["RUN_STATUS.json", "configuration.json", "summary.json", "per_scenario_metrics.csv",
                "ADAPTIVE_HYBRID_KV_REPORT.md"]
    for name in required:
        if not (OUT / name).is_file(): raise RuntimeError(f"Missing output {OUT/name}")
    status = json.loads((OUT / "RUN_STATUS.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    scenario_files = list((OUT / "scenario_results").glob("task*_state*.json"))
    visuals = list((OUT / "mask_visualizations").glob("*.png"))
    if status.get("status") != "complete" or len(status.get("completed", [])) != 9:
        raise RuntimeError("Runner did not mark all nine scenarios complete")
    if summary.get("status") != "complete" or summary.get("scenario_count") != 9:
        raise RuntimeError("Aggregate summary is incomplete")
    if len(scenario_files) != 9: raise RuntimeError(f"Expected nine scenario result files; found {len(scenario_files)}")
    if len(visuals) < 18: raise RuntimeError(f"Expected at least 18 diagnostic figures; found {len(visuals)}")
    for p in scenario_files:
        obj = json.loads(p.read_text())
        if obj.get("configuration_sha256") != summary["configuration"]["configuration_sha256"]:
            raise RuntimeError(f"Configuration mismatch in {p.name}")
    provenance = json.loads((OUT / "source_commit.json").read_text())
    report = OUT / "ADAPTIVE_HYBRID_KV_REPORT.md"
    text = report.read_text()
    if "## Source and publication provenance" not in text:
        with report.open("a") as f:
            f.write(f"\n## Source and publication provenance\n\nImplementation source commit: `{provenance['source_commit']}` on `{BRANCH}`. The aggregate result commit and remote verification are recorded in `SUPERVISOR_STATUS.json`.\n")
    return {"scenario_files": len(scenario_files), "visualizations": len(visuals),
            "successes": summary["successes"], "scenarios": summary["scenario_count"]}


def publish():
    compact = [OUT / n for n in ("RUN_STATUS.json", "configuration.json", "summary.json",
        "per_scenario_metrics.csv", "ADAPTIVE_HYBRID_KV_REPORT.md", "validation.json",
        "validation_configuration.json", "ADAPTIVE_VALIDATION_REPORT.md", "source_commit.json",
        "mask_visualizations", "validation_visualizations")]
    rel = [str(p.relative_to(REPO)) for p in compact if p.exists()]
    run("git", "-C", str(REPO), "add", "-f", *rel)
    run("git", "-C", str(REPO), "diff", "--cached", "--check")
    changed = run("git", "-C", str(REPO), "diff", "--cached", "--quiet", check=False)
    if changed.returncode == 1:
        run("git", "-C", str(REPO), "commit", "-m", "Publish adaptive hybrid KV experiment results")
    sha = run("git", "-C", str(REPO), "rev-parse", "HEAD", capture=True).stdout.strip()
    run("git", "-C", str(REPO), "push", REMOTE, BRANCH, timeout=90, capture=True)
    remote_sha = run("git", "ls-remote", REMOTE, f"refs/heads/{BRANCH}", capture=True, timeout=30).stdout.split()[0]
    if remote_sha != sha: raise RuntimeError(f"Remote SHA mismatch after push: local={sha} remote={remote_sha}")
    return sha


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "logs").mkdir(exist_ok=True); LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            set_status("already_running", message="Another adaptive supervisor owns the lock")
            return
        atomic_json(STATUS, {"status": "preflight", "job": CONTAINER, "started_unix": time.time()})
        try:
            checks = preflight(); set_status("preflight_passed", checks=checks)
            atomic_json(OUT / "source_commit.json", {"source_commit": checks["source_commit"],
                "starting_reference_commit": "168dd80eefd1b0a6d6926d89f309f8be6db45d13",
                "remote_source_commit_verified": checks["remote_sha"], "branch": BRANCH,
                "remote": REMOTE, "v2_checkpoint_sha256": V2_SHA256})
            existing = run("docker", "inspect", "-f", "{{.State.Running}}", CONTAINER, check=False, capture=True)
            if existing.returncode == 0 and existing.stdout.strip() == "true":
                raise RuntimeError(f"Named adaptive worker {CONTAINER} is already running; refusing duplicate")
            if existing.returncode == 0: run("docker", "rm", CONTAINER, capture=True)
            for attempt in range(1, 4):
                set_status("launching", attempt=attempt, max_attempts=3)
                cid = launch_container(attempt); log(f"Started {CONTAINER} id={cid} attempt={attempt}")
                set_status("running", container_id=cid, attempt=attempt,
                           log=str(OUT/f"logs/experiment_attempt_{attempt:02d}.log"))
                while True:
                    state = run("docker", "inspect", "-f", "{{.State.Running}}", CONTAINER, capture=True).stdout.strip()
                    done = 0
                    try: done = len(json.loads((OUT/"RUN_STATUS.json").read_text()).get("completed", []))
                    except (OSError, ValueError): pass
                    set_status("running" if state == "true" else "worker_exited", completed_scenarios=done,
                               attempt=attempt, container_id=cid)
                    if state != "true": break
                    time.sleep(20)
                code = int(run("docker", "inspect", "-f", "{{.State.ExitCode}}", CONTAINER, capture=True).stdout.strip())
                log(f"Worker exit={code}, attempt={attempt}")
                if code == 0:
                    try: validated = validate_results(); break
                    except Exception as e: log(f"Result validation failed: {e}")
                if attempt == 3:
                    set_status("failed", error=f"worker exited {code} after three attempts; partials retained")
                    return
                run("docker", "rm", CONTAINER, capture=True)
                set_status("retrying", attempt=attempt+1, message="Resume completed scenarios; archive interrupted partial episode")
            sha = publish()
            set_status("complete", validated=validated, result_commit=sha,
                       report_url=f"https://github.com/dybenko1/TrackingKVinOpenVLA/blob/{BRANCH}/experiments/kv_cache/outputs/adaptive_hybrid_kv_cache/primary_20261125/ADAPTIVE_HYBRID_KV_REPORT.md",
                       remote_sha_verified=True)
            log(f"Completed and remote branch verified at {sha}")
        except Exception as exc:
            set_status("failed", error=f"{type(exc).__name__}: {exc}")
            log(f"ERROR {type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__": main()
