"""Persistent host-side Docker/tmux supervisor for the Stage 3C matrix.

Run from a detached tmux session. It performs one CUDA/runtime validation,
then the 10/25/50 percent matrices sequentially, validates saved outputs, and
pushes only this experiment's artifacts. It never changes or kills unrelated
containers/processes and never force-pushes.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OPENVLA = REPO
OUT = REPO / "experiments" / "kv_cache" / "outputs" / "stage3c_oracle_task_relevance"
SESSION = "stage3c-task-relevance"
CONTAINER = "stage3c-task-relevance"
IMAGE = "openvla-cached:latest"
LOG = Path("/tmp/stage3c_oracle_task_relevance_supervisor.log")
STATUS = Path("/tmp/stage3c_oracle_task_relevance_queue_status.json")
OUTPUT_LOG = OUT / "supervisor.log"
OUTPUT_STATUS = OUT / "queue_status.json"
LOCK = Path("/tmp/stage3c-task-relevance.lock")
SCRIPT = "experiments.kv_cache.experiments.stage3c_oracle_task_relevance"
MOUNTS = [
    (str(OPENVLA), "/opt/openvla"),
    ("/home/oscar/projects/TrackingKVinOpenVLA/LIBERO", "/opt/LIBERO"),
    ("/home/oscar/projects/TrackingKVinOpenVLA/dusty-robosuite", "/opt/robosuite"),
    ("/tmp", "/tmp"),
]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def emit(message):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with LOG.open("a") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def save_status(data):
    data["updated_at"] = now()
    STATUS.write_text(json.dumps(data, indent=2))
    mirror_supervisor_files()


def mirror_supervisor_files():
    """Copy host-writable /tmp state into the root-owned bind-mounted output dir."""
    if not container_id():
        return
    for source, target in ((LOG, OUTPUT_LOG), (STATUS, OUTPUT_STATUS)):
        if source.exists():
            subprocess.run(["docker", "exec", CONTAINER, "cp", str(source), str(target)],
                           cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(command, *, check=True, capture=False):
    emit("RUN " + " ".join(map(str, command)))
    result = subprocess.run(command, cwd=REPO, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    if capture and result.stdout:
        for line in result.stdout.splitlines():
            emit(line)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result


def container_id():
    result = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^{CONTAINER}$"],
                            cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"Docker query failed: {result.stderr.strip()}")
    return result.stdout.strip()


def ensure_container():
    existing = container_id()
    if existing:
        inspect = run(["docker", "inspect", "-f", "{{.Config.Image}}", CONTAINER], capture=True)
        image = inspect.stdout.strip().splitlines()[-1]
        if image != IMAGE:
            raise RuntimeError(f"Refusing to reuse named container with unexpected image {image!r}.")
        state = run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER], capture=True).stdout.strip().splitlines()[-1]
        if state != "true":
            run(["docker", "start", CONTAINER])
        return
    args = ["docker", "run", "-d", "--runtime", "nvidia", "--network", "host", "--group-add", "2011",
            "--name", CONTAINER, "-e", "NVIDIA_DRIVER_CAPABILITIES=all", "-e", "MUJOCO_GL=egl",
            "-e", "PYOPENGL_PLATFORM=egl", "-e", "MPLCONFIGDIR=/tmp/matplotlib",
            "-e", "TF_CPP_MIN_LOG_LEVEL=3"]
    for host, target in MOUNTS:
        args += ["-v", f"{host}:{target}"]
    args += ["-w", "/opt/openvla", IMAGE, "bash", "-lc", "sleep infinity"]
    run(args, capture=True)


def docker_exec(command, *, check=True):
    return run(["docker", "exec", CONTAINER, *command], check=check, capture=True)


def git_sync(paths, message):
    branch = run(["git", "branch", "--show-current"], capture=True).stdout.strip().splitlines()[-1]
    if branch != "main":
        raise RuntimeError(f"Expected existing main branch, found {branch!r}; refusing publication.")
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO)
    if staged.returncode != 0:
        raise RuntimeError("Git index is not empty; refusing to combine or overwrite unrelated staged work.")
    mirror_supervisor_files()
    common = ["experiments/kv_cache/outputs/stage3c_oracle_task_relevance/queue_status.json",
              "experiments/kv_cache/outputs/stage3c_oracle_task_relevance/supervisor.log"]
    # outputs/ is globally gitignored because it contains large experiment
    # artifacts; explicitly force-add only the requested Stage 3C paths.
    run(["git", "add", "-f", "--", *paths, *common])
    run(["git", "diff", "--cached", "--check"])
    has_changes = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO)
    if has_changes.returncode == 0:
        emit(f"No new tracked changes to commit for: {message}")
        return
    run(["git", "commit", "-m", message])
    run(["git", "push", "origin", "main"])
    sha = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip().splitlines()[-1]
    emit(f"PUSHED commit={sha}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        emit("Another Stage 3C supervisor owns the lock; refusing duplicate execution.")
        return 3
    status = {"experiment": "Stage 3C — Oracle Background + Task Relevance, Full-Layer K+V, R5",
              "session": SESSION, "container": CONTAINER, "branch": "main", "state": "starting",
              "jobs": {"runtime_validation": "pending", "protect_10": "pending",
                       "protect_25": "pending", "protect_50": "pending", "finalization": "pending"}}
    save_status(status)
    try:
        ensure_container()
        gpu = docker_exec(["python", "-c", "import torch; assert torch.cuda.is_available(); x=torch.ones(1, device='cuda'); torch.cuda.synchronize(); print({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'devices':torch.cuda.device_count(),'gpu':torch.cuda.get_device_name(0),'probe':x.item()})"])
        status["cuda_probe"] = gpu.stdout.strip().splitlines()[-1]
        status["jobs"]["runtime_validation"] = "running"
        save_status(status)
        validation = docker_exec(["python", "-m", SCRIPT, "--validate-only"], check=False)
        status["jobs"]["runtime_validation"] = "complete" if validation.returncode == 0 else "failed"
        status["validation_exit_code"] = validation.returncode
        if validation.returncode:
            status["state"] = "blocked_validation_failure"
            save_status(status)
            git_sync(["experiments/kv_cache/outputs/stage3c_oracle_task_relevance/validation.json",
                      "experiments/kv_cache/outputs/stage3c_oracle_task_relevance/supervisor.log",
                      "experiments/kv_cache/outputs/stage3c_oracle_task_relevance/queue_status.json"],
                     "Record Stage 3C validation failure")
            return 2
        save_status(status)
        for pct in (10, 25, 50):
            name = f"protect_{pct}"
            existing = OUT / name / "summary.json"
            if existing.exists():
                try:
                    if json.loads(existing.read_text()).get("status") == "complete":
                        status["jobs"][name] = "complete_reused"
                        save_status(status)
                        emit(f"Skipping already complete threshold {pct}%")
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            status["state"] = f"running_{name}"
            status["jobs"][name] = "running"
            save_status(status)
            result = docker_exec(["python", "-m", SCRIPT, "--threshold", str(pct / 100)], check=False)
            status["jobs"][name] = "complete" if result.returncode == 0 else "failed"
            status[f"{name}_exit_code"] = result.returncode
            save_status(status)
            if result.returncode == 0:
                try:
                    # Validate the saved per-threshold aggregate before publishing.
                    docker_exec(["python", "-c", f"from pathlib import Path; from experiments.kv_cache.experiments.stage3c_oracle_task_relevance import validate_summary; validate_summary(Path('experiments/kv_cache/outputs/stage3c_oracle_task_relevance/{name}/summary.json')); print('artifact validation passed')"])
                    git_sync([f"experiments/kv_cache/outputs/stage3c_oracle_task_relevance/{name}"],
                             f"Publish Stage 3C {pct}% task relevance results")
                except Exception as post_error:
                    status["jobs"][name] = "complete_publish_failed"
                    status[f"{name}_publish_error"] = repr(post_error)
                    save_status(status)
                    emit(f"Threshold {pct}% completed but validation/publication failed: {post_error!r}; continuing.")
            else:
                emit(f"Threshold {pct}% failed with exit={result.returncode}; preserving partials and continuing to next threshold.")
        status["state"] = "finalizing"
        status["jobs"]["finalization"] = "running"
        save_status(status)
        final = docker_exec(["python", "-m", SCRIPT, "--finalize-only"], check=False)
        status["jobs"]["finalization"] = "complete" if final.returncode == 0 else "failed"
        status["finalization_exit_code"] = final.returncode
        status["state"] = "complete" if final.returncode == 0 and all(
            status["jobs"][f"protect_{pct}"] in ("complete", "complete_reused", "complete_publish_failed") for pct in (10, 25, 50)
        ) else "incomplete"
        save_status(status)
        try:
            git_sync(["experiments/kv_cache/outputs/stage3c_oracle_task_relevance"],
                     "Publish Stage 3C task relevance aggregate report")
        except Exception as publish_error:
            status["state"] = "complete_local_unpublished" if status["state"] == "complete" else status["state"]
            status["final_push_error"] = repr(publish_error)
            save_status(status)
            emit(f"Final report push failed; local artifacts retained: {publish_error!r}")
        emit(f"QUEUE_FINISHED state={status['state']}")
        return 0 if status["state"] == "complete" else 1
    except Exception as exc:
        status["state"] = "supervisor_error"
        status["error"] = repr(exc)
        save_status(status)
        emit(f"SUPERVISOR_ERROR {exc!r}")
        try:
            git_sync(["experiments/kv_cache/outputs/stage3c_oracle_task_relevance/queue_status.json",
                      "experiments/kv_cache/outputs/stage3c_oracle_task_relevance/supervisor.log"],
                     "Record Stage 3C supervisor error")
        except Exception as push_error:
            emit(f"Could not publish failure status: {push_error!r}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
