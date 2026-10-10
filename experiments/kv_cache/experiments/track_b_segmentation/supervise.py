"""Persistent Track B supervisor with an explicit Stage 3B completion gate and GitHub sync."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .dataset import OUT, sha256

REPO = Path(__file__).resolve().parents[4]
CONTAINER_DEFAULT = os.environ.get("OPENVLA_CONTAINER", "f5240fe761b5")
STAGE3B_SUPERVISOR_PID_DEFAULT = 1683056
STAGE3B_SUPERVISOR_COMMAND = "experiments/kv_cache/experiments/supervise_stage3b.py"
STATUS_PATH = OUT / "run_status.json"
SUPERVISOR_LOG = OUT / "logs" / "supervisor.log"
EXECUTION_LOG = OUT / "logs" / "execution.log"


def log(message: str) -> None:
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {message}"
    print(text, flush=True)
    try:
        _container_append(SUPERVISOR_LOG, text + "\n")
    except BaseException:
        # A local fallback helps diagnose a lost container, while output artifacts remain container-owned.
        with (Path("/tmp") / "track_b_segmentation_supervisor.log").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def _container_path(path: Path) -> str:
    relative = path.relative_to(REPO)
    return str(Path("/opt/openvla") / relative)


def _container_mkdir(path: Path, container: str | None = None) -> None:
    result = subprocess.run(["docker", "exec", container or CONTAINER_DEFAULT, "mkdir", "-p", _container_path(path)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Could not create container output directory: {result.stdout.strip()}")


def _container_write_text(path: Path, content: str, append: bool = False, container: str | None = None) -> None:
    selected_container = container or CONTAINER_DEFAULT
    _container_mkdir(path.parent, selected_container)
    code = (
        "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); "
        + ("p.open('a',encoding='utf-8').write(sys.stdin.read())" if append
           else "tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(sys.stdin.read(),encoding='utf-8'); tmp.replace(p)")
    )
    result = subprocess.run(["docker", "exec", "-i", selected_container, "python", "-c", code, _container_path(path)],
                            input=content, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Could not write container output {path}: {result.stdout.strip()}")


def _container_append(path: Path, content: str, container: str | None = None) -> None:
    _container_write_text(path, content, append=True, container=container)


def _container_atomic_json(path: Path, value: Any, container: str | None = None) -> None:
    _container_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n", container=container)


def set_status(status: str, **metadata: Any) -> None:
    current = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}
    current.update({"status": status, "updated_at_unix": time.time(), **metadata})
    _container_atomic_json(STATUS_PATH, current)


def git(*args: str, capture: bool = False) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    return result.stdout.strip() if capture and result.stdout else ""


def _process_command(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return None
    except PermissionError as exc:
        raise RuntimeError(f"Cannot inspect dependency PID {pid}: {exc}") from exc
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _container_stage3b_processes(container: str) -> list[str]:
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-lc", "ps -eo pid=,args="],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect Stage 3B container processes: {result.stdout.strip()}")
    return [line.strip() for line in result.stdout.splitlines()
            if "stage3b_oracle_full_kv_matrix.py --matrix" in line]


def wait_for_stage3b(pid: int, expected_command: str, container: str) -> dict[str, Any]:
    initial = _process_command(pid)
    if initial is not None and expected_command not in initial:
        raise RuntimeError(f"Stage 3B dependency PID {pid} has unexpected command: {initial}")
    log(f"Waiting on Stage 3B supervisor PID {pid}; expected command contains {expected_command!r}.")
    set_status("pending", phase="waiting_for_stage3b", stage3b_supervisor_pid=pid,
               stage3b_container=container, process_completion_dependency=True)
    while True:
        command = _process_command(pid)
        if command is None or expected_command not in command:
            break
        log(f"Stage 3B supervisor remains active as PID {pid}; no Track B compute is running.")
        time.sleep(30)
    # The parent supervisor is the process that performs Stage 3B result audit and GitHub finalization.
    # Also wait for its container matrix child to disappear before checking resources.
    while True:
        processes = _container_stage3b_processes(container)
        if not processes:
            break
        log(f"Stage 3B matrix process remains active in {container}: {processes}; Track B remains pending.")
        time.sleep(15)
    status_path = REPO / "experiments/kv_cache/outputs/stage3b_oracle_full_kv_matrix/run_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "missing"}
    log(f"Stage 3B supervisor exited; final visible Stage 3B status is {status.get('status')}.")
    return {"supervisor_pid": pid, "final_status": status, "completed_at_unix": time.time()}


def _docker_resource_snapshot(container: str) -> dict[str, Any]:
    name_result = subprocess.run(["docker", "inspect", "--format", "{{.Name}}", container],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if name_result.returncode != 0:
        raise RuntimeError(f"Could not resolve target container name: {name_result.stdout.strip()}")
    target_name = name_result.stdout.strip().lstrip("/")
    stats = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.CPUPerc}} {{.MemUsage}} {{.PIDs}}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if stats.returncode != 0:
        raise RuntimeError(f"docker stats failed: {stats.stdout.strip()}")
    top = subprocess.run(["docker", "top", container, "-eo", "pid,ppid,stat,etime,%cpu,%mem,args"],
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if top.returncode != 0:
        raise RuntimeError(f"docker top failed: {top.stdout.strip()}")
    probe = r'''python - <<'PY'
import json
import torch
available = bool(torch.cuda.is_available())
payload = {"cuda_available": available, "device_count": torch.cuda.device_count() if available else 0}
if available:
    free, total = torch.cuda.mem_get_info(0)
    payload.update({"device_name": torch.cuda.get_device_name(0), "free_bytes": int(free), "total_bytes": int(total)})
print("TRACK_B_RESOURCE_JSON=" + json.dumps(payload))
PY'''
    device = subprocess.run(["docker", "exec", container, "bash", "-lc", probe],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    marker = next((line.split("=", 1)[1] for line in device.stdout.splitlines()
                   if line.startswith("TRACK_B_RESOURCE_JSON=")), None)
    snapshot = {
        "docker_stats": stats.stdout.strip().splitlines(),
        "target_container_processes": top.stdout.strip().splitlines(),
        "cuda_probe_return_code": device.returncode,
        "cuda_probe_output": device.stdout.strip(),
        "cuda": json.loads(marker) if marker else None,
        "cuda_probe_completed_at_unix": time.time(),
    }
    if device.returncode != 0 or snapshot["cuda"] is None:
        raise RuntimeError("CUDA resource probe did not return usable device and memory information: " + device.stdout.strip())
    if not snapshot["cuda"]["cuda_available"]:
        raise RuntimeError("CUDA is unavailable after Stage 3B; Track B will not fall back to CPU execution.")
    if snapshot["cuda"]["free_bytes"] < 6 * 1024**3:
        raise RuntimeError(f"Only {snapshot['cuda']['free_bytes'] / 1024**3:.2f} GiB CUDA memory is free; 6 GiB is required.")
    # These containers are shared with unrelated work. High memory or CPU use is a conservative signal to defer.
    for line in snapshot["docker_stats"]:
        if line.split(maxsplit=1)[0] == target_name:
            continue
        fields = line.split()
        if len(fields) >= 4:
            try:
                cpu_pct = float(fields[1].rstrip("%"))
                memory_text = fields[2]
                if "GiB" in memory_text:
                    memory_gib = float(memory_text.split("GiB", 1)[0])
                elif "MiB" in memory_text:
                    memory_gib = float(memory_text.split("MiB", 1)[0]) / 1024
                elif "KiB" in memory_text:
                    memory_gib = float(memory_text.split("KiB", 1)[0]) / (1024 ** 2)
                else:
                    memory_gib = 0.0
            except ValueError:
                continue
            if cpu_pct > 20 or memory_gib > 4:
                raise RuntimeError(f"Other Docker container appears resource-active; deferring Track B: {line}")
    return snapshot


def _run_track_b(container: str, code_commit: str) -> int:
    execution_path = _container_path(EXECUTION_LOG)
    runner = (
        "cd /opt/openvla && mkdir -p experiments/kv_cache/outputs/track_b_segmentation/logs && "
        "set -o pipefail && python -m experiments.kv_cache.experiments.track_b_segmentation.main "
        f"run --code-commit {shlex.quote(code_commit)} 2>&1 | tee -a {shlex.quote(execution_path)}"
    )
    command = [
        "docker", "exec", container, "bash", "-lc", runner,
    ]
    process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return process.wait()


def _run_container_smoke_test(container: str) -> dict[str, Any]:
    path = _container_path(OUT / "logs" / "smoke_test.log")
    command = [
        "docker", "exec", container, "bash", "-lc",
        "cd /opt/openvla && mkdir -p experiments/kv_cache/outputs/track_b_segmentation/logs && "
        "set -o pipefail && python -m experiments.kv_cache.experiments.track_b_segmentation.main "
        f"smoke-test 2>&1 | tee -a {shlex.quote(path)}",
    ]
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        set_status("failed", phase="target_container_smoke_test", smoke_test_exit_code=result.returncode,
                   error=result.stdout.strip()[-2000:])
        raise RuntimeError(f"Target-container CPU smoke test exited {result.returncode}: {result.stdout.strip()}")
    try:
        data = json.loads((OUT / "logs" / "smoke_test.json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Target-container smoke test did not produce valid JSON: {exc}") from exc
    if data.get("status") != "passed" or data.get("model_forwards_run") or data.get("action_generation_run"):
        raise RuntimeError(f"Target-container smoke test returned an invalid result: {data}")
    log(f"Target-container CPU smoke test passed for {data['sampled_observations_available']} replayable observations.")
    return data


def _validate_outputs() -> dict[str, Any]:
    required = [
        "summary.json", "metrics.csv", "per_sample_metrics.csv", "per_trajectory_metrics.csv",
        "threshold_sweep.csv", "temporal_consistency.csv", "calibration_thresholds.csv",
        "dataset_manifest.json", "configuration.json", "feature_mapping_report.md", "REPORT.md",
    ]
    missing = [name for name in required if not (OUT / name).is_file() or (OUT / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Required Track B outputs are missing or empty: {missing}")
    summary = json.loads((OUT / "summary.json").read_text())
    manifest = json.loads((OUT / "dataset_manifest.json").read_text())
    if summary.get("status") != "complete" or manifest.get("sample_count", 0) < 200:
        raise RuntimeError("Summary or dataset manifest does not indicate a validated complete benchmark.")
    if manifest.get("feature_extraction", {}).get("status") != "complete":
        raise RuntimeError("Dataset manifest does not indicate complete feature extraction.")
    for row in manifest["samples"]:
        feature_path = OUT / row["feature_path"]
        if not feature_path.is_file() or row.get("feature_status") != "complete":
            raise RuntimeError(f"Missing native feature file for {row['sample_id']}.")
        if row.get("feature_sha256") and sha256(feature_path) != row["feature_sha256"]:
            raise RuntimeError(f"Feature checksum mismatch for {row['sample_id']}.")
        for field, digest_field in (("rgb_observation_path", "rgb_observation_sha256"),
                                    ("rgb_path", "policy_rgb_sha256"),
                                    ("instance_segmentation_path", "instance_segmentation_sha256"),
                                    ("oracle_semantic_mask_path", "oracle_semantic_mask_sha256"),
                                    ("oracle_token_mask_path", "oracle_token_mask_sha256")):
            artifact = OUT / row[field]
            expected = row.get("artifact_sha256", {}).get(digest_field)
            if not artifact.is_file() or not expected or sha256(artifact) != expected:
                raise RuntimeError(f"Missing or checksum-invalid {field} for {row['sample_id']}.")
    figure_paths = list((OUT / "figures").glob("*.png")) + list((OUT / "figures").glob("*.pdf"))
    required_figures = [
        "segmentation_overlays.png", "error_maps.png", "dilation_comparison.png",
        "reuse_opportunity_vs_unsafe_fg_reuse.png", "feature_mapping_overlay.png", "comparison_report.pdf",
    ]
    missing_figures = [name for name in required_figures if not (OUT / "figures" / name).is_file()]
    if missing_figures or len(list((OUT / "figures").glob("temporal_*.png"))) < 3:
        raise RuntimeError(f"Required comparison/temporal figures are missing: {missing_figures}")
    with (OUT / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    if len(metric_rows) != 15:
        raise RuntimeError(f"Expected 15 method/variant metric rows, got {len(metric_rows)}.")
    return {
        "sample_count": manifest["sample_count"],
        "figure_count": len(figure_paths),
        "metric_rows": len(metric_rows),
        "required_files": required,
        "validation": "passed",
    }


def _tracked_result_files() -> list[Path]:
    stable = [
        "summary.json", "metrics.csv", "per_sample_metrics.csv", "per_trajectory_metrics.csv",
        "threshold_sweep.csv", "temporal_consistency.csv", "calibration_thresholds.csv",
        "dataset_manifest.json", "configuration.json", "feature_mapping_report.md", "REPORT.md",
        "run_status.json", "push_metadata.json",
    ]
    paths = [OUT / name for name in stable]
    paths += sorted((OUT / "figures").glob("*.png"))
    paths += sorted((OUT / "figures").glob("*.pdf"))
    paths += [OUT / "logs" / "smoke_test.json", OUT / "logs" / "smoke_test.log", OUT / "logs" / "resource_snapshot.json",
              OUT / "logs" / "execution_excerpt.log"]
    return [path for path in paths if path.is_file()]


def _commit_and_push(code_commit: str, validation: dict[str, Any]) -> dict[str, Any]:
    branch = git("branch", "--show-current", capture=True)
    if branch != "main":
        raise RuntimeError(f"Expected existing main branch, found {branch!r}.")
    remote = git("remote", "get-url", "origin", capture=True)
    before = git("status", "--porcelain", capture=True)
    # User changes may exist in parallel; only explicit Track B paths are ever staged.
    if git("diff", "--cached", "--name-only", capture=True):
        raise RuntimeError("The Git index already contains staged changes; refusing to commit Track B outputs.")
    result_paths = [p for p in _tracked_result_files() if p.name not in ("run_status.json", "push_metadata.json")]
    relative = [str(path.relative_to(REPO)) for path in result_paths]
    if not relative:
        raise RuntimeError("No validated lightweight result files are available to commit.")
    git("add", "-f", *relative)
    git("diff", "--cached", "--check")
    git("commit", "-m", "results(kv-cache): publish Track B native-feature segmentation benchmark")
    results_commit = git("rev-parse", "HEAD", capture=True)

    metadata = {
        "branch": "main",
        "remote": remote,
        "repository_url": remote.replace("git@github.com:", "https://github.com/").removesuffix(".git"),
        "implementation_commit": code_commit,
        "results_commit": results_commit,
        "output_validation": validation,
        "raw_features_committed": False,
        "feature_arrays_local_path": str((OUT / "data" / "samples").relative_to(REPO)),
        "sync_status": "pending",
        "created_at_unix": time.time(),
    }
    _container_atomic_json(OUT / "push_metadata.json", metadata)
    set_status("completed", phase="awaiting_github_sync", code_commit=code_commit,
               results_commit=results_commit, github_sync_status="pending", output_validation=validation)
    sync_paths = [OUT / "push_metadata.json", STATUS_PATH]
    git("add", "-f", *[str(path.relative_to(REPO)) for path in sync_paths])
    git("diff", "--cached", "--check")
    git("commit", "-m", "docs(kv-cache): record Track B result synchronization metadata")
    metadata_commit = git("rev-parse", "HEAD", capture=True)

    try:
        git("push", "origin", "main")
    except BaseException as exc:
        log(f"[GITHUB_SYNC_FAILED] {type(exc).__name__}: {exc}")
        metadata["sync_status"] = "failed"
        metadata["sync_error"] = f"{type(exc).__name__}: {exc}"
        metadata["metadata_commit"] = metadata_commit
        _container_atomic_json(OUT / "push_metadata.json", metadata)
        set_status("failed", phase="github_sync", code_commit=code_commit,
                   results_commit=results_commit, metadata_commit=metadata_commit,
                   github_sync_status="failed", github_sync_error=metadata["sync_error"])
        git("add", "-f", str((OUT / "push_metadata.json").relative_to(REPO)), str(STATUS_PATH.relative_to(REPO)))
        git("commit", "-m", "docs(kv-cache): record Track B GitHub synchronization failure")
        raise

    metadata["sync_status"] = "pushed"
    metadata["metadata_commit"] = metadata_commit
    metadata["push_succeeded_at_unix"] = time.time()
    _container_atomic_json(OUT / "push_metadata.json", metadata)
    set_status("completed", phase="github_synced", code_commit=code_commit,
               results_commit=results_commit, metadata_commit=metadata_commit,
               github_sync_status="pushed", repository_url=metadata["repository_url"])
    git("add", "-f", str((OUT / "push_metadata.json").relative_to(REPO)), str(STATUS_PATH.relative_to(REPO)))
    git("diff", "--cached", "--check")
    git("commit", "-m", "docs(kv-cache): confirm Track B GitHub synchronization")
    sync_confirmation_commit = git("rev-parse", "HEAD", capture=True)
    git("push", "origin", "main")
    log(f"[GITHUB_SYNC_SUCCESS] results={results_commit} metadata={metadata_commit} final_sync_commit={sync_confirmation_commit} remote={metadata['repository_url']}")
    return {"results_commit": results_commit, "metadata_commit": metadata_commit,
            "sync_confirmation_commit": sync_confirmation_commit, "remote": remote,
            "repository_url": metadata["repository_url"], "git_status_before_result_commit": before}


def run(args: argparse.Namespace) -> None:
    _container_mkdir(OUT / "logs", args.container)
    code_commit = args.code_commit or git("rev-parse", "HEAD", capture=True)
    git_status = git("status", "--porcelain", capture=True)
    if git_status:
        raise RuntimeError("Track B must start from a clean committed implementation; worktree is not clean: " + git_status)
    dependency = wait_for_stage3b(args.stage3b_supervisor_pid, STAGE3B_SUPERVISOR_COMMAND, args.container)
    set_status("running", phase="target_container_smoke_test", code_commit=code_commit, stage3b_dependency=dependency)
    _run_container_smoke_test(args.container)
    snapshot = _docker_resource_snapshot(args.container)
    _container_atomic_json(OUT / "logs" / "resource_snapshot.json", snapshot, args.container)
    set_status("running", phase="vision_feature_extraction", code_commit=code_commit,
               stage3b_dependency=dependency, resource_snapshot_path=str((OUT / "logs" / "resource_snapshot.json").relative_to(REPO)))
    log(f"Resource gate passed: CUDA {snapshot['cuda']['device_name']}, {snapshot['cuda']['free_bytes'] / 1024**3:.2f} GiB free.")
    code = _run_track_b(args.container, code_commit)
    if code != 0:
        set_status("failed", phase="runner", runner_exit_code=code)
        raise RuntimeError(f"Track B dataset/evaluation process exited with status {code}.")
    set_status("running", phase="output_validation", runner_exit_code=0)
    validation = _validate_outputs()
    log(f"Track B output validation passed: {validation}.")
    # A short excerpt is committed; the full supervisor/extraction logs remain local and durable.
    excerpt_candidates = [SUPERVISOR_LOG, EXECUTION_LOG]
    excerpt_lines: list[str] = []
    for path in excerpt_candidates:
        if path.exists():
            for line in path.read_text(errors="replace").splitlines():
                if any(term in line for term in ("[track_b]", "[GITHUB_SYNC", "Traceback", "RuntimeError", "passed", "[")):
                    excerpt_lines.append(line[-500:])
    _container_write_text(OUT / "logs" / "execution_excerpt.log", "\n".join(excerpt_lines[-200:]) + "\n",
                          container=args.container)
    sync = _commit_and_push(code_commit, validation)
    set_status("completed", phase="github_synced", code_commit=code_commit,
               results_commit=sync["results_commit"], metadata_commit=sync["metadata_commit"],
               sync_confirmation_commit=sync["sync_confirmation_commit"], github_sync_status="pushed",
               repository_url=sync["repository_url"], output_validation=validation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default=CONTAINER_DEFAULT)
    parser.add_argument("--stage3b-supervisor-pid", type=int, default=STAGE3B_SUPERVISOR_PID_DEFAULT)
    parser.add_argument("--code-commit", default=None)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        log(f"[TRACK_B_FAILED] {error}")
        set_status("failed", phase="supervisor", error=error)
        raise


if __name__ == "__main__":
    main()
