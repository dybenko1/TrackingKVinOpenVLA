"""Unattended Stage 3B gate, matrix execution, report, and GitHub sync."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments/kv_cache/outputs/stage3b_oracle_full_kv_matrix"
RUNNER = "experiments/kv_cache/experiments/stage3b_oracle_full_kv_matrix.py"
CONTAINER = os.environ.get("STAGE3B_CONTAINER", "f5240fe761b5")
CODE_COMMIT = ""
interrupted = False


def on_signal(signum, _frame):
    global interrupted
    interrupted = True
    raise InterruptedError(f"supervisor received signal {signum}")


def log_run(args, log_path: Path):
    command = ["docker", "exec", CONTAINER, "bash", "-lc", f"cd /opt/openvla && {' '.join(args)}"]
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n"); log.flush()
        process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            assert process.stdout
            for line in process.stdout:
                sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise


def scenario_paths():
    return [OUT / f"task{task}_state{state}" / "results.json" for task in (4, 5, 7) for state in (0, 1, 2)]


def audit_summary(path: Path):
    data = json.loads(path.read_text())
    assert data.get("status") == "complete" and len(data.get("scenarios", [])) == 9
    for scenario in data["scenarios"]:
        assert scenario["fresh"]["all_valid"] and scenario["stage3b_true_skip"]["all_valid"]
        skip = scenario["stage3b_true_skip"]
        assert len(skip["skipped_k_rows_by_layer_aggregate"]) == 32
        assert len(skip["skipped_v_rows_by_layer_aggregate"]) == 32
        for row in skip["records"]:
            assert row["foreground_tokens"] + row["background_tokens"] == 256
            assert row["source_age"] == row["step"] % 5
            if row["step"] % 5:
                assert row["source_step"] == row["step"] - row["step"] % 5
                assert len(row["layer_instrumentation"]) == 32
                assert all(x["k_projected_rows"] == x["v_projected_rows"] == 1 + row["foreground_tokens"] + row["language_tokens"] for x in row["layer_instrumentation"])
                assert all(x["k_skipped_background_rows"] == x["v_skipped_background_rows"] == row["background_tokens"] for x in row["layer_instrumentation"])
    for p in scenario_paths():
        assert p.exists() and json.loads(p.read_text())
    return data


def make_report(data):
    stage3a_root = REPO / "experiments/kv_cache/outputs/stage3a_oracle_masked_matrix"
    rows = []
    a_fresh = a_reuse = b_fresh = b_reuse = 0
    for sc in data["scenarios"]:
        task, state = sc["task_id"], sc["state_id"]
        old = json.loads((stage3a_root / f"task{task}_state{state}" / "results.json").read_text())
        a, b = old["masked_true_skip"], sc["stage3b_true_skip"]
        assert old["fresh"]["seed"] == sc["fresh"]["seed"] == 7
        assert old["config"]["R"] == 5 and old["config"]["layers"] == list(range(14))
        a_fresh += old["fresh"]["success"]; a_reuse += a["success"]
        b_fresh += sc["fresh"]["success"]; b_reuse += b["success"]
        rows.append(f"| {task} | {state} | {'Success' if old['fresh']['success'] else 'Fail'} ({old['fresh']['steps']}) | {'Success' if a['success'] else 'Fail'} ({a['steps']}) | {'Success' if sc['fresh']['success'] else 'Fail'} ({sc['fresh']['steps']}) | {'Success' if b['success'] else 'Fail'} ({b['steps']}) | {b['refresh_count']} | {b['reuse_count']} | {b['reuse_fraction']*100:.1f}% | {b['total_skipped_k_projection_rows']:,} | {b['total_skipped_v_projection_rows']:,} |")
    table = "\n".join(rows)
    return f"""# Stage 3B: full-layer oracle-masked K/V reuse

## Run status

Complete. Nine task/state scenarios ran with paired Fresh and Stage 3B policies. No latency benchmark was performed. The result commit contains per-step raw records and this aggregate report.

## Configuration and implementation revision

- Suite: LIBERO Spatial; tasks 4, 5, 7; states 0, 1, 2.
- Seed 7, official-style 10 settling steps, horizon 220, greedy OpenVLA checkpoint `openvla/openvla-7b-finetuned-libero-spatial`.
- R=5; refresh at `step % 5 == 0`; age 1–4 reuses the most recent independent fresh source.
- All 32 Llama layers; visual cache positions 1–256; background K and V reused; foreground K and V projected fresh; Q full sequence; BOS/text fresh; action decoding delegated normally.
- Implementation commit: `{CODE_COMMIT}`.
- Exact commands: `python {RUNNER} --validate-only`; after a passing validation, `python {RUNNER} --matrix` (both run inside container `{CONTAINER}`, working directory `/opt/openvla`).

## Results

| Task | State | Stage 3A Fresh (steps) | Stage 3A K-only (steps) | Stage 3B Fresh (steps) | Stage 3B K+V (steps) | Refresh | Reuse | Temporal reuse | K rows omitted | V rows omitted |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
{table}

Stage 3A success counts: Fresh **{a_fresh}/9**, K-only masked reuse **{a_reuse}/9**. Stage 3B: Fresh **{b_fresh}/9**, full-layer K+V masked reuse **{b_reuse}/9**. For each Stage 3B scenario, `summary.json` reports skipped projection row totals by layer, aggregate K/V rows, effective K/V and combined row fractions, spatial background-token fraction, and temporal reuse fraction. The approximately 80% temporal reuse rate is distinct from the foreground-mask spatial fraction and the actual projection-row fraction. No measured latency is included.

## Reproduction and validation

The supervisor first ran the six-step task-4/state-0 same-observation/source comparison between compute-then-overwrite and true skip. It required all 32 layers to be instrumented, exact stale-background K/V cache provenance and cache shape agreement, full Q rows, original-position RoPE for fresh K, unrotated V, delegated one-token decoding, no unexpected fallbacks, finite actions, and valid R5 source ages. Validation JSON is `validation.json`.

After validation passed, the supervisor ran the full matrix sequentially. Each scenario JSON was saved immediately, and `summary.json` was updated after every scenario. Reproduce the full sequence with `python experiments/kv_cache/experiments/supervise_stage3b.py` on the lab host. The supervisor performs post-run integrity checks, writes a compact status log excerpt and file-hash manifest, then commits and pushes the artifacts.

Result files and hashes are listed in `manifest.json`; the compact execution excerpt is `execution_excerpt.log`. Runtime log is `supervisor.log` locally.
"""


def finalize(status: str, message: str):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_status.json").write_text(json.dumps({"status": status, "message": message, "experiment_code_commit": CODE_COMMIT}, indent=2))
    (OUT / "REPORT.md").write_text(f"# Stage 3B run status\n\nStatus: **{status.upper()}**.\n\n{message}\n\nImplementation commit: `{CODE_COMMIT}`.\n")
    log = OUT / "supervisor.log"
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        selected = [line for line in lines if "[VALIDATION_PASS]" in line or "[SCENARIO_COMPLETE]" in line or "Traceback" in line or "AssertionError" in line or "RuntimeError" in line or "$ docker exec" in line]
        (OUT / "execution_excerpt.log").write_text("\n".join(selected[-100:]) + "\n")
    paths = [p for p in [OUT / "summary.json", OUT / "validation.json", OUT / "REPORT.md", OUT / "run_status.json", OUT / "execution_excerpt.log"] + scenario_paths() if p.exists()]
    manifest = [{"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


def commit_push_results():
    files = [OUT / "summary.json", OUT / "validation.json", OUT / "REPORT.md", OUT / "run_status.json", OUT / "execution_excerpt.log", OUT / "manifest.json"] + scenario_paths()
    existing = [str(p.relative_to(REPO)) for p in files if p.exists()]
    if not existing:
        raise RuntimeError("No Stage 3B artifacts exist to commit.")
    subprocess.run(["git", "add", "-f", *existing], cwd=REPO, check=True)
    subprocess.run(["git", "diff", "--cached", "--check"], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-m", "results(kv-cache): publish Stage 3B oracle-masked full-layer K/V matrix"], cwd=REPO, check=True)
    results_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    (OUT / "push_metadata.json").write_text(json.dumps({"branch": "main", "experiment_code_commit": CODE_COMMIT, "results_commit": results_commit}, indent=2))
    subprocess.run(["git", "add", "-f", str((OUT / "push_metadata.json").relative_to(REPO))], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-m", "docs(kv-cache): record Stage 3B results commit hash"], cwd=REPO, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=REPO, check=True)


def main():
    global CODE_COMMIT
    signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)
    OUT.mkdir(parents=True, exist_ok=True)
    CODE_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise RuntimeError("Working tree must be clean before unattended Stage 3B supervisor starts.")
    log_path = OUT / "supervisor.log"
    matrix_started = False
    try:
        status = log_run(["python", RUNNER, "--validate-only"], log_path)
        if status != 0:
            raise RuntimeError(f"Stage 3B validation exited with status {status}; matrix was not launched.")
        validation_path = OUT / "validation.json"
        validation = json.loads(validation_path.read_text())
        assert validation["all_valid"] and validation["steps"] == 6 and validation["layers"] == list(range(32))
        with log_path.open("a") as f: f.write("[VALIDATION_PASS] gate accepted; launching matrix\n")
        matrix_started = True
        status = log_run(["python", RUNNER, "--matrix"], log_path)
        if status != 0:
            raise RuntimeError(f"Stage 3B matrix exited with status {status}.")
        data = audit_summary(OUT / "summary.json")
        (OUT / "REPORT.md").write_text(make_report(data))
        (OUT / "run_status.json").write_text(json.dumps({"status": "complete", "scenarios_completed": 9,
            "experiment_code_commit": CODE_COMMIT, "latency_measured": False}, indent=2))
        lines = log_path.read_text(errors="replace").splitlines()
        excerpt = [line for line in lines if "[VALIDATION_PASS]" in line or "[SCENARIO_COMPLETE]" in line]
        (OUT / "execution_excerpt.log").write_text("\n".join(excerpt) + "\n")
        files = [OUT / "summary.json", OUT / "validation.json", OUT / "REPORT.md", OUT / "run_status.json", OUT / "execution_excerpt.log"] + scenario_paths()
        manifest = [{"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
        commit_push_results()
    except BaseException as exc:
        finalize("incomplete", f"{type(exc).__name__}: {exc}. Check supervisor.log and scenario files; incomplete results are not described as completed.")
        if matrix_started:
            try:
                commit_push_results()
            except BaseException as push_exc:
                with (OUT / "supervisor.log").open("a") as f: f.write(f"\n[POSTRUN_SYNC_FAILED] {type(push_exc).__name__}: {push_exc}\n")
        raise

if __name__ == "__main__":
    main()
