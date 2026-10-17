"""Publish ignored Stage 3C outputs once each threshold is validated.

The global repository ignores experiments/kv_cache/outputs, so this small
host-side watcher force-adds only the Stage 3C result directory after the main
supervisor completes its validation/publish attempt for each threshold.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments" / "kv_cache" / "outputs" / "stage3c_oracle_task_relevance"
STATUS = Path("/tmp/stage3c_oracle_task_relevance_queue_status.json")
LOG = Path("/tmp/stage3c_oracle_task_relevance_publisher.log")
PUBLISHED = Path("/tmp/stage3c_oracle_task_relevance_published.json")
LOCK = Path("/tmp/stage3c-result-git.lock")
BRANCH = "main"
THRESHOLDS = (10, 25, 50)


def emit(message):
    line = f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {message}"
    with LOG.open("a") as handle:
        handle.write(line + "\n"); handle.flush()
    print(line, flush=True)


def command(args, *, check=True):
    result = subprocess.run(args, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.stdout:
        emit(result.stdout.strip())
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args)
    return result


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def validate_threshold(path: Path) -> bool:
    data = load_json(path)
    if not data or data.get("status") != "complete" or len(data.get("scenarios", [])) != 9:
        return False
    seen = set()
    for scenario in data["scenarios"]:
        key = (scenario.get("task_id"), scenario.get("state_id"))
        if key in seen or not scenario.get("all_valid") or scenario.get("status") != "complete":
            return False
        seen.add(key)
        if len(scenario.get("records", [])) != scenario.get("steps"):
            return False
        if not scenario.get("records") or scenario.get("max_source_age", 0) > 4:
            return False
        if not scenario.get("fresh_reference") or not scenario.get("stage3b_reference"):
            return False
        for row in scenario["records"]:
            if row["foreground_tokens"] + row["background_tokens"] != 256:
                return False
            if row["type"] == "reuse" and row["source_age"] not in (1, 2, 3, 4):
                return False
            if len(row["skipped_k_rows_per_layer"]) != 32 or len(row["skipped_v_rows_per_layer"]) != 32:
                return False
    return len(seen) == 9


def publish(paths: list[str], message: str) -> str | None:
    lock_handle = LOCK.open("a+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    try:
        branch = command(["git", "branch", "--show-current"]).stdout.strip()
        if branch != BRANCH:
            raise RuntimeError(f"Expected {BRANCH}, current branch is {branch}.")
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO)
        if staged.returncode:
            raise RuntimeError("Git index already contains staged changes; publication deferred to avoid mixing work.")
        command(["git", "add", "-f", "--", *paths])
        command(["git", "diff", "--cached", "--check"])
        changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO)
        if changed.returncode == 0:
            emit(f"No tracked changes to publish for {message}.")
            return None
        command(["git", "commit", "-m", message])
        command(["git", "push", "origin", BRANCH])
        return command(["git", "rev-parse", "HEAD"]).stdout.strip()
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def main():
    published = load_json(PUBLISHED) or {"threshold_shas": {}, "final_sha": None}
    emit("Watching Stage 3C threshold outputs for validated completion.")
    while True:
        status = load_json(STATUS) or {}
        for pct in THRESHOLDS:
            key = str(pct)
            result_dir = OUT / f"protect_{pct:02d}"
            if key in published["threshold_shas"] or not validate_threshold(result_dir / "summary.json"):
                continue
            # Main supervisor makes a no-op attempt because outputs are ignored.
            time.sleep(15)
            try:
                sha = publish([str(result_dir.relative_to(REPO))],
                              f"Publish Stage 3C {pct}% task relevance results")
                published["threshold_shas"][key] = sha or "already-published"
                PUBLISHED.write_text(json.dumps(published, indent=2))
                emit(f"Threshold {pct}% validation/publication complete sha={sha}")
            except Exception as exc:
                emit(f"Threshold {pct}% publication failed; outputs retained locally: {exc!r}")
        root = load_json(OUT / "summary.json") or {}
        final = status.get("state") in ("complete", "incomplete", "complete_local_unpublished")
        if final and root.get("status") in ("complete", "incomplete") and not published.get("final_sha"):
            time.sleep(15)
            try:
                sha = publish([str(OUT.relative_to(REPO))], "Publish Stage 3C task relevance aggregate report")
                published["final_sha"] = sha or "already-published"
                PUBLISHED.write_text(json.dumps(published, indent=2))
                emit(f"Final Stage 3C artifacts published sha={sha}")
            except Exception as exc:
                emit(f"Final Stage 3C publication failed; artifacts retained locally: {exc!r}")
        if final:
            break
        time.sleep(20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
