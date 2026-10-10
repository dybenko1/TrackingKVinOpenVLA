# Stage 3 research queue

`stage3_research_queue.py` is the persistent host-side supervisor for the Track B → Stage 3C → Stage 3D → factorial-analysis queue. It runs jobs serially, keeps incremental scenario outputs, records status/exit codes in `experiments/kv_cache/outputs/stage3_research_queue/queue_status.json`, and appends durable output to `queue.log`.

The supervisor is intended to run inside the `stage3-research-queue` tmux session. It survives SSH disconnects while the host remains running; it does **not** configure host-reboot recovery. A process lock prevents a duplicate queue supervisor. Stage 3C/3D scenario outputs are independently resumable and are committed/pushed after each completed matrix. Track B uses its existing output validation and result-sync supervisor. The factorial report is marked incomplete if any required scenario is missing or invalid.

From the repository root, monitor with:

```bash
tmux attach -t stage3-research-queue
cat experiments/kv_cache/outputs/stage3_research_queue/queue_status.json
tail -f experiments/kv_cache/outputs/stage3_research_queue/queue.log
```

If the host reboots or the tmux session is lost, first verify that no queue/experiment process remains, then restart with:

```bash
tmux new-session -d -s stage3-research-queue 'python3 experiments/kv_cache/experiments/stage3_research_queue.py'
```

The script resumes completed Track B artifacts and valid completed Stage 3C/3D scenario files; it does not overwrite incompatible scenario artifacts. Review `queue_status.json` before restarting. Git operations target only explicitly enumerated queue and experiment artifacts on the existing `main` branch; force-push is never used.
