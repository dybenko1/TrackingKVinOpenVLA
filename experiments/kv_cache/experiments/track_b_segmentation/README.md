# Track B: native-feature segmentation

This module benchmarks token-level foreground/background masks from the installed OpenVLA vision branches. It replays Stage 3A Fresh action traces to reconstruct LIBERO frames and oracle instance masks, then runs only the loaded OpenVLA vision backbone. It does not generate actions, run policy rollouts, or change KV-cache behavior.

## Commands

Run the CPU smoke check:

```bash
python -m experiments.kv_cache.experiments.track_b_segmentation.main smoke-test
```

Run all B1/B2 stages after Stage 3B is complete:

```bash
python -m experiments.kv_cache.experiments.track_b_segmentation.main run --code-commit <implementation-commit>
```

The persistent supervisor is:

```bash
python -m experiments.kv_cache.experiments.track_b_segmentation.supervise \
  --container f5240fe761b5 \
  --stage3b-supervisor-pid 1683056 \
  --code-commit <implementation-commit>
```

The host supervisor waits for the existing Stage 3B supervisor process and its matrix child, checks CUDA availability/free memory and other active containers, invokes the runner inside the existing OpenVLA container, validates outputs, and commits/pushes only lightweight Track B artifacts. Raw frames, masks, features, and score arrays remain local under `experiments/kv_cache/outputs/track_b_segmentation/data/`.

## Progress and result paths

The durable status is `outputs/track_b_segmentation/run_status.json`. The full local logs are `outputs/track_b_segmentation/logs/supervisor.log` and `execution.log`. The final `REPORT.md`, CSV metrics, dataset manifest, feature mapping report, and figures are written under `outputs/track_b_segmentation/`.

To inspect a persistent run after reconnecting:

```bash
tmux ls
tmux attach -t track-b-segmentation
tail -f experiments/kv_cache/outputs/track_b_segmentation/logs/supervisor.log
cat experiments/kv_cache/outputs/track_b_segmentation/run_status.json
```
