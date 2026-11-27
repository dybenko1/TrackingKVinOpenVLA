#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER=trackb-v1-coco-object-20261126
WT=/opt/openvla/cache/worktrees/trackb-dense-grounding-v2
MOD=experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.run_coco
OUT=/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_baseline_validation/coco_object
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"
printf '%s\n' "launcher_pid=$$" "started=$(date -Is)" > "$LOGDIR/launcher.pid"

echo "[$(date -Is)] smoke test starting in $CONTAINER"
docker exec "$CONTAINER" bash -lc "export PYTHONPATH=/opt/openvla/cache/venvs/v1-coco-eval-opencv:$WT; cd $WT; python -u -m $MOD --smoke"
echo "[$(date -Is)] smoke test passed; full 5000-image evaluation starting"
docker exec "$CONTAINER" bash -lc "export PYTHONPATH=/opt/openvla/cache/venvs/v1-coco-eval-opencv:$WT; cd $WT; python -u -m $MOD"
echo "[$(date -Is)] COCO evaluation command finished"
