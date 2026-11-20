#!/usr/bin/env bash
set -uo pipefail

CONTAINER="trackb-v2-libero-e2-runtime"
IMAGE="openvla-talk2dino-siglip:20261116"
TREE="/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/worktrees/trackb-dense-grounding-v2"
OUT="$TREE/experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/libero_ground_truth_e2_completion"
DATA="/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/datasets/libero_ground_truth_v2_official"
CACHE="/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/datasets/libero_ground_truth_v2_evalcache"
mkdir -p "$OUT/logs" "$DATA" "$CACHE"
exec 9>"$OUT/.worker.lock"
if ! flock -n 9; then
  echo "A LIBERO E2 worker already holds $OUT/.worker.lock" >&2
  exit 73
fi
printf '%s\n' "$$" > "$OUT/host_worker.pid"
set +e
docker run --rm --name "$CONTAINER" --runtime nvidia --network host --group-add 2011 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e MPLCONFIGDIR=/tmp/matplotlib-libero-e2 \
  -v /home/oscar/projects/TrackingKVinOpenVLA/openvla:/opt/openvla \
  -v /home/oscar/projects/TrackingKVinOpenVLA/LIBERO:/opt/LIBERO \
  -v /home/oscar/projects/TrackingKVinOpenVLA/dusty-robosuite:/opt/robosuite \
  -v /tmp:/tmp \
  -w /opt/openvla \
  -e NUMBA_CACHE_DIR=/tmp/numba-cache-libero-e2 \
  -e LIBERO_CONFIG_PATH=/opt/openvla/cache/libero_config_e2 \
  -e PYTHONPATH=/opt/openvla/cache/worktrees/trackb-dense-grounding-v2 \
  "$IMAGE" bash -lc '
    mkdir -p "$NUMBA_CACHE_DIR"
    mkdir -p "$LIBERO_CONFIG_PATH"
    if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
      cat > "$LIBERO_CONFIG_PATH/config.yaml" <<"YAML"
benchmark_root: /opt/LIBERO/libero/libero
bddl_files: /opt/LIBERO/libero/libero/bddl_files
init_states: /opt/LIBERO/libero/libero/init_files
datasets: /opt/LIBERO/libero/datasets
assets: /opt/LIBERO/libero/libero/assets
YAML
    fi
    cd /opt/openvla/cache/worktrees/trackb-dense-grounding-v2
    python -u experiments/kv_cache/experiments/trackb_talk2dino_siglip/transfer_validation/libero_demo_ground_truth.py \
      --repo-root /opt/openvla \
      --libero-root /opt/LIBERO \
      --robosuite-root /opt/robosuite \
      --dataset-root /opt/openvla/cache/datasets/libero_ground_truth_v2_official \
      --cache-root /opt/openvla/cache/datasets/libero_ground_truth_v2_evalcache \
      --output-dir /opt/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/libero_ground_truth_e2_completion
  ' >> "$OUT/logs/container_worker.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/worker.exit_code"
if [[ -f "$OUT/RUN_STATUS.json" ]]; then
  python - "$OUT/RUN_STATUS.json" "$rc" <<'PY'
import json, os, sys, tempfile
p, rc = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(p))
except Exception:
    d = {}
d["host_worker_exit_code"] = rc
tmp = p + ".tmp"
with open(tmp, "w") as f:
    json.dump(d, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp, p)
PY
fi
exit "$rc"
