#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="trackc-exp1"
REPO_HOST="/home/oscar/projects/TrackingKVinOpenVLA/openvla"
OUT_HOST="$REPO_HOST/experiments/kv_cache/outputs/trackc_kv_block_memory"
SCRIPT="/opt/openvla/experiments/kv_cache/experiments/trackc_kv_block_memory/run_queue.py"

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")
  echo "Container $CONTAINER_NAME already exists (state=$state); refusing to launch a duplicate."
  echo "Inspect with: docker logs $CONTAINER_NAME"
  exit 2
fi

mkdir -p "$OUT_HOST/logs"
container_id=$(docker run -d \
  --name "$CONTAINER_NAME" \
  --runtime nvidia \
  --network host \
  --group-add 2011 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e TF_CPP_MIN_LOG_LEVEL=3 \
  -v "$REPO_HOST:/opt/openvla" \
  -v /home/oscar/projects/TrackingKVinOpenVLA/LIBERO:/opt/LIBERO \
  -v /home/oscar/projects/TrackingKVinOpenVLA/dusty-robosuite:/opt/robosuite \
  -v /tmp:/tmp \
  -w /opt/openvla \
  openvla-cached:latest \
  python3 "$SCRIPT" --warmups 20 --repetitions 50)

printf '%s\n' "$container_id" > "$OUT_HOST/logs/process.pid"
printf 'Detached launch %s image=openvla-cached:latest at %s\n' "$container_id" "$(date -u +%FT%TZ)" >> "$OUT_HOST/logs/launcher.log"

echo "Launched detached container: $CONTAINER_NAME"
echo "Logs: docker logs -f $CONTAINER_NAME"
echo "Status: $OUT_HOST/logs/progress.json"
