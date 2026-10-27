#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/oscar/projects/TrackingKVinOpenVLA/openvla
OUT="$ROOT/experiments/kv_cache/outputs/trackb_unmatch_cache_safety"
mkdir -p "$OUT/logs"
LOG="$OUT/logs/matrix_supervisor.log"
printf '\n[%s] Starting Track B UNMATCH cache-safety matrix in openvla-cached:latest\n' "$(date -Is)" | tee -a "$LOG"

docker run --rm --runtime nvidia \
  --network host \
  --group-add 2011 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e TF_CPP_MIN_LOG_LEVEL=3 \
  -e PYTHONPATH=/tmp/stage4a_openclip_pkg \
  -v /home/oscar/projects/TrackingKVinOpenVLA/openvla:/opt/openvla \
  -v /home/oscar/projects/TrackingKVinOpenVLA/LIBERO:/opt/LIBERO \
  -v /home/oscar/projects/TrackingKVinOpenVLA/dusty-robosuite:/opt/robosuite \
  -v /tmp:/tmp \
  -w /opt/openvla \
  openvla-cached:latest \
  python -u experiments/kv_cache/experiments/trackb_unmatch_cache_safety.py --mode matrix 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
printf '%s\n' "$RC" > "$OUT/logs/matrix_exit_code.txt"
printf '[%s] Matrix container exit code: %s\n' "$(date -Is)" "$RC" | tee -a "$LOG"
exit "$RC"
