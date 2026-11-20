#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/worktrees/trackb-dense-grounding-v2
REPO=/home/oscar/projects/TrackingKVinOpenVLA/openvla
LIBERO=/home/oscar/projects/TrackingKVinOpenVLA/LIBERO
ROBOSUITE=/home/oscar/projects/TrackingKVinOpenVLA/dusty-robosuite
DATASET="$REPO/cache/datasets/talk2dino/coco2014"
FEATURES="$DATASET/features_training_v2_dense_20261118"
TRAINING="$WORKTREE/experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118"
V1="$REPO/cache/worktrees/trackb-talk2dino-siglip/experiments/kv_cache/outputs/trackb_talk2dino_siglip/full_coco_training_20261117/checkpoints/epoch_050.pt"
SIGLIP="$REPO/cache/models/timm/ViT-SO400M-14-SigLIP-384"
SAVED="$REPO/experiments/kv_cache/outputs/trackb_dinov2_segmentation_visual_trial"
OUT="$WORKTREE/experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/transfer_validation_20261120"
NAME=trackb-v2-transfer-validation-20261120
IMAGE=openvla-talk2dino-siglip:20261116
mkdir -p "$OUT/logs"
branch=$(git -C "$WORKTREE" branch --show-current)
[[ "$branch" == "thesis/trackb-dense-grounding-v2" ]] || { echo "Expected branch thesis/trackb-dense-grounding-v2, got $branch" >&2; exit 4; }

if docker inspect "$NAME" >/dev/null 2>&1; then
  state=$(docker inspect -f '{{.State.Status}}' "$NAME")
  echo "Container $NAME already exists (state=$state); refusing to duplicate it."
  exit 2
fi
for path in "$WORKTREE" "$LIBERO" "$ROBOSUITE" "$V1" "$SIGLIP/open_clip_model.safetensors" "$TRAINING/best_projection.pt" "$TRAINING/validation_manifest.json" "$FEATURES/patch_features_val2014_manifest.json"; do
  test -e "$path" || { echo "Required input missing: $path" >&2; exit 3; }
done

docker run -d \
  --name "$NAME" \
  --restart on-failure:3 \
  --runtime nvidia --network host --group-add 2011 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e MPLCONFIGDIR=/tmp/matplotlib -e TF_CPP_MIN_LOG_LEVEL=3 \
  -v "$REPO:/opt/openvla" -v "$LIBERO:/opt/LIBERO" -v "$ROBOSUITE:/opt/robosuite" -v /tmp:/tmp \
  -w /opt/openvla "$IMAGE" \
  bash -lc "export PYTHONPATH=/opt/openvla/cache/worktrees/trackb-dense-grounding-v2:/opt/LIBERO/libero:\$PYTHONPATH; cd /opt/openvla/cache/worktrees/trackb-dense-grounding-v2; exec python -u -m experiments.kv_cache.experiments.trackb_talk2dino_siglip.transfer_validation.orchestrate --repo /opt/openvla/cache/worktrees/trackb-dense-grounding-v2 --dataset-root /opt/openvla/cache/datasets/talk2dino/coco2014 --feature-root /opt/openvla/cache/datasets/talk2dino/coco2014/features_training_v2_dense_20261118 --training-output /opt/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118 --v1-checkpoint /opt/openvla/cache/worktrees/trackb-talk2dino-siglip/experiments/kv_cache/outputs/trackb_talk2dino_siglip/full_coco_training_20261117/checkpoints/epoch_050.pt --siglip-checkpoint /opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384 --libero-root /opt/LIBERO --robosuite-root /opt/robosuite --saved-frames /opt/openvla/experiments/kv_cache/outputs/trackb_dinov2_segmentation_visual_trial --output-dir /opt/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/transfer_validation_20261120"
