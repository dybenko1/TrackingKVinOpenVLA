# Original Talk2DINO Baseline Validation

This module evaluates the frozen upstream Talk2DINO ViT-Large model on COCO Object val2017 and the saved paired E2 LIBERO RGB/mask frames. It does not train, run policy actions, or change OpenVLA.

## Runtime

Use the already validated container `trackb-v1-coco-object-20261126`. The module expects the mounted repository at `/opt/openvla`, the isolated worktree at `/opt/openvla/cache/worktrees/trackb-dense-grounding-v2`, and the cached official Talk2DINO, DINOv2, CLIP/MLP weights and datasets listed in `run_coco.py`.

```bash
export PYTHONPATH=/opt/openvla/cache/venvs/v1-coco-eval-opencv:/opt/openvla/cache/worktrees/trackb-dense-grounding-v2
cd /opt/openvla/cache/worktrees/trackb-dense-grounding-v2
python -m experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.run_coco --smoke
python -m experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.run_coco
python -m experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.run_libero
python experiments/kv_cache/experiments/trackb_talk2dino_baseline_validation/make_libero_figures.py
python experiments/kv_cache/experiments/trackb_talk2dino_baseline_validation/make_report.py
```

The COCO evaluator resumes from verified per-image confusion matrices. LIBERO segment caches preserve source task directories in their output paths to avoid collisions. The validated run used the named detached tmux session `talk2dino-coco`; status, per-image progress and logs are under the output directory.

## Evaluation protocol and limits

COCO follows the upstream 448×448 square DINOText transform, the COCO test scale `(2048, 448)`, 448 sliding crop, 224 stride, `sub_imagenet_template`, 0.54 background score, and a global 81-class confusion matrix including background. The upstream DINOText model calls are used directly. Because a compatible MMCV/MMseg runtime was not present, the slide-window/test/evaluation wrapper is a single-process equivalent and must be treated as a protocol deviation until the >1 pp published-result discrepancies are explained.

LIBERO uses 1,152 previously inspected 224×224 E2 RGB frames and their paired simulator labels. Upstream DINOText resizes each RGB frame to 448. The fixed COCO threshold is diagnostic only; it is not calibrated for LIBERO. Metrics use fractional 16×16 patch occupancy and are not instance-level pixel IoU or policy success.

Do not cite the retained `libero/` or `libero_corrected/` provisional directories. Only `libero_final/` uses collision-free segment paths and the correct separate E2 `relevant_gt` union label.
