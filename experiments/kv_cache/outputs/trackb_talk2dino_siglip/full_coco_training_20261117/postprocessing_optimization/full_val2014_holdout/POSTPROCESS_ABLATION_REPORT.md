# Full COCO val2014 retrieval and segmentation validation

## Design and split integrity

- Retrieval covered **40,504 images and 202,654 captions**. Development: 8,101 images / 40,542 captions; final evaluation: 32,403 images / 162,112 captions.
- Segmentation used 500 development and 2000 final images with COCO instance masks. The manifests are disjoint and exclude all previously examined Phase 3 manifest images.
- Checkpoints were frozen epochs 20, 50, 100; the DINO weights are OpenVLA’s existing backbone, and features use the same 224px Talk2DINO sidecar transform used for full-COCO training and Phase 3 (short-edge bicubic resize, center crop, ImageNet normalization). The standard OpenVLA policy preprocessing/forward path was not changed. No training, CLIP baseline, or 448px input was used.
- Threshold calibration used development images only. Final evaluation metrics were not used to select the checkpoint or thresholds.

## Full val2014 multi-positive retrieval

| Epoch | Split | I→T R@1 | I→T R@5 | I→T R@10 | T→I R@1 | T→I R@5 | T→I R@10 | Symmetric multi-positive InfoNCE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | development | 0.1223 | 0.4855 | 0.6384 | 0.0921 | 0.3800 | 0.5139 | 4.5363 |
| 100 | evaluation | 0.0558 | 0.2658 | 0.3839 | 0.0448 | 0.2047 | 0.2954 | 5.8976 |
| 20 | development | 0.1490 | 0.5359 | 0.6812 | 0.0988 | 0.3989 | 0.5324 | 4.4504 |
| 20 | evaluation | 0.0732 | 0.3110 | 0.4324 | 0.0498 | 0.2220 | 0.3159 | 5.8082 |
| 50 | development | 0.1305 | 0.5056 | 0.6660 | 0.0981 | 0.3946 | 0.5286 | 4.4880 |
| 50 | evaluation | 0.0632 | 0.2878 | 0.4076 | 0.0485 | 0.2174 | 0.3103 | 5.8461 |

Image-to-text retrieval counts any caption for that image as positive. Text-to-image uses its paired image. Similarities are streamed in chunks; no full similarity matrix is materialized. Ranking ties are treated as not outranking a positive.

## Segmentation postprocessing ablation

Development-selected primary native-grid condition: **epoch_050 + pamr_raw**, threshold 0.55, development macro class mIoU 0.3094.

| Checkpoint | Method | Scale | Dev-calibrated macro class mIoU | AUROC | Precision | Recall | Pointing |
|---|---|---|---:|---:|---:|---:|---:|---:|
| epoch_020 | cleaned | aligned224 | 0.2995 | 0.8555 | 0.3993 | 0.4838 | 0.5673 |
| epoch_020 | cleaned | grid16 | 0.2958 | 0.8362 | 0.3735 | 0.4906 | 0.5551 |
| epoch_020 | cleaned_pamr | aligned224 | 0.3158 | 0.8604 | 0.4087 | 0.4492 | 0.6287 |
| epoch_020 | cleaned_pamr | grid16 | 0.3171 | 0.8617 | 0.3806 | 0.4491 | 0.6463 |
| epoch_020 | pamr_raw | aligned224 | 0.3223 | 0.8598 | 0.4246 | 0.4627 | 0.6271 |
| epoch_020 | pamr_raw | grid16 | 0.3224 | 0.8612 | 0.3943 | 0.4617 | 0.6441 |
| epoch_020 | raw | aligned224 | 0.3057 | 0.8550 | 0.4075 | 0.4986 | 0.5602 |
| epoch_020 | raw | grid16 | 0.2985 | 0.8357 | 0.3811 | 0.5040 | 0.5501 |
| epoch_050 | cleaned | aligned224 | 0.3018 | 0.8605 | 0.4147 | 0.4787 | 0.5782 |
| epoch_050 | cleaned | grid16 | 0.2972 | 0.8417 | 0.3846 | 0.4846 | 0.5669 |
| epoch_050 | cleaned_pamr | aligned224 | 0.3203 | 0.8658 | 0.4231 | 0.4410 | 0.6482 |
| epoch_050 | cleaned_pamr | grid16 | 0.3230 | 0.8670 | 0.3962 | 0.4408 | 0.6524 |
| epoch_050 | pamr_raw | aligned224 | 0.3275 | 0.8653 | 0.4409 | 0.4598 | 0.6459 |
| epoch_050 | pamr_raw | grid16 | 0.3290 | 0.8666 | 0.4111 | 0.4591 | 0.6514 |
| epoch_050 | raw | aligned224 | 0.3073 | 0.8601 | 0.4206 | 0.4993 | 0.5736 |
| epoch_050 | raw | grid16 | 0.3003 | 0.8412 | 0.3922 | 0.5059 | 0.5636 |
| epoch_100 | cleaned | aligned224 | 0.2934 | 0.8421 | 0.3946 | 0.4350 | 0.5719 |
| epoch_100 | cleaned | grid16 | 0.2910 | 0.8272 | 0.3695 | 0.4434 | 0.5578 |
| epoch_100 | cleaned_pamr | aligned224 | 0.3083 | 0.8457 | 0.3933 | 0.4006 | 0.6234 |
| epoch_100 | cleaned_pamr | grid16 | 0.3095 | 0.8480 | 0.3648 | 0.3999 | 0.6415 |
| epoch_100 | pamr_raw | aligned224 | 0.3161 | 0.8451 | 0.4119 | 0.4216 | 0.6222 |
| epoch_100 | pamr_raw | grid16 | 0.3193 | 0.8475 | 0.3860 | 0.4215 | 0.6400 |
| epoch_100 | raw | aligned224 | 0.3003 | 0.8416 | 0.4033 | 0.4578 | 0.5661 |
| epoch_100 | raw | grid16 | 0.2962 | 0.8267 | 0.3792 | 0.4649 | 0.5527 |

### Small-object and postprocessing diagnostics (epoch_050, native grid)

Size groups are based on the number of occupied 16×16 cells in a per-image category-union mask: small <4, medium 4–15, large ≥16. This is a grid-resolution proxy, not physical object-size annotation.

| Method | Small mIoU (pairs) | Medium mIoU (pairs) | Large mIoU (pairs) |
|---|---:|---:|---:|
| raw | 0.0667 (1148) | 0.2066 (1341) | 0.4016 (2445) |
| cleaned | 0.0605 (1148) | 0.2025 (1341) | 0.3873 (2445) |
| pamr_raw | 0.0644 (1148) | 0.2211 (1341) | 0.4201 (2445) |
| cleaned_pamr | 0.0620 (1148) | 0.2134 (1341) | 0.4046 (2445) |

Held-out macro category mIoU change relative to raw at this same checkpoint/mask scale: cleaned -0.0030, pamr_raw +0.0288, cleaned_pamr +0.0227. These paired aggregate differences indicate whether cleaning/refinement helped; they do not identify causal failure mechanisms by themselves.

For historical context only, epoch-100 cleaned at the fixed 0.54 threshold scored 0.2910 on this new final holdout versus the prior 0.3059 on the previously examined 80-image subset (descriptive difference -0.0149). The subsets differ, so this is not a paired estimate of improvement.


Per-category metrics are in `segmentation/per_category_metrics.csv`; per-image metrics are retained locally in `segmentation/per_image_metrics.jsonl` and in compressed form as `segmentation/per_image_metrics.jsonl.gz`. The gallery contains example RGB/GT/raw/cleaned/PAMR maps for the final evaluation subset.

## Limits

- This is semantic category-union evaluation against COCO instance masks projected to the 16×16 patch grid and aligned 224×224 view; it is not instance segmentation AP.
- The evaluation subset was selected from annotation availability and geometric visibility, without model predictions. It excludes the older 100-image Phase 3 manifest but remains one COCO dataset/domain.
- The 0.3059 prior mIoU is a historical result on the previously examined 80-image subset and is not a matched test-set baseline for this holdout.
- PAMR and attention-based cleaning are postprocessing; there is no policy/KV reuse or LIBERO action rollout.

## LIBERO feasibility

- Offline mask review covers 27 saved RGB frames across tasks 4, 5, 7, states 0, 1, 2.
- Selected configuration: epoch_050 + `pamr_raw`, development-calibrated threshold 0.55.
- Sparse-frame temporal mask IoU by prompt: `{"background scene": 0.24485668040941114, "black bowl": 0.6040870849225448, "robot arm": 0.4810399841147012, "robot gripper": 0.41860056966435527, "table": 1.0}`; source/current patch flips: `{"background scene": 39.05555555555556, "black bowl": 14.88888888888889, "robot arm": 40.05555555555556, "robot gripper": 35.77777777777778, "table": 0.0}`.
- These RGB-only artifacts have no paired oracle segmentation mask; the sparse frames are not adjacent control steps. This is qualitative feasibility evidence, not accuracy or cache-policy validation.
- See `libero_feasibility/LIBERO_FEASIBILITY_REPORT.md` and its per-frame visualizations.

## Artifacts

- `validation_split_manifest.json`, `segmentation_manifest.json` — frozen image-level splits and selection criteria.
- `full_val2014_retrieval_summary.json` — full retrieval metrics.
- `segmentation/summary.json`, `segmentation/threshold_calibration.json`, `segmentation/per_category_metrics.csv`, `segmentation/per_image_metrics.jsonl` — segmentation metrics.
- `configuration.json` and `run_status.json` — exact data paths, hashes, software/checkpoint identities, and completion status.
