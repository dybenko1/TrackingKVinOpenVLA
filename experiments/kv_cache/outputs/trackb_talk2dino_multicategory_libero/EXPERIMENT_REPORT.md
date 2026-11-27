# Original Talk2DINO multi-category background detection on LIBERO

## Result

This run tests original standalone Talk2DINO on 90 cached E2 RGB/oracle pairs with independently encoded object queries processed simultaneously. It does not train, run policy rollouts, use OpenVLA DINO features, or alter the KV cache.

The primary multi-category decision is the original background-rejection rule: take the maximum cleaned (or raw) category score at each 16×16 patch; below threshold is background. All metrics compare against the E2 task-relevant fractional patch-occupancy union. Display masks binarize oracle occupancy at >0 only for visualization.

## Split and inference

- Images: 90 total; 30 development and 60 evaluation.
- Three task demonstrations per suite; ten evenly spaced frames per demonstration. Split is by entire demonstration segment (one segment per suite for development; two per suite for evaluation).
- Original Meta DINOv2 ViT-L/14-register, OpenAI CLIP ViT-B/16, original projection MLP, original `sub_imagenet_template`, original per-head attention cleaning with λ=0.2. Inference input is 224×224, yielding a 16×16 patch grid. PAMR disabled.
- Each task's projected query embeddings are encoded once, then all listed object/robot phrases are scored in the same upstream DINOText pass.
- Threshold sweep uses stored score arrays; no additional vision inference is run for the sweep.

## Main evaluation-set comparison

The 0.54 rows are the original COCO reference threshold. The 95%-recall thresholds were selected on development demonstrations only; evaluation columns use those frozen thresholds.

| Variant / threshold policy | Threshold | FG recall | FG precision | FG IoU | BG precision | BG recall | BG IoU | BG coverage | Foreground missed as BG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW at 0.54 | 0.540 | 0.998 | 0.280 | 0.280 | 0.999 | 0.535 | 0.534 | 45.3% | 0.2% |
| CLEAN at 0.54 | 0.540 | 0.978 | 0.312 | 0.310 | 0.994 | 0.610 | 0.607 | 51.9% | 2.2% |
| RAW dev-selected ≥95% recall | 0.550 | 0.975 | 0.317 | 0.314 | 0.993 | 0.619 | 0.616 | 52.8% | 2.5% |
| CLEAN dev-selected ≥95% recall | 0.540 | 0.978 | 0.312 | 0.310 | 0.994 | 0.610 | 0.607 | 51.9% | 2.2% |

Metrics are pooled fractional patch occupancy, not per-image binary IoU. BG is the complement of the task-relevant oracle and predicted FG on valid patches.

## Development-only operating points

For each variant, the displayed operating threshold maximizes development BG IoU subject to pooled FG recall ≥95% (and separately ≥98%). It is then frozen for the evaluation split.

| Variant | Target dev FG recall | Selected threshold | Dev FG recall | Dev BG IoU | Dev BG coverage | Eval FG recall | Eval BG IoU | Eval BG coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW | 95% | 0.550 | 0.987 | 0.473 | 39.7% | 0.975 | 0.616 | 52.8% |
| RAW | 98% | 0.550 | 0.987 | 0.473 | 39.7% | 0.975 | 0.616 | 52.8% |
| CLEAN | 95% | 0.540 | 0.991 | 0.459 | 38.5% | 0.978 | 0.607 | 51.9% |
| CLEAN | 98% | 0.540 | 0.991 | 0.459 | 38.5% | 0.978 | 0.607 | 51.9% |

## Foreground recall at protected-token budgets

These ranking diagnostics protect the top-scoring patches per image. They are not the thresholded background detector and do not prove KV safety.

| Split | Variant | 10% | 25% | 50% | 75% | 90% |
|---|---|---:|---:|---:|---:|---:|
| development | RAW | 0.383 | 0.732 | 0.938 | 0.992 | 1.000 |
| development | CLEAN | 0.383 | 0.765 | 0.941 | 0.988 | 1.000 |
| evaluation | RAW | 0.371 | 0.759 | 0.973 | 1.000 | 1.000 |
| evaluation | CLEAN | 0.413 | 0.778 | 0.975 | 1.000 | 1.000 |

## Object-role recall

Role recall evaluates each independent category map against the corresponding coarser E2 role channel. For categories such as ramekin/table, the simulator label is a role union rather than a category-specific instance mask; interpret those rows as role-level diagnostics.

Evaluation split at the official 0.54 threshold:

| Role | RAW recall | CLEAN recall |
|---|---:|---:|
| destination_object | 0.804 | 0.622 |
| destination_or_fixture | 0.855 | 0.422 |
| manipulated_object | 0.540 | 0.432 |
| robot_arm | 1.000 | 0.999 |
| robot_gripper | 1.000 | 0.998 |

These per-query recalls use each query's own score map and a coarser role label. They do not establish category-specific instance accuracy.

## Visual examples

### libero_goal — put_the_wine_bottle_on_the_rack

Queries: wine bottle, wine rack, robot arm, robot gripper

![Binary multi-category FG/BG comparison](libero_goal_multicategory_binary.png)

### libero_object — pick_up_the_tomato_sauce_and_place_it_in_the_basket

Queries: tomato sauce, basket, robot arm, robot gripper

![Binary multi-category FG/BG comparison](libero_object_multicategory_binary.png)

### libero_spatial — pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate

Queries: black bowl, plate, table, robot arm, robot gripper

![Binary multi-category FG/BG comparison](libero_spatial_multicategory_binary.png)

## Cost

One inference forward per image returned RAW and CLEAN maps. Recorded mean synchronized scoring time: **83.02 ms** over 90 images. Per-image median/p95/min/max are unavailable in the preserved timing artifact and are not reported.
Projected category-text embeddings were computed once per task segment: mean **109.08 ms** per segment across 9 segments (reported total 981.75 ms).
Peak CUDA allocation: 1.472066879272461 GB; peak CUDA reserved: 1.556640625 GB.

## Limitations and conclusions

This covers three demonstration segments per suite (one development segment and two evaluation segments), not broad generalization. E2 labels are fractional 16×16 patch occupancies, so these measurements cannot establish pixel-level object boundary quality. The 0.54 threshold is the COCO reference, not a LIBERO-calibrated setting.

A high-recall background operating point is only a segmentation operating point. It does not establish that stale K/V can be reused safely or improve robot task success.

### Research questions

1. **Can original Talk2DINO identify task-relative background with simultaneous queries?** At development-selected thresholds, evaluation FG recall is 97.5% RAW and 97.8% CLEAN, while BG coverage is 52.8% and 51.9%, respectively. This is a useful coarse separation signal, with moderate BG IoU (0.616 RAW, 0.607 CLEAN).
2. **Does cleaning improve at matched foreground recall?** At fixed 0.54, CLEAN raises BG IoU from 0.534 to 0.607, while FG recall drops from 99.8% to 97.8%. At development-selected high-recall thresholds, RAW is slightly better on evaluation BG IoU and coverage (0.616/52.8%) than CLEAN (0.607/51.9%); cleaning does not improve the matched-recall result in this sample.
3. **How much background at high foreground recall?** About 52.8% RAW and 51.9% CLEAN, at held-out FG recall near 98%. Both evaluation recalls fall slightly short of 98%, although the development constraint was met.
4. **Is further OpenVLA integration justified?** Yes for a bounded diagnostic integration; the current evidence does not justify selective KV reuse. The sample is small and uses coarse patch labels, and it includes no cache or task-success evaluation.

## Reproducibility

Source: `experiments/kv_cache/experiments/trackb_talk2dino_multicategory_libero/run_experiment.py`. See `configuration.json`, `frame_manifest.json`, `validation.json`, `threshold_sweep.csv` and `latency_metrics.json` for exact identities, hashes, selected demos, queries and metrics.
