# Track B V2 Metric and Prompt Audit

## Status and scope

The 40-epoch V2 training is preserved; this audit performs no training, policy rollouts, or KV-cache changes. All corrected metrics below are based on the existing frozen checkpoints and saved COCO val2014 features/masks.

## Corrected development checkpoint metric

The saved epoch-40 checkpoint has native-grid pooled macro-category mIoU **0.2794** at threshold **0.45**. Corrected small-object IoU is **0.1262** (effective valid occupancy >0 and <4 patch equivalents; absent categories excluded). The old reported value was 0.0048; it included absent categories and is invalid. The macro mIoU and threshold are unchanged.

The saved best epoch was selected under the original metric. Only `best_dev.pt`, `latest.pt`, and `initial_mlp.pt` are available; earlier epoch weights are not. Thus this is a corrected evaluation of the saved checkpoint, not corrected epoch selection.

## Comparison results

All IoU values in the table are percentages. `Pooled macro IoU` sums intersections/unions across all positive image/category pairs per category, then averages categories; absent-category images are excluded from both measures. `Mean pair IoU` averages IoU over positive image/category pairs within each category, then macro-averages categories; this matches the prior V1 Phase 3 aggregation.

| Result | Split / checkpoint | Prompt configuration | 16x16 pooled macro IoU | 16x16 mean positive-pair macro IoU | 224x224 pooled macro IoU | Threshold |
|---|---|---|---:|---:|---:|---|
| V1 published baseline | prior 80-image evaluation; V1 epoch 100 | seven templates; no PAMR | — | 30.59% (reported) | — | 0.54 |
| V1 prior larger holdout reference | prior 2,000-image holdout; epoch 50 + raw PAMR | seven templates | — | 32.90% | — | prior dev calibration |
| V1 recomputed | new 3,000-image holdout; V1 epoch 50 + raw PAMR | faithful seven-template mean | 39.72% | 26.81% | 42.44% | fixed 0.55 |
| V2 raw | same new 3,000-image holdout; best V2 epoch 40 | single training prompt | 49.26% | 34.38% | 47.70% | development-calibrated |
| V2 + PAMR | same new 3,000-image holdout; best V2 epoch 40 | single training prompt | 45.04% | 27.53% | 50.61% | development-calibrated |

## Prompt and representation differences

V1 uses the seven template strings from the existing Phase 3 `encode_class_templates()` path: `itap of a {}`, `a bad photo of a {}`, `a origami {}`, `a photo of the large {}`, `a {} in a video game`, `art of the {}`, and `a photo of the small {}`. Their raw text embeddings are averaged before V1 projection and normalization. V2 uses the single `a photo of a <category>` prompt used during training. V1 uses final-layer normalized DINO patches; V2 uses the penultimate DINO patches from OpenVLA's standard path. Therefore the comparison is a system-level comparison, not an isolated loss ablation.

## Ground truth and aggregation audit

COCO val2014 non-crowd instance masks are unioned by semantic category. Crowd pixels are ignored. Grid targets are fractional 16x16 occupancy with fractional valid-cell weights; 224x224 targets are binary and use valid non-crowd pixels. Categories with no valid positive ground-truth occupancy are excluded. Pointing accuracy selects the highest-scoring valid location. Both pooled per-category IoU and mean positive image/category IoU are emitted.

## Previous evaluator defect and held-out disclosure

The earlier evaluator repeatedly opened each per-image JSONL in `w` mode. Although it processed the configured 1,000/3,000 images, only one record survived in each file; the old `evaluation_complete.json` therefore records 1/1 and its metrics are invalid. A later audit attempt also applied the V2 default 0.55 threshold instead of the development-selected 0.20 and is preserved as invalid in `../evaluation_audit_20261120/`. The corrected run writes to this separate directory, uses the selected threshold, and refuses completion unless exact split counts are present.

The final split was already processed by the earlier run and its one-record aggregate was inspected. Corrected thresholds are selected only on the fixed development split, never the final split, but the final evaluation is not an untouched blind test. See `development_thresholds.json`, `evaluation_metadata.json`, and `summary.json` for provenance.

## Files

- `summary.json`: corrected metrics, provenance, and disclosure.
- `development_metrics.json` and `evaluation_metrics.json`: summary and per-category metrics.
- `per_category_metrics.csv`: corrected held-out per-category metrics.
- `development_thresholds.json`: development-only threshold calibration.
- `development_per_image.jsonl` and `evaluation_per_image.jsonl`: complete image records.
- `evaluation_metadata.json`: prompt, feature, and ground-truth definitions.
- `corrected_training_dev_metrics.json`: corrected small-object metric and checkpoint-selection caveat.
- `artifact_manifest.json`: hashes and sizes for local-only per-image rows, log, and figures.

## Limitations

The V1 and V2 feature layers and prompt aggregation differ. V1 historical numbers use another subset and metric/reporting context. The selected V2 checkpoint cannot be retroactively reselected without earlier epoch weights. The final holdout has been previously inspected. These data do not establish LIBERO task success, safe KV reuse, or latency gains.
