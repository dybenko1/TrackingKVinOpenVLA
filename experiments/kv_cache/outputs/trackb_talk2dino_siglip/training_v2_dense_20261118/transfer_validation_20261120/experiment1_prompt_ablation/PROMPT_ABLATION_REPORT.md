# Prompt Robustness and Language Ablation

Frozen V2 epoch 40 was evaluated with cached OpenVLA penultimate DINO patches. No weights were updated. V1 used its seven-template prompt ensemble.

The binary scores are `sigmoid(10 × (cosine − 0.20))`. Conditions A–D reword known COCO categories; E averages embeddings and F averages probability maps. Thresholds were selected only on the 1,000-image development split by pooled category IoU. The 3,000-image split was previously inspected and is secondary, not a new blind test.

All aggregate threshold-dependent metrics were rebuilt from saved per-image threshold sweeps; model inference was not rerun. Small-object IoU and empty-mask frequency now use each method's reported threshold.

Development selected **A_training_prompt** at threshold **0.20**. V1 remains at its published 0.55 threshold; its development recalibration is reported separately at **0.55**.

## Development metrics (calibrated threshold)

| Condition | Threshold | Pooled macro IoU | Positive-pair macro IoU | Precision | Recall | Small-object IoU | Pointing | Empty masks | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_training_prompt | 0.20 | 0.4928 | 0.3422 | 0.6210 | 0.7237 | 0.1441 | 0.7598 | 0.1730 | 0.8673 |
| B_natural_description | 0.35 | 0.3816 | 0.2639 | 0.5074 | 0.5876 | 0.0962 | 0.6732 | 0.2333 | 0.8096 |
| C_picture_phrase | 0.30 | 0.4654 | 0.3152 | 0.6060 | 0.6854 | 0.1310 | 0.7277 | 0.2240 | 0.8432 |
| D_noun_phrase | 0.30 | 0.4156 | 0.2887 | 0.5385 | 0.6465 | 0.1070 | 0.7159 | 0.2155 | 0.8490 |
| E_embedding_ensemble | 0.30 | 0.4652 | 0.3186 | 0.6087 | 0.6803 | 0.1352 | 0.7497 | 0.2166 | 0.8585 |
| F_score_ensemble | 0.30 | 0.4557 | 0.3122 | 0.6050 | 0.6598 | 0.1297 | 0.7442 | 0.2309 | 0.8527 |
| V1_seven_template | 0.55 | 0.3987 | 0.2673 | 0.4807 | 0.7469 | 0.0738 | 0.6589 | 0.2865 | 0.7910 |

## Secondary previously inspected 3,000-image holdout

| Condition | Threshold | Pooled macro IoU | Positive-pair macro IoU | Recall | Small-object IoU | Pointing |
|---|---:|---:|---:|---:|---:|---:|
| A_training_prompt | 0.20 | 0.4926 | 0.3438 | 0.7243 | 0.1325 | 0.7743 |
| B_natural_description | 0.35 | 0.3862 | 0.2697 | 0.5942 | 0.0830 | 0.6881 |
| C_picture_phrase | 0.30 | 0.4705 | 0.3192 | 0.6902 | 0.1232 | 0.7447 |
| D_noun_phrase | 0.30 | 0.4132 | 0.2911 | 0.6439 | 0.0984 | 0.7255 |
| E_embedding_ensemble | 0.30 | 0.4679 | 0.3222 | 0.6819 | 0.1281 | 0.7641 |
| F_score_ensemble | 0.30 | 0.4596 | 0.3164 | 0.6644 | 0.1213 | 0.7573 |
| V1_seven_template | 0.55 | 0.3972 | 0.2681 | 0.7336 | 0.0681 | 0.6707 |

## Interpretation and limits

The prompt experiment measures paraphrase robustness for COCO classes present in V2 training. It does not establish zero-shot performance for unseen attributes such as `black bowl` or concepts such as `plate`, `drawer`, or `robot gripper`; those require independent masks.

`mean_positive_pair_macro_iou` averages image/category IoU over positive pairs and then categories. `pooled_macro_iou` pools TP/FP/FN within each category before macro-averaging. Pointing accuracy and AUROC do not depend on the binary threshold. Small-object IoU includes positive image/category examples with occupied area below four 16×16 patch equivalents.

See `development_metrics.json`, `development_thresholds.json`, and `evaluation_metrics_secondary_previously_inspected.json` for category-level results. Earlier aggregates are preserved with `_pre_metric_correction` suffixes.
