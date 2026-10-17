# Track B — OpenVLA native-feature foreground/background segmentation

## Run status

Complete. This is an offline segmentation benchmark over replayed LIBERO observations. It does not implement selective KV skipping and does not measure closed-loop task success, KV-reuse latency, inference speedup, or action preservation.

## Data and split

- 315 observations across tasks 4, 5, and 7; sampled as seven five-frame R5 windows from each of nine Stage 3A Fresh trajectories.
- Complete trajectories are separated by initial state: training=105, calibration=105, evaluation=105 samples. Each split has three trajectories, one per task.
- Evaluation oracle labels were not used for classifier fitting, PCA polarity, or threshold selection. Classifier PCA dimensionality reduction is fitted on training features only; native DINOv2 PCA bases, polarity, and thresholds are fitted/calibrated on calibration trajectories.
- Stage 3A action records reconstruct observations and segmentation maps. No OpenVLA action generation was run during Track B extraction.

## 1. Semantic segmentation accuracy

The table reports evaluation micro averages across 256 visual tokens per observation. Recall, precision, IoU, false-negative rate, and false-positive background rate describe semantic foreground/background agreement. The source union and dilation rows also show how candidate masks change at token resolution.

| Method | Variant | FG recall | FG precision | FG IoU | Unsafe FG reuse | FP background rate | Reuse opportunity |
|---|---|---:|---:|---:|---:|---:|---:|
| DINOv2 PCA · per frame | current | 0.950 | 0.508 | 0.494 | 0.050 | 0.965 | 0.043 |
| DINOv2 PCA · per frame | dilated | 0.990 | 0.509 | 0.507 | 0.010 | 0.999 | 0.006 |
| DINOv2 PCA · per frame | source_union | 0.967 | 0.510 | 0.501 | 0.033 | 0.973 | 0.030 |
| DINOv2 PCA · fixed basis | current | 0.978 | 0.746 | 0.733 | 0.022 | 0.349 | 0.329 |
| DINOv2 PCA · fixed basis | dilated | 1.000 | 0.574 | 0.574 | 0.000 | 0.776 | 0.109 |
| DINOv2 PCA · fixed basis | source_union | 0.979 | 0.742 | 0.730 | 0.021 | 0.357 | 0.325 |
| DINOv2 LDA | current | 0.963 | 0.939 | 0.907 | 0.037 | 0.066 | 0.475 |
| DINOv2 LDA | dilated | 1.000 | 0.660 | 0.660 | 0.000 | 0.540 | 0.225 |
| DINOv2 LDA | source_union | 0.967 | 0.933 | 0.904 | 0.033 | 0.073 | 0.470 |
| SigLIP LDA | current | 0.960 | 0.981 | 0.942 | 0.040 | 0.019 | 0.500 |
| SigLIP LDA | dilated | 1.000 | 0.674 | 0.674 | 0.000 | 0.506 | 0.242 |
| SigLIP LDA | source_union | 0.965 | 0.977 | 0.943 | 0.035 | 0.024 | 0.495 |
| Fused LDA | current | 0.966 | 0.970 | 0.938 | 0.034 | 0.031 | 0.491 |
| Fused LDA | dilated | 1.000 | 0.678 | 0.678 | 0.000 | 0.496 | 0.246 |
| Fused LDA | source_union | 0.970 | 0.966 | 0.938 | 0.030 | 0.035 | 0.487 |

Thresholds were selected using calibration labels by maximizing reuse opportunity under a 5% calibration unsafe-foreground-reuse target. If no threshold met the target, the minimum-unsafe threshold was selected and that deviation is recorded in `summary.json`. The full held-out threshold sweep is in `threshold_sweep.csv`; evaluation curves are descriptive and were not used to choose thresholds.

## 2. Potential background-token KV-reuse suitability

Unsafe foreground reuse is oracle foreground predicted background divided by total oracle foreground tokens. Reuse opportunity is the predicted-background fraction of all 256 visual tokens. The dilation rows measure a conservative one-patch foreground expansion; the source-union rows protect locations predicted foreground in either the R5 source or current frame. These are mask-level opportunity and risk proxies, not measured compute savings.

## Temporal reuse safety

`temporal_consistency.csv` reports adjacent-frame token-mask agreement within the sampled five-frame R5 windows. `per_sample_metrics.csv` reports both the current-frame mask and the source/current foreground union. The union protects a token marked foreground in either frame; its changes in unsafe reuse and reuse opportunity can be read from `metrics.csv`.

## Computational overhead

- Median OpenVLA DINOv2+SigLIP feature forward: 93.74970011413097 ms per observation; p95 109.33177368715405 ms.
- `summary.json` reports each segmentation method's median and p95 classifier overhead separately, measured on evaluation samples and excluding the vision-backbone forward.
- These are component measurements, not end-to-end latency or speedup claims.

## 3. Actual KV-reuse performance

Not measured in Track B. There was no selective KV-cache skipping, decoder execution, action generation, latency benchmark, or task-success evaluation. Segmentation quality alone does not establish speedup or downstream action preservation.

## Visual results

See `figures/segmentation_overlays.png`, `figures/error_maps.png`, `figures/dilation_comparison.png`, the three `figures/temporal_*.png` sequences, `figures/reuse_opportunity_vs_unsafe_fg_reuse.png`, and the combined `figures/comparison_report.pdf`. The gallery includes task/state medians plus the strongest and weakest held-out samples by average method IoU; examples are labeled with their split and should not be read as a substitute for the aggregate metrics.
- Figures were rendered with Matplotlib.

## Method notes and limitations

1. DINOv2 PCA per frame is unsupervised in its per-frame component fit, but label-based calibration supplies a global polarity and threshold. Per-frame components are sign-aligned to the calibration PCA direction.
2. Calibration-basis DINOv2 PCA fits its basis to calibration observations and holds it fixed for all other frames. Its foreground polarity and threshold use calibration labels.
3. LDA methods train only on the training trajectories. To condition a high-dimensional covariance, each branch uses per-token L2 normalization and a training-only randomized PCA to 128 dimensions, followed by binary pooled-covariance LDA with 10% shrinkage toward a diagonal target and a 1e-6 relative ridge. This is a documented regularized linear discriminant implementation; the same reduction and regularization are used for all three comparisons, and explained variance/ridge values are recorded in `summary.json`.
4. Oracle semantics use the Stage 3A rendered-label utility: explicit background name hints are background; robot hints are foreground; all other names are foreground by default and are listed as `foreground_default_unmapped`. Nonzero segmentation IDs not represented by the name map are also recorded in `dataset_manifest.json`.
5. Oracle pixels are mapped to the 16×16 token grid by the existing Stage 3A nearest-neighbor resize/crop and any-foreground-pixel reduction. The feature mapping report and overlay document the spatial assumptions.
6. The benchmark samples 315 observations from nine pre-existing trajectories; the held-out evaluation split contains three complete state-2 trajectories, one per task. Generalization to other tasks, camera views, object taxonomies, or policies is not established.
7. Segmentation results indicate mask quality and theoretical background-token opportunity only. Actual KV-reuse performance, projection savings, latency, downstream action relevance, and task-success preservation were not measured.

## Reproducibility artifacts

- `configuration.json`: experiment parameters and code revision.
- `dataset_manifest.json`: action-trace source hashes, trajectory split manifests, oracle mapping audit, sample paths/checksums, encoder mapping, and feature extraction status.
- `data/`: local-only RGB, instance masks, semantic masks, token masks, native feature tensors, and evaluation scores. Large arrays are not committed.
- `logs/`: supervisor and execution logs; raw feature checksums are in `dataset_manifest.json`.
