# E2 Offline Threshold Comparison

This analysis reuses the reconciled E2 probability scores, simulator occupancy masks, prompt mappings, validity masks, and the original `collect_metrics()` implementation. No model inference, feature extraction, simulator replay, or policy action was run.

Manifest SHA-256: `d9cf911a1cb81b8bd635b8991d49e1977c2a50e164fc38cb38ebb1d5a371d156`. Cached sequences/frames: development 18/576; held-out 18/576.

E2 held-out scores were previously inspected while auditing role mapping. The held-out split is therefore a regression diagnostic, not an untouched test set. The predetermined primary threshold for the live experiment remains 0.15; this diagnostic comparison did not change it.

## Task-relevant union

| Split | Probability cutoff | Recall | Precision | False-background rate | Candidate BG coverage | Mean reusable tokens |
|---|---:|---:|---:|---:|---:|---:|
| development | 0.10 | 0.9588 | 0.2377 | 0.0412 | 0.2938 | 75.2/256 |
| development | 0.15 | 0.8235 | 0.3136 | 0.1765 | 0.5401 | 138.3/256 |
| development | 0.20 | 0.5272 | 0.3336 | 0.4728 | 0.7233 | 185.2/256 |
| held_out | 0.10 | 0.9202 | 0.2174 | 0.0798 | 0.2851 | 73.0/256 |
| held_out | 0.15 | 0.6362 | 0.3489 | 0.3638 | 0.6921 | 177.2/256 |
| held_out | 0.20 | 0.3958 | 0.5419 | 0.6042 | 0.8766 | 224.4/256 |

## Role recall at the primary cutoff 0.15

| Split | Manipulated | Destination object | Destination/fixture prompt | Task fixture | Robot arm | Robot gripper |
|---|---:|---:|---:|---:|---:|---:|
| development | 0.3605 | 0.0081 | 0.5146 | 0.8785 | 0.5923 | 0.9718 |
| held_out | 0.2637 | 0.0000 | 0.1407 | 0.2161 | 0.5866 | 0.9540 |

## Across-task variability at 0.15

Task-level recall values are in `offline_threshold_per_task.csv`. The task-relevant union includes manipulated objects, destination objects, task-relevant fixtures, robot arm, gripper, and robot mount per the E2 role-reconciled evaluator. The physical foreground labels and prompt roles are diagnostics; candidate background is not proven KV-safe.
