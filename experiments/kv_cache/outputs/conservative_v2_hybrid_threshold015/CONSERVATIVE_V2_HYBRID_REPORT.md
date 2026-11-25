# Conservative V2 Hybrid KV Reuse — Probability Cutoff 0.15

Status: **complete**. Success: **5/9**.

## Frozen method

This is the fixed Stage 3C-derived R5 hybrid (not the adaptive rolling policy). It changes exactly the external V2 foreground-probability classification cutoff from 0.20 to 0.15. The trained scoring function remains `sigmoid(10 × (cosine_similarity − 0.20))`; its cosine offset is unchanged. Frozen V2 epoch 40, current-frame native penultimate OpenVLA DINO features, existing E2 instruction prompts and SigLIP text embeddings, top-10% attention protection of current predicted background, layer-15 attention captured at full-refresh anchors, immutable independent R5 anchor, full Q and actual K/V skipping over 32 layers are retained. No confidence gate, rolling cache, continuous attention, or explicit FG-to-BG transition protection is used.

## Offline E2 threshold diagnostic

The pre-existing E2 cached score/mask/occupancy arrays were reused; no weights or features were regenerated. E2 held-out scores were previously inspected during the role-mapping audit, so these are regression diagnostics rather than a blind test. The predetermined live configuration remains 0.15; thresholds were not selected from these outputs.

| Split | Cutoff | Task-relevant recall | Precision | False-background rate | Candidate BG coverage | Mean reusable tokens |
|---|---:|---:|---:|---:|---:|---:|
| development | 0.10 | 0.9588 | 0.2377 | 0.0412 | 0.2938 | 75.2/256 |
| development | 0.15 | 0.8235 | 0.3136 | 0.1765 | 0.5401 | 138.3/256 |
| development | 0.20 | 0.5272 | 0.3336 | 0.4728 | 0.7233 | 185.2/256 |
| held_out | 0.10 | 0.9202 | 0.2174 | 0.0798 | 0.2851 | 73.0/256 |
| held_out | 0.15 | 0.6362 | 0.3489 | 0.3638 | 0.6921 | 177.2/256 |
| held_out | 0.20 | 0.3958 | 0.5419 | 0.6042 | 0.8766 | 224.4/256 |

Offline reports: `offline_preflight/OFFLINE_THRESHOLD_REPORT.md`, `offline_preflight/offline_threshold_metrics.json`, `offline_preflight/offline_threshold_per_task.csv`; representative saved-score heatmaps are in `offline_preflight/visualizations/`.

## Live LIBERO regression matrix

| Task | State | Success | Steps |
|---:|---:|:---:|---:|
| 4 | 0 | no | 220 |
| 4 | 1 | no | 220 |
| 4 | 2 | no | 220 |
| 5 | 0 | yes | 109 |
| 5 | 1 | yes | 108 |
| 5 | 2 | yes | 101 |
| 7 | 0 | yes | 157 |
| 7 | 1 | yes | 121 |
| 7 | 2 | no | 220 |

## Efficiency and timing

Realized reuse including full-refresh decisions: 44.8% (114.6/256 tokens per policy decision). Selective-step reuse: 56.1%. Full-refresh frequency: 20.2% (298/1476); selective decisions: 1178.

Skipped projection rows across all 32 layers: K 5,413,888, V 5,413,888. Mean skipped rows per decision: K 3667.9, V 3667.9. These counts are structural and are not a speedup claim.

Selection time: 1.431 ms mean. Attention capture on R5 refreshes: 3.854 ms mean; amortized 0.778 ms per decision. Synchronized end-to-end latency: refresh 1405.74 ms mean; reuse 1461.58 ms mean. Historical baselines were not timed under fully matched instrumentation, so no latency improvement is claimed.

## Historical regression references

| Policy | Success on established 9-scenario matrix |
|---|---:|
| Fresh OpenVLA | 8/9 |
| Stage 3C oracle FG + attention | 9/9 |
| Stage 4A attention-only | 8/9 |
| V2 hybrid R5, cutoff 0.20 | 7/9 |
| Adaptive hybrid | 5/9 |
| Conservative V2 hybrid R5, cutoff 0.15 | 5/9 |

The established nine scenarios are a regression benchmark, not an independent blind holdout. Different policies produce different observation trajectories. See `per_scenario_metrics.csv` and `scenario_results/` for per-scenario and per-step outcomes.

## Mask and validation artifacts

- `validation_smoke.json` — five-step CUDA smoke, full-refresh action/logit parity, R5 cache checks and 32-layer projection skip.
- `mask_visualizations/` — live semantic, attention-protection, and reuse-candidate masks.
- `offline_preflight/visualizations/` — 12 E2 probability-grid visual comparisons.
- `configuration.json`, `summary.json`, `per_scenario_metrics.csv`, `RUN_STATUS.json` — frozen setup and metrics.

## Interpretation and next protocol

The offline caches show a recall/coverage tradeoff, but only the live nine-scenario outcome tests whether it affects policy success. Any association between more semantic protection and task success is not causal in this single threshold comparison. No broader threshold calibration was performed. For a later study, freeze a development scenario set disjoint by task from the held-out set before tuning 0.10/0.15/0.20, choose by task success then completion efficiency, measured reuse, and timing, and evaluate the frozen choice once on held-out tasks. Do not use these nine regression scenarios as independent validation.
