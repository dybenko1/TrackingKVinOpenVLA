# Track B — UNMATCH score cache-safety experiment

## Method

The recovered Stage 4A score implementation was kept unchanged: FP32-normalized penultimate OpenVLA SigLIP patch and paired-checkpoint text token features; cosine matrix `[valid text tokens, 256 patches]`; patch score is the maximum cosine across valid instruction tokens. Otsu is applied per current frame; high-affinity scores are fresh candidates. Stage 3C's lagged layer-15 attention protection selects the top 10% of predicted background (half-up rounding, stable ties). Remaining predicted background K/V uses the latest fresh R5 source. Oracle masks are computed after policy selection solely for post-hoc metrics.

Otsu uses exact unique-score split midpoints; equal values stay together; lower threshold wins objective ties. Constant, nonfinite, empty-class, or invalid score vectors fall back to all-fresh. No oracle mask selects cache rows. This is a per-frame foreground candidate rule, not validated semantic segmentation.

## Offline Otsu preflight

Mean predicted fresh tokens across saved frames: 116.4/256. Oracle comparison: mean precision 0.499, recall 0.451, IoU 0.313, score AUROC 0.485. These oracle statistics are descriptive only. Gallery: `preflight_visuals/otsu_previews.png`.

## Runtime validation

Status: True. Validation record: `validation.json`. It verifies score dimension/finite values, exact fresh logits/token/action invariance with scoring instrumentation, source cloning, R5 age 1 reuse, all 32 layers, full Q, actual K/V row skips, normal greedy decode, and no oracle policy dependency.

## Closed-loop outcomes

See `per_scenario.csv` and `summary.json`. Historical references are not paired trajectories; policies produce different observations.

| Condition | Successes | Scenarios |
|---|---:|---:|
| unmatch_otsu | 7 | 9 |
| matched_random | 6 | 9 |

Fresh historical reference: 8/9. Stage 3C oracle-assisted + 10% task-relevance protection: 9/9. Stage 3B oracle background full-layer KV: 5/9. Stage 4A oracle-free attention results are a separate mask policy; see `historical_references.json`. None of these small scenario sets establishes generalization.

## Efficiency and limitations

Synchronized policy-path latency was measured from processor input preparation through `predict_action`, including one normal vision forward, current score/mask generation, and policy inference (environment stepping excluded): UNMATCH+Otsu mean 902.1 ms, p50 889.8 ms, p95 1082.5 ms; matched-random mean 884.2 ms, p50 890.1 ms, p95 894.2 ms. This is not a paired Fresh-vs-cache speedup benchmark and should not be interpreted as one. Mean score-generation/mask-selection times for UNMATCH+Otsu were 1.87/0.33 ms. Skipped row counts are structural, not latency claims. SigLIP masking reuses the current OpenVLA vision-backbone output already needed by the policy; the runner captures its penultimate SigLIP branch during that same forward. Text states are computed from the paired pretrained text tower, cached by task instruction. The 16-token context can truncate task 4/5 instructions. Current-frame masks lag the K/V source by design, and source/current mask disagreement is observational only.

## Files and reproducibility

Code: `experiments/kv_cache/experiments/trackb_unmatch_cache_safety.py`. Config: `configuration.json`. Validation: `validation.json`. Results are incrementally saved per scenario. No Stage 3/4 output was overwritten.
