# Adaptive Hybrid KV Cache — V2 + Task Relevance

Completed scenarios: **5/9**. This evaluates the combined policy; it does not isolate the effects of confidence gating, transition protection, or rolling storage.

## Runtime model identity

Model: `openvla/openvla-7b-finetuned-libero-spatial` at revision `962318cec55ac10993ff0f5f43eda9a270b4c873`; Transformers `4.49.0`, PyTorch `2.6.0`, attention backend `sdpa`. Native DINO capture is `[1,256,1024]`; fused `[1,256,2176]` is cross-checked against the prior E3 live profile. V2 projection SHA-256 `8232ec8f657da47645d73741b16560a1140bfea89cce6826948b9457ce4854d4`. The auxiliary SigLIP tower identity and context are in `configuration.json`.

## Policy and action-confidence definition

Every observation uses passive native penultimate OpenVLA DINO features to compute `sigmoid(10 × (cosine − 0.20))`; foreground is probability ≥ 0.20. During autoregressive decoding, for each of the seven generated action tokens, probability distributions are softmax-normalized over the full model vocabulary, matching greedy GenerationMixin token selection. The confidence is the mean top-1 minus top-2 probability margin. Step t uses step t−1's confidence to decide full refresh; margin < 0.65 triggers a full refresh. This matches the Gated VLA-Cache OpenVLA definition, so the provisional threshold was used as-is without calibration on these nine scenarios.

Layer-15 relevance is captured during each prefill. On selective steps it is computed from the same full Q and assembled post-RoPE K tensor used by SDPA, including both retained rolling rows and newly projected rows; skipped K projections are not recomputed. That attention only controls the next decision. At t, top 10% (nearest integer, half up) of current predicted background are attention-protected using scores from t−1. Foreground-to-background transitions are also refreshed. All K/V source entries update per token after successful prefill; no periodic refresh or maximum age is imposed.

## Scenario outcomes

| Task | State | Adaptive | Fresh historical | Stage 3C oracle historical | Stage 4A attention-only 90%-fresh historical | V2 hybrid R5 historical |
|---:|---:|---|---|---|
| 4 | 0 | failure / 220 | success / 140 | success / 129 | success / 129 | success / 149 |
| 4 | 1 | failure / 220 | success / 145 | success / 143 | failure / 220 | failure / 220 |
| 4 | 2 | success / 188 | success / 130 | success / 131 | success / 131 | success / 135 |
| 5 | 0 | success / 113 | success / 105 | success / 111 | success / 104 | success / 110 |
| 5 | 1 | failure / 220 | success / 89 | success / 100 | success / 88 | success / 96 |
| 5 | 2 | success / 111 | success / 90 | success / 88 | success / 84 | success / 183 |
| 7 | 0 | success / 122 | success / 111 | success / 152 | success / 117 | failure / 220 |
| 7 | 1 | failure / 220 | success / 120 | success / 124 | success / 140 | success / 118 |
| 7 | 2 | success / 145 | failure / 220 | success / 126 | success / 117 | success / 123 |

The Stage 4A column is its attention-only, 90%-fresh R5 condition. Historical trajectories are not paired: different policies change subsequent states. The nine-scenario matrix is small and does not establish generalization.

## Cache use and confidence gate

Across 1559 decisions including full refreshes, realized reuse was 56.3% (144.1/256 tokens per decision). Selective steps reused 73.2% on average. Refreshes: 361/1559 (23.2%). Mean skipped projection rows per decision: K 4610, V 4610. Total omitted rows across all layers: K 7,187,072, V 7,187,072.

Refresh reasons: `{"initial_full_refresh": 9, "previous_action_margin_below_0.65": 352, "selective_rolling_reuse": 1198}`. Action confidence margin p50/p95: 0.7630/0.9721; threshold .65. Maximum visual-token age was 51 decisions (there is intentionally no age cap). Mean attention-protected tokens on selective steps: 21.1; mean transition-protected tokens: 2.9.

## Runtime and interpretation

V2 selection mean: 1.372 ms. Attention capture mean: full refresh 4.245 ms; selective step 1.983 ms; amortized 2.506 ms/decision. Confidence margin calculation mean: 1.325 ms. Rolling cache update mean: 0.132 ms. Synchronized policy-call latency: refresh mean 1429.84 ms, selective mean 1531.40 ms.

Skipped-row counts are structural and do not demonstrate acceleration. End-to-end decision latency includes V2 scoring, device-to-host transfer and mask composition on both refresh and selective decisions; attention extraction is inside the synchronized policy-call interval and is separately reported as a component, not double-counted. Policy-call latency includes the already-existing greedy decode verification pass. Historical timing is not directly matched; no net speedup is claimed. Selective K/V provenance tracks last recomputation but deeper-layer values still depend on the reused contextual hidden states that produced them; it is not equivalent to fully fresh inference.

## Threshold follow-up protocol

For a later, separate threshold study (not run here), freeze development tasks 0–3, 6, 8, 9 with states 0–2 and hold out tasks 4, 5, 7 with states 0–2. Compare τFG ∈ {0.10, 0.15, 0.20} only on development, rank by task success, completion efficiency, actual reuse, then measured inference cost, and freeze the choice before a single held-out evaluation. The current nine comparison scenarios were not used to tune this run.

## Artifacts

- `summary.json` and `per_scenario_metrics.csv` contain aggregate/scenario outcomes.
- `scenario_results/` contains per-step confidence, refresh reason, token ages, K/V row counts, and attention.
- `mask_visualizations/` shows semantic, attention, transition, union, reuse, and age maps.
- `configuration.json`, `validation.json`, and `RUN_STATUS.json` preserve run identity and validation.

## Source and publication provenance

Policy-execution source commit: `4315ef667965f46c9c1a8d8a49317b616bd1c000`. Finalization/reporting source commit: `c4e27bb258f9739c4e70554d5d031b233ad05dcd` on `thesis/trackb-dense-grounding-v2`. The aggregate result commit and remote verification are recorded in `SUPERVISOR_STATUS.json`.
