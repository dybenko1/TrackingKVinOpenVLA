# Hybrid V2 + Task-Relevance KV Reuse

Execution status: **complete**. Completed scenarios: **9/9**, successes: **7/9**.

Frozen V2 epoch 40 at probability threshold 0.20 marks foreground. The top 10% of current V2-background tokens ranked by the latest Stage 3C layer-15 text-to-visual scores are additionally protected. V2 selection is recomputed each frame; attention scores refresh only at independent R5 anchors. Reused K/V comes only from the latest independent full refresh.

## Success and completion-step comparison

Historical outcomes use the same scenario identifiers/seeds but are not paired trajectories; different actions change subsequent scenes.

| Task | State | Hybrid | Fresh | Stage 3C oracle + 10% | Stage 4A attention 90% fresh | Stage 4A 75% fresh | Stage 4A 60% fresh |
|---:|---:|---|---|---|---|---|---|
| 4 | 0 | success / 149 steps | success / 140 steps | success / 129 steps | success / 129 steps | success / 142 steps | failure / 220 steps |
| 4 | 1 | failure / 220 steps | success / 145 steps | success / 143 steps | failure / 220 steps | success / 130 steps | success / 136 steps |
| 4 | 2 | success / 135 steps | success / 130 steps | success / 131 steps | success / 131 steps | success / 129 steps | success / 127 steps |
| 5 | 0 | success / 110 steps | success / 105 steps | success / 111 steps | success / 104 steps | success / 96 steps | success / 103 steps |
| 5 | 1 | success / 96 steps | success / 89 steps | success / 100 steps | success / 88 steps | success / 90 steps | success / 100 steps |
| 5 | 2 | success / 183 steps | success / 90 steps | success / 88 steps | success / 84 steps | success / 88 steps | success / 83 steps |
| 7 | 0 | failure / 220 steps | success / 111 steps | success / 152 steps | success / 117 steps | success / 109 steps | success / 112 steps |
| 7 | 1 | success / 118 steps | success / 120 steps | success / 124 steps | success / 140 steps | success / 128 steps | success / 119 steps |
| 7 | 2 | success / 123 steps | failure / 220 steps | success / 126 steps | success / 117 steps | success / 131 steps | success / 134 steps |

## Reuse and actual projection skipping

Across 1354 decisions (full refreshes counted as zero reuse), mean actual reuse was 153.04/256 tokens, or 59.8%. Reuse decisions alone averaged 191.69 candidate tokens. Actual skipped rows: K 6,630,880, V 6,630,880; all-layer totals are accumulated over 32 Llama layers.

4,152 reused tokens (2.00%) were V2 foreground at the source refresh. This is a diagnostic only; it did not affect selection.

## Selection overhead and latency

Complete V2 hybrid selection averaged 1.559 ms (p50 1.525, p95 1.672), including CPU transfer/synchronization and mask union/threshold/ranking. GPU scoring averaged 0.990 ms; scoring plus CPU transfer wall time averaged 1.295 ms; ranking/composition averaged 0.264 ms.

Refresh-only attention capture averaged 4.37 ms per refresh and 0.88 ms amortized per policy decision.

End-to-end refresh decision latency mean/p50/p95: 1415.78/1408.90/1421.48 ms. Reuse decision latency: 1464.07/1463.51/1475.91 ms. Historical latency runs are not treated as directly comparable; no speedup claim is made.

Per-scenario coverage, latency distributions, source-mask overlap, and projection-row counts are in `per_scenario_metrics.csv`. Per-step diagnostics are preserved in local scenario records.

## Validation and interpretation

The smoke validation confirmed fresh logits/action-token parity, passive one-pass DINO capture, 256-position alignment, all 32 patched layers, full Q, true K/V skip rows, immutable anchors, and ages 1–4. This is a small fixed scenario matrix, not evidence of broad generalization. Projection rows skipped are structural counts, not a latency claim.

## Artifacts

- `RUN_STATUS.json` — persistent job status.
- `per_scenario_metrics.csv` — outcome, coverage, projection rows, and timing by scenario.
- `scenario_results/` — per-scenario/per-step records.
- `mask_visualizations/` — representative hybrid masks.
- `configuration.json` — frozen configuration and checkpoint identity.
