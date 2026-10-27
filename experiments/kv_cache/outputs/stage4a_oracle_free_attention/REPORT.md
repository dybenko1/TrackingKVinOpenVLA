# Stage 4A — Oracle-Free Lagged Attention Foreground Selection

Status: **complete**. This is the revised attention-based Stage 4A, not successful UNMATCH-style SigLIP grounding.

At each R5 refresh, a read-only layer-15 sidecar captures text-query-to-visual-key attention during the normal prefill. The attention-ranked mask is held through the following four steps. The top-K patches are operationally selected for fresh K/V; the remaining patches are reuse candidates. No oracle segmentation values enter policy selection. A seeded random mask uses the same exact fresh-token count as a nonsemantic control.

Fresh budgets are 90%, 75%, and 60% (rounded to 230, 192, and 154 visual positions; reuse candidates 26, 64, and 102). K and V true projection skipping covers all 32 Llama layers; Q remains full. Refresh sources are independent full fresh prefill snapshots only; reuse outputs are never promoted.

| Fresh budget | Mask | Success | Mean candidate K rows skipped / decision | Mean candidate V rows skipped / decision |
|---:|---|---:|---:|---:|
| 90% | attention | 8/9 | 663.1 | 663.1 |
| 90% | random | 8/9 | 663.6 | 663.6 |
| 75% | attention | 9/9 | 1631.6 | 1631.6 |
| 75% | random | 4/9 | 1636.0 | 1636.0 |
| 60% | attention | 8/9 | 2601.1 | 2601.1 |
| 60% | random | 6/9 | 2604.8 | 2604.8 |

Fresh and Stage 3B outcomes are reused from matching saved scenario records; Stage 3C oracle-assisted outcomes are attached per scenario. The comparisons are descriptive, not evidence of significance or generalization.

Score maps are lagged by up to four control decisions. A relevant patch that becomes important after a refresh remains eligible for reuse until the next refresh. The selected set means fresh-computation priority, not confirmed object foreground. Skipped projection rows are structural counts, not latency savings.

Inference wall-time values are synchronized per-call diagnostics collected with read-only projection hooks; they are not a dedicated timing benchmark. Sidecar capture, mask-selection, and policy wall times are separately logged.
