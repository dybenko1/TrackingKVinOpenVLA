# Stage 3C — Oracle Background + Task Relevance, Full-Layer K+V, R5

Status: **complete**; completed protection budgets: [10, 25, 50].

For each fresh R5 refresh, a read-only layer-15 sidecar computes text-query to visual-key attention from the already-produced Q/K projections after RoPE, applies causal masking and averages heads and prompt text queries for the 256 visual key positions. Its scores are reused for the following four reuse decisions. All oracle foreground patches and the selected top-ranked fraction of oracle-background patches remain fresh; only remaining background K/V rows are skipped and sourced from the last independent fresh prefill.

The policy attention backend remains SDPA, and its output is not replaced. The sidecar uses FP32 QK/softmax for ranking, so it is an attention-derived relevance estimate rather than a dump of SDPA's internal probabilities. Its measured refresh-sidecar and mask-ranking costs are recorded separately. No full inference latency benchmark was run.

| Protected background budget | Success | Mean K rows skipped / rollout | Mean V rows skipped / rollout |
|---:|---:|---:|---:|
| 10% | 9/9 | 359808 | 359808 |
| 25% | 8/9 | 311972 | 311972 |
| 50% | 9/9 | 188341 | 188341 |

Reference success counts on the same nine scenarios: Fresh 8/9, Stage 3A 8/9, Stage 3B 5/9. Exact scenario pairs and structural row counts are in the CSV files and threshold subdirectories. Completion-step differences are descriptive only.

This tests one task-relevance safety filter inside the existing oracle-masked full-layer K+V projection-skipping policy. It does not test the complete VLA-Cache method, and skipped projection rows are not measured latency savings.
