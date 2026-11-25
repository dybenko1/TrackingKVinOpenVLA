# Hybrid V2 + Task-Relevance KV Reuse

Status: **smoke validation complete (one scenario, one R5 window)**. No nine-scenario results are claimed here.

V2 epoch-40 threshold 0.20 identifies the current-frame candidate foreground. The top 10% of predicted background, ranked by the most recent Stage 3C layer-15 text-to-visual attention vector, is additionally refreshed. R5 reuses only from an independent full refresh. DINO features are tapped from the same OpenVLA vision call.

| Scenario | Status | Steps |
|---|---|---:|
| Task 4, state 0 | Five-step smoke; success not scored | 5 |

The smoke validation passed the integration assertions. See `VALIDATION_SMOKE_REPORT.md` and `validation_smoke.json`. This report is a placeholder until the detached nine-scenario worker completes.
