# Stage 4A oracle-free attention selection — validation

Status: **PASS**. The unchanged SDPA policy path produced exactly matching full-fresh prefill logits, greedy action-token IDs, and policy action with/without the read-only Q/K sidecar. The layer-15 sidecar returned 256 finite scores with normalized FP32 probability rows.

A same-observation R5 reuse validation used the top-90%-fresh mask. Hooks verified full-sequence Q rows, 266 K and 266 V projection rows (BOS + 230 selected patches + 35 language positions), and 26 reused visual rows/layer across all 32 layers. Thus 832 K and 832 V rows were omitted per reuse prefill. Cache shape, independent fresh source, original source cache, and normal one-token decode checks passed. The reuse action matched the same-observation fresh counterfactual exactly in this validation sample.

The policy runner contains no oracle segmentation calls or references. Fresh masks are generated from refresh attention rankings; random controls use a deterministic seeded mask. This validates the inherited Stage 3B cache context for one scenario and one budget; it is not evidence of task success preservation.
