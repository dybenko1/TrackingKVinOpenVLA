# Hybrid V2 + Task-Relevance — Configuration Audit

Status: implementation/smoke validation only. The nine-scenario evaluation is paused pending review of the temporal selection behavior described below.

## Model and feature compatibility

- OpenVLA model: `openvla/openvla-7b-finetuned-libero-spatial`; locally resolved Hugging Face snapshot/revision `962318cec55ac10993ff0f5f43eda9a270b4c873`. This is the same checkpoint family used by E2/E3.
- E2/E3 preprocessing: `PrismaticImageProcessor`, resize-naive to 224×224, bicubic interpolation, no letterbox; policy `pixel_values` are `[1,6,224,224]`. The standard OpenVLA path is left unchanged.
- The passive DINO hook observes the existing penultimate DINOv2 output `[1,256,1024]` once per policy input. E3's recorded native-integration result confirms the fused visual features are `[1,256,2176]`; the hybrid smoke independently confirms the DINO shape and its mapping to LLM visual positions `[1,256]` (sequence positions 1–256).
- V2 projection: frozen epoch-40 `best_projection.pt`, SHA-256 `8232ec8f657da47645d73741b16560a1140bfea89cce6826948b9457ce4854d4`, matching E2.
- Auxiliary text tower: frozen `timm/ViT-SO400M-14-SigLIP-384`, revision `ac16108d567c4389e6cd2b11c9b8585f7474435b`; 64-token context, 1,152 output dimensions, strict text-only state loading, with no SigLIP visual tower loaded. E2 recorded OpenCLIP 2.32.0 and PyTorch 2.6.0.
- Stage 3C's sidecar explicitly refuses to run unless the model attention implementation is `sdpa`; successful completion of the five-step smoke therefore confirms SDPA was active and the policy's regular attention path was retained.

**Runtime caveat:** the successful smoke used Transformers 4.49.0 and tokenizers 0.21.0. The OpenVLA remote model code emitted its warning that it expects Transformers 4.40.1 and tokenizers 0.19.1. Thus the live path is functionally smoke-tested, but the dependency versions are not the checkpoint code's expected pins. No package was changed for this audit. This mismatch should be considered when interpreting parity beyond the same-runtime smoke comparison.

## V2 scoring and E2 parity

The hybrid computes normalized cosine similarity between the normalized native DINO patches and normalized projected SigLIP text embeddings, then applies exactly:

```text
prob = sigmoid(10 * (cosine_similarity - 0.20))
foreground = prob >= 0.20
```

The text prompts come from E2's `instruction_phrases()` function and use the same `a photo of a {phrase}` form. That function appends the fixed `robot arm` and `robot gripper` prompts. It does not add a fixed `table` prompt; a table/fixture phrase is included only when extracted from the task instruction. Text embeddings are encoded/projected once per episode/task and L2-normalized; patch features are L2-normalized per frame, as in E2.

The hybrid takes the maximum probability over prompts before thresholding. This is equivalent to E2's `any(prompt_probability >= threshold)` union. I replayed that comparison against the saved E2 probability arrays: **36 cache files, 1,152 frames, zero foreground-mask mismatches** at threshold 0.20. This validates the union/threshold logic on the exact saved E2 score inputs. The saved E2 caches contain probabilities rather than a separately persisted binary mask array, so this check reconstructs the E2 mask from its saved scores. It is not a fresh live-model score comparison on the same RGB frames as the hybrid policy smoke.

## Actual temporal selection behavior

The implementation currently differs from the earlier “compute a combined mask on refresh and hold it for four reuse steps” specification:

1. **At each R5 full refresh (age 0):** the normal OpenVLA vision forward supplies the passive DINO features; V2 scores/mask are computed. The layer-15 Q/K sidecar computes attention relevance during the same full Llama prefill. The two masks are combined and an independent full K/V anchor is saved.
2. **At each reuse step (ages 1–4):** the normal current-frame OpenVLA vision forward again supplies DINO features, and V2 scores/mask are recomputed before the Llama layers run. The attention vector remains the one captured at the most recent independent refresh. The top 10% is re-ranked among the *current* V2-background tokens on every reuse step.
3. Therefore, **the V2 portion and final reusable-token set can change within one R5 window**. Attention scores do not change until the next refresh.
4. The selector does not update the source cache: **every reused K/V value still comes only from the last independent full refresh**, with age 1–4. Smoke assertions confirmed anchor immutability and this source-age range.

This per-frame V2 choice was made because its current DINO features are already available before Llama, so it avoids an extra DINO forward and lets the current-frame mask govern the projections being skipped. It is nevertheless a real temporal-policy deviation and is not being changed in this audit.

**Overhead:** no second DINO forward and no per-frame SigLIP text encoding are performed. Reuse frames add patch normalization, prompt-to-patch matrix multiplication, sigmoid, thresholding, and re-ranking over 256 tokens. The five-step smoke did not isolate/timestamp this V2 selection cost. The refresh sidecar separately recorded about 83.6 ms for its extra attention-score calculation on the smoke input; that is refresh-only. No net latency improvement is established.

## Smoke and pause state

The published `VALIDATION_SMOKE_REPORT.md` records one R5 cycle: step 0 refresh, steps 1–4 reuse; exact fresh logits/action-token parity passed; 32 layers validated; Q remained full; selected K/V rows were skipped; and the independent source anchor remained unchanged. Four CPU helper tests also passed. This is integration evidence, not task-success evidence.

Before this review request arrived, the detached full-run worker had started task 4/state 0 and was stopped at step 125 (126 steps recorded). That partial scenario is preserved locally and is **not** counted as a completed scenario or experimental result. The nine-scenario evaluation remains paused; do not resume it until the configuration is approved.
