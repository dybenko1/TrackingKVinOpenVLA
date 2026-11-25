# Adaptive Hybrid KV — Live Integration Validation

Status: **PASSED**. These are implementation and cache-correctness checks only, not nine-scenario policy results.

## Runtime identity

- OpenVLA: `openvla/openvla-7b-finetuned-libero-spatial` at Hugging Face revision `962318cec55ac10993ff0f5f43eda9a270b4c873`.
- Transformers `4.49.0`, PyTorch `2.6.0`, SDPA attention backend.
- Native DINO module `VisionTransformer` captured once per observation as `[1,256,1024]`. E3's previous live integration profile verified the fused OpenVLA visual sequence `[1,256,2176]` and unchanged normal forward path.
- Frozen E2 epoch-40 projection checkpoint: `training_v2_dense_20261118/best_projection.pt`, SHA-256 `8232ec8f657da47645d73741b16560a1140bfea89cce6826948b9457ce4854d`.
- Frozen paired SigLIP Text checkpoint: `timm/ViT-SO400M-14-SigLIP-384`, revision `ac16108d567c4389e6cd2b11c9b8585f7474435b`, context 64, dimension 1152. Strict text-only load passed with 0 missing keys and no unexpected keys.
- Scoring: `sigmoid(10 * (cosine - 0.20))`; foreground when probability ≥ 0.20. The cosine offset and probability cutoff are independent.

## Checks that passed

- Fresh instrumentation produced exactly the same action token IDs and prefill logits as the ordinary policy call.
- Native DINO ran once per observation; visual positions aligned with LLM positions 1–256.
- All 32 layers retained full Q projection and performed selected K/V projection skipping.
- Current K/V on selected rows matched fresh projection outputs; retained rows matched the latest rolling source; source storage remained immutable during each prefill.
- An eight-step live trace demonstrated selective reuse through token age 5, no periodic R5 refresh, and a full reset at step 6.
- A separate live sequence triggered a confidence-based refresh on step 1 using a validation-only threshold 0.99; production threshold remains 0.65.
- Six CPU unit tests passed.

Action confidence is the mean over the seven generated action tokens of top-1 minus top-2 probability after softmax over the full vocabulary. The manual greedy token IDs matched `predict_action()`, verifying the logits and decoding semantics used for the margin. The previous action's margin controls the next prefill. The provisional OpenVLA threshold 0.65 matches the top-two probability-gap definition reported in [Gated VLA-Cache](https://arxiv.org/abs/2608.10824); it was not tuned on these nine comparison scenarios.

## Limits

This validates the mechanics, not task preservation, cache equivalence to fully fresh inference, acceleration or generalization. Later-layer K/V still depend on hidden states that consumed retained earlier-layer values. The container reports Transformers 4.49.0/tokenizers 0.21.0 while OpenVLA source warns that it expects 4.40.1/0.19.1; the established image and dependencies were left unchanged. Host `nvidia-smi` returned an Orin device-handle error, but PyTorch inside the approved Docker image reported CUDA available and the Orin device; live validation passed.

Machine-readable evidence: `validation.json`, `validation_configuration.json`. Per-step traces remain local under the ignored `validation_prelaunch_20261125_retry9/` folder.
