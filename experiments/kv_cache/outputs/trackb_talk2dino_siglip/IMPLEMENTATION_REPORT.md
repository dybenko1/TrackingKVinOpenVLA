# Talk2DINO–SigLIP implementation report

Date: 2026-11-16. Worktree: `cache/worktrees/trackb-talk2dino-siglip`, branch
`thesis/trackb-talk2dino-siglip`. Upstream Talk2DINO reference is pinned at
`6dac47dbe3500df63f3b3df20dd28334290919cc`.

## Implementation

The new isolated package is
`experiments/kv_cache/experiments/trackb_talk2dino_siglip/`. It implements the
Talk2DINO sidecar without changing `DinoSigLIPViTBackbone.forward()` or the
OpenVLA projector, policy, preprocessing, or action path.

- **DINOv2:** `vit_large_patch14_reg4_dinov2.lvd142m`, using the existing
  OpenVLA fine-tuned DINO branch when available. The sidecar calls the final
  TIMM block with `norm=True`, reads the model's prefix count/grid/dimension,
  and excludes CLS plus four registers. It captures final-block QKV through a
  temporary hook removed in `finally`; uses model head count, Q/K norms and
  scale; and softmaxes the CLS-query logits over spatial patch keys. It returns
  final patch features `[B,256,1024]`, attention `[B,16,256]`, and the upstream
  weighted-patch **mean** region descriptor `[B,16,1024]`.
- **Preprocessing:** the sidecar uses bicubic short-edge resize to 224, center
  crop, tensor conversion and ImageNet normalization. This is separate from,
  and does not replace, the standard OpenVLA preprocessing route.
- **SigLIP text:** the loader targets the paired
  `timm/ViT-SO400M-14-SigLIP-384` text tower and loads only `text.*` safetensors
  parameters; it does not instantiate a second vision tower. Context, width,
  output size and positional embedding shape are checked against checkpoint
  configuration. No download or 224 fallback is attempted. A shared OpenCLIP
  tokenizer bundle uses the checkpoint's `canonicalize` cleaning function for
  feature extraction, inference, and untruncated caption auditing.
- **Projection/training:** `Linear(1152,1024) -> Tanh -> Linear(1024,1024)`,
  2,230,272 trainable parameters. It L2-normalizes text and 16 visual region
  descriptors and uses max-score head alignment. The default objective is
  symmetric bidirectional cross entropy at fixed temperature 0.07. An explicit
  legacy option reproduces upstream's additional division by `batch_size**2`.
  Adam, learning rate `1e-4`, batch size 128, 100 epochs and seed 123 are
  configured. Only MLP parameters enter the optimizer; a checkpoint is saved
  only after explicit `--execute`, and existing outputs are not overwritten.
- **Data:** image-level region arrays and caption-annotation text arrays are
  separate. The dataset joins each caption annotation to its image feature,
  preserves both IDs, and logs repeated image IDs within contrastive batches;
  it does not silently deduplicate captions or change to multi-positive loss.
  Train and validation feature manifests are checked for identical feature
  definitions. The upstream train/validation visual-feature mismatch is avoided.
- **Inference adapter:** accepts the already initialized OpenVLA DINO module,
  frozen text tower/tokenizer, and trained MLP. It exposes category-wise 16×16
  similarity and attention-guided maps without patching or altering policy
  inference.

## Checkpoint and caption audit status

The only local SigLIP checkpoint found is the previously verified
`timm/ViT-SO400M-14-SigLIP` revision
`c85ba2402be4b49b5d178f2e2384fdf50f8c7961`; its config and learned positional
embedding specify a **16-token** context. The required paired
`timm/ViT-SO400M-14-SigLIP-384` checkpoint/config is absent. It was not
substituted, downloaded, or loaded. Therefore the 64-token production text
encoder, strict weight-key compatibility, and new context-64 COCO truncation
statistics are not yet validated.

The new shared-tokenizer audit script writes
`caption_audit_siglip384.json`. Its current status is
`blocked_missing_required_checkpoint`, with the precise missing config path.
Earlier `caption_token_statistics.json` values (train 15.1133% and validation
15.1929% longer than 16 tokens) belong to the prior Stage 4A regex cleaner and
16-position checkpoint. They are historical only and must not be interpreted
as Talk2DINO/SigLIP-384 statistics. Once the 384 checkpoint is available, rerun
the audit script to obtain counts using the same OpenCLIP `canonicalize`
wrapper as extraction and inference.

## Validation performed

In the existing isolated `openvla-talk2dino-siglip:20261116` image, with
`--network none` and `CUDA_VISIBLE_DEVICES=''`:

- Python AST syntax validation passed for 22 module and test files.
- 10 CPU-only synthetic unit tests passed. They cover global-token exclusion,
  256×1024 patches, 16-head normalized attention, region construction and hook
  cleanup; tokenizer cleaner consistency and context/positional checks;
  annotation-level ID joins and duplicate-image logging; projection dimensions,
  parameter count and gradients; canonical/legacy losses; frozen feature
  tower behavior; adapter maps; train/validation feature contracts; and the
  fact that the standard OpenVLA backbone source has no sidecar dependency.
- A separate synthetic OpenCLIP 2.32.0 API check built a small text-only tower
  with `_build_text_tower`; it returned finite `[2,12]` output. This validates
  the builder API shape, not compatibility with the unavailable SigLIP-384
  checkpoint weights.
- The existing derived container could run CPU tests. Direct host Python had
  no Torch/OpenCLIP and Docker daemon access required the existing approved
  container route; no dependencies were installed or changed.

Earlier saved GPU smoke evidence in `validation.json` remains historical: it
verified the OpenVLA standard fused feature path and DINO/SigLIP dimensions
under the then-available environment. No model was loaded for this coding
task, and those earlier measurements do not validate the new transform or
SigLIP-384 weight loading.

## Status and scope

- No COCO feature extraction started.
- No GPU training started.
- No LIBERO experiment or rollout started.
- No OpenVLA policy, projector, default backbone forward, model weights,
  dependencies, or environment were changed.
- No commit or push was made. The package remains untracked on the requested
  worktree; no branch switch, reset, stash, or staging occurred.

## SigLIP-384 checkpoint gate (2026-11-16)

The checkpoint dependency was resolved in the separate validation task. The
official `timm/ViT-SO400M-14-SigLIP-384` safetensors artifact was downloaded at
revision `ac16108d567c4389e6cd2b11c9b8585f7474435b`, and its SHA-256 was
independently verified against the pinned revision metadata. The text-only
loader strictly loaded all 330 text tensors with no missing or unexpected
keys; its embeddings were exactly equal to those from the text tower loaded
from the full checkpoint on the same CPU input. See
`SIGLIP384_CHECKPOINT_VALIDATION.md` and `REAL_MODEL_SMOKE_TEST.md`.

The official tokenizer and `canonicalize` process were used to audit all
COCO 2014 captions at the model's 64-token context. Only 1 of 414,113 training
captions exceeded the context; none of 202,654 validation captions did. The
complete statistics and the historical context-16 comparison are recorded in
`caption_token_statistics_64.json` and
`CAPTION_TOKEN_STATISTICS_64.md`.

The bounded real-weight smoke test passed: SigLIP text outputs were finite,
1152-dimensional and deterministic; DINO region outputs had shape
`[B,16,1024]`; the projection MLP, similarity and loss worked; gradients were
confined to the MLP; and the standard OpenVLA fused feature route was exactly
unchanged before and after optional sidecar extraction. The checkpoint gate is
therefore resolved and the code is ready for a controlled 500–1,000-image
extraction pilot. The pilot, full extraction, training and LIBERO evaluation
have **not** been started and require separate authorization. No dependencies,
standard OpenVLA path, previous experiment results, or checkpoint weights were
modified.
