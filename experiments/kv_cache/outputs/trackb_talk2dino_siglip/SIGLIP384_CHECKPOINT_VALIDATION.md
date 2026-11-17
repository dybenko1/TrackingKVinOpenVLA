# SigLIP-384 Checkpoint Validation

Date: 2026-11-16

## Artifact identity and integrity

- Hugging Face repository: [`timm/ViT-SO400M-14-SigLIP-384`](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384)
- Pinned revision: [`ac16108d567c4389e6cd2b11c9b8585f7474435b`](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384/tree/ac16108d567c4389e6cd2b11c9b8585f7474435b)
- Local checkpoint directory: `/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384/`
- Weights: `open_clip_model.safetensors` (3,511,918,424 bytes)
- SHA-256: `97faeafa0576c1ba4c92bae739ca06ef5f5d0c2db92f86a6086653d82a262043`
- Result: local SHA-256 matches the official file metadata for the pinned revision.
- The redundant PyTorch `.bin` weights were not downloaded. The existing SigLIP-224 checkpoint was left untouched.

The small accompanying files were downloaded from the same pinned revision:

| File | Bytes |
|---|---:|
| `open_clip_config.json` | 921 |
| `special_tokens_map.json` | 2,200 |
| `tokenizer.json` | 2,422,934 |
| `tokenizer_config.json` | 20,608 |

The detached worker completed successfully. Its state and durable log are at:

- `checkpoint_download/status.json`
- `checkpoint_download/download.log`

The destination is under the ignored `openvla/cache/` tree and is not tracked by Git.

## Configuration and text-tower loading

The pinned `open_clip_config.json` reports `context_length=64`, `embed_dim=1152`, text width 1152, 27 text transformer layers and 16 heads. The text positional embedding shape loaded from weights is `[64,1152]`. The tokenizer is `T5TokenizerFast`, configured with `clean=canonicalize`; padding and EOS IDs are both 1.

OpenCLIP 2.32.0 loaded the full 674-tensor safetensors state on CPU with strict loading. A text-only extraction selected the 330 `text.*` tensors, removed the `text.` namespace prefix solely to load them into the standalone text-tower module, and then loaded strictly with zero missing and zero unexpected keys. No vision tower was moved to GPU for this validation. On identical tokenized inputs, full-checkpoint and text-only text embeddings were exactly equal (maximum and mean absolute difference 0, relative L2 0). Repeated evaluation-mode calls were also exactly equal. Outputs were finite with shape `[3,1152]`.

Numerical comparison used identical CPU model weights, tokenizer output and input tensors; equality was exact, stricter than the configured `atol=2e-5`, `rtol=1e-5` check.

## Runtime and scope

The download used the existing isolated Docker image and official Hugging Face route. The pinned revision, file length and checksum were checked; no package or system dependency was changed. The worker exited successfully after verification.

This validates checkpoint identity, strict text-weight loading, tokenizer configuration and deterministic text embeddings. It does **not** validate patch-text grounding quality, train the model, or establish segmentation performance. No COCO feature extraction, training or LIBERO rollout was run.
