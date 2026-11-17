# Talk2DINO-SigLIP Preflight Audit

Audit date: 2026-11-17 (Asia/Shanghai)

## A. Git safety

- The active audit worktree is `/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/worktrees/trackb-talk2dino-siglip`; its Git root is that directory and its branch is the expected `thesis/trackb-talk2dino-siglip`.
- The original parent checkout remains at `/home/oscar/projects/TrackingKVinOpenVLA/openvla` on `thesis/trackc-kv-block-memory`; it was not switched or changed.
- The audit worktree has no staged or tracked-file modifications. The Talk2DINO module is an existing untracked package from the preparation work; this audit added `audit_coco_caption_lengths.py` to that package. Generated output and reference source reside under ignored cache/output paths. No files were staged, committed, or pushed.
- There is one applicable `.gitignore` in each checkout (no nested `.gitignore`). The parent checkout's root rule at `.gitignore:148` is `cache/`; it ignores the dataset and DINOv2 reference clone. `git check-ignore -v` matched the extracted train/validation annotations, sample train/validation images, and the DINOv2 reference source to this rule.
- `git ls-files cache/datasets/talk2dino/coco2014` returned no paths. The parent checkout reports the cache as ignored (`!! cache/`); no dataset files are tracked or pending additions. Representative annotation and image files remain present. The validated archives remain on disk.
- No `.gitignore` edit or `git rm --cached` was needed.

## B. Official DINOv2 reference

An apparent PyTorch Hub cache directory existed at `cache/models/torch/hub/facebookresearch_dinov2_main`, but it has no independent Git metadata: Git resolution climbed into the parent OpenVLA checkout. Its selected DINO source files matched the official source after retrieval, but the directory alone did not establish provenance. A separate shallow clone was therefore made, without weights, at:

`/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/references/dinov2/`

- Remote: `https://github.com/facebookresearch/dinov2.git`
- Branch: `main`
- Commit: `7764ea0f912e53c92e82eb78a2a1631e92725fc8` (`Safely load weights from specified URLs (#598)`)
- Clone time: 2026-11-17 13:23 Asia/Shanghai.
- The clone is clean and ignored by the parent repository's `cache/` rule.

Relevant code inspected:

- `dinov2/models/vision_transformer.py:216-231` prepares CLS, patch and register tokens; `:257-271` returns normalized CLS/register/patch states; `:301-321` extracts intermediate states and removes CLS plus register tokens from spatial patches.
- `dinov2/layers/attention.py:36-77` defines QKV projection and the fused scaled-dot-product attention path; it returns the attended output, not the attention probability matrix. `:80-100` contains the memory-efficient xFormers path.
- `dinov2/layers/block.py:43-100` defines the transformer block and residual path.
- `dinov2/hub/backbones.py:144-153` registers the ViT-L/14-register backbone with four register tokens; `hubconf.py:7-8` exports the regular and register variants.

These are the appropriate references for the final-layer patch and CLS-to-patch attention adaptation. They also show why attention probabilities need explicit QKV-based reconstruction or a different instrumentation path: the official fast attention forward does not expose them. The original OpenVLA model path remains untouched.

## C. SigLIP text architecture

- Checkpoint: `timm/ViT-SO400M-14-SigLIP`, revision `c85ba2402be4b49b5d178f2e2384fdf50f8c7961`.
- Local weights: `/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/models/timm/ViT-SO400M-14-SigLIP/c85ba2402be4b49b5d178f2e2384fdf50f8c7961/open_clip_model.safetensors`.
- The local safetensors file is 3,509,517,656 bytes and its SHA-256 matches the pinned official metadata: `0a04a48b797e187a568335ab67e57a8958c32cee5a707a3feb6d4c97149dcd9c`.
- `open_clip_config.json` specifies text context length 16, width 1,152, vocabulary size 32,000, 27 transformer layers, 16 heads, and last-token pooling. The text positional embedding is `[16, 1152]`; the text projection output dimension is 1,152. The paired checkpoint's vision architecture is `vit_so400m_patch14_siglip_224`.
- The checkpoint tokenizer is local `T5TokenizerFast`, identified as `timm/ViT-SO400M-14-SigLIP`. Its special IDs include pad/EOS ID 1. The 16-position limit is architectural: the config and learned positional embedding both have length 16, not merely a tokenizer-side truncation setting.
- Existing Stage 4A code loads the checkpoint-local fast tokenizer and sets pad ID to EOS (`stage4a_unmatch_siglip_diagnostics.py:105-107`). It canonicalizes text by lowercasing, replacing underscores with spaces, removing punctuation, and collapsing whitespace, then tokenizes with special tokens (`:110-120`). It excludes special/pad positions for local matching (`:122-125`).
- The isolated SigLIP text-tower smoke test produced finite `[3,1152]` outputs with repeat max-absolute difference 0.0; recorded in `outputs/trackb_talk2dino_siglip/validation.json`.

## D. COCO caption-length statistics

The CPU-only audit tokenized every caption without truncation, including special tokens, using the exact Stage 4A canonicalization and local `T5TokenizerFast`. A caption counts as exceeding the model context when its raw token count is greater than 16. Percentile is nearest-rank (`sorted[ceil(q*n)-1]`).

| Split | Captions | ≤16 tokens | >16 tokens | >16 (%) | Mean | Median | P90 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 414,113 | 351,527 | 62,586 | 15.1133% | 13.7789 | 13 | 18 | 65 |
| Validation | 202,654 | 171,865 | 30,789 | 15.1929% | 13.7779 | 13 | 18 | 62 |

Examples longer than the context include a train caption describing a doorway framing two rooms and their furniture (65 tokens), and a validation caption describing a car model, hotdog and fries, condiments, and people reaching for food (62 tokens). The JSON artifact contains the full examples, canonicalized text, and token pieces.

Machine-readable results: `caption_token_statistics.json` in this directory. Reproduction script: `experiments/kv_cache/experiments/trackb_talk2dino_siglip/audit_coco_caption_lengths.py`; it runs CPU-only in the isolated `openvla-talk2dino-siglip:20261116` image with the local dataset and tokenizer files.

**Normalization caveat:** the prepared `extract_text.py` currently calls OpenCLIP's `canonicalize` cleaner, while this audit intentionally follows the exact manual regex canonicalization in the prior Stage 4A diagnostic. Both use the same checkpoint tokenizer, but those normalization functions are not asserted equivalent here. Align this before interpreting extraction-time truncation metadata as numerically identical to this audit.

## E. Final status

- COCO downloads and extraction were already complete before this audit; archive CRC validation passed for all three ZIPs. Captions and image files remain intact.
- No GPU feature extraction, segmentation pilot, LIBERO experiment, or training was started. The caption audit ran on CPU only.
- No OpenVLA policy, dependency, checkpoint, or environment was modified.
- No commits or pushes were made. No unrelated tracked files were changed.
- The official DINOv2 source clone and audit outputs are in ignored cache/output locations; source code remains separate from the OpenVLA runtime.
