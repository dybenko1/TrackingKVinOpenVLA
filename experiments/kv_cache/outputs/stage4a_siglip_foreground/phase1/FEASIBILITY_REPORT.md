# Stage 4A — Phase 1 SigLIP patch-text feasibility

Status: **NOT ESTABLISHED — stop before foreground-mask implementation or rollouts.**

## What was verified

- The Stage 3B/Track B runtime is available in `openvla-cached:latest`: PyTorch 2.6.0, CUDA 12.8, `timm` 0.9.16, Transformers 4.49.0, CUDA visible on an Orin with 46.54 GiB reported free. No other Docker workload was active during these read-only checks.
- The loaded OpenVLA checkpoint metadata and Track B manifest identify the visual backbone as `dinosiglip-vit-so-224px`, with SigLIP branch `vit_so400m_patch14_siglip_224`. Track B saved 315 native OpenVLA SigLIP feature tensors, each `[256,1152]`, from the second-to-last (block index 25 of 27) vision-transformer block. The row-major 16×16 mapping is already validated by Track B.
- Inside the established container, `timm.get_pretrained_cfg("vit_so400m_patch14_siglip_224")` resolves to `timm/ViT-SO400M-14-SigLIP`, file `open_clip_pytorch_model.bin`. Its paired OpenCLIP configuration specifies the same 1,152-wide SO400M/14 vision tower, a learned MAP image pooler, no additional visual projection, and a 1,152-wide, 27-layer text tower with maximum context length 16. This is a plausible matching pretrained text/vision source, not evidence that patch vectors themselves are text-aligned. [timm model card/config](https://huggingface.co/timm/ViT-SO400M-14-SigLIP/commit/c85ba2402be4b49b5d178f2e2384fdf50f8c7961)
- The SigLIP objective and public inference interface score pooled image/text representations. The OpenVLA features are intermediate patch states, before the final vision block and MAP pooling. Consequently, direct cosine similarity between those patch states and projected text embeddings would not reproduce the trained global scoring path and is not accepted here as a semantic patch score. [SigLIP paper](https://arxiv.org/abs/2303.15343), [Transformers SigLIP docs](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/siglip)

## Blocking diagnostic result

The exact matching paired weights were not available in the container cache. A Transformers load of `google/siglip-so400m-patch14-224` began fetching its 3,509,549,856-byte checkpoint but remained at 20 MiB for more than two minutes. A bounded 10 MB range request took 8.71 seconds; four concurrent 10 MB reads took 79.22 seconds total. The partial download was interrupted and its container removed; a fresh-container check found no persistent partial checkpoint. No prior experiment outputs or persistent model cache were changed. At that transfer rate, obtaining and validating the matching tower in this session is not a safe bounded diagnostic.

The exact SigLIP tokenizer further limits the raw task language: the task 4 instruction is 20 tokens and task 5 is 18 tokens (including EOS), while the matching text context is 16; task 7 is 15. Thus two of the three task descriptions cannot be encoded in full without a separately specified compact-instruction policy. Truncating them would discard target/action wording and is not accepted as an unexamined default.

No patch-text heatmaps were generated: without the matching text weights and a verified image/text pathway, producing a heatmap from raw feature/text cosine or from Track B's oracle-trained LDA would be scientifically misleading. Oracle labels were not used to fit or select anything. No action inference, KV intervention, or rollout was run.

## Feasibility judgment and next step

A defensible *candidate* is possible in principle: encode the task with the paired SigLIP text tower; continue the already-computed OpenVLA SigLIP patch sequence through its final vision block, final normalization, and learned MAP pooler; then use gradient-times-activation or patch ablation to attribute the trained global image–text score back to the 256 input patches. That is a text-conditioned attribution map, **not** a native SigLIP patch-text similarity objective. Before any mask or policy experiment, it must pass (1) numerical alignment of OpenVLA's saved block-25 features with the paired checkpoint, (2) matched versus mismatched instruction controls on held-out Track B frames, (3) spatial review of maps and score variation, and (4) explicit handling of the 16-token text context. These diagnostics remain undone, so Phase 1 has not passed.

Potential alternatives if this attribution is not useful are a separately approved compact task-description schema or, after supervisor approval, a small learned head. Neither is implemented here. Do not proceed to Stage 4A mask generation, KV reuse, or closed-loop evaluation until the Phase 1 diagnostic is completed and reviewed.

## Reproduction / runtime notes

- Branch: `thesis/stage4a-siglip-foreground`.
- Existing Track B inputs: `experiments/kv_cache/outputs/track_b_segmentation/data/samples/` and `dataset_manifest.json` (these raw local artifacts are git-ignored).
- Exact-runtime checks used the existing `openvla-cached:latest` image with `--runtime nvidia`; the container was not kept running.
- The attempted full-model download was intentionally stopped before inference. No dependency or model-weight changes were made to the repository.

## Checkpoint-download infrastructure follow-up (2026-10-25)

The download blocker is **large-file transfer reliability, not DNS or Hub metadata reachability**. The session is on the project machine (`hostname=ubuntu`, project cwd `/home/oscar/projects/TrackingKVinOpenVLA/openvla`); the existing `openvla-cached:latest` container reports Python at `/opt/venv/bin/python`, PyTorch 2.6.0/CUDA 12.8, and an NVIDIA Orin device. The container's active Hugging Face cache is `/data/models/huggingface` (about 29 GB); a read-only search found no matching SO400M SigLIP weights or pre-existing incomplete SigLIP download. `/tmp` has about 34 GB free. No Hugging Face or proxy environment-variable names were set. Installed versions are `huggingface_hub 0.29.1`, `timm 0.9.16`, `safetensors 0.5.3`; `hf_xet` and `open_clip_torch` are not installed. No packages were changed.

The exact compatible repository is `timm/ViT-SO400M-14-SigLIP`, pinned to revision `c85ba2402be4b49b5d178f2e2384fdf50f8c7961` (the revision referenced above), not a 384px or SigLIP2 variant. The official Hub API confirms `open_clip_model.safetensors` is 3,509,517,656 bytes with LFS SHA-256 `0a04a48b797e187a568335ab67e57a8958c32cee5a707a3feb6d4c97149dcd9c`. Its paired `open_clip_config.json` declares image size 224 and `vit_so400m_patch14_siglip_224`; its text tower is 1,152-wide, 27 layers, context length 16. The official repository lists both `.safetensors` and `.bin`; only the safetensors file was selected.

Both `huggingface.co` and `hf-mirror.com`, including their model metadata APIs, returned HTTP 200. Small byte-range reads of the exact safetensors object returned HTTP 206 through the official storage host (`us.aws.cdn.hf.co`) and the mirror's CAS host (`cas-bridge.xethub.hf.co`). The standard Hub CLI initially stalled at 20 MiB on the weight file, and its mirror endpoint returned HTTP 429 on one metadata request. A resumable range downloader with per-chunk range checks, retries across the official and mirror endpoints, and a final official-hash check completed the exact file. Transfer speed varied and a few requests timed out or were rate limited, but retries recovered them. The installed Hub client used ordinary HTTP/range transfer (`hf_xet` is absent); the mirror's data route resolves through a Xet/CAS host. The evidence points to intermittent/throttled large-object CDN/CAS delivery, rather than DNS or metadata reachability. No proxy environment variables were configured.

The verified small companion files are now available from the pinned revision at:

- `/tmp/stage4a_siglip_checkpoint_224/files/open_clip_config.json` (917 bytes)
- `/tmp/stage4a_siglip_checkpoint_224/files/tokenizer.json` (2,422,934 bytes)
- `/tmp/stage4a_siglip_checkpoint_224/files/tokenizer_config.json` (20,608 bytes)
- `/tmp/stage4a_siglip_checkpoint_224/files/special_tokens_map.json` (2,200 bytes)

The complete weight file is available at `/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/models/timm/ViT-SO400M-14-SigLIP/c85ba2402be4b49b5d178f2e2384fdf50f8c7961/open_clip_model.safetensors`. Its size is exactly `3509517656` bytes and its computed SHA-256 exactly matches the pinned official LFS metadata: `0a04a48b797e187a568335ab67e57a8958c32cee5a707a3feb6d4c97149dcd9c`. The file opened successfully with the installed `safetensors` reader without loading the model onto GPU. It contains 674 tensors, including 330 `text.*` tensors; observed text parameters include a `[32000,1152]` token embedding, `[16,1152]` positional embeddings, and 27 transformer blocks. The config, tokenizer, and special-token files each matched their official pinned-revision Git blob IDs and byte sizes. The tokenizer loaded locally as `T5TokenizerFast` using the existing Transformers installation. The paired text-encoder weights and tokenizer assets are therefore available for a later feasibility investigation; no text/image scoring was run.

The original completed download and its verification record remain under `/tmp/stage4a_siglip_checkpoint_224/`. The verified checkpoint and companion files were copied to the Git-ignored persistent cache path above; the persistent copy independently returned the same SHA-256. The initial 20 MiB CLI and 32 MiB range partials remain in `/tmp` as harmless diagnostic files and are not used. Existing model caches and unrelated worktree edits were left untouched. The temporary detached downloader completed and exited.

No second visual tower was fetched: the source repository has one monolithic safetensors file rather than separate vision/text weight files, so the full file was required to obtain and verify the paired text tower. The alternative `.bin` file was not downloaded. `open_clip_torch` is not installed; installing or using a text-encoder implementation is outside this download task and can be considered during the next approved feasibility investigation.

The scientific conclusion is unchanged: **patch-level SigLIP/text scoring remains unvalidated**. No scoring, heatmaps, model inference, or LIBERO rollouts were performed.
