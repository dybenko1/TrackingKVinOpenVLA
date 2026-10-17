# OpenVLA native feature extraction and spatial mapping

## Installed model and preprocessing

- Checkpoint: `openvla/openvla-7b-finetuned-libero-spatial` (`dinosiglip-vit-so-224px`); both feature arrays come from this loaded checkpoint.
- Configured TIMM encoders: `vit_large_patch14_reg4_dinov2.lvd142m` for DINOv2 and `vit_so400m_patch14_siglip_224` for SigLIP.
- The OpenVLA image processor emitted `[1, 6, 224, 224]` using `resize-naive` preprocessing. Per-branch input sizes are `[[3, 224, 224], [3, 224, 224]]`, resize interpolation is `['bicubic', 'bicubic']`, resize parameters are `[{'antialias': True, 'interpolation': 3, 'max_size': None, 'size': [224, 224]}, {'antialias': True, 'interpolation': 3, 'max_size': None, 'size': [224, 224]}]`, center-crop parameters are `[{'output_size': [224, 224]}, {'output_size': [224, 224]}]`, and normalization parameters are `[{'inplace': False, 'mean': [0.484375, 0.455078125, 0.40625], 'std': [0.228515625, 0.2236328125, 0.224609375]}, {'inplace': False, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}]`. Stored means and standard deviations are `[[0.485, 0.456, 0.406], [0.5, 0.5, 0.5]]` / `[[0.229, 0.224, 0.225], [0.5, 0.5, 0.5]]`. Letterbox mode: `False`; fill: `None`.
- The feature extractor loaded the full checkpoint to preserve its exact branch weights, left the language model on CPU, and moved only `model.vision_backbone` to CUDA. No language-model forward, action generation, or robot rollout was run for Track B.

## Feature layers and token mapping

| Branch | TIMM encoder | Selected block (zero based) | Blocks | Prefix tokens in ViT | Returned grid | Dimension |
|---|---|---:|---:|---:|---|---:|
| DINOv2 | `vit_large_patch14_reg4_dinov2.lvd142m` | 22 | 24 | 5 | [16, 16] | 1024 |
| SigLIP | `vit_so400m_patch14_siglip_224` | 25 | 27 | 0 | [16, 16] | 1152 |

Both branches use the second-to-last transformer block through the same `get_intermediate_layers()` wrappers used by `PrismaticVisionBackbone`. The installed TIMM accessor strips all prefix tokens before returning its patch sequence; its defaults are `norm=False` for DINOv2 and `norm=False` for SigLIP, with `return_prefix_tokens=False` and `False`. Thus the class token and DINOv2 register tokens are not returned. Runtime shape checks require `[1,256,1024]` and `[1,256,1152]`. Each feature sequence is a row-major 16×16 patch raster from a 224×224 input with patch size 14.

OpenVLA's fused backbone splits the processor's six channels into DINOv2 and SigLIP inputs, then concatenates the branches on the feature axis in DINOv2-then-SigLIP order. It preserves the 256 patch positions. The language-model prefill places those projected tokens at positions 1–256, as independently recorded by the existing Stage 3A cache layout. The benchmark stores the native branch arrays and their concatenation `2176`-dimensionally; it does not run the projector or LLM.

## Spatial alignment check

The raw LIBERO RGB observation and the processor image are both saved. The processor image is produced by the same Stage 3A helper: rotate LIBERO's camera image by 180°, resize it with the existing evaluator path, and apply its centered 0.9-area crop. The semantic mask is flipped in the same orientation and passed through the existing `policy_mask()` nearest-neighbor resize and center crop; `token_mask()` then marks any foreground pixel in each 14×14 region. The saved comparison figures overlay these exact 16×16 labels on the processor input and draw token boundaries, providing a visual patch-to-token alignment check.

The extracted observations replay existing Stage 3A Fresh action traces to reconstruct pixels and instance maps; they do not invoke OpenVLA for actions. Oracle names classified by the explicit background-name rule, robot foreground hints, or foreground-default-unmapped are enumerated in `dataset_manifest.json`. Nonzero segmentation IDs absent from the rendered-label map are also recorded with pixel counts.

## Runtime visual alignment figure

See [`figures/feature_mapping_overlay.png`](figures/feature_mapping_overlay.png): it displays the exact processor input, oracle mask and 16×16 patch grid used for the feature/token alignment audit.
