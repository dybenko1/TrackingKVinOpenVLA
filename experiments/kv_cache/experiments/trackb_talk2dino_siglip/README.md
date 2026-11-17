# Talk2DINO with SigLIP Text

This isolated module adapts Talk2DINO's frozen DINOv2 features, attention-head
regions, text projection, and contrastive objective to a SigLIP text tower. It
is not wired into OpenVLA action inference, KV reuse, or LIBERO.

## Frozen feature routes

The ordinary OpenVLA route in `prismatic/models/backbones/vision/dinosiglip_vit.py`
is unchanged. It continues to use the checkpoint's paired DINOv2/SigLIP vision
branches, preprocessing, penultimate features, concatenation, and projector.
The sidecar route accepts the already initialized OpenVLA DINOv2 TIMM module and
calls its final-block `get_intermediate_layers(..., norm=True)` path. It excludes
CLS plus four register tokens and captures final-block QKV with a temporary
removed-in-`finally` hook. CLS-to-spatial-patch attention uses the module's
head count, Q/K normalization, and scale. Per-head patch weights are softmaxed
only over the 256 spatial patches. A region is the upstream Talk2DINO reduction
`mean(attention * normalized_patch, dim=patches)`.

Talk2DINO's independent sidecar transform is bicubic short-edge resize to 224,
center crop to 224, tensor conversion, and ImageNet normalization. It does not
replace OpenVLA preprocessing.

## SigLIP text checkpoint gate

The intended text-only checkpoint is `timm/ViT-SO400M-14-SigLIP-384` (64-token
context, 1152 output dimensions). The loader reads the checkpoint's own
OpenCLIP configuration, verifies its positional embedding shape, tokenizer
identity and dimensions, and loads only `text.*` safetensors keys into an
OpenCLIP text tower. It never constructs the paired visual tower, downloads
weights, or falls back to the available 224/context-16 checkpoint. The shared
tokenizer wrapper is used for feature extraction, inference and caption-length
auditing, including the checkpoint's `canonicalize` cleaning function.

The official 384 checkpoint is present in the local ignored cache at pinned
revision `ac16108d567c4389e6cd2b11c9b8585f7474435b`. Its strict text-only
loading, tokenizer configuration, deterministic embeddings, and 64-token COCO
caption audit are documented under
`experiments/kv_cache/outputs/trackb_talk2dino_siglip/`.

## COCO convention and projection

Visual region features are cached once per image. Text vectors are cached once
per caption annotation and joined through `(annotation_id, image_id)`. Training
uses one row per annotation, like the original repository; repeated image IDs
within a batch are logged as potential diagonal-label false negatives rather
than deduplicated or converted to multi-positive labels.

The only trainable model is `Linear(1152,1024) -> Tanh -> Linear(1024,1024)`.
Visual regions and projected text are L2 normalized; `max_score` selects the
highest cosine-scoring head per image/text pair. The fixed-scale canonical
loss is symmetric cross entropy at temperature 0.07. An explicit flag provides
the upstream repository's additional division by `batch_size**2`; it is off by
default. Adam uses lr `1e-4`, batch size 128, 100 epochs, seed 123; no weight
decay is set. Training is impossible without `--execute`, and checkpoints are
never overwritten.

## Commands

Run inside the isolated image/container after the 384 checkpoint is available:

```bash
TB=/opt/openvla/experiments/kv_cache/experiments/trackb_talk2dino_siglip
python3 -m py_compile "$TB"/*.py
PYTHONPATH=/opt/openvla python3 "$TB/audit_coco_caption_lengths.py"
PYTHONPATH=/opt/openvla python3 "$TB/extract_visual.py" --split train --limit 8
PYTHONPATH=/opt/openvla python3 "$TB/extract_text.py" --split train --limit 8
python3 -m unittest discover -s "$TB/tests" -v
```

Full extraction and training remain separate tasks. Extraction supports CPU or
CUDA (CPU extraction will be slow); `train.py` requires `--execute` before
optimization. Feature arrays/checkpoints belong under ignored `cache/` paths,
not Git.

Reference source is `lorebianchi98/Talk2DINO`, pinned locally at
`6dac47dbe3500df63f3b3df20dd28334290919cc`. The upstream README's `--model`
argument/parser mismatch is handled with aliases; both train and validation
manifests require the same visual feature definition.
