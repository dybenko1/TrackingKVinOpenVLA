"""Talk2DINO-faithful patch-map and COCO-mask utilities for Phase 3."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def short_edge_resize_size(height: int, width: int, output_size: int = 224) -> tuple[int, int]:
    """Match torchvision Resize(int): return (height, width), rounding long edge down."""
    if min(height, width, output_size) <= 0:
        raise ValueError("Image dimensions and output size must be positive")
    if width < height:
        return int(output_size * height / width), output_size
    return output_size, int(output_size * width / height)


def center_crop_offsets(height: int, width: int, output_size: int = 224) -> tuple[int, int]:
    """Match torchvision CenterCrop's rounded (top, left) offsets."""
    return int(round((height - output_size) / 2.0)), int(round((width - output_size) / 2.0))


def sigmoid_masks(similarity: torch.Tensor) -> torch.Tensor:
    """Upstream DINOTextSim2Mask soft-mask path (no stochastic Gumbel)."""
    return torch.sigmoid(similarity)


def weighted_background_cleaning(
    soft_mask: torch.Tensor,
    patch_features: torch.Tensor,
    attention: torch.Tensor,
    text_features: torch.Tensor,
    lambda_bg: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Faithful per-image port of DINOText.similarity_assignment_weighted.

    Args use upstream layout: masks [B,N,H,W], features [B,C,H,W],
    attention [B,heads,H*W], normalized text embeddings [N,C].
    Returns cleaned maps and class-conditioned normalized attention maps.
    """
    if soft_mask.ndim != 4 or patch_features.ndim not in (3, 4) or attention.ndim != 3:
        raise ValueError("Expected mask [B,N,H,W], features [B,C,H,W] or [B,C,P], attention [B,H,P]")
    b, classes, h, w = soft_mask.shape
    if patch_features.shape[0] != b or attention.shape[0] != b:
        raise ValueError("Batch dimensions disagree")
    p = h * w
    if patch_features.ndim == 4 and tuple(patch_features.shape[-2:]) != (h, w):
        raise ValueError("Patch feature map dimensions do not match the score map")
    if patch_features.ndim == 4:
        patch_features = patch_features.reshape(b, patch_features.shape[1], p)
    if patch_features.shape[-1] != p or attention.shape[-1] != p:
        raise ValueError("Patch count does not match map dimensions")
    if text_features.shape != (classes, patch_features.shape[1]):
        raise ValueError("Text features must be [N,C] and align with mask categories")

    image_features = patch_features.reshape(b, patch_features.shape[1], p)
    # [B,heads,C]: mean over patch locations exactly as in upstream.
    head_regions = (attention.unsqueeze(2) * image_features.unsqueeze(1)).mean(dim=-1)
    head_regions = head_regions / head_regions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    head_regions = head_regions.permute(0, 2, 1)  # [B,C,heads]
    text_head_similarity = torch.matmul(text_features.unsqueeze(0), head_regions).softmax(dim=-1)

    # Category-conditioned weighted mixture of the original CLS->patch maps.
    attention_by_class = (
        attention.unsqueeze(1) * text_head_similarity.unsqueeze(-1)
    ).sum(dim=2) / text_head_similarity.sum(dim=-1).unsqueeze(-1).clamp_min(1e-12)
    attention_by_class = attention_by_class.softmax(dim=-1)

    # The upstream implementation min/max scales over its batch and class maps.
    # For this one-image evaluation, this is equivalent to scaling the full
    # [N,P] tensor together, preserving cross-category normalization.
    min_attention = attention_by_class.min()
    max_attention = attention_by_class.max()
    scale = torch.maximum(max_attention, max_attention - min_attention)
    attention_by_class = (attention_by_class - min_attention) / scale.clamp_min(1e-12)
    mask_flat = soft_mask.reshape(b, classes, p)
    mask_min, mask_max = mask_flat.min(), mask_flat.max()
    attention_by_class = attention_by_class * (mask_max - mask_min) + mask_min
    cleaned = (mask_flat + float(lambda_bg) * attention_by_class.unsqueeze(0)) / (1.0 + float(lambda_bg))
    return cleaned.reshape(b, classes, h, w), attention_by_class.reshape(b, classes, h, w)


def decode_compressed_coco_rle(counts: str | bytes) -> list[int]:
    """Decode COCO maskApi's compressed run-length string to integer counts."""
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    out: list[int] = []
    pos = 0
    while pos < len(counts):
        value = 0
        shift = 0
        more = True
        while more:
            code = ord(counts[pos]) - 48
            pos += 1
            value |= (code & 0x1F) << shift
            more = bool(code & 0x20)
            shift += 5
            if not more and (code & 0x10):
                value |= -1 << shift
        if len(out) > 2:
            value += out[-2]
        out.append(value)
    return out


def decode_coco_rle(segmentation: dict, height: int, width: int) -> torch.Tensor:
    """Decode polygon-free COCO RLE (compressed or uncompressed), Fortran order."""
    size = tuple(map(int, segmentation["size"]))
    if size != (height, width):
        raise ValueError(f"RLE size {size} does not match image {(height, width)}")
    counts = segmentation["counts"]
    if isinstance(counts, (str, bytes)):
        counts = decode_compressed_coco_rle(counts)
    flat = torch.zeros(height * width, dtype=torch.uint8)
    offset, value = 0, 0
    for run in counts:
        end = offset + int(run)
        if end > flat.numel() or run < 0:
            raise ValueError("Invalid COCO RLE run")
        if value:
            flat[offset:end] = 1
        offset, value = end, 1 - value
    if offset != flat.numel():
        raise ValueError(f"COCO RLE covers {offset} pixels; expected {flat.numel()}")
    return flat.reshape(width, height).T.contiguous()


def transform_mask_to_policy(mask, output_size: int = 224) -> torch.Tensor:
    """Resize-short-edge + center crop; nearest-neighbor discrete mask geometry."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(mask, dtype=np.uint8)
    h, w = arr.shape
    if h <= 0 or w <= 0:
        raise ValueError("Empty mask")
    new_h, new_w = short_edge_resize_size(h, w, output_size)
    image = Image.fromarray(arr * 255).resize((new_w, new_h), Image.Resampling.NEAREST)
    # torchvision.transforms.CenterCrop uses round((dimension - crop) / 2),
    # which differs from floor division for odd margins. Match that exact
    # geometry so the annotation and Talk2DINO image transform stay aligned.
    top, left = center_crop_offsets(new_h, new_w, output_size)
    image = image.crop((left, top, left + output_size, top + output_size))
    tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).float().div_(255)
    return tensor.unsqueeze(0).unsqueeze(0)


def binary_mask_from_polygons(polygons: Sequence[Sequence[float]], height: int, width: int):
    """Rasterize COCO polygon segments in original image coordinates."""
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        if len(polygon) >= 6:
            draw.polygon(list(zip(polygon[0::2], polygon[1::2])), outline=1, fill=1)
    return torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy())


def resize_scores_to_224(scores: torch.Tensor) -> torch.Tensor:
    """Upsample [N,16,16] soft maps using upstream bilinear align_corners=True."""
    if scores.ndim != 3:
        raise ValueError("scores must be [N,H,W]")
    return F.interpolate(scores.unsqueeze(0), (224, 224), mode="bilinear", align_corners=True)[0]

