"""Final TIMM DINOv2 patches and Talk2DINO disentangled attention regions."""
from __future__ import annotations

from typing import Any

import torch


def final_patch_and_region_features(dino: Any, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return final normalized spatial patches, per-head attention, and regions.

    Attention follows Talk2DINO's last-block QK reconstruction: use the CLS
    query, discard all prefix keys, and softmax separately over spatial keys for
    each head. Region descriptors reproduce its ``disentangled_self_attn``
    reduction: mean(attention * final normalized patch state, over patches).
    """
    if pixels.ndim != 4 or tuple(pixels.shape[-2:]) != (224, 224):
        raise ValueError(f"Expected [B,3,224,224], got {tuple(pixels.shape)}")
    num_prefix = int(getattr(dino, "num_prefix_tokens", -1))
    num_reg = int(getattr(dino, "num_reg_tokens", max(0, num_prefix - 1)))
    if num_prefix != num_reg + 1:
        raise AssertionError(f"Expected one CLS plus register prefix tokens, got prefix={num_prefix}, registers={num_reg}.")
    grid = tuple(int(x) for x in getattr(dino.patch_embed, "grid_size", ()))
    if len(grid) != 2:
        raise AssertionError(f"TIMM DINOv2 must expose a 2D patch grid, got {grid}.")
    num_patches = grid[0] * grid[1]
    feature_dim = int(getattr(dino, "num_features", getattr(dino, "embed_dim", 0)))
    if feature_dim <= 0:
        raise AssertionError("TIMM DINOv2 does not expose a positive feature dimension.")
    if not dino.blocks:
        raise AssertionError("DINOv2 exposes no transformer blocks.")

    qkv_capture: list[torch.Tensor] = []
    last_attn = dino.blocks[-1].attn
    handle = last_attn.qkv.register_forward_hook(lambda _m, _i, out: qkv_capture.append(out))
    try:
        # Calling get_intermediate_layers directly bypasses OpenVLA's wrapper
        # forward override (which intentionally selects the penultimate block).
        outputs = dino.get_intermediate_layers(
            pixels,
            n={len(dino.blocks) - 1},
            norm=True,
            return_prefix_tokens=True,
        )
    finally:
        handle.remove()
    if len(qkv_capture) != 1:
        raise AssertionError(f"Expected one last-block QKV capture, got {len(qkv_capture)}.")
    item = outputs[0]
    if isinstance(item, (tuple, list)):
        patches = next((x for x in item if isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[1] == num_patches), None)
    else:
        patches = item[:, num_prefix:, :]
    if patches is None or tuple(patches.shape[1:]) != (num_patches, feature_dim):
        raise AssertionError(f"Expected final normalized patch states [B,{num_patches},{feature_dim}], got {None if patches is None else tuple(patches.shape)}.")

    raw_qkv = qkv_capture[0]
    batch, token_count, qkv_width = raw_qkv.shape
    heads = int(last_attn.num_heads)
    if heads <= 0 or qkv_width % (3 * heads):
        raise AssertionError(f"Invalid QKV width/head count: width={qkv_width}, heads={heads}.")
    head_dim = qkv_width // (3 * heads)
    if qkv_width != 3 * feature_dim or token_count != num_prefix + num_patches:
        raise AssertionError(f"Unexpected final QKV shape/attention layout: {tuple(raw_qkv.shape)}, heads={heads}, prefix={num_prefix}, patches={num_patches}.")
    qkv = raw_qkv.reshape(batch, token_count, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    q_norm, k_norm = getattr(last_attn, "q_norm", None), getattr(last_attn, "k_norm", None)
    if q_norm is not None:
        q = q_norm(q)
    if k_norm is not None:
        k = k_norm(k)
    scale = float(getattr(last_attn, "scale", head_dim ** -0.5))
    # Talk2DINO reconstructs the same pre-dropout attention from qkv. FP32
    # accumulation improves diagnostic stability; it does not feed the policy.
    logits = torch.matmul((q.float() * scale), k.float().transpose(-2, -1))
    cls_to_patches = logits[:, :, 0, num_prefix:]
    attention = cls_to_patches.softmax(dim=-1)
    if tuple(attention.shape) != (batch, heads, num_patches):
        raise AssertionError(f"Expected [B,{heads},{num_patches}] CLS-to-patch attention, got {tuple(attention.shape)}.")
    # This is Talk2DINO's actual disentangled_self_attn aggregation: the
    # attention-weighted normalized patch states are averaged over positions.
    regions = (patches.unsqueeze(1) * attention.to(dtype=patches.dtype).unsqueeze(-1)).mean(dim=2)
    if not torch.isfinite(patches).all() or not torch.isfinite(attention).all() or not torch.isfinite(regions).all():
        raise FloatingPointError("DINOv2 feature extraction produced nonfinite values.")
    if not torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-5, rtol=1e-5):
        raise AssertionError("Per-head spatial attention does not normalize to one.")
    return patches, attention, regions
