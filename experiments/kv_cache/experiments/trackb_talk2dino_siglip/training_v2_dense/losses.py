"""Numerically stable dense grounding and patch-pooled InfoNCE losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dense_pair_losses(logits: torch.Tensor, occupancy: torch.Tensor, valid: torch.Tensor,
                      occupied_equivalents: torch.Tensor, hard_negatives: int = 16,
                      negative_cutoff: float = 0.02, eps: float = 1e-6):
    """Losses for paired examples, shapes [B,C,P]; valid excludes crowd pixels."""
    if logits.shape != occupancy.shape or logits.shape != valid.shape or logits.ndim != 3:
        raise ValueError("logits, occupancy and valid must share [B,C,P]")
    z, y, m = logits.float(), occupancy.float().clamp(0, 1), valid.float().clamp(0, 1)
    positive = y * float(4.0)
    negative = (1.0 - y)
    weights = torch.where(y > 0, positive, negative) * m
    bce_raw = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    per_den = weights.sum(-1).clamp_min(eps)
    bce = (bce_raw * weights).sum(-1) / per_den
    p = torch.sigmoid(z) * m
    yv = y * m
    dice = 1.0 - (2.0 * (p * yv).sum(-1) + eps) / (p.sum(-1) + yv.sum(-1) + eps)
    negative_mask = (m >= 0.98) & (y < float(negative_cutoff))
    neg_logits = z.masked_fill(~negative_mask, -torch.inf)
    k = min(int(hard_negatives), z.shape[-1])
    top = neg_logits.topk(k, dim=-1).values
    top_valid = torch.isfinite(top)
    hard = (F.softplus(torch.where(top_valid, top, torch.zeros_like(top))) * top_valid).sum(-1) / top_valid.sum(-1).clamp_min(1)

    # Sparse/no-positive masks cannot be treated as foreground examples.
    has_positive = ((yv.sum(-1) > 0) & (m.sum(-1) > 0))
    area = occupied_equivalents.float()
    small_weight = torch.where(area < 4, 2.5, torch.where(area < 16, 1.5, 1.0))
    small_weight = small_weight * has_positive
    norm = small_weight.sum().clamp_min(1.0)
    dense_bce = (bce * small_weight).sum() / norm
    dense_dice = (dice * small_weight).sum() / norm
    dense_hard = (hard * small_weight).sum() / norm
    return {"bce": dense_bce, "dice": dense_dice, "hard_negative": dense_hard,
            "valid_positive_pairs": has_positive.sum(), "small_object_weight_mean": small_weight.sum() / has_positive.sum().clamp_min(1)}


def patch_logmeanexp_scores(patches: torch.Tensor, projected_text: torch.Tensor, tau: float = 0.10):
    """Differentiable image-caption scores [B,B], pooling over 256 patches."""
    patches = F.normalize(patches.float(), dim=-1)
    text = F.normalize(projected_text.float(), dim=-1)
    sims = torch.einsum("bpd,cd->bcp", patches, text)
    return float(tau) * (torch.logsumexp(sims / float(tau), dim=-1) - torch.log(torch.tensor(float(patches.shape[1]), device=patches.device)))


def symmetric_infonce(scores: torch.Tensor, temperature: float = 0.07):
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("InfoNCE expects square unique-image scores")
    labels = torch.arange(scores.shape[0], device=scores.device)
    return (F.cross_entropy(scores / temperature, labels) + F.cross_entropy(scores.T / temperature, labels)) / 2
