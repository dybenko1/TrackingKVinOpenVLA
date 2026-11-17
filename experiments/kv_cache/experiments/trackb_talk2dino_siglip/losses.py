"""Talk2DINO bidirectional contrastive objective variants."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def symmetric_infonce(
    scores: torch.Tensor,
    temperature: float = 0.07,
    legacy_extra_batch_square: bool = False,
) -> torch.Tensor:
    """Symmetric image/text InfoNCE with fixed, non-learned temperature.

    ``legacy_extra_batch_square`` reproduces the upstream ``loss.py`` behavior,
    which divides an already mean-reduced pair of cross-entropies by B**2.
    The canonical objective is the mean of the two mean-reduced cross-entropies.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("InfoNCE requires a square paired score matrix.")
    if scores.shape[0] < 1:
        raise ValueError("InfoNCE requires a non-empty batch.")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite image-text scores")
    labels = torch.arange(scores.shape[0], device=scores.device)
    logits = scores / float(temperature)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
    if legacy_extra_batch_square:
        loss = loss / (scores.shape[0] ** 2)
    return loss
