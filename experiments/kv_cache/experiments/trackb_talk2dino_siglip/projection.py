"""Trainable Talk2DINO text projection and max-score region alignment."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RegionTextProjection(nn.Module):
    """Project SigLIP text vectors and align them to DINO attention-head regions."""

    def __init__(self, text_dim: int = 1152, hidden_dim: int = 1024, region_dim: int = 1024):
        super().__init__()
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, region_dim),
        )
        self.text_dim = int(text_dim)
        self.region_dim = int(region_dim)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        regions: torch.Tensor,
        text: torch.Tensor,
        *,
        return_head_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return image/text cosine scores; `max_score` chooses the best head.

        Taking a maximum over cosine scores is equivalent to the upstream
        procedure's softmax-over-heads, argmax, gather, then selected cosine.
        The selected index is optionally returned for reproducibility/debugging.
        """
        if regions.ndim != 3 or text.ndim != 2:
            raise ValueError(f"Expected regions [B,H,D] and text [N,D], got {regions.shape}, {text.shape}.")
        if regions.shape[-1] != self.region_dim or text.shape[-1] != self.text_dim:
            raise ValueError(f"Feature dimensions must be DINO {self.region_dim} and SigLIP {self.text_dim}.")
        if regions.shape[1] == 0:
            raise ValueError("At least one attention-head region is required.")
        projected_text = self.text_projection(text.float())
        regions = F.normalize(regions.float(), dim=-1)
        projected_text = F.normalize(projected_text, dim=-1)
        similarities = torch.einsum("bhd,nd->bnh", regions, projected_text)
        scores, head_indices = similarities.max(dim=-1)
        return (scores, head_indices) if return_head_indices else scores


def symmetric_infonce(scores: torch.Tensor, temperature: float = 0.07, legacy_extra_batch_square: bool = False) -> torch.Tensor:
    """Compatibility export; implementation lives in :mod:`losses`."""
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.losses import symmetric_infonce as _loss

    return _loss(scores, temperature=temperature, legacy_extra_batch_square=legacy_extra_batch_square)
