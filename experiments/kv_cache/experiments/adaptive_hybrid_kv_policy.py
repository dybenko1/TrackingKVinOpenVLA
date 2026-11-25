"""Pure adaptive selection, action-confidence and rolling-age helpers.

This module does not touch OpenVLA.  Runtime cache operations live in the
separate adaptive runner so the already-published R5 experiment stays fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def compose_adaptive_mask(
    probabilities: np.ndarray,
    attention_scores: np.ndarray,
    previous_foreground: np.ndarray,
    *,
    probability_threshold: float = 0.20,
    attention_background_fraction: float = 0.10,
) -> dict:
    """Union current V2 foreground, previous-step relevance, and FG->BG edges.

    Attention protects a rounded top fraction of current V2-background patches.
    Its scores must be from t-1; this helper cannot verify time ownership, so the
    runner records and asserts that provenance separately.
    """
    p = np.asarray(probabilities, dtype=np.float32)
    a = np.asarray(attention_scores, dtype=np.float32)
    prev = np.asarray(previous_foreground, dtype=bool)
    if p.ndim == 2:
        if p.shape[1] != 256 or p.shape[0] < 1:
            raise ValueError(f"probabilities must be [prompts,256], got {p.shape}")
        p = p.max(axis=0)
    if p.shape != (256,) or a.shape != (256,) or prev.shape != (256,):
        raise ValueError(f"Expected three 256-entry vectors, got {p.shape}, {a.shape}, {prev.shape}")
    if not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("Selection scores contain nonfinite values")
    if not 0 <= probability_threshold <= 1 or not 0 <= attention_background_fraction <= 1:
        raise ValueError("Selection thresholds must lie in [0,1]")

    foreground = p >= probability_threshold
    background_idx = np.flatnonzero(~foreground)
    n_attention = int(np.floor(len(background_idx) * attention_background_fraction + 0.5))
    order = np.lexsort((background_idx, -a[background_idx]))
    attention = np.zeros(256, dtype=bool)
    attention[background_idx[order[:n_attention]]] = True
    transition = prev & ~foreground
    fresh = foreground | attention | transition
    reused = ~fresh
    return {
        "foreground": foreground,
        "attention_protected": attention,
        "transition_protected": transition,
        "fresh": fresh,
        "reused": reused,
        "counts": {
            "foreground": int(foreground.sum()),
            "attention_protected": int(attention.sum()),
            "transition_protected": int(transition.sum()),
            "attention_transition_overlap": int((attention & transition).sum()),
            "fresh_union": int(fresh.sum()),
            "reused": int(reused.sum()),
        },
    }


def top_two_probability_margin(logits: torch.Tensor) -> float:
    """Mean top-1 minus top-2 softmax probability for action-token logits.

    ``logits`` has one vocabulary distribution per generated action dimension;
    normalization is over the complete model vocabulary, matching greedy
    GenerationMixin decoding (rather than an action-bin-only renormalization).
    """
    z = torch.as_tensor(logits).float()
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 2:
        raise ValueError(f"Expected [action_dimensions,vocabulary>=2], got {tuple(z.shape)}")
    if not torch.isfinite(z).all():
        raise ValueError("Action logits contain nonfinite values")
    p = torch.softmax(z, dim=-1)
    top = torch.topk(p, k=2, dim=-1).values
    return float((top[:, 0] - top[:, 1]).mean().item())


@dataclass
class TokenAgeTracker:
    """Per-visual-token last-recomputation provenance, independent of storage."""
    token_count: int = 256
    last_recomputed_step: np.ndarray | None = None

    def __post_init__(self):
        if self.token_count != 256:
            raise ValueError("This OpenVLA cache layout requires 256 visual tokens")
        if self.last_recomputed_step is not None:
            self.last_recomputed_step = np.asarray(self.last_recomputed_step, dtype=np.int64).copy()
            if self.last_recomputed_step.shape != (self.token_count,):
                raise ValueError("last_recomputed_step must have one entry per token")

    def ages(self, step: int) -> np.ndarray:
        if self.last_recomputed_step is None:
            raise RuntimeError("Token age tracker has not been initialized by a full refresh")
        if np.any(self.last_recomputed_step > step):
            raise ValueError("Token provenance points into the future")
        return int(step) - self.last_recomputed_step

    def commit_successful_prefill(self, step: int, recomputed: np.ndarray, *, full_refresh: bool = False) -> np.ndarray:
        mask = np.asarray(recomputed, dtype=bool)
        if mask.shape != (self.token_count,):
            raise ValueError("recomputed mask must have one entry per visual token")
        if full_refresh:
            mask = np.ones_like(mask)
        if self.last_recomputed_step is None and not full_refresh:
            raise RuntimeError("Cannot selectively update before the initial full refresh")
        if self.last_recomputed_step is None:
            self.last_recomputed_step = np.full(self.token_count, int(step), dtype=np.int64)
        else:
            self.last_recomputed_step[mask] = int(step)
        return self.ages(step)


def confidence_requires_refresh(previous_margin: float | None, threshold: float = 0.65) -> bool:
    """Fail-safe one-step gate; no previous confidence means full refresh."""
    if not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must lie in [0,1]")
    return previous_margin is None or not np.isfinite(previous_margin) or previous_margin < threshold


def rolling_merge_rows(source: torch.Tensor, fresh_rows: torch.Tensor, positions: torch.Tensor, *, axis: int = 2) -> torch.Tensor:
    """Create a new rolling tensor by replacing only this step's fresh rows.

    The source is never mutated. ``fresh_rows`` follows ``positions`` in the
    selected axis and retains the transformer's original token positions.
    """
    if source.ndim != fresh_rows.ndim or source.ndim < axis + 1:
        raise ValueError("source and fresh_rows ranks are incompatible")
    idx = torch.as_tensor(positions, dtype=torch.long, device=source.device)
    if idx.ndim != 1 or fresh_rows.shape[axis] != idx.numel():
        raise ValueError("positions must match the fresh row count")
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= source.shape[axis]):
        raise ValueError("rolling update position is out of bounds")
    result = source.clone()
    result.index_copy_(axis, idx, fresh_rows.to(device=source.device, dtype=source.dtype))
    return result
