"""V2 foreground + Stage 3C task-relevance mask composition helpers."""
from __future__ import annotations

import numpy as np


def compose_hybrid_mask(v2_probabilities: np.ndarray, attention_scores: np.ndarray,
                        v2_threshold: float = 0.20,
                        attention_background_fraction: float = 0.10):
    """Return fresh/reuse masks and exact token accounting for one frame.

    V2 scores are probabilities shaped [prompts, 256] (or [256]); attention
    scores are Stage 3C's lagged text-query-to-visual-key vector [256].
    Attention protects the highest-scoring 10% of *V2-predicted background*.
    Ties at the selection boundary are resolved by ascending patch index.
    """
    p = np.asarray(v2_probabilities, dtype=np.float32)
    a = np.asarray(attention_scores, dtype=np.float32)
    if p.ndim == 2:
        if p.shape[1] != 256 or p.shape[0] < 1:
            raise ValueError(f"V2 probabilities must be [prompts,256], got {p.shape}")
        p = p.max(axis=0)
    if p.shape != (256,) or a.shape != (256,):
        raise ValueError(f"Expected two 256-token vectors; got V2={p.shape}, attention={a.shape}")
    if not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("Hybrid mask scores must be finite")
    if not 0 <= v2_threshold <= 1 or not 0 <= attention_background_fraction <= 1:
        raise ValueError("Threshold and attention budget must be in [0,1]")

    semantic_fg = p >= v2_threshold
    candidate_bg = ~semantic_fg
    bg_idx = np.flatnonzero(candidate_bg)
    n_protected = int(np.floor(len(bg_idx) * attention_background_fraction + 0.5))
    # Stable tie break: descending relevance, then ascending visual-token index.
    order = np.lexsort((bg_idx, -a[bg_idx]))
    attention_only = np.zeros(256, dtype=bool)
    attention_only[bg_idx[order[:n_protected]]] = True
    fresh = semantic_fg | attention_only
    reusable = ~fresh
    assert not np.any(semantic_fg & attention_only)
    assert int(fresh.sum() + reusable.sum()) == 256
    return {
        "v2_foreground": semantic_fg,
        "attention_only_protected": attention_only,
        "fresh_visual": fresh,
        "reusable_visual": reusable,
        "counts": {
            "v2_foreground": int(semantic_fg.sum()),
            "v2_background": int(candidate_bg.sum()),
            "attention_only_protected": n_protected,
            "fresh_visual": int(fresh.sum()),
            "reusable_visual": int(reusable.sum()),
        },
    }
