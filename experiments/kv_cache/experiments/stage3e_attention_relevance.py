"""Offline helpers for VLA-Cache-style text-query to visual-key relevance.

This module deliberately does not capture model attention or patch inference.
It accepts already-captured attention probabilities in the standard
``[batch, heads, query, key]`` layout and implements only the deterministic
aggregation, ranking, and oracle-background filtering steps. Real attention
capture/backend validation must be completed in the target CUDA runtime.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    """Convert a CPU tensor/array to NumPy without silently accepting CUDA data."""
    if hasattr(value, "detach"):
        value = value.detach()
        if getattr(value, "device", None) is not None and value.device.type != "cpu":
            raise ValueError("Attention tensors must be copied to CPU explicitly before offline analysis.")
        if getattr(value, "dtype", None) is not None and str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def text_to_visual_scores(
    attentions: Sequence[Any],
    query_positions: Sequence[int],
    *,
    query_token_positions: Sequence[int] | None = None,
    visual_start: int = 1,
    visual_count: int = 256,
    probability_tolerance: float = 5e-3,
) -> np.ndarray:
    """Average text-query attention mass for each visual key across layers.

    Each input must be a probability tensor shaped ``[B,H,Q,K]``. Normally it
    is a full self-attention matrix; a compact query-row tensor is also accepted
    when ``query_token_positions`` supplies the original sequence positions of
    those rows. Selected queries must follow the entire visual block so every
    visual key is causally visible. Reductions match the VLA-Cache convention:
    average heads, selected text queries, then selected layers. Averaging batch
    as well makes this helper usable with batched CPU fixtures.

    The returned values are raw mean attention probabilities, not normalized
    to sum to one over visual tokens. Callers choose the layers and query
    positions; this function does not infer prompt/token boundaries.
    """
    if not attentions:
        raise ValueError("At least one layer of attention probabilities is required.")
    queries = np.asarray(query_positions, dtype=np.int64)
    if queries.ndim != 1 or queries.size == 0:
        raise ValueError("query_positions must be a non-empty one-dimensional sequence.")
    if visual_start < 0 or visual_count < 1:
        raise ValueError("visual_start and visual_count must define a non-empty nonnegative span.")
    token_positions = queries if query_token_positions is None else np.asarray(query_token_positions, dtype=np.int64)
    if token_positions.shape != queries.shape:
        raise ValueError("query_token_positions must have one original sequence position per selected query row.")

    layer_scores = []
    expected_q = expected_k = None
    for layer_index, raw in enumerate(attentions):
        probs = _as_numpy(raw)
        if probs.ndim != 4:
            raise ValueError(f"Layer {layer_index}: expected [B,H,Q,K], got shape {probs.shape}.")
        batch, heads, q_len, k_len = probs.shape
        if batch < 1 or heads < 1:
            raise ValueError(f"Layer {layer_index}: expected non-empty [B,H,Q,K] attention.")
        if expected_q is not None and (q_len, k_len) != (expected_q, expected_k):
            raise ValueError("All selected layers must have matching query/key sequence lengths.")
        expected_q, expected_k = q_len, k_len
        visual_end = visual_start + visual_count
        if visual_end > k_len:
            raise ValueError(f"Layer {layer_index}: visual span ends at {visual_end}, sequence has {k_len} keys.")
        if np.any(queries < 0) or np.any(queries >= q_len):
            raise ValueError(f"Layer {layer_index}: query-row index is outside query dimension {q_len}.")
        if np.any(token_positions < 0) or np.any(token_positions >= k_len):
            raise ValueError(f"Layer {layer_index}: original query token position is outside key sequence length {k_len}.")
        if np.any(token_positions < visual_end - 1):
            raise ValueError("Selected queries must follow the complete visual block for causal visibility.")
        if not np.isfinite(probs).all():
            raise ValueError(f"Layer {layer_index}: attention contains non-finite probabilities.")
        if np.any(probs < -probability_tolerance):
            raise ValueError(f"Layer {layer_index}: attention contains negative values.")
        row_sums = probs.sum(axis=-1, dtype=np.float64)
        if not np.allclose(row_sums, 1.0, atol=probability_tolerance, rtol=probability_tolerance):
            max_error = float(np.max(np.abs(row_sums - 1.0)))
            raise ValueError(f"Layer {layer_index}: rows are not normalized attention probabilities (max error {max_error:g}).")

        # Axes are [batch, heads, query, key]: select text query rows and
        # visual key columns, then average batch/head/query dimensions.
        selected = probs[:, :, queries, visual_start:visual_end]
        layer_scores.append(selected.astype(np.float64, copy=False).mean(axis=(0, 1, 2)))

    return np.mean(np.stack(layer_scores, axis=0), axis=0)


def protect_top_background(
    relevance: Any,
    oracle_foreground: Any,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Protect a deterministic top fraction of oracle-background tokens.

    The count uses nearest-integer rounding (half up), capped at the number of
    background positions. Stable descending ranking breaks score ties by
    ascending visual-token index. Returns ``(protected_bg, reusable_bg)``.
    """
    scores = _as_numpy(relevance).astype(np.float64, copy=False)
    fg = _as_numpy(oracle_foreground).astype(bool, copy=False)
    if scores.shape != (256,) or fg.shape != (256,):
        raise ValueError(f"Expected 256 relevance scores and 256 foreground flags; got {scores.shape}, {fg.shape}.")
    if not np.isfinite(scores).all():
        raise ValueError("Relevance scores must be finite.")
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("fraction must be finite and in [0, 1].")
    bg_indices = np.flatnonzero(~fg)
    count = min(len(bg_indices), int(np.floor(len(bg_indices) * fraction + 0.5)))
    # lexsort uses the last key as primary: descending score, ascending index.
    order = np.lexsort((bg_indices, -scores[bg_indices]))
    protected = np.zeros(256, dtype=bool)
    protected[bg_indices[order[:count]]] = True
    reusable = (~fg) & (~protected)
    return protected, reusable


def protect_random_background(
    oracle_foreground: Any,
    protected_count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a reproducible random background set with an exact matched budget."""
    fg = _as_numpy(oracle_foreground).astype(bool, copy=False)
    if fg.shape != (256,):
        raise ValueError(f"Expected 256 foreground flags, got {fg.shape}.")
    bg_indices = np.flatnonzero(~fg)
    if protected_count < 0 or protected_count > len(bg_indices):
        raise ValueError("protected_count must be between zero and the oracle-background count.")
    chosen = np.random.default_rng(seed).choice(bg_indices, size=protected_count, replace=False)
    protected = np.zeros(256, dtype=bool)
    protected[chosen] = True
    return protected, (~fg) & (~protected)
