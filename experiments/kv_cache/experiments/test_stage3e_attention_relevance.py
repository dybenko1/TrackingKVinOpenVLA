"""CPU-only tests for the Stage 3E relevance aggregation/filter helpers."""
from __future__ import annotations

import unittest

import numpy as np

from experiments.kv_cache.experiments.stage3e_attention_relevance import (
    protect_random_background,
    protect_top_background,
    text_to_visual_scores,
)


def _fixture_attention(length: int, visual_start: int, visual_count: int, queries: tuple[int, ...], salient: int) -> np.ndarray:
    probs = np.zeros((1, 2, length, length), dtype=np.float64)
    for query in range(length):
        probs[0, :, query, :query + 1] = 1.0 / (query + 1)
    for query in queries:
        probs[0, :, query, :] = 0.0
        probs[0, :, query, 0] = 0.5
        probs[0, :, query, visual_start + salient] = 0.5
    return probs


class AttentionRelevanceTests(unittest.TestCase):
    def test_query_rows_visual_key_columns_and_layer_average(self):
        start, count, query_positions = 1, 256, (257, 258)
        layer0 = _fixture_attention(260, start, count, query_positions, salient=37)
        layer1 = _fixture_attention(260, start, count, query_positions, salient=91)
        scores = text_to_visual_scores([layer0, layer1], query_positions, visual_start=start, visual_count=count)
        self.assertEqual(scores.shape, (256,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(int(np.argmax(scores)), 37)  # layer0 and layer1 contribute equal mass to their own peak; first wins tie
        self.assertAlmostEqual(float(scores[37]), 0.25)
        self.assertAlmostEqual(float(scores[91]), 0.25)

    def test_compact_text_query_rows_keep_original_sequence_positions(self):
        full = _fixture_attention(260, 1, 256, (257, 258), salient=37)
        compact = full[:, :, [257, 258], :]
        scores = text_to_visual_scores(
            [compact], [0, 1], query_token_positions=[257, 258],
            visual_start=1, visual_count=256,
        )
        self.assertEqual(scores.shape, (256,))
        self.assertEqual(int(np.argmax(scores)), 37)

    def test_top_fraction_excludes_oracle_foreground_and_random_matches_budget(self):
        fg = np.zeros(256, dtype=bool)
        fg[[0, 1, 2, 3]] = True
        scores = np.arange(256, dtype=np.float64)
        protected, reusable = protect_top_background(scores, fg, 0.1)
        self.assertFalse(np.any(protected & fg))
        self.assertEqual(int(protected.sum()), 25)  # round-half-up of 252 * 10%
        self.assertEqual(int(reusable.sum()), 252 - 25)
        self.assertEqual(set(np.flatnonzero(protected)), set(range(231, 256)))
        random_a, reusable_a = protect_random_background(fg, int(protected.sum()), seed=17)
        random_b, reusable_b = protect_random_background(fg, int(protected.sum()), seed=17)
        self.assertTrue(np.array_equal(random_a, random_b))
        self.assertTrue(np.array_equal(reusable_a, reusable_b))
        self.assertEqual(int(random_a.sum()), int(protected.sum()))
        self.assertFalse(np.any(random_a & fg))

    def test_rejects_invalid_axes_or_nonprobabilities(self):
        probs = _fixture_attention(260, 1, 256, (257,), salient=9)
        with self.assertRaisesRegex(ValueError, "follow the complete visual block"):
            text_to_visual_scores([probs], [100])
        probs[0, 0, 257, 0] += 0.25
        with self.assertRaisesRegex(ValueError, "not normalized"):
            text_to_visual_scores([probs], [257])


if __name__ == "__main__":
    unittest.main()
