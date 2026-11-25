"""CPU-only contract tests for V2 + lagged task-relevance mask composition."""
import unittest

import numpy as np

from experiments.kv_cache.experiments.hybrid_v2_task_relevance import compose_hybrid_mask
from experiments.kv_cache.experiments.stage3e_attention_relevance import protect_top_background


class HybridMaskTests(unittest.TestCase):
    def test_counts_and_stage3c_rounding_match(self):
        probs = np.zeros((2, 256), dtype=np.float32)
        probs[:, :57] = 0.21
        scores = np.arange(256, dtype=np.float32)
        got = compose_hybrid_mask(probs, scores, 0.20, 0.10)
        protected, reusable = protect_top_background(scores, got["v2_foreground"], 0.10)
        self.assertEqual(got["counts"], {
            "v2_foreground": 57, "v2_background": 199,
            "attention_only_protected": 20, "fresh_visual": 77,
            "reusable_visual": 179,
        })
        np.testing.assert_array_equal(protected, got["attention_only_protected"])
        np.testing.assert_array_equal(reusable, got["reusable_visual"])

    def test_v2_foreground_never_reused_and_attention_only_is_background(self):
        probs = np.linspace(0, 1, 256, dtype=np.float32)[None, :]
        scores = np.linspace(1, 0, 256, dtype=np.float32)
        got = compose_hybrid_mask(probs, scores, 0.20, 0.10)
        self.assertTrue(np.all(got["fresh_visual"][got["v2_foreground"]]))
        self.assertFalse(np.any(got["attention_only_protected"] & got["v2_foreground"]))
        self.assertFalse(np.any(got["reusable_visual"] & got["fresh_visual"]))
        self.assertEqual(int(got["fresh_visual"].sum() + got["reusable_visual"].sum()), 256)

    def test_equal_attention_scores_break_ties_by_visual_index(self):
        probs = np.zeros(256, dtype=np.float32)
        scores = np.ones(256, dtype=np.float32)
        got = compose_hybrid_mask(probs, scores, 0.20, 0.10)
        np.testing.assert_array_equal(np.flatnonzero(got["attention_only_protected"]), np.arange(26))

    def test_invalid_scores_fail_closed(self):
        with self.assertRaises(ValueError):
            compose_hybrid_mask(np.zeros(256), np.full(256, np.nan))


if __name__ == "__main__":
    unittest.main()
