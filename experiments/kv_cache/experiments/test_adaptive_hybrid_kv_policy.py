import unittest

import numpy as np
import torch

from experiments.kv_cache.experiments.adaptive_hybrid_kv_policy import (
    TokenAgeTracker,
    compose_adaptive_mask,
    confidence_requires_refresh,
    rolling_merge_rows,
    top_two_probability_margin,
)


class AdaptiveHelpersTest(unittest.TestCase):
    def test_union_and_overlap_accounting(self):
        p = np.full(256, 0.1, np.float32)
        p[5:10] = 0.3
        prev = np.zeros(256, bool)
        prev[0:5] = True  # Five tokens stayed foreground; five transition to background.
        attention = np.arange(256, dtype=np.float32)
        selected = compose_adaptive_mask(p, attention, prev)
        self.assertEqual(selected["counts"]["foreground"], 5)
        self.assertEqual(selected["counts"]["transition_protected"], 5)
        self.assertEqual(selected["counts"]["attention_protected"], 25)
        self.assertEqual(selected["counts"]["attention_transition_overlap"], 0)
        self.assertEqual(selected["counts"]["fresh_union"], 35)
        self.assertEqual(selected["counts"]["reused"], 221)
        self.assertTrue(np.all(selected["fresh"] | selected["reused"]))
        self.assertFalse(np.any(selected["fresh"] & selected["reused"]))

    def test_attention_transition_overlap_counted_once_in_union(self):
        p = np.full(256, 0.1, np.float32)
        prev = np.zeros(256, bool)
        prev[250:] = True
        attn = np.arange(256, dtype=np.float32)
        result = compose_adaptive_mask(p, attn, prev)
        self.assertEqual(result["counts"]["transition_protected"], 6)
        self.assertEqual(result["counts"]["attention_protected"], 26)
        self.assertEqual(result["counts"]["attention_transition_overlap"], 6)
        self.assertEqual(result["counts"]["fresh_union"], 26)

    def test_threshold_is_probability_threshold(self):
        p = np.linspace(0, 1, 256, dtype=np.float32)
        result = compose_adaptive_mask(p, np.zeros(256), np.zeros(256), probability_threshold=0.20)
        self.assertEqual(result["counts"]["foreground"], int((p >= 0.20).sum()))

    def test_confidence_margin_matches_full_vocab_softmax_definition(self):
        logits = torch.tensor([[2.0, 1.0, -1.0], [0.5, 0.4, 0.3]])
        p = logits.softmax(-1)
        expected = (p.topk(2, dim=-1).values[:, 0] - p.topk(2, dim=-1).values[:, 1]).mean()
        self.assertAlmostEqual(top_two_probability_margin(logits), float(expected), places=7)

    def test_refresh_gate_and_rolling_ages(self):
        self.assertTrue(confidence_requires_refresh(None, 0.65))
        self.assertTrue(confidence_requires_refresh(0.64, 0.65))
        self.assertFalse(confidence_requires_refresh(0.65, 0.65))
        ages = TokenAgeTracker()
        self.assertTrue(np.all(ages.commit_successful_prefill(0, np.zeros(256), full_refresh=True) == 0))
        mask = np.zeros(256, bool)
        mask[3:8] = True
        ages.commit_successful_prefill(1, mask)
        a = ages.ages(4)
        self.assertEqual(a[3], 3)
        self.assertEqual(a[0], 4)
        self.assertTrue(np.all(ages.commit_successful_prefill(5, np.zeros(256), full_refresh=True) == 0))

    def test_rolling_update_is_latest_value_per_selected_token_and_source_is_immutable(self):
        source = torch.arange(2 * 3 * 9 * 4, dtype=torch.float32).reshape(2, 3, 9, 4)
        original = source.clone()
        positions = torch.tensor([1, 4, 7])
        fresh = torch.full((2, 3, 3, 4), -17.0)
        updated = rolling_merge_rows(source, fresh, positions, axis=2)
        self.assertTrue(torch.equal(source, original))
        self.assertTrue(torch.equal(updated.index_select(2, positions), fresh))
        retained = torch.tensor([0, 2, 3, 5, 6, 8])
        self.assertTrue(torch.equal(updated.index_select(2, retained), source.index_select(2, retained)))
        next_fresh = torch.full((2, 3, 1, 4), 99.0)
        next_step = rolling_merge_rows(updated, next_fresh, torch.tensor([4]), axis=2)
        self.assertTrue(torch.all(next_step[:, :, 4, :] == 99.0))
        self.assertTrue(torch.equal(next_step[:, :, 1, :], updated[:, :, 1, :]))


if __name__ == "__main__":
    unittest.main()
