import unittest

from v2_temporal_consistency_audit import reconstruct_pair, summarize_pairs


class TransitionTests(unittest.TestCase):
    def test_reconstructs_all_four_cells(self):
        row = reconstruct_pair(100, 90, 30)
        self.assertEqual((row["fg_to_fg"], row["fg_to_bg"], row["bg_to_fg"], row["bg_to_bg"]),
                         (70, 30, 20, 136))
        self.assertEqual(sum(row[k] for k in ("fg_to_fg", "fg_to_bg", "bg_to_fg", "bg_to_bg")), 256)
        self.assertAlmostEqual(row["flip_rate"], 50/256)
        self.assertAlmostEqual(row["temporal_foreground_iou"], 70/120)

    def test_rejects_impossible_fg_to_bg_count(self):
        with self.assertRaises(ValueError):
            reconstruct_pair(10, 250, 10)

    def test_rejects_negative_implied_bg_to_fg(self):
        with self.assertRaises(ValueError):
            reconstruct_pair(100, 20, 0)

    def test_random_baseline_matches_independent_draw_formula(self):
        row = reconstruct_pair(64, 128, 32)
        expected = (64/256)*(1-128/256)+(1-64/256)*(128/256)
        self.assertAlmostEqual(row["expected_random_flip_rate"], expected)
        self.assertAlmostEqual(row["chance_adjusted_agreement"], 1-row["flip_rate"]/expected)

    def test_all_identical_masks_have_full_retention_and_iou(self):
        row = reconstruct_pair(80, 80, 0)
        self.assertEqual(row["flip_rate"], 0)
        self.assertEqual(row["fg_retention_rate"], 1)
        self.assertEqual(row["bg_retention_rate"], 1)
        self.assertEqual(row["temporal_foreground_iou"], 1)


if __name__ == "__main__":
    unittest.main()
