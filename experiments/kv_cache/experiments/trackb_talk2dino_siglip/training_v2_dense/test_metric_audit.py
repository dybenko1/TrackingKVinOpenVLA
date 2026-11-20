import unittest
import numpy as np
import torch

from .evaluate import aggregate, confusion, thresholds_for_method
from .train import eval_dev


class IdentityProjection:
    def __call__(self, value):
        return value


class MockModel:
    text_projection = IdentityProjection()

    def __init__(self, category_text):
        self._category_text = category_text


class MetricAuditTests(unittest.TestCase):
    def test_grid16_confusion_accepts_flat_fractional_targets(self):
        score = np.zeros((16, 16), dtype=np.float32)
        score[0, 0] = 1
        target = np.zeros(256, dtype=np.float32)
        target[0] = 1
        tp, fp, fn, tn = confusion(score, target, np.ones(256), .5)
        self.assertEqual((tp, fp, fn), (1.0, 0.0, 0.0))
        self.assertEqual(tn, 255.0)

    def test_absent_categories_do_not_enter_small_object_iou(self):
        patches = np.zeros((1, 256, 1024), dtype=np.float16)
        patches[0, 0, 0] = 1
        patches[0, 1, 1] = 1  # confident false positive for absent category
        occupancy = np.zeros((1, 2, 256), dtype=np.float16)
        occupancy[0, 0, 0] = 1
        valid = np.ones((1, 256), dtype=np.float16)
        category_text = torch.zeros((2, 1024), dtype=torch.float32)
        category_text[0, 0] = 1
        category_text[1, 1] = 1
        result = eval_dev(MockModel(category_text), patches, occupancy, valid,
                          np.array([0]), torch.device("cpu"), batch=1)
        self.assertAlmostEqual(result["best_small_object_iou"], 1.0, places=6)

    def test_reports_pooled_and_positive_pair_iou_separately(self):
        def rec(iou, tp, fp, fn):
            entry = {"tp": tp, "fp": fp, "fn": fn, "tn": 10,
                     "iou": iou, "precision": tp / max(1, tp + fp),
                     "recall": tp / max(1, tp + fn)}
            return {"categories": {"1": {
                "occupied_patch_equivalents": 1,
                "methods": {"v2_raw": {
                    "grid16": {"0.50": entry}, "224": {"0.50": entry},
                    "grid16_auroc": .5, "pointing_grid16": 1,
                    "pointing_224": 1,
                }},
            }}}
        summary, _ = aggregate([rec(.5, 1, 0, 1), rec(.25, 1, 3, 0)],
                               {"v2_raw": .5})
        metrics = summary["v2_raw"]["grid16"]
        self.assertAlmostEqual(metrics["macro_category_miou_pooled"], 2 / 6)
        self.assertAlmostEqual(metrics["macro_mean_positive_image_category_iou"], .375)

    def test_selected_eval_threshold_is_materialized_not_defaulted_to_point_five_five(self):
        self.assertEqual(thresholds_for_method("v2_raw", {"v2_raw": .20}), [.20])
        self.assertEqual(thresholds_for_method("v1_raw_pamr", {"v1_raw_pamr": .4}), [.55])


if __name__ == "__main__":
    unittest.main()
