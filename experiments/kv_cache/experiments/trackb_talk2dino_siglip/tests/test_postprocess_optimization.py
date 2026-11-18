import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "postprocess_optimization.py"
SPEC = importlib.util.spec_from_file_location("postprocess_optimization_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_threshold_sweep_matches_brute_force():
    scores = np.array([[0.1, 0.2, 0.8], [0.8, 0.4, 0.2]], dtype=np.float32)
    target = np.array([[0, 1, 1], [0, 1, 0]], dtype=bool)
    thresholds = np.array([0.2, 0.4, 0.5, 0.8], dtype=np.float64)
    actual = MODULE.sweep_iou(scores, target, thresholds)
    expected = []
    for threshold in thresholds:
        pred = scores >= threshold
        tp = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        expected.append(tp / union if union else np.nan)
    np.testing.assert_allclose(actual, expected)


def test_threshold_sweep_counts_ties_in_positive_class():
    scores = np.array([0.5, 0.5, 0.1])
    target = np.array([True, False, False])
    np.testing.assert_allclose(MODULE.sweep_iou(scores, target, np.array([0.5])), [0.5])


def test_metric_returns_binary_and_threshold_independent_measures():
    scores = np.array([[0.9, 0.1], [0.6, 0.2]])
    target = np.array([[1, 0], [0, 0]], dtype=bool)
    result = MODULE.one_metrics(scores, target, 0.5)
    assert result["iou"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert result["pointing_accuracy"] == 1.0
    assert 0.5 <= result["auroc"] <= 1.0


def test_vectorized_auc_matches_upstream_tie_aware_auc():
    rng = np.random.default_rng(14)
    for _ in range(5):
        # Quantization deliberately creates ties, as in upsampled patch maps.
        scores = rng.integers(0, 8, size=(31, 29)).astype(np.float32) / 7
        target = rng.random((31, 29)) > 0.63
        expected = MODULE.phase3.rank_auc(scores.ravel(), target.ravel())
        actual = MODULE.fast_auc(scores, target)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
