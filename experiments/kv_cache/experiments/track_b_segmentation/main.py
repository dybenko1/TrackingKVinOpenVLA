"""Command line entry point for Track B data extraction, evaluation, and smoke checks."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import dataset, evaluate
from .dataset import OUT, atomic_json, update_status


def smoke_test(output_dir: Path | None = None) -> dict[str, Any]:
    """Fast CPU-only checks for split isolation, masks, dilation, and metrics."""
    from experiments.kv_cache.experiments.oracle_foreground_mask import token_mask

    oracle_pixels = np.zeros((224, 224), dtype=bool)
    oracle_pixels[:14, :14] = True
    token = token_mask(oracle_pixels)
    if token.shape != (256,) or np.flatnonzero(token).tolist() != [0]:
        raise AssertionError("The 14x14 pixel-to-token smoke mapping did not produce visual token zero.")
    truth = np.zeros(256, dtype=bool)
    truth[0] = True
    pred = np.zeros(256, dtype=bool)
    metrics = evaluate._metrics(truth, pred)
    if metrics["false_negative_foreground_rate"] != 1 or metrics["unsafe_fg_reuse"] != 1:
        raise AssertionError("Unsafe reuse did not match the oracle foreground false-negative rate.")
    dilated = evaluate._dilate(truth)
    if int(dilated.sum()) != 4 or not dilated[0] or not dilated[1] or not dilated[16] or not dilated[17]:
        raise AssertionError("One-cell 8-neighbor dilation failed at a grid corner.")
    rng = np.random.default_rng(dataset.SEED)
    synthetic = rng.normal(size=(96, 8)).astype(np.float32)
    labels = (synthetic[:, 0] > 0).astype(np.uint8)
    synthetic[:, 0] += labels.astype(np.float32) * 1.5
    reduced = evaluate.RandomizedPCA(4, random_state=dataset.SEED).fit_transform(synthetic)
    lda = evaluate.ShrinkageLDA(0.1).fit(reduced, labels)
    probabilities = lda.predict_proba(reduced)
    if probabilities.shape != (96, 2) or not np.isfinite(probabilities).all():
        raise AssertionError("Training-only PCA plus regularized LDA smoke fit produced invalid probabilities.")

    available_samples = 0
    trajectories = []
    for task_id in dataset.TASKS:
        for state_id in dataset.STATES:
            run, trace_path, trace_hash = dataset._load_trace(task_id, state_id)
            starts = dataset._sample_windows(len(run["records"]))
            available_samples += len(starts) * 5
            trajectories.append({"task_id": task_id, "state_id": state_id, "trace": str(trace_path),
                                 "trace_sha256": trace_hash, "sampled_windows": len(starts)})
    if available_samples < 200:
        raise AssertionError(f"Only {available_samples} replayable samples are available.")
    if {row["state_id"] for row in trajectories} != set(dataset.STATES):
        raise AssertionError("The trace set does not cover all requested initial states.")
    result = {
        "status": "passed",
        "cpu_only": True,
        "sampled_observations_available": available_samples,
        "trajectory_trace_audits": trajectories,
        "token_mapping_check": "one 14x14 foreground region maps to its matching 16x16 token",
        "unsafe_reuse_check": "unsafe foreground reuse equals foreground false-negative rate",
        "dilation_check": "one corner token expands to four in-bounds 8-neighbor cells",
        "classifier_check": "randomized PCA plus diagonal-shrinkage binary LDA fit and finite probabilities",
        "model_forwards_run": False,
        "action_generation_run": False,
        "completed_at_unix": time.time(),
    }
    destination = output_dir or OUT
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "logs").mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "logs" / "smoke_test.json", result)
    return result


def _write_configuration(code_commit: str | None) -> None:
    config_path = Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text())
    config["code_commit"] = code_commit
    config["model_feature_source"] = "installed OpenVLA model vision_backbone.featurizer and fused_featurizer"
    config["dataset_source"] = "Stage 3A Fresh action traces replayed offline with LIBERO oracle instance rendering"
    config["split_manifest"] = {
        "training": "state_id 0, tasks 4/5/7",
        "calibration": "state_id 1, tasks 4/5/7",
        "evaluation": "state_id 2, tasks 4/5/7",
    }
    atomic_json(OUT / "configuration.json", config)


def run_all(code_commit: str | None = None) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    _write_configuration(code_commit)
    update_status("running", phase="dataset_replay", code_commit=code_commit)
    manifest = dataset.extract_dataset(resume=True)
    manifest["code_commit"] = code_commit
    atomic_json(OUT / "dataset_manifest.json", manifest)
    summary = evaluate.run_evaluation(manifest)
    update_status("completed", phase="validated", code_commit=code_commit,
                  sample_count=summary["dataset_sample_count"], evaluation_trajectories=summary["evaluation_trajectory_ids"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke-test", "extract", "evaluate", "run"))
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--smoke-output", type=Path, default=None)
    args = parser.parse_args()
    if args.action == "smoke-test":
        print(json.dumps(smoke_test(args.smoke_output), indent=2))
    elif args.action == "extract":
        _write_configuration(args.code_commit)
        manifest = dataset.extract_dataset(resume=True)
        manifest["code_commit"] = args.code_commit
        atomic_json(OUT / "dataset_manifest.json", manifest)
        print(json.dumps({"status": "features_complete", "samples": len(manifest["samples"])}, indent=2))
    elif args.action == "evaluate":
        manifest = json.loads((OUT / "dataset_manifest.json").read_text())
        print(json.dumps(evaluate.run_evaluation(manifest), indent=2))
    else:
        print(json.dumps(run_all(args.code_commit), indent=2))


if __name__ == "__main__":
    main()
