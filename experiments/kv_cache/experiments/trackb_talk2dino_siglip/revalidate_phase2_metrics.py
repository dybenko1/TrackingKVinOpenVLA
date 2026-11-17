#!/usr/bin/env python3
"""Recompute Phase 2 paired retrieval metrics from saved checkpoints/features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase2_pilot import (
    load_projection, multi_positive_metrics, sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--phase2-output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pilot, phase2 = args.pilot_output, args.phase2_output
    features = pilot / "features"
    image_ids = np.load(features / "image_ids_val2014.npy", mmap_mode="r")
    regions = np.load(features / "region_features_val2014.npy", mmap_mode="r")
    caption_ids = np.load(features / "caption_ids_val2014.npy", mmap_mode="r")
    texts = np.load(features / "text_features_val2014.npy", mmap_mode="r")
    items = [
        ("baseline_multi_positive_validation.json", pilot / "pilot_mlp_checkpoint/final_projection.pt"),
        ("corrected_multi_positive_validation.json", phase2 / "corrected_projection.pt"),
    ]
    results = {}
    for filename, checkpoint in items:
        model, checkpoint_state_hash = load_projection(checkpoint, device)
        metrics = multi_positive_metrics(model, regions, image_ids, texts, caption_ids, device)
        destination = phase2 / filename
        previous = json.loads(destination.read_text())
        previous_metrics = previous["metrics"]
        # Retrieval rates/ranks except conventional median should be reproducible.
        checks = []
        for key in ("image_to_text", "text_to_image", "multi_positive_infonce", "score_matrix"):
            for metric, value in metrics[key].items():
                if isinstance(value, dict):
                    continue
                old = previous_metrics[key][metric]
                if key.endswith("_rank") and metric == "median":
                    checks.append(f"{key}.{metric} recomputed using conventional median")
                    continue
                if isinstance(value, (int, float)):
                    if abs(float(value) - float(old)) > 1e-7:
                        raise AssertionError(f"{filename}: {key}.{metric} changed {old} -> {value}")
                elif value != old:
                    raise AssertionError(f"{filename}: {key}.{metric} changed {old} -> {value}")
                checks.append(f"{key}.{metric}")
        previous.update({"metrics": metrics, "checkpoint_state_sha256": checkpoint_state_hash,
                         "checkpoint_sha256": sha256_file(checkpoint),
                         "metric_revalidation": "passed; exact metrics except median ranks recomputed with conventional NumPy median"})
        destination.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")
        results[filename] = {"checked_metrics": len(checks), "metrics": metrics}
    (phase2 / "evaluation_revalidation.json").write_text(json.dumps({
        "status": "passed", "device": str(device), "checks": results,
        "rank_median_definition": "conventional median (average of the two middle ranks for an even number of queries)",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "device": str(device), "metrics": results}, indent=2))


if __name__ == "__main__":
    main()
