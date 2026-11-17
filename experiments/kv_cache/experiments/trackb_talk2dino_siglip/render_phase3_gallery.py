#!/usr/bin/env python3
"""Re-render panels from saved Phase 3 maps/metrics without model inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase3_evaluate import (
    CHECKPOINTS, OUT, read_coco, write_qualitative,
)


def main():
    manifest = json.loads((OUT / "evaluation_manifest.json").read_text())
    metric_data = json.loads((OUT / "segmentation_metrics.json").read_text())
    _, _, categories = read_coco()
    cache = {}
    by_id = {int(r["image_id"]): r for r in manifest["images"]}
    dest = OUT / "qualitative_results"
    prior_gallery = dest / "qualitative_manifest.json"
    if not prior_gallery.is_file():
        raise FileNotFoundError("No qualitative manifest from the completed evaluation")
    # Use precisely the images recorded by the evaluator, never stale NPZs left
    # by a previous manifest/run.
    result_ids = [int(Path(row["file"]).stem.split("_")[-1]) for row in
                  json.loads(prior_gallery.read_text())["images"]]
    for iid in result_ids:
        path = OUT / "qualitative_results" / f"predictions_image_{iid}.npz"
        data = np.load(path, allow_pickle=False)
        category_ids = [int(x) for x in data["category_ids"].tolist()]
        category_names = [str(x) for x in data["category_names"].tolist()]
        source = by_id[iid]
        image_path = f"/opt/openvla/cache/datasets/talk2dino/coco2014/val2014/{source['file_name']}"
        cp_data = {}
        for cp in CHECKPOINTS:
            prefix = cp + "__"
            metrics = metric_data["per_image"][cp][str(iid)]["categories"]
            cp_data[cp] = {
                "categories": category_names,
                "category_ids": category_ids,
                "gt16": data["ground_truth_16"].astype(bool),
                "similarity": data[prefix + "raw_cosine"],
                "raw": data[prefix + "sigmoid_map"],
                "cleaned": data[prefix + "background_cleaned_map"],
                "binary_cleaned": data[prefix + "category_binary_masks"].astype(bool),
                "pamr": data[prefix + "pamr_map"] if prefix + "pamr_map" in data else None,
                "metric_items": {int(k): v for k, v in metrics.items()},
                "image_path": image_path,
            }
        cache[iid] = cp_data
    for old in dest.glob("comparison_*.png"):
        old.unlink()
    write_qualitative(cache, categories, with_pamr=True)
    current_gallery = json.loads((dest / "qualitative_manifest.json").read_text())
    keep_png = {row["file"] for row in current_gallery["images"]}
    keep_ids = {int(Path(name).stem.split("_")[-1]) for name in keep_png}
    for path in dest.glob("comparison_*.png"):
        if path.name not in keep_png:
            path.unlink()
    for path in dest.glob("predictions_image_*.npz"):
        if int(path.stem.split("_")[-1]) not in keep_ids:
            path.unlink()
    print(f"Rendered {len(cache)} comparison panels from saved predictions; no model inference ran.")


if __name__ == "__main__":
    main()
