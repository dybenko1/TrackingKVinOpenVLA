"""Recompute threshold-dependent aggregates from saved per-image sweeps."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .prompt_ablation import METHODS, aggregate, choose_thresholds


def read_jsonl(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def backup(path: Path):
    destination = path.with_name(path.stem + "_pre_metric_correction" + path.suffix)
    if path.exists() and not destination.exists():
        shutil.copy2(path, destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    dev = read_jsonl(out / "development_per_image.jsonl")
    final = read_jsonl(out / "evaluation_per_image.jsonl")
    if not dev or not final:
        raise RuntimeError("Both completed per-image JSONL files are required")

    old = json.loads((out / "prompt_ablation_metrics.json").read_text())
    thresholds, calibration = choose_thresholds(dev)
    fixed = {"V1_seven_template": .55, **{m: .50 for m in METHODS if m != "V1_seven_template"}}
    dev_fixed = aggregate(dev, fixed)
    dev_cal = aggregate(dev, {m: thresholds[m] for m in METHODS})
    dev_cal["V1_development_recalibrated"] = dev_cal["V1_seven_template"]
    final_thresholds = {m: (.55 if m == "V1_seven_template" else thresholds[m]) for m in METHODS}
    final_metrics = aggregate(final, final_thresholds)
    final_v1_recal = aggregate(
        final, {**final_thresholds, "V1_seven_template": thresholds["V1_development_recalibrated"]}
    )["V1_seven_template"]

    for name in (
        "development_metrics.json", "development_thresholds.json",
        "evaluation_metrics_secondary_previously_inspected.json", "prompt_ablation_metrics.json",
    ):
        backup(out / name)
    development_doc = {
        "metric_schema": "threshold-consistent-v2",
        "small_object_definition": "positive occupied patch equivalents below 4; IoU recomputed at each method's reported threshold",
        "empty_mask_definition": "no predicted foreground patches at the reported threshold",
        "fixed_original_thresholds": dev_fixed,
        "development_calibrated_thresholds": dev_cal,
    }
    thresholds_doc = {
        "manifest_sha256": dev[0].get("split_manifest_sha256"),
        "selected_thresholds": thresholds,
        "calibration": calibration,
        "method": "maximize native-grid pooled category IoU on development only; thresholds fixed before secondary previously inspected evaluation",
        "metric_schema": "threshold-consistent-v2",
    }
    final_doc = {
        "previously_inspected_holdout": True,
        "thresholds_from_development": {m: final_thresholds[m] for m in METHODS},
        "metrics": final_metrics,
        "V1_development_recalibrated_secondary": final_v1_recal,
        "image_count": len(final),
        "metric_schema": "threshold-consistent-v2",
    }
    summary = {
        **old,
        "development_calibrated": dev_cal,
        "development_fixed": dev_fixed,
        "secondary_holdout_selected_condition": final_metrics,
        "secondary_holdout_image_count": len(final),
        "secondary_holdout_previously_inspected": True,
        "selected_threshold": thresholds[old["selected_prompt_strategy"]],
        "v1_development_recalibrated_threshold": thresholds["V1_development_recalibrated"],
        "metric_schema": "threshold-consistent-v2",
        "small_object_metric": "positive occupied patch equivalents below 4; threshold-consistent pooled IoU",
        "empty_mask_metric": "fraction of positive image/category pairs with no predicted foreground at that method's threshold",
        "aggregation_source": "saved per-image threshold sweep counts; no model inference was rerun",
    }

    def write(path: Path, value):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        tmp.replace(path)

    write(out / "development_metrics.json", development_doc)
    write(out / "development_thresholds.json", thresholds_doc)
    write(out / "evaluation_metrics_secondary_previously_inspected.json", final_doc)
    write(out / "prompt_ablation_metrics.json", summary)

    selected = summary["selected_prompt_strategy"]
    lines = [
        "# Prompt Robustness and Language Ablation",
        "",
        "Frozen V2 epoch 40 was evaluated with cached OpenVLA penultimate DINO patches. No weights were updated. V1 used its seven-template prompt ensemble.",
        "",
        "The binary scores are `sigmoid(10 × (cosine − 0.20))`. Conditions A–D reword known COCO categories; E averages embeddings and F averages probability maps. Thresholds were selected only on the 1,000-image development split by pooled category IoU. The 3,000-image split was previously inspected and is secondary, not a new blind test.",
        "",
        "All aggregate threshold-dependent metrics were rebuilt from saved per-image threshold sweeps; model inference was not rerun. Small-object IoU and empty-mask frequency now use each method's reported threshold.",
        "",
        f"Development selected **{selected}** at threshold **{thresholds[selected]:.2f}**. V1 remains at its published 0.55 threshold; its development recalibration is reported separately at **{thresholds['V1_development_recalibrated']:.2f}**.",
        "",
        "## Development metrics (calibrated threshold)",
        "",
        "| Condition | Threshold | Pooled macro IoU | Positive-pair macro IoU | Precision | Recall | Small-object IoU | Pointing | Empty masks | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        m = dev_cal[method]
        lines.append(
            f"| {method} | {m['threshold']:.2f} | {m['pooled_macro_iou']:.4f} | {m['mean_positive_pair_macro_iou']:.4f} | {m['precision_macro_category']:.4f} | {m['recall_macro_category']:.4f} | {m['small_object_pooled_macro_iou']:.4f} | {m['pointing_accuracy_mean']:.4f} | {m['empty_mask_frequency']:.4f} | {m['mean_pair_auroc']:.4f} |"
        )
    lines += ["", "## Secondary previously inspected 3,000-image holdout", "", "| Condition | Threshold | Pooled macro IoU | Positive-pair macro IoU | Recall | Small-object IoU | Pointing |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        m = final_metrics[method]
        lines.append(f"| {method} | {m['threshold']:.2f} | {m['pooled_macro_iou']:.4f} | {m['mean_positive_pair_macro_iou']:.4f} | {m['recall_macro_category']:.4f} | {m['small_object_pooled_macro_iou']:.4f} | {m['pointing_accuracy_mean']:.4f} |")
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "The prompt experiment measures paraphrase robustness for COCO classes present in V2 training. It does not establish zero-shot performance for unseen attributes such as `black bowl` or concepts such as `plate`, `drawer`, or `robot gripper`; those require independent masks.",
        "",
        "`mean_positive_pair_macro_iou` averages image/category IoU over positive pairs and then categories. `pooled_macro_iou` pools TP/FP/FN within each category before macro-averaging. Pointing accuracy and AUROC do not depend on the binary threshold. Small-object IoU includes positive image/category examples with occupied area below four 16×16 patch equivalents.",
        "",
        "See `development_metrics.json`, `development_thresholds.json`, and `evaluation_metrics_secondary_previously_inspected.json` for category-level results. Earlier aggregates are preserved with `_pre_metric_correction` suffixes.",
    ]
    (out / "PROMPT_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "selected": selected, "threshold": thresholds[selected], "development_images": len(dev), "secondary_previously_inspected_images": len(final), "metric_schema": "threshold-consistent-v2"}, indent=2))


if __name__ == "__main__":
    main()
