#!/usr/bin/env python3
"""Offline E2 threshold audit and deterministic RGB mask previews.

Uses the reconciled E2 probability/occupancy arrays only. No feature extraction,
model loading, simulator replay, or policy actions are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DEFAULT_E2 = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/libero_ground_truth_e2_completion"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_audit(e2_dir: Path, out: Path, thresholds=(0.10, 0.15, 0.20)) -> dict:
    # Import the exact E2 aggregation and metric definitions to avoid changing
    # prompt-union semantics, fractional GT treatment, or valid-patch weights.
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.transfer_validation.libero_demo_ground_truth import (
        aggregate_arrays, collect_metrics,
    )

    manifest_path = e2_dir / "libero_task_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cache_root = e2_dir / "role_reconciled_scoring_cache"
    paths = sorted(cache_root.rglob("*.npz"))
    manifest_sha = manifest["manifest_sha256"]
    datasets = {split: aggregate_arrays(paths, split, manifest_sha)
                for split in ("development", "held_out")}
    if not all(datasets.values()):
        raise RuntimeError("E2 cache must contain both development and held-out sequences")

    payload = {
        "status": "complete",
        "method": "E2 collect_metrics reused unchanged over reconciled saved probabilities and role occupancies",
        "e2_manifest_sha256": manifest_sha,
        "cache_file_count": len(paths),
        "sequence_counts": {k: len(v) for k, v in datasets.items()},
        "frame_counts": {k: sum(a["v2_probabilities"].shape[0] for a, _ in v)
                         for k, v in datasets.items()},
        "thresholds": {},
        "prior_e2_context": {
            "heldout_scores_previously_inspected": True,
            "heldout_thresholds_used_for_selection": False,
            "interpretation": "Exploratory regression-benchmark diagnostics, not an untouched validation set.",
        },
    }
    task_rows = []
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        payload["thresholds"][key] = {}
        for split, rows in datasets.items():
            result = collect_metrics(rows, "v2", float(threshold))
            payload["thresholds"][key][split] = result
            for task_id, values in result["per_task_metrics"].items():
                fg = values.get("task_relevant_foreground", {})
                for role in ("manipulated_objects", "destination_or_fixture",
                             "task_relevant_fixtures", "robot_arm", "robot_gripper"):
                    rv = values.get(role)
                    if rv is not None:
                        task_rows.append({"threshold": float(threshold), "split": split,
                                          "task_id": task_id, "role": role,
                                          "recall": rv["recall"],
                                          "precision": rv["precision"],
                                          "false_background_rate": rv["false_background_rate"],
                                          "tp": rv["tp_fp_fn_tn"][0], "fp": rv["tp_fp_fn_tn"][1],
                                          "fn": rv["tp_fp_fn_tn"][2], "tn": rv["tp_fp_fn_tn"][3]})
                if fg:
                    task_rows.append({"threshold": float(threshold), "split": split,
                                      "task_id": task_id, "role": "task_relevant_foreground",
                                      "recall": fg["recall"], "precision": fg["precision"],
                                      "false_background_rate": fg["false_background_rate"],
                                      "tp": fg["tp_fp_fn_tn"][0], "fp": fg["tp_fp_fn_tn"][1],
                                      "fn": fg["tp_fp_fn_tn"][2], "tn": fg["tp_fp_fn_tn"][3]})

    out.mkdir(parents=True, exist_ok=True)
    payload["source_hashes"] = {"manifest": sha256(manifest_path),
                               "role_audit": sha256(e2_dir / "ground_truth_role_audit.json")}
    (out / "offline_threshold_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out / "offline_threshold_per_task.csv").open("w", newline="") as f:
        fields = ["threshold", "split", "task_id", "role", "recall", "precision",
                  "false_background_rate", "tp", "fp", "fn", "tn"]
        w = csv.DictWriter(f, fields, lineterminator="\n"); w.writeheader(); w.writerows(task_rows)
    lines = ["# E2 Offline Threshold Comparison", "",
             "This analysis reuses the reconciled E2 probability scores, simulator occupancy masks, prompt mappings, validity masks, and the original `collect_metrics()` implementation. No model inference, feature extraction, simulator replay, or policy action was run.", "",
             f"Manifest SHA-256: `{manifest_sha}`. Cached sequences/frames: development {payload['sequence_counts']['development']}/{payload['frame_counts']['development']}; held-out {payload['sequence_counts']['held_out']}/{payload['frame_counts']['held_out']}.", "",
             "E2 held-out scores were previously inspected while auditing role mapping. The held-out split is therefore a regression diagnostic, not an untouched test set. The predetermined primary threshold for the live experiment remains 0.15; this diagnostic comparison did not change it.", "",
             "## Task-relevant union", "",
             "| Split | Probability cutoff | Recall | Precision | False-background rate | Candidate BG coverage | Mean reusable tokens |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("development", "held_out"):
        for threshold in thresholds:
            m = payload["thresholds"][f"{threshold:.2f}"][split]
            fg = m["role_metrics"]["task_relevant_foreground"]
            lines.append(f"| {split} | {threshold:.2f} | {fg['recall']:.4f} | {fg['precision']:.4f} | {fg['false_background_rate']:.4f} | {m['foreground']['mean_candidate_background_coverage']:.4f} | {m['foreground']['mean_reusable_tokens']:.1f}/256 |")
    lines += ["", "## Role recall at the primary cutoff 0.15", "",
              "| Split | Manipulated | Destination object | Destination/fixture prompt | Task fixture | Robot arm | Robot gripper |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("development", "held_out"):
        m = payload["thresholds"]["0.15"][split]["role_metrics"]
        vals = [m.get(r, {}).get("recall") for r in ("manipulated_objects", "destination_objects", "destination_or_fixture", "task_relevant_fixtures", "robot_arm", "robot_gripper")]
        lines.append(f"| {split} | " + " | ".join("—" if v is None else f"{v:.4f}" for v in vals) + " |")
    lines += ["", "## Across-task variability at 0.15", "",
              "Task-level recall values are in `offline_threshold_per_task.csv`. The task-relevant union includes manipulated objects, destination objects, task-relevant fixtures, robot arm, gripper, and robot mount per the E2 role-reconciled evaluator. The physical foreground labels and prompt roles are diagnostics; candidate background is not proven KV-safe.", ""]
    (out / "OFFLINE_THRESHOLD_REPORT.md").write_text("\n".join(lines))
    return payload


def render_previews(e2_dir: Path, out: Path, max_tasks: int = 12) -> int:
    from PIL import Image, ImageDraw

    manifest = json.loads((e2_dir / "libero_task_manifest.json").read_text())
    by_id = {t["task_id"]: t for t in manifest["tasks"]}
    npzs = sorted((e2_dir / "role_reconciled_scoring_cache").rglob("*.npz"))
    chosen = {}
    for path in npzs:
        with np.load(path, allow_pickle=False) as z:
            md = json.loads(str(z["metadata"].item()))
        chosen.setdefault(md["task_id"], path)
    vis = out / "visualizations"; vis.mkdir(parents=True, exist_ok=True)
    written = 0
    for task_id, npz in list(sorted(chosen.items()))[:max_tasks]:
        task = by_id[task_id]
        with np.load(npz, allow_pickle=False) as z:
            md = json.loads(str(z["metadata"].item()))
            scores = z["v2_probabilities"][0].astype(np.float32)
            roles = z["role_gt_occupancy"][0].astype(np.float32)
            role_keys = md["role_keys"]
            state = int(z["state_indices"][0])
        role_map = {r: i for i, r in enumerate(role_keys)}
        role_union = np.zeros(256, bool)
        for r in ("manipulated_objects", "destination_objects", "task_relevant_fixtures",
                  "robot_arm", "robot_gripper", "robot_mount"):
            if r in role_map: role_union |= roles[role_map[r]] > 0
        fg = {t: np.max(scores, axis=0) >= t for t in (0.10, 0.15, 0.20)}
        # A lower cutoff should only add positives on identical score maps.
        if not (np.all(fg[.20] <= fg[.15]) and np.all(fg[.15] <= fg[.10])):
            raise AssertionError("Threshold monotonicity violated")
        panel_w, panel_h, title_h = 256, 256, 48
        canvas = Image.new("RGB", (panel_w * 5, panel_h + title_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), f"{md['split']} | {task_id} | {md['demo']} state {state}", fill="black")
        draw.text((4, 16), task["instruction"][:130], fill="black")
        labels = ["Task-relevant GT", "Max prompt probability", "V2 p>=0.10", "V2 p>=0.15", "V2 p>=0.20"]
        counts = [int(role_union.sum()), None, int(fg[.10].sum()), int(fg[.15].sum()), int(fg[.20].sum())]
        grids = [role_union.reshape(16,16), np.max(scores, axis=0).reshape(16,16),
                 fg[.10].reshape(16,16), fg[.15].reshape(16,16), fg[.20].reshape(16,16)]
        for col, (label, grid_values, count) in enumerate(zip(labels, grids, counts)):
            x = col * panel_w
            draw.text((x + 4, 20), label if count is None else f"{label} ({count}/256)", fill="black")
            if col == 1:
                values = np.clip((grid_values - 0.0) / 1.0, 0, 1)
                # Compact purple-to-yellow score map, computed only from saved scores.
                tile_array = np.stack((np.clip(55 + values*200,0,255),
                                       np.clip(20 + values*170,0,255),
                                       np.clip(90 - values*75,0,255)), axis=-1).astype(np.uint8)
            else:
                tile_array = np.zeros((16,16,3), dtype=np.uint8)
                tile_array[:] = (42, 45, 52)
                tile_array[np.asarray(grid_values, dtype=bool)] = (35, 190, 95)
            tile = Image.fromarray(tile_array).resize((panel_w, panel_h), Image.NEAREST)
            td = ImageDraw.Draw(tile)
            for n in range(1,16):
                xx = n*16; yy = n*16
                td.line((xx,0,xx,panel_h), fill=(240,240,240), width=1)
                td.line((0,yy,panel_w,yy), fill=(240,240,240), width=1)
            canvas.paste(tile, (x, title_h))
        path = vis / f"{written+1:02d}_{task_id.replace('/', '_')}.png"
        canvas.save(path); written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2-dir", type=Path, default=DEFAULT_E2)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    payload = build_audit(args.e2_dir, args.output_dir)
    n = render_previews(args.e2_dir, args.output_dir)
    payload["visualization_count"] = n
    (args.output_dir / "offline_threshold_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "sequences": payload["sequence_counts"],
                      "frames": payload["frame_counts"], "visualizations": n,
                      "metrics": {t: {s: {"task_relevant_recall": m["role_metrics"]["task_relevant_foreground"]["recall"],
                                                   "candidate_background_coverage": m["foreground"]["mean_candidate_background_coverage"],
                                                   "mean_reusable_tokens": m["foreground"]["mean_reusable_tokens"]}
                                            for s, m in splits.items()}
                                  for t, splits in payload["thresholds"].items()}}, indent=2))


if __name__ == "__main__": main()
