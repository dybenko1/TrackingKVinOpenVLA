#!/usr/bin/env python3
"""Render binary RGB mask panels from cached Talk2DINO/E2 LIBERO arrays.

No model inference is performed. Masks and predictions live on E2's native
16x16 visual-token grid; visual displays use nearest-neighbour expansion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE = Path("/opt/openvla/cache/datasets/talk2dino_native_e2/scoring")
OUT = Path(
    "/opt/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/"
    "trackb_talk2dino_baseline_validation/libero_final/binary_visual_gallery"
)
THRESHOLD = 0.54
ROLE_TO_GT = {
    "manipulated_object": ("manipulated_objects",),
    "destination_or_fixture": ("destination_objects", "task_relevant_fixtures"),
    "task_fixture": ("task_relevant_fixtures",),
    "robot_arm": ("robot_arm",),
    "robot_gripper": ("robot_gripper",),
}
DISPLAY_ROLES = ("manipulated_object", "destination_or_fixture", "task_fixture")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(source: Path):
    records = []
    for path in sorted(source.glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as z:
            rgb = z["rgb"].copy()
            roles = z["role_gt"].astype(np.float32)
            valid = z["valid"].astype(np.float32)
            scores = z["a_clean"].astype(np.float32)  # official standalone Meta-DINO condition
            meta = json.loads(str(z["metadata"].item()))
        if rgb.ndim != 4 or rgb.shape[1:] != (224, 224, 3):
            raise ValueError(f"Unexpected RGB shape in {path}: {rgb.shape}")
        if roles.shape[0] != len(rgb) or scores.shape != (len(rgb), len(meta["prompt_specs"]), 256):
            raise ValueError(f"RGB/GT/prediction alignment error in {path}")
        if roles.shape[-1] != 256 or valid.shape != (len(rgb), 256):
            raise ValueError(f"Unexpected patch layout in {path}")
        records.append({"path": path, "rgb": rgb, "roles": roles, "valid": valid,
                        "scores": scores, "meta": meta})
    return records


def gt_for(record, role: str, frame: int) -> np.ndarray:
    keys = record["meta"]["role_keys"]
    channels = ROLE_TO_GT[role]
    y = np.zeros(256, dtype=np.float32)
    for name in channels:
        if name in keys:
            y = np.maximum(y, record["roles"][frame, keys.index(name)])
    return y


def select_examples(records):
    """Select median positive GT area per task/role, independent of predictions."""
    tasks = {}
    for r in records:
        tasks.setdefault(r["meta"]["task_id"], []).append(r)
    selected = []
    for task_id in sorted(tasks):
        group = tasks[task_id]
        # Favor object queries; task fixture is used only where no object query exists.
        available_roles = []
        for role in DISPLAY_ROLES:
            if any(s["role"] == role for r in group for s in r["meta"]["prompt_specs"]):
                available_roles.append(role)
        if "manipulated_object" not in available_roles and "destination_or_fixture" not in available_roles:
            available_roles = [r for r in available_roles if r == "task_fixture"]
        for role in available_roles:
            candidates = []
            for r in group:
                spec_ids = [i for i, s in enumerate(r["meta"]["prompt_specs"]) if s["role"] == role]
                if not spec_ids:
                    continue
                qi = spec_ids[0]
                for frame in range(len(r["rgb"])):
                    y = gt_for(r, role, frame)
                    valid = r["valid"][frame] > 0
                    area = float((y * valid).sum())
                    if area > 0:
                        candidates.append((area, r, frame, qi, y, valid))
            if not candidates:
                continue
            candidates.sort(key=lambda x: (x[0], x[1]["meta"]["demo"], x[3], x[2]))
            area, r, frame, qi, y, valid = candidates[(len(candidates) - 1) // 2]
            selected.append({"task_id": task_id, "role": role, "phrase": r["meta"]["prompt_specs"][qi]["phrase"],
                             "instruction": r["meta"]["instruction"], "record": r,
                             "frame": frame, "query_index": qi, "gt_fraction": y,
                             "gt": (y > 0) & valid, "valid": valid, "gt_area": area})
    return selected


def render(item, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = item["record"]
    frame, qi = item["frame"], item["query_index"]
    rgb = r["rgb"][frame]
    gt = item["gt"].reshape(16, 16)
    valid = item["valid"].reshape(16, 16)
    prob = r["scores"][frame, qi].reshape(16, 16)
    pred = (prob >= THRESHOLD) & valid
    bg = (~pred) & valid

    tp = pred & gt
    fp = pred & (~gt) & valid
    fn = (~pred) & gt
    tn = (~pred) & (~gt) & valid
    precision = tp.sum() / max(pred.sum(), 1)
    recall = tp.sum() / max(gt.sum(), 1)
    iou = tp.sum() / max((tp | fp | fn).sum(), 1)
    fg_coverage = pred.sum() / max(valid.sum(), 1)
    bg_coverage = bg.sum() / max(valid.sum(), 1)

    # Binary token masks are expanded as 14x14 blocks to preserve the exact
    # 16x16 token alignment; they are not pixel-boundary segmentation masks.
    gt_px = np.repeat(np.repeat(gt, 14, axis=0), 14, axis=1)
    pred_px = np.repeat(np.repeat(pred, 14, axis=0), 14, axis=1)
    bg_px = np.repeat(np.repeat(bg, 14, axis=0), 14, axis=1)
    err = np.zeros((224, 224, 3), np.float32)
    err[np.repeat(np.repeat(tp, 14, 0), 14, 1)] = (0.0, 0.8, 0.0)  # correct FG
    err[np.repeat(np.repeat(fp, 14, 0), 14, 1)] = (0.95, 0.12, 0.08)  # false FG
    err[np.repeat(np.repeat(fn, 14, 0), 14, 1)] = (1.0, 0.85, 0.0)  # missed FG

    panels = [
        ("Original RGB", rgb, "rgb"),
        ("Oracle foreground\n(black elsewhere)", rgb * gt_px[..., None], "rgb"),
        (f"Predicted foreground\n{item['phrase']}", rgb * pred_px[..., None], "rgb"),
        ("Predicted background\n(black elsewhere)", rgb * bg_px[..., None], "rgb"),
        ("Errors over RGB\nTP green · FP red · FN yellow", np.clip(rgb * 0.45 + err * 255 * 0.55, 0, 255).astype(np.uint8), "rgb"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(19, 5.25), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.05]})
    for ax, (title, image, _) in zip(axes, panels):
        ax.imshow(image.astype(np.uint8))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    metrics = (
        f"Task: {item['task_id']}    Demo: {r['meta']['demo']}    Frame: {frame} / {len(r['rgb'])-1}\n"
        f"Query: {item['phrase']}    |    Official standalone Talk2DINO (Meta DINO), cleaned score threshold={THRESHOLD:.2f}\n"
        f"Binary 16×16 token metrics (GT occupancy > 0; invalid tokens excluded):  "
        f"Precision {precision:.3f}   Recall {recall:.3f}   IoU {iou:.3f}   "
        f"Predicted FG {pred.sum()}/{valid.sum()} ({fg_coverage:.1%})   BG {bg.sum()}/{valid.sum()} ({bg_coverage:.1%})\n"
        f"TP {tp.sum()} · FP {fp.sum()} · FN {fn.sum()} · TN {tn.sum()}    "
        "Masks are 16×16 patches enlarged with nearest-neighbor blocks; they are not full-resolution object contours."
    )
    fig.suptitle(item["instruction"], fontsize=12, y=0.99)
    fig.text(0.5, 0.015, metrics, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    key = item["task_id"].replace("/", "__") + "__" + item["role"]
    path = output / f"{key}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"file": path.name, "task_id": item["task_id"], "suite": r["meta"]["suite"],
            "instruction": item["instruction"], "role": item["role"], "query": item["phrase"],
            "demo": r["meta"]["demo"], "source_frame_index": frame, "source_array": str(r["path"]),
            "source_array_sha256": sha256(r["path"]), "threshold": THRESHOLD,
            "ground_truth_binary_rule": "fractional occupancy > 0 on valid E2 16x16 token grid",
            "prediction_source": "a_clean: official standalone Talk2DINO, official Meta DINO, attention-cleaned score",
            "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()), "tn": int(tn.sum()),
            "precision": float(precision), "recall": float(recall), "iou": float(iou),
            "predicted_foreground_tokens": int(pred.sum()), "predicted_background_tokens": int(bg.sum()),
            "valid_tokens": int(valid.sum()), "foreground_coverage": float(fg_coverage),
            "background_coverage": float(bg_coverage)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    records = load_records(args.source)
    selected = select_examples(records)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = [render(x, args.output) for x in selected]
    manifest_doc = {
        "experiment": "Original Talk2DINO baseline validation — individual-object binary gallery",
        "model_condition": "official standalone Talk2DINO with Meta DINOv2",
        "selection": "one frame per available task/query; median positive ground-truth patch area, independent of predictions",
        "input_archive_count": len(records), "unique_task_count": len({r['meta']['task_id'] for r in records}),
        "figure_count": len(manifest), "threshold": THRESHOLD,
        "metric_grid": "16x16 E2 visual-token grid; binary oracle is fractional occupancy > 0; prediction is cached cleaned score >= 0.54; invalid patches excluded",
        "visual_resolution_note": "RGB is original 224x224; binary masks expand each 16x16 token to 14x14 pixels using nearest-neighbor replication",
        "examples": manifest}
    (args.output / "gallery_manifest.json").write_text(json.dumps(manifest_doc, indent=2) + "\n")
    lines = [
        "# Talk2DINO LIBERO binary-mask gallery",
        "",
        "Individual-query visualizations for all 12 tasks represented in the completed LIBERO scoring cache. "
        "These are saved RGB observations and cached official standalone Talk2DINO predictions; no inference or rollout was run to make this gallery.",
        "",
        "## How to read the panels",
        "",
        "Each panel shows RGB, oracle foreground, predicted foreground, predicted background, and a color-coded error overlay. "
        "Green is a correctly selected foreground patch (TP), red is a false-positive foreground patch (FP), and yellow is missed foreground (FN). "
        "Metrics use binary 16×16 patches: oracle occupancy > 0, prediction is the cached attention-cleaned score ≥ 0.54, and invalid tokens are excluded. "
        "Each patch is enlarged to a 14×14 pixel block on the 224×224 RGB image; this is not pixel-boundary segmentation.",
        "",
        "Frames were selected per task and query by median positive oracle patch area, without inspecting model scores or predictions. "
        "The selected queries are manipulated objects, destination/fixtures, or the task fixture when no object query is available.",
        "",
        "## Examples",
        "",
    ]
    for e in manifest:
        lines += [
            f"### {e['suite']} — `{e['task_id'].split('/', 1)[1]}`",
            "",
            f"**Query:** {e['query']} ({e['role']}) · **Demo/frame:** {e['demo']} / {e['source_frame_index']} · "
            f"**P/R/IoU:** {e['precision']:.3f} / {e['recall']:.3f} / {e['iou']:.3f} · "
            f"**FG/BG coverage:** {e['foreground_coverage']:.1%} / {e['background_coverage']:.1%}",
            "",
            f"![{e['suite']} {e['role']} query: {e['query']}]({e['file']})",
            "",
        ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"source_archives": len(records), "unique_tasks": len({r['meta']['task_id'] for r in records}),
                      "figures": len(manifest), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
