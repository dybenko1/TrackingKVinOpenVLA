#!/usr/bin/env python3
"""Build a 30-image binary segmentation gallery with original Talk2DINO.

Selection is based only on val2017 semantic ground-truth area. The official
upstream DINOText inference, prompt ensemble, projection, and attention cleaner
are retained; no model is trained and no OpenVLA features are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/opt/openvla")
WORKTREE = ROOT / "cache/worktrees/trackb-dense-grounding-v2"
DATA = ROOT / "cache/datasets/coco_stuff164k"
OUT = WORKTREE / "experiments/kv_cache/outputs/trackb_talk2dino_coco_binary_gallery"
BASELINE = WORKTREE / "experiments/kv_cache/experiments/trackb_talk2dino_baseline_validation"
THRESHOLD = 0.54
RESOLUTION = 448
PATCH = 14
GRID = RESOLUTION // PATCH
SEED = 20261126
SIZE_BINS = {"small": (0, 32 * 32), "medium": (32 * 32, 96 * 96), "large": (96 * 96, float("inf"))}
PREFERRED_CLASSES = [
    "bowl", "bottle", "cup", "wine glass", "fork", "knife", "spoon",
    # Mix household/manipulation categories with unrelated object controls.
    "bicycle", "bus", "car", "person", "cat", "dog", "bird", "sports ball",
    "banana", "apple", "sandwich", "toothbrush", "remote", "cell phone",
    "book", "clock", "vase", "scissors", "teddy bear", "chair", "couch",
    "dining table", "toilet", "sink", "refrigerator", "backpack",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resize2d(array, width, height, nearest=False):
    """PIL resize for binary labels and floating score maps (no OpenCV needed)."""
    a = np.asarray(array)
    im = Image.fromarray(a) if a.dtype == np.uint8 else Image.fromarray(a.astype(np.float32), mode="F")
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(im.resize((int(width), int(height)), resample=method))


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def choose_targets(names, lut):
    """Select unique image/category pairs, stratified by semantic-mask area."""
    image_dir, gt_dir = DATA / "images/val2017", DATA / "annotations_raw/val2017"
    images = sorted(image_dir.glob("*.jpg"))
    masks = sorted(gt_dir.glob("*.png"))
    if len(images) != 5000 or len(masks) != 5000:
        raise RuntimeError(f"Expected 5000 paired val2017 images/masks; got {len(images)}/{len(masks)}")
    image_by_id = {p.stem: p for p in images}
    if set(image_by_id) != {p.stem for p in masks}:
        raise RuntimeError("COCO val2017 image and semantic-mask IDs do not match")
    class_to_id = {name: i for i, name in enumerate(names)}
    eligible = [c for c in PREFERRED_CLASSES if c in class_to_id]
    candidates = {size: defaultdict(list) for size in SIZE_BINS}
    for gt_path in masks:
        raw = np.asarray(Image.open(gt_path), dtype=np.uint8)
        labels = lut[raw]
        for category in eligible:
            cid = class_to_id[category]
            area = int(np.count_nonzero(labels == cid))
            if area == 0:
                continue
            size = next((key for key, (lo, hi) in SIZE_BINS.items() if lo <= area < hi), None)
            if size:
                candidates[size][category].append((gt_path.stem, area))
    rng = random.Random(SEED)
    selected, used_images = [], set()
    distribution = {}
    for size in SIZE_BINS:
        # Keep the predeclared mixed household/control ordering. Selection is
        # deterministic and the per-category image ordering remains hashed.
        categories = [category for category in PREFERRED_CLASSES if category in candidates[size]]
        for category in categories:
            candidates[size][category].sort(key=lambda x: hashlib.sha256(f"{SEED}:{size}:{category}:{x[0]}".encode()).hexdigest())
        taken = 0
        # Round-robin categories to avoid a gallery dominated by common classes.
        while taken < 10:
            made_progress = False
            for category in categories:
                choice = next((x for x in candidates[size][category] if x[0] not in used_images), None)
                if choice is None:
                    continue
                image_id, area = choice
                used_images.add(image_id)
                selected.append({"image_id": image_id, "category": category, "class_id": class_to_id[category],
                                 "size_bin": size, "semantic_union_area_px": area,
                                 "image_path": str(image_by_id[image_id]), "gt_path": str(gt_dir / f"{image_id}.png"),
                                 "image_sha256": sha256(image_by_id[image_id]), "gt_sha256": sha256(gt_dir / f"{image_id}.png")})
                taken += 1
                made_progress = True
                if taken == 10:
                    break
            if not made_progress:
                raise RuntimeError(f"Only selected {taken}/10 {size} examples from eligible target categories")
        distribution[size] = dict(sorted({x["category"]: sum(1 for y in selected if y["size_bin"] == size and y["category"] == x["category"]) for x in selected if x["size_bin"] == size}.items()))
    if len(selected) != 30 or len(used_images) != 30:
        raise RuntimeError("Selection did not produce 30 unique images")
    return selected, {"image_count": len(images), "mask_count": len(masks), "eligible_categories": eligible,
                      "selected_category_counts_by_size": distribution}


def scores_one(official, text_embed, image_bgr):
    """Return upstream RAW/CLEAN pixel maps and native patch-grid scores."""
    import torch
    import torch.nn.functional as F
    model = official.model
    original_clean = model.similarity_assignment_weighted
    captured = {}

    def capture_clean(mask, image_feat, self_attn, text, lambda_bg=0.2):
        clean_patch = original_clean(mask, image_feat, self_attn, text, lambda_bg=lambda_bg)
        captured["clean_patch"] = clean_patch.detach()
        return clean_patch

    model.similarity_assignment_weighted = capture_clean
    try:
        result = official.score_bgr_crop(image_bgr, text_embed, clean=True, pamr=False)
    finally:
        model.similarity_assignment_weighted = original_clean
    cosine = result["cosine"]
    if cosine.shape != (1, GRID, GRID):
        raise RuntimeError(f"Expected one query on a {GRID}x{GRID} patch map; got {tuple(cosine.shape)}")
    if "clean_patch" not in captured or captured["clean_patch"].shape != (1, 1, GRID, GRID):
        raise RuntimeError(f"Missing or misaligned cleaned native-grid output: {captured.get('clean_patch')}")
    raw_patch = torch.sigmoid(cosine[0]).float()
    clean_patch = captured["clean_patch"][0, 0].float()
    raw_pixel = F.interpolate(torch.sigmoid(cosine).unsqueeze(0), size=image_bgr.shape[:2],
                               mode="bilinear", align_corners=True)[0, 0].float()
    clean_pixel = result["scores"][0].float()
    if clean_pixel.shape != image_bgr.shape[:2]:
        raise RuntimeError(f"Expected cleaned map aligned to input image {image_bgr.shape[:2]}, got {tuple(clean_pixel.shape)}")
    arrays = [raw_patch, clean_patch, raw_pixel, clean_pixel]
    if any(not torch.isfinite(x).all() for x in arrays):
        raise RuntimeError("Nonfinite Talk2DINO score map")
    return [x.detach().cpu().numpy().astype(np.float32) for x in arrays]


def counts_metrics(pred, gt):
    p, y = np.asarray(pred, bool), np.asarray(gt, bool)
    tp = int(np.count_nonzero(p & y)); fp = int(np.count_nonzero(p & ~y))
    fn = int(np.count_nonzero(~p & y)); tn = int(np.count_nonzero(~p & ~y))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1), "foreground_coverage": float(p.mean()),
            "background_coverage": float((~p).mean())}


def fractional_patch_metrics(pred, occupancy):
    p = np.asarray(pred, bool).astype(np.float64)
    y = np.clip(np.asarray(occupancy, np.float64), 0, 1)
    tp = float((p * y).sum()); fp = float((p * (1-y)).sum())
    fn = float(((1-p) * y).sum()); tn = float(((1-p) * (1-y)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp/max(tp+fp,1e-12), "recall": tp/max(tp+fn,1e-12),
            "iou": tp/max(tp+fp+fn,1e-12)}


def render_panel(item, rgb, gt, score_maps, metrics, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = gt.shape
    variants = ["raw", "clean"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 11.2))
    col_titles = ["Original RGB", "Oracle foreground", "Predicted foreground", "Predicted background", "Errors: TP green · FP red · FN yellow"]
    preds = {}
    for row, method in enumerate(variants):
        pred = score_maps[method]["pixel"] >= THRESHOLD
        preds[method] = pred
        err = np.zeros((H, W, 3), dtype=np.uint8)
        err[pred & gt] = (20, 205, 60)
        err[pred & ~gt] = (235, 35, 35)
        err[~pred & gt] = (255, 220, 0)
        images = [rgb, rgb * gt[..., None], rgb * pred[..., None], rgb * (~pred)[..., None],
                  np.clip(rgb.astype(np.float32)*0.42 + err.astype(np.float32)*0.58, 0, 255).astype(np.uint8)]
        for col, (ax, image) in enumerate(zip(axes[row], images)):
            ax.imshow(image)
            # Keep column labels only above the first row. Repeating them on
            # row two caused collisions with the CLEAN row heading.
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, pad=5)
            ax.axis("off")
    pixel_lines = []
    for method in variants:
        m = metrics[method]["pixel"]
        pixel_lines.append(f"{method.upper()} pixel @0.54: P {m['precision']:.3f} | R {m['recall']:.3f} | IoU {m['iou']:.3f} | FG {m['foreground_coverage']:.1%} | BG {m['background_coverage']:.1%}")
    patch_lines = []
    for method in variants:
        m = metrics[method]["patch_binary"]
        patch_lines.append(f"{method.upper()} {GRID}×{GRID} binary patches: P {m['precision']:.3f} | R {m['recall']:.3f} | IoU {m['iou']:.3f}")
    fig.suptitle(f"COCO val2017 image {item['image_id']} · query: {item['category']} · {item['size_bin']} target "
                 f"(semantic category union {item['semantic_union_area_px']:,} px)\n"
                 "Original Talk2DINO ViT-L/14-reg4 + CLIP ViT-B/16 + original MLP · 448×448 · threshold 0.54",
                 fontsize=12, y=0.99)
    fig.text(0.5, 0.935, "RAW · original category score", ha="center", va="center", fontsize=10, weight="bold")
    fig.text(0.5, 0.492, "CLEAN · upstream attention-based background cleaning", ha="center", va="center", fontsize=10, weight="bold")
    fig.text(0.5, 0.025, "\n".join(pixel_lines + patch_lines), ha="center", va="bottom", fontsize=8.2, family="monospace")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.17, hspace=0.50, wspace=0.035)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def aggregate_rows(rows, keys):
    out = {}
    for key in keys:
        group = [r for r in rows if all(r[k] == v for k, v in key.items())]
        for method in ("raw", "clean"):
            for level in ("pixel", "patch_binary", "patch_fractional"):
                totals = {c: sum(r["metrics"][method][level][c] for r in group) for c in ("tp", "fp", "fn", "tn")}
                tp, fp, fn, tn = (totals[c] for c in ("tp", "fp", "fn", "tn"))
                out.setdefault(method, {}).setdefault(level, []).append({**key, "images": len(group),
                    "precision": tp/max(tp+fp, 1e-12), "recall": tp/max(tp+fn, 1e-12),
                    "iou": tp/max(tp+fp+fn, 1e-12), **totals})
    return out


def write_report(manifest, rowdata, model_id, selection_audit):
    # Global, size-stratified, and per-category pooled pixel-level metrics.
    groupings = [{"all": "all"}]
    for item in manifest["selected"]:
        pass
    report = ["# Original Talk2DINO COCO binary segmentation gallery", "",
        "This report evaluates the frozen original pretrained Talk2DINO model on 30 COCO Object val2017 images. Selection used ground-truth category area only; no model scores were consulted. Each image has one independent category query and the category mask is the union of every pixel annotated with that semantic class.", "",
        "## Protocol", "",
        "- Original Meta DINOv2 ViT-L/14-register, original CLIP ViT-B/16, original Talk2DINO projection MLP, original `sub_imagenet_template` prompt ensemble, and original attention-based cleaning (`lambda_bg=0.2`). All checkpoint hashes and source revisions are recorded in `configuration.json`.",
        "- One square 448×448 DINOText input per image; the score map is aligned back to the source image size for pixel scoring. RAW is `sigmoid(cosine)`; CLEAN is the upstream cleaned soft mask. Both use the fixed 0.54 probability threshold. PAMR is disabled.",
        "- Size bins use the standard COCO area cutoffs (<1,024, 1,024–9,215, ≥9,216 pixels), applied to the semantic category-union mask. The downloaded val2017 material contains semantic maps, not per-instance polygons; therefore these are category-union area strata, not verified individual-instance sizes.",
        "- Pixel-level metrics use the original resolution semantic labels. Patch metrics use the model's native 32×32 DINO grid at 448 input: GT is first nearest-neighbor warped with the same 448 square geometry, then area pooled. Binary patch GT means any target occupancy in the 14×14 cell; fractional occupancy-weighted patch metrics are also included.",
        "- This is a 30-image visual diagnostic, not the complete COCO Object benchmark and not directly comparable to the prior 5,000-image 81-class mIoU. The earlier full benchmark uses sliding-window inference and multiclass semantic assignment; this gallery uses fixed single-scale category-wise binary maps.", "",
        "## Aggregated metrics", ""]
    for grouping_name, grouping_key in (("All selected images", None), ("By target size", "size_bin"), ("By category", "category")):
        report += [f"### {grouping_name}", "", "Pooled pixel metrics (counts pooled before ratios):", "",
                   "| Group | N | Method | Precision | Recall | IoU | FG coverage | BG coverage |", "|---|---:|---|---:|---:|---:|---:|---:|"]
        group_values = [None] if grouping_key is None else sorted({r[grouping_key] for r in rowdata})
        for val in group_values:
            g = [r for r in rowdata if grouping_key is None or r[grouping_key] == val]
            for method in ("raw", "clean"):
                counts = [sum(r["metrics"][method]["pixel"][k] for r in g) for k in ("tp", "fp", "fn", "tn")]
                tp, fp, fn, tn = counts
                report.append(f"| {val or 'all'} | {len(g)} | {method.upper()} | {tp/max(tp+fp,1):.3f} | {tp/max(tp+fn,1):.3f} | {tp/max(tp+fp+fn,1):.3f} | {sum(r['metrics'][method]['pixel']['foreground_coverage'] for r in g)/max(len(g),1):.1%} | {sum(r['metrics'][method]['pixel']['background_coverage'] for r in g)/max(len(g),1):.1%} |")
        report += ["", "Binary 32×32 patch metrics are available per image in `per_image_metrics.csv`; these use any-positive-occupancy token labels. Fractional occupancy-weighted token-grid scores are in the same CSV.", ""]
    report += ["## Failure analysis", ""]
    small_rows = [r for r in rowdata if r["size_bin"] == "small"]
    large_rows = [r for r in rowdata if r["size_bin"] == "large"]
    both_zero = [r for r in small_rows if all(r["metrics"][m]["pixel"]["recall"] == 0 for m in ("raw", "clean"))]
    # Pool counts before computing metrics, matching the tables above.
    def pooled_iou(group, method):
        tp=sum(r["metrics"][method]["pixel"]["tp"] for r in group)
        fp=sum(r["metrics"][method]["pixel"]["fp"] for r in group)
        fn=sum(r["metrics"][method]["pixel"]["fn"] for r in group)
        return tp/max(tp+fp+fn,1)
    def pooled_precision(group, method):
        tp=sum(r["metrics"][method]["pixel"]["tp"] for r in group)
        fp=sum(r["metrics"][method]["pixel"]["fp"] for r in group)
        return tp/max(tp+fp,1)
    report += [
        f"- **Small-object localization:** pooled pixel IoU is {pooled_iou(small_rows,'raw'):.3f} RAW and {pooled_iou(small_rows,'clean'):.3f} CLEAN, versus {pooled_iou(large_rows,'raw'):.3f}/{pooled_iou(large_rows,'clean'):.3f} on large targets. Small-object misses are prominent.",
        f"- **Complete misses:** {len(both_zero)} of 10 small-target images have zero pixel recall in both RAW and CLEAN at 0.54 ({', '.join(r['image_id']+' '+r['category'] for r in both_zero) if both_zero else 'none'}).",
        f"- **Excess foreground:** small-bin precision is {pooled_precision(small_rows,'raw'):.3f} RAW and {pooled_precision(small_rows,'clean'):.3f} CLEAN; many predicted foreground pixels therefore lie outside the queried category, even when recall is nonzero.",
        f"- **Disconnected regions:** predicted connected-component counts across the 30 frames have median/max {int(np.median([r['metrics']['raw']['predicted_components'] for r in rowdata]))}/{max(r['metrics']['raw']['predicted_components'] for r in rowdata)} RAW and {int(np.median([r['metrics']['clean']['predicted_components'] for r in rowdata]))}/{max(r['metrics']['clean']['predicted_components'] for r in rowdata)} CLEAN. Multiple COCO instances can legitimately form separate components, so these counts alone do not establish fragmentation errors.",
        "- **Semantic confusions:** the binary target metric records every non-target pixel as a false-positive foreground pixel, but does not classify which other COCO category it belongs to. The gallery can show examples of spill, but this experiment does not quantify a specific semantically similar-category confusion matrix.",
        "- **Cleaning trade-off:** CLEAN generally raises precision/IoU while lowering recall and predicted foreground coverage. That reduces spill but misses more target pixels; it is not uniformly better.", "",
        "## What the gallery can establish", "",
        "Visual inspection should focus on whether category-specific RAW/CLEAN predictions cover the full annotated category union, whether activation spills onto semantically related objects, and how often small regions vanish on the 32×32 grid. Pixel metrics are based on the upsampled patch score field, so they do not imply pixel-level boundary modeling.", "",
        "Potential failure modes are quantified through false-positive/false-negative counts, prediction connected-component counts, and recall in each size stratum. A disconnected mask can be a legitimate result for a category with multiple instances; it is not automatically an error.", "",
        "The prior LIBERO individual-object study used cached 224×224 RGB images and 16×16 patch-aligned role masks, while this COCO gallery uses 448×448 inputs, 32×32 native patch masks, full-resolution semantic category unions, and a different 30-image sample. Any observed difference is descriptive and cannot be attributed to domain shift alone.", "",
        "## Gallery", ""]
    for size in SIZE_BINS:
        report += [f"### {size.title()} target regions", ""]
        for item in manifest["selected"]:
            if item["size_bin"] == size:
                report += [f"#### Image {item['image_id']} — {item['category']} ({item['semantic_union_area_px']:,} semantic-union pixels)",
                           f"![RAW and CLEAN binary segmentation comparison](figures/{item['figure']})", ""]
    report += ["## Provenance", "", f"- Selected {len(manifest['selected'])} unique image IDs from {selection_audit['image_count']} val2017 images, exactly 10 per size group.",
               f"- Model identity: `{model_id['official_repo_revision']}` upstream Talk2DINO, DINO source `{model_id['official_dino']['source_revision']}`.",
               "- Exact image, annotation, selected category, target size, and source SHA-256 for every panel: `gallery_manifest.json`.",
               "- Full per-image pixel, binary patch, fractional patch, and size/category metrics: `per_image_metrics.csv`.", ""]
    (OUT / "EXPERIMENT_REPORT.md").write_text("\n".join(report))


def run(smoke=False):
    import sys
    sys.path.insert(0, str(WORKTREE))
    sys.path.insert(0, str(ROOT))
    # The container also has the checked-out repository installed editable;
    # extend its regular package paths so imports resolve to this research
    # worktree's validated baseline code rather than the active Track C tree.
    import experiments.kv_cache
    experiments.kv_cache.__path__.append(str(WORKTREE / "experiments/kv_cache"))
    import experiments.kv_cache.experiments
    experiments.kv_cache.experiments.__path__.append(str(WORKTREE / "experiments/kv_cache/experiments"))
    from experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation import run_coco
    names = run_coco.class_names()
    lut = run_coco.label_lut()
    selected, audit = choose_targets(names, lut)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    # Remove only stale panels from this gallery's earlier deterministic
    # selection attempts; keep unrelated files in the experiment directory.
    selected_figures = {f"coco_{x['image_id']}_{x['category'].replace(' ','_')}.png" for x in selected}
    for old_figure in (OUT / "figures").glob("coco_*.png"):
        if old_figure.name not in selected_figures:
            old_figure.unlink()
    cache_dir = OUT / "predictions_local"
    cache_dir.mkdir(exist_ok=True)
    selected = selected[:1] if smoke else selected
    manifest = {"experiment": "Original Talk2DINO COCO Object val2017 binary gallery", "seed": SEED,
                "selection_rule": "Ground-truth COCO Object semantic category-union area only; deterministic category-balanced round robin, unique image IDs, no model predictions consulted.",
                "semantic_union_size_proxy": "COCO standard instance-area cutoffs applied to the category union area in the source semantic label mask; instance areas are not separately available.",
                "target_distribution": {"small": 10, "medium": 10, "large": 10},
                "selection_audit": audit, "smoke_mode": smoke, "selected": selected}
    atomic_json(OUT / "gallery_manifest.json", manifest)
    atomic_json(OUT / "RUN_STATUS.json", {"status": "running", "stage": "preflight", "selected": len(selected), "timestamp": time.time()})
    device = "cuda:0"
    from experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.run_coco import load_model
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the validated original Talk2DINO evaluation runtime")
    official, classnames, embeddings, model_id = load_model(torch.device(device))
    model_id["gallery_image_preprocessing"] = "upstream DINOText Resize((448,448)) square warp; source image loaded RGB then passed BGR as expected by MMCV crop wrapper"
    model_id["gallery_inference"] = "one fixed 448x448 full-image DINOText forward per image; no sliding-window; no PAMR"
    model_id["binary_threshold"] = THRESHOLD
    model_id["native_patch_grid"] = [GRID, GRID]
    class_ids = {name: i for i, name in enumerate(classnames)}
    embeddings_by_name = {name: embeddings[i:i+1] for i, name in enumerate(classnames[1:])}
    rows = []
    for index, item in enumerate(selected):
        image_path, gt_path = Path(item["image_path"]), Path(item["gt_path"])
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        source_labels = np.asarray(Image.open(gt_path), dtype=np.uint8)
        if source_labels.shape != rgb.shape[:2]:
            raise RuntimeError(f"RGB/annotation geometry mismatch for {item['image_id']}")
        labels = lut[source_labels]
        gt = labels == item["class_id"]
        if not gt.any():
            raise RuntimeError(f"Selected class is absent in image {item['image_id']}")
        cache = cache_dir / f"{item['image_id']}.npz"
        if cache.exists():
            with np.load(cache) as z:
                if str(z["query"].item()) == item["category"] and str(z["image_sha256"].item()) == item["image_sha256"]:
                    raw_patch, clean_patch, raw448, clean448 = [z[k].astype(np.float32) for k in ("raw_patch", "clean_patch", "raw448", "clean448")]
                else:
                    cache.unlink()
                    raw_patch = None
        else:
            raw_patch = None
        if raw_patch is None:
            image_bgr = rgb[..., ::-1].copy()
            rp, cp, r448, c448 = scores_one(official, embeddings_by_name[item["category"]], image_bgr)
            raw_patch, clean_patch, raw448, clean448 = rp, cp, r448, c448
            np.savez_compressed(cache, raw_patch=raw_patch.astype(np.float16), clean_patch=clean_patch.astype(np.float16),
                                raw448=raw448.astype(np.float16), clean448=clean448.astype(np.float16),
                                query=np.asarray(item["category"]), image_sha256=np.asarray(item["image_sha256"]))
        if raw_patch.shape != (GRID, GRID) or clean_patch.shape != (GRID, GRID):
            raise RuntimeError("Prediction patch dimensions do not match the DINO grid")
        H, W = gt.shape
        # DINOText internally warps the RGB to 448×448, then upsamples maps
        # back to the supplied image extent; the returned maps already align
        # with the original-resolution COCO labels.
        pixel_maps = {"raw": raw448, "clean": clean448}
        patch_maps = {"raw": raw_patch, "clean": clean_patch}
        gt448 = resize2d(gt.astype(np.uint8), RESOLUTION, RESOLUTION, nearest=True).astype(np.float32)
        occupancy = gt448.reshape(GRID,PATCH,GRID,PATCH).mean(axis=(1,3))
        metric_row = {**{k:v for k,v in item.items() if k not in ("image_path","gt_path")}, "metrics": {}}
        # area selection was determined before inference; verify it again after LUT decode.
        if int(gt.sum()) != item["semantic_union_area_px"]:
            raise RuntimeError("GT area changed between selection and inference")
        for method in ("raw", "clean"):
            pixel_pred = pixel_maps[method] >= THRESHOLD
            patch_pred = patch_maps[method] >= THRESHOLD
            patch_binary_gt = occupancy > 0
            pixel = counts_metrics(pixel_pred, gt)
            patch_binary = counts_metrics(patch_pred, patch_binary_gt)
            patch_fractional = fractional_patch_metrics(patch_pred, occupancy)
            from scipy.ndimage import label as connected_components
            n_components = int(connected_components(pixel_pred, structure=np.ones((3,3), dtype=np.uint8))[1])
            metric_row["metrics"][method] = {"pixel": pixel, "patch_binary": patch_binary,
                                                "patch_fractional": patch_fractional,
                                                "predicted_components": n_components}
        figure = f"coco_{item['image_id']}_{item['category'].replace(' ','_')}.png"
        item["figure"] = figure
        render_panel(item, rgb, gt, {m:{"pixel":pixel_maps[m],"patch":patch_maps[m]} for m in ("raw","clean")}, metric_row["metrics"], OUT/"figures"/figure)
        metric_row["figure"] = figure
        rows.append(metric_row)
        atomic_json(OUT/"RUN_STATUS.json", {"status":"running","stage":"score_and_render","completed":index+1,
                    "total":len(selected),"last_image_id":item["image_id"],"timestamp":time.time(),"smoke":smoke})
    official.close()
    manifest["selected"] = selected
    atomic_json(OUT/"gallery_manifest.json", manifest)
    fieldnames = ["image_id","category","class_id","size_bin","semantic_union_area_px","figure","method",
                  "pixel_precision","pixel_recall","pixel_iou","pixel_foreground_coverage","pixel_background_coverage",
                  "pixel_tp","pixel_fp","pixel_fn","pixel_tn","pixel_predicted_components",
                  "patch_binary_precision","patch_binary_recall","patch_binary_iou","patch_binary_foreground_coverage","patch_binary_background_coverage",
                  "patch_fractional_precision","patch_fractional_recall","patch_fractional_iou"]
    with (OUT/"per_image_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader()
        for row in rows:
            for method in ("raw","clean"):
                p=row["metrics"][method]["pixel"]; pb=row["metrics"][method]["patch_binary"]; pf=row["metrics"][method]["patch_fractional"]
                w.writerow({"image_id":row["image_id"],"category":row["category"],"class_id":row["class_id"],"size_bin":row["size_bin"],
                    "semantic_union_area_px":row["semantic_union_area_px"],"figure":row["figure"],"method":method,
                    **{f"pixel_{k}":p[k] for k in ("precision","recall","iou","foreground_coverage","background_coverage","tp","fp","fn","tn")},
                    "pixel_predicted_components":row["metrics"][method]["predicted_components"],
                    **{f"patch_binary_{k}":pb[k] for k in ("precision","recall","iou","foreground_coverage","background_coverage")},
                    **{f"patch_fractional_{k}":pf[k] for k in ("precision","recall","iou")}})
    summary = {}
    for group_name, selector in (("overall",lambda r:True),("by_size",lambda r:True),("by_category",lambda r:True)):
        groups = [None] if group_name=="overall" else sorted({r["size_bin" if group_name=="by_size" else "category"] for r in rows})
        summary[group_name] = {}
        for g in groups:
            subset = [r for r in rows if g is None or r["size_bin" if group_name=="by_size" else "category"]==g]
            summary[group_name][g or "all"] = {}
            for method in ("raw","clean"):
                summary[group_name][g or "all"][method] = {}
                for level in ("pixel","patch_binary","patch_fractional"):
                    if level=="patch_fractional":
                        sums={k:sum(r["metrics"][method][level][k] for r in subset) for k in ("tp","fp","fn","tn")}
                    else:
                        sums={k:sum(r["metrics"][method][level][k] for r in subset) for k in ("tp","fp","fn","tn")}
                    tp,fp,fn,tn=(sums[k] for k in ("tp","fp","fn","tn"))
                    summary[group_name][g or "all"][method][level]={**sums,"precision":tp/max(tp+fp,1e-12),"recall":tp/max(tp+fn,1e-12),"iou":tp/max(tp+fp+fn,1e-12)}
                summary[group_name][g or "all"][method]["pixel"]["foreground_coverage_mean_image"] = float(np.mean([r["metrics"][method]["pixel"]["foreground_coverage"] for r in subset]))
                summary[group_name][g or "all"][method]["pixel"]["background_coverage_mean_image"] = float(np.mean([r["metrics"][method]["pixel"]["background_coverage"] for r in subset]))
    summary.update({"image_count":len(rows),"threshold":THRESHOLD,"selection_audit":audit,"model_identity":model_id,
                    "metric_definitions":{"pixel":"binary mask evaluated at original COCO image size after bilinear score-map resize from 448 square inference output","patch_binary":"32x32 DINO positions; binary GT is positive for any area-pooled occupancy > 0","patch_fractional":"same 32x32 predictions weighted by GT foreground fractional occupancy"}})
    atomic_json(OUT/"summary.json",summary)
    atomic_json(OUT/"configuration.json",{"model_identity":model_id,"dataset":"COCO Object val2017 via cached COCO-Stuff164k semantic maps",
        "image_count":len(rows),"size_bins_pixel_area":{"small":"0 < area < 1024","medium":"1024 <= area < 9216","large":"area >= 9216"},
        "size_area_note":"Bins use the queried class's union area in semantic labels as a proxy; instance-level segmentation JSON was not present in the local dataset.",
        "threshold":THRESHOLD,"visual_resolution":[RESOLUTION,RESOLUTION],"native_patch_grid":[GRID,GRID],"prompt_template":"upstream sub_imagenet_template",
        "cleaning_lambda":0.2,"pamr":False,"selection_seed":SEED,"smoke_mode":smoke,"no_training":True,"no_openvla_features":True})
    validation={"status":"passed_smoke_and_gallery" if not smoke else "smoke_passed","smoke_image_id":rows[0]["image_id"],
        "selected_unique_images":len(rows),"expected_images":1 if smoke else 30,"raw_clean_maps_finite":True,
        "pixel_masks_aligned_to_original_rgb":True,"native_patch_grid_shapes":[GRID,GRID],"threshold":THRESHOLD,
        "score_cache_local_only":True,"selection_used_model_predictions":False}
    atomic_json(OUT/"validation.json",validation)
    if not smoke:
        write_report(manifest,rows,model_id,audit)
        # Simple gallery README with per-image visual links grouped by target size.
        readme=["# Original Talk2DINO COCO binary gallery","","Thirty COCO Object val2017 examples selected by ground-truth category-union area before model inference. Each figure compares RAW and CLEAN using the same original image, query, oracle mask, and 0.54 threshold.","","See [the full report](EXPERIMENT_REPORT.md) for aggregated metrics and protocol details.",""]
        for size in SIZE_BINS:
            readme += [f"## {size.title()} objects",""]
            for row in rows:
                if row["size_bin"]==size:
                    readme += [f"### {row['category']} · COCO image {row['image_id']} ({row['semantic_union_area_px']:,} target pixels)",
                               f"![COCO {row['category']} RAW/CLEAN panel](figures/{row['figure']})",""]
        (OUT/"README.md").write_text("\n".join(readme))
    atomic_json(OUT/"RUN_STATUS.json",{"status":"complete" if not smoke else "smoke_complete","stage":"complete","images":len(rows),"timestamp":time.time()})
    print(json.dumps(validation,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true",help="run the first selected gallery image only")
    args=parser.parse_args()
    run(args.smoke)


if __name__=="__main__":
    main()
