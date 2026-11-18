#!/usr/bin/env python3
"""Post-training threshold/cleaning/PAMR evaluation on the frozen Phase 3 cache.

Only cached 224px DINO patch/attention features are consumed; no DINO or OpenVLA
forward, training, or policy inference is performed. Thresholds use dev only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.kv_cache.experiments.trackb_talk2dino_siglip import phase3_evaluate as phase3
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection

FULL = ROOT / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/full_coco_training_20261117"
DEST = FULL / "postprocessing_optimization"
MANIFEST = FULL / "spatial_evaluation/evaluation_manifest.json"
FEATURES = ROOT / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/pilot_1000/features"
CHECKPOINTS = {
    "epoch_020": FULL / "checkpoints/epoch_020.pt",
    "epoch_050": FULL / "checkpoints/epoch_050.pt",
    "epoch_100": FULL / "checkpoints/epoch_100.pt",
}
METHODS = ("raw", "cleaned", "pamr_raw", "cleaned_pamr")
SCALES = ("grid16", "aligned224")
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
HISTORICAL_THRESHOLD = 0.54
UPSTREAM_THRESHOLD = 0.55
LAMBDA_BG = 0.2
TEMPLATES = phase3.TEMPLATES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(tmp, path)


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def load_checkpoints(device: torch.device):
    models, hashes = {}, {}
    for label, path in CHECKPOINTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        expected_epoch = int(label.rsplit("_", 1)[1])
        if not isinstance(blob, dict) or int(blob.get("completed_epoch", -1)) != expected_epoch:
            raise AssertionError(f"{path} is not the requested epoch {expected_epoch} checkpoint")
        model = RegionTextProjection()
        model.load_state_dict(blob["model"], strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[label], hashes[label] = model, digest
    return models, hashes


def load_class_text(device: torch.device, dataset_root: Path):
    phase3.DATA = dataset_root
    phase3.SIGLIP = Path("/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384")
    coco, _, categories = phase3.read_coco()
    ids = sorted(categories)
    names = [categories[cid]["name"] for cid in ids]
    prompts = [template.format(name) for name in names for template in TEMPLATES]
    tower, tokenizer, metadata = phase3.load_siglip_text_tower(phase3.SIGLIP, device=str(device))
    tokens = tokenizer(prompts).to(device)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(prompts), 64):
            vectors.append(phase3.encode_text_embeddings(tower, tokens[start:start + 64]))
    vectors = torch.cat(vectors).reshape(len(names), len(TEMPLATES), -1).mean(dim=1)
    del tower
    return coco, categories, ids, vectors, metadata


def rgb224_tensor(rgb: Image.Image, device: torch.device):
    width, height = rgb.size
    new_h, new_w = phase3.short_edge_resize_size(height, width, 224)
    resized = rgb.resize((new_w, new_h), Image.Resampling.BICUBIC)
    top, left = phase3.center_crop_offsets(new_h, new_w, 224)
    image = resized.crop((left, top, left + 224, top + 224))
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0).to(device)
    return image, tensor


def make_maps(model, patch, attention, class_text, selected_indices, rgb, pamr):
    patch = F.normalize(patch.float(), dim=-1)
    attention = attention.float() / attention.float().sum(-1, keepdim=True).clamp_min(1e-12)
    text = F.normalize(model.text_projection(class_text.float()).index_select(0, selected_indices), dim=-1)
    cosine = text @ patch.T
    raw = torch.sigmoid(cosine).reshape(-1, 16, 16)
    cleaned, _ = phase3.weighted_background_cleaning(
        raw.unsqueeze(0), patch.T.reshape(1, patch.shape[-1], 16, 16),
        attention.unsqueeze(0), text, LAMBDA_BG)
    cleaned = cleaned[0]
    raw224 = F.interpolate(raw.unsqueeze(0), (224, 224), mode="bilinear", align_corners=True)[0]
    clean224 = F.interpolate(cleaned.unsqueeze(0), (224, 224), mode="bilinear", align_corners=True)[0]
    pamr_raw224 = pamr(rgb, raw224.unsqueeze(0))[0]
    pamr_clean224 = pamr(rgb, clean224.unsqueeze(0))[0]
    out = {
        "raw": {"grid16": raw, "aligned224": raw224},
        "cleaned": {"grid16": cleaned, "aligned224": clean224},
        "pamr_raw": {
            "grid16": F.interpolate(pamr_raw224.unsqueeze(0), (16, 16), mode="nearest")[0],
            "aligned224": pamr_raw224},
        "cleaned_pamr": {
            "grid16": F.interpolate(pamr_clean224.unsqueeze(0), (16, 16), mode="nearest")[0],
            "aligned224": pamr_clean224},
    }
    out = {m: {s: x.detach().float().cpu().numpy() for s, x in pair.items()} for m, pair in out.items()}
    return out, cosine.detach().float().cpu().numpy()


def sweep_iou(scores: np.ndarray, target: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Exact threshold IoU after sorting each score map once."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(target, dtype=bool).ravel()
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    ctp = np.cumsum(labels[order], dtype=np.int64)
    count = np.searchsorted(-sorted_values, -thresholds, side="right")
    tp = np.zeros(len(thresholds), dtype=np.int64)
    valid_count = count > 0
    tp[valid_count] = ctp[count[valid_count] - 1]
    positives = int(labels.sum())
    union = count + positives - tp
    result = np.full(len(thresholds), np.nan, dtype=np.float64)
    valid = union > 0
    result[valid] = tp[valid] / union[valid]
    return result


def calibrate(dev_records, image_by_id, anns_by_image, category_by_id, class_ids,
              image_ids, patches, attentions, models, device, class_text, pamr):
    sums, counts = {}, {}
    class_positions = {cid: i for i, cid in enumerate(class_ids)}
    for ri, rec in enumerate(dev_records, 1):
        iid, row = int(rec["image_id"]), int(rec["feature_row"])
        if int(image_ids[row]) != iid:
            raise AssertionError(f"Feature row/image ID mismatch at {row}")
        meta = image_by_id[iid]
        gt_ids, gt224, gt16, gt_sizes = phase3.make_gt(meta, anns_by_image, category_by_id)
        selected = list(rec["category_ids"])
        if not set(selected).issubset(gt_ids) or any(gt_sizes.get(cid, 0) < 1 for cid in selected):
            raise AssertionError(f"Frozen manifest labels mismatch for image {iid}")
        selector = torch.tensor([class_positions[c] for c in selected], device=device)
        patch = torch.from_numpy(np.array(patches[row], dtype=np.float32, copy=True)).to(device)
        attn = torch.from_numpy(np.array(attentions[row], dtype=np.float32, copy=True)).to(device)
        rgb = Image.open(phase3.DATA / "val2014" / meta["file_name"]).convert("RGB")
        _, rgb_tensor = rgb224_tensor(rgb, device)
        targets = {cid: (gt16[gt_ids.index(cid)].numpy().astype(bool),
                         gt224[gt_ids.index(cid)].numpy().astype(bool)) for cid in selected}
        for checkpoint, model in models.items():
            maps, _ = make_maps(model, patch, attn, class_text, selector, rgb_tensor, pamr)
            for method in METHODS:
                for si, scale in enumerate(SCALES):
                    key = (checkpoint, method, scale)
                    sums.setdefault(key, {})
                    counts.setdefault(key, {})
                    for ci, cid in enumerate(selected):
                        values = sweep_iou(maps[method][scale][ci], targets[cid][si], THRESHOLDS)
                        if cid not in sums[key]:
                            sums[key][cid] = np.zeros_like(THRESHOLDS)
                            counts[key][cid] = 0
                        good = np.isfinite(values)
                        sums[key][cid][good] += values[good]
                        counts[key][cid] += 1
        atomic_json(DEST / "status.json", {"status": "running", "stage": "dev_threshold_calibration",
                                           "development_images_completed": ri, "development_images_total": len(dev_records),
                                           "last_image_id": iid})
        print(f"calibration {ri}/{len(dev_records)} image={iid}", flush=True)
    thresholds, dev_scores = {}, {}
    for (checkpoint, method, scale), class_sums in sums.items():
        class_values = np.stack([class_sums[c] / counts[(checkpoint, method, scale)][c] for c in class_sums])
        macro = np.nanmean(class_values, axis=0)
        best_value = np.nanmax(macro)
        candidates = np.flatnonzero(np.isclose(macro, best_value, rtol=0, atol=1e-12))
        best_idx = min(candidates, key=lambda j: (abs(float(THRESHOLDS[j]) - UPSTREAM_THRESHOLD), float(THRESHOLDS[j])))
        thresholds.setdefault(checkpoint, {}).setdefault(method, {})[scale] = {
            "threshold": float(THRESHOLDS[best_idx]), "development_macro_class_miou": float(macro[best_idx]),
            "candidate_thresholds": THRESHOLDS.tolist(), "category_count": len(class_sums),
            "category_image_pairs": int(sum(counts[(checkpoint, method, scale)].values())),
            "objective": "macro mean over categories of mean per-image IoU",
            "tie_break": "closest to upstream 0.55, then lower threshold"}
        dev_scores.setdefault(checkpoint, {}).setdefault(method, {})[scale] = {
            "calibrated_threshold": float(THRESHOLDS[best_idx]), "macro_class_miou": float(macro[best_idx])}
    return thresholds, dev_scores


def one_metrics(scores, target, threshold, shared=None):
    scores = np.asarray(scores, dtype=np.float32)
    target = np.asarray(target, dtype=bool)
    pred = scores >= float(threshold)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    result = {"iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn}
    if shared is None:
        result["auroc"] = fast_auc(scores, target)
        result["pointing_accuracy"] = phase3.pointing(scores, target)
    else:
        result.update(shared)
    return result


def fast_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware Mann–Whitney AUROC with vectorized average ranks."""
    x = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    npos = int(y.sum()); nneg = len(y) - npos
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    starts = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    average_ranks = (starts + 1 + ends) / 2.0
    sorted_ranks = np.repeat(average_ranks, ends - starts)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = sorted_ranks
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def aggregate_rows(rows):
    if not rows:
        return {}
    def avg(k, subset):
        vals = [float(x[k]) for x in subset if x.get(k) is not None and np.isfinite(x[k])]
        return float(np.mean(vals)) if vals else None
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[str(row["category_id"])].append(row)
    cat_means = [avg("iou", vals) for vals in by_cat.values()]
    sizes = {}
    for group in ("small", "medium", "large"):
        sub = [r for r in rows if r["size_group"] == group]
        if sub:
            sizes[group] = {k: avg(k, sub) for k in ("iou", "auroc", "precision", "recall", "pointing_accuracy")}
            sizes[group]["pairs"] = len(sub)
    return {
        "category_image_pairs": len(rows), "categories_present": len(by_cat),
        "macro_mean_category_iou": float(np.nanmean(cat_means)),
        "mean_pair_iou": avg("iou", rows), "mean_auroc": avg("auroc", rows),
        "mean_precision": avg("precision", rows), "mean_recall": avg("recall", rows),
        "mean_pointing_accuracy": avg("pointing_accuracy", rows), "by_size_group": sizes,
        "per_category": {cid: {"category": vals[0]["category"], "n_images": len(vals),
                               "iou": avg("iou", vals), "auroc": avg("auroc", vals),
                               "precision": avg("precision", vals), "recall": avg("recall", vals),
                               "pointing_accuracy": avg("pointing_accuracy", vals)}
                         for cid, vals in sorted(by_cat.items(), key=lambda p: int(p[0]))},
    }


def render_gallery(image_id, category_id, category_name, image_path, ground_truth,
                   maps_by_checkpoint, calibrated, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = Image.open(image_path).convert("RGB")
    policy_rgb, _ = rgb224_tensor(rgb, torch.device("cpu"))
    gt = np.asarray(ground_truth, dtype=bool)
    columns = ("RGB", "ground truth", "raw", "cleaned", "PAMR raw", "cleaned + PAMR")
    fig, axes = plt.subplots(3, len(columns), figsize=(18, 9), squeeze=False)
    for ri, checkpoint in enumerate(CHECKPOINTS):
        axes[ri, 0].imshow(policy_rgb)
        axes[ri, 0].set_title(f"{checkpoint}\nRGB")
        axes[ri, 1].imshow(policy_rgb)
        axes[ri, 1].imshow(np.ma.masked_where(~gt, gt), cmap="autumn", alpha=0.55, vmin=0, vmax=1)
        axes[ri, 1].set_title("ground truth")
        for ci, method in enumerate(METHODS, start=2):
            item = maps_by_checkpoint[checkpoint][method]
            if method in ("pamr_raw", "cleaned_pamr"):
                score = item["aligned224"]
            else:
                score = F.interpolate(torch.from_numpy(item["grid16"])[None, None], (224, 224),
                                      mode="bilinear", align_corners=True)[0, 0].numpy()
            threshold = calibrated[checkpoint][method]["aligned224"]["threshold"]
            pred = score >= threshold
            axes[ri, ci].imshow(policy_rgb)
            axes[ri, ci].imshow(np.ma.masked_where(~pred, pred), cmap="winter", alpha=0.55, vmin=0, vmax=1)
            axes[ri, ci].set_title(f"{method}\nt={threshold:.2f}")
        for ax in axes[ri]:
            ax.axis("off")
    fig.suptitle(f"COCO val2014 image {image_id}: {category_name}, 224px policy crop", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    global DEST, MANIFEST, FEATURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/opt/openvla/cache/datasets/talk2dino/coco2014"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--feature-root", type=Path, default=FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEST)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-evaluation-images", type=int, default=None, help="bounded smoke check only")
    args = parser.parse_args()
    DEST, MANIFEST, FEATURES = args.output_dir, args.manifest, args.feature_root
    DEST.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA {device} unavailable")
    phase3.DATA = args.dataset_root
    phase3.REFERENCE = Path("/opt/openvla/cache/references/Talk2DINO/src/open_vocabulary_segmentation/models/dinotext")
    phase3.OUT = DEST
    coco, categories, class_ids, class_text, text_metadata = load_class_text(device, args.dataset_root)
    image_by_id = {int(x["id"]): x for x in coco["images"]}
    category_by_id = categories
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
    manifest = json.loads(MANIFEST.read_text())
    records = manifest["images"]
    dev_records = [r for r in records if r["subset"] == "development"]
    eval_records = [r for r in records if r["subset"] == "evaluation"]
    if not dev_records or not eval_records:
        raise AssertionError("Both development and final evaluation subsets must be nonempty")
    dev_image_ids = {int(r["image_id"]) for r in dev_records}
    eval_image_ids = {int(r["image_id"]) for r in eval_records}
    if dev_image_ids & eval_image_ids:
        raise AssertionError("Development and final evaluation image IDs overlap")
    if manifest["selection"].get("selection_used_model_predictions") is not False:
        raise AssertionError("Evaluation manifest was prediction-dependent")
    patch_path = FEATURES / "patch_features_spatial_val2014.npy"
    attention_path = FEATURES / "attention_maps_spatial_val2014.npy"
    spatial_ids_path = FEATURES / "spatial_image_ids_val2014.npy"
    if not patch_path.is_file():
        patch_path = FEATURES / "patch_features_val2014.npy"
        attention_path = FEATURES / "attention_maps_val2014.npy"
        spatial_ids_path = FEATURES / "image_ids_val2014.npy"
    patches = np.load(patch_path, mmap_mode="r")
    attentions = np.load(attention_path, mmap_mode="r")
    image_ids = np.load(spatial_ids_path).astype(np.int64)
    if patches.shape != (len(image_ids), 256, 1024) or attentions.shape != (len(image_ids), 16, 256):
        raise AssertionError(f"Unexpected cache shapes: {patches.shape}, {attentions.shape}, {image_ids.shape}")
    if not np.isfinite(patches).all() or not np.isfinite(attentions).all():
        raise FloatingPointError("DINO patch/attention cache contains nonfinite data")
    for rec in records:
        if int(image_ids[int(rec["feature_row"])]) != int(rec["image_id"]):
            raise AssertionError("Feature cache and Phase 3 manifest are misaligned")
    models, hashes = load_checkpoints(device)
    pamr = phase3.load_pamr(device)
    config = {
        "experiment": "Talk2DINO-SigLIP post-training segmentation postprocessing",
        "checkpoint_sha256": hashes, "checkpoint_epochs": [20, 50, 100],
        "phase3_manifest_sha256": sha256(MANIFEST), "feature_cache_sha256": {
            name: sha256(FEATURES / name) for name in (patch_path.name, attention_path.name, spatial_ids_path.name)},
        "feature_definition": "cached final DINO patches plus CLS-to-patch attention; 224 input, 16x16 grid",
        "methods": list(METHODS), "scales": list(SCALES), "lambda_bg": LAMBDA_BG,
        "pamr": {"iterations": 10, "dilations": [1, 2, 4, 8, 12, 24]},
        "historical_threshold": HISTORICAL_THRESHOLD, "upstream_coco_object_threshold": UPSTREAM_THRESHOLD,
        "threshold_candidates": THRESHOLDS.tolist(), "text_templates": list(TEMPLATES),
        "development_images": len(dev_records), "evaluation_images": len(eval_records),
        "preprocessing_resolution": 224, "no_448px": True, "no_training": True,
        "no_dino_forward": True, "openvla_policy_unchanged": True,
        "siglip_text_metadata": text_metadata, "device": str(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "runtime_platform": platform.platform(),
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["configuration_sha256"] = config_hash
    atomic_json(DEST / "configuration.json", config)
    atomic_json(DEST / "checkpoint_integrity.json", {
        k: {"path": str(CHECKPOINTS[k]), "sha256": hashes[k], "completed_epoch": int(k.rsplit("_", 1)[1]), "strict_load": True}
        for k in CHECKPOINTS})
    validation = phase3.validate_implementation(device)
    if not validation["weighted_cleaning_allclose_atol_1e-7_rtol_1e-6"]:
        raise AssertionError("Upstream weighted-cleaning parity failed")
    atomic_json(DEST / "implementation_validation.json", validation)
    calibration_file = DEST / "threshold_calibration.json"
    cached_calibration = None
    if args.resume and calibration_file.is_file():
        old = json.loads(calibration_file.read_text())
        if old.get("configuration_sha256") == config_hash:
            cached_calibration = old
    if cached_calibration:
        thresholds, dev_scores = cached_calibration["calibrated_thresholds"], cached_calibration["development_scores"]
    else:
        thresholds, dev_scores = calibrate(dev_records, image_by_id, anns_by_image, category_by_id,
            class_ids, image_ids, patches, attentions, models, device, class_text, pamr)
        atomic_json(calibration_file, {"configuration_sha256": config_hash, "calibrated_thresholds": thresholds,
            "development_scores": dev_scores, "split": f"{len(dev_records)} development images only",
            "objective": "macro class mIoU; separate thresholds per checkpoint, postprocess, and reported scale"})
    candidates = []
    for cp in CHECKPOINTS:
        for method in METHODS:
            item = thresholds[cp][method]["grid16"]
            candidates.append({"checkpoint": cp, "method": method, **item})
    best = max(candidates, key=lambda x: x["development_macro_class_miou"])
    atomic_json(DEST / "development_selection.json", {
        "selected_primary_native_grid16": best, "all_conditions": dev_scores,
        "prior_phase3_eval_subset_already_examined": True,
        "selection_rule": "highest dev macro class mIoU across checkpoints/methods, each calibrated on dev only"})
    eval_to_run = eval_records if args.max_evaluation_images is None else eval_records[:args.max_evaluation_images]
    jsonl = DEST / "per_image_metrics.jsonl"
    if not args.resume and jsonl.exists():
        jsonl.unlink()
    completed = set()
    if args.resume and jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                item = json.loads(line)
                if item.get("configuration_sha256") == config_hash:
                    completed.add(int(item["image_id"]))
            except Exception:
                pass
    gallery_ids = {int(r["image_id"]) for r in eval_records[:12]}
    for ri, rec in enumerate(eval_to_run, 1):
        iid, row = int(rec["image_id"]), int(rec["feature_row"])
        if iid in completed:
            continue
        if int(image_ids[row]) != iid:
            raise AssertionError(f"Feature row {row} is not image {iid}")
        meta = image_by_id[iid]
        rgb = Image.open(args.dataset_root / "val2014" / meta["file_name"]).convert("RGB")
        _, rgb_tensor = rgb224_tensor(rgb, device)
        gt_ids, gt224, gt16, gt_sizes = phase3.make_gt(meta, anns_by_image, category_by_id)
        selected = list(rec["category_ids"])
        if not set(selected).issubset(gt_ids) or any(gt_sizes.get(cid, 0) < 1 for cid in selected):
            raise AssertionError(f"Frozen manifest labels mismatch for image {iid}")
        class_positions = {cid: i for i, cid in enumerate(class_ids)}
        selector = torch.tensor([class_positions[cid] for cid in selected], device=device)
        patch = torch.from_numpy(np.array(patches[row], dtype=np.float32, copy=True)).to(device)
        attention = torch.from_numpy(np.array(attentions[row], dtype=np.float32, copy=True)).to(device)
        gt = {cid: (gt16[gt_ids.index(cid)].numpy().astype(bool), gt224[gt_ids.index(cid)].numpy().astype(bool))
              for cid in selected}
        output = {"configuration_sha256": config_hash, "image_id": iid, "feature_row": row, "checkpoint_metrics": {}}
        gallery_maps = {}
        for cp, model in models.items():
            maps, cosine = make_maps(model, patch, attention, class_text, selector, rgb_tensor, pamr)
            cp_item = {"raw_cosine_mean": float(cosine.mean()), "raw_cosine_std": float(cosine.std()), "categories": {}}
            for ci, cid in enumerate(selected):
                positive = int(gt[cid][0].sum())
                size = "small" if positive < 4 else "medium" if positive < 16 else "large"
                category_result = {"category_id": cid, "category": category_by_id[cid]["name"],
                    "positive_grid_cells": positive, "size_group": size, "methods": {}}
                for method in METHODS:
                    category_result["methods"][method] = {}
                    for si, scale in enumerate(SCALES):
                        score = maps[method][scale][ci]
                        target = gt[cid][si]
                        shared_auc = fast_auc(score, target)
                        shared_pointing = phase3.pointing(np.asarray(score), np.asarray(target))
                        tests = {
                            "historical_phase3_0.54": HISTORICAL_THRESHOLD,
                            "upstream_0.55": UPSTREAM_THRESHOLD,
                            "development_calibrated": float(thresholds[cp][method][scale]["threshold"]),
                        }
                        for label, threshold in tests.items():
                            metric = one_metrics(score, target, threshold,
                                {"auroc": shared_auc, "pointing_accuracy": shared_pointing})
                            category_result["methods"][method][f"{scale}__{label}"] = metric
                        category_result["methods"][method][f"{scale}__calibrated_threshold"] = tests["development_calibrated"]
                    if iid in gallery_ids:
                        gallery_maps.setdefault(cp, {})[method] = {
                            "grid16": maps[method]["grid16"][ci].astype(np.float16),
                            "aligned224": maps[method]["aligned224"][ci].astype(np.float16)}
                cp_item["categories"][str(cid)] = category_result
            output["checkpoint_metrics"][cp] = cp_item
        output["category_ids"] = selected
        output["category_names"] = [category_by_id[c]["name"] for c in selected]
        with jsonl.open("a") as stream:
            stream.write(json.dumps(output, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        completed.add(iid)
        if iid in gallery_ids and selected:
            chosen_cat = max(selected, key=lambda c: int(gt[c][0].sum()))
            render_gallery(iid, chosen_cat, category_by_id[chosen_cat]["name"],
                args.dataset_root / "val2014" / meta["file_name"], gt[chosen_cat][1],
                gallery_maps, thresholds, DEST / "qualitative_results" / f"image_{iid}_category_{chosen_cat}.png")
        atomic_json(DEST / "status.json", {"status": "running", "stage": "evaluation",
            "evaluation_images_completed": len(completed), "evaluation_images_total": len(eval_to_run),
            "last_image_id": iid, "device": str(device), "elapsed_seconds": time.perf_counter() - started})
        print(f"evaluation {len(completed)}/{len(eval_to_run)} image={iid}", flush=True)
    jsonl_items = {}
    for line in jsonl.read_text().splitlines():
        item = json.loads(line)
        if item.get("configuration_sha256") == config_hash:
            jsonl_items[int(item["image_id"])] = item
    expected_ids = {int(x["image_id"]) for x in eval_to_run}
    if not expected_ids.issubset(jsonl_items):
        raise RuntimeError("Evaluation records are incomplete")
    rows = []
    for iid in sorted(expected_ids):
        item = jsonl_items[iid]
        for cp, cp_item in item["checkpoint_metrics"].items():
            for cid_text, cat in cp_item["categories"].items():
                for method in METHODS:
                    for scale in SCALES:
                        for label in ("historical_phase3_0.54", "upstream_0.55", "development_calibrated"):
                            result = cat["methods"][method][f"{scale}__{label}"]
                            threshold = (cat["methods"][method][f"{scale}__calibrated_threshold"]
                                         if label == "development_calibrated" else HISTORICAL_THRESHOLD if label == "historical_phase3_0.54" else UPSTREAM_THRESHOLD)
                            rows.append({"image_id": iid, "checkpoint": cp, "epoch": int(cp.rsplit("_", 1)[1]),
                                "method": method, "scale": scale, "threshold_policy": label, "threshold": threshold,
                                "category_id": int(cid_text), "category": cat["category"],
                                "positive_grid_cells": cat["positive_grid_cells"], "size_group": cat["size_group"], **result})
    aggregate = {}
    for cp in CHECKPOINTS:
        for method in METHODS:
            for scale in SCALES:
                for label in ("historical_phase3_0.54", "upstream_0.55", "development_calibrated"):
                    subset = [r for r in rows if r["checkpoint"] == cp and r["method"] == method
                              and r["scale"] == scale and r["threshold_policy"] == label]
                    aggregate.setdefault(cp, {}).setdefault(method, {}).setdefault(scale, {})[label] = aggregate_rows(subset)
    with (DEST / "per_category_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    selected_eval = aggregate[best["checkpoint"]][best["method"]]["grid16"]["development_calibrated"]
    summary = {
        "status": "complete" if args.max_evaluation_images is None else "smoke_test_partial",
        "development_selection": best,
        "evaluation_aggregate": aggregate,
        "selected_configuration_evaluation_grid16": selected_eval,
        "historical_full_coco_epoch100_cleaned_grid16_miou_0_54": 0.30588934948829305,
        "evaluation_images": len(expected_ids), "development_images": len(dev_records),
        "category_image_pairs": len({(r["image_id"], r["category_id"]) for r in rows}),
        "evaluation_subset_note": "Segmentation development and final evaluation IDs are disjoint and were selected without model predictions. All IDs from the previously examined Phase 3 manifest were excluded from both new segmentation subsets. Retrieval uses the full val2014 split partitioned by image ID into disjoint development and evaluation candidates.",
        "runtime": {"seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
            "cached_vision_forward_rerun": False, "training": False},
    }
    atomic_json(DEST / "summary.json", summary)
    atomic_json(DEST / "status.json", {"status": summary["status"], "stage": "complete",
        "evaluation_images_processed": len(expected_ids), "development_images": len(dev_records),
        "finished_unix_time": time.time()})
    print(f"complete: {DEST / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
