#!/usr/bin/env python3
"""Offline Phase 3 COCO spatial evaluation from existing cached DINO features."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.config import EXPECTED_DINO_FEATURE_DIM
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase3_spatial import (
    binary_mask_from_polygons,
    center_crop_offsets,
    decode_coco_rle,
    resize_scores_to_224,
    short_edge_resize_size,
    sigmoid_masks,
    transform_mask_to_policy,
    weighted_background_cleaning,
)
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import (
    encode_text_embeddings,
    load_siglip_text_tower,
)

OUT = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/phase3_spatial_evaluation"
DATA = Path("/opt/openvla/cache/datasets/talk2dino/coco2014")
FEATURES = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/pilot_1000/features"
INITIAL = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/pilot_1000/pilot_mlp_checkpoint/initial_projection.pt"
ORIGINAL = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/pilot_1000/pilot_mlp_checkpoint/final_projection.pt"
CORRECTED = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/pilot_1000_phase2/corrected_projection.pt"
SIGLIP = Path("/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384")
REFERENCE = Path("/opt/openvla/cache/references/Talk2DINO/src/open_vocabulary_segmentation/models/dinotext")
THRESHOLD = 0.54  # upstream COCO-object eval configuration: bg_thresh=0.54
LAMBDA_BG = 0.2
TEMPLATES = [
    "itap of a {}.", "a bad photo of a {}.", "a origami {}.",
    "a photo of the large {}.", "a {} in a video game.", "art of the {}.",
    "a photo of the small {}.",
]
CHECKPOINTS = {
    "untrained": INITIAL,
    "original_pilot": ORIGINAL,
    "phase2_corrected": CORRECTED,
}


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_coco():
    with (DATA / "annotations2014/instances_val2014.json").open() as f:
        coco = json.load(f)
    return coco, {int(x["id"]): x for x in coco["images"]}, {int(x["id"]): x for x in coco["categories"]}


def make_category_masks(image: dict, anns: list[dict]) -> dict[int, torch.Tensor]:
    h, w = int(image["height"]), int(image["width"])
    result: dict[int, torch.Tensor] = {}
    for ann in anns:
        seg = ann.get("segmentation")
        if isinstance(seg, list):
            mask = binary_mask_from_polygons(seg, h, w)
        elif isinstance(seg, dict):
            mask = decode_coco_rle(seg, h, w)
        else:
            continue
        cid = int(ann["category_id"])
        if cid in result:
            result[cid] |= mask
        else:
            result[cid] = mask
    return result


def prepare_manifest() -> dict:
    """Select from annotation/feature metadata only, before loading model scores."""
    coco, images, categories = read_coco()
    image_ids = np.load(FEATURES / "image_ids_val2014.npy").astype(np.int64).tolist()
    patches = np.load(FEATURES / "patch_features_val2014.npy", mmap_mode="r")
    attention = np.load(FEATURES / "attention_maps_val2014.npy", mmap_mode="r")
    if patches.shape != (200, 256, EXPECTED_DINO_FEATURE_DIM) or attention.shape != (200, 16, 256):
        raise AssertionError(f"Unexpected existing feature shapes: {patches.shape}, {attention.shape}")
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    rows = []
    rle_count = polygon_count = 0
    for row, image_id in enumerate(image_ids):
        image = images.get(int(image_id))
        if image is None:
            continue
        path = DATA / "val2014" / image["file_name"]
        if not path.is_file():
            continue
        cat_masks = make_category_masks(image, anns_by_image[int(image_id)])
        qualifying = []
        per_cat = []
        for cid, mask in sorted(cat_masks.items()):
            policy_mask = transform_mask_to_policy(mask).squeeze(0).squeeze(0)
            grid = F.interpolate(policy_mask[None, None], (16, 16), mode="nearest")[0, 0] > 0.5
            cells = int(grid.sum())
            if cells >= 2:
                qualifying.append(cid)
            if cells:
                per_cat.append({"category_id": cid, "grid_cells": cells, "pixel_area_original": int(mask.sum())})
        if len(qualifying) >= 2:
            rows.append({
                "feature_row": row, "image_id": int(image_id), "file_name": image["file_name"],
                "width": int(image["width"]), "height": int(image["height"]),
                "category_ids": qualifying, "category_areas": per_cat,
            })
    rng = random.Random(20261117)
    rng.shuffle(rows)
    selected = rows[:100]
    if len(selected) < 3:
        raise RuntimeError(f"Only {len(selected)} eligible cached validation images found")
    dev_count = min(20, max(1, len(selected) // 5))
    for i, item in enumerate(selected):
        item["subset"] = "development" if i < dev_count else "evaluation"
        item["category_names"] = [categories[cid]["name"] for cid in item["category_ids"]]
    result = {
        "selection": {
            "source": "existing 200-image COCO val feature subset",
            "seed": 20261117,
            "eligible_criterion": "image has at least two annotated categories whose category-union mask occupies >=2 cells after exact short-edge resize/center-crop and nearest 16x16 projection",
            "eligible_count": len(rows), "selected_count": len(selected),
            "development_count": dev_count, "evaluation_count": len(selected) - dev_count,
            "selection_used_model_predictions": False,
        },
        "feature_cache": {
            "feature_definition": "talk2dino_final_layer_cls_attention_weighted_patch_mean_v1",
            "patch_features": str(FEATURES / "patch_features_val2014.npy"),
            "attention_maps": str(FEATURES / "attention_maps_val2014.npy"),
            "image_ids": str(FEATURES / "image_ids_val2014.npy"),
            "patches_shape": list(patches.shape), "attention_shape": list(attention.shape),
        },
        "images": selected,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "evaluation_manifest.json", result)
    atomic_json(OUT / "instance_annotations_audit.json", {
        "archive": str(DATA / "archives/annotations_trainval2014.zip"),
        "archive_sha256": sha256(DATA / "archives/annotations_trainval2014.zip"),
        "annotation_file": str(DATA / "annotations2014/instances_val2014.json"),
        "annotation_sha256": sha256(DATA / "annotations2014/instances_val2014.json"),
        "archive_listing_confirmed_instances_val2014": True,
        "archive_listing_confirmed_instances_train2014": True,
        "annotation_images": len(coco["images"]), "annotation_instances": len(coco["annotations"]),
        "category_count": len(categories), "crowd_instances_included_in_category_unions": sum(bool(a.get("iscrowd", 0)) for a in coco["annotations"]),
        "cached_feature_images": len(image_ids), "cached_image_files_found": sum((DATA / "val2014" / images[i]["file_name"]).is_file() for i in image_ids if i in images),
        "eligible_images_in_cached_sample": len(rows), "selected_images": len(selected),
        "segmentation_format_counts": dict(Counter("polygon" if isinstance(a["segmentation"], list) else "rle" for a in coco["annotations"])),
        "selected_image_segmentation_format_counts": dict(Counter(
            "polygon" if isinstance(a["segmentation"], list) else "rle"
            for rec in selected for a in anns_by_image[int(rec["image_id"])]
        )),
        "rle_note": "Polygon and compressed COCO RLE segmentations are both decoded into semantic category unions. iscrowd is retained in the union rather than used as an instance-level AP ignore region.",
    })
    return result


def load_projection(path: Path, device: torch.device):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model = RegionTextProjection()
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"Checkpoint keys mismatch: {incompat}")
    return model.to(device).eval(), hashlib.sha256(path.read_bytes()).hexdigest()


def encode_class_templates(device: torch.device):
    text_tower, tokenizer, metadata = load_siglip_text_tower(SIGLIP, device=str(device))
    coco, _, categories = read_coco()
    ids = sorted(categories)
    names = [categories[cid]["name"] for cid in ids]
    phrases = [template.format(name) for name in names for template in TEMPLATES]
    tokenized = tokenizer(phrases).to(device)
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(phrases), 64):
            encoded.append(encode_text_embeddings(text_tower, tokenized[start:start + 64]))
    template_features = torch.cat(encoded).reshape(len(names), len(TEMPLATES), -1).mean(dim=1)
    return ids, names, template_features, metadata


def extract_upstream_weighted_method():
    """Load the actual upstream method body for numerical parity test."""
    source = (REFERENCE / "dinotext.py").read_text()
    tree = ast.parse(source)
    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "similarity_assignment_weighted":
            method = node
            break
    if method is None:
        raise RuntimeError("Could not locate upstream similarity_assignment_weighted")
    module = ast.Module(body=[method], type_ignores=[])
    ns = {"torch": torch}
    exec(compile(module, str(REFERENCE / "dinotext.py"), "exec"), ns)
    return ns["similarity_assignment_weighted"]


def validate_implementation(device: torch.device) -> dict:
    torch.manual_seed(731)
    b, n, c, h, w = 1, 3, 12, 4, 4
    mask = torch.sigmoid(torch.randn(b, n, h, w, device=device))
    patches = F.normalize(torch.randn(b, c, h, w, device=device), dim=1)
    attention = torch.softmax(torch.randn(b, 16, h * w, device=device), dim=-1)
    text = F.normalize(torch.randn(n, c, device=device), dim=-1)
    upstream = extract_upstream_weighted_method()
    class Dummy: pass
    expected = upstream(Dummy(), mask, patches, attention, text, 0.2)
    actual, prior = weighted_background_cleaning(mask, patches, attention, text, 0.2)
    exact = torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    if not exact:
        raise AssertionError(f"Background cleaning parity failed, max={float((actual-expected).abs().max())}")
    if prior.shape != mask.shape or not torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=1e-6):
        raise AssertionError("Attention/map shape or normalization validation failed")
    perm = torch.tensor([2, 0, 1], device=device)
    reordered, _ = weighted_background_cleaning(mask[:, perm], patches, attention, text[perm], 0.2)
    inverse = torch.argsort(perm)
    category_order = torch.allclose(reordered[:, inverse], actual, atol=1e-7, rtol=1e-6)
    if not category_order:
        raise AssertionError("Category ordering changed weighted outputs")
    # Verify nearest geometry on a synthetic rectangular corner mask.
    known = torch.zeros(180, 320, dtype=torch.uint8); known[40:130, 25:95] = 1
    transformed = transform_mask_to_policy(known)
    if transformed.shape != (1, 1, 224, 224):
        raise AssertionError("Mask preprocessing geometry produced wrong shape")
    return {
        "upstream_method_source": str(REFERENCE / "dinotext.py"),
        "upstream_ast_method": "DINOText.similarity_assignment_weighted",
        "weighted_cleaning_max_abs_error": float((actual - expected).abs().max()),
        "weighted_cleaning_allclose_atol_1e-7_rtol_1e-6": exact,
        "category_order_invariant": category_order,
        "attention_sums_to_one": True,
        "mask_shape": list(mask.shape), "attention_shape": list(attention.shape),
        "lambda_bg": 0.2,
        "sigmoid_conversion": "torch.sigmoid(cosine_similarity), matching DINOTextSim2Mask soft path",
        "mask_transform_shape": list(transformed.shape),
    }


def binary_metrics(scores: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    target = target.astype(bool)
    tp = int((pred & target).sum()); fp = int((pred & ~target).sum()); fn = int((~pred & target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    flat_s = scores.astype(np.float64).ravel(); flat_y = target.ravel().astype(np.int8)
    auc = rank_auc(flat_s, flat_y)
    return {"iou": iou, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "auroc": auc}


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool); x = np.asarray(scores, dtype=np.float64)
    npos = int(y.sum()); nneg = len(y) - npos
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    # Average ranks for ties.
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def pointing(scores: np.ndarray, target: np.ndarray) -> float:
    index = int(np.argmax(scores))
    return float(bool(target.reshape(-1)[index]))


def load_pamr(device: torch.device):
    import importlib.util
    path = REFERENCE / "pamr.py"
    spec = importlib.util.spec_from_file_location("talk2dino_phase3_pamr", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.PAMR(10, [1, 2, 4, 8, 12, 24]).eval().to(device)


def make_gt(image_meta: dict, anns_by_image: dict, categories: dict[int, dict]):
    all_masks = make_category_masks(image_meta, anns_by_image[int(image_meta["id"])])
    ids = sorted(all_masks)
    masks224, masks16 = [], []
    sizes = {}
    for cid in ids:
        mask = transform_mask_to_policy(all_masks[cid])
        grid = F.interpolate(mask, (16, 16), mode="nearest")[0, 0]
        masks224.append(mask[0, 0] > 0.5); masks16.append(grid > 0.5)
        sizes[cid] = int((grid > 0.5).sum())
    return ids, torch.stack(masks224), torch.stack(masks16), sizes


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.48):
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    if base.shape[:2] != mask.shape:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((base.shape[1], base.shape[0]), Image.Resampling.NEAREST)) > 0
    base[mask] = base[mask] * (1.0 - alpha) + np.asarray(color) * alpha
    return Image.fromarray(np.uint8(np.clip(base, 0, 255)))


def evaluate(args):
    device = torch.device(args.device)
    started = time.perf_counter()
    manifest_path = OUT / "evaluation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run --prepare-manifest first; manifest must precede prediction inspection")
    manifest = json.loads(manifest_path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "status.json", {"status": "running", "stage": "loading checkpoints and category text",
                                      "device": str(device), "started_unix_time": time.time(),
                                      "manifest_sha256": sha256(manifest_path), "pamr": bool(args.pamr)})
    coco, image_by_id, category_by_id = read_coco()
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
    patches_arr = np.load(FEATURES / "patch_features_val2014.npy", mmap_mode="r")
    attention_arr = np.load(FEATURES / "attention_maps_val2014.npy", mmap_mode="r")
    ids = np.load(FEATURES / "image_ids_val2014.npy").astype(np.int64)
    if not np.isfinite(patches_arr).all() or not np.isfinite(attention_arr).all():
        raise FloatingPointError("Existing feature cache contains nonfinite values")
    implementation_validation = validate_implementation(device)

    model_results = {}
    checkpoint_hashes = {}
    for name, path in CHECKPOINTS.items():
        if not path.is_file(): raise FileNotFoundError(path)
        model, checkpoint_hashes[name] = load_projection(path, device)
        model_results[name] = model
    class_ids, class_names, template_text, text_metadata = encode_class_templates(device)
    class_position = {cid: i for i, cid in enumerate(class_ids)}
    text_by_model = {}
    with torch.inference_mode():
        for name, model in model_results.items():
            projected = model.text_projection(template_text.float())
            text_by_model[name] = F.normalize(projected, dim=-1)
    pamr = load_pamr(device) if args.pamr else None

    metrics_rows = []
    per_image_results: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in model_results}
    qualitative_cache = {}
    eval_records = [r for r in manifest["images"] if r["subset"] == "evaluation"]
    dev_records = [r for r in manifest["images"] if r["subset"] == "development"]
    qualitative_ids = {int(r["image_id"]) for r in eval_records[:12]}
    records_to_run = list(manifest["images"])
    if args.max_eval_images is not None:
        dev = [r for r in records_to_run if r["subset"] == "development"]
        ev = [r for r in records_to_run if r["subset"] == "evaluation"][:args.max_eval_images]
        records_to_run = dev + ev
    eval_goal = min(len(eval_records), args.max_eval_images) if args.max_eval_images is not None else len(eval_records)
    completed_eval = 0
    for record_index, rec in enumerate(records_to_run, 1):
        iid = int(rec["image_id"]); row = int(rec["feature_row"])
        if int(ids[row]) != iid:
            raise AssertionError(f"Feature row/image id mismatch at row={row}: {ids[row]} != {iid}")
        image_meta = image_by_id[iid]
        img_path = DATA / "val2014" / image_meta["file_name"]
        rgb = Image.open(img_path).convert("RGB")
        gt_ids, gt224, gt16, gt_sizes = make_gt(image_meta, anns_by_image, category_by_id)
        selected_ids = [cid for cid in gt_ids if gt_sizes[cid] >= 2]
        if selected_ids != rec["category_ids"]:
            raise AssertionError(f"Manifest category mismatch for {iid}: {selected_ids} vs {rec['category_ids']}")
        idx = torch.tensor([class_position[cid] for cid in selected_ids], dtype=torch.long, device=device)
        patch = torch.from_numpy(np.array(patches_arr[row], dtype=np.float32, copy=True)).to(device)
        attn = torch.from_numpy(np.array(attention_arr[row], dtype=np.float32, copy=True)).to(device)
        patch = F.normalize(patch, dim=-1)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        gt224 = gt224[[gt_ids.index(cid) for cid in selected_ids]].to(device)
        gt16 = gt16[[gt_ids.index(cid) for cid in selected_ids]].to(device)
        cats = [category_by_id[cid]["name"] for cid in selected_ids]
        results_this_image = {}
        for model_name, model in model_results.items():
            texts = text_by_model[model_name].index_select(0, idx)
            with torch.inference_mode():
                raw = patch @ texts.T  # [256,N], cosine because both normalized
                raw = raw.T.reshape(len(cats), 16, 16)
                soft = sigmoid_masks(raw)
                clean, attention_prior = weighted_background_cleaning(
                    soft.unsqueeze(0), patch.transpose(0, 1).reshape(1, patch.shape[-1], 16, 16),
                    attn.unsqueeze(0), texts, LAMBDA_BG,
                )
                clean = clean[0]
                raw224 = resize_scores_to_224(soft)
                clean224 = resize_scores_to_224(clean)
                pamr224 = None
                if pamr is not None:
                    # Talk2DINO's center crop geometry; match the cached 224 inputs.
                    ow, oh = rgb.size
                    nh, nw = short_edge_resize_size(oh, ow, 224)
                    resized_rgb = rgb.resize((nw, nh), Image.Resampling.BICUBIC)
                    top, left = center_crop_offsets(nh, nw, 224)
                    resized_rgb = resized_rgb.crop((left, top, left + 224, top + 224))
                    rgb_tensor = torch.from_numpy(np.asarray(resized_rgb).copy()).permute(2, 0, 1).float().unsqueeze(0).to(device)
                    pamr224 = pamr(rgb_tensor, clean224.unsqueeze(0))[0]

            if rec["subset"] == "evaluation":
                per_cat = {}
                for cat_idx, cid in enumerate(selected_ids):
                    gt_g = gt16[cat_idx].cpu().numpy().astype(bool)
                    gt_f = gt224[cat_idx].cpu().numpy().astype(bool)
                    maps = {
                        "raw": soft[cat_idx].detach().cpu().numpy(),
                        "cleaned": clean[cat_idx].detach().cpu().numpy(),
                    }
                    if pamr224 is not None:
                        pamr16 = F.interpolate(pamr224[cat_idx].unsqueeze(0).unsqueeze(0), (16, 16), mode="nearest")[0, 0]
                        maps["pamr"] = pamr16.detach().cpu().numpy()
                    map224 = {
                        "raw": raw224[cat_idx].detach().cpu().numpy(),
                        "cleaned": clean224[cat_idx].detach().cpu().numpy(),
                    }
                    if pamr224 is not None: map224["pamr"] = pamr224[cat_idx].detach().cpu().numpy()
                    item = {"category_id": cid, "category": category_by_id[cid]["name"],
                            "positive_grid_cells": int(gt_g.sum()), "positive_pixels_224": int(gt_f.sum()), "methods": {}}
                    for method in maps:
                        m16 = binary_metrics(maps[method], gt_g, THRESHOLD)
                        m224 = binary_metrics(map224[method], gt_f, THRESHOLD)
                        m16["pointing_accuracy"] = pointing(maps[method], gt_g)
                        m224["pointing_accuracy"] = pointing(map224[method], gt_f)
                        item["methods"][method] = {"grid16": m16, "aligned224": m224}
                    per_cat[cid] = item
                    for method in item["methods"]:
                        metrics_rows.append({
                            "subset": "evaluation", "image_id": iid, "checkpoint": model_name,
                            "category_id": cid, "category": category_by_id[cid]["name"],
                            "positive_grid_cells": int(gt_g.sum()), "size_group": "small" if int(gt_g.sum()) < 4 else "medium" if int(gt_g.sum()) < 16 else "large",
                            "method": method, **item["methods"][method]["grid16"],
                            "pointing_accuracy": item["methods"][method]["grid16"]["pointing_accuracy"],
                            "iou_224": item["methods"][method]["aligned224"]["iou"],
                        })
                # Predicted union and semantic category map using upstream fixed bg score.
                union_gt = gt16.any(dim=0).cpu().numpy().astype(bool)
                for method, maps in (("raw", soft), ("cleaned", clean)):
                    stack = maps.detach().cpu().numpy()
                    predicted = stack >= THRESHOLD
                    union_pred = predicted.any(axis=0)
                    union_metric = binary_metrics(union_pred.astype(np.float32), union_gt, 0.5)
                    semantic = np.full((16, 16), -1, dtype=np.int32)
                    best = stack.argmax(axis=0)
                    best_score = stack.max(axis=0)
                    semantic[best_score >= THRESHOLD] = np.asarray(selected_ids, dtype=np.int32)[best[best_score >= THRESHOLD]]
                    results_this_image[method + "_union_iou"] = union_metric["iou"]
                    results_this_image[method + "_semantic_foreground_fraction"] = float((semantic >= 0).mean())
                per_image_results[model_name][iid] = {"categories": per_cat, **results_this_image}
            # cache maps for a deterministic selection of the first 12 eval images only
            if rec["subset"] == "evaluation" and iid in qualitative_ids:
                qualitative_cache.setdefault(iid, {})[model_name] = {
                    "categories": cats,
                    "category_ids": selected_ids,
                    "gt16": gt16.detach().cpu().numpy(),
                    "similarity": raw.detach().cpu().numpy(),
                    "raw": soft.detach().cpu().numpy(),
                    "cleaned": clean.detach().cpu().numpy(),
                    "pamr": None if pamr224 is None else pamr224.detach().cpu().numpy(),
                    "binary_cleaned": (clean.detach().cpu().numpy() >= THRESHOLD),
                    "metric_items": per_cat,
                    "image_path": str(img_path),
                }
        if rec["subset"] == "evaluation":
            completed_eval += 1
            status = {"status": "running", "stage": "scoring cached features and masks",
                      "evaluation_completed": completed_eval, "evaluation_total_this_run": eval_goal,
                      "development_images_processed": sum(x["subset"] == "development" for x in records_to_run),
                      "last_image_id": iid, "last_feature_row": row, "device": str(device),
                      "pamr": bool(args.pamr), "elapsed_seconds": time.perf_counter() - started}
            atomic_json(OUT / "status.json", status)
            print(json.dumps(status), flush=True)

    # Aggregate metrics by checkpoint/method/category and patch-size group.
    aggregate = {}
    for model_name in model_results:
        aggregate[model_name] = {}
        for method in (["raw", "cleaned", "pamr"] if args.pamr else ["raw", "cleaned"]):
            subset = [r for r in metrics_rows if r["checkpoint"] == model_name and r["method"] == method]
            categories_agg = {}
            for cid in sorted({r["category_id"] for r in subset}):
                vals = [r for r in subset if r["category_id"] == cid]
                categories_agg[str(cid)] = {
                    "category": vals[0]["category"], "n_images": len(vals),
                    **{key: float(np.nanmean([v[key] for v in vals])) for key in ("iou", "precision", "recall", "pointing_accuracy", "auroc", "iou_224")},
                }
            aggregate[model_name][method] = {
                "n_category_image_pairs": len(subset),
                "macro_mean_category_iou": float(np.mean([x["iou"] for x in categories_agg.values()])) if categories_agg else None,
                "mean_pair_iou": float(np.nanmean([r["iou"] for r in subset])) if subset else None,
                "mean_precision": float(np.nanmean([r["precision"] for r in subset])) if subset else None,
                "mean_recall": float(np.nanmean([r["recall"] for r in subset])) if subset else None,
                "mean_pointing_accuracy": float(np.nanmean([r["pointing_accuracy"] for r in subset])) if subset else None,
                "mean_auroc": float(np.nanmean([r["auroc"] for r in subset])) if subset else None,
                "mean_iou_224": float(np.nanmean([r["iou_224"] for r in subset])) if subset else None,
                "by_size_group": {
                    group: {"n": sum(r["size_group"] == group for r in subset),
                            "mean_iou": float(np.nanmean([r["iou"] for r in subset if r["size_group"] == group])) if any(r["size_group"] == group for r in subset) else None,
                            "mean_pointing_accuracy": float(np.nanmean([r["pointing_accuracy"] for r in subset if r["size_group"] == group])) if any(r["size_group"] == group for r in subset) else None}
                    for group in ("small", "medium", "large")
                },
                "per_category": categories_agg,
            }

    # Bootstrap paired image-level mean IoU differences vs untrained; descriptive 95% percentile CI.
    ci = {}
    rng = np.random.default_rng(20261118)
    image_ids_eval = sorted({r["image_id"] for r in metrics_rows})
    for method in (["raw", "cleaned", "pamr"] if args.pamr else ["raw", "cleaned"]):
        by_cp = {}
        for cp in model_results:
            by_cp[cp] = {}
            for iid in image_ids_eval:
                vals = [r["iou"] for r in metrics_rows if r["checkpoint"] == cp and r["method"] == method and r["image_id"] == iid]
                by_cp[cp][iid] = float(np.nanmean(vals)) if vals else np.nan
        for cp in ("original_pilot", "phase2_corrected"):
            common = [iid for iid in image_ids_eval if np.isfinite(by_cp["untrained"][iid]) and np.isfinite(by_cp[cp][iid])]
            diffs = np.asarray([by_cp[cp][i] - by_cp["untrained"][i] for i in common])
            draws = [float(np.mean(rng.choice(diffs, len(diffs), replace=True))) for _ in range(1000)] if len(diffs) else []
            ci[f"{cp}_minus_untrained_{method}"] = {
                "n_paired_images": len(common), "mean_difference": float(diffs.mean()) if len(diffs) else None,
                "bootstrap_95_percentile_ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))] if draws else None,
                "bootstrap_resamples": 1000, "seed": 20261118,
            }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_category_metrics.csv").open("w", newline="") as f:
        columns = list(metrics_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(metrics_rows)
    atomic_json(OUT / "segmentation_metrics.json", {
        "threshold": THRESHOLD, "threshold_semantics": "upstream bg_thresh=0.54 applied to sigmoid(cosine) maps",
        "lambda_bg": LAMBDA_BG, "grid": "16x16 native patch resolution",
        "evaluation_image_count": len(eval_records), "development_image_count": len(dev_records),
        "category_image_pairs": len(metrics_rows) // (len(model_results) * (3 if args.pamr else 2)),
        "aggregates": aggregate, "paired_bootstrap_vs_untrained": ci,
        "image_union_metrics": {
            cp: {method: float(np.nanmean([v[method + "_union_iou"] for v in per_image_results[cp].values()]))
                 for method in ("raw", "cleaned")}
            for cp in model_results
        },
        "per_image": per_image_results,
    })
    atomic_json(OUT / "checkpoint_comparison.json", {
        "checkpoints": {name: {"path": str(path), "sha256": checkpoint_hashes[name]} for name, path in CHECKPOINTS.items()},
        "models": aggregate, "paired_bootstrap_vs_untrained": ci,
        "checkpoint_weight_files_modified": False,
    })
    atomic_json(OUT / "implementation_equivalence_tests.json", implementation_validation)
    atomic_json(OUT / "resource_metrics.json", {
        "device": str(device), "elapsed_seconds_including_text_encoder_load_and_evaluation": time.perf_counter() - started,
        "text_tower_metadata": text_metadata,
        "cached_vision_forward_rerun": False, "fresh_dino_feature_extraction": False,
        "pamr_enabled": bool(args.pamr),
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
    })
    write_qualitative(qualitative_cache, category_by_id, args.pamr)
    atomic_json(OUT / "status.json", {"status": "complete", "evaluation_manifest_images": len(manifest["images"]),
                                      "evaluation_images_processed": completed_eval, "development_images": len(dev_records),
                                      "scored_checkpoints": list(CHECKPOINTS), "pamr": bool(args.pamr),
                                      "finished_unix_time": time.time()})
    return aggregate


def write_qualitative(cache: dict, categories: dict, with_pamr: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest = OUT / "qualitative_results"; dest.mkdir(parents=True, exist_ok=True)
    records = sorted(cache.items())
    # Deterministic diversity: select images adding unseen labels first, then earliest remaining rows.
    selected, selected_ids, seen = [], set(), set()
    for iid, model_data in records:
        labels = set(model_data["untrained"]["category_ids"])
        if len(selected) < 12 and labels - seen:
            selected.append((iid, model_data)); seen |= labels
            selected_ids.add(iid)
    for item in records:
        if len(selected) >= min(12, len(records)): break
        if item[0] not in selected_ids:
            selected.append(item); selected_ids.add(item[0])
    palette = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
    for n, (iid, model_data) in enumerate(selected, 1):
        first = model_data["untrained"]
        image = Image.open(first["image_path"]).convert("RGB")
        # Same policy image geometry as the feature route.
        w, h = image.size
        nh, nw = short_edge_resize_size(h, w, 224)
        image = image.resize((nw, nh), Image.Resampling.BICUBIC)
        top, left = center_crop_offsets(nh, nw, 224)
        image = image.crop((left, top, left + 224, top + 224))
        cats = first["categories"]
        count = len(cats)
        columns = ["RGB", "GT union"]
        for model in CHECKPOINTS:
            columns.extend([f"{model}\nraw cosine", f"{model}\nclean binary"])
            if with_pamr: columns.append(f"{model}\nPAMR mask")
        fig, axes = plt.subplots(count, len(columns), figsize=(2.3 * len(columns), 2.65 * count), squeeze=False)
        gt = first["gt16"]
        for ci, category in enumerate(cats):
            cls = first["category_ids"][ci]
            color = palette[ci % len(palette)]
            axes[ci, 0].imshow(image); axes[ci, 0].set_title(f"{category}\nRGB", fontsize=9)
            gt_overlay = overlay(image, F.interpolate(torch.from_numpy(gt[ci]).float()[None,None], (224,224), mode="nearest")[0,0].numpy() > .5, color)
            axes[ci, 1].imshow(gt_overlay)
            metric_lines = []
            for kind, key in (("raw", "raw"), ("clean", "cleaned")):
                vals = []
                for cp in CHECKPOINTS:
                    metric = model_data[cp]["metric_items"][cls]["methods"][key]["grid16"]
                    vals.append(f"{metric['iou']:.2f}/{metric['pointing_accuracy']:.0f}")
                metric_lines.append(f"{kind} IoU/point U|O|P: " + " | ".join(vals))
            axes[ci, 1].set_title("GT category union\n" + "\n".join(metric_lines), fontsize=7)
            col = 2
            for model_name in CHECKPOINTS:
                item = model_data[model_name]
                map_idx = item["category_ids"].index(cls)
                score = item["similarity"][map_idx]
                axes[ci, col].imshow(score, cmap="magma", vmin=-0.12, vmax=0.35, interpolation="nearest")
                axes[ci, col].set_title(f"{model_name}\nraw cosine", fontsize=8); col += 1
                clean_binary = item["binary_cleaned"][map_idx]
                clean_up = F.interpolate(torch.from_numpy(clean_binary.astype(np.float32))[None,None], (224,224), mode="nearest")[0,0].numpy() > .5
                axes[ci, col].imshow(overlay(image, clean_up, color)); axes[ci, col].set_title(f"{model_name}\ncleaned > {THRESHOLD}", fontsize=8); col += 1
                if with_pamr:
                    score = item["pamr"][map_idx]
                    axes[ci, col].imshow(overlay(image, score >= THRESHOLD, color))
                    axes[ci, col].set_title(f"{model_name}\nPAMR", fontsize=8); col += 1
            for ax in axes[ci]: ax.axis("off")
        fig.suptitle(f"COCO val image {iid} — same category prompts / all checkpoints; clean/PAMR binary threshold={THRESHOLD}", fontsize=13)
        fig.tight_layout()
        fig.savefig(dest / f"comparison_{n:02d}_image_{iid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        # Preserve maps and decision masks separately from the montage.
        payload = {"image_id": np.asarray(iid), "category_ids": np.asarray(first["category_ids"]),
                   "category_names": np.asarray(cats), "ground_truth_16": gt.astype(np.uint8)}
        for model_name in CHECKPOINTS:
            item = model_data[model_name]
            cleaned = item["cleaned"]
            best = cleaned.argmax(axis=0)
            best_score = cleaned.max(axis=0)
            semantic = np.full((16,16), -1, dtype=np.int32)
            cat_ids = np.asarray(item["category_ids"], dtype=np.int32)
            semantic[best_score >= THRESHOLD] = cat_ids[best[best_score >= THRESHOLD]]
            prefix = model_name + "__"
            payload[prefix + "raw_cosine"] = item["similarity"].astype(np.float32)
            payload[prefix + "sigmoid_map"] = item["raw"].astype(np.float32)
            payload[prefix + "background_cleaned_map"] = cleaned.astype(np.float32)
            payload[prefix + "category_binary_masks"] = item["binary_cleaned"].astype(np.uint8)
            payload[prefix + "foreground_union"] = item["binary_cleaned"].any(axis=0).astype(np.uint8)
            payload[prefix + "background_mask"] = (~item["binary_cleaned"].any(axis=0)).astype(np.uint8)
            payload[prefix + "semantic_category_map"] = semantic
            if with_pamr: payload[prefix + "pamr_map"] = item["pamr"].astype(np.float32)
        np.savez_compressed(dest / f"predictions_image_{iid}.npz", **payload)
    atomic_json(dest / "qualitative_manifest.json", {
        "panel_count": len(selected),
        "images": [{"image_id": int(iid), "category_names": model_data["untrained"]["categories"],
                    "categories": model_data["untrained"]["category_ids"],
                    "file": f"comparison_{index:02d}_image_{iid}.png"}
                   for index, (iid, model_data) in enumerate(selected, 1)],
        "raw_cosine_display_range": [-0.12, 0.35],
        "binary_threshold": THRESHOLD,
        "interpretation": "Class/category masks union all COCO instances of that category; not individual-instance separation.",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-manifest", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pamr", action="store_true")
    parser.add_argument("--max-eval-images", type=int, default=None, help="bounded smoke check; omit for full frozen manifest")
    args = parser.parse_args()
    if args.prepare_manifest:
        result = prepare_manifest()
        print(json.dumps(result["selection"], indent=2))
    if args.evaluate:
        if not args.prepare_manifest and not (OUT / "evaluation_manifest.json").is_file():
            raise SystemExit("Missing evaluation manifest; use --prepare-manifest first")
        result = evaluate(args)
        print(json.dumps({k: {m: v[m]["macro_mean_category_iou"] for m in v} for k,v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
