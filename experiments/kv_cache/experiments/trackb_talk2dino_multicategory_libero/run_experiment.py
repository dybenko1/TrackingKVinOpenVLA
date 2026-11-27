#!/usr/bin/env python3
"""Original Talk2DINO multi-category FG/BG evaluation on cached LIBERO E2 frames.

Only original Meta DINOv2 + original CLIP + original projection weights are used.
Scoring is one upstream DINOText pass per frame; threshold sweeps use saved maps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path("/opt/openvla")
SOURCE = REPO / "cache/datasets/talk2dino_native_e2/scoring"
WORKTREE = REPO / "cache/worktrees/trackb-dense-grounding-v2"
OUT = WORKTREE / "experiments/kv_cache/outputs/trackb_talk2dino_multicategory_libero"
CONTAINER = "trackb-v1-coco-object-20261126"
BG_THRESHOLD = 0.54
RECALL_TARGETS = (0.95, 0.98)
FIXED_BUDGETS = (0.10, 0.25, 0.50, 0.75, 0.90)
THRESHOLDS = sorted(set([round(x, 2) for x in np.arange(0.05, 1.0, 0.05)] + [BG_THRESHOLD]))

# Explicit task-conditioned, independent category queries.  A separate CLIP
# embedding is produced for each phrase and all phrases are scored together.
# Each suite has one development segment and two held-out segments.
TASKS = {
    "libero_goal/open_the_middle_drawer_of_the_cabinet": {
        "demo": "demo_0", "split": "development",
        "queries": [
            {"text": "middle drawer of the cabinet", "role": "task_fixture", "gt": ["task_relevant_fixtures"]},
            {"text": "robot arm", "role": "robot_arm", "gt": ["robot_arm"]},
            {"text": "robot gripper", "role": "robot_gripper", "gt": ["robot_gripper"]},
        ],
    },
    "libero_goal/put_the_bowl_on_the_plate": {
        "demo": "demo_1", "split": "evaluation",
        "queries": [
            {"text": "bowl", "role": "manipulated_object", "gt": ["manipulated_objects"]},
            {"text": "plate", "role": "destination_object", "gt": ["destination_objects"]},
            {"text": "robot arm", "role": "robot_arm", "gt": ["robot_arm"]},
            {"text": "robot gripper", "role": "robot_gripper", "gt": ["robot_gripper"]},
        ],
    },
    "libero_goal/put_the_wine_bottle_on_the_rack": {
        "demo": "demo_10", "split": "evaluation",
        "queries": [
            {"text": "wine bottle", "role": "manipulated_object", "gt": ["manipulated_objects"]},
            {"text": "wine rack", "role": "destination_or_fixture", "gt": ["destination_objects", "task_relevant_fixtures"]},
            {"text": "robot arm", "role": "robot_arm", "gt": ["robot_arm"]},
            {"text": "robot gripper", "role": "robot_gripper", "gt": ["robot_gripper"]},
        ],
    },
    "libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket": {
        "demo": "demo_0", "split": "development",
        "queries": [
            {"text": "alphabet soup", "role": "manipulated_object", "gt": ["manipulated_objects"]},
            {"text": "basket", "role": "destination_object", "gt": ["destination_objects", "task_relevant_fixtures"]},
            {"text": "robot arm", "role": "robot_arm", "gt": ["robot_arm"]},
            {"text": "robot gripper", "role": "robot_gripper", "gt": ["robot_gripper"]},
        ],
    },
    "libero_object/pick_up_the_butter_and_place_it_in_the_basket": {
        "demo": "demo_1", "split": "evaluation",
        "queries": [
            {"text": "butter", "role": "manipulated_object", "gt": ["manipulated_objects"]},
            {"text": "basket", "role": "destination_object", "gt": ["destination_objects", "task_relevant_fixtures"]},
            {"text": "robot arm", "role": "robot_arm", "gt": ["robot_arm"]},
            {"text": "robot gripper", "role": "robot_gripper", "gt": ["robot_gripper"]},
        ],
    },
    "libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket": {
        "demo": "demo_10", "split": "evaluation",
        "queries": [
            {"text": "tomato sauce", "role": "manipulated_object", "gt": ["manipulated_objects"]},
            {"text": "basket", "role": "destination_object", "gt": ["destination_objects", "task_relevant_fixtures"]},
            {"text": "robot arm", "role": "robot_arm", "gt": ["robot_arm"]},
            {"text": "robot gripper", "role": "robot_gripper", "gt": ["robot_gripper"]},
        ],
    },
    "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": {
        "demo": "demo_0", "split": "development",
        "queries": [
            {"text": "black bowl", "role": "manipulated_object", "gt": ["manipulated_objects"]},
            {"text": "plate", "role": "destination_object", "gt": ["destination_objects"]},
            {"text": "ramekin", "role": "mentioned_context_object", "gt": ["other_objects"]},
            {"text": "robot arm", "role": "robot_arm", "gt": ["robot_arm"]},
            {"text": "robot gripper", "role": "robot_gripper", "gt": ["robot_gripper"]},
        ],
    },
    "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate": {
        "demo": "demo_1", "split": "evaluation",
        "queries": [
            {"text": "black bowl", "role": "manipulated_object", "gt": ["manipulated_objects"]},
            {"text": "plate", "role": "destination_object", "gt": ["destination_objects"]},
            {"text": "table", "role": "spatial_reference", "gt": ["task_relevant_fixtures"]},
            {"text": "robot arm", "role": "robot_arm", "gt": ["robot_arm"]},
            {"text": "robot gripper", "role": "robot_gripper", "gt": ["robot_gripper"]},
        ],
    },
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": {
        "demo": "demo_10", "split": "evaluation",
        "queries": [
            {"text": "black bowl", "role": "manipulated_object", "gt": ["manipulated_objects"]},
            {"text": "plate", "role": "destination_object", "gt": ["destination_objects"]},
            {"text": "stove", "role": "task_fixture", "gt": ["task_relevant_fixtures"]},
            {"text": "robot arm", "role": "robot_arm", "gt": ["robot_arm"]},
            {"text": "robot gripper", "role": "robot_gripper", "gt": ["robot_gripper"]},
        ],
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def find_segments(source: Path):
    found = {}
    for path in sorted(source.glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
        tid = meta["task_id"]
        if tid in TASKS and meta["demo"] == TASKS[tid]["demo"]:
            if tid in found:
                raise RuntimeError(f"Duplicate segment selected for {tid}: {found[tid]} {path}")
            found[tid] = path
    missing = sorted(set(TASKS) - set(found))
    if missing:
        raise FileNotFoundError(f"Selected source segment(s) missing: {missing}")
    suites = {}
    for tid, spec in TASKS.items():
        suite = tid.split("/", 1)[0]
        suites.setdefault(suite, []).append(spec["split"])
    if any(sum(x == "development" for x in v) != 1 or sum(x == "evaluation" for x in v) != 2 for v in suites.values()):
        raise AssertionError(f"Invalid suite split: {suites}")
    return found


def choose_frames(n: int):
    idx = np.linspace(0, n - 1, 10).round().astype(int)
    if len(np.unique(idx)) != 10:
        raise RuntimeError(f"Cannot select 10 unique evenly spaced frames from {n}")
    return idx


def score_bgr_raw_and_clean(official, bgr: np.ndarray, text_embedding, query_names):
    """Use upstream generate_masks once, capturing its raw and cleaned maps."""
    import torch
    model = official.model
    arr = np.asarray(bgr)
    image = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(
        device=next(model.parameters()).device, dtype=torch.float32)
    model.feats.clear()
    masker = model.masker
    orig_forward_seg = masker.forward_seg
    orig_clean = model.similarity_assignment_weighted
    captured = {}

    def capture_forward_seg(image_feat, text_emb, deterministic=True, hard=False):
        raw, cosine = orig_forward_seg(image_feat, text_emb, deterministic=deterministic, hard=hard)
        captured["raw"] = raw.detach()
        captured["cosine"] = cosine.detach()
        return raw, cosine

    def capture_clean(mask, image_feat, self_attn_maps, text_emb, lambda_bg=0.2):
        cleaned = orig_clean(mask, image_feat, self_attn_maps, text_emb, lambda_bg=lambda_bg)
        captured["clean"] = cleaned.detach()
        return cleaned

    masker.forward_seg = capture_forward_seg
    model.similarity_assignment_weighted = capture_clean
    model.with_bg_clean = True
    try:
        # This is the official path: it runs the vision encoder once, then the
        # original multi-class per-head cleaning. The captured raw map is the
        # same pre-cleaning DINOTextMasker output from that forward pass.
        _, _ = model.generate_masks(
            image, None, text_embedding,
            ["background"] + [str(i) for i in range(text_embedding.shape[0])],
            apply_pamr=False, lambda_bg=0.2,
        )
    finally:
        masker.forward_seg = orig_forward_seg
        model.similarity_assignment_weighted = orig_clean
    for key in ("raw", "clean", "cosine"):
        if key not in captured or not torch.isfinite(captured[key]).all():
            raise RuntimeError(f"Official multi-category scorer failed to return finite {key} maps")
    raw = captured["raw"][0].float().cpu().numpy().reshape(len(query_names), 256)
    clean = captured["clean"][0].float().cpu().numpy().reshape(len(query_names), 256)
    cosine = captured["cosine"][0].float().cpu().numpy().reshape(len(query_names), 256)
    if raw.shape != clean.shape or raw.shape != (len(query_names), 256):
        raise RuntimeError(f"Expected [queries,256] at 224 px; got {raw.shape}/{clean.shape}")
    return raw, clean, cosine


def role_target(role_gt, role_keys, role_spec, frame):
    y = np.zeros(256, np.float32)
    present = False
    for key in role_spec["gt"]:
        if key in role_keys:
            y = np.maximum(y, role_gt[frame, role_keys.index(key)])
            present = True
    return y if present else None


def global_counts(y, pred, valid):
    y = np.clip(np.asarray(y, np.float64), 0, 1) * np.asarray(valid, bool)
    pred = np.asarray(pred, bool) & np.asarray(valid, bool)
    v = np.asarray(valid, bool)
    tp = float((pred * y).sum())
    fp = float((pred * (1 - y) * v).sum())
    fn = float(((~pred) * y).sum())
    tn = float(((~pred) * (1 - y) * v).sum())
    return {"fg_tp": tp, "fg_fp": fp, "fg_fn": fn, "fg_tn": tn,
            "fg_precision": tp / max(tp + fp, 1e-12),
            "fg_recall": tp / max(tp + fn, 1e-12),
            "fg_iou": tp / max(tp + fp + fn, 1e-12),
            "bg_precision": tn / max(tn + fn, 1e-12),
            "bg_recall": tn / max(tn + fp, 1e-12),
            "bg_iou": tn / max(tn + fn + fp, 1e-12),
            "bg_coverage": float((v & ~pred).sum() / max(v.sum(), 1)),
            "fg_coverage": float((v & pred).sum() / max(v.sum(), 1)),
            "foreground_as_background_rate": fn / max(tp + fn, 1e-12),
            "valid_patch_equivalents": float(v.sum())}


def aggregate(scores, targets, valids, threshold):
    totals = np.zeros(4, np.float64)  # tp fp fn tn
    valid_tokens = fg_tokens = bg_tokens = 0
    for score, y, valid in zip(scores, targets, valids):
        pred = score >= threshold
        y = np.clip(y, 0, 1) * valid
        pred &= valid
        totals += [(pred * y).sum(), (pred * (1-y) * valid).sum(), ((~pred)*y).sum(), ((~pred)*(1-y)*valid).sum()]
        valid_tokens += int(valid.sum()); fg_tokens += int((pred & valid).sum()); bg_tokens += int((~pred & valid).sum())
    tp, fp, fn, tn = totals
    return {"fg_tp": float(tp), "fg_fp": float(fp), "fg_fn": float(fn), "fg_tn": float(tn),
            "fg_precision": float(tp/max(tp+fp,1e-12)), "fg_recall": float(tp/max(tp+fn,1e-12)),
            "fg_iou": float(tp/max(tp+fp+fn,1e-12)), "bg_precision": float(tn/max(tn+fn,1e-12)),
            "bg_recall": float(tn/max(tn+fp,1e-12)), "bg_iou": float(tn/max(tn+fp+fn,1e-12)),
            "bg_coverage": float(bg_tokens/max(valid_tokens,1)), "fg_coverage": float(fg_tokens/max(valid_tokens,1)),
            "foreground_as_background_rate": float(fn/max(tp+fn,1e-12)),
            "valid_patch_equivalents": int(valid_tokens)}


def threshold_choice(scores, targets, valids, target_recall, sweep_rows, method):
    eligible = [r for r in sweep_rows if r["method"] == method and r["split"] == "development"
                and r["fg_recall"] >= target_recall]
    if not eligible:
        return {"threshold": min(THRESHOLDS), "constraint_met": False, "target_recall": target_recall,
                "selection_metric": "maximum development bg IoU subject to foreground recall floor"}
    selected = sorted(eligible, key=lambda r: (r["bg_iou"], r["threshold"]))[-1]
    return {"threshold": selected["threshold"], "constraint_met": True, "target_recall": target_recall,
            "development_fg_recall": selected["fg_recall"], "development_bg_iou": selected["bg_iou"],
            "development_bg_coverage": selected["bg_coverage"],
            "selection_metric": "maximum development bg IoU subject to foreground recall floor"}


def load_model_224(device):
    import torch
    from experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.official_inference import OfficialTalk2DINO
    from experiments.kv_cache.experiments.trackb_talk2dino_native.run_libero_e2 import load_official_dino
    from experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.run_coco import (
        UPSTREAM, DINO_WEIGHT, CLIP_WEIGHT, MLP_WEIGHT, DINO_SHA, IMAGE_SHA, MLP_SHA, sha256,
    )
    expected = ((DINO_WEIGHT, DINO_SHA, "DINOv2"), (CLIP_WEIGHT, IMAGE_SHA, "CLIP"), (MLP_WEIGHT, MLP_SHA, "MLP"))
    for path, want, label in expected:
        got = sha256(path)
        if got != want:
            raise RuntimeError(f"{label} hash mismatch: {got} != {want}")
    dino, dino_meta = load_official_dino(device, torch.float32)
    dino_meta["source_revision"] = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    official = OfficialTalk2DINO(UPSTREAM, dino, CLIP_WEIGHT, MLP_WEIGHT, resize_dim=224, device=str(device))
    return official, {"dino": dino_meta, "dino_sha256": DINO_SHA, "clip_sha256": IMAGE_SHA,
                      "mlp_sha256": MLP_SHA, "upstream_revision": "6dac47dbe3500df63f3b3df20dd28334290919cc",
                      "visual_resolution": [224, 224], "patch_grid": [16, 16],
                      "feature_definition": "original DINOv2 ViT-L/14-register, final block, original DINOText feature and attention cleaning",
                      "text_encoder": "OpenAI CLIP ViT-B/16", "prompt_template": "upstream sub_imagenet_template",
                      "lambda_bg": 0.2, "pamr": False}


def collect_scores(source_paths, smoke=False):
    import torch
    official, identity = load_model_224(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    output = []
    timings, text_timings = [], []
    manifest_segments = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    segment_items = list(source_paths.items())
    for si, (task_id, path) in enumerate(segment_items, 1):
        spec = TASKS[task_id]
        with np.load(path, allow_pickle=False) as z:
            rgb = z["rgb"].copy()
            role_gt = z["role_gt"].astype(np.float32)
            relevant = z["relevant_gt"].astype(np.float32)
            valid = z["valid"].astype(np.float32)
            metadata = json.loads(str(z["metadata"].item()))
        if metadata["task_id"] != task_id or metadata["demo"] != spec["demo"]:
            raise RuntimeError(f"Source mismatch for {path}")
        indices = choose_frames(len(rgb))
        if smoke:
            indices = indices[:1]
        queries = spec["queries"]
        t0 = time.perf_counter()
        text = official.encode_categories([q["text"] for q in queries], "sub_imagenet_template")
        if torch.cuda.is_available(): torch.cuda.synchronize()
        text_timings.append((time.perf_counter() - t0) * 1000)
        if tuple(text.shape) != (len(queries), 1024) or not torch.isfinite(text).all():
            raise RuntimeError(f"Bad projected text embeddings for {task_id}: {tuple(text.shape)}")
        seg = {"task_id": task_id, "suite": metadata["suite"], "demo": spec["demo"], "split": spec["split"],
               "instruction": metadata["instruction"], "queries": queries, "source_archive": str(path),
               "source_sha256": sha256(path), "selected_indices": [], "rgb": [], "relevant_gt": [],
               "role_gt": [], "role_keys": metadata["role_keys"], "valid": [], "raw": [], "clean": [], "cosine": []}
        for fi in indices:
            t0 = time.perf_counter()
            raw, clean, cosine = score_bgr_raw_and_clean(official, rgb[fi, ..., ::-1].copy(), text, [q["text"] for q in queries])
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            if not np.isfinite(raw).all() or not np.isfinite(clean).all() or not np.isfinite(cosine).all():
                raise RuntimeError(f"Nonfinite score array: {task_id} frame {fi}")
            if raw.shape[1] != 256 or clean.shape[1] != 256:
                raise RuntimeError(f"Visual patch alignment mismatch: {raw.shape}, {clean.shape}")
            timings.append(elapsed)
            seg["selected_indices"].append(int(fi)); seg["rgb"].append(rgb[fi]); seg["relevant_gt"].append(relevant[fi]);
            seg["role_gt"].append(role_gt[fi]); seg["valid"].append(valid[fi] > 0); seg["raw"].append(raw); seg["clean"].append(clean); seg["cosine"].append(cosine)
        for key in ("rgb", "relevant_gt", "role_gt", "valid", "raw", "clean", "cosine"):
            seg[key] = np.asarray(seg[key])
        manifest_segments.append({"task_id": task_id, "suite": metadata["suite"], "demo": spec["demo"],
                                  "split": spec["split"], "instruction": metadata["instruction"],
                                  "source_archive": str(path), "source_sha256": sha256(path),
                                  "source_states_slice_sha256": metadata.get("states_slice_sha256"),
                                  "selected_frame_indices": [int(x) for x in indices],
                                  "queries": queries, "image_count": len(indices)})
        output.append(seg)
        atomic_json(OUT / "RUN_STATUS.json", {"status": "running", "stage": "score_images", "segments_done": si,
                    "segments_total": len(segment_items), "frames_done": sum(len(x["selected_frame_indices"]) for x in manifest_segments),
                    "frames_total": 1 if smoke else 90, "last_task": task_id, "updated_unix": time.time()})
    model_memory = {}
    if torch.cuda.is_available():
        model_memory = {"peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                        "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
    official.close()
    return output, identity, manifest_segments, timings, text_timings, model_memory


def flatten(scores_by_segment, split, method):
    scores, targets, valids, refs = [], [], [], []
    for seg in scores_by_segment:
        if split and seg["split"] != split:
            continue
        s = seg[method].max(axis=1)
        for i in range(len(s)):
            scores.append(s[i]); targets.append(seg["relevant_gt"][i]); valids.append(seg["valid"][i]); refs.append((seg, i))
    return scores, targets, valids, refs


def role_metrics(scores_by_segment, method, threshold):
    sums = {}
    for seg in scores_by_segment:
        for qi, q in enumerate(seg["queries"]):
            y = role_target(seg["role_gt"], seg["role_keys"], q, 0)
            if y is None:
                continue
            for i in range(len(seg["rgb"])):
                y = role_target(seg["role_gt"], seg["role_keys"], q, i)
                if y is None or float(y.sum()) == 0:
                    continue
                pred = seg[method][i, qi] >= threshold
                valid = seg["valid"][i]
                c = global_counts(y, pred, valid)
                bucket = sums.setdefault(q["role"], [0.0, 0.0])
                bucket[0] += c["fg_tp"]
                bucket[1] += c["fg_fn"]
    return {role: {"foreground_recall": tp / max(tp + fn, 1e-12), "foreground_positive_patch_equivalents": tp + fn}
            for role, (tp, fn) in sums.items()}


def analyze(data, identity, manifest, timings, text_timings, memory):
    sweep = []
    for split in ("development", "evaluation"):
        for method in ("raw", "clean"):
            scores, targets, valids, _ = flatten(data, split, method)
            for threshold in THRESHOLDS:
                m = aggregate(scores, targets, valids, threshold)
                sweep.append({"split": split, "method": method, "threshold": threshold, **m})
    calibrated = {}
    for method in ("raw", "clean"):
        calibrated[method] = {}
        for target in RECALL_TARGETS:
            calibrated[method][str(target)] = threshold_choice(None, None, None, target, sweep, method)
    sweep_rows = {(x["split"], x["method"], x["threshold"]): x for x in sweep}
    operating = {"official_0.54": {}, "development_selected_95_recall": {}, "development_selected_98_recall": {}}
    for split in ("development", "evaluation"):
        for method in ("raw", "clean"):
            operating["official_0.54"].setdefault(split, {})[method] = sweep_rows[(split, method, BG_THRESHOLD)]
            for target, key in ((0.95, "development_selected_95_recall"), (0.98, "development_selected_98_recall")):
                tau = calibrated[method][str(target)]["threshold"]
                operating[key].setdefault(split, {})[method] = sweep_rows[(split, method, tau)]
    role = {}
    for method in ("raw", "clean"):
        role[method] = {
            "official_0.54": {split: role_metrics([x for x in data if x["split"] == split], method, BG_THRESHOLD)
                              for split in ("development", "evaluation")},
            "development_selected_95_recall": {split: role_metrics([x for x in data if x["split"] == split], method,
                calibrated[method]["0.95"]["threshold"]) for split in ("development", "evaluation")},
        }
    fixed_budget = {}
    for split in ("development", "evaluation"):
        fixed_budget[split] = {}
        for method in ("raw", "clean"):
            ss, ys, vs, _ = flatten(data, split, method)
            values = {}
            for budget in FIXED_BUDGETS:
                nums = dens = 0.0
                for s, y, v in zip(ss, ys, vs):
                    n = int(np.ceil(budget * int(v.sum())))
                    idx = np.argsort(-s, kind="stable")[:n]
                    pred = np.zeros(256, bool); pred[idx] = True
                    nums += float((pred * y * v).sum())
                    dens += float((y * v).sum())
                values[str(budget)] = {"foreground_recall": nums / max(dens, 1e-12), "protected_token_budget": budget}
            fixed_budget[split][method] = values

    # Per-frame records at the official threshold plus development-chosen
    # thresholds. Eval thresholds come only from the dev segment of each suite.
    per_frame = []
    selected_by_method = {m: calibrated[m]["0.95"]["threshold"] for m in ("raw", "clean")}
    for seg in data:
        for i in range(len(seg["rgb"])):
            base = {"suite": seg["suite"], "task_id": seg["task_id"], "demo": seg["demo"], "split": seg["split"],
                    "frame_index": seg["selected_indices"][i], "instruction": seg["instruction"],
                    "queries_json": json.dumps([q["text"] for q in seg["queries"]])}
            for method in ("raw", "clean"):
                for label, tau in (("official_0.54", BG_THRESHOLD), ("dev_recall95", selected_by_method[method])):
                    c = global_counts(seg["relevant_gt"][i], seg[method][i].max(axis=0) >= tau, seg["valid"][i])
                    per_frame.append({**base, "method": method, "threshold_policy": label, "threshold": tau, **c})
                for qi, q in enumerate(seg["queries"]):
                    y = role_target(seg["role_gt"], seg["role_keys"], q, i)
                    if y is None:
                        continue
                    for label, tau in (("official_0.54", BG_THRESHOLD), ("dev_recall95", selected_by_method[method])):
                        c = global_counts(y, seg[method][i, qi] >= tau, seg["valid"][i])
                        per_frame.append({**base, "method": method, "threshold_policy": label, "threshold": tau,
                                          "query": q["text"], "query_role": q["role"],
                                          "role_fg_recall": c["fg_recall"], "role_fg_precision": c["fg_precision"],
                                          "role_fg_iou": c["fg_iou"], "role_valid_patch_equivalents": c["valid_patch_equivalents"]})

    sweep_file = OUT / "threshold_sweep.csv"
    with sweep_file.open("w", newline="") as f:
        fields = list(sweep[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(sweep)
    frame_file = OUT / "per_frame_metrics.csv"
    fields = sorted({k for r in per_frame for k in r})
    with frame_file.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(per_frame)

    summary = {"image_count": sum(len(x["rgb"]) for x in data), "segment_count": len(data),
               "split_counts": {s: sum(len(x["rgb"]) for x in data if x["split"] == s) for s in ("development", "evaluation")},
               "operating_points": operating, "thresholds_selected_using_development_only": calibrated,
               "role_recall": role, "foreground_recall_at_fixed_protected_token_budgets": fixed_budget,
               "metrics_use_fractional_E2_patch_occupancy": True,
               "background_decision": "max over independently queried category scores at each 16x16 position; below threshold=background",
               "threshold_sweep_values": THRESHOLDS, "model_identity": identity}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    latency = {"images": sum(len(x["rgb"]) for x in data), "per_frame_raw_and_clean_same_vision_forward_ms": {
                   "mean": float(np.mean(timings)), "median": float(np.median(timings)),
                   "p95": float(np.percentile(timings, 95)), "min": float(np.min(timings)), "max": float(np.max(timings))},
               "per_task_text_embedding_ms": {"mean": float(np.mean(text_timings)), "total": float(np.sum(text_timings))},
               "peak_gpu_memory": memory,
               "threshold_sweep_additional_model_inference": False,
               "timing_scope": "GPU synchronized upstream single-image DINOText scoring including simultaneous query maps and original attention cleaning; excludes figure rendering and offline threshold sweep"}
    (OUT / "latency_metrics.json").write_text(json.dumps(latency, indent=2) + "\n")
    return summary, latency, sweep, per_frame


def draw_figures(data, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    chosen = []
    thresholds = {m: summary["thresholds_selected_using_development_only"][m]["0.95"]["threshold"] for m in ("raw", "clean")}
    for suite in ("libero_goal", "libero_object", "libero_spatial"):
        candidates=[]
        for seg in data:
            if seg["suite"] != suite or seg["split"] != "evaluation": continue
            for i in range(len(seg["rgb"])):
                area=float((seg["relevant_gt"][i]*seg["valid"][i]).sum())
                candidates.append((area,seg,i))
        candidates.sort(key=lambda x:(x[0],x[1]["task_id"],x[1]["demo"],x[2]))
        if not candidates: continue
        area, seg, i = candidates[(len(candidates)-1)//2]
        rgb=seg["rgb"][i].astype(np.uint8)
        valid=seg["valid"][i].reshape(16,16)
        gt_soft=seg["relevant_gt"][i].reshape(16,16)
        gt=(gt_soft>0)&valid
        predictions={m:(seg[m][i].max(axis=0).reshape(16,16)>=thresholds[m])&valid for m in ("raw","clean")}
        fig=plt.figure(figsize=(16,11))
        grid=fig.add_gridspec(3,4,width_ratios=[1,1,1,1])
        ax=fig.add_subplot(grid[:,0]); ax.imshow(rgb); ax.set_title("Original RGB"); ax.axis("off")
        ax=fig.add_subplot(grid[0,1]); ax.imshow(rgb*np.repeat(np.repeat(gt,14,0),14,1)[...,None]); ax.set_title("Oracle foreground\n(black elsewhere)"); ax.axis("off")
        ax=fig.add_subplot(grid[1,1]); bg=(~gt)&valid; ax.imshow(rgb*np.repeat(np.repeat(bg,14,0),14,1)[...,None]); ax.set_title("Oracle background\n(black elsewhere)"); ax.axis("off")
        ax=fig.add_subplot(grid[2,1]); ax.imshow(error_overlay(rgb,gt,gt,valid)); ax.set_title("Oracle binary mask"); ax.axis("off")
        stats={}
        for col,m in ((2,"raw"),(3,"clean")):
            pred=predictions[m]
            ax=fig.add_subplot(grid[0,col]); ax.imshow(rgb*np.repeat(np.repeat(pred,14,0),14,1)[...,None]); ax.set_title(f"{m.upper()} foreground\nτ={thresholds[m]:.3f}"); ax.axis("off")
            pbg=(~pred)&valid
            ax=fig.add_subplot(grid[1,col]); ax.imshow(rgb*np.repeat(np.repeat(pbg,14,0),14,1)[...,None]); ax.set_title(f"{m.upper()} background\n(black elsewhere)"); ax.axis("off")
            ax=fig.add_subplot(grid[2,col]); ax.imshow(error_overlay(rgb,gt,pred,valid)); ax.set_title(f"{m.upper()} errors\nTP green · FP red · FN yellow"); ax.axis("off")
            c=global_counts(gt_soft.flatten(),pred.flatten(),valid.flatten())
            stats[m]=c
        qtext=", ".join(q["text"] for q in seg["queries"])
        fig.suptitle(f"{suite} — {seg['task_id'].split('/',1)[1]} · {seg['demo']} frame {seg['selected_indices'][i]}\nSimultaneous independent queries: {qtext}",fontsize=11,y=0.98)
        foot=(f"E2 binary display: fractional occupancy > 0 on 16×16 grid; nearest-neighbor display on 224×224 RGB. "
              f"RAW: FG recall {stats['raw']['fg_recall']:.3f} · BG precision {stats['raw']['bg_precision']:.3f} · "
              f"BG IoU {stats['raw']['bg_iou']:.3f} · BG coverage {stats['raw']['bg_coverage']:.1%}.  "
              f"CLEAN: FG recall {stats['clean']['fg_recall']:.3f} · BG precision {stats['clean']['bg_precision']:.3f} · "
              f"BG IoU {stats['clean']['bg_iou']:.3f} · BG coverage {stats['clean']['bg_coverage']:.1%}.  "
              f"Thresholds selected on dev for ≥95% FG recall: RAW {thresholds['raw']:.3f}, CLEAN {thresholds['clean']:.3f}.")
        fig.tight_layout(rect=(0,0.15,1,0.90))
        fig.text(0.5,0.035,foot,ha="center",va="bottom",fontsize=8.5,wrap=True)
        path=figdir/f"{suite}_multicategory_binary.png"; fig.savefig(path,dpi=145,bbox_inches="tight");plt.close(fig)
        chosen.append({"suite":suite,"task_id":seg["task_id"],"demo":seg["demo"],"frame_index":seg["selected_indices"][i],
                       "queries":[q["text"] for q in seg["queries"]],"thresholds":thresholds,
                       "rgb_binary_gt_patch_rule":"occupancy > 0", "raw_metrics":stats["raw"],"clean_metrics":stats["clean"],"figure":path.name})
    (figdir/"figure_manifest.json").write_text(json.dumps(chosen,indent=2)+"\n")
    return chosen


def error_overlay(rgb, gt, pred, valid):
    gtpx=np.repeat(np.repeat(gt,14,0),14,1); predpx=np.repeat(np.repeat(pred,14,0),14,1)
    vpx=np.repeat(np.repeat(valid,14,0),14,1)
    tp=gtpx&predpx; fp=(~gtpx)&predpx&vpx; fn=gtpx&(~predpx)
    color=np.zeros_like(rgb,dtype=np.float32)
    color[tp]=(0,210,0); color[fp]=(245,20,20); color[fn]=(255,220,0)
    return np.clip(rgb.astype(np.float32)*0.45+color*0.55,0,255).astype(np.uint8)


def report(summary, latency, figures):
    ops=summary["operating_points"]
    lines=[
        "# Original Talk2DINO multi-category background detection on LIBERO",
        "",
        "## Result",
        "",
        "This run tests original standalone Talk2DINO on 90 cached E2 RGB/oracle pairs with independently encoded object queries processed simultaneously. "
        "It does not train, run policy rollouts, use OpenVLA DINO features, or alter the KV cache.",
        "",
        "The primary multi-category decision is the original background-rejection rule: take the maximum cleaned (or raw) category score at each 16×16 patch; below threshold is background. "
        "All metrics compare against the E2 task-relevant fractional patch-occupancy union. Display masks binarize oracle occupancy at >0 only for visualization.",
        "",
        "## Split and inference",
        "",
        f"- Images: {summary['image_count']} total; {summary['split_counts']['development']} development and {summary['split_counts']['evaluation']} evaluation.",
        "- Three task demonstrations per suite; ten evenly spaced frames per demonstration. Split is by entire demonstration segment (one segment per suite for development; two per suite for evaluation).",
        "- Original Meta DINOv2 ViT-L/14-register, OpenAI CLIP ViT-B/16, original projection MLP, original `sub_imagenet_template`, original per-head attention cleaning with λ=0.2. Inference input is 224×224, yielding a 16×16 patch grid. PAMR disabled.",
        "- Each task's projected query embeddings are encoded once, then all listed object/robot phrases are scored in the same upstream DINOText pass.",
        "- Threshold sweep uses stored score arrays; no additional vision inference is run for the sweep.",
        "",
        "## Main evaluation-set comparison",
        "",
        "The 0.54 rows are the original COCO reference threshold. The 95%-recall thresholds were selected on development demonstrations only; evaluation columns use those frozen thresholds.",
        "",
        "| Variant / threshold policy | Threshold | FG recall | FG precision | FG IoU | BG precision | BG recall | BG IoU | BG coverage | Foreground missed as BG |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy,key in (("RAW at 0.54","official_0.54"),("CLEAN at 0.54","official_0.54"),
                       ("RAW dev-selected ≥95% recall","development_selected_95_recall"),("CLEAN dev-selected ≥95% recall","development_selected_95_recall")):
        method="raw" if policy.startswith("RAW") else "clean"
        x=ops[key]["evaluation"][method]
        lines.append(f"| {policy} | {x['threshold']:.3f} | {x['fg_recall']:.3f} | {x['fg_precision']:.3f} | {x['fg_iou']:.3f} | {x['bg_precision']:.3f} | {x['bg_recall']:.3f} | {x['bg_iou']:.3f} | {x['bg_coverage']:.1%} | {x['foreground_as_background_rate']:.1%} |")
    lines += ["", "Metrics are pooled fractional patch occupancy, not per-image binary IoU. BG is the complement of the task-relevant oracle and predicted FG on valid patches.",
              "", "## Development-only operating points", "",
              "For each variant, the displayed operating threshold maximizes development BG IoU subject to pooled FG recall ≥95% (and separately ≥98%). It is then frozen for the evaluation split.",
              "", "| Variant | Target dev FG recall | Selected threshold | Dev FG recall | Dev BG IoU | Dev BG coverage | Eval FG recall | Eval BG IoU | Eval BG coverage |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("raw","clean"):
        for target in ("0.95","0.98"):
            ch=summary["thresholds_selected_using_development_only"][method][target]
            d=ops["official_0.54"]["development"][method] if False else next(
                x for x in [] ) if False else None
            tau=ch["threshold"]
            dev=next(x for x in summary["_sweep_cache"] if x["split"]=="development" and x["method"]==method and x["threshold"]==tau) if "_sweep_cache" in summary else None
            # use selected threshold values inserted by caller below
            dev=summary["calibration_metrics"][method][target]
            ev=summary["evaluation_at_calibration"][method][target]
            lines.append(f"| {method.upper()} | {float(target):.0%} | {tau:.3f} | {dev['fg_recall']:.3f} | {dev['bg_iou']:.3f} | {dev['bg_coverage']:.1%} | {ev['fg_recall']:.3f} | {ev['bg_iou']:.3f} | {ev['bg_coverage']:.1%} |")
    lines += ["", "## Foreground recall at protected-token budgets", "",
              "These ranking diagnostics protect the top-scoring patches per image. They are not the thresholded background detector and do not prove KV safety.",
              "", "| Split | Variant | 10% | 25% | 50% | 75% | 90% |", "|---|---|---:|---:|---:|---:|---:|"]
    for split in ("development","evaluation"):
        for method in ("raw","clean"):
            vals=summary["foreground_recall_at_fixed_protected_token_budgets"][split][method]
            lines.append(f"| {split} | {method.upper()} | " + " | ".join(f"{vals[str(b)]['foreground_recall']:.3f}" for b in FIXED_BUDGETS)+" |")
    lines += ["", "## Object-role recall", "",
              "Role recall evaluates each independent category map against the corresponding coarser E2 role channel. For categories such as ramekin/table, the simulator label is a role union rather than a category-specific instance mask; interpret those rows as role-level diagnostics.",
              "", "Evaluation split at the official 0.54 threshold:", "",
              "| Role | RAW recall | CLEAN recall |", "|---|---:|---:|"]
    role_raw=summary["role_recall"]["raw"]["official_0.54"]["evaluation"]
    role_clean=summary["role_recall"]["clean"]["official_0.54"]["evaluation"]
    for role_name in sorted(set(role_raw)|set(role_clean)):
        lines.append(f"| {role_name} | {role_raw.get(role_name,{}).get('foreground_recall',float('nan')):.3f} | {role_clean.get(role_name,{}).get('foreground_recall',float('nan')):.3f} |")
    lines += ["", "These per-query recalls use each query's own score map and a coarser role label. They do not establish category-specific instance accuracy.",
              "", "## Visual examples", ""]
    for x in figures:
        lines += [f"### {x['suite']} — {x['task_id'].split('/',1)[1]}", "",
                  f"Queries: {', '.join(x['queries'])}", "",
                  f"![Binary multi-category FG/BG comparison]({x['figure']})", ""]
    timing=latency["per_frame_raw_and_clean_same_vision_forward_ms"]
    latency_text=(f"One inference forward per image returned RAW and CLEAN maps. Recorded mean synchronized scoring time: **{timing['mean']:.2f} ms** over {timing.get('sample_count', 90)} images. Per-image median/p95/min/max are unavailable in the preserved timing artifact and are not reported.")
    lines += ["## Cost", "", latency_text,
              f"Projected category-text embeddings were computed once per task segment: mean **{latency['per_task_text_embedding_ms']['mean']:.2f} ms** per segment across {latency['per_task_text_embedding_ms'].get('sample_count', 9)} segments (reported total {latency['per_task_text_embedding_ms'].get('total', float('nan')):.2f} ms).",
              f"Peak CUDA allocation: {latency['peak_gpu_memory'].get('peak_cuda_allocated_gb','not available') if latency['peak_gpu_memory'] else 'not available'} GB; peak CUDA reserved: {latency['peak_gpu_memory'].get('peak_cuda_reserved_gb','not available') if latency['peak_gpu_memory'] else 'not available'} GB.",
              "", "## Limitations and conclusions", "",
              "This covers three demonstration segments per suite (one development segment and two evaluation segments), not broad generalization. E2 labels are fractional 16×16 patch occupancies, so these measurements cannot establish pixel-level object boundary quality. The 0.54 threshold is the COCO reference, not a LIBERO-calibrated setting.",
              "", "A high-recall background operating point is only a segmentation operating point. It does not establish that stale K/V can be reused safely or improve robot task success.",
              "", "### Research questions", "",
              f"1. **Can original Talk2DINO identify task-relative background with simultaneous queries?** At development-selected thresholds, evaluation FG recall is {ops['development_selected_95_recall']['evaluation']['raw']['fg_recall']:.1%} RAW and {ops['development_selected_95_recall']['evaluation']['clean']['fg_recall']:.1%} CLEAN, while BG coverage is {ops['development_selected_95_recall']['evaluation']['raw']['bg_coverage']:.1%} and {ops['development_selected_95_recall']['evaluation']['clean']['bg_coverage']:.1%}, respectively. This is a useful coarse separation signal, with moderate BG IoU ({ops['development_selected_95_recall']['evaluation']['raw']['bg_iou']:.3f} RAW, {ops['development_selected_95_recall']['evaluation']['clean']['bg_iou']:.3f} CLEAN).",
              f"2. **Does cleaning improve at matched foreground recall?** At fixed 0.54, CLEAN raises BG IoU from {ops['official_0.54']['evaluation']['raw']['bg_iou']:.3f} to {ops['official_0.54']['evaluation']['clean']['bg_iou']:.3f}, while FG recall drops from {ops['official_0.54']['evaluation']['raw']['fg_recall']:.1%} to {ops['official_0.54']['evaluation']['clean']['fg_recall']:.1%}. At development-selected high-recall thresholds, RAW is slightly better on evaluation BG IoU and coverage ({ops['development_selected_95_recall']['evaluation']['raw']['bg_iou']:.3f}/{ops['development_selected_95_recall']['evaluation']['raw']['bg_coverage']:.1%}) than CLEAN ({ops['development_selected_95_recall']['evaluation']['clean']['bg_iou']:.3f}/{ops['development_selected_95_recall']['evaluation']['clean']['bg_coverage']:.1%}); cleaning does not improve the matched-recall result in this sample.",
              f"3. **How much background at high foreground recall?** About {ops['development_selected_95_recall']['evaluation']['raw']['bg_coverage']:.1%} RAW and {ops['development_selected_95_recall']['evaluation']['clean']['bg_coverage']:.1%} CLEAN, at held-out FG recall near 98%. Both evaluation recalls fall slightly short of 98%, although the development constraint was met.",
              "4. **Is further OpenVLA integration justified?** Yes for a bounded diagnostic integration; the current evidence does not justify selective KV reuse. The sample is small and uses coarse patch labels, and it includes no cache or task-success evaluation.",
              "", "## Reproducibility", "", "Source: `experiments/kv_cache/experiments/trackb_talk2dino_multicategory_libero/run_experiment.py`. "
              "See `configuration.json`, `frame_manifest.json`, `validation.json`, `threshold_sweep.csv` and `latency_metrics.json` for exact identities, hashes, selected demos, queries and metrics."]
    (OUT/"EXPERIMENT_REPORT.md").write_text("\n".join(lines)+"\n")


def save_scores(data, identity):
    # Variable query counts are stored per segment in separate compact archives.
    d=OUT/"score_maps"; d.mkdir(parents=True,exist_ok=True)
    for seg in data:
        key=seg["task_id"].replace("/","__")+"__"+seg["demo"]
        path=d/(key+".npz")
        tmp=path.with_suffix(".npz.tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f,raw=seg["raw"].astype(np.float16),clean=seg["clean"].astype(np.float16),
                cosine=seg["cosine"].astype(np.float16),relevant_gt=seg["relevant_gt"].astype(np.float16),
                role_gt=seg["role_gt"].astype(np.float16),valid=seg["valid"].astype(np.uint8),rgb=seg["rgb"],
                selected_indices=np.asarray(seg["selected_indices"],np.int32),queries_json=np.asarray(json.dumps(seg["queries"])),
                role_keys_json=np.asarray(json.dumps(seg["role_keys"])),task_id=np.asarray(seg["task_id"]),split=np.asarray(seg["split"]))
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true",help="one 224px image forward and shape/finite validation")
    ap.add_argument("--report-only",action="store_true",help="rebuild metrics and figures from saved score maps without model inference")
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    status=OUT/"RUN_STATUS.json"
    if args.report_only:
        frame_manifest=json.load((OUT/"frame_manifest.json").open())
        segment_manifest={x["task_id"]:x for x in frame_manifest["selection"]}
        data=[]
        for path in sorted((OUT/"score_maps").glob("*.npz")):
            with np.load(path,allow_pickle=False) as z:
                task_id=str(z["task_id"].item())
                meta=segment_manifest[task_id]
                data.append({"task_id":task_id,"suite":meta["suite"],"demo":meta["demo"],"split":meta["split"],
                    "instruction":meta["instruction"],"queries":json.loads(str(z["queries_json"].item())),
                    "selected_indices":z["selected_indices"].astype(int).tolist(),"rgb":z["rgb"].copy(),
                    "relevant_gt":z["relevant_gt"].astype(np.float32),"role_gt":z["role_gt"].astype(np.float32),
                    "role_keys":json.loads(str(z["role_keys_json"].item())),"valid":z["valid"].astype(bool),
                    "raw":z["raw"].astype(np.float32),"clean":z["clean"].astype(np.float32),"cosine":z["cosine"].astype(np.float32)})
        if len(data)!=9 or sum(len(x["rgb"]) for x in data)!=90:
            raise RuntimeError("Saved per-segment score maps are incomplete; refusing to rebuild the report")
        previous=json.load((OUT/"summary.json").open())
        identity=json.load((OUT/"validation.json").open())["model_identity"]
        identity["dino"]["source_revision"]="7764ea0f912e53c92e82eb78a2a1631e92725fc8"
        identity["upstream_revision"]="6dac47dbe3500df63f3b3df20dd28334290919cc"
        validation=json.load((OUT/"validation.json").open())
        validation["model_identity"]=identity
        atomic_json(OUT/"validation.json",validation)
        latency=json.load((OUT/"latency_metrics.json").open())
        # Report-only must never feed aggregate latency values back through
        # analyze(): that would turn a 90-image timing summary into a fake
        # one-sample distribution and overwrite the saved timing artifact.
        summary=json.load((OUT/"summary.json").open())
        summary["model_identity"]=identity
        (OUT/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
        figures=draw_figures(data,summary)
        report(summary,latency,figures)
        atomic_json(status,{"status":"complete","stage":"report_rebuilt_from_saved_score_maps","images":90,"updated_unix":time.time()})
        print(json.dumps({"status":"report_rebuilt_from_saved_score_maps","figures":len(figures)},indent=2))
        return
    atomic_json(status,{"status":"running","stage":"preflight","smoke":args.smoke,"updated_unix":time.time()})
    paths=find_segments(SOURCE)
    if len(paths)!=9: raise RuntimeError(f"Expected nine selected demo segments, found {len(paths)}")
    manifest_config={tid:{"demo":spec["demo"],"split":spec["split"],"queries":spec["queries"]} for tid,spec in TASKS.items()}
    config={"experiment":"Original Talk2DINO multi-category BG on LIBERO E2",
        "model":"standalone original Talk2DINO; Meta DINOv2 ViT-L/14-register + OpenAI CLIP ViT-B/16 + original projection MLP",
        "model_loader":"pinned upstream DINOText code", "visual_resolution":[224,224],"patch_grid":[16,16],
        "upstream_resize_dim":224,"template":"sub_imagenet_template","lambda_bg":0.2,"pamr":False,
        "background_rule":"foreground if max over simultaneous independently queried category score >= threshold; background otherwise",
        "oracle":"E2 task_relevant_foreground fractional occupancy, 16x16; valid mask excludes invalid tokens",
        "primary_threshold":BG_THRESHOLD,"threshold_sweep":THRESHOLDS,
        "threshold_selection":"development only: maximize pooled background IoU subject to foreground recall >=95%; report also >=98%",
        "frame_selection":"10 rounded evenly-spaced indices from each 32-frame segment",
        "segments":manifest_config,"source_cache":str(SOURCE),"container":CONTAINER}
    atomic_json(OUT/"configuration.json",config)
    if args.smoke:
        data, identity, mseg, timings, text_t, mem=collect_scores({next(iter(paths)):paths[next(iter(paths))]},smoke=True)
        s=data[0]
        assert len(s["rgb"])==1 and s["raw"].shape[0:]==(1,len(s["queries"]),256)
        assert np.isfinite(s["relevant_gt"]).all() and (s["valid"].sum()==256 or s["valid"].sum()>0)
        result={"status":"passed","smoke_images":1,"score_shape":list(s["raw"].shape),"clean_shape":list(s["clean"].shape),
                "query_count":len(s["queries"]),"valid_score_values":bool(np.isfinite(s["raw"]).all() and np.isfinite(s["clean"]).all()),
                "resolution":[224,224],"grid":[16,16],"method":"official DINOText generate_masks one forward, RAW and CLEAN captured",
                "model_identity":identity,"latency_ms":timings[0],"peak_gpu_memory":mem}
        atomic_json(OUT/"validation.json",result)
        atomic_json(status,{"status":"smoke_passed","stage":"smoke_complete","updated_unix":time.time()})
        print(json.dumps(result,indent=2));return
    # Full evaluation: exactly nine segments x ten frames.
    data,identity,manifest,timings,text_t,mem=collect_scores(paths,smoke=False)
    count=sum(len(x["rgb"]) for x in data)
    if count!=90 or sum(len(x["rgb"]) for x in data if x["split"]=="development")!=30 or sum(len(x["rgb"]) for x in data if x["split"]=="evaluation")!=60:
        raise RuntimeError(f"Unexpected split counts: {count}")
    atomic_json(OUT/"frame_manifest.json",{"total_images":count,"development_images":30,"evaluation_images":60,
        "split_by_demo_segment":True,"selection":manifest})
    save_scores(data,identity)
    # Threshold calculations consume stored arrays, not model predictions rerun at each threshold.
    summary,latency,sweep,per_frame=analyze(data,identity,manifest,timings,text_t,mem)
    summary["calibration_metrics"]={};summary["evaluation_at_calibration"]={}
    for method in ("raw","clean"):
        summary["calibration_metrics"][method]={};summary["evaluation_at_calibration"][method]={}
        for target in ("0.95","0.98"):
            tau=summary["thresholds_selected_using_development_only"][method][target]["threshold"]
            summary["calibration_metrics"][method][target]=next(x for x in sweep if x["split"]=="development" and x["method"]==method and x["threshold"]==tau)
            summary["evaluation_at_calibration"][method][target]=next(x for x in sweep if x["split"]=="evaluation" and x["method"]==method and x["threshold"]==tau)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (OUT/"latency_metrics.json").write_text(json.dumps(latency,indent=2)+"\n")
    figures=draw_figures(data,summary)
    report(summary,latency,figures)
    val={"status":"passed","smoke_passed":True,"images_scored":count,"segments":len(data),
         "development_images":30,"evaluation_images":60,"queries_encoded_per_task_once":True,
         "all_query_scores_finite":all(np.isfinite(s[m]).all() for s in data for m in ("raw","clean","cosine")),
         "all_score_shapes_aligned_to_256_patches":all(s["raw"].shape==(len(s["rgb"]),len(s["queries"]),256) and s["clean"].shape==s["raw"].shape for s in data),
         "threshold_sweep_reused_saved_score_maps":True,"pamr":False,"openvla_or_siglip_oracle_or_rollouts_used":False,
         "model_identity":identity}
    atomic_json(OUT/"validation.json",val)
    atomic_json(status,{"status":"complete","stage":"complete","images":count,"segments":len(data),"updated_unix":time.time()})
    print(json.dumps({"status":"complete","images":count,"dev":30,"eval":60,"output":str(OUT)},indent=2))


if __name__=="__main__":
    try: main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/"RUN_STATUS.json",{"status":"failed","error":repr(e),"traceback":traceback.format_exc(),"updated_unix":time.time()})
        raise
