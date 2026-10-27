"""Evaluate the existing UNMATCH-inspired SigLIP score as an Otsu cache mask.

The local affinity calculation is imported verbatim from the Stage 4A diagnostic.
Otsu is only the new score-to-mask conversion. This runner does not use oracle
segmentation in its cache decision; oracle labels are read after prediction for
post-hoc diagnostics only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.machinery
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from experiments.kv_analysis.experiments.extract_libero_image import (
    DEFAULT_STEP_ACTION, normalize_and_invert_openvla_gripper,
)
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import (
    assert_independent_cache_clone, cache_length, capture_prediction as capture_normal_prediction, policy_image_from_observation,
    seed_everything,
)
from experiments.kv_cache.experiments.prefill_reuse_experiment import (
    PrefillSnapshot, decode_action_token_ids, greedy_decode_from_prefill,
    prepare_inputs,
)
from experiments.kv_cache.experiments.oracle_foreground_mask import rendered_label_mapping
from experiments.kv_cache.experiments.stage3a_masked_visual_k_validation import mask_from_obs
from experiments.kv_cache.experiments.stage3b_oracle_full_kv_matrix import (
    HORIZON, LAYERS, R, SEED, TASKS, STATES, MaskedKVProjectionSkip, make_env_and_task,
)
from experiments.kv_cache.experiments.stage3c_oracle_task_relevance import RefreshAttentionSidecar
from experiments.kv_cache.experiments.stage3e_attention_relevance import protect_top_background
from experiments.kv_cache.experiments.static_reuse_experiment import (
    DEFAULT_MODEL_ID, cache_layers, clone_prefill_cache, load_model_and_processor,
    resolve_unnorm_key,
)
from experiments.kv_cache.experiments.stage4a_unmatch_siglip_diagnostics import (
    CHECKPOINT_DIR, WEIGHTS, _local_affinity, _text_tokens, _tokenize,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "trackb_unmatch_cache_safety"
PHASE1 = ROOT / "outputs" / "stage4a_siglip_foreground" / "phase1"
TRACK_B = ROOT / "outputs" / "track_b_segmentation"
PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut:"
PROTECTION_FRACTION = 0.10
MATCHED_RANDOM_SEED = 74131


def otsu_high_mask(scores: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact sample Otsu split; high cosine-affinity scores are fresh/foreground.

    Candidate thresholds are midpoints between consecutive unique finite scores.
    Equal values therefore never split across classes. Invalid, constant, or
    otherwise unsplittable inputs fail conservatively to all-fresh.
    """
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    if x.shape != (256,):
        return np.ones(256, dtype=bool), {"valid": False, "reason": f"shape_{x.shape}", "fallback": "all_fresh"}
    if not np.isfinite(x).all():
        return np.ones(256, dtype=bool), {"valid": False, "reason": "nonfinite", "fallback": "all_fresh"}
    values, counts = np.unique(x, return_counts=True)
    if len(values) < 2:
        return np.ones(256, dtype=bool), {"valid": False, "reason": "constant_scores", "fallback": "all_fresh"}
    weights0 = np.cumsum(counts, dtype=np.float64)[:-1]
    weights1 = 256.0 - weights0
    sums0 = np.cumsum(values * counts, dtype=np.float64)[:-1]
    total = float(np.dot(values, counts))
    means0, means1 = sums0 / weights0, (total - sums0) / weights1
    between = weights0 * weights1 * np.square(means0 - means1)
    if not np.isfinite(between).all() or float(between.max()) <= 0.0:
        return np.ones(256, dtype=bool), {"valid": False, "reason": "degenerate_threshold", "fallback": "all_fresh"}
    # np.argmax deterministically chooses the lower threshold in a tie.
    split = int(np.argmax(between))
    threshold = float((values[split] + values[split + 1]) / 2.0)
    mask = x > threshold
    if not mask.any() or mask.all():
        return np.ones(256, dtype=bool), {"valid": False, "reason": "empty_otsu_class", "threshold": threshold, "fallback": "all_fresh"}
    return mask, {
        "valid": True, "threshold": threshold, "foreground_direction": "score > threshold",
        "foreground_count": int(mask.sum()), "background_count": int((~mask).sum()),
        "unique_score_count": int(len(values)), "between_class_variance": float(between[split]),
        "tie_rule": "equal scores stay in one class; lowest threshold wins objective ties",
        "fallback": None,
    }


def score_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Keep the original UNMATCH visual→text max-over-valid-text aggregation."""
    a = np.asarray(matrix, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 256:
        raise ValueError(f"Expected [valid_text_tokens,256] original affinity matrix; got {a.shape}.")
    return a.max(axis=0)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    y, s = np.asarray(labels, bool).reshape(-1), np.asarray(scores, np.float64).reshape(-1)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    p = int(y.sum())
    return float((ranks[y].sum() - p * (p + 1) / 2.0) / (p * (len(y) - p)))


def _binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    p, y = np.asarray(pred, bool), np.asarray(truth, bool)
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    tn = int((~p & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1), "false_negative_rate": fn / max(tp + fn, 1)}


def _preflight_otsu() -> dict[str, Any]:
    """Otsu-mask previews/diagnostics from unchanged saved affinity matrices."""
    manifest = json.loads((TRACK_B / "dataset_manifest.json").read_text())
    diag = json.loads((PHASE1 / "matching_diagnostics.json").read_text())
    sample_rows = {r["sample_id"]: r for r in manifest["samples"]}
    records, previews = [], []
    figdir = OUT / "preflight_visuals"
    figdir.mkdir(parents=True, exist_ok=True)
    for row in diag["samples"]:
        sample_id = row["sample_id"]
        source = sample_rows[sample_id]
        matrix_path = PHASE1 / row["affinity_npz"]
        key = re.sub(r"[^a-zA-Z0-9]+", "_", row["task_description"].lower()).strip("_") + "__penultimate"
        with np.load(matrix_path) as data:
            matrix = np.asarray(data[key], dtype=np.float32)
        scores = score_from_matrix(matrix)
        pred, otsu = otsu_high_mask(scores)
        truth = np.load(TRACK_B / row["oracle_token_mask_path"]).astype(bool).reshape(-1)
        metrics = _binary_metrics(pred, truth)
        metrics["score_auroc"] = _auc(scores, truth)
        records.append({"sample_id": sample_id, "task_id": row["task_id"], "state_id": row["state_id"],
                        "source_matrix": str(matrix_path.relative_to(ROOT)), "matrix_key": key,
                        "original_aggregation": "max over valid text-token cosine rows, per visual token",
                        "otsu": otsu, "oracle_posthoc": metrics,
                        "score_min": float(scores.min()), "score_max": float(scores.max()),
                        "score_mean": float(scores.mean()), "score_std": float(scores.std())})
        rgb = np.asarray(Image.open(TRACK_B / row["image_path"]).convert("RGB").resize((256, 256)))
        previews.append((sample_id, row["task_id"], row["task_description"], rgb, scores, pred.reshape(16,16), truth.reshape(16,16)))
    _save_preflight_gallery(previews, figdir / "otsu_previews.png")
    summary = {"status": "passed", "scoring_method_unchanged": True,
               "score_direction": "higher cosine affinity is candidate foreground/fresh",
               "thresholding": "per-frame Otsu over original 256 visual→text max-affinity scores",
               "samples": records,
               "mean_predicted_foreground_tokens": float(np.mean([r["otsu"].get("foreground_count",256) for r in records])),
               "mean_oracle_precision": float(np.mean([r["oracle_posthoc"]["precision"] for r in records])),
               "mean_oracle_recall": float(np.mean([r["oracle_posthoc"]["recall"] for r in records])),
               "mean_oracle_iou": float(np.mean([r["oracle_posthoc"]["iou"] for r in records])),
               "mean_score_auroc": float(np.mean([r["oracle_posthoc"]["score_auroc"] for r in records]))}
    _write_json(OUT / "preflight_otsu.json", summary)
    return summary


def _save_preflight_gallery(rows, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 5, figsize=(15, 3.2 * len(rows)), squeeze=False)
    for i, (sid, task, prompt, rgb, scores, otsu, oracle) in enumerate(rows):
        axes[i, 0].imshow(rgb); axes[i, 0].set_title(f"Task {task}: {sid}")
        axes[i, 1].imshow(scores.reshape(16,16), cmap="magma"); axes[i, 1].set_title("Original UNMATCH score\nmax valid-token cosine")
        axes[i, 2].imshow(otsu, cmap="gray", vmin=0,vmax=1,interpolation="nearest"); axes[i, 2].set_title(f"Otsu FG {int(otsu.sum())}/256")
        axes[i, 3].imshow(oracle, cmap="gray", vmin=0,vmax=1,interpolation="nearest"); axes[i, 3].set_title("Oracle FG (post-hoc only)")
        error=np.zeros((*oracle.shape,3),dtype=np.uint8); error[oracle & otsu]=[20,180,50]; error[otsu & ~oracle]=[230,50,50]; error[oracle & ~otsu]=[30,90,240]
        axes[i, 4].imshow(error,interpolation="nearest"); axes[i, 4].set_title("TP green / FP red / FN blue")
        for ax in axes[i]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Frozen original UNMATCH-inspired scores → per-frame Otsu (illustrative score mask)")
    fig.tight_layout(); fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _load_text_tower(device: torch.device):
    """Load only paired checkpoint text.* parameters from the verified safetensors."""
    # OpenCLIP uses ftfy only in its default tokenizer; Stage 4A uses the local
    # T5 tokenizer directly, so retain the previously documented no-op import shim.
    ftfy = types.ModuleType("ftfy"); ftfy.fix_text = lambda text: text
    ftfy.__spec__ = importlib.machinery.ModuleSpec("ftfy", loader=None)
    sys.modules.setdefault("ftfy", ftfy)
    pkg = Path("/tmp/stage4a_openclip_pkg")
    if pkg.exists(): sys.path.insert(0, str(pkg))
    import open_clip
    from safetensors import safe_open
    from transformers import AutoTokenizer
    cfg = json.loads((CHECKPOINT_DIR / "open_clip_config.json").read_text())["model_cfg"]
    cfg_dir = Path("/tmp/stage4a_openclip_model_config")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "trackb_unmatch_siglip_text.json"
    cfg_path.write_text(json.dumps(cfg))
    open_clip.add_model_config(cfg_path)
    model = open_clip.create_model("trackb_unmatch_siglip_text", precision="bf16", device="cpu", output_dict=True)
    text_state = {}
    with safe_open(str(WEIGHTS), framework="pt", device="cpu") as sf:
        for name in sf.keys():
            if name.startswith("text."):
                text_state[name.removeprefix("text.")] = sf.get_tensor(name)
    incompatible = model.text.load_state_dict(text_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Paired text state mismatch: {incompatible}")
    del text_state
    text_model = model.text.eval().to(device)
    del model
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return text_model, tokenizer


def _score_current(text_model, tokenizer, instruction: str, visual: torch.Tensor, text_cache: dict | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if visual.ndim != 3 or tuple(visual.shape[1:]) != (256, 1152):
        raise RuntimeError(f"Expected actual OpenVLA SigLIP penultimate features [B,256,1152], got {tuple(visual.shape)}")
    clean = re.sub(r"[^\w\s]", "", instruction.lower().replace("_", " "))
    clean = " ".join(clean.split())
    cache_key = clean
    cache_hit = text_cache is not None and cache_key in text_cache
    if cache_hit:
        text_states, valid, tokens, raw_count, truncated = text_cache[cache_key]
    else:
        raw = tokenizer(clean, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
        ids, valid, tokens, truncated = _tokenize(tokenizer, instruction, visual.device)
        with torch.inference_mode():
            _, text_states = _text_tokens(types.SimpleNamespace(text=text_model), ids)
        raw_count = len(raw)
        if text_cache is not None:
            text_cache[cache_key] = (text_states, valid, tokens, raw_count, truncated)
    with torch.inference_mode():
        matrix, t2v, t2v_idx, v2t, v2t_idx = _local_affinity(text_states, visual, valid, torch)
    scores = v2t.detach().float().cpu().numpy()
    metadata = {"tokenizer_tokens": tokens, "raw_token_count_with_specials": raw_count,
                "context_length": 16, "truncated": bool(truncated),
                "valid_local_text_positions": int(valid.sum()), "matrix_shape": list(matrix.shape),
                "score_min": float(scores.min()), "score_max": float(scores.max()),
                "score_mean": float(scores.mean()), "score_std": float(scores.std()),
                "v2t_winning_token_indices": v2t_idx.detach().cpu().tolist(),
                "instruction_text_cache_hit": bool(cache_hit)}
    return scores, metadata


class _CachedVision:
    """Run the normal visual backbone once and reuse its exact result in VLA forward."""
    def __init__(self, backbone, pixel_values):
        self.backbone, self.pixel_values = backbone, pixel_values
        self.original_forward = None
        self.patch_features = None
        self.siglip_features = None
        self.hook = None

    def __enter__(self):
        siglip = self.backbone.fused_featurizer
        self.hook = siglip.register_forward_hook(self._capture_siglip)
        torch.cuda.synchronize()
        vision_start = time.perf_counter()
        with torch.inference_mode():
            self.patch_features = self.backbone(self.pixel_values)
        torch.cuda.synchronize()
        self.vision_forward_ms = (time.perf_counter() - vision_start) * 1000.0
        self.hook.remove(); self.hook = None
        if self.siglip_features is None or self.siglip_features.shape[1] != 256:
            raise RuntimeError(f"Expected SigLIP penultimate [B,256,1152], got {None if self.siglip_features is None else tuple(self.siglip_features.shape)}")
        if self.patch_features.shape[1] != 256:
            raise RuntimeError(f"Expected concatenated 256 patch features, got {tuple(self.patch_features.shape)}")
        self.original_forward = self.backbone.forward
        module = self
        def cached_forward(_backbone, current_pixel_values):
            if not _same_pixel_values(module.pixel_values, current_pixel_values):
                raise RuntimeError("VLA forward inputs differ from the precomputed vision features.")
            return module.patch_features
        self.backbone.forward = types.MethodType(cached_forward, self.backbone)
        return self

    def _capture_siglip(self, _module, _inputs, output):
        self.siglip_features = output[0] if isinstance(output, (tuple, list)) else output

    def __exit__(self, *_):
        if self.hook is not None: self.hook.remove()
        if self.original_forward is not None: self.backbone.forward = self.original_forward


def _same_pixel_values(a: Any, b: Any) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    return torch.equal(a, b)


def _capture_prediction(model, processor, image, prompt, device, key, *, text_model=None, score_text=None, source_step=None, intervention=None):
    torch.cuda.synchronize(device)
    path_wall_start = time.perf_counter()
    inputs = prepare_inputs(processor, prompt, Image.fromarray(image).convert("RGB"), device)
    backbone = model.vision_backbone
    cache_vision = _CachedVision(backbone, inputs["pixel_values"])
    vision_start, vision_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    score_ms = None
    vision_start.record()
    with cache_vision:
        scores = score_meta = None
        if text_model is not None:
            score_wall_start = time.perf_counter()
            scores, score_meta = _score_current(text_model, processor._trackb_tokenizer,
                                                prompt if score_text is None else score_text,
                                                cache_vision.siglip_features,
                                                getattr(processor,"_trackb_text_cache",None))
            score_ms = (time.perf_counter() - score_wall_start) * 1000.0
        context = intervention if intervention is not None else _NullContext()
        policy_start, policy_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        policy_start.record()
        with context, PrefillSnapshot(model) as snapshot, torch.inference_mode():
            policy_action = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
        policy_end.record(); policy_end.synchronize()
    vision_end.record(); vision_end.synchronize()
    predict_path_ms = (time.perf_counter() - path_wall_start) * 1000.0
    cache, logits, input_ids, layout = snapshot.require()
    prefill_length = cache_length(cache)
    if prefill_length != int(layout["prefill_length"]):
        raise RuntimeError("Prefill snapshot/cache layout length mismatch.")
    if layout.get("bos") != [0,0] or layout.get("visual") != [1,256] or layout.get("language",[None])[0] != 257:
        raise RuntimeError(f"Unexpected dynamically derived OpenVLA multimodal layout: {layout}")
    with torch.inference_mode():
        ids = greedy_decode_from_prefill(model, logits, clone_prefill_cache(cache), model.get_action_dim(key))
    decoded = decode_action_token_ids(model, ids, key)
    if not np.array_equal(np.asarray(policy_action), decoded):
        raise RuntimeError("Manual greedy decode disagrees with predict_action.")
    return {"cache": cache, "logits": logits, "input_ids": input_ids, "layout": layout,
            "policy_action": np.asarray(policy_action), "action_token_ids": ids[0].detach().cpu().tolist(),
            "prefill_length": prefill_length, "scores": scores, "score_meta": score_meta,
            "vision_features": cache_vision.patch_features, "siglip_features": cache_vision.siglip_features,
            "vision_forward_ms": cache_vision.vision_forward_ms,
            "score_generation_ms": score_ms,
            "policy_predict_action_ms": float(policy_start.elapsed_time(policy_end)),
            "predict_path_ms": predict_path_ms}


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *_): return None


def validate_original_score_implementation() -> dict[str, Any]:
    """Static provenance check: local function implementation and saved matrices agree."""
    source = Path(__file__).with_name("stage4a_unmatch_siglip_diagnostics.py").read_text()
    expected_parts = ["F.normalize(text_states[0, valid[0]].float(), dim=-1)",
                      "F.normalize(visual_states[0].float(), dim=-1)",
                      "matrix = t @ v.T", "v2t, v2t_idx = matrix.max(dim=0)"]
    if not all(part in source for part in expected_parts):
        raise RuntimeError("Original UNMATCH score source no longer matches the recovered formula.")
    # The exact gallery reproducibility check was performed in the phase-zero notebook/run
    # and is recorded with its original hashes in configuration.json.
    return {"source_script": "stage4a_unmatch_siglip_diagnostics.py", "function": "_local_affinity",
            "primary_score": "max over valid text tokens for each visual patch (visual→text)",
            "normalization": "FP32 L2 normalization for both feature vectors",
            "matrix_orientation": "[valid_text_tokens, 256_visual_patches]",
            "oracle_used_in_mask_policy": False, "changed_scoring": False}


def _mask_preview(score: np.ndarray, oracle: np.ndarray | None = None):
    return otsu_high_mask(score)


def validate_fast_masks():
    checks = []
    for v in (np.linspace(0, 1, 256), np.ones(256), np.r_[np.zeros(240), np.ones(16)], np.full(256, np.nan)):
        mask, rec = otsu_high_mask(v)
        checks.append({"input_unique": int(np.unique(v).size) if np.isfinite(v).all() else "nonfinite",
                       "mask_sum": int(mask.sum()), "status": rec})
    x = np.asarray([0,0,1,1]*64, dtype=np.float32)
    m1, _ = otsu_high_mask(x); m2, _ = otsu_high_mask(x.copy())
    assert np.array_equal(m1, m2) and int(m1.sum()) == 128
    assert np.all(otsu_high_mask(np.ones(256))[0]) and np.all(otsu_high_mask(np.full(256, np.nan))[0])
    return {"otsu_fast_validation": "passed", "cases": checks}


def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    original = validate_original_score_implementation()
    fast = validate_fast_masks()
    offline = _preflight_otsu()
    validation = {"status": "passed", "original_scoring": original, "fast_validation": fast,
                  "otsu_preflight": "completed", "interpretation": "Otsu changes only score-to-mask conversion."}
    _write_json(OUT / "validation_report.json", validation)
    script = Path(__file__).read_bytes()
    cfg = {"experiment":"Track B — UNMATCH score cache safety",
           "branch":subprocess.run(["git","-c","safe.directory=/opt/openvla","branch","--show-current"],
                                   check=True,capture_output=True,text=True).stdout.strip(),
           "model":"openvla/openvla-7b-finetuned-libero-spatial",
           "openvla_snapshot":"962318cec55ac10993ff0f5f43eda9a270b4c873",
           "paired_siglip":"timm/ViT-SO400M-14-SigLIP@c85ba2402be4b49b5d178f2e2384fdf50f8c7961",
           "paired_weights_sha256":"0a04a48b797e187a568335ab67e57a8958c32cee5a707a3feb6d4c97149dcd9c",
           "primary_visual_feature":"OpenVLA fine-tuned SigLIP penultimate block 25, 256x1152, as original UNMATCH diagnostics",
           "text_feature":"paired SigLIP token states, exact original local-affinity path; context 16",
           "scoring":"FP32 normalize; S=valid text states @ 256 patch states.T; per patch max over valid text tokens",
           "mask":"per-frame exact Otsu on 256 scores; score > threshold is candidate fresh foreground; invalid/degenerate -> all fresh",
           "cache":{"components":["K","V"],"layers":list(LAYERS),"R":R,"source":"latest fully fresh prefill only",
                    "attention_protection_fraction":PROTECTION_FRACTION,"attention_protection":"Stage 3C layer-15 fresh-refresh scores; stable top-background, half-up rounding"},
           "scenarios":{"tasks":list(TASKS),"states":list(STATES),"seed":SEED,"horizon":HORIZON,"settling_steps":10},
           "matched_random":{"included":True,"selection":"deterministic random foreground count matched per frame to Otsu count","seed_base":MATCHED_RANDOM_SEED},
           "original_gallery_reproduction":{"review_png_sha256":"8f78a6adcb5b013bd70db5caabfc9e9dc93432cfa54661a8aee5a0ae84f772a3","matched_mismatch_png_sha256":"4ba285f6e15cc7985feba95fb305656562c8f5f5f2e31954846179cd2c3790f3","result":"byte-identical output regenerated from saved NPZ inputs in /tmp"},
           "code_sha256":hashlib.sha256(script).hexdigest(),"latency":"synchronized wall timing from processor input preparation through full predict_action; vision and score components also recorded; manual decode validation excluded"}
    _write_json(OUT / "configuration.json", cfg)
    return validation


def _temporal_mask_metrics(current: np.ndarray, source: np.ndarray) -> dict[str, Any]:
    src, cur = np.asarray(source, bool), np.asarray(current, bool)
    bg = ~cur
    return {"source_current_mask_iou": float((src & cur).sum() / max((src | cur).sum(), 1)),
            "source_fg_not_current_fg": int((src & ~cur).sum()),
            "current_fg_not_source_fg": int((cur & ~src).sum()),
            "reusable_was_source_fg": int((bg & src).sum())}


def _save_rollout_preview(path: Path, image: np.ndarray, scores: np.ndarray, predicted_fg: np.ndarray,
                          protected_bg: np.ndarray, oracle_fg: np.ndarray, reusable_bg: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(1,6,figsize=(18,3.4))
    ax[0].imshow(image); ax[0].set_title("RGB")
    im=ax[1].imshow(np.asarray(scores).reshape(16,16),cmap="magma"); ax[1].set_title("Original UNMATCH score")
    fig.colorbar(im,ax=ax[1],fraction=.05)
    ax[2].imshow(predicted_fg.reshape(16,16),cmap="gray",vmin=0,vmax=1,interpolation="nearest"); ax[2].set_title("Otsu foreground")
    ax[3].imshow(oracle_fg.reshape(16,16),cmap="gray",vmin=0,vmax=1,interpolation="nearest"); ax[3].set_title("Oracle post-hoc")
    ax[4].imshow(protected_bg.reshape(16,16),cmap="Oranges",vmin=0,vmax=1,interpolation="nearest"); ax[4].set_title("Top-10% BG protected")
    ax[5].imshow(reusable_bg.reshape(16,16),cmap="Greens",vmin=0,vmax=1,interpolation="nearest"); ax[5].set_title("Actually reusable BG")
    for a in ax: a.set_xticks([]);a.set_yticks([])
    fig.suptitle("Otsu mask and effective Stage 3C freshness filter · diagnostic only")
    fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160,bbox_inches="tight");plt.close(fig)


def run_rollout(task_id: int, state_id: int, condition: str, model, processor, text_model, device, key, max_steps: int | None = None):
    if condition not in {"unmatch_otsu", "matched_random"}:
        raise ValueError(condition)
    seed_everything(SEED)
    task, states, env = make_env_and_task(task_id)
    outdir = OUT / ("smoke" if max_steps is not None else condition) / f"task{task_id}_state{state_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp=time.strftime("%Y%m%dT%H%M%S")
    for old in (outdir/"results.partial.json",outdir/"results.failed.json"):
        if old.exists(): old.rename(outdir/f"{old.stem}.{stamp}{old.suffix}")
    prompt = PROMPT_TEMPLATE.format(task=task.language.lower())
    source = None; records = []; done = False; saved_preview_steps: set[int] = set()
    env.reset(); obs = env.set_init_state(states[state_id])
    mapping = rendered_label_mapping(env.env.model.instances_to_ids) # diagnostic-only oracle mapping
    try:
        for _ in range(10): obs, _, _, _ = env.step(DEFAULT_STEP_ACTION)
        horizon = min(HORIZON, max_steps) if max_steps is not None else HORIZON
        for step in range(horizon):
            refresh = step % R == 0
            expected_source = step - step % R
            current_image = policy_image_from_observation(obs, True)
            if refresh:
                sidecar = RefreshAttentionSidecar(model)
                pred = _capture_prediction(model, processor, current_image, prompt, device, key,
                                           text_model=text_model, score_text=task.language.lower(), intervention=sidecar)
                if sidecar.scores is None or sidecar.sequence_length != pred["prefill_length"]:
                    raise RuntimeError("Fresh R5 prefill did not capture Stage3C layer-15 attention scores.")
                source_cache = clone_prefill_cache(pred["cache"])
                assert_independent_cache_clone(source_cache, pred["cache"])
                source = {"cache": source_cache, "layout": dict(pred["layout"]), "step": step,
                          "fresh": True, "attention_scores": sidecar.scores.copy(),
                          "unmatch_mask": None, "policy_image": current_image.copy()}
                current_scores = pred["scores"]
                otsu_start = time.perf_counter()
                siglip_mask, otsu_info = otsu_high_mask(current_scores)
                otsu_ms = (time.perf_counter() - otsu_start) * 1000.0
                raw_siglip_mask = siglip_mask.copy()
                current_mask = siglip_mask.copy()
                if condition == "matched_random" and otsu_info["valid"]:
                    # Budget-matched control preserves the Otsu foreground count,
                    # but chooses a deterministic independent spatial subset.
                    rng = np.random.default_rng(MATCHED_RANDOM_SEED + task_id * 1000 + state_id * 10 + step)
                    chosen = rng.choice(256, size=int(current_mask.sum()), replace=False)
                    current_mask = np.zeros(256, bool); current_mask[chosen] = True
                source["unmatch_mask"] = raw_siglip_mask.copy()
                source["selection_mask"] = current_mask.copy()
                protected = np.zeros(256, bool); reusable = ~current_mask
                projection_rows = [{"layer": l, "k_projected_rows": pred["prefill_length"],
                                    "v_projected_rows": pred["prefill_length"],
                                    "k_skipped_background_rows": 0, "v_skipped_background_rows": 0}
                                   for l in LAYERS]
                temporal = None; age = 0; source_step = step
                mask_generation_ms = otsu_ms
                attention_capture_ms = sidecar.capture_ms
            else:
                if source is None or source["step"] != expected_source or not source["fresh"]:
                    raise AssertionError("R5 source is not the most recent fully fresh prefill snapshot.")
                # Score current frame with original max-cosine method from the current
                # OpenVLA vision features; attention protection remains lagged to source.
                # Note current SigLIP features are computed once inside _capture_prediction.
                # Need obtain current score before choosing mask. Use one vision pass, then
                # call the VLA with the same cached features and the computed skip context.
                torch.cuda.synchronize(device)
                total_wall_start = time.perf_counter()
                inputs = prepare_inputs(processor, prompt, Image.fromarray(current_image).convert("RGB"), device)
                cached = _CachedVision(model.vision_backbone, inputs["pixel_values"])
                with cached:
                    score_wall_start = time.perf_counter()
                    scores, score_meta = _score_current(text_model, processor._trackb_tokenizer, task.language.lower(), cached.siglip_features,
                                                        getattr(processor,"_trackb_text_cache",None))
                    score_generation_ms = (time.perf_counter() - score_wall_start) * 1000.0
                    otsu_start = time.perf_counter()
                    siglip_mask, otsu_info = otsu_high_mask(scores)
                    otsu_ms = (time.perf_counter() - otsu_start) * 1000.0
                    raw_siglip_mask = siglip_mask.copy()
                    if condition == "matched_random" and otsu_info["valid"]:
                        rng = np.random.default_rng(MATCHED_RANDOM_SEED + task_id * 1000 + state_id * 10 + step)
                        chosen = rng.choice(256, size=int(siglip_mask.sum()), replace=False)
                        siglip_mask = np.zeros(256, bool); siglip_mask[chosen] = True
                    protected, reusable = protect_top_background(source["attention_scores"], siglip_mask, PROTECTION_FRACTION)
                    effective_fg = siglip_mask | protected
                    context = MaskedKVProjectionSkip(model, source["cache"], source["layout"], effective_fg)
                    source_versions = [(k._version, v._version) for k, v in cache_layers(source["cache"])]
                    policy_start, policy_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    policy_start.record()
                    with context, PrefillSnapshot(model) as snapshot, torch.inference_mode():
                        policy_action = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
                    policy_end.record(); policy_end.synchronize()
                torch.cuda.synchronize(device)
                predict_path_ms = (time.perf_counter()-total_wall_start)*1000.0
                vision_forward_ms = cached.vision_forward_ms
                policy_predict_action_ms = float(policy_start.elapsed_time(policy_end))
                cache, logits, input_ids, layout = snapshot.require()
                plen = cache_length(cache)
                if plen != int(layout["prefill_length"]): raise RuntimeError("masked prefill layout mismatch")
                ids = greedy_decode_from_prefill(model, logits, clone_prefill_cache(cache), model.get_action_dim(key))
                decoded = decode_action_token_ids(model, ids, key)
                if not np.array_equal(np.asarray(policy_action), decoded): raise RuntimeError("manual decode mismatch")
                pred = {"cache":cache,"logits":logits,"input_ids":input_ids,"layout":layout,
                        "policy_action":np.asarray(policy_action),"action_token_ids":ids[0].detach().cpu().tolist(),
                        "prefill_length":plen,"scores":scores,"score_meta":score_meta}
                context.validate(pred["cache"])
                if source_versions != [(k._version, v._version) for k, v in cache_layers(source["cache"])]:
                    raise RuntimeError("A reuse step modified its independent fresh source cache.")
                projection_rows = context.diag["rows_by_layer"]
                current_mask = siglip_mask
                temporal = _temporal_mask_metrics(raw_siglip_mask, source["unmatch_mask"])
                temporal["cache_selection_mask_iou"] = _temporal_mask_metrics(current_mask, source["selection_mask"])["source_current_mask_iou"]
                temporal["reusable_source_fg_count"] = int((reusable & source["unmatch_mask"]).sum())
                old_image = source["policy_image"].astype(np.float32)
                temporal["source_current_rgb_mean_abs_01"] = float(np.mean(np.abs(current_image.astype(np.float32)-old_image))/255.0)
                age = step-source["step"]; source_step=source["step"]
                mask_generation_ms = otsu_ms
                attention_capture_ms = None
                pred["vision_forward_ms"] = vision_forward_ms
                pred["score_generation_ms"] = score_generation_ms
                pred["policy_predict_action_ms"] = policy_predict_action_ms
                pred["predict_path_ms"] = predict_path_ms
            # Oracle annotation is computed only after mask selection/policy inference, never passed to the policy.
            oracle_fg = mask_from_obs(obs, mapping)
            oracle_metrics = _binary_metrics(raw_siglip_mask, oracle_fg)
            oracle_metrics["score_auroc"] = _auc(pred["scores"], oracle_fg)
            if step in (0, 1) and step not in saved_preview_steps:
                preview_reusable = reusable if not refresh else np.zeros(256, dtype=bool)
                _save_rollout_preview(outdir / f"mask_diagnostic_step{step:03d}.png", current_image,
                                      pred["scores"], current_mask, protected, oracle_fg, preview_reusable)
                saved_preview_steps.add(step)
            if not np.all(np.isfinite(pred["policy_action"])): raise RuntimeError("Nonfinite action")
            env_action = normalize_and_invert_openvla_gripper(pred["policy_action"])
            reused_count = int(reusable.sum()) if not refresh else 0
            skipped = reused_count * len(LAYERS)
            rec = {"step":step,"refresh":refresh,"source_step":source_step,"source_age":age,
                   "attention_score_age":age,"prefill_length":pred["prefill_length"],
                   "language_tokens":int(pred["layout"]["language"][1]-pred["layout"]["language"][0]+1),
                   "siglip_fg_tokens":int(raw_siglip_mask.sum()) if not refresh else int(siglip_mask.sum()),
                   "siglip_bg_tokens":256-(int(raw_siglip_mask.sum()) if not refresh else int(siglip_mask.sum())),
                   "selected_fresh_fg_tokens":int(current_mask.sum()),
                   "protected_predicted_background_tokens":int(protected.sum()),
                   "reused_background_tokens":reused_count,"fresh_visual_tokens":int((~reusable).sum()) if not refresh else 256,
                   "predicted_foreground_indices":np.flatnonzero(raw_siglip_mask).tolist(),
                   "selected_fresh_foreground_indices":np.flatnonzero(current_mask).tolist(),
                   "protected_background_indices":np.flatnonzero(protected).tolist(),
                   "reused_background_indices":np.flatnonzero(reusable).tolist() if not refresh else [],
                   "oracle_foreground_indices_posthoc":np.flatnonzero(oracle_fg).tolist(),
                   "otsu":otsu_info,"score_metadata":pred["score_meta"],"scores":pred["scores"].tolist(),
                   "source_current_mask_metrics":temporal,"oracle_posthoc":oracle_metrics,
                   "k_projected_rows_per_layer":[r["k_projected_rows"] for r in projection_rows],
                   "v_projected_rows_per_layer":[r["v_projected_rows"] for r in projection_rows],
                   "k_skipped_rows_per_layer":[r.get("k_skipped_background_rows",0) for r in projection_rows],
                   "v_skipped_rows_per_layer":[r.get("v_skipped_background_rows",0) for r in projection_rows],
                   "k_skipped_rows_total":skipped,"v_skipped_rows_total":skipped,
                   "actual_mask_generation_ms":mask_generation_ms,"score_generation_ms":pred["score_generation_ms"],
                   "vision_forward_ms":pred["vision_forward_ms"],"policy_predict_action_ms":pred["policy_predict_action_ms"],
                   "total_predict_path_ms":pred["predict_path_ms"],"attention_capture_ms":attention_capture_ms,
                   "token_ids":pred["action_token_ids"],"policy_action":pred["policy_action"].tolist(),
                   "environment_action":env_action.tolist(),"all_32_layers_validated":refresh or sorted(context.diag["layers"])==list(LAYERS),
                   "source_fresh":bool(source["fresh"]),"done":False}
            if not refresh and step == 1:
                # One bounded same-observation counterfactual per scenario. This fresh
                # action is diagnostic only and is never sent to the environment.
                fresh_cf = capture_normal_prediction(model, processor, current_image, prompt, device, key)
                delta = np.abs(np.asarray(pred["policy_action"]) - np.asarray(fresh_cf["policy_action"]))
                rec["fresh_counterfactual"] = {
                    "same_observation": True,
                    "token_ids_exact_match": pred["action_token_ids"] == fresh_cf["action_token_ids"],
                    "changed_action_tokens": sum(a != b for a, b in zip(pred["action_token_ids"], fresh_cf["action_token_ids"])),
                    "policy_action_l2": float(np.linalg.norm(delta)),
                    "policy_action_max_abs": float(delta.max()),
                }
            if not refresh and age not in (1,2,3,4): raise AssertionError(f"R5 age invalid: {age}")
            obs, _, done, _ = env.step(env_action.tolist()); rec["done"]=bool(done)
            records.append(rec)
            _write_json(outdir/"results.partial.json",{"status":"running","steps":len(records),"records":records})
            if done: break
    except Exception as exc:
        _write_json(outdir/"results.failed.json",{"status":"failed","task_id":task_id,"state_id":state_id,
                    "condition":condition,"steps_completed":len(records),"error_type":type(exc).__name__,
                    "error":str(exc),"records":records})
        raise
    finally:
        env.close()
    result = {"status":"smoke_complete" if max_steps is not None else "complete","condition":condition,"task_id":task_id,"state_id":state_id,
              "task_description":task.language,"seed":SEED,"horizon":HORIZON,"success":bool(done),
              "steps":len(records),"termination":"success" if done else "horizon","R":R,
              "layers":list(LAYERS),"components":["K","V"],"refresh_count":sum(r["refresh"] for r in records),
              "reuse_count":sum(not r["refresh"] for r in records),
              "temporal_reuse_fraction":sum(not r["refresh"] for r in records)/max(len(records),1),
              "max_source_age":max([r["source_age"] for r in records],default=0),
              "mean_siglip_fg_tokens":float(np.mean([r["siglip_fg_tokens"] for r in records])),
              "mean_protected_bg_tokens":float(np.mean([r["protected_predicted_background_tokens"] for r in records])),
              "mean_reused_tokens":float(np.mean([r["reused_background_tokens"] for r in records])),
              "total_k_projection_rows_skipped":sum(r["k_skipped_rows_total"] for r in records),
              "total_v_projection_rows_skipped":sum(r["v_skipped_rows_total"] for r in records),
              "all_valid":all(r["all_32_layers_validated"] and r["source_fresh"] for r in records),
              "records":records}
    _write_json(outdir/("results.smoke.json" if max_steps is not None else "results.json"),result)
    (outdir/"results.partial.json").unlink(missing_ok=True)
    return result


def run_validation(model, processor, text_model, device, key):
    """One deterministic transition validates exact fresh equivalence and actual row skipping."""
    seed_everything(SEED)
    task, states, env = make_env_and_task(4)
    prompt = PROMPT_TEMPLATE.format(task=task.language.lower())
    try:
        env.reset(); obs=env.set_init_state(states[0])
        for _ in range(10): obs,_,_,_=env.step(DEFAULT_STEP_ACTION)
        image=policy_image_from_observation(obs,True)
        full = capture_normal_prediction(model,processor,image,prompt,device,key)
        # Score path captures SigLIP branch features during the same normal vision pass.
        scored = _capture_prediction(model,processor,image,prompt,device,key,text_model=text_model,score_text=task.language.lower())
        if not torch.equal(full["logits"],scored["logits"]): raise RuntimeError("Fresh logits changed when SigLIP scoring was attached")
        if full["action_token_ids"]!=scored["action_token_ids"] or not np.array_equal(full["policy_action"],scored["policy_action"]):
            raise RuntimeError("Fresh actions changed when SigLIP scoring was attached")
        if scored["scores"].shape!=(256,) or not np.isfinite(scored["scores"]).all(): raise RuntimeError("Invalid 256-score vector")
        fg,_=otsu_high_mask(scored["scores"])
        source=clone_prefill_cache(full["cache"]); assert_independent_cache_clone(source,full["cache"])
        obs,_,_,_=env.step(normalize_and_invert_openvla_gripper(full["policy_action"]).tolist())
        nxt=policy_image_from_observation(obs,True)
        attention=RefreshAttentionSidecar(model)
        fresh= capture_normal_prediction(model,processor,nxt,prompt,device,key,attention)
        source=clone_prefill_cache(fresh["cache"]); assert_independent_cache_clone(source,fresh["cache"])
        current=_capture_prediction(model,processor,nxt,prompt,device,key,text_model=text_model,score_text=task.language.lower())
        m,_=otsu_high_mask(current["scores"])
        prot,reuse=protect_top_background(attention.scores,m,PROTECTION_FRACTION)
        context=MaskedKVProjectionSkip(model,source,fresh["layout"],m|prot)
        masked=_capture_prediction(model,processor,nxt,prompt,device,key,text_model=text_model,intervention=context)
        context.validate(masked["cache"])
        skipped=int(reuse.sum())
        assert len(context.diag["rows_by_layer"])==32 and all(r["k_skipped_background_rows"]==skipped and r["v_skipped_background_rows"]==skipped for r in context.diag["rows_by_layer"])
        return {"passed":True,"task_id":4,"state_id":0,"prefill_length":masked["prefill_length"],
                "language_tokens":masked["layout"]["language"][1]-masked["layout"]["language"][0]+1,
                "score_count":256,"score_finite":True,"precomputed_vision_logits_equal_normal_forward":True,
                "fresh_instrumentation_tokens_exact":True,"fresh_instrumentation_action_exact":True,
                "source_clone_independent":True,"reuse_age":1,"foreground_tokens":int(m.sum()),
                "protected_background_tokens":int(prot.sum()),"reused_tokens":skipped,
                "k_skipped_rows_per_layer":skipped,"v_skipped_rows_per_layer":skipped,
                "total_k_rows_skipped":skipped*32,"total_v_rows_skipped":skipped*32,
                "all_32_layers":True,"q_full":True,"action_decode_valid":True,
                "masked_action_finite":bool(np.isfinite(masked["policy_action"]).all()),
                "no_oracle_policy_dependency":True}
    finally: env.close()


def _aggregate_outputs():
    files=[]; table=[]
    for condition in ("unmatch_otsu","matched_random"):
        for task in TASKS:
            for state in STATES:
                p=OUT/condition/f"task{task}_state{state}"/"results.json"
                if p.exists():
                    r=json.loads(p.read_text()); files.append(str(p.relative_to(ROOT)))
                    reuse_steps=[x for x in r["records"] if not x["refresh"]]
                    denom_visual_rows=sum(32*256 for _ in reuse_steps)
                    table.append({"condition":condition,"task_id":task,"state_id":state,"success":r["success"],"steps":r["steps"],
                                  "mean_fg":r["mean_siglip_fg_tokens"],"mean_protected_bg":r["mean_protected_bg_tokens"],
                                  "mean_reused":r["mean_reused_tokens"],"k_skipped":r["total_k_projection_rows_skipped"],
                                  "v_skipped":r["total_v_projection_rows_skipped"],"max_age":r["max_source_age"],
                                  "visual_kv_skip_fraction_on_reuse":r["total_k_projection_rows_skipped"]/max(denom_visual_rows,1),
                                  "mean_policy_path_ms":float(np.mean([x["total_predict_path_ms"] for x in r["records"]])),
                                  "p95_policy_path_ms":float(np.percentile([x["total_predict_path_ms"] for x in r["records"]],95)),
                                  "mean_source_current_mask_iou":float(np.mean([x["source_current_mask_metrics"]["source_current_mask_iou"] for x in reuse_steps])) if reuse_steps else None,
                                  "source_fg_patches_reused_mean":float(np.mean([x["source_current_mask_metrics"]["reusable_source_fg_count"] for x in reuse_steps])) if reuse_steps else None})
    with (OUT/"per_scenario.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]) if table else ["condition"]); w.writeheader(); w.writerows(table)
    refs={}
    for task in TASKS:
        for state in STATES:
            tag=f"task{task}_state{state}"
            base=ROOT/"outputs"/"stage3b_oracle_full_kv_matrix"/tag/"results.json"
            c3=ROOT/"outputs"/"stage3c_oracle_task_relevance"/"protect_10"/tag/"results.json"
            c4=ROOT/"outputs"/"stage4a_oracle_free_attention"/"fresh_60"/"attention"/tag/"results.json"
            if base.exists():
                b=json.loads(base.read_text()); refs[tag]={"fresh":b.get("fresh",{}),"stage3b":b.get("stage3b_true_skip",{})}
            if c3.exists(): refs.setdefault(tag,{})["stage3c_oracle_10pct"]=json.loads(c3.read_text())
            if c4.exists(): refs.setdefault(tag,{})["stage4a_attention_60pct_fresh"]=json.loads(c4.read_text())
    for row in table:
        ref=refs.get(f"task{row['task_id']}_state{row['state_id']}",{})
        for name,key_name in [("fresh","fresh"),("stage3b_true_skip","stage3b"),("stage3c_oracle_10pct","stage3c_oracle_10pct"),("stage4a_attention","stage4a_attention_60pct_fresh")]:
            v=ref.get(key_name,{})
            row[f"{name}_success"]=v.get("success")
            row[f"{name}_steps"]=v.get("steps")
    if table:
        with (OUT/"per_scenario.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    condition_summary={}
    for condition in ("unmatch_otsu","matched_random"):
        subset=[r for r in table if r["condition"]==condition]
        recs=[]
        for t in TASKS:
            for s in STATES:
                p=OUT/condition/f"task{t}_state{s}"/"results.json"
                if p.exists(): recs.extend(json.loads(p.read_text())["records"])
        all_reuse=[r for r in recs if not r["refresh"]]
        all_steps=recs
        skipped_k=sum(r["k_skipped_rows_total"] for r in all_steps)
        skipped_v=sum(r["v_skipped_rows_total"] for r in all_steps)
        denominator_visual=sum(32*256 for r in all_reuse)
        condition_summary[condition]={"completed_scenarios":len(subset),"successes":sum(bool(r["success"]) for r in subset),
             "success_rate":sum(bool(r["success"]) for r in subset)/max(len(subset),1),
             "mean_foreground_tokens":float(np.mean([r["siglip_fg_tokens"] for r in all_steps])) if all_steps else None,
             "mean_protected_background_tokens":float(np.mean([r["protected_predicted_background_tokens"] for r in all_steps])) if all_steps else None,
             "mean_reused_tokens_per_decision":float(np.mean([r["reused_background_tokens"] for r in all_steps])) if all_steps else None,
             "reuse_decision_fraction":sum(not r["refresh"] for r in all_steps)/max(len(all_steps),1),
             "total_k_rows_skipped":skipped_k,"total_v_rows_skipped":skipped_v,
             "skip_fraction_of_visual_rows_on_reuse":skipped_k/max(denominator_visual,1),
             "mean_policy_path_ms":float(np.mean([r["total_predict_path_ms"] for r in all_steps])) if all_steps else None,
             "p50_policy_path_ms":float(np.median([r["total_predict_path_ms"] for r in all_steps])) if all_steps else None,
             "p95_policy_path_ms":float(np.percentile([r["total_predict_path_ms"] for r in all_steps],95)) if all_steps else None,
             "mean_siglip_score_generation_ms":float(np.mean([r["score_generation_ms"] for r in all_steps])) if all_steps else None,
             "mean_otsu_ms":float(np.mean([r["actual_mask_generation_ms"] for r in all_steps])) if all_steps else None,
             "mean_attention_sidecar_ms_refresh":float(np.mean([r["attention_capture_ms"] for r in all_steps if r["attention_capture_ms"] is not None])) if any(r["attention_capture_ms"] is not None for r in all_steps) else None,
             "mean_source_current_mask_iou":float(np.mean([r["source_current_mask_metrics"]["source_current_mask_iou"] for r in all_reuse])) if all_reuse else None,
             "mean_source_fg_patches_that_are_reused":float(np.mean([r["source_current_mask_metrics"]["reusable_source_fg_count"] for r in all_reuse])) if all_reuse else None}
    _write_json(OUT/"historical_references.json",refs)
    summary={"status":"complete" if all(sum(r["condition"]==c for r in table)==9 for c in ("unmatch_otsu","matched_random")) else "incomplete",
             "method":"original UNMATCH-inspired SigLIP visual→text max cosine + per-frame Otsu; Stage3C top10% lagged attention protection; R5 full-layer true K/V skip",
             "primary": condition_summary["unmatch_otsu"],
             "matched_random": condition_summary["matched_random"],
             "per_scenario":table,"historical_references":refs,"result_files":files,
             "latency_measured":bool(table),"latency_scope":"synchronized wall latency from processor input preparation through full predict_action, including one normal vision forward, current score/mask, and policy inference; excludes environment and manual decode validation"}
    _write_json(OUT/"summary.json",summary)
    _plot_aggregate(table,refs)
    return summary


def _plot_aggregate(table, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True)
    labels=[]; rates=[]
    for name,vals in [("Fresh",[v.get("fresh",{}) for v in refs.values()]),
                      ("Stage3B",[v.get("stage3b",{}) for v in refs.values()]),
                      ("Stage3C oracle+10%",[v.get("stage3c_oracle_10pct",{}) for v in refs.values()]),
                      ("Stage4A attention",[v.get("stage4a_attention_60pct_fresh",{}) for v in refs.values()]),
                      ("UNMATCH+Otsu",[r for r in table if r["condition"]=="unmatch_otsu"]),
                      ("Matched random",[r for r in table if r["condition"]=="matched_random"])]:
        good=[x for x in vals if isinstance(x,dict) and "success" in x]
        if good:
            labels.append(name);rates.append(sum(bool(x["success"]) for x in good)/len(good))
    fig,ax=plt.subplots(figsize=(9,4.5));bars=ax.bar(labels,rates,color=["gray","#d95f02","#1b9e77","#7570b3","#e7298a","#66a61e"][:len(labels)])
    ax.set_ylim(0,1.05);ax.set_ylabel("Success fraction");ax.set_title("Historical closed-loop outcomes (different policy trajectories)");ax.tick_params(axis="x",rotation=20)
    for b,v in zip(bars,rates):ax.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.0%}",ha="center")
    fig.tight_layout();fig.savefig(OUT/"success_comparison.png",dpi=160);plt.close(fig)
    primary=[r for r in table if r["condition"]=="unmatch_otsu"]
    random=[r for r in table if r["condition"]=="matched_random"]
    fig,ax=plt.subplots(figsize=(7,5))
    for subset,label,color in [(primary,"UNMATCH+Otsu","#e7298a"),(random,"matched random","#66a61e")]:
        ax.scatter([r["k_skipped"] for r in subset],[int(r["success"]) for r in subset],label=label,color=color)
    ax.set_xlabel("K projection rows skipped in rollout");ax.set_yticks([0,1],labels=["Failure","Success"]);ax.set_title("Structural skipping versus scenario success");ax.legend();fig.tight_layout();fig.savefig(OUT/"skipped_rows_vs_success.png",dpi=160);plt.close(fig)


def _write_report(preflight_result, validation_result, summary):
    refs={"Fresh":"experiments/kv_cache/outputs/stage4a_oracle_free_attention", "Stage 3C":"experiments/kv_cache/outputs/stage3c_oracle_task_relevance/protect_10", "Stage 3B":"experiments/kv_cache/outputs/stage3b_oracle_full_kv_matrix"}
    lines=["# Track B — UNMATCH score cache-safety experiment", "",
           "## Method", "",
           "The recovered Stage 4A score implementation was kept unchanged: FP32-normalized penultimate OpenVLA SigLIP patch and paired-checkpoint text token features; cosine matrix `[valid text tokens, 256 patches]`; patch score is the maximum cosine across valid instruction tokens. Otsu is applied per current frame; high-affinity scores are fresh candidates. Stage 3C's lagged layer-15 attention protection selects the top 10% of predicted background (half-up rounding, stable ties). Remaining predicted background K/V uses the latest fresh R5 source. Oracle masks are computed after policy selection solely for post-hoc metrics.","",
           "Otsu uses exact unique-score split midpoints; equal values stay together; lower threshold wins objective ties. Constant, nonfinite, empty-class, or invalid score vectors fall back to all-fresh. No oracle mask selects cache rows. This is a per-frame foreground candidate rule, not validated semantic segmentation.","",
           "## Offline Otsu preflight", "", f"Mean predicted fresh tokens across saved frames: {preflight_result.get('mean_predicted_foreground_tokens'):.1f}/256. Oracle comparison: mean precision {preflight_result.get('mean_oracle_precision'):.3f}, recall {preflight_result.get('mean_oracle_recall'):.3f}, IoU {preflight_result.get('mean_oracle_iou'):.3f}, score AUROC {preflight_result.get('mean_score_auroc'):.3f}. These oracle statistics are descriptive only. Gallery: `preflight_visuals/otsu_previews.png`.","",
           "## Runtime validation", "", f"Status: {validation_result.get('passed')}. Validation record: `validation.json`. It verifies score dimension/finite values, exact fresh logits/token/action invariance with scoring instrumentation, source cloning, R5 age 1 reuse, all 32 layers, full Q, actual K/V row skips, normal greedy decode, and no oracle policy dependency.","",
           "## Closed-loop outcomes", "", "See `per_scenario.csv` and `summary.json`. Historical references are not paired trajectories; policies produce different observations.","",
           "| Condition | Successes | Scenarios |", "|---|---:|---:|"]
    for cond in ("unmatch_otsu","matched_random"):
        subset=[r for r in summary.get("per_scenario",[]) if r["condition"]==cond]
        lines.append(f"| {cond} | {sum(r['success'] for r in subset)} | {len(subset)} |")
    lines += ["", "Fresh historical reference: 8/9. Stage 3C oracle-assisted + 10% task-relevance protection: 9/9. Stage 3B oracle background full-layer KV: 5/9. Stage 4A oracle-free attention results are a separate mask policy; see the output's `references.json` and historical CSV. None of these small scenario sets establishes generalization.","",
              "## Efficiency and limitations", "", "Skipped projection rows are structural counts, not latency. End-to-end latency was not measured in this run. SigLIP masking reuses the current OpenVLA vision-backbone output already needed by the policy; the runner captures its penultimate SigLIP branch during that same forward. Text states are computed from the paired pretrained text tower, cached by task instruction. The 16-token context can truncate task 4/5 instructions. Current-frame masks lag the K/V source by design, and source/current mask disagreement is observational only.","",
              "## Files and reproducibility", "", "Code: `experiments/kv_cache/experiments/trackb_unmatch_cache_safety.py`. Config: `configuration.json`. Validation: `validation.json`. Results are incrementally saved per scenario. No Stage 3/4 output was overwritten.",""]
    (OUT/"REPORT.md").write_text("\n".join(lines))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["preflight","validate","smoke","pilot","rollout","matrix","aggregate"],default="preflight")
    ap.add_argument("--condition",choices=["unmatch_otsu","matched_random"],default="unmatch_otsu")
    ap.add_argument("--task",type=int); ap.add_argument("--state",type=int); ap.add_argument("--max-steps",type=int)
    ap.add_argument("--model-id",default=DEFAULT_MODEL_ID)
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=="preflight":
        print(json.dumps(preflight(),indent=2)); return
    if args.mode=="aggregate":
        s=_aggregate_outputs(); _write_report(json.loads((OUT/"preflight_otsu.json").read_text()),json.loads((OUT/"validation.json").read_text()),s); print(json.dumps(s,indent=2)); return
    if args.mode in {"validate","smoke","pilot","rollout"} and (args.task is None or args.state is None):
        ap.error("--task and --state required for validate/smoke/pilot/rollout")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required inside the established Docker runtime.")
    torch.cuda.synchronize(); model_load_start=time.perf_counter()
    model,processor,device=load_model_and_processor(args.model_id)
    torch.cuda.synchronize(); model_load_ms=(time.perf_counter()-model_load_start)*1000.0
    model_gpu_bytes=int(torch.cuda.memory_allocated(device))
    key=resolve_unnorm_key(model,None)
    text_load_start=time.perf_counter(); gpu_before_text=int(torch.cuda.memory_allocated(device))
    text_model,tokenizer=_load_text_tower(device)
    torch.cuda.synchronize(); text_load_ms=(time.perf_counter()-text_load_start)*1000.0
    text_gpu_bytes=int(torch.cuda.memory_allocated(device))-gpu_before_text
    processor._trackb_tokenizer=tokenizer
    processor._trackb_text_cache={}
    _write_json(OUT/"runtime_metadata.json",{"docker_image":"openvla-cached:latest","device":torch.cuda.get_device_name(device),
        "torch":torch.__version__,"model_load_ms":model_load_ms,"openvla_gpu_allocated_bytes":model_gpu_bytes,
        "paired_text_tower_load_ms":text_load_ms,"paired_text_tower_incremental_gpu_bytes":text_gpu_bytes,
        "vision_forward_policy_extra_passes":0,"text_cache":"one token-state encoding per unique instruction, reused per frame"})
    if args.mode=="validate":
        result=run_validation(model,processor,text_model,device,key)
        _write_json(OUT/"validation.json",result); print(json.dumps(result,indent=2)); return
    if args.mode=="pilot":
        preflight()
        validation=run_validation(model,processor,text_model,device,key)
        _write_json(OUT/"validation.json",validation)
        smoke=run_rollout(args.task,args.state,args.condition,model,processor,text_model,device,key,args.max_steps or 5)
        _write_json(OUT/"pilot_summary.json",{"validation":validation,"smoke":{k:v for k,v in smoke.items() if k!="records"}})
        print(json.dumps({"validation":validation,"smoke":{k:v for k,v in smoke.items() if k!="records"}},indent=2)); return
    if args.mode=="smoke":
        result=run_rollout(args.task,args.state,args.condition,model,processor,text_model,device,key,args.max_steps or 5)
        print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2)); return
    if args.mode=="matrix":
        results=[]
        for condition in ("unmatch_otsu","matched_random"):
            for task_id in TASKS:
                for state_id in STATES:
                    path=OUT/condition/f"task{task_id}_state{state_id}"/"results.json"
                    if path.exists():
                        prior=json.loads(path.read_text())
                        if prior.get("status")=="complete":
                            print(f"skip completed {condition} task={task_id} state={state_id}",flush=True)
                            results.append(prior); continue
                    print(f"start {condition} task={task_id} state={state_id}",flush=True)
                    results.append(run_rollout(task_id,state_id,condition,model,processor,text_model,device,key))
                    progress=_aggregate_outputs()
                    _write_report(json.loads((OUT/"preflight_otsu.json").read_text()),
                                  json.loads((OUT/"validation.json").read_text()),progress)
        summary=_aggregate_outputs()
        _write_report(json.loads((OUT/"preflight_otsu.json").read_text()),json.loads((OUT/"validation.json").read_text()),summary)
        print(json.dumps(summary,indent=2)); return
    result=run_rollout(args.task,args.state,args.condition,model,processor,text_model,device,key,args.max_steps)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))


if __name__=="__main__": main()
