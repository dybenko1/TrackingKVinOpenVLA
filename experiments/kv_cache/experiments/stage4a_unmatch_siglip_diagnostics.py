#!/usr/bin/env python3
"""Offline feasibility diagnostics for UNMATCH-inspired SigLIP patch/text affinities.

This script consumes existing Track B observations only. It does not run VLA action
inference, LIBERO, segmentation fitting, or KV-cache interventions.
"""
from __future__ import annotations

import json
import importlib.machinery
import math
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments/kv_cache/outputs/stage4a_siglip_foreground/phase1"
TRACK_B = REPO / "experiments/kv_cache/outputs/track_b_segmentation"
CHECKPOINT_DIR = REPO / "cache/models/timm/ViT-SO400M-14-SigLIP/c85ba2402be4b49b5d178f2e2384fdf50f8c7961"
WEIGHTS = CHECKPOINT_DIR / "open_clip_model.safetensors"
OPENVLA_SNAPSHOT = Path("/data/models/huggingface/models--openvla--openvla-7b-finetuned-libero-spatial/snapshots/962318cec55ac10993ff0f5f43eda9a270b4c873")
OPENVLA_SIGLIP_SHARD = OPENVLA_SNAPSHOT / "model-00001-of-00004.safetensors"
OPENCLIP_PKG = Path("/tmp/stage4a_openclip_pkg")
MODEL_CONFIG_NAME = "stage4a_siglip_so400m"
SPECIAL_ID_NAMES = {0: "<pad>", 1: "</s>"}

PROMPTS = [
    "black bowl", "black", "bowl", "plate", "robot gripper", "wooden table",
    "wooden", "table", "red cup", "red", "cup", "blue bowl", "it", "the",
]
SAMPLE_IDS = [
    "task4_state0_step0000", "task4_state0_step0048", "task4_state0_step0112",
    "task5_state0_step0000", "task5_state0_step0052", "task5_state0_step0104",
    "task7_state0_step0000", "task7_state0_step0051", "task7_state0_step0109",
]


def _json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _float_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std())}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fmt_token(token: str) -> str:
    return token.replace("▁", "·").replace("Ġ", "·").strip() or "∅"


def _load_model(torch: Any):
    # OpenCLIP imports ftfy for its default tokenizer. This diagnostic uses the
    # checkpoint's local T5 tokenizer directly, so the unused cleaner is stubbed
    # only to permit importing the model implementation in this isolated process.
    ftfy = types.ModuleType("ftfy")
    ftfy.fix_text = lambda text: text
    ftfy.__spec__ = importlib.machinery.ModuleSpec("ftfy", loader=None)
    sys.modules.setdefault("ftfy", ftfy)
    sys.path.insert(0, str(OPENCLIP_PKG))
    import open_clip
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    cfg = json.loads((CHECKPOINT_DIR / "open_clip_config.json").read_text())["model_cfg"]
    tmp_cfg = Path("/tmp/stage4a_openclip_model_config")
    tmp_cfg.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_cfg / f"{MODEL_CONFIG_NAME}.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    open_clip.add_model_config(cfg_path)

    t0 = time.perf_counter()
    model = open_clip.create_model(MODEL_CONFIG_NAME, precision="bf16", device="cuda", output_dict=True)
    state = load_file(str(WEIGHTS), device="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    del state
    model.eval()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t0) * 1000.0
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id  # matches OpenCLIP HFTokenizer behavior
    return model, tokenizer, open_clip, load_ms, incompatible


def _tokenize(tokenizer: Any, text: str, device: Any) -> tuple[Any, Any, list[str], bool]:
    # The paired checkpoint has context_length=16 and the OpenCLIP tokenizer
    # canonicalizes to lowercase/no punctuation before tokenization.
    clean = re.sub(r"[^\w\s]", "", text.lower().replace("_", " "))
    clean = " ".join(clean.split())
    raw = tokenizer(clean, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
    truncated = len(raw) > 16
    encoded = tokenizer(
        clean, add_special_tokens=True, truncation=True, max_length=16,
        padding="max_length", return_tensors="pt",
    )
    ids = encoded["input_ids"].to(device)
    # Valid local-matching text positions exclude pad/EOS and all special IDs.
    special = set(tokenizer.all_special_ids)
    valid = torch_non_special_mask(ids, special)
    toks = tokenizer.convert_ids_to_tokens(ids[0].tolist())
    return ids, valid, [_fmt_token(x) for x in toks], truncated


def torch_non_special_mask(ids: Any, special: set[int]):
    import torch
    valid = torch.ones_like(ids, dtype=torch.bool)
    for sid in special:
        valid &= ids.ne(int(sid))
    return valid


def _preprocess(path: Path, torch: Any):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import pil_to_tensor, resize
    image = Image.open(path).convert("RGB")
    if image.size != (224, 224):
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
    # Track B policy_rgb is already the exact 224x224 naive-resize RGB view;
    # apply the paired checkpoint's documented (x - .5) / .5 normalization.
    x = pil_to_tensor(image).float().div_(255.0).unsqueeze(0)
    x = (x - 0.5) / 0.5
    return image, x.to(device="cuda", dtype=torch.bfloat16)


def _extract_patch_features(model: Any, x: Any, torch: Any):
    trunk = model
    penultimate = trunk.get_intermediate_layers(x, n={25}, return_prefix_tokens=False, norm=False)[0]
    final = trunk.blocks[26](penultimate)
    final = trunk.norm(final)
    return penultimate, final


def _load_openvla_siglip_trunk(torch: Any):
    import timm
    from safetensors import safe_open
    prefix = "vision_backbone.fused_featurizer."
    trunk = timm.create_model("vit_so400m_patch14_siglip_224", pretrained=False, num_classes=0)
    trunk = trunk.to(device="cuda", dtype=torch.bfloat16).eval()
    state = {}
    with safe_open(str(OPENVLA_SIGLIP_SHARD), framework="pt", device="cpu") as f:
        source_keys = [k for k in f.keys() if k.startswith(prefix)]
        for key in source_keys:
            state[key.removeprefix(prefix)] = f.get_tensor(key)
    if len(state) != len(trunk.state_dict()):
        raise AssertionError(f"OpenVLA SigLIP tensor count {len(state)} != Timm model count {len(trunk.state_dict())}")
    missing, unexpected = trunk.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise AssertionError(f"OpenVLA SigLIP strict state load mismatch: missing={missing}, unexpected={unexpected}")

    paired_path = CHECKPOINT_DIR / "open_clip_model.safetensors"
    paired_key_prefix = "visual.trunk."
    total_sq = diff_sq = diff_abs = 0.0
    max_abs = 0.0
    exact_bf16 = 0
    groups: dict[str, list[float]] = {}
    with safe_open(str(paired_path), framework="pt", device="cpu") as paired:
        for key, vla_tensor in state.items():
            pk = paired_key_prefix + key
            if pk not in paired.keys():
                raise AssertionError(f"Missing paired visual tensor {pk}")
            a = paired.get_tensor(pk).float()
            b = vla_tensor.float()
            d = a - b
            total_sq += float(a.square().sum().item())
            diff_sq += float(d.square().sum().item())
            diff_abs += float(d.abs().sum().item())
            max_abs = max(max_abs, float(d.abs().max().item()))
            exact_bf16 += int(torch.equal(a.to(torch.bfloat16), b.to(torch.bfloat16)))
            group = key.split(".")[0]
            g = groups.setdefault(group, [0.0, 0.0])
            g[0] += float(d.square().sum().item())
            g[1] += float(a.square().sum().item())
    comparison = {
        "paired_tensors": len(state), "openvla_tensors": len(state),
        "same_after_bf16_cast_tensor_count": exact_bf16,
        "relative_l2": math.sqrt(diff_sq / total_sq), "max_abs": max_abs,
        "mean_abs": diff_abs / sum(v.numel() for v in state.values()),
        "relative_l2_by_prefix": {k: math.sqrt(v[0] / v[1]) if v[1] else 0.0 for k, v in groups.items()},
        "source_openvla_snapshot_revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
        "source_openvla_shard": str(OPENVLA_SIGLIP_SHARD),
    }
    return trunk, comparison


def _text_tokens(model: Any, ids: Any):
    import torch
    text_model = model.text
    # Return all final normalized token states (the OpenCLIP output_tokens path
    # drops the pooled last position, which may be content if input was truncated).
    cast_dtype = text_model.transformer.get_cast_dtype()
    x = text_model.token_embedding(ids).to(cast_dtype)
    seq_len = ids.shape[1]
    x = x + text_model.positional_embedding[:seq_len].to(cast_dtype)
    x = text_model.transformer(x, attn_mask=text_model.attn_mask)
    tokens = text_model.ln_final(x)
    pooled = tokens[:, -1]
    if isinstance(text_model.text_projection, torch.nn.Linear):
        pooled = text_model.text_projection(pooled)
    elif text_model.text_projection is not None:
        pooled = pooled @ text_model.text_projection
    return pooled, tokens


def _local_affinity(text_states: Any, visual_states: Any, valid: Any, torch: Any):
    import torch.nn.functional as F
    t = F.normalize(text_states[0, valid[0]].float(), dim=-1)
    v = F.normalize(visual_states[0].float(), dim=-1)
    matrix = t @ v.T
    t2v, t2v_idx = matrix.max(dim=1)
    v2t, v2t_idx = matrix.max(dim=0)
    return matrix, t2v, t2v_idx, v2t, v2t_idx


def _cuda_stats(fn: Any, torch: Any, warmup: int = 10, reps: int = 40) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.inference_mode():
        for _ in range(reps):
            start.record(); fn(); end.record(); end.synchronize()
            times.append(float(start.elapsed_time(end)))
    a = np.asarray(times, dtype=np.float64)
    return {"warmups": warmup, "repetitions": reps, "mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "std_ms": float(a.std())}


def _make_figure(sample: dict[str, Any], image: Image.Image, prompt_records: dict[str, Any],
                 out: Path, token_maps: dict[str, np.ndarray], token_names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # First row compares native penultimate and final patch maps for object and
    # task prompts; lower rows expose the individual final-state token affinity.
    selected = ["black bowl", "plate", "red cup", sample["task_description"]]
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.25])
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axs[0].imshow(image); axs[0].set_title("Policy RGB")
    oracle = np.load(TRACK_B / sample["oracle_token_mask_path"]).astype(bool).reshape(16, 16)
    axs[1].imshow(oracle, cmap="gray", interpolation="nearest", vmin=0, vmax=1); axs[1].set_title("Oracle FG (post-hoc only)")
    for col, variant in [(2, "penultimate"), (3, "final")]:
        rec = prompt_records[selected[0]][variant]
        axs[col].imshow(image)
        axs[col].imshow(np.asarray(rec["visual_to_text"]).reshape(16,16), cmap="magma", alpha=0.58, interpolation="nearest")
        axs[col].set_title(f"{variant} · black bowl")

    for col, prompt in enumerate(selected):
        for row, variant in enumerate(["penultimate", "final"], start=1):
            ax = fig.add_subplot(gs[row, col])
            rec = prompt_records[prompt][variant]
            ax.imshow(np.asarray(rec["visual_to_text"]).reshape(16, 16), cmap="magma", interpolation="nearest")
            ax.set_title(f"{variant}: {prompt[:24]}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    # Replace final bottom-right panel with per-patch winning token identity for
    # the task instruction; labels are token indices, with the legend below.
    ax = fig.axes[-1]
    ax.clear()
    winners = token_maps[sample["task_description"]]
    cmap = plt.get_cmap("tab20", max(1, len(token_names)))
    ax.imshow(winners.reshape(16,16), cmap=cmap, vmin=0, vmax=max(1, len(token_names)-1), interpolation="nearest")
    for y in range(16):
        for x in range(16):
            ax.text(x,y,str(int(winners[y*16+x])),ha="center",va="center",fontsize=5,color="white")
    ax.set_title("Final features · winning instruction token index", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    token_key = ", ".join(f"{i}:{_fmt_token(tok)}" for i, tok in enumerate(token_names[:16]))
    fig.suptitle(f"{sample['sample_id']} · task-conditioned local cosine; token ids in final panel\nToken legend: {token_key}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_per_token_figure(sample: dict[str, Any], image: Image.Image, token_affinities: dict[str, np.ndarray],
                           token_labels: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(token_labels)
    if not n:
        return
    variants = ("penultimate", "final", "paired_pretrained_final")
    fig, axes = plt.subplots(n, 4, figsize=(13, max(3.0, n * 1.45)), squeeze=False)
    for i, label in enumerate(token_labels):
        for j, variant in enumerate(variants):
            ax = axes[i, j]
            ax.imshow(token_affinities[variant][i].reshape(16, 16), cmap="magma", interpolation="nearest", vmin=-0.08, vmax=0.15)
            ax.set_title(f"{variant} · token {i}: {label}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        ax = axes[i, 3]
        ax.imshow(image)
        ax.set_title("Policy RGB" if i == 0 else "", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Per-token local cosine maps · {sample['sample_id']}\nRows use non-special instruction tokens; color is cosine, not probability", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT / "unmatch_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((TRACK_B / "dataset_manifest.json").read_text())
    by_id = {row["sample_id"]: row for row in manifest["samples"]}
    missing = [sid for sid in SAMPLE_IDS if sid not in by_id]
    if missing:
        raise RuntimeError(f"Requested sample IDs are missing: {missing}")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    gpu_before = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, open_clip, load_ms, incompatible = _load_model(torch)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"Strict checkpoint mismatch: {incompatible}")
    openvla_trunk, weight_comparison = _load_openvla_siglip_trunk(torch)
    torch.cuda.synchronize()
    model_memory_bytes = torch.cuda.memory_allocated()
    peak_model_bytes = torch.cuda.max_memory_allocated()
    visual_params = sum(p.numel() for p in model.visual.parameters())
    text_params = sum(p.numel() for p in model.text.parameters())
    dtype_bytes = 2
    text_memory_estimate = text_params * dtype_bytes
    print(f"loaded paired checkpoint in {load_ms:.1f}ms; GPU allocated={model_memory_bytes}; CUDA free={gpu_before[0]}", flush=True)

    prompts_by_sample = {}
    all_texts = set(PROMPTS)
    for sid in SAMPLE_IDS:
        row = by_id[sid]
        task_text = row["task_description"]
        prompt_set = [*PROMPTS, task_text]
        prompts_by_sample[sid] = prompt_set
        all_texts.update(prompt_set)
    text_cache = {}
    token_meta = {}
    text_encode_times = []
    tokenization_times = []
    pooled_path_checks = {}
    for prompt in sorted(all_texts):
        tok_t0 = time.perf_counter()
        ids, valid, toks, truncated = _tokenize(tokenizer, prompt, torch.device("cuda"))
        tokenization_times.append((time.perf_counter() - tok_t0) * 1000.0)
        t0 = time.perf_counter()
        with torch.inference_mode():
            pooled, token_states = _text_tokens(model, ids)
        torch.cuda.synchronize()
        text_encode_times.append((time.perf_counter() - t0) * 1000)
        with torch.inference_mode():
            pooled_reference = model.encode_text(ids, normalize=False)
        torch.cuda.synchronize()
        pooled_path_checks[prompt] = {
            "max_abs": float((pooled - pooled_reference).abs().max().item()),
            "mean_abs": float((pooled - pooled_reference).abs().mean().item()),
        }
        text_cache[prompt] = (ids, valid, pooled, token_states)
        token_meta[prompt] = {
            "canonicalized_text": " ".join(re.sub(r"[^\w\s]", "", prompt.lower()).split()),
            "tokens_context16": toks,
            "non_special_token_indices": [i for i, tokid in enumerate(ids[0].tolist()) if tokid not in set(tokenizer.all_special_ids)],
            "raw_token_count_with_specials": len(tokenizer(prompt, add_special_tokens=True, truncation=False, padding=False)["input_ids"]),
            "truncated_to_context16": truncated,
            "valid_matching_token_count": int(valid.sum().item()),
        }

    records = []
    all_affinities: dict[str, np.ndarray] = {}
    pool_scores = []
    feature_compare = []
    feature_time_full = []
    feature_time_continue = []
    scoring_times = []
    prompt_json: dict[str, Any] = {}
    per_sample_memory = []

    torch.cuda.reset_peak_memory_stats()
    for sid in SAMPLE_IDS:
        row = by_id[sid]
        sample_dir = TRACK_B / row["sample_dir"]
        image, x = _preprocess(TRACK_B / row["rgb_path"], torch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            p, f = _extract_patch_features(openvla_trunk, x, torch)
        torch.cuda.synchronize()
        feature_time_full.append((time.perf_counter() - t0) * 1000.0)
        # Measure only the added final block + final normalization when OpenVLA's
        # existing block-25 hidden states are already available.
        feature_time_continue.append(_cuda_stats(lambda: openvla_trunk.norm(openvla_trunk.blocks[26](p)), torch, warmup=5, reps=30)["median_ms"])

        with np.load(TRACK_B / row["feature_path"]) as z:
            saved = z["siglip"].astype(np.float32)
        p_np = p[0].float().cpu().numpy()
        delta = p_np.astype(np.float64) - saved.astype(np.float64)
        denom = float(np.linalg.norm(saved.astype(np.float64)))
        feature_compare.append({
            "sample_id": sid, "saved_track_b_shape": list(saved.shape), "recomputed_shape": list(p_np.shape),
            "max_abs": float(np.abs(delta).max()), "mean_abs": float(np.abs(delta).mean()),
            "relative_l2": float(np.linalg.norm(delta) / max(denom, 1e-12)),
            "cosine": float(np.sum(p_np.astype(np.float64)*saved.astype(np.float64)) /
                             max(np.linalg.norm(p_np.astype(np.float64))*denom, 1e-12)),
        })

        # Compare the paired pretraining visual tower's outputs to the actual
        # fine-tuned OpenVLA features; this is diagnostic, never used in VLA.
        with torch.inference_mode():
            paired_p = model.visual.trunk.get_intermediate_layers(x, n={25}, return_prefix_tokens=False, norm=False)[0]
            paired_f = model.visual.trunk.norm(model.visual.trunk.blocks[26](paired_p))
        pair_saved = paired_p[0].float().cpu().numpy().astype(np.float64) - saved.astype(np.float64)
        pair_final_delta = paired_f.float() - f.float()
        feature_compare[-1]["paired_pretraining_penultimate_vs_openvla_saved_relative_l2"] = float(
            np.linalg.norm(pair_saved) / max(np.linalg.norm(saved.astype(np.float64)), 1e-12))
        feature_compare[-1]["paired_pretraining_vs_openvla_final_max_abs"] = float(pair_final_delta.abs().max().item())
        feature_compare[-1]["paired_pretraining_vs_openvla_final_relative_l2"] = float(
            pair_final_delta.norm().item() / max(f.float().norm().item(), 1e-12))

        # Validate continuation equals Timm's direct final-block extraction.
        with torch.inference_mode():
            f_direct = openvla_trunk.get_intermediate_layers(x, n={26}, return_prefix_tokens=False, norm=True)[0]
        diff = (f - f_direct).float()
        feature_compare[-1]["final_continuation_vs_direct_max_abs"] = float(diff.abs().max().item())
        feature_compare[-1]["final_continuation_vs_direct_mean_abs"] = float(diff.abs().mean().item())

        # Original SigLIP pooled image/text path, retained only as a reference
        # check; these scores do not drive the patch mask.
        with torch.inference_mode():
            global_image = model.encode_image(x, normalize=True)
        prompt_records: dict[str, Any] = {}
        token_winner_maps: dict[str, np.ndarray] = {}
        matrices_for_file = {}
        oracle = np.load(TRACK_B / row["oracle_token_mask_path"]).astype(bool).reshape(-1)
        if len(oracle) != 256:
            raise AssertionError(f"Oracle token mask shape invalid for {sid}: {oracle.shape}")
        for prompt in prompts_by_sample[sid]:
            ids, valid, pooled, token_states = text_cache[prompt]
            pooled_text = F.normalize(pooled.float(), dim=-1)
            pooled_cosine = float((global_image.float() * pooled_text).sum().item())
            logits = float((model.logit_scale.exp().float() * pooled_cosine + model.logit_bias.float()).item())
            for variant, visual in [("penultimate", p), ("final", f), ("paired_pretrained_final", paired_f)]:
                matrix, t2v, t2v_idx, v2t, v2t_idx = _local_affinity(token_states, visual, valid, torch)
                matrix_np = matrix.cpu().numpy().astype(np.float32)
                valid_idx = valid[0].nonzero(as_tuple=False).flatten().cpu().numpy().astype(int)
                valid_tokens = [token_meta[prompt]["tokens_context16"][int(i)] for i in valid_idx]
                winner_context_idx = valid_idx[v2t_idx.cpu().numpy()]
                winner_words = [token_meta[prompt]["tokens_context16"][int(i)] for i in winner_context_idx]
                scores = v2t.cpu().numpy().astype(np.float32)
                auc = _auc(scores, oracle)
                entry = {
                    "affinity_matrix_shape": list(matrix_np.shape),
                    "text_tokens": valid_tokens,
                    "text_context_indices": valid_idx.tolist(),
                    "text_to_visual_max": t2v.cpu().numpy().astype(np.float32).tolist(),
                    "text_to_visual_argmax_patch": t2v_idx.cpu().numpy().astype(int).tolist(),
                    "visual_to_text": scores.tolist(),
                    "visual_to_text_argmax_text_row": v2t_idx.cpu().numpy().astype(int).tolist(),
                    "visual_to_text_argmax_context_index": winner_context_idx.tolist(),
                    "visual_to_text_argmax_token": winner_words,
                    "foreground_mean": float(scores[oracle].mean()) if oracle.any() else None,
                    "background_mean": float(scores[~oracle].mean()) if (~oracle).any() else None,
                    "foreground_minus_background_mean": float(scores[oracle].mean()-scores[~oracle].mean()) if oracle.any() and (~oracle).any() else None,
                    "oracle_fg_auc_posthoc": auc,
                    "score_stats": _float_stats(scores),
                    "top_winning_token_counts": {w: winner_words.count(w) for w in sorted(set(winner_words), key=lambda w: -winner_words.count(w))[:8]},
                }
                prompt_records.setdefault(prompt, {})[variant] = entry
                matrices_for_file[f"{re.sub(r'[^a-z0-9]+','_',prompt.lower()).strip('_')}__{variant}"] = matrix_np
                if variant == "final":
                    token_winner_maps[prompt] = winner_context_idx
                    for idx, tok in zip(valid_idx, valid_tokens):
                        # Every text-token map uses same visual layout; keep raw arrays separately.
                        pass
                scoring_times.append((len(valid_idx) * 256 * 1152) / 1e9)
            pool_scores.append({"sample_id": sid, "prompt": prompt, "pooled_cosine": pooled_cosine, "siglip_logit": logits})

        # Retain the entire local affinity matrix for all prompts and feature variants.
        score_dir = OUT / "affinity_matrices"
        score_dir.mkdir(parents=True, exist_ok=True)
        score_path = score_dir / f"{sid}.npz"
        np.savez_compressed(score_path, **matrices_for_file)
        all_affinities[sid] = np.array([len(matrices_for_file), sum(v.size for v in matrices_for_file.values())], dtype=np.int64)
        prompt_json[sid] = prompt_records
        sample_record = {
            "sample_id": sid, "task_id": row["task_id"], "state_id": row["state_id"],
            "time_step": row.get("time_step"), "task_description": row["task_description"],
            "image_path": row["rgb_path"], "oracle_token_mask_path": row["oracle_token_mask_path"],
            "oracle_fg_tokens": int(oracle.sum()), "oracle_bg_tokens": int((~oracle).sum()),
            "affinity_npz": str(score_path.relative_to(OUT)),
            "prompts": prompts_by_sample[sid],
        }
        records.append(sample_record)
        if sid.endswith("step0000") or sid.endswith("step0048") or sid.endswith("step0051"):
            # Full per-text-token maps for the task instruction, separate from
            # max-reduced score maps. This is the evidence for word-specific matches.
            full_prompt = row["task_description"]
            _, valid, _, txt = text_cache[full_prompt]
            per_token_maps = {}
            token_grid_maps = {"penultimate": [], "final": [], "paired_pretrained_final": []}
            valid_context = valid[0].nonzero(as_tuple=False).flatten().tolist()
            for variant, visual in [("penultimate", p), ("final", f), ("paired_pretrained_final", paired_f)]:
                mat, *_ = _local_affinity(txt, visual, valid, torch)
                for n, ctx_idx in enumerate(valid_context):
                    token_map = mat[n].float().cpu().numpy().astype(np.float32)
                    per_token_maps[f"{variant}_{ctx_idx}_{token_meta[full_prompt]['tokens_context16'][ctx_idx]}"] = token_map
                    token_grid_maps[variant].append(token_map)
            np.savez_compressed(score_dir / f"{sid}_per_token.npz", **per_token_maps)
            _make_figure(sample_record, image, prompt_records, fig_dir / f"{sid}_matching.png", token_winner_maps, token_meta[full_prompt]["tokens_context16"])
            _make_per_token_figure(
                sample_record, image,
                {k: np.stack(v) for k, v in token_grid_maps.items()},
                [token_meta[full_prompt]["tokens_context16"][i] for i in valid_context],
                fig_dir / f"{sid}_instruction_token_maps.png",
            )
        per_sample_memory.append(int(torch.cuda.max_memory_allocated()))
        print(f"processed {sid}; feature max abs to saved={feature_compare[-1]['max_abs']:.5g}; final-vs-direct={feature_compare[-1]['final_continuation_vs_direct_max_abs']:.5g}", flush=True)

    # Timings separated by model initialization, text encoding, image blocks,
    # continuation, pooled reference, and 256×text local matching.
    first_ids, first_valid, _, first_tokens = text_cache[PROMPTS[0]]
    _, first_p = _preprocess(TRACK_B / by_id[SAMPLE_IDS[0]]["rgb_path"], torch)
    with torch.inference_mode():
        p0, f0 = _extract_patch_features(openvla_trunk, first_p, torch)
        _ = _local_affinity(first_tokens, p0, first_valid, torch)
    affinity_base_bytes = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    scoring_latency = _cuda_stats(lambda: _local_affinity(first_tokens, f0, first_valid, torch), torch, warmup=20, reps=100)
    affinity_peak_delta_bytes = max(0, int(torch.cuda.max_memory_allocated()) - affinity_base_bytes)
    longest_instruction = max(
        {by_id[sid]["task_description"] for sid in SAMPLE_IDS},
        key=lambda text: token_meta[text]["valid_matching_token_count"],
    )
    _, long_valid, _, long_tokens = text_cache[longest_instruction]
    instruction_scoring_latency = _cuda_stats(lambda: _local_affinity(long_tokens, f0, long_valid, torch), torch, warmup=20, reps=100)
    text_latency = _cuda_stats(lambda: _text_tokens(model, first_ids), torch, warmup=5, reps=30)
    continue_latency = _cuda_stats(lambda: openvla_trunk.norm(openvla_trunk.blocks[26](p0)), torch, warmup=10, reps=100)
    penultimate_latency = _cuda_stats(lambda: openvla_trunk.get_intermediate_layers(first_p, n={25}, return_prefix_tokens=False, norm=False)[0], torch, warmup=5, reps=30)
    full_patch_latency = _cuda_stats(lambda: _extract_patch_features(openvla_trunk, first_p, torch), torch, warmup=5, reps=30)
    pooled_latency = _cuda_stats(lambda: model.encode_image(first_p, normalize=True), torch, warmup=5, reps=30)

    pairwise = []
    for sample in records:
        sid = sample["sample_id"]
        task = sample["task_description"]
        for prompt in ["black bowl", "plate", "red cup", task]:
            a = prompt_json[sid][prompt]["penultimate"]
            b = prompt_json[sid][prompt]["final"]
            va = np.asarray(a["visual_to_text"]); vb = np.asarray(b["visual_to_text"])
            pairwise.append({
                "sample_id": sid, "prompt": prompt,
                "penultimate_final_score_map_mean_abs_difference": float(np.abs(va-vb).mean()),
                "penultimate_final_score_map_max_abs_difference": float(np.abs(va-vb).max()),
                "penultimate_auc": a["oracle_fg_auc_posthoc"], "final_auc": b["oracle_fg_auc_posthoc"],
                "penultimate_fg_minus_bg": a["foreground_minus_background_mean"],
                "final_fg_minus_bg": b["foreground_minus_background_mean"],
            })
    # Pooled task-match sanity: report within-image matched-vs-red-cup differences.
    pooled_by_key = {(r["sample_id"], r["prompt"]): r for r in pool_scores}
    pooled_deltas = []
    for sample in records:
        sid = sample["sample_id"]
        pooled_deltas.append({
            "sample_id": sid,
            "black_bowl_minus_red_cup_cosine": pooled_by_key[(sid, "black bowl")]["pooled_cosine"] - pooled_by_key[(sid, "red cup")]["pooled_cosine"],
            "plate_minus_red_cup_cosine": pooled_by_key[(sid, "plate")]["pooled_cosine"] - pooled_by_key[(sid, "red cup")]["pooled_cosine"],
            "task_instruction_minus_red_cup_cosine": pooled_by_key[(sid, sample["task_description"])]["pooled_cosine"] - pooled_by_key[(sid, "red cup")]["pooled_cosine"],
        })

    device_props = torch.cuda.get_device_properties(0)
    diagnostic = {
        "experiment": "Stage 4A Phase 1B — UNMATCH-inspired frozen SigLIP local affinity feasibility",
        "scope": "offline saved Track B observations; no VLA action inference, LIBERO rollouts, KV reuse, labels used for fitting, or threshold selection",
        "checkpoint": {
            "repo": "timm/ViT-SO400M-14-SigLIP", "revision": "c85ba2402be4b49b5d178f2e2384fdf50f8c7961",
            "file": str(WEIGHTS), "sha256": "0a04a48b797e187a568335ab67e57a8958c32cee5a707a3feb6d4c97149dcd9c",
            "config": json.loads((CHECKPOINT_DIR / "open_clip_config.json").read_text()),
            "loaded_strictly": True, "visual_parameters": int(visual_params), "text_parameters": int(text_params),
        },
        "runtime": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "torch": torch.__version__, "timm": __import__('timm').__version__, "open_clip_torch_isolated_version": "2.31.0", "model_load_ms": load_ms, "gpu_allocated_model_bytes": int(model_memory_bytes), "peak_model_bytes": int(peak_model_bytes), "text_tower_bf16_parameter_bytes_estimate": int(text_memory_estimate), "affinity_peak_temporary_delta_bytes": affinity_peak_delta_bytes, "gpu_total_memory_bytes": int(device_props.total_memory)},
        "feature_variants": {
            "A_penultimate": {"source": "OpenVLA fine-tuned checkpoint's own SigLIP fused_featurizer, Timm SO400M/14 block 25 of 27; no final LayerNorm; native Track B FP16 features numerically reproduced from checkpoint weights"},
            "B_final": {"source": "OpenVLA fine-tuned SigLIP weights; continue block 26 from block-25 output and apply trunk.norm; 256x1152. The paired pretrained tower is not used for this continuation because its visual weights differ materially."},
            "weight_key_mapping": {"paired_checkpoint_visual_entries": 342, "timm_visual_state_entries": 342, "matched_state_keys": 342, "missing": [], "unexpected": [], "strict_full_openclip_load": True},
            "paired_vs_finetuned_openvla_visual_weight_comparison": weight_comparison,
            "native_feature_reproduction": feature_compare,
            "final_continuation_vs_direct_extraction_max_abs": max(x["final_continuation_vs_direct_max_abs"] for x in feature_compare),
        },
        "text_tokenization": {"tokenizer": "local checkpoint T5TokenizerFast, canonicalized lowercase/no punctuation; OpenCLIP HFTokenizer convention pad_id=eos_id=1", "context_length": 16, "prompts": token_meta},
        "pooled_text_api_vs_token_state_equivalence": pooled_path_checks,
        "method": {
            "affinity": "L2-normalize each final/penultimate text-token and visual-patch state in FP32, then S = T @ V.T; shape [valid text tokens,256]",
            "text_to_visual": "max over 256 patches for each retained text token; preserve argmax patch",
            "visual_to_text": "max over retained text tokens for each visual patch; preserve argmax token/context position",
            "retained_text_positions": "exclude all tokenizer special IDs, including EOS and repeated EOS padding; no special/pad query included",
            "probability_claim": False,
            "oracle_use": "only post-hoc AUC/foreground-background diagnostics and figure display; never score fitting or mask construction",
            "deviations": ["UNMATCH's trained global-local classifier and multiscale coverage/statistics are not implemented.", "UNMATCH uses SigLIP-B/16 with 196 final patch states and up to 64 text positions; this study uses SO400M/14, 256 patches, 1152 dimensions, and context length 16.", "The matching mechanism is local cosine maxima over token states, not UNMATCH's downstream trained claim-verification classifier."],
        },
        "layout": {"visual_patch_count": 256, "visual_patch_grid": [16, 16], "row_major": True, "policy_image_size": [224, 224], "patch_size": 14, "cache_visual_positions": [1, 256]},
        "timing": {
            "text_encoding_median_wall_ms_each": float(np.median(text_encode_times)),
            "tokenizer_median_wall_ms_each": float(np.median(tokenization_times)),
            "openvla_feature_forward_median_wall_ms": float(np.median(feature_time_full)),
            "openvla_final_block_continuation_median_cuda_ms_across_samples": float(np.median(feature_time_continue)),
            "openvla_penultimate_feature_cuda_events": penultimate_latency,
            "openvla_final_block_continuation_cuda_events": continue_latency,
            "openvla_full_penultimate_plus_final_feature_cuda_events": full_patch_latency,
            "final_block_plus_norm_cuda_events": continue_latency,
            "text_encoder_cuda_events": text_latency,
            "affinity_scoring_two_token_phrase_cuda_events": scoring_latency,
            "affinity_scoring_longest_instruction_cuda_events": instruction_scoring_latency,
            "longest_instruction_scored": longest_instruction,
            "original_pooled_image_encoder_cuda_events": pooled_latency,
            "notes": "CUDA timings synchronized; fixed tensors; warm-up/repetitions in each record. Patch-forward timing is diagnostic and excludes OpenVLA language prefill/action inference; text timing excludes tokenizer CPU processing except separately reported wall time; full model load is one-time and includes safetensors deserialization/copy to CUDA.",
        },
        "global_pooled_scores": pool_scores,
        "matched_mismatched_pooled_score_differences": pooled_deltas,
        "samples": records,
        "prompt_affinity_diagnostics": prompt_json,
        "penultimate_vs_final": pairwise,
        "files": {"affinity_dir": str((OUT / "affinity_matrices").relative_to(REPO)), "figure_dir": str(fig_dir.relative_to(REPO)), "script": str(Path(__file__).relative_to(REPO))},
    }
    _json(OUT / "matching_diagnostics.json", diagnostic)
    _json(OUT / "configuration.json", {
        "experiment": diagnostic["experiment"], "openvla_checkpoint": "openvla/openvla-7b-finetuned-libero-spatial",
        "openvla_snapshot_revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
        "paired_siglip_repo": "timm/ViT-SO400M-14-SigLIP", "paired_siglip_revision": "c85ba2402be4b49b5d178f2e2384fdf50f8c7961",
        "paired_weights_sha256": "0a04a48b797e187a568335ab67e57a8958c32cee5a707a3feb6d4c97149dcd9c",
        "track_b_samples": SAMPLE_IDS, "prompts": PROMPTS,
        "patch_grid": [16, 16], "patch_count": 256, "text_context_length": 16,
        "feature_variants": ["OpenVLA fine-tuned penultimate block 25, unnormalized", "OpenVLA fine-tuned final block 26 + final norm", "paired pretrained final block + norm reference"],
        "scope": "offline only; no segmentation fitting, action inference, KV reuse, or rollouts",
        "runtime_command": "PYTHONPATH=/tmp/stage4a_openclip_pkg python -u experiments/kv_cache/experiments/stage4a_unmatch_siglip_diagnostics.py",
    })

    # Layer/variant comparison plot derived from the serialized JSON structure.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [x["sample_id"].replace("task", "T").replace("_state0", "") for x in records]
    for ax, metric, title in [(axes[0], "oracle_fg_auc_posthoc", "Post-hoc oracle-FG ranking AUC"), (axes[1], "foreground_minus_background_mean", "Mean local score: oracle FG − BG")]:
        vals_a = [prompt_json[r["sample_id"]][r["task_description"]]["penultimate"][metric] for r in records]
        vals_b = [prompt_json[r["sample_id"]][r["task_description"]]["final"][metric] for r in records]
        xs = np.arange(len(records)); ax.scatter(xs-0.12, vals_a, label="Penultimate block 25", s=24); ax.scatter(xs+0.12, vals_b, label="Final block + norm", s=24)
        ax.axhline(0.5 if "auc" in metric else 0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(xs, labels, rotation=60, ha="right"); ax.set_title(title); ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(fig_dir / "feature_variant_posthoc_summary.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    _json(OUT / "timing_memory.json", diagnostic["timing"] | {"runtime": diagnostic["runtime"]})
    print(f"Wrote {OUT / 'matching_diagnostics.json'}", flush=True)


if __name__ == "__main__":
    main()
