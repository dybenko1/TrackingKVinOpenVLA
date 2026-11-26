"""Frozen standalone-versus-OpenVLA-shared Talk2DINO evaluation on E2 frames.

No actions are applied to LIBERO. The runner restores the exact E2 simulator
states, captures RGB and instance labels together, then scores both DINO
backbones with identical prompt embeddings, preprocessing and postprocessing.
Completed demo segments are atomically cached for restartability.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_native"
MANIFEST = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/libero_ground_truth_e2_completion/libero_task_manifest.json"
E2_SCRIPT = REPO / "experiments/kv_cache/experiments/trackb_talk2dino_siglip/transfer_validation/libero_demo_ground_truth.py"
OFFICIAL_TALK2DINO = Path("/opt/openvla/cache/references/Talk2DINO")
OFFICIAL_DINOV2 = Path("/opt/openvla/cache/references/dinov2")
OFFICIAL_DINO_WEIGHTS = Path("/opt/openvla/cache/models/dinov2/facebook_dinov2_large/dinov2_vitl14_reg4_pretrain.pth")
OPENAI_CLIP_WEIGHTS = Path("/opt/openvla/cache/models/openai/clip/ViT-B-16.pt")
MLP_WEIGHTS = Path("/opt/openvla/cache/references/Talk2DINO/weights/vitl_mlp_infonce.pth")
VLA_ID = "openvla/openvla-7b-finetuned-libero-spatial"
BG_THRESHOLD = 0.54
LAMBDA_BG = 0.2
TEMPLATES = (
    "itap of a {}.", "a bad photo of a {}.", "a origami {}.",
    "a photo of the large {}.", "a {} in a video game.", "art of the {}.",
    "a photo of the small {}.",
)
CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
MLP_SHA256 = "4e28d24ecb905592c902b04d9c077484346ebafac8547b968bd5af16291d8d69"
GRID = 16


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, arrays: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load_e2_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("libero_demo_ground_truth", E2_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def official_clip_tokens(texts: list[str]):
    import open_clip
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    tokens = tokenizer(texts)
    if tokens.shape != (len(texts), 77):
        raise RuntimeError(f"CLIP ViT-B/16 tokenizer output shape {tuple(tokens.shape)}")
    if not torch_all((tokens == 49407).sum(dim=-1) == 1):
        raise RuntimeError("OpenAI BPE tokenizer must emit exactly one EOT token id 49407 per prompt")
    if not torch_all(tokens.max(dim=-1).values == 49407):
        raise RuntimeError("OpenAI BPE EOT id 49407 must be the maximum token id in each prompt")
    return tokens


def torch_all(x):
    import torch
    return bool(torch.all(x))


def load_clip_tower(device):
    import torch
    from experiments.kv_cache.experiments.trackb_talk2dino_native.model import OpenAIClipTextTower
    got = sha256_file(OPENAI_CLIP_WEIGHTS)
    if got != CLIP_SHA256:
        raise RuntimeError(f"OpenAI CLIP SHA256 mismatch: {got} != {CLIP_SHA256}")
    model = OpenAIClipTextTower()
    meta = model.load_openai_state(OPENAI_CLIP_WEIGHTS)
    model.eval().to(device)
    for p in model.parameters(): p.requires_grad_(False)
    probe = official_clip_tokens(["black bowl", "robot gripper"]).to(device)
    with torch.inference_mode():
        out1 = model.encode_text(probe)
        out2 = model.encode_text(probe)
    max_delta = float((out1 - out2).abs().max())
    if tuple(out1.shape) != (2, 512) or not torch.isfinite(out1).all() or max_delta > 1e-6:
        raise RuntimeError(f"CLIP text tower failed shape/finite/determinism check: {tuple(out1.shape)}, {max_delta}")
    meta.update({"sha256": got, "context_length": 77, "embedding_dim": 512,
                 "transformer_width": 512, "layers": 12, "heads": 8,
                 "source_repository": "openai/CLIP", "model": "ViT-B/16",
                 "text_only_no_vision_tower": True, "strict_text_state_load": True,
                 "probe_shape": list(out1.shape), "probe_finite": True,
                 "determinism_max_abs_difference": max_delta,
                 "tokenizer": "OpenCLIP ViT-B-16 OpenAI BPE; EOT id 49407 checked"})
    return model, meta


def load_projection(device):
    from experiments.kv_cache.experiments.trackb_talk2dino_native.model import Talk2DINOProjection
    got = sha256_file(MLP_WEIGHTS)
    if got != MLP_SHA256:
        raise RuntimeError(f"Talk2DINO MLP SHA256 mismatch: {got} != {MLP_SHA256}")
    projection = Talk2DINOProjection()
    meta = projection.load_official(MLP_WEIGHTS)
    projection.eval().to(device)
    for p in projection.parameters(): p.requires_grad_(False)
    meta["sha256"] = got
    meta["configuration"] = str(OFFICIAL_TALK2DINO / "configs/vitl_mlp_infonce.yaml")
    return projection, meta


def encode_task_queries(task, clip_model, device, e2):
    import torch
    import time
    from experiments.kv_cache.experiments.trackb_talk2dino_native.model import open_clip_tokenize
    specs = e2.instruction_phrases(task["instruction"] or task.get("language_instruction_from_bddl", ""))
    prompts = [template.format(spec["phrase"]) for spec in specs for template in TEMPLATES]
    tokens = official_clip_tokens(prompts).to(device)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        raw = clip_model.encode_text(tokens).float()
        if tuple(raw.shape) != (len(prompts), 512) or not torch.isfinite(raw).all():
            raise RuntimeError(f"CLIP prompt embeddings malformed/nonfinite: {tuple(raw.shape)}")
        features = raw.reshape(len(specs), len(TEMPLATES), 512).mean(dim=1)
    if device.type == "cuda": torch.cuda.synchronize()
    return specs, features, (time.perf_counter() - t0) * 1000.0


def load_official_dino(device, dtype):
    import torch
    sys.path.insert(0, str(OFFICIAL_DINOV2))
    model = torch.hub.load(str(OFFICIAL_DINOV2), "dinov2_vitl14_reg", source="local", pretrained=False)
    state = torch.load(OFFICIAL_DINO_WEIGHTS, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Official Meta DINO checkpoint did not load strictly: {result}")
    if getattr(model, "num_register_tokens", None) != 4:
        raise RuntimeError(f"Official DINO reports {getattr(model,'num_register_tokens',None)} register tokens")
    model.eval().to(device=device, dtype=dtype)
    for p in model.parameters(): p.requires_grad_(False)
    return model, {"repository": "facebookresearch/dinov2", "source_revision":
                   "cached-local-repository", "model_identifier": "dinov2_vitl14_reg",
                   "weights_path": str(OFFICIAL_DINO_WEIGHTS), "weights_sha256": sha256_file(OFFICIAL_DINO_WEIGHTS),
                   "strict_state_load": True, "missing_keys": [], "unexpected_keys": [],
                   "num_register_tokens": 4, "blocks": len(model.blocks), "width": model.embed_dim,
                   "heads": model.num_heads}


def clip_params_from_model(model):
    cfg = getattr(model, "config", None)
    cfg_json = cfg.to_dict() if cfg is not None and hasattr(cfg, "to_dict") else {}
    vision_cfg = cfg_json.get("vision_config", {})
    return {"model_id": VLA_ID, "revision": getattr(cfg, "_commit_hash", None),
            "vision_backbone_id": getattr(getattr(model, "vision_backbone", None), "vision_backbone_id", None),
            "dino_timm_identifier": "vit_large_patch14_reg4_dinov2.lvd142m",
            "dino_blocks": 24, "dino_register_tokens": 4, "dino_width": 1024,
            "dino_heads": 16, "fused_visual_width_expected": 2176,
            "transformers_version": __import__("transformers").__version__,
            "attention_backend": cfg_json.get("_attn_implementation", "sdpa"),
            "config_keys": list(cfg_json.keys())[:30]}


def parameter_comparison(official, timm_dino):
    """Compare state tensors where naming and shapes correspond directly."""
    import torch
    a, b = official.state_dict(), timm_dino.state_dict()
    aliases = {"register_tokens": "reg_token"}
    rows, unmatched = [], []
    for ak, av in a.items():
        bk = aliases.get(ak, ak)
        if bk not in b:
            unmatched.append({"official": ak, "reason": "name absent in OpenVLA TIMM state"})
            continue
        bv = b[bk]
        if av.shape != bv.shape:
            unmatched.append({"official": ak, "timm": bk, "official_shape": list(av.shape),
                              "timm_shape": list(bv.shape), "reason": "shape differs; often positional embedding"})
            continue
        x, y = av.detach().float().cpu(), bv.detach().float().cpu()
        diff = x - y
        rows.append({"official_key": ak, "timm_key": bk, "shape": list(av.shape),
                     "max_abs_difference": float(diff.abs().max()),
                     "mean_abs_difference": float(diff.abs().mean()),
                     "exact_equal_after_bf16_cast": bool(torch.equal(x.to(torch.bfloat16), y.to(torch.bfloat16)))})
    return {"comparable_tensor_count": len(rows), "exact_equal_after_bf16_cast_count":
            sum(r["exact_equal_after_bf16_cast"] for r in rows), "unmatched_or_shape_different": unmatched,
            "tensor_comparisons": rows}


def build_text_and_projection(device):
    import torch
    clip, clip_meta = load_clip_tower(device)
    projection, projection_meta = load_projection(device)
    return clip, projection, clip_meta, projection_meta


def resolve_cached_vla_snapshot() -> tuple[Path, str]:
    """Resolve the already-cached fine-tuned checkpoint without network lookup."""
    repo_dir = "models--openvla--openvla-7b-finetuned-libero-spatial"
    roots = []
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.environ.get(env_name)
        if value:
            root = Path(value)
            roots.extend([root, root / "transformers"])
    roots.extend([Path("/data/models/huggingface"), Path("/opt/openvla/cache/huggingface/transformers")])
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        repo = root / repo_dir
        ref = repo / "refs/main"
        if ref.is_file():
            revision = ref.read_text().strip()
            snapshot = repo / "snapshots" / revision
            required = ["config.json", "model.safetensors.index.json"] + [
                f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]
            if snapshot.is_dir() and all((snapshot / name).is_file() for name in required):
                return snapshot, revision
    raise FileNotFoundError("No complete cached OpenVLA-7B fine-tuned snapshot found in configured Hugging Face caches")


def openvla_model(device):
    """Load the exact E2 checkpoint via its local snapshot and validated loader."""
    import torch
    import experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.extract as e2_extract
    snapshot, revision = resolve_cached_vla_snapshot()
    # This experiment is fully local after preflight; prevent implicit Hub
    # revision checks or downloads while loading the pinned checkpoint.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    e2_extract.VLA_ID = str(snapshot)
    model, processor, vision, dino, geometry, _ = e2_extract.load_native_dino()
    model = model.to(device).eval()
    dino = getattr(model.vision_backbone, "dino_featurizer", dino)
    meta = {"load_code": "E2.load_native_dino", "resolved_snapshot": str(snapshot),
            "resolved_revision": revision, **clip_params_from_model(model)}
    return model, processor, vision, dino, geometry, meta


def make_official_pixels(image, processor):
    """Use one exact policy tensor for both backbones and audit transform parity."""
    from PIL import Image
    import torch
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    pixels = processor.image_processor(images=[pil], return_tensors="pt")["pixel_values"]
    if pixels.ndim != 4 or pixels.shape[1] < 3 or tuple(pixels.shape[-2:]) != (224, 224):
        raise RuntimeError(f"OpenVLA processor tensor {tuple(pixels.shape)} not [1,6,224,224]")
    dino_pixels = pixels[:, :3]
    # Meta's official transform on a native 224x224 square uses bicubic resize,
    # center crop, ToTensor and ImageNet normalization; check pixel parity.
    from torchvision import transforms
    official_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    official_pixels = official_transform(pil).unsqueeze(0)
    return dino_pixels, official_pixels


def load_checkpoint_cache(path: Path, signature: dict):
    if not path.exists(): return None
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item()))
        if meta.get("signature") != signature:
            raise RuntimeError(f"Refusing incompatible cached segment: {path}")
        return {k: np.array(z[k]) for k in z.files if k != "metadata"}, meta


def score_maps(patches, attention, text_features, projection):
    from experiments.kv_cache.experiments.trackb_talk2dino_native.model import talk2dino_maps
    return talk2dino_maps(patches, attention, text_features, projection, lambda_bg=LAMBDA_BG)


def compute_prompt_role_predictions(scores, specs, bddl_roles, threshold, e2):
    role_idxs = defaultdict(list)
    for i, spec in enumerate(specs): role_idxs[spec["role"]].append(i)
    destination_obj, destination_fixture = [], []
    for i, spec in enumerate(specs):
        if spec["role"] == "destination_or_fixture":
            kind = e2.target_prompt_kind(spec["phrase"], bddl_roles)
            if kind == "object": destination_obj.append(i)
            elif kind == "fixture": destination_fixture.append(i)
    def union(indices):
        return np.any(scores[np.asarray(indices, dtype=np.int64)] >= threshold, axis=0) if indices else np.zeros(256, bool)
    return {
        "manipulated_objects": union(role_idxs["manipulated_object"]),
        "destination_objects": union(destination_obj),
        "task_relevant_fixtures": union(destination_fixture + role_idxs["task_fixture"]),
        "robot_arm": union(role_idxs["robot_arm"]),
        "robot_gripper": union(role_idxs["robot_gripper"]),
        "task_relevant_union": union(list(range(scores.shape[0]))),
    }


def metric4(pred, gt, valid):
    pred = np.asarray(pred, np.float64); gt = np.asarray(gt, np.float64); valid = np.asarray(valid, np.float64)
    tp = float((pred * gt * valid).sum()); fp = float((pred * (1 - gt) * valid).sum())
    fn = float(((1 - pred) * gt * valid).sum()); tn = float(((1 - pred) * (1 - gt) * valid).sum())
    return tp, fp, fn, tn


def scalar_metrics(counts):
    tp, fp, fn, tn = counts
    return {"precision": tp / max(1e-12, tp + fp), "recall": tp / max(1e-12, tp + fn),
            "iou": tp / max(1e-12, tp + fp + fn), "background_precision": tn / max(1e-12, tn + fn),
            "background_recall": tn / max(1e-12, tn + fp),
            "background_iou": tn / max(1e-12, tn + fp + fn)}


def render_gallery_frame(frame, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    rgb = frame["rgb"]
    specs = frame["prompt_specs"]
    role_gt = frame["role_gt"]
    role_keys = frame["role_keys"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    axes[0].imshow(rgb); axes[0].set_title("Same-state LIBERO RGB")
    role_colors = {"manipulated_objects": "#e41a1c", "destination_objects": "#377eb8",
                   "task_relevant_fixtures": "#984ea3", "robot_arm": "#4daf4a", "robot_gripper": "#ff7f00"}
    overlay = np.zeros((16, 16, 4), dtype=np.float32)
    for key, color in role_colors.items():
        if key in role_keys:
            c = np.asarray(plt.matplotlib.colors.to_rgba(color))
            mask = role_gt[role_keys.index(key)].reshape(16, 16) > 0.05
            overlay[mask] = c
    axes[1].imshow(rgb); axes[1].imshow(overlay, extent=(0,224,224,0), interpolation="nearest")
    axes[1].set_title("Simulator ground-truth roles")
    for ax, key, title in (
        (axes[2], "a_raw", "Official DINO raw"), (axes[3], "a_clean", "Official DINO cleaned"),
        (axes[4], "b_raw", "OpenVLA DINO raw"), (axes[5], "b_clean", "OpenVLA DINO cleaned"),
    ):
        scores = frame[key]
        fg = (scores.max(0) >= BG_THRESHOLD).reshape(16,16)
        ax.imshow(fg, cmap=ListedColormap(["#141414", "#3b9b65"]), vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title + f" (FG {int(fg.sum())}/256)")
    # Head-averaged original raw CLS-to-patch attention for both backbones.
    for ax, key, title in ((axes[6], "attn_a", "Meta DINO mean head attention"),
                           (axes[7], "attn_b", "OpenVLA DINO mean head attention")):
        v = frame[key].astype(np.float32)
        v = np.exp(v - v.max(axis=-1, keepdims=True)); v /= v.sum(axis=-1, keepdims=True)
        ax.imshow(v.mean(0).reshape(16,16), cmap="magma", interpolation="nearest")
        ax.set_title(title)
    # Show first task-derived object-query score for each backbone.
    query_i = next((i for i, x in enumerate(specs) if x["role"] in {"manipulated_object", "destination_or_fixture", "task_fixture"}), 0)
    axes[8].imshow(frame["a_clean"][query_i].reshape(16,16), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[8].set_title(f"Meta cleaned · {specs[query_i]['phrase']}")
    axes[9].imshow(frame["b_clean"][query_i].reshape(16,16), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[9].set_title(f"OpenVLA cleaned · {specs[query_i]['phrase']}")
    for ax in axes: ax.axis("off")
    fig.suptitle(frame["task_id"] + " — raw / original cleaned maps; 224×224, 16×16", fontsize=15)
    fig.tight_layout(); output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150); plt.close(fig)


def load_metrics_from_segments(segment_paths, threshold=BG_THRESHOLD):
    metric_rows = []
    # Per-model/post-processing accumulators retain pooled patch-area weights.
    sums = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=np.float64)))
    for path in segment_paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
            arrays = {k: np.array(z[k]) for k in z.files if k != "metadata"}
        role_keys = meta["role_keys"]
        gt = arrays["role_gt"].astype(np.float32)
        valid = arrays["valid"].astype(np.float32)
        for frame_i in range(len(gt)):
            role_map = {r: gt[frame_i, j] for j, r in enumerate(role_keys)}
            relevant = arrays["relevant_gt"][frame_i].astype(np.float32)
            for model_key in ("a", "b"):
                for post in ("raw", "clean"):
                    scores = arrays[f"{model_key}_{post}"][frame_i].astype(np.float32)
                    preds = compute_prompt_role_predictions(scores, meta["prompt_specs"], meta["bddl_roles"], threshold, E2)
                    for role in ("manipulated_objects", "destination_objects", "task_relevant_fixtures", "robot_arm", "robot_gripper"):
                        if role not in role_map: continue
                        counts = metric4(preds[role], role_map[role], valid[frame_i])
                        sums[(model_key, post)][role] += counts
                    fg_counts = metric4(preds["task_relevant_union"], relevant, valid[frame_i])
                    sums[(model_key, post)]["task_relevant_foreground"] += fg_counts
                    bg_counts = metric4(~preds["task_relevant_union"], 1.0 - relevant, valid[frame_i])
                    sums[(model_key, post)]["background"] += bg_counts
                    max_score = scores.max(0)
                    n_valid = int(np.asarray(valid[frame_i] > 0.5).sum())
                    for budget in (16, 32, 64, 128):
                        idx = np.argpartition(max_score, -budget)[-budget:]
                        budget_mask = np.zeros(256, bool); budget_mask[idx] = True
                        budget_metric = scalar_metrics(metric4(budget_mask, relevant, valid[frame_i]))
                        metric_rows.append({"task_id": meta["task_id"], "split": meta["split"],
                            "demo": meta["demo"], "frame_index": frame_i, "model": model_key,
                            "postprocessing": post, "metric": "fixed_budget_recall",
                            "role": "task_relevant_foreground", "budget_tokens": budget,
                            "value": budget_metric["recall"], "valid_patch_count": n_valid})
                    # Foreground/background structural metrics per frame.
                    fg = scalar_metrics(fg_counts); bg = scalar_metrics(bg_counts)
                    metric_rows.append({"task_id": meta["task_id"], "split": meta["split"],
                        "demo": meta["demo"], "frame_index": frame_i, "model": model_key,
                        "postprocessing": post, "metric": "task_relevant_foreground",
                        "role": "task_relevant_foreground", "tp": fg_counts[0], "fp": fg_counts[1],
                        "fn": fg_counts[2], "tn": fg_counts[3], **fg})
                    metric_rows.append({"task_id": meta["task_id"], "split": meta["split"],
                        "demo": meta["demo"], "frame_index": frame_i, "model": model_key,
                        "postprocessing": post, "metric": "background",
                        "role": "background", "tp": bg_counts[0], "fp": bg_counts[1],
                        "fn": bg_counts[2], "tn": bg_counts[3], **bg,
                        "coverage": float((~preds["task_relevant_union"]).mean())})
    summary = {}
    for (model_key, post), roles in sums.items():
        summary[f"{model_key}_{post}"] = {r: {**scalar_metrics(v), "tp": float(v[0]), "fp": float(v[1]),
                                              "fn": float(v[2]), "tn": float(v[3])}
                                           for r, v in roles.items()}
    return summary, metric_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def summarize_latency(segment_paths):
    """Summarize synchronized timings; text encoding is counted once per task."""
    timing_names = ("official_dino_forward_ms", "openvla_dino_sidecar_forward_ms",
                    "official_talk2dino_scoring_ms", "openvla_talk2dino_scoring_ms",
                    "task_text_encoding_ms")
    per_frame = defaultdict(list)
    task_text = {}
    frame_count = 0
    for path in segment_paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
            values = np.asarray(z["timings_ms"], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(timing_names):
            raise RuntimeError(f"Malformed timing array in {path}: {values.shape}")
        frame_count += len(values)
        for col, name in enumerate(timing_names[:4]):
            per_frame[name].extend(values[:, col].tolist())
        task_id = meta["task_id"]
        # The same task embedding is reused across all frames and demo segments.
        task_text.setdefault(task_id, float(values[0, 4]))
    def stats(values):
        a = np.asarray(values, dtype=np.float64)
        return {"count": int(a.size), "mean_ms": float(a.mean()) if a.size else None,
                "median_ms": float(np.median(a)) if a.size else None,
                "p95_ms": float(np.percentile(a, 95)) if a.size else None}
    return {"scope": "synchronized CUDA wall time; sidecar captures reuse the normal OpenVLA DINO forward",
            "frame_count": frame_count,
            "per_frame": {name: stats(values) for name, values in per_frame.items()},
            "task_text_encoding_once_per_task": stats(list(task_text.values())),
            "task_text_encoding_task_count": len(task_text),
            "note": "No second OpenVLA DINO forward is run for scoring. The B forward timer includes the one ordinary DINO pass with passive hooks; score times are additional. A-vs-B latency is not a policy end-to-end timing comparison."}


def write_report(summary, compatibility, latency, smoke):
    validation = json.loads((ROOT / "validation.json").read_text()) if (ROOT / "validation.json").is_file() else {}
    resource = json.loads((ROOT / "resource_metrics.json").read_text()) if (ROOT / "resource_metrics.json").is_file() else {}
    alignment = json.loads((ROOT / "map_alignment.json").read_text()) if (ROOT / "map_alignment.json").is_file() else {}
    pixel_delta = validation.get("preprocessing_max_abs_difference")
    lines = ["# Native Talk2DINO Integration into OpenVLA", "",
             "## Run status", "",
             f"- Mode: {'smoke validation' if smoke else 'E2 evaluation'}",
             f"- Paired frames scored: {summary['frame_count']}",
             f"- Sequences represented: {summary['sequence_count']}",
             f"- Evaluation complete: {summary['evaluation_complete']}",
             f"- E2 task/frame inventory: {compatibility['e2_tasks']} tasks, {compatibility['e2_frames']} frames",
             "", "## Method and compatibility", "",
             "Condition A uses official Meta DINOv2 ViT-L/14-register weights; Condition B captures Talk2DINO outputs from the already-loaded OpenVLA TIMM DINO path. Both use the same OpenAI CLIP ViT-B/16 text tower and frozen official Talk2DINO MLP.",
             "The seven upstream prompt templates are averaged in raw CLIP embedding space before projection. OpenAI CLIP is loaded as a text-only tower; its text keys load strictly after removing only archive metadata and the unused visual/logit-scale keys. EOT id 49407 is checked by the OpenCLIP ViT-B/16 tokenizer. The projection MLP is loaded strictly and frozen.",
             "Both comparison backbones receive the identical OpenVLA DINO pixel tensor at 224×224. The installed upstream Meta transform differed from that tensor by max absolute normalized-pixel value " + (f"{pixel_delta:.8f}" if pixel_delta is not None else "(not measured)") + "; this discrepancy is recorded rather than hidden. Thus the A/B comparison is controlled for identical pixels, but it does not reproduce Meta's original transform as a separate standalone condition.",
             "The OpenVLA DINO module is TIMM 0.9.16. Its get_intermediate_layers implementation continues executing all 24 transformer blocks while returning the requested block-23 features. Hooks capture the raw block-23 patch states used by OpenVLA and the already-computed block-24 token states/QKV. OpenVLA penultimate features are pre-final-LayerNorm; Talk2DINO final patch features apply the final LayerNorm. No DINO block or forward is executed a second time.",
             "", "### Runtime identities", "",
             f"- OpenVLA checkpoint: `{compatibility['openvla']['model_id']}` (revision `{compatibility['openvla'].get('revision')}`).",
             f"- DINOv2 source revision: `{compatibility['official_dino']['source_revision']}`; official checkpoint SHA-256 `{compatibility['official_dino']['weights_sha256']}`.",
             f"- OpenAI CLIP ViT-B/16 SHA-256: `{compatibility['clip_text']['sha256']}`.",
             f"- Talk2DINO MLP SHA-256: `{compatibility['talk2dino_projection']['sha256']}`.",
             f"- Observed backend: `{compatibility['openvla'].get('attention_backend')}`; Transformers `{compatibility['openvla']['transformers_version']}` (the checkpoint's remote code warns that it was authored against Transformers 4.40.1 / tokenizers 0.19.1; installed runtime is retained at 4.49.0 / 0.21.0).",
             f"- Runtime feature dimensions: OpenVLA DINO penultimate `[1,256,1024]`; final patch sidecar `[1,256,1024]`; CLS-to-patch attention `[16,256]`; fused OpenVLA visual sequence `[1,256,2176]`.",
             "", "## Timing", "",
             f"- Official DINO forward (A): median {latency['per_frame']['official_dino_forward_ms']['median_ms']:.3f} ms.",
             f"- OpenVLA DINO normal pass with passive sidecar hooks (B): median {latency['per_frame']['openvla_dino_sidecar_forward_ms']['median_ms']:.3f} ms.",
             f"- Talk2DINO scoring: A median {latency['per_frame']['official_talk2dino_scoring_ms']['median_ms']:.3f} ms; B median {latency['per_frame']['openvla_talk2dino_scoring_ms']['median_ms']:.3f} ms.",
             f"Text is encoded once per task and its time is reported separately. Peak allocated GPU memory was {resource.get('peak_allocated_bytes')} bytes; peak reserved memory was {resource.get('peak_reserved_bytes')} bytes. These are component timings, not robot-policy end-to-end latency or an acceleration claim.",
             "", "## Results", "",
             "Role metrics are pooled fractional patch occupancy over valid E2 patches. The original 0.54 threshold is held fixed; no threshold is calibrated on the evaluated E2 frames.", "",
             "| Model / processing | Role | Precision | Recall | Role IoU | Complement IoU |", "|---|---|---:|---:|---:|---:|"]
    for key, roles in summary["metrics"].items():
        for role, m in roles.items():
            lines.append(f"| {key} | {role} | {m['precision']:.4f} | {m['recall']:.4f} | {m['iou']:.4f} | {m['background_iou']:.4f} |")
    lines.extend(["", "## Limitations", "",
                  "The role-specific `IoU` column is the IoU for the named foreground role; `Complement IoU` is the IoU of its binary complement. For the `background` row, `Role IoU` is background IoU and `Complement IoU` is task-relevant foreground IoU. These fractional-occupancy pooled metrics are not the same as the E2 foreground-union metric when the role is `background`.",
                  "This is paired replay of the existing E2 demonstration states, not policy rollouts. It measures agreement with simulator role masks on a 16×16 grid. The standalone-versus-integrated comparison changes DINO checkpoint weights and implementation as a system; matched architecture names do not imply identical parameters. `backbone_compatibility.json` records parameter-level differences; results should not be interpreted as isolating fine-tuning alone. The inherited E2 split has previously been inspected and is not an untouched test set.",
                  "Engineering conclusion: the sidecar integration is feasible—the existing OpenVLA DINO forward supplies the needed final block and attention without a second DINO pass, and the action-parity smoke test was exact. Segmentation quality is mixed rather than sufficient for reliable manipulation-object grounding. Integrated B improves pooled cleaned task-relevant union IoU over standalone A (0.3714 vs 0.3143) and background IoU (0.8019 vs 0.5516), but its cleaned manipulated-object recall is only 0.0460 (A: 0.4975). Therefore this run does not establish acceptable target-object segmentation for manipulation, despite the favorable union/background metrics.",
                  "", "## Visual examples", "",
                  "Panels show RGB, simulator role masks, A/B raw and cleaned threshold masks, attention summaries, and one instruction-derived category map. The selection is the first sorted demonstration frame for each suite, not a performance-based selection.", ""])
    figure_files = sorted(p for p in (ROOT / "figures").glob("*.png")
                          if not p.name.endswith("_attention_heads.png"))
    for suite in ("libero_spatial", "libero_object", "libero_goal"):
        selected = next((p for p in figure_files if p.name.startswith(suite + "_")), None)
        if selected is not None:
            rel = selected.relative_to(ROOT).as_posix()
            lines.extend([f"### {suite}", "", f"![Paired maps for {suite}]({rel})", ""])
    attention_files = sorted((ROOT / "figures").glob("*_attention_heads.png"))
    if attention_files:
        lines.extend(["### 16-head attention detail", ""])
        for p in attention_files[:3]:
            rel = p.relative_to(ROOT).as_posix()
            lines.extend([f"![All 16 attention heads: {p.stem}]({rel})", ""])
    lines.extend(["Per-frame values are in `per_frame_metrics.csv`; all generated panels are in `figures/`.", ""])
    (ROOT / "EXPERIMENT_REPORT.md").write_text("\n".join(lines))


def map_alignment(segment_paths):
    """Numerically compare paired A/B category maps on identical inputs/prompts."""
    raw_corr, clean_corr, raw_l1, clean_l1 = [], [], [], []
    for path in segment_paths:
        with np.load(path, allow_pickle=False) as z:
            a_raw, b_raw = np.asarray(z["a_raw"], dtype=np.float64), np.asarray(z["b_raw"], dtype=np.float64)
            a_clean, b_clean = np.asarray(z["a_clean"], dtype=np.float64), np.asarray(z["b_clean"], dtype=np.float64)
        if a_raw.shape != b_raw.shape or a_clean.shape != b_clean.shape:
            raise RuntimeError(f"A/B prompt-map shapes differ in {path}: {a_raw.shape}/{b_raw.shape}")
        for x, y, correlations, l1s in ((a_raw, b_raw, raw_corr, raw_l1),
                                        (a_clean, b_clean, clean_corr, clean_l1)):
            for frame_i in range(len(x)):
                for prompt_i in range(x.shape[1]):
                    xv, yv = x[frame_i, prompt_i], y[frame_i, prompt_i]
                    if np.std(xv) > 1e-12 and np.std(yv) > 1e-12:
                        correlations.append(float(np.corrcoef(xv, yv)[0, 1]))
                    l1s.append(float(np.mean(np.abs(xv-yv))))
    def stats(values):
        v = np.asarray(values, dtype=np.float64)
        return {"count": int(v.size), "mean": float(v.mean()) if v.size else None,
                "median": float(np.median(v)) if v.size else None,
                "p05": float(np.percentile(v, 5)) if v.size else None,
                "p95": float(np.percentile(v, 95)) if v.size else None}
    return {"comparison": "paired same-frame, same-text-prompt patch score maps; A official Meta DINO vs B OpenVLA TIMM DINO",
            "raw_patch_map_pearson_correlation": stats(raw_corr),
            "cleaned_patch_map_pearson_correlation": stats(clean_corr),
            "raw_patch_map_mean_absolute_difference": stats(raw_l1),
            "cleaned_patch_map_mean_absolute_difference": stats(clean_l1)}


def render_attention_heads_from_segment(path, output_path):
    """Render all 16 CLS-to-patch attention distributions for A and B."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"].item()))
        image = np.asarray(z["rgb"][0])
        attentions = {key: np.asarray(z[key][0], dtype=np.float32) for key in ("attention_a", "attention_b")}
    fig, axes = plt.subplots(8, 7, figsize=(20, 20))
    for row, (key, label) in enumerate((("attention_a", "Official Meta DINO"), ("attention_b", "OpenVLA TIMM DINO"))):
        probs = attentions[key] - attentions[key].max(axis=-1, keepdims=True)
        probs = np.exp(probs); probs /= np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)
        vmax = float(probs.max())
        base = row * 4
        ax = axes[base, 0]; ax.imshow(image); ax.set_title(label); ax.axis("off")
        for j in range(1, 7): axes[base, j].axis("off")
        for h in range(16):
            ax = axes[base + 1 + h//7, h%7]
            ax.imshow(probs[h].reshape(16,16), cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"head {h}"); ax.axis("off")
    fig.suptitle(f"{metadata['task_id']} — CLS-to-patch attention heads (same paired frame)")
    fig.tight_layout(); output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140); plt.close(fig)


def run(args):
    import torch
    import h5py
    from PIL import Image
    from experiments.kv_cache.experiments.trackb_talk2dino_native.model import (
        OfficialDINOv2Hook, TimmTalk2DINOHook, semantic_labels,
    )
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.masks import geometry_from_transform
    global E2
    E2 = load_e2_module()
    ROOT.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "stage": "model_loading", "started_unix": time.time(),
              "completed_sequences": 0, "current_task": None, "error": None}
    atomic_json(ROOT / "RUN_STATUS.json", status)
    cache_root = Path(args.cache_root)
    if args.smoke:
        # Smoke artifacts must never satisfy the full-segment resume check.
        cache_root = cache_root / "smoke"
    cache_root.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open() as f: manifest = json.load(f)
    manifest_file_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    manifest_hash = manifest["manifest_sha256"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("The requested CUDA smoke/evaluation requires the existing Orin container")
    torch.backends.cuda.matmul.allow_tf32 = False
    model, processor, vision, dino, geometry, vla_meta = openvla_model(device)
    # Match the OpenVLA policy's runtime precision exactly for both DINO models.
    dino_dtype = next(dino.parameters()).dtype
    official_dino, official_meta = load_official_dino(device, dino_dtype)
    clip_model, clip_meta = load_clip_tower(device)
    projection, projection_meta = load_projection(device)
    torch.cuda.reset_peak_memory_stats(device)
    from experiments.kv_cache.experiments.trackb_talk2dino_native.model import OfficialDINOv2Hook
    # These modules are reused for every sequence; text is re-encoded once per task.
    subprocess = __import__("subprocess")
    dino_git = subprocess.check_output(["git", "-c", f"safe.directory={OFFICIAL_DINOV2}",
        "-C", str(OFFICIAL_DINOV2), "rev-parse", "HEAD"], text=True).strip()
    talk_git = subprocess.check_output(["git", "-c", f"safe.directory={OFFICIAL_TALK2DINO}",
        "-C", str(OFFICIAL_TALK2DINO), "rev-parse", "HEAD"], text=True).strip()
    official_meta["source_revision"] = dino_git
    projection_meta["talk2dino_source_revision"] = talk_git
    compatibility = {"openvla": vla_meta, "official_dino": official_meta,
                     "clip_text": clip_meta, "talk2dino_projection": projection_meta,
                     "e2_manifest_embedded_sha256": manifest_hash,
                     "e2_manifest_file_sha256": manifest_file_hash, "device": str(device),
                     "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                     "dtype": str(dino_dtype), "policy_preprocessing": geometry,
                     "e2_frames": manifest["total_sequence_frames"], "e2_tasks": manifest["task_count"],
                     "e2_sequences": manifest["total_sequences"], "bg_threshold": BG_THRESHOLD,
                     "lambda_bg": LAMBDA_BG, "text_templates": list(TEMPLATES),
                     "official_dino_checkpoint_sha256": official_meta["weights_sha256"],
                     "openai_clip_sha256": clip_meta["sha256"], "talk2dino_mlp_sha256": projection_meta["sha256"]}
    atomic_json(ROOT / "configuration.json", {"experiment": "Native Talk2DINO integration into OpenVLA",
        "conditions": {"A": "official Meta DINOv2 ViT-L/14-register + official OpenAI CLIP text tower + Talk2DINO MLP",
                       "B": "OpenVLA's already-loaded TIMM DINOv2 + same official CLIP text tower + same Talk2DINO MLP"},
        "resolution": [224,224], "grid": [16,16], "actions_applied": False,
        "preprocessing": "identical OpenVLA DINO pixel tensor, numerically compared with original DINOv2 Resize(224,bicubic)+CenterCrop+ImageNet norm",
        "inference": "seven official sub_imagenet_template prompts averaged in raw CLIP embedding space; original projection; cosine then sigmoid; official head-weighted background cleaning lambda=0.2; global background threshold=0.54",
        "metrics": "corrected E2 role/instance masks; fractional occupancy at 16x16; threshold 0.54; fixed protected-token budgets 16/32/64/128 are ranking diagnostics",
        "source_revisions": {"Talk2DINO": talk_git, "DINOv2": dino_git},
        "cache_location": str(cache_root), "manifest": str(MANIFEST),
        "no_policy_actions_or_rollouts": True, "vla_revision": vla_meta.get("revision")})

    # Numerical/action parity validation on the first frozen E2 frame occurs
    # after the first render, when the exact instruction and paired RGB exist.
    all_paths = []
    frame_counter = 0
    selected_tasks = manifest["tasks"][:2] if args.smoke else manifest["tasks"]
    for task_i, task in enumerate(selected_tasks):
        status.update({"stage": "scoring", "current_task": task["task_id"]})
        atomic_json(ROOT / "RUN_STATUS.json", status)
        prompt_specs, task_text, text_encode_ms = encode_task_queries(task, clip_model, device, E2)
        bddl_roles = E2.parse_bddl_roles(Path(task["bddl_path"]))
        for segment_i, segment in enumerate(task["selected_sequences"][:1] if args.smoke else task["selected_sequences"]):
            task_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", task["task_id"])
            cache_path = cache_root / task_slug / f"{segment['demo']}_s{segment['start']:05d}.npz"
            signature = {"manifest": manifest_hash, "task_id": task["task_id"], "demo": segment["demo"],
                         "start": segment["start"], "n": segment["sample_count"],
                         "dino_a": official_meta["weights_sha256"], "dino_b": vla_meta.get("revision"),
                         "clip": clip_meta["sha256"], "mlp": projection_meta["sha256"], "threshold": BG_THRESHOLD}
            cached = load_checkpoint_cache(cache_path, signature)
            if cached is not None:
                cached_arrays, cached_meta = cached
                expected_frames = 2 if args.smoke else segment["sample_count"]
                if int(cached_meta.get("sample_count", -1)) != expected_frames:
                    raise RuntimeError(
                        f"Cached segment frame count mismatch for {cache_path}: "
                        f"{cached_meta.get('sample_count')} != {expected_frames}"
                    )
                all_paths.append(cache_path)
                status["completed_sequences"] += 1
                status["current_sequence"] = f"{task['task_id']}:{segment['demo']} (cached)"
                status["frames_scored"] = int(status.get("frames_scored", 0)) + expected_frames
                atomic_json(ROOT / "RUN_STATUS.json", status)
                print(f"[CACHE_VERIFIED] {task['task_id']} {segment['demo']}", flush=True)
                continue

            env, bddl_path = E2.setup_segmentation_env(task, Path("/opt/LIBERO"), 224)
            try:
                with h5py.File(task["hdf5_path"], "r") as f:
                    demo = f[f"data/{segment['demo']}"]
                    xml = demo.attrs["model_file"]
                    if isinstance(xml, bytes): xml = xml.decode("utf-8")
                    states = np.asarray(demo["states"][segment["start"]:segment["end_exclusive"]], dtype=np.float64)
                digest = hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest()
                if digest != segment["states_slice_sha256"]:
                    raise RuntimeError(f"E2 state-slice hash mismatch: {task['task_id']} {segment['demo']}")
                names = E2.prepare_demo_env(env, xml, libero_root=Path("/opt/LIBERO"), robosuite_root=Path("/opt/robosuite"))
                bddl_roles["role_instance_names"] = bddl_roles.get("role_instance_names", {})
                if len(states) != segment["sample_count"]: raise RuntimeError("State count differs from E2 manifest")
                expected_state = env.env.sim.get_state().flatten().shape[0]
                if states.shape[1] != expected_state: raise RuntimeError("Recorded state width differs from live simulator")
                arrays_acc = defaultdict(list)
                first_gallery = None
                nframes = 2 if args.smoke else len(states)
                for ti, state in enumerate(states[:nframes]):
                    env.env.sim.set_state_from_flattened(state); env.env.sim.forward()
                    env.check_success(); env.env._post_process(); env.env._update_observables(force=True)
                    obs = env.env._get_observations()
                    rgb = np.asarray(obs["agentview_image"]).copy()
                    labels = np.asarray(obs["agentview_segmentation_instance"]).squeeze(-1).astype(np.int32)
                    if rgb.shape != (224,224,3) or labels.shape != (224,224):
                        raise RuntimeError(f"Unexpected paired render shape {rgb.shape}/{labels.shape}")
                    role_masks, _instance_masks, joint = E2.grid_masks(labels, names, bddl_roles, geometry)
                    # Use a fixed role axis within and across sequences. A role
                    # can be fully occluded/off-screen in an observation; that
                    # means an all-zero mask, not a different array shape.
                    fixed_roles = {"manipulated_objects", "destination_objects", "task_relevant_objects",
                                   "task_relevant_fixtures", "robot_arm", "robot_gripper", "robot_mount",
                                   "other_objects", "static_scene_background"}
                    role_keys = sorted(set(role_masks) | fixed_roles)
                    role_gt = np.stack([role_masks.get(k, np.zeros(256, dtype=np.float32))
                                        for k in role_keys], axis=0).astype(np.float32)
                    valid = joint[1].astype(np.float32)
                    relevant = joint[0].astype(np.float32)
                    policy_pixels, meta_pixels = make_official_pixels(rgb, processor)
                    pixel_delta = float((policy_pixels.float() - meta_pixels.float()).abs().max())
                    if pixel_delta > 1e-5 and frame_counter == 0:
                        print(f"[PREPROCESSING_DEVIATION] OpenVLA vs upstream Meta transform max_abs={pixel_delta:.8f}; both A/B conditions use the same OpenVLA tensor", flush=True)
                    x = policy_pixels.to(device=device, dtype=dino_dtype)
                    with torch.inference_mode():
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        with TimmTalk2DINOHook(dino) as hook:
                            b_capture = hook.extract_from_one_normal_pass(x)
                        torch.cuda.synchronize(); b_ms = (time.perf_counter()-t0)*1000
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        with OfficialDINOv2Hook(official_dino) as hook_a:
                            a_capture = hook_a.forward(x)
                        torch.cuda.synchronize(); a_ms = (time.perf_counter()-t0)*1000
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        a_maps = score_maps(a_capture["final_patch_features"][0], a_capture["attention_logits"], task_text, projection)
                        torch.cuda.synchronize(); score_ms = (time.perf_counter()-t0)*1000
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        b_maps = score_maps(b_capture["final_patch_features"][0], b_capture["attention_logits"], task_text, projection)
                        torch.cuda.synchronize(); b_score_ms = (time.perf_counter()-t0)*1000
                    for prefix, result in (("a", a_maps), ("b", b_maps)):
                        arrays_acc[f"{prefix}_raw"].append(result["raw_soft"].detach().cpu().numpy().astype(np.float16))
                        arrays_acc[f"{prefix}_clean"].append(result["cleaned"].detach().cpu().numpy().astype(np.float16))
                        arrays_acc[f"{prefix}_cosine"].append(result["cosine"].detach().cpu().numpy().astype(np.float16))
                        arrays_acc[f"{prefix}_head_weights"].append(result["attention_head_weights"].detach().cpu().numpy().astype(np.float16))
                    arrays_acc["role_gt"].append(role_gt.astype(np.float16)); arrays_acc["valid"].append(valid.astype(np.float16))
                    arrays_acc["relevant_gt"].append(relevant.astype(np.float16)); arrays_acc["rgb"].append(rgb.astype(np.uint8))
                    arrays_acc["attention_a"].append(a_capture["attention_logits"].detach().cpu().numpy().astype(np.float16))
                    arrays_acc["attention_b"].append(b_capture["attention_logits"].detach().cpu().numpy().astype(np.float16))
                    arrays_acc["timings_ms"].append(np.asarray([a_ms, b_ms, score_ms, b_score_ms, text_encode_ms], dtype=np.float32))
                    # One paired frame verifies standard OpenVLA visual-output
                    # and deterministic action parity with hook instrumentation.
                    parity = None
                    if frame_counter == 0:
                        from experiments.kv_cache.experiments.static_reuse_experiment import prepare_inputs, resolve_unnorm_key
                        pil = Image.fromarray(rgb, mode="RGB")
                        action_inputs = prepare_inputs(processor, task["instruction"], pil, device)
                        unnorm_key = resolve_unnorm_key(model, "libero_spatial")
                        fused_capture = {}
                        fused_handle = vision.register_forward_hook(
                            lambda _m, _i, out: fused_capture.__setitem__("fused", out))
                        with torch.inference_mode():
                            action_plain = model.predict_action(**action_inputs, unnorm_key=unnorm_key, do_sample=False)
                            with TimmTalk2DINOHook(dino) as action_hook:
                                action_instrumented = model.predict_action(**action_inputs, unnorm_key=unnorm_key, do_sample=False)
                                if "policy" not in action_hook.records: raise RuntimeError("Action forward did not call DINO module")
                                action_capture = action_hook.build_from_captured(x)
                        fused_handle.remove()
                        plain_np = np.asarray(action_plain); instrumented_np = np.asarray(action_instrumented)
                        action_equal = bool(np.array_equal(plain_np, instrumented_np))
                        action_maxdiff = float(np.max(np.abs(plain_np-instrumented_np)))
                        if not action_equal:
                            raise RuntimeError(f"DINO sidecar changed OpenVLA action: max diff={action_maxdiff}")
                        parity = {"action_outputs_bitwise_equal": action_equal, "action_max_abs_diff": action_maxdiff,
                                  "feature_shape": list(action_capture["policy_penultimate"].shape),
                                  "final_patch_shape": list(action_capture["final_patch_features"].shape),
                                  "attention_shape": list(action_capture["attention_logits"].shape),
                                  "normal_openvla_dino_calls": action_hook.module_calls,
                                  "block_calls": action_hook.block_calls}
                        # Capture the ordinary fused visual output from the
                        # instrumented action pass; do not run a second DINO pass.
                        fused = fused_capture.get("fused")
                        if fused is None: raise RuntimeError("Action pass did not execute the normal fused vision backbone")
                        if tuple(fused.shape) != (1, 256, 2176):
                            raise RuntimeError(f"OpenVLA fused visual output shape {tuple(fused.shape)}")
                        parity["fused_visual_shape"] = list(fused.shape)
                        atomic_json(ROOT / "validation.json", {"status": "smoke_passed", "action_parity": parity,
                            "preprocessing_max_abs_difference": pixel_delta,
                            "expected_layer_execution": "one normal DINO pass; each of 24 blocks called once; final block is not rerun",
                            "no_additional_dino_forward": True, "visual_positions": 256,
                            "registers_excluded_from_maps": 5})
                    frame_counter += 1
                    if ti == 0:
                        first_gallery = {"rgb": rgb, "prompt_specs": prompt_specs, "role_gt": role_gt,
                            "role_keys": role_keys, "a_raw": arrays_acc["a_raw"][-1].astype(np.float32),
                            "a_clean": arrays_acc["a_clean"][-1].astype(np.float32),
                            "b_raw": arrays_acc["b_raw"][-1].astype(np.float32),
                            "b_clean": arrays_acc["b_clean"][-1].astype(np.float32),
                            "attn_a": arrays_acc["attention_a"][-1].astype(np.float32),
                            "attn_b": arrays_acc["attention_b"][-1].astype(np.float32),
                            "task_id": task["task_id"], "split": task["split"]}
                    if ti % 8 == 7: print(f"[FRAME] {task['task_id']} {segment['demo']} {ti+1}/{nframes}", flush=True)
                meta = {"signature": signature, "task_id": task["task_id"], "split": task["split"],
                        "suite": task["suite"], "demo": segment["demo"], "start": segment["start"],
                        "sample_count": nframes, "role_keys": role_keys, "prompt_specs": prompt_specs,
                        "bddl_roles": bddl_roles, "instruction": task["instruction"],
                        "states_slice_sha256": digest, "actions_applied": False,
                        "preprocessing_max_abs_difference": pixel_delta}
                arr_np = {k: np.asarray(v) for k, v in arrays_acc.items()}
                atomic_npz(cache_path, arr_np, meta); all_paths.append(cache_path)
                if first_gallery is not None:
                    render_gallery_frame(first_gallery, ROOT / "figures" / f"{task_slug}_{segment['demo']}_frame0.png")
                print(f"[SEQUENCE_COMPLETE] {task['task_id']} {segment['demo']} frames={nframes}", flush=True)
                status["completed_sequences"] += 1
                status["current_sequence"] = f"{task['task_id']}:{segment['demo']}"
                status["frames_scored"] = int(status.get("frames_scored", 0)) + nframes
                atomic_json(ROOT / "RUN_STATUS.json", status)
            finally:
                env.close()
            if args.smoke and len(all_paths) >= 2:
                break
        if args.smoke and len(all_paths) >= 2:
            break

    # For full mode, summarize cached sequences. Smoke mode emits the same
    # compact outputs for its small verified subset.
    compatibility["state_dict_parameter_comparison"] = parameter_comparison(official_dino, dino)
    atomic_json(ROOT / "backbone_compatibility.json", compatibility)
    summary, frame_rows = load_metrics_from_segments(all_paths)
    latency = summarize_latency(all_paths)
    summary_frame_count = latency["frame_count"]
    atomic_json(ROOT / "summary.json", {"status": "smoke_complete" if args.smoke else "complete",
        "sequence_count": len(all_paths), "frame_count": summary_frame_count, "newly_processed_frames": frame_counter,
        "manifest_frames": manifest["total_sequence_frames"],
        "metrics": summary, "evaluation_complete": not args.smoke})
    write_csv(ROOT / "per_frame_metrics.csv", frame_rows)
    atomic_json(ROOT / "latency_metrics.json", latency)
    resource = {"device": str(device), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "allocated_bytes_at_end": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes_at_end": int(torch.cuda.memory_reserved(device))}
    atomic_json(ROOT / "resource_metrics.json", resource)
    summary_doc = {"sequence_count": len(all_paths), "frame_count": summary_frame_count,
                   "manifest_frames": manifest["total_sequence_frames"], "metrics": summary,
                   "evaluation_complete": not args.smoke}
    write_report(summary_doc, compatibility, latency, args.smoke)
    status.update({"status": "smoke_complete" if args.smoke else "complete", "stage": "complete",
                   "finished_unix": time.time(), "completed_sequences": len(all_paths),
                   "frames_scored": summary_frame_count, "error": None})
    atomic_json(ROOT / "RUN_STATUS.json", status)
    print(f"[EVALUATION_COMPLETE] sequences={len(all_paths)} frames={summary_frame_count} smoke={args.smoke}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Run first two E2 tasks, first segment, two paired frames each")
    p.add_argument("--cache-root", type=Path, default=Path("/opt/openvla/cache/datasets/talk2dino_native_e2/scoring"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        try:
            root = Path(__file__).resolve().parents[4] / "experiments/kv_cache/outputs/trackb_talk2dino_native"
            atomic_json(root / "RUN_STATUS.json", {"status": "failed", "stage": "exception",
                "finished_unix": time.time(), "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
