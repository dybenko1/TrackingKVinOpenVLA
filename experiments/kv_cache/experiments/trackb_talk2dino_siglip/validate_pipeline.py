#!/usr/bin/env python3
"""Bounded real-checkpoint validation; no extraction, training, or rollout."""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

MODULE = Path(__file__).resolve().parent
REPO = MODULE.parents[3]
sys.path.insert(0, str(REPO))

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.dino_features import final_patch_and_region_features
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.extract_visual import load_vision, process_images, standard_openvla_visual_features
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.losses import symmetric_infonce
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import encode_text_embeddings, load_siglip_text_tower
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.tokenizer import load_siglip_tokenizer, read_siglip_config, validate_text_config
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.config import SIGLIP_TEXT_MODEL_REVISION


MODEL_NAME = "ViT-SO400M-14-SigLIP-384"
DEFAULT_SIGLIP = Path("/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384")
DEFAULT_REPORT = Path("/opt/openvla/experiments/kv_cache/outputs/trackb_talk2dino_siglip/real_model_smoke_test.json")
DEFAULT_CAPTIONS = (
    "A robot arm and objects on a tabletop.",
    "A small object near a robot gripper.",
    "Objects arranged on a table in a robot workspace.",
)


def rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB; this is the process high-water RSS.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    diff = (reference.float() - candidate.float()).abs()
    rel = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return {
        "exact_equal": bool(torch.equal(reference, candidate)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "relative_l2": float(rel.item()),
        "allclose_atol_2e-5_rtol_1e-5": bool(torch.allclose(reference, candidate, atol=2e-5, rtol=1e-5)),
    }


def full_openclip_reference(checkpoint: Path, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Load the complete checkpoint on CPU solely to compare its text output."""
    import open_clip

    model = open_clip.create_model(MODEL_NAME, pretrained=None, device="cpu", force_custom_text=True)
    state = load_file(str(checkpoint / "open_clip_model.safetensors"), device="cpu")
    load_info = model.load_state_dict(state, strict=True)
    del state
    if load_info.missing_keys or load_info.unexpected_keys:
        raise RuntimeError(f"Full OpenCLIP checkpoint strict load mismatch: {load_info}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        output = model.encode_text(tokens, normalize=False)
    if output.shape != (tokens.shape[0], 1152) or not torch.isfinite(output).all():
        raise AssertionError(f"Full checkpoint text output invalid: {tuple(output.shape)}")
    info = {
        "open_clip_version": open_clip.__version__,
        "strict_full_state_load": True,
        "state_tensor_count": len(model.state_dict()),
        "loaded_device": "cpu",
        "vision_tower_loaded_to_gpu": False,
    }
    del model
    return output.cpu(), info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, nargs="+", required=True, help="One to four previously saved RGB observations")
    parser.add_argument("--caption", nargs="+", default=None, help="One caption per image; defaults to short generic prompts")
    parser.add_argument("--siglip-dir", type=Path, default=DEFAULT_SIGLIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-images", type=int, default=4)
    args = parser.parse_args()
    if len(args.image) > args.max_images:
        raise ValueError(f"Bounded smoke is limited to {args.max_images} images")
    captions = list(args.caption or DEFAULT_CAPTIONS[:len(args.image)])
    if len(captions) != len(args.image):
        raise ValueError("Supply exactly one --caption per image")
    missing_images = [str(path) for path in args.image if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Smoke images are missing: {missing_images}")

    # Assert the exact checkpoint config and tokenizer metadata before loading.
    cfg = read_siglip_config(args.siglip_dir)
    dims = validate_text_config(cfg)
    text_cfg = cfg["model_cfg"]["text_cfg"]
    expected = {"context_length": 64, "embed_dim": 1152, "width": 1152, "layers": 27, "heads": 16, "clean": "canonicalize"}
    observed = {"context_length": dims["context_length"], "embed_dim": dims["output_dim"], "width": dims["text_width"], "layers": int(text_cfg["layers"]), "heads": int(text_cfg["heads"]), "clean": text_cfg.get("tokenizer_kwargs", {}).get("clean")}
    if observed != expected:
        raise AssertionError(f"SigLIP-384 checkpoint architecture mismatch: {observed}")

    tokenizer = load_siglip_tokenizer(args.siglip_dir)
    token_ids = tokenizer(captions)
    if tuple(token_ids.shape) != (len(captions), dims["context_length"]):
        raise AssertionError(f"Tokenizer output has unexpected shape {tuple(token_ids.shape)}")
    if tokenizer.context_length != dims["context_length"]:
        raise AssertionError("Tokenizer and model context lengths differ")

    full_started = time.perf_counter()
    full_output, full_load_info = full_openclip_reference(args.siglip_dir, token_ids)
    full_reference_seconds = time.perf_counter() - full_started

    text_tower, text_tokenizer, text_meta = load_siglip_text_tower(args.siglip_dir, "cpu")
    if text_meta["text_positional_embedding_shape"] != [64, 1152] or not text_meta["strict_text_state_load"]:
        raise AssertionError(f"Text-only checkpoint validation failed: {text_meta}")
    if text_tokenizer.context_length != tokenizer.context_length:
        raise AssertionError("Loaded text encoder and shared tokenizer context mismatch")
    text_a = encode_text_embeddings(text_tower, token_ids, expected_dim=1152).cpu()
    text_b = encode_text_embeddings(text_tower, token_ids, expected_dim=1152).cpu()
    text_compare = compare_tensors(full_output, text_a)
    deterministic = compare_tensors(text_a, text_b)
    if not text_compare["allclose_atol_2e-5_rtol_1e-5"]:
        raise AssertionError(f"Text-only tower differs from full checkpoint: {text_compare}")
    if not deterministic["exact_equal"]:
        raise AssertionError(f"Text tower is not deterministic in eval mode: {deterministic}")

    # Now load OpenVLA's existing vision module. SigLIP text and the full
    # checkpoint reference remain on CPU; only OpenVLA vision is on CUDA.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real OpenVLA DINOv2 feature smoke test")
    model, processor, device, _ = load_vision()
    torch.cuda.synchronize(device)
    cuda_live_after_model_load = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    visual_rows = []
    region_rows = []
    sidecar_isolation = []
    for image_path in args.image:
        image = Image.open(image_path).convert("RGB")
        standard_before = standard_openvla_visual_features(model, processor, image, device, torch).detach().clone()
        rows, regions, patches, attention = process_images(model, processor, [image_path], device, torch, verify_standard=False)
        standard_after = standard_openvla_visual_features(model, processor, image, device, torch)
        comparison = compare_tensors(standard_before, standard_after)
        if not comparison["exact_equal"]:
            raise AssertionError(f"Inactive sidecar changed normal OpenVLA visual output: {comparison}")
        if patches[0].shape != (256, 1024) or attention[0].shape != (16, 256) or regions[0].shape != (16, 1024):
            raise AssertionError(f"Unexpected DINO feature shapes: {patches[0].shape}, {attention[0].shape}, {regions[0].shape}")
        if not np.isfinite(regions[0]).all():
            raise FloatingPointError("Nonfinite DINO region features")
        visual_rows.append(rows[0])
        region_rows.append(torch.from_numpy(regions[0].astype(np.float32)))
        sidecar_isolation.append(comparison)

    regions_tensor = torch.stack(region_rows)
    # SigLIP inference returns inference-mode tensors. The following CPU MLP
    # backward test needs ordinary tensors even though those features stay frozen.
    with torch.inference_mode(False):
        text_a = text_a.clone()
        regions_tensor = regions_tensor.clone()
    projection = RegionTextProjection().cpu()
    projection_optimizer = torch.optim.Adam(projection.parameters(), lr=1e-4)
    projection_optimizer.zero_grad(set_to_none=True)
    scores = projection(regions_tensor, text_a)
    if tuple(scores.shape) != (len(args.image), len(args.image)) or not torch.isfinite(scores).all():
        raise AssertionError(f"Cross-modal similarity matrix invalid: {tuple(scores.shape)}")
    loss = symmetric_infonce(scores, temperature=0.07)
    loss.backward()
    mlp_grads = [p.grad for p in projection.parameters()]
    if not all(g is not None and torch.isfinite(g).all() for g in mlp_grads):
        raise AssertionError("MLP gradients are missing or nonfinite")
    if any(parameter.grad is not None for parameter in text_tower.parameters()):
        raise AssertionError("Frozen SigLIP Text tower received gradients")
    if any(parameter.grad is not None for parameter in model.vision_backbone.parameters()):
        raise AssertionError("Frozen OpenVLA vision backbone received gradients")
    optimizer_ids = {id(p) for group in projection_optimizer.param_groups for p in group["params"]}
    if optimizer_ids != {id(p) for p in projection.parameters()}:
        raise AssertionError("Projection optimizer contains non-MLP parameters")

    torch.cuda.synchronize(device)
    report = {
        "status": "passed",
        "checkpoint": {"repository": "timm/ViT-SO400M-14-SigLIP-384", "revision": SIGLIP_TEXT_MODEL_REVISION, "path": str(args.siglip_dir), "config": observed, "tokenizer_shape": list(token_ids.shape)},
        "full_checkpoint_reference": {**full_load_info, "reference_load_and_text_seconds": full_reference_seconds, "text_only_vs_full": text_compare},
        "text_only_tower": {**text_meta, "embedding_shape": list(text_a.shape), "finite": bool(torch.isfinite(text_a).all()), "repeat": deterministic, "frozen": all(not p.requires_grad for p in text_tower.parameters())},
        "dino_smoke": {"openvla_checkpoint": "openvla/openvla-7b-finetuned-libero-spatial", "device": str(device), "images": [str(p) for p in args.image], "samples": visual_rows, "standard_route_before_after_sidecar": sidecar_isolation},
        "cross_modal_smoke": {"region_shape": list(regions_tensor.shape), "text_shape": list(text_a.shape), "similarity_shape": list(scores.shape), "similarities_finite": bool(torch.isfinite(scores).all()), "loss": float(loss.detach()), "loss_finite": bool(torch.isfinite(loss)), "mlp_gradients_finite": True, "frozen_backbone_gradients_absent": True, "optimizer_contains_only_mlp": True},
        "resource_usage": {"process_peak_rss_bytes": rss_bytes(), "cuda_live_allocated_after_openvla_load_bytes": cuda_live_after_model_load, "cuda_peak_allocated_bytes_after_model_load": int(torch.cuda.max_memory_allocated(device)), "cuda_peak_reserved_bytes_after_model_load": int(torch.cuda.max_memory_reserved(device)), "gpu_name": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda},
        "policy_actions_generated": False,
        "training_started": False,
        "full_feature_extraction_started": False,
        "libero_rollouts_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "text_only_vs_full": text_compare, "text_shape": list(text_a.shape), "regions": list(regions_tensor.shape), "similarity": list(scores.shape), "resource_usage": report["resource_usage"]}, indent=2))


if __name__ == "__main__":
    main()
