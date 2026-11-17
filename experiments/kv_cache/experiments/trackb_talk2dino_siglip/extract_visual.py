#!/usr/bin/env python3
"""Extract frozen DINOv2 final patch and attention-head region features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODULE = Path(__file__).resolve().parent
REPO = MODULE.parents[3]
sys.path.insert(0, str(REPO))

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.dino_features import final_patch_and_region_features
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.preprocessing import preprocess_talk2dino_image

MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_vision() -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    )
    if not hasattr(model, "vision_backbone") or not hasattr(model.vision_backbone, "fused_featurizer"):
        raise TypeError("Expected the OpenVLA fused DINOv2/SigLIP checkpoint")
    model.vision_backbone.to(device).eval()
    for parameter in model.language_model.parameters():
        if parameter.device.type != "cpu":
            raise AssertionError("The language model was unexpectedly moved to the accelerator")
    return model, processor, device, torch


def standard_openvla_visual_features(model: Any, processor: Any, image: Any, device: Any, torch: Any) -> Any:
    """Evaluate the untouched standard fused route for sidecar isolation checks."""
    vision = model.vision_backbone
    pixels = processor.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    if tuple(pixels.shape) != (1, 6, 224, 224):
        raise AssertionError(f"OpenVLA processor returned {tuple(pixels.shape)}, expected [1,6,224,224].")
    dino_pixels = pixels[:, :3].to(dtype=next(vision.featurizer.parameters()).dtype)
    siglip_pixels = pixels[:, 3:].to(dtype=next(vision.fused_featurizer.parameters()).dtype)
    with torch.inference_mode():
        return torch.cat([
            vision.featurizer(dino_pixels),
            vision.fused_featurizer(siglip_pixels),
        ], dim=-1)[0]


def process_images(model: Any, processor: Any, paths: list[Path], device: Any, torch: Any, verify_standard: bool = False, output_dtype: Any = np.float16) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    vision = model.vision_backbone
    dino = vision.featurizer
    rows: list[dict[str, Any]] = []
    regions_out: list[np.ndarray] = []
    patches_out: list[np.ndarray] = []
    attention_out: list[np.ndarray] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        dino_pixels = preprocess_talk2dino_image(image).unsqueeze(0).to(
            device=device, dtype=next(dino.parameters()).dtype
        )
        if tuple(dino_pixels.shape) != (1, 3, 224, 224):
            raise AssertionError(f"Talk2DINO transform returned {tuple(dino_pixels.shape)}")
        standard_a = None
        standard_ms = None
        std_max = None
        if verify_standard:
            # Smoke-only control: independently check the unchanged standard
            # OpenVLA path. Its pixels are not used by the Talk2DINO sidecar.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            standard_a = standard_openvla_visual_features(model, processor, image, device, torch)
            standard_b = standard_openvla_visual_features(model, processor, image, device, torch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            standard_ms = (time.perf_counter() - before) * 500.0
            std_max = float((standard_a - standard_b).abs().max().float().cpu())
            if not torch.equal(standard_a, standard_b):
                raise AssertionError(f"Standard OpenVLA vision path changed across identical calls (max diff={std_max}).")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode():
            patches, attn, regions = final_patch_and_region_features(dino, dino_pixels)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        final_ms = (time.perf_counter() - start) * 1000.0
        # Keep legacy full-dataset extraction in float16 by default; controlled
        # pilots may request float32 without changing feature computation.
        regions_out.append(regions[0].float().cpu().numpy().astype(output_dtype))
        patches_out.append(patches[0].float().cpu().numpy().astype(output_dtype))
        attention_out.append(attn[0].float().cpu().numpy().astype(output_dtype))
        rows.append({
            "image_path": str(path),
            "talk2dino_pixels_shape": list(dino_pixels.shape),
            "talk2dino_preprocessing": "Resize(short edge, bicubic, 224); CenterCrop(224); ToTensor; ImageNet mean/std",
            "dino_input_dtype": str(dino_pixels.dtype),
            "standard_openvla_fused_shape": None if standard_a is None else list(standard_a.shape),
            "standard_repeat_exact": None if standard_a is None else True,
            "standard_repeat_max_abs_diff": std_max,
            "standard_two_pass_mean_ms": standard_ms,
            "final_patch_shape": list(patches.shape),
            "cls_patch_attention_shape": list(attn.shape),
            "region_shape": list(regions.shape),
            "attention_min": float(attn.min().float().cpu()),
            "attention_max": float(attn.max().float().cpu()),
            "attention_entropy_mean": float((-(attn.float() * attn.float().clamp_min(1e-12).log()).sum(-1)).mean().cpu()),
            "attention_patch_sum_min": float(attn.sum(-1).min().float().cpu()),
            "attention_patch_sum_max": float(attn.sum(-1).max().float().cpu()),
            "final_path_ms": final_ms,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "patch_mean_l2": float(patches.float().norm(dim=-1).mean().cpu()),
            "patch_mean_abs": float(patches.float().abs().mean().cpu()),
        })
    return rows, regions_out, patches_out, attention_out


def _coco_image_paths(root: Path, split: str) -> list[tuple[int, Path]]:
    ann = root / "annotations2014" / f"captions_{split}2014.json"
    payload = json.loads(ann.read_text())
    image_dir = root / f"{split}2014"
    return [(int(row["id"]), image_dir / row["file_name"]) for row in payload["images"]]


def extract_full(args: argparse.Namespace) -> None:
    import torch

    root = args.dataset_root
    splits = ("train", "val") if args.split == "all" else (args.split,)
    feature_root = root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    model, processor, device, torch = load_vision()
    dino = model.vision_backbone.featurizer
    feature_dim = int(dino.num_features)
    heads = int(dino.blocks[-1].attn.num_heads)
    grid = tuple(int(v) for v in dino.patch_embed.grid_size)
    if grid != (16, 16):
        raise AssertionError(f"Expected 16x16 patch grid at 224px, got {grid}")
    if heads != 16 or feature_dim != 1024:
        raise AssertionError(f"Unexpected OpenVLA DINO architecture: heads={heads}, feature_dim={feature_dim}")
    for split in splits:
        entries = _coco_image_paths(root, split)
        if args.limit:
            entries = entries[: args.limit]
        ids_path = feature_root / f"image_ids_{split}2014.npy"
        regions_path = feature_root / f"region_features_{split}2014.npy"
        done_path = feature_root / f"region_done_{split}2014.npy"
        manifest_path = feature_root / f"visual_{split}2014_manifest.json"
        ids = np.asarray([image_id for image_id, _ in entries], dtype=np.int64)
        if ids_path.exists():
            old_ids = np.load(ids_path, mmap_mode="r")
            if not np.array_equal(old_ids, ids):
                raise RuntimeError(f"Existing image index does not match current annotation order: {ids_path}")
        else:
            np.save(ids_path, ids, allow_pickle=False)
        regions_new = not regions_path.exists()
        done_new = not done_path.exists()
        region_shape = (len(ids), heads, feature_dim)
        regions = np.lib.format.open_memmap(regions_path, mode="r+" if not regions_new else "w+", dtype=np.float16, shape=region_shape)
        done = np.lib.format.open_memmap(done_path, mode="r+" if not done_new else "w+", dtype=np.uint8, shape=(len(ids),))
        if done_new:
            done[:] = 0
        if regions_new:
            regions[:] = 0
        pending = np.flatnonzero(np.asarray(done) == 0).tolist()
        chunk = max(1, args.batch_size)
        times = []
        for offset in range(0, len(pending), chunk):
            batch_rows = pending[offset : offset + chunk]
            recs, feats, _patches, _attention = process_images(model, processor, [entries[i][1] for i in batch_rows], device, torch)
            for i, feature in zip(batch_rows, feats):
                regions[i] = feature
                done[i] = 1
            regions.flush()
            done.flush()
            times.extend(row["final_path_ms"] for row in recs)
            atomic_json(manifest_path, {"split": split, "count": len(ids), "completed": int(np.asarray(done).sum()), "region_shape": list(region_shape), "dtype": "float16", "feature_definition": "talk2dino_final_layer_cls_attention_weighted_patch_mean_v1", "preprocessing": "Resize(short edge, bicubic, 224); CenterCrop(224); ToTensor; ImageNet mean/std", "dino_model": "vit_large_patch14_reg4_dinov2.lvd142m", "dino_revision": MODEL_ID, "patch_grid": list(grid), "attention_heads": heads, "feature_dim": feature_dim, "source_image_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "recent_batch": recs, "mean_final_path_ms": float(np.mean(times)) if times else None, "updated_at_unix": time.time()})
            print(f"{split}2014: {int(np.asarray(done).sum())}/{len(ids)} images", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("/opt/openvla/cache/datasets/talk2dino/coco2014"))
    p.add_argument("--split", choices=("train", "val", "all"), default="train")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="bounded validation only; 0 means all images")
    p.add_argument("--smoke-images", nargs="*", type=Path)
    p.add_argument("--output", type=Path, default=REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/smoke")
    args = p.parse_args()
    if args.smoke_images:
        import torch

        model, processor, device, torch = load_vision()
        rows, regions, patches, attentions = process_images(model, processor, args.smoke_images, device, torch, verify_standard=True)
        args.output.mkdir(parents=True, exist_ok=True)
        np.save(args.output / "smoke_regions_fp16.npy", np.stack(regions), allow_pickle=False)
        np.save(args.output / "smoke_final_patches_fp16.npy", np.stack(patches), allow_pickle=False)
        np.save(args.output / "smoke_cls_patch_attention_fp16.npy", np.stack(attentions), allow_pickle=False)
        atomic_json(args.output / "smoke_report.json", {"model": MODEL_ID, "samples": rows, "region_array_shape": list(np.stack(regions).shape), "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda})
        print(json.dumps({"samples": len(rows), "report": str(args.output / "smoke_report.json")}, indent=2))
    else:
        extract_full(args)


if __name__ == "__main__":
    main()
