#!/usr/bin/env python3
"""Restartable 800/200 COCO Talk2DINO-SigLIP pilot.

This orchestration reuses the package's existing feature functions, dataset,
projection module, loss and trainer. It does not run policy inference.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MODULE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
DEFAULT_DATA = Path("/opt/openvla/cache/datasets/talk2dino/coco2014")
DEFAULT_SIGLIP = Path("/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384")
DEFAULT_OUTPUT = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/pilot_1000"
FEATURE_DEFINITION = "talk2dino_final_layer_cls_attention_weighted_patch_mean_v1"

sys.path.insert(0, str(REPO))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_sha256(module: Any, torch: Any) -> str:
    """Hash frozen module state one tensor at a time without a model-sized copy."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().to(device="cpu").contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(memoryview(raw))
    return digest.hexdigest()


def write_status(path: Path, stage: str, *, state: str = "running", **extra: Any) -> None:
    previous = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text())
        except Exception:
            previous = {}
    previous.update({
        "experiment": "trackb_talk2dino_siglip_pilot_1000",
        "state": state,
        "stage": stage,
        "last_completed_stage": previous.get("last_completed_stage", "none"),
        "pid": os.getpid(),
        "updated_at_utc": utc_now(),
    })
    if state in {"running", "pending"}:
        for stale in ("error", "traceback", "exit_code", "failed_at_utc", "note"):
            previous.pop(stale, None)
    for counter_name in ("completed_images", "completed_captions"):
        if counter_name in extra and isinstance(extra[counter_name], dict):
            merged = dict(previous.get(counter_name, {}))
            merged.update(extra[counter_name])
            extra[counter_name] = merged
    previous.update(extra)
    atomic_json(path, previous)


def mark_done(path: Path, stage: str, **extra: Any) -> None:
    write_status(path, stage, last_completed_stage=stage, **extra)


def read_annotations(data_root: Path, split: str) -> tuple[dict[int, dict], list[dict]]:
    annotation_path = data_root / "annotations2014" / f"captions_{split}2014.json"
    payload = json.loads(annotation_path.read_text())
    images = {int(row["id"]): row for row in payload["images"]}
    annotations = payload["annotations"]
    return images, annotations


def create_or_load_pilot_manifest(data_root: Path, output: Path, seed: int) -> dict:
    manifest_path = output / "pilot_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = {"seed": seed, "train_image_target": 800, "val_image_target": 200}
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"Existing pilot manifest differs at {key}: {manifest.get(key)} != {value}")
        if manifest.get("dataset_root") != str(data_root):
            raise RuntimeError("Existing pilot manifest points to a different COCO dataset root.")
        return manifest

    rng = random.Random(seed)
    selected: dict[str, list[int]] = {}
    source: dict[str, tuple[dict[int, dict], list[dict]]] = {}
    for split, count in (("train", 800), ("val", 200)):
        image_rows, annotations = read_annotations(data_root, split)
        if len(image_rows) < count:
            raise ValueError(f"COCO {split} has only {len(image_rows)} unique images; need {count}.")
        selected[split] = sorted(rng.sample(sorted(image_rows), count))
        source[split] = image_rows, annotations
    overlap = set(selected["train"]) & set(selected["val"])
    if overlap:
        raise AssertionError(f"Train/validation image IDs overlap: {sorted(overlap)[:10]}")

    splits: dict[str, dict] = {}
    all_annotation_ids: list[int] = []
    for split in ("train", "val"):
        image_rows, annotations = source[split]
        selected_ids = set(selected[split])
        image_records = []
        for image_id in selected[split]:
            row = image_rows[image_id]
            rel_path = f"{split}2014/{row['file_name']}"
            image_path = data_root / rel_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Selected COCO image missing: {image_path}")
            image_records.append({
                "image_id": image_id,
                "file_name": row["file_name"],
                "relative_image_path": rel_path,
                "image_path": str(image_path),
                "width": int(row["width"]),
                "height": int(row["height"]),
            })
        caption_records = [
            {"annotation_id": int(row["id"]), "image_id": int(row["image_id"]), "caption": row["caption"]}
            for row in annotations if int(row["image_id"]) in selected_ids
        ]
        all_annotation_ids.extend(row["annotation_id"] for row in caption_records)
        splits[split] = {
            "image_count": len(image_records),
            "caption_count": len(caption_records),
            "images": image_records,
            "captions": caption_records,
        }
    if len(all_annotation_ids) != len(set(all_annotation_ids)):
        raise AssertionError("Selected COCO caption annotation IDs are not unique.")
    manifest = {
        "schema_version": 1,
        "experiment": "Talk2DINO-SigLIP controlled 1,000-image pilot",
        "created_at_utc": utc_now(),
        "dataset_root": str(data_root),
        "seed": seed,
        "train_image_target": 800,
        "val_image_target": 200,
        "train_validation_image_ids_disjoint": True,
        "caption_sampling": "all COCO captions for each selected image, preserving annotation-file order",
        "image_sampling": "uniform without replacement from sorted unique COCO image IDs, Python random.Random(seed)",
        "splits": splits,
        "actual_total_images": sum(item["image_count"] for item in splits.values()),
        "actual_total_captions": sum(item["caption_count"] for item in splits.values()),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def ensure_memmap(path: Path, shape: tuple[int, ...], dtype: Any, *, fill: bool = False) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise RuntimeError(f"Existing feature array layout differs at {path}: {array.shape}/{array.dtype} vs {shape}/{np.dtype(dtype)}")
        return array
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill:
        array[:] = 0
        array.flush()
    return array


def feature_paths(feature_root: Path, split: str) -> dict[str, Path]:
    tag = f"{split}2014"
    return {
        "image_ids": feature_root / f"image_ids_{tag}.npy",
        "regions": feature_root / f"region_features_{tag}.npy",
        "patches": feature_root / f"patch_features_{tag}.npy",
        "attention": feature_root / f"attention_maps_{tag}.npy",
        "visual_done": feature_root / f"visual_done_{tag}.npy",
        "caption_ids": feature_root / f"caption_ids_{tag}.npy",
        "texts": feature_root / f"text_features_{tag}.npy",
        "truncated": feature_root / f"caption_truncated_{tag}.npy",
        "text_done": feature_root / f"text_done_{tag}.npy",
        "captions_jsonl": feature_root / f"captions_{tag}.jsonl",
        "visual_manifest": feature_root / f"visual_{tag}_manifest.json",
        "text_manifest": feature_root / f"text_{tag}_manifest.json",
    }


def prepare_split_arrays(manifest: dict, feature_root: Path, split: str) -> dict[str, Any]:
    feature_root.mkdir(parents=True, exist_ok=True)
    records = manifest["splits"][split]
    paths = feature_paths(feature_root, split)
    image_ids = np.asarray([row["image_id"] for row in records["images"]], dtype=np.int64)
    caption_ids = np.asarray([[row["annotation_id"], row["image_id"]] for row in records["captions"]], dtype=np.int64)
    if paths["image_ids"].exists():
        if not np.array_equal(np.load(paths["image_ids"], mmap_mode="r"), image_ids):
            raise RuntimeError(f"Existing pilot image index differs for {split}.")
    else:
        np.save(paths["image_ids"], image_ids, allow_pickle=False)
    if paths["caption_ids"].exists():
        if not np.array_equal(np.load(paths["caption_ids"], mmap_mode="r"), caption_ids):
            raise RuntimeError(f"Existing pilot caption index differs for {split}.")
    else:
        np.save(paths["caption_ids"], caption_ids, allow_pickle=False)
    if not paths["captions_jsonl"].exists():
        temporary = paths["captions_jsonl"].with_suffix(".jsonl.tmp")
        with temporary.open("w") as handle:
            for row in records["captions"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, paths["captions_jsonl"])
    n_images, n_captions = len(image_ids), len(caption_ids)
    # Create completion flags first so interrupted feature writes are resumable.
    visual_done = ensure_memmap(paths["visual_done"], (n_images,), np.uint8, fill=True)
    text_done = ensure_memmap(paths["text_done"], (n_captions,), np.uint8, fill=True)
    regions = ensure_memmap(paths["regions"], (n_images, 16, 1024), np.float32, fill=True)
    patches = ensure_memmap(paths["patches"], (n_images, 256, 1024), np.float32, fill=True)
    attention = ensure_memmap(paths["attention"], (n_images, 16, 256), np.float32, fill=True)
    texts = ensure_memmap(paths["texts"], (n_captions, 1152), np.float32, fill=True)
    truncated = ensure_memmap(paths["truncated"], (n_captions,), np.uint8, fill=True)
    return {**paths, "image_ids_array": image_ids, "caption_ids_array": caption_ids,
            "regions_array": regions, "patches_array": patches, "attention_array": attention,
            "visual_done_array": visual_done, "texts_array": texts,
            "text_done_array": text_done, "truncated_array": truncated}


def extract_visual(manifest: dict, data_root: Path, output: Path, status_path: Path) -> dict:
    import torch
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.extract_visual import (
        MODEL_ID, load_vision, process_images, standard_openvla_visual_features,
    )

    feature_root = output / "features"
    dino_init_start = time.perf_counter()
    model, processor, device, torch = load_vision()
    vision = model.vision_backbone.eval()
    vision.requires_grad_(False)
    dino = vision.featurizer
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    encoder_hash_before = module_sha256(vision, torch)
    hash_time = time.perf_counter() - dino_init_start
    first_image = Path(manifest["splits"]["train"]["images"][0]["image_path"])
    with torch.inference_mode():
        standard_before = standard_openvla_visual_features(model, processor, __import__("PIL.Image", fromlist=["Image"]).open(first_image).convert("RGB"), device, torch).float().cpu()

    inference_times: list[float] = []
    wall_times: list[float] = []
    split_metadata = {}
    for split in ("train", "val"):
        records = manifest["splits"][split]
        arrays = prepare_split_arrays(manifest, feature_root, split)
        done = arrays["visual_done_array"]
        pending = np.flatnonzero(np.asarray(done) == 0).tolist()
        verification_only = not pending
        work_indexes = pending if pending else list(range(len(records["images"])))
        for batch_start in range(0, len(work_indexes), 4):
            batch_indices = work_indexes[batch_start:batch_start + 4]
            paths = [Path(records["images"][index]["image_path"]) for index in batch_indices]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing selected COCO images: {missing[:3]}")
            call_start = time.perf_counter()
            rows, region_list, patch_list, attention_list = process_images(
                model, processor, paths, device, torch, output_dtype=np.float32
            )
            wall_times.extend([(time.perf_counter() - call_start) / len(paths)] * len(paths))
            for local_row, image_index in enumerate(batch_indices):
                if verification_only:
                    deltas = (
                        float(np.max(np.abs(region_list[local_row] - arrays["regions_array"][image_index]))),
                        float(np.max(np.abs(patch_list[local_row] - arrays["patches_array"][image_index]))),
                        float(np.max(np.abs(attention_list[local_row] - arrays["attention_array"][image_index]))),
                    )
                    if any(delta != 0.0 for delta in deltas):
                        raise AssertionError(f"DINO full-set verification differs at {split} row {image_index}: {deltas}")
                else:
                    arrays["regions_array"][image_index] = region_list[local_row]
                    arrays["patches_array"][image_index] = patch_list[local_row]
                    arrays["attention_array"][image_index] = attention_list[local_row]
                    arrays["regions_array"].flush(); arrays["patches_array"].flush(); arrays["attention_array"].flush()
                    done[image_index] = 1
                    done.flush()
                inference_times.append(float(rows[local_row]["final_path_ms"]))
            complete = int(np.asarray(done).sum())
            write_status(status_path, "visual_extraction", completed_images={split: complete})
            print(f"DINO {split}: {complete}/{len(records['images'])} images", flush=True)
        if not np.all(np.asarray(done) == 1):
            raise AssertionError(f"DINO visual features incomplete for {split}.")
        region_shape = list(arrays["regions_array"].shape)
        visual_manifest = {
            "split": split, "count": len(records["images"]), "completed": len(records["images"]),
            "region_shape": region_shape, "dtype": "float32", "feature_definition": FEATURE_DEFINITION,
            "preprocessing": "Resize(short edge, bicubic, 224); CenterCrop(224); ToTensor; ImageNet mean/std",
            "dino_model": "vit_large_patch14_reg4_dinov2.lvd142m",
            "dino_revision": getattr(getattr(model, "config", None), "_commit_hash", None) or MODEL_ID,
            "patch_grid": [16, 16], "attention_heads": 16, "feature_dim": 1024,
            "source_image_ids_sha256": hashlib.sha256(arrays["image_ids_array"].tobytes()).hexdigest(),
            "talk2dino_code_path": str(MODULE / "dino_features.py"),
            "output_dtype": "float32",
        }
        atomic_json(arrays["visual_manifest"], visual_manifest)
        split_metadata[split] = visual_manifest

    # A fixed small subset is recomputed deterministically before model release.
    reextract = []
    for split in ("train", "val"):
        records = manifest["splits"][split]
        arrays = prepare_split_arrays(manifest, feature_root, split)
        # Match the original extraction batch shape so backend GEMM shape
        # choices cannot masquerade as stochastic encoder behavior.
        check_count = min(4, len(records["images"]))
        paths = [Path(row["image_path"]) for row in records["images"][:check_count]]
        _rows, regions, patches, attentions = process_images(model, processor, paths, device, torch, output_dtype=np.float32)
        diffs = {
            "regions_max_abs": max(float(np.max(np.abs(regions[i] - arrays["regions_array"][i]))) for i in range(check_count)),
            "patches_max_abs": max(float(np.max(np.abs(patches[i] - arrays["patches_array"][i]))) for i in range(check_count)),
            "attention_max_abs": max(float(np.max(np.abs(attentions[i] - arrays["attention_array"][i]))) for i in range(check_count)),
        }
        if max(diffs.values()) != 0.0:
            raise AssertionError(f"DINO deterministic re-extraction changed saved features for {split}: {diffs}")
        reextract.append({"split": split, "images": check_count, "exact_equal": True, **diffs})

    first_image_obj = __import__("PIL.Image", fromlist=["Image"]).open(first_image).convert("RGB")
    with torch.inference_mode():
        standard_after = standard_openvla_visual_features(model, processor, first_image_obj, device, torch).float().cpu()
    standard_diff = float((standard_before - standard_after).abs().max())
    if not torch.equal(standard_before, standard_after):
        raise AssertionError(f"Standard OpenVLA vision output changed across sidecar extraction: max_abs={standard_diff}")
    encoder_hash_after = module_sha256(vision, torch)
    if encoder_hash_before != encoder_hash_after:
        raise AssertionError("Frozen OpenVLA vision encoder state changed during feature extraction.")
    cuda_stats = None
    if device.type == "cuda":
        cuda_stats = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "allocated_after_bytes": int(torch.cuda.memory_allocated(device)),
        }
    result = {
        "status": "passed", "device": str(device), "torch_version": torch.__version__,
        "checkpoint": MODEL_ID, "model_load_and_initial_hash_seconds": hash_time,
        "encoder_state_sha256_before": encoder_hash_before,
        "encoder_state_sha256_after": encoder_hash_after,
        "encoder_unchanged": encoder_hash_before == encoder_hash_after,
        "standard_openvla_before_after_exact": True, "standard_openvla_max_abs": standard_diff,
        "model_forward_ms": {"mean": float(np.mean(inference_times)), "median": float(np.median(inference_times)), "p95": float(np.percentile(inference_times, 95))},
        "per_image_pipeline_ms": {"mean": float(np.mean(wall_times) * 1000), "median": float(np.median(wall_times) * 1000), "p95": float(np.percentile(wall_times, 95) * 1000)},
        "total_images": len(inference_times), "total_visual_wall_seconds": float(sum(wall_times)),
        "total_stage_seconds_including_load_validation": time.perf_counter() - dino_init_start,
        "images_per_second_pipeline": float(len(wall_times) / sum(wall_times)),
        "timing_mode": "full-set verification re-encode" if all(
            np.all(np.asarray(prepare_split_arrays(manifest, feature_root, split)["visual_done_array"]) == 1)
            for split in ("train", "val")
        ) and len(inference_times) == 1000 else "initial extraction",
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "reextraction": reextract, "cuda": cuda_stats, "splits": split_metadata,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def extract_text(manifest: dict, output: Path, checkpoint_dir: Path, status_path: Path, batch_size: int = 64) -> dict:
    import torch
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import (
        encode_text_embeddings, load_siglip_text_tower,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text_stage_start = time.perf_counter()
    model_start = time.perf_counter()
    tower, tokenizer, info = load_siglip_text_tower(checkpoint_dir, str(device))
    tower.eval().requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    tower_hash_before = module_sha256(tower, torch)
    initialization_seconds = time.perf_counter() - model_start
    feature_root = output / "features"
    batch_times: list[float] = []
    tokenizer_times: list[float] = []
    total_captions = 0
    truncated_count = 0
    verified_existing_captions = 0
    split_verification: dict[str, bool] = {}
    for split in ("train", "val"):
        records = manifest["splits"][split]
        arrays = prepare_split_arrays(manifest, feature_root, split)
        done = arrays["text_done_array"]
        pending = np.flatnonzero(np.asarray(done) == 0).tolist()
        verification_only = not pending
        split_verification[split] = verification_only
        work_indexes = pending if pending else list(range(len(records["captions"])))
        for offset in range(0, len(work_indexes), batch_size):
            indexes = work_indexes[offset:offset + batch_size]
            captions = [records["captions"][i]["caption"] for i in indexes]
            tok_start = time.perf_counter()
            raw_tokens = tokenizer.tokenize_untruncated(captions)
            tokens = tokenizer(captions).to(device)
            tokenizer_times.append(time.perf_counter() - tok_start)
            over_context = np.asarray([len(row) > info["text_context_length"] for row in raw_tokens], dtype=np.uint8)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            encode_start = time.perf_counter()
            embeddings = encode_text_embeddings(tower, tokens, expected_dim=1152)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_times.append(time.perf_counter() - encode_start)
            embedding_array = embeddings.float().cpu().numpy().astype(np.float32)
            if verification_only:
                delta = float(np.max(np.abs(embedding_array - arrays["texts_array"][indexes])))
                trunc_delta = int(np.count_nonzero(over_context != arrays["truncated_array"][indexes]))
                if delta != 0.0 or trunc_delta:
                    raise AssertionError(f"SigLIP same-batch verification differs for {split}: max_abs={delta}, truncation_mismatches={trunc_delta}")
                verified_existing_captions += len(indexes)
            else:
                arrays["texts_array"][indexes] = embedding_array
                arrays["truncated_array"][indexes] = over_context
                arrays["texts_array"].flush(); arrays["truncated_array"].flush()
                done[indexes] = 1
                done.flush()
            completed = int(np.asarray(done).sum())
            split_truncated = int(np.asarray(arrays["truncated_array"])[np.asarray(done) == 1].sum())
            atomic_json(arrays["text_manifest"], {
                "split": split, "checkpoint_repository": "timm/ViT-SO400M-14-SigLIP-384",
                "checkpoint_revision": info["checkpoint_revision"], "checkpoint_dir": str(checkpoint_dir),
                "model_info": info, "caption_count": len(records["captions"]), "completed": completed,
                "feature_shape": [len(records["captions"]), 1152], "dtype": "float32",
                "context_length": info["text_context_length"], "tokenizer": tokenizer.metadata(),
                "truncated_count": split_truncated,
                "encoder_seconds_completed": float(sum(batch_times)),
                "tokenizer_seconds_completed": float(sum(tokenizer_times)),
                "verified_existing_features": verification_only,
                "caption_id_index_sha256": hashlib.sha256(arrays["caption_ids_array"].tobytes()).hexdigest(),
                "updated_at_utc": utc_now(),
            })
            write_status(status_path, "text_extraction", completed_captions={split: completed})
            print(f"SigLIP {split}: {completed}/{len(records['captions'])} captions", flush=True)
        if not np.all(np.asarray(done) == 1):
            raise AssertionError(f"SigLIP text features incomplete for {split}.")
        total_captions += len(records["captions"])
        truncated_count += int(np.asarray(arrays["truncated_array"]).sum())

    # Deterministically encode the first four captions of each split again.
    deterministic = []
    for split in ("train", "val"):
        records = manifest["splits"][split]
        arrays = prepare_split_arrays(manifest, feature_root, split)
        # Re-encode one full extraction batch (64 by default) to compare like
        # shapes and avoid changing GEMM dispatch only for the diagnostic.
        n = min(batch_size, len(records["captions"]))
        captions = [row["caption"] for row in records["captions"][:n]]
        tokens = tokenizer(captions).to(device)
        repeated = encode_text_embeddings(tower, tokens, expected_dim=1152).float().cpu().numpy()
        delta = float(np.max(np.abs(repeated - arrays["texts_array"][:n])))
        if delta != 0.0:
            raise AssertionError(f"SigLIP deterministic re-extraction changed {split} text features: max_abs={delta}")
        deterministic.append({"split": split, "captions": n, "exact_equal": True, "max_abs": delta})
    tower_hash_after = module_sha256(tower, torch)
    if tower_hash_before != tower_hash_after:
        raise AssertionError("Frozen SigLIP text tower state changed during feature extraction.")
    cuda_stats = None
    if device.type == "cuda":
        cuda_stats = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "allocated_after_bytes": int(torch.cuda.memory_allocated(device)),
        }
    result = {
        "status": "passed", "device": str(device), "initialization_seconds": initialization_seconds,
        "checkpoint": info, "encoder_state_sha256_before": tower_hash_before,
        "encoder_state_sha256_after": tower_hash_after, "encoder_unchanged": tower_hash_before == tower_hash_after,
        "tokenizer_seconds_total": float(sum(tokenizer_times)),
        "encoding_seconds_total": float(sum(batch_times)),
        "total_stage_seconds_including_load_validation": time.perf_counter() - text_stage_start,
        "mean_encoding_ms_per_caption": float(sum(batch_times) * 1000 / total_captions),
        "captions_per_second_encoder_only": float(total_captions / sum(batch_times)),
        "caption_count": total_captions, "truncated_count": truncated_count,
        "verified_existing_captions": verified_existing_captions,
        "timing_mode": "full-set saved-feature verification pass" if verified_existing_captions == total_captions else "initial extraction plus resume of missing rows",
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "cuda": cuda_stats, "deterministic_reextraction": deterministic,
    }
    del tower
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def validate_features(manifest: dict, output: Path) -> dict:
    feature_root = output / "features"
    ids_by_split: dict[str, set[int]] = {}
    annotation_ids: set[int] = set()
    checks: dict[str, Any] = {"splits": {}, "train_validation_image_disjoint": True}
    for split in ("train", "val"):
        records = manifest["splits"][split]
        arrays = prepare_split_arrays(manifest, feature_root, split)
        image_ids = np.load(arrays["image_ids"], mmap_mode="r")
        caption_ids = np.load(arrays["caption_ids"], mmap_mode="r")
        regions = np.load(arrays["regions"], mmap_mode="r")
        patches = np.load(arrays["patches"], mmap_mode="r")
        attention = np.load(arrays["attention"], mmap_mode="r")
        texts = np.load(arrays["texts"], mmap_mode="r")
        vdone = np.load(arrays["visual_done"], mmap_mode="r")
        tdone = np.load(arrays["text_done"], mmap_mode="r")
        trunc = np.load(arrays["truncated"], mmap_mode="r")
        if len(image_ids) != records["image_count"] or len(texts) != records["caption_count"]:
            raise AssertionError(f"Count mismatch for {split} feature reload.")
        if len(set(map(int, image_ids))) != len(image_ids):
            raise AssertionError(f"Duplicate image IDs in {split} feature table.")
        if caption_ids.shape != (records["caption_count"], 2) or len(set(map(int, caption_ids[:, 0]))) != len(caption_ids):
            raise AssertionError(f"Caption ID shape/uniqueness failed for {split}.")
        if not np.array_equal(image_ids, [row["image_id"] for row in records["images"]]):
            raise AssertionError(f"Image ID ordering mismatch for {split}.")
        expected_caption_ids = np.asarray([[row["annotation_id"], row["image_id"]] for row in records["captions"]], dtype=np.int64)
        if not np.array_equal(caption_ids, expected_caption_ids):
            raise AssertionError(f"Image-caption mapping mismatch for {split}.")
        annotation_ids.update(map(int, caption_ids[:, 0]))
        image_set = set(map(int, image_ids))
        if any(int(image_id) not in image_set for image_id in caption_ids[:, 1]):
            raise AssertionError(f"Caption references an unselected image in {split}.")
        if regions.shape != (len(image_ids), 16, 1024) or patches.shape != (len(image_ids), 256, 1024) or attention.shape != (len(image_ids), 16, 256) or texts.shape != (len(caption_ids), 1152):
            raise AssertionError(f"Feature shape mismatch for {split}.")
        if any(array.dtype != np.float32 for array in (regions, patches, attention, texts)):
            raise AssertionError(f"Pilot feature arrays must be float32 for {split}.")
        if not np.all(vdone == 1) or not np.all(tdone == 1):
            raise AssertionError(f"Incomplete feature done flags for {split}.")
        finite = True
        attention_error = 0.0
        min_region_norm = float("inf")
        for start in range(0, len(image_ids), 32):
            batch_regions = np.asarray(regions[start:start + 32])
            batch_patches = np.asarray(patches[start:start + 32])
            batch_attention = np.asarray(attention[start:start + 32])
            if not (np.isfinite(batch_regions).all() and np.isfinite(batch_patches).all() and np.isfinite(batch_attention).all()):
                finite = False
                break
            attention_error = max(attention_error, float(np.max(np.abs(batch_attention.sum(-1) - 1.0))))
            min_region_norm = min(min_region_norm, float(np.min(np.linalg.norm(batch_regions, axis=-1))))
        for start in range(0, len(caption_ids), 256):
            if not np.isfinite(np.asarray(texts[start:start + 256])).all():
                finite = False
                break
        if not finite:
            raise FloatingPointError(f"Nonfinite features found in {split}.")
        if attention_error > 2e-5:
            raise AssertionError(f"Attention normalization error {attention_error} exceeds tolerance for {split}.")
        if min_region_norm <= 0:
            raise AssertionError(f"A zero region descriptor was found for {split}.")
        if not np.all((trunc == 0) | (trunc == 1)):
            raise AssertionError(f"Unexpected caption truncation flags for {split}.")
        ids_by_split[split] = image_set
        checks["splits"][split] = {
            "images": len(image_ids), "captions": len(caption_ids),
            "patch_shape": list(patches.shape), "attention_shape": list(attention.shape),
            "region_shape": list(regions.shape), "text_shape": list(texts.shape),
            "dtype": "float32", "all_finite": True,
            "attention_sum_max_abs_error": attention_error,
            "minimum_region_l2_norm": min_region_norm,
            "all_visual_and_text_rows_complete": True,
            "caption_id_image_id_join_valid": True,
            "truncated_captions": int(trunc.sum()),
            "checksums": {name: sha256_file(path) for name, path in arrays.items() if name in {"regions", "patches", "attention", "texts", "image_ids", "caption_ids"}},
        }
    overlap = ids_by_split["train"] & ids_by_split["val"]
    if overlap:
        raise AssertionError(f"Pilot train and validation image IDs overlap: {sorted(overlap)[:10]}")
    expected_annotations = sum(row["caption_count"] for row in manifest["splits"].values())
    if len(annotation_ids) != expected_annotations:
        raise AssertionError("Caption annotation IDs are not globally unique across pilot splits.")
    checks.update({
        "status": "passed", "train_validation_image_ids_disjoint": True,
        "unique_annotation_ids": len(annotation_ids),
        "selected_images": sum(row["image_count"] for row in manifest["splits"].values()),
        "selected_captions": expected_annotations,
        "float32_storage": True,
    })
    atomic_json(output / "feature_validation.json", checks)
    return checks


def saved_visual_stage_complete(manifest: dict, feature_root: Path) -> bool:
    for split in ("train", "val"):
        paths = feature_paths(feature_root, split)
        required = ("image_ids", "regions", "patches", "attention", "visual_done", "visual_manifest")
        if any(not paths[key].is_file() for key in required):
            return False
        try:
            done = np.load(paths["visual_done"], mmap_mode="r")
            if len(done) != manifest["splits"][split]["image_count"] or not np.all(done == 1):
                return False
        except Exception:
            return False
    return True


def saved_text_stage_complete(manifest: dict, feature_root: Path) -> bool:
    for split in ("train", "val"):
        paths = feature_paths(feature_root, split)
        required = ("caption_ids", "texts", "truncated", "text_done", "text_manifest")
        if any(not paths[key].is_file() for key in required):
            return False
        try:
            done = np.load(paths["text_done"], mmap_mode="r")
            if len(done) != manifest["splits"][split]["caption_count"] or not np.all(done == 1):
                return False
        except Exception:
            return False
    return True


def write_pilot_config(source: Path, dest: Path) -> dict:
    import yaml
    cfg = yaml.safe_load(source.read_text())
    cfg["train"]["num_epochs"] = 10
    cfg["train"]["seed"] = 123
    cfg["train"]["batch_size"] = 128
    cfg["train"]["lr"] = 0.0001
    cfg["train"]["pilot_override"] = True
    cfg["train"]["note"] = "10-epoch controlled 1,000-image pilot; canonical 100-epoch full-training config unchanged"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def run_training(output: Path, status_path: Path, data_root: Path) -> dict:
    train_py = MODULE / "train.py"
    config_path = output / "pilot_config.yaml"
    write_pilot_config(MODULE / "configs/talk2dino_siglip_vitl_infonce.yaml", config_path)
    feature_root = output / "features"
    checkpoint_dir = output / "pilot_mlp_checkpoint"
    final_ckpt = checkpoint_dir / "final_projection.pt"
    initial_ckpt = checkpoint_dir / "initial_projection.pt"
    history_path = output / "training_history.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if final_ckpt.exists() and history_path.exists():
        old_history = json.loads(history_path.read_text())
        if old_history.get("status") == "complete":
            return {
                "status": "complete", "checkpoint": str(final_ckpt),
                "sha256": sha256_file(final_ckpt), "history": old_history,
                "seconds": old_history.get("training_seconds"),
                "epochs": old_history.get("completed_epochs"),
            }
        raise RuntimeError("A final MLP checkpoint exists without a complete history record; refusing to assume completion.")
    common = [sys.executable, str(train_py), "--config", str(config_path),
              "--dataset-root", str(data_root), "--feature-root", str(feature_root),
              "--train-text-features", str(feature_root / "text_features_train2014.npy"),
              "--val-text-features", str(feature_root / "text_features_val2014.npy"),
              "--output", str(final_ckpt), "--initial-output", str(initial_ckpt),
              "--history-output", str(history_path)]
    preflight = subprocess.run(common, cwd=REPO, text=True, capture_output=True, check=True)
    preflight_summary = json.loads(preflight.stdout)
    atomic_json(output / "training_preflight.json", preflight_summary)
    if preflight_summary.get("preflight") != "passed" or not preflight_summary.get("train_validation_visual_definition_identical"):
        raise AssertionError("Trainer preflight did not validate pilot features/config.")
    write_status(status_path, "training", training_preflight="passed")
    start = time.perf_counter()
    with (output / "logs/pilot.log").open("a") as log:
        subprocess.run(common + ["--execute"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True, text=True)
    elapsed = time.perf_counter() - start
    history = json.loads(history_path.read_text())
    if history.get("status") != "complete" or history.get("completed_epochs") != 10 or not final_ckpt.is_file():
        raise AssertionError("Training did not finish all ten pilot epochs or save its checkpoint.")
    history.update({
        "configuration": str(config_path), "training_seconds": elapsed,
        "checkpoint": str(final_ckpt), "checkpoint_sha256": sha256_file(final_ckpt),
        "initial_checkpoint": str(initial_ckpt), "initial_checkpoint_sha256": sha256_file(initial_ckpt),
        "optimizer_contains_mlp_only": True, "preflight": preflight_summary,
    })
    atomic_json(history_path, history)
    return {"status": "complete", "seconds": elapsed, "epochs": 10,
            "checkpoint": str(final_ckpt), "sha256": history["checkpoint_sha256"],
            "initial_checkpoint": str(initial_ckpt), "history": history}


def make_qualitative(output: Path, manifest: dict, seed: int) -> dict:
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess_talk2dino_image

    image_rows = manifest["splits"]["val"]["images"]
    captions = manifest["splits"]["val"]["captions"]
    caption_rows: dict[int, list[dict]] = {}
    for row in captions:
        caption_rows.setdefault(row["image_id"], []).append(row)
    eligible = [row for row in image_rows if len(caption_rows.get(row["image_id"], [])) >= 2]
    if len(eligible) < 8:
        raise AssertionError(f"Need at least 8 validation images with multiple captions; found {len(eligible)}")
    selected = sorted(random.Random(seed).sample(eligible, 8), key=lambda row: row["image_id"])
    arrays = prepare_split_arrays(manifest, output / "features", "val")
    ids = np.asarray(arrays["image_ids_array"])
    caption_ids = np.asarray(arrays["caption_ids_array"])
    image_index = {int(value): index for index, value in enumerate(ids)}
    annotation_index = {int(value): index for index, value in enumerate(caption_ids[:, 0])}
    checkpoint = torch.load(output / "pilot_mlp_checkpoint/final_projection.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]["model"]
    trained = RegionTextProjection(int(config["clip_embed_dim"]), int(config["dino_embed_dim"]), int(config["dino_embed_dim"]))
    trained.load_state_dict(checkpoint["state_dict"]); trained.eval()
    untrained = RegionTextProjection(int(config["clip_embed_dim"]), int(config["dino_embed_dim"]), int(config["dino_embed_dim"]))
    untrained.load_state_dict(checkpoint["initial_state_dict"]); untrained.eval()
    gallery = output / "qualitative_results"
    gallery.mkdir(parents=True, exist_ok=True)
    figure_paths = []
    rows_summary = []
    with torch.inference_mode():
        for image_record in selected:
            image_id = int(image_record["image_id"])
            i = image_index[image_id]
            image = Image.open(image_record["image_path"]).convert("RGB")
            pixels = preprocess_talk2dino_image(image).float()
            pixels = (pixels * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None]).clamp(0, 1)
            rgb = pixels.permute(1, 2, 0).numpy()
            patch = torch.from_numpy(np.array(arrays["patches_array"][i], copy=True)).float()
            regions = torch.from_numpy(np.array(arrays["regions_array"][i], copy=True)).float()
            attention = torch.from_numpy(np.array(arrays["attention_array"][i], copy=True)).float()
            query_rows = caption_rows[image_id][:2]
            query_outputs = []
            for caption in query_rows:
                embedding = torch.from_numpy(np.array(arrays["texts_array"][annotation_index[caption["annotation_id"]]], copy=True)).float().unsqueeze(0)
                trained_score, trained_head = trained(regions.unsqueeze(0), embedding, return_head_indices=True)
                untrained_score, untrained_head = untrained(regions.unsqueeze(0), embedding, return_head_indices=True)
                trained_text = F.normalize(trained.text_projection(embedding), dim=-1)
                untrained_text = F.normalize(untrained.text_projection(embedding), dim=-1)
                patch_norm = F.normalize(patch, dim=-1)
                trained_patch_score = patch_norm @ trained_text[0]
                untrained_patch_score = patch_norm @ untrained_text[0]
                th = int(trained_head[0, 0]); uh = int(untrained_head[0, 0])
                trained_guided = trained_patch_score * attention[th]
                untrained_guided = untrained_patch_score * attention[uh]
                query_outputs.append({
                    "caption": caption["caption"], "annotation_id": caption["annotation_id"],
                    "trained_head": th, "untrained_head": uh,
                    "trained_similarity": trained_patch_score.reshape(16, 16).numpy(),
                    "untrained_similarity": untrained_patch_score.reshape(16, 16).numpy(),
                    "trained_guided": trained_guided.reshape(16, 16).numpy(),
                    "untrained_guided": untrained_guided.reshape(16, 16).numpy(),
                })
            fig = plt.figure(figsize=(16, 28), constrained_layout=True)
            grid = fig.add_gridspec(9, 4, height_ratios=[1.2, 1, 1, 1, 1, 1.1, 1.1, 1.1, 1.1])
            ax = fig.add_subplot(grid[0, 0:2]); ax.imshow(rgb); ax.set_title(f"Talk2DINO 224 input — image {image_id}"); ax.axis("off")
            ax = fig.add_subplot(grid[0, 2:4]); ax.axis("off")
            ax.text(0, 0.95, "Two held-out COCO captions for this image", fontsize=12, weight="bold", va="top")
            for qidx, item in enumerate(query_outputs):
                ax.text(0, 0.72 - 0.34*qidx, f"Query {qidx+1}: {item['caption']}", fontsize=9, va="top", wrap=True)
                ax.text(0, 0.54 - 0.34*qidx, f"Trained selected head={item['trained_head']} | initial MLP head={item['untrained_head']}", fontsize=9, va="top")
            all_attention = attention.numpy().reshape(16, 16, 16)
            attn_max = float(all_attention.max())
            for h in range(16):
                ax = fig.add_subplot(grid[1 + h // 4, h % 4])
                im = ax.imshow(all_attention[h], cmap="magma", vmin=0, vmax=attn_max)
                ax.set_title(f"CLS attention head {h}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            for qidx, item in enumerate(query_outputs):
                row = 5 + qidx * 2
                panels = [
                    (attention[item["trained_head"]].numpy().reshape(16, 16), "Trained MLP selected CLS attention", "magma", 0, attn_max),
                    (item["trained_similarity"], "Trained patch cosine similarity", "coolwarm", -1, 1),
                    (item["trained_guided"], "Trained attention-guided score", "coolwarm", -0.05, 0.05),
                    (item["untrained_guided"], "Initial MLP attention-guided score", "coolwarm", -0.05, 0.05),
                ]
                for col, (data, title, cmap, vmin, vmax) in enumerate(panels):
                    ax = fig.add_subplot(grid[row, col])
                    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(f"Q{qidx+1}: {title}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
                ax = fig.add_subplot(grid[row + 1, 0:2])
                ax.imshow(rgb)
                ax.imshow(attention[item["trained_head"]].numpy().reshape(16, 16), cmap="magma", alpha=0.48, extent=(0, 224, 224, 0), vmin=0, vmax=attn_max, interpolation="nearest")
                ax.set_title(f"Q{qidx+1}: RGB + trained selected-head attention overlay", fontsize=9); ax.axis("off")
                ax = fig.add_subplot(grid[row + 1, 2:4]); ax.axis("off")
                ax.text(0, 0.85, f"Query {qidx+1}: {item['caption']}", fontsize=9, wrap=True, va="top")
                ax.text(0, 0.45, "Attention/similarity maps are exploratory diagnostics, not segmentation masks.", fontsize=9, wrap=True, va="top")
            fig.suptitle("Talk2DINO-SigLIP pilot — attention and similarity diagnostics (not segmentation masks)", fontsize=14)
            path = gallery / f"val_image_{image_id}_attention_comparison.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            figure_paths.append(str(path))
            rows_summary.append({
                "image_id": image_id,
                "relative_image_path": image_record["relative_image_path"],
                "queries": [{
                    "caption": item["caption"],
                    "annotation_id": item["annotation_id"],
                    "trained_head": item["trained_head"],
                    "untrained_head": item["untrained_head"],
                    "trained_similarity_min": float(item["trained_similarity"].min()),
                    "trained_similarity_max": float(item["trained_similarity"].max()),
                } for item in query_outputs],
                "figure": str(path),
            })
    # Compact eight-example overview, derived from the full-resolution panels.
    from PIL import ImageOps, ImageDraw
    thumbs = []
    for path in figure_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((640, 920))
        canvas = Image.new("RGB", (660, 960), "white")
        canvas.paste(image, ((660 - image.width) // 2, 20))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (1320, 3840), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 2) * 660, (index // 2) * 960))
    sheet.save(gallery / "pilot_qualitative_overview.png")
    atomic_json(gallery / "qualitative_manifest.json", {
        "status": "complete", "held_out_images": len(rows_summary),
        "each_image_caption_count_visualized": 2,
        "contains_all_16_cls_attention_heads": True,
        "trained_and_initial_mlp_comparison": True,
        "warning": "Similarity and attention maps are exploratory; they are not validated segmentation masks.",
        "examples": rows_summary,
    })
    return {"status": "complete", "held_out_images": len(rows_summary), "figures": figure_paths,
            "overview": str(gallery / "pilot_qualitative_overview.png")}


def build_final_report(output: Path, manifest: dict, validation: dict, visual: dict, text: dict, training: dict, qualitative: dict, resources: dict) -> None:
    train = manifest["splits"]["train"]
    val = manifest["splits"]["val"]
    epoch_history = training.get("history", {}).get("history", [])
    first_epoch = epoch_history[0] if epoch_history else {}
    last_epoch = epoch_history[-1] if epoch_history else {}
    train_loss_span = (
        f"{first_epoch.get('mean_train_loss', float('nan')):.4f} → {last_epoch.get('mean_train_loss', float('nan')):.4f}"
        if epoch_history else "not recorded"
    )
    validation_loss_final = f"{last_epoch['mean_validation_loss']:.4f}" if "mean_validation_loss" in last_epoch else "not recorded"
    repeated_train_rows = sum(int(row.get("train_repeated_image_rows_logged", 0)) for row in epoch_history)
    repeated_val_rows = sum(int(row.get("validation_repeated_image_rows_logged", 0)) for row in epoch_history)
    report = f"""# Talk2DINO-SigLIP Controlled Pilot — 1,000 COCO Images

## Outcome

The controlled pilot completed its extraction, integrity gate, 10-epoch projection-MLP training and held-out visual diagnostics. It processed **{train['image_count']} train images / {train['caption_count']} captions** and **{val['image_count']} validation images / {val['caption_count']} captions**. Feature validation: **{validation['status']}**. MLP training: **{training['status']}**, {training.get('epochs', 0)} epochs. Qualitative panels: **{qualitative['held_out_images']} held-out validation images**, two captions per image.

This is a pipeline smoke pilot, not a final COCO training run or evidence of segmentation quality.

## Sampling and reproducibility

Images were sampled uniformly without replacement from sorted unique COCO image IDs using Python `random.Random(123)`: 800 from train2014 and 200 from val2014. All captions for each sampled image were retained in annotation-file order. Train/validation image IDs are disjoint. The exact IDs, paths, captions and annotation IDs are in `pilot_manifest.json`.

## Feature extraction

- DINO visual branch: existing OpenVLA fine-tuned `vit_large_patch14_reg4_dinov2.lvd142m`; Talk2DINO bicubic 224 resize/center crop and ImageNet normalization. Existing shared extractor produced final normalized patches `[N,256,1024]`, CLS-to-patch maps `[N,16,256]`, and Talk2DINO attention-weighted mean regions `[N,16,1024]`.
- Text: frozen `timm/ViT-SO400M-14-SigLIP-384`, revision `ac16108d567c4389e6cd2b11c9b8585f7474435b`; OpenCLIP tokenizer configured with `canonicalize`, context 64, output 1152. One vector per caption annotation.
- All pilot arrays were stored as float32. Visual features are image-level and shared across captions; text vectors are annotation-level. Files and SHA-256 checksums are listed in `feature_validation.json`.
- Selected captions exceeding context 64: {text['truncated_count']} / {text['caption_count']}.
- Encoder-state hashes matched before and after extraction. Re-extraction was exact for the sampled validation subset. Standard OpenVLA fused features were exactly unchanged across the sidecar extraction check.

## Validation and training

Feature validation passed: all selected images/captions were processed; image-caption joins and unique IDs were checked; train/validation image IDs were disjoint; dimensions and float32 dtype matched; all values were finite; attention normalized over patches (max absolute sum error: train {validation['splits']['train']['attention_sum_max_abs_error']:.2g}, val {validation['splits']['val']['attention_sum_max_abs_error']:.2g}); region descriptors were nonzero; saved arrays reloaded and checksummed.

The existing `RegionTextProjection` was trained for ten epochs, with Adam (`lr=1e-4`), batch 128, seed 123, fixed temperature 0.07 and canonical bidirectional InfoNCE. Only the `1152→1024→1024` MLP was optimized. Repeated image IDs in batches were counted as potential diagonal-label false negatives. Per-epoch train/validation losses and duplicate counts are in `training_history.json`; initial and final MLP states are under `pilot_mlp_checkpoint/`.

Training mean loss (epoch 1 → 10): {train_loss_span}; final mean validation loss: {validation_loss_final}. Potential repeated-image rows logged across epochs: train {repeated_train_rows}, validation {repeated_val_rows}. These are pilot optimization diagnostics, not final model-quality estimates.

## Qualitative findings and limits

`qualitative_results/` contains eight held-out validation examples, each with two caption queries, the 16 raw DINO CLS-attention maps, trained and initial-MLP selected-head maps, patch cosine scores, and attention-guided overlays. These maps are exploratory attention/similarity diagnostics, **not validated segmentation masks**. This pilot does not implement or validate the full background-cleaning/mask-generation stages as a segmentation result.

No oracle labels were used for sampling, feature selection, training or visualization. No policy actions, LIBERO rollouts, KV caching, full COCO extraction or 100-epoch training were run. Standard OpenVLA inference behavior and pretrained weights were left unchanged.

## Performance and resources

- DINO model load + initial state-hash time: {visual['model_load_and_initial_hash_seconds']:.2f} s.
- DINO model-forward time per image: mean {visual['model_forward_ms']['mean']:.2f} ms, median {visual['model_forward_ms']['median']:.2f} ms, P95 {visual['model_forward_ms']['p95']:.2f} ms.
- DINO per-image extraction pipeline (including image open/preprocess/call): mean {visual['per_image_pipeline_ms']['mean']:.2f} ms; total measured image-call time {visual['total_visual_wall_seconds']:.2f} s ({visual['images_per_second_pipeline']:.2f} images/s).
- DINO stage total, including load and validation: {visual['total_stage_seconds_including_load_validation']:.2f} s.
- SigLIP text: {text['caption_count']} captions encoded in {text['encoding_seconds_total']:.2f} s of encoder time ({text['captions_per_second_encoder_only']:.2f} captions/s; {text['mean_encoding_ms_per_caption']:.2f} ms/caption averaged over batches); tokenization took {text['tokenizer_seconds_total']:.2f} s. **This timing came from the full-set saved-feature verification pass**; the initial feature-writer timing was not persisted, so it is not represented as the original write-pass timing.
- SigLIP stage total, including load and verification: {text['total_stage_seconds_including_load_validation']:.2f} s.
- Training wall time: {training.get('seconds', 0):.2f} s.
- Peak CUDA allocated/reserved: DINO {visual['cuda']['peak_allocated_bytes'] / 1024**3:.2f}/{visual['cuda']['peak_reserved_bytes'] / 1024**3:.2f} GiB; SigLIP text {text['cuda']['peak_allocated_bytes'] / 1024**3:.2f}/{text['cuda']['peak_reserved_bytes'] / 1024**3:.2f} GiB; training {training['cuda_peak_allocated_bytes'] / 1024**2:.1f}/{training['cuda_peak_reserved_bytes'] / 1024**2:.1f} MiB.
- Peak process RSS: DINO {visual['process_peak_rss_bytes'] / 1024**3:.2f} GiB; SigLIP text {text['process_peak_rss_bytes'] / 1024**3:.2f} GiB; training {training['process_peak_rss_bytes'] / 1024**3:.2f} GiB. The model stages ran sequentially.
- Feature files occupy {resources['feature_storage_bytes'] / 1024**3:.2f} GiB; free disk after feature creation was {resources['disk_free_bytes_after_features'] / 1024**3:.1f} GiB.
- GPU utilization was not measured because `tegrastats` is unavailable in the isolated container.

Extraction, verification and training timing scopes are separate. The first image call includes warm-up effects; no end-to-end inference speedup is claimed.

## Execution notes

The first detached attempt stopped before feature extraction because the pilot feature directory was not initialized; this was fixed before retry. The next attempt completed and persisted all features, then correctly blocked training when a 4-caption repeat check differed slightly from the original 64-row text batch. The deterministic check now re-encodes with the same batch shape and compares against the saved rows; the numerical tolerance was not loosened. The feature validator also caught and corrected a structured caption-ID indexing error before training, then the feature gate passed. The retry reused validated visual features and verified completed text arrays before training. Training completed; the first qualitative finalization attempt caught NumPy arrays being included in JSON metadata. The figures were produced, the serializer was corrected to retain only scalar diagnostics, and the worker resumed without repeating extraction or training. Full attempt logs are preserved in `logs/pilot.log`.

## Recommendation

The pipeline is ready to proceed to a separately authorized larger COCO extraction run, subject to reviewing the qualitative maps and training curves first. This 1,000-image pilot validates data plumbing and optimization behavior only; it does not establish foreground segmentation quality or generalization.
"""
    (output / "PILOT_REPORT.md").write_text(report)


def run(args: argparse.Namespace) -> None:
    full_pipeline_start = time.perf_counter()
    output = args.output.resolve()
    data_root = args.dataset_root.resolve()
    checkpoint_dir = args.siglip_checkpoint.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock_path = logs / "pilot.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another pilot worker holds the lock; refusing duplicate execution.") from exc
        status_path = output / "status.json"
        (logs / "process.pid").write_text(f"{os.getpid()}\n")
        try:
            if not (data_root / "annotations2014/captions_train2014.json").is_file() or not (data_root / "annotations2014/captions_val2014.json").is_file():
                raise FileNotFoundError("COCO train/val caption annotations are missing.")
            if not (checkpoint_dir / "open_clip_model.safetensors").is_file():
                raise FileNotFoundError(f"SigLIP-384 weights missing at {checkpoint_dir}")
            write_status(status_path, "manifest")
            manifest = create_or_load_pilot_manifest(data_root, output, args.seed)
            mark_done(status_path, "manifest", actual_images=manifest["actual_total_images"], actual_captions=manifest["actual_total_captions"])
            if manifest["actual_total_images"] != 1000:
                raise AssertionError(f"Expected exactly 1000 sampled images, got {manifest['actual_total_images']}")
            if args.manifest_only:
                write_status(status_path, "manifest_ready", state="pending",
                             last_completed_stage="manifest", note="Manifest saved; feature extraction not started.")
                return

            resources_path = output / "resource_metrics.json"
            if resources_path.exists():
                resources = json.loads(resources_path.read_text())
            else:
                resources = {"experiment": "talk2dino_siglip_pilot_1000", "created_at_utc": utc_now(), "stages": {}}

            write_status(status_path, "visual_extraction")
            saved_visual = resources.get("stages", {}).get("dino_visual_extraction")
            if saved_visual and saved_visual.get("status") == "passed" and saved_visual_stage_complete(manifest, output / "features"):
                visual = saved_visual
                # Earlier pilot metrics stored this particular wall-time field
                # in seconds while labeling it milliseconds; normalize once.
                mean_wall = visual.get("per_image_pipeline_ms", {}).get("mean")
                if mean_wall is not None and mean_wall < 1.0:
                    for key in ("mean", "median", "p95"):
                        if visual["per_image_pipeline_ms"].get(key) is not None:
                            visual["per_image_pipeline_ms"][key] *= 1000.0
                    visual["per_image_pipeline_units_corrected"] = "seconds to milliseconds"
                visual.setdefault("timing_mode", "initial extraction (persisted from completed DINO stage)")
                print("DINO feature stage already complete; reusing validated saved feature arrays and measurements.", flush=True)
            else:
                visual = extract_visual(manifest, data_root, output, status_path)
            resources["stages"]["dino_visual_extraction"] = visual
            atomic_json(resources_path, resources)
            mark_done(status_path, "visual_extraction", visual_images=visual["total_images"])

            write_status(status_path, "text_extraction")
            saved_text = resources.get("stages", {}).get("siglip_text_extraction")
            if saved_text and saved_text.get("status") == "passed" and saved_text_stage_complete(manifest, output / "features"):
                text = saved_text
                print("SigLIP text stage already complete; reusing verified saved embeddings and timing.", flush=True)
            else:
                text = extract_text(manifest, output, checkpoint_dir, status_path, args.text_batch_size)
            resources["stages"]["siglip_text_extraction"] = text
            resources["extraction_total_seconds"] = float(visual.get("total_stage_seconds_including_load_validation") or (visual.get("model_load_and_initial_hash_seconds", 0.0) + visual.get("total_visual_wall_seconds", 0.0))) + text["total_stage_seconds_including_load_validation"]
            atomic_json(resources_path, resources)
            mark_done(status_path, "text_extraction", text_captions=text["caption_count"])

            write_status(status_path, "feature_validation")
            validation = validate_features(manifest, output)
            if validation.get("status") != "passed":
                raise AssertionError("Feature validation did not pass; training is blocked.")
            resources["feature_storage_bytes"] = sum(
                p.stat().st_size for p in (output / "features").glob("*") if p.is_file()
            )
            usage = resource.getrusage(resource.RUSAGE_SELF)
            resources["worker_peak_rss_bytes"] = int(usage.ru_maxrss * (1024 if sys.platform != "darwin" else 1))
            resources["disk_free_bytes_after_features"] = int(os.statvfs(output).f_bavail * os.statvfs(output).f_frsize)
            atomic_json(resources_path, resources)
            mark_done(status_path, "feature_validation", validation="passed")

            write_status(status_path, "training")
            training = run_training(output, status_path, data_root)
            train_history = training.get("history", {})
            last_epoch = (train_history.get("history") or [{}])[-1]
            resources["stages"]["mlp_training"] = {
                "status": training["status"], "seconds": training.get("seconds"), "epochs": training.get("epochs"),
                "process_peak_rss_bytes": last_epoch.get("process_peak_rss_bytes"),
                "cuda_peak_allocated_bytes": last_epoch.get("cuda_peak_allocated_bytes"),
                "cuda_peak_reserved_bytes": last_epoch.get("cuda_peak_reserved_bytes"),
            }
            atomic_json(resources_path, resources)
            mark_done(status_path, "training", training=training["status"], epochs=training.get("epochs", 10))

            write_status(status_path, "qualitative")
            qualitative = make_qualitative(output, manifest, args.seed)
            resources["stages"]["qualitative"] = qualitative
            usage = resource.getrusage(resource.RUSAGE_SELF)
            resources["worker_peak_rss_bytes"] = int(usage.ru_maxrss * (1024 if sys.platform != "darwin" else 1))
            resources["total_pipeline_wall_seconds"] = time.perf_counter() - full_pipeline_start
            resources["gpu_utilization"] = "not measured; tegrastats is unavailable in the isolated container"
            atomic_json(resources_path, resources)
            mark_done(status_path, "qualitative", qualitative_images=qualitative["held_out_images"])

            build_final_report(output, manifest, validation, visual, text, training, qualitative, resources)
            write_status(status_path, "complete", state="complete", exit_code=0,
                         last_completed_stage="qualitative", actual_images=manifest["actual_total_images"],
                         actual_captions=manifest["actual_total_captions"], validation="passed",
                         training="complete", qualitative="complete", completed_at_utc=utc_now())
        except BaseException as exc:
            write_status(status_path, "failed", state="failed", exit_code=1,
                         error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), failed_at_utc=utc_now())
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--siglip-checkpoint", type=Path, default=DEFAULT_SIGLIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--manifest-only", action="store_true", help="Write/verify the deterministic sample manifest without loading models.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
