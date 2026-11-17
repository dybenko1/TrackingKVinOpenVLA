#!/usr/bin/env python3
"""Talk2DINO projection training; optimization requires explicit --execute."""
from __future__ import annotations

import argparse
import json
import os
import resource
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.config import validate_feature_manifests
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.dataset import CocoAnnotationDataset, repeated_image_id_count
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.losses import symmetric_infonce
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the isolated training config") from exc
    return yaml.safe_load(path.read_text())


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(p) for group in optimizer.param_groups for p in group["params"]}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_validation(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                   temperature: float, legacy_scaling: bool) -> tuple[float, int]:
    model.eval()
    losses: list[float] = []
    duplicate_rows = 0
    with torch.inference_mode():
        for regions, texts, image_ids, _annotation_ids in loader:
            duplicate_rows += repeated_image_id_count(image_ids)
            scores = model(regions.to(device), texts.to(device))
            losses.append(float(symmetric_infonce(scores, temperature, legacy_scaling).cpu()))
    return float(np.mean(losses)), duplicate_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    default_config = Path(__file__).resolve().parent / "configs/talk2dino_siglip_vitl_infonce.yaml"
    parser.add_argument("--config", "--model-config", "--model_config", "--model", dest="config", type=Path, default=default_config)
    parser.add_argument("--dataset-root", type=Path, default=Path("/opt/openvla/cache/datasets/talk2dino/coco2014"))
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--train-text-features", type=Path,
                        help="Optional explicit training text-feature array path (pilot/cache override).")
    parser.add_argument("--val-text-features", type=Path,
                        help="Optional explicit validation text-feature array path (pilot/cache override).")
    parser.add_argument("--initial-output", type=Path,
                        help="Optional path to save/verify the deterministic initial MLP weights.")
    parser.add_argument("--history-output", type=Path,
                        help="Optional path for an atomically updated per-epoch history JSON.")
    parser.add_argument("--output", type=Path, default=Path("/opt/openvla/cache/checkpoints/talk2dino_siglip/projection.pt"))
    parser.add_argument("--execute", action="store_true", help="required to start training; otherwise only validate inputs/config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_cfg, model_cfg = cfg["train"], cfg["model"]
    seed = int(train_cfg["seed"])
    batch_size, epochs, lr = int(train_cfg["batch_size"]), int(train_cfg["num_epochs"]), float(train_cfg["lr"])
    temperature = float(model_cfg.get("temperature", 0.07))
    legacy_scaling = bool(model_cfg.get("legacy_extra_batch_square", False))
    feature_root = args.feature_root or args.dataset_root / "features"
    files = {
        "train_image_ids": feature_root / "image_ids_train2014.npy",
        "train_regions": feature_root / "region_features_train2014.npy",
        "train_caption_ids": feature_root / "caption_ids_train2014.npy",
        "train_text": args.train_text_features or feature_root / "text_features_train2014_fp16.npy",
        "val_image_ids": feature_root / "image_ids_val2014.npy",
        "val_regions": feature_root / "region_features_val2014.npy",
        "val_caption_ids": feature_root / "caption_ids_val2014.npy",
        "val_text": args.val_text_features or feature_root / "text_features_val2014_fp16.npy",
        "train_manifest": feature_root / "visual_train2014_manifest.json",
        "val_manifest": feature_root / "visual_val2014_manifest.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Feature extraction is not complete: " + ", ".join(missing))
    train_manifest = json.loads(files["train_manifest"].read_text())
    val_manifest = json.loads(files["val_manifest"].read_text())
    validate_feature_manifests(train_manifest, val_manifest)
    train_data = CocoAnnotationDataset(files["train_image_ids"], files["train_regions"], files["train_caption_ids"], files["train_text"])
    val_data = CocoAnnotationDataset(files["val_image_ids"], files["val_regions"], files["val_caption_ids"], files["val_text"])
    if min(len(train_data), len(val_data)) < batch_size:
        raise ValueError(f"Need at least {batch_size} caption annotations in each split; got {len(train_data)}, {len(val_data)}")
    summary = {
        "preflight": "passed", "train_annotations": len(train_data), "val_annotations": len(val_data),
        "train_missing_image_feature_rows": train_data.dropped_missing_images,
        "val_missing_image_feature_rows": val_data.dropped_missing_images,
        "batch_size": batch_size, "epochs": epochs, "optimizer": train_cfg["optimizer"],
        "lr": lr, "seed": seed, "feature_definition": train_manifest["feature_definition"],
        "train_validation_visual_definition_identical": True, "execute_required": True,
    }
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing projection checkpoint: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; no training was started.")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = RegionTextProjection(
        text_dim=int(model_cfg["clip_embed_dim"]),
        hidden_dim=int(model_cfg["dino_embed_dim"]),
        region_dim=int(model_cfg["dino_embed_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_parameter_ids(optimizer) != {id(p) for p in model.parameters()}:
        raise AssertionError("Optimizer must contain projection MLP parameters only")
    initial_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if args.initial_output is not None:
        args.initial_output.parent.mkdir(parents=True, exist_ok=True)
        if args.initial_output.exists():
            previous = torch.load(args.initial_output, map_location="cpu", weights_only=True)
            if set(previous) != set(initial_state_dict) or any(
                not torch.equal(previous[key], initial_state_dict[key]) for key in initial_state_dict
            ):
                raise RuntimeError("Existing pilot initial MLP state differs from the seed/configuration; refusing to overwrite.")
        else:
            temporary_initial = args.initial_output.with_suffix(args.initial_output.suffix + ".tmp")
            torch.save(initial_state_dict, temporary_initial)
            temporary_initial.replace(args.initial_output)
    history: list[dict] = []
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    if args.history_output is not None:
        atomic_json(args.history_output, {"status": "training", "completed_epochs": 0, "history": []})
    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        duplicate_rows = 0
        for regions, texts, image_ids, _annotation_ids in train_loader:
            duplicate_rows += repeated_image_id_count(image_ids)
            regions, texts = regions.to(device, non_blocking=True), texts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = symmetric_infonce(model(regions, texts), temperature=temperature,
                                     legacy_extra_batch_square=legacy_scaling)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss, val_duplicate_rows = run_validation(model, val_loader, device, temperature, legacy_scaling)
        record = {
            "epoch": epoch + 1,
            "mean_train_loss": float(np.mean(train_losses)),
            "mean_validation_loss": val_loss,
            "train_batches": len(train_losses),
            "validation_batches": len(val_loader),
            "train_repeated_image_rows_logged": duplicate_rows,
            "validation_repeated_image_rows_logged": val_duplicate_rows,
            "elapsed_seconds": time.perf_counter() - training_start,
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        }
        history.append(record)
        if args.history_output is not None:
            atomic_json(args.history_output, {"status": "training", "completed_epochs": len(history), "history": history})
        print(json.dumps(record), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save({
        "state_dict": model.cpu().state_dict(), "initial_state_dict": initial_state_dict,
        "config": cfg, "history": history,
        "seed": seed, "train_manifest": train_manifest, "validation_manifest": val_manifest,
        "trainable_parameter_count": model.trainable_parameter_count,
        "optimizer": "Adam", "learning_rate": lr,
        "training_seconds": time.perf_counter() - training_start,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
    }, temp)
    temp.replace(args.output)
    if args.history_output is not None:
        atomic_json(args.history_output, {"status": "complete", "completed_epochs": len(history), "history": history, "checkpoint": str(args.output)})
    print(f"Saved final projection checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()
