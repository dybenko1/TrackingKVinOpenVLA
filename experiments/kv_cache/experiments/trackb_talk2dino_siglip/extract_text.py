#!/usr/bin/env python3
"""Cache frozen paired-SigLIP text features for every COCO caption."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import sys
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import encode_text_embeddings, load_siglip_text_tower
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.config import SIGLIP_TEXT_MODEL_REVISION

DEFAULT_CHECKPOINT = Path("/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run(args: argparse.Namespace) -> None:
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = args.dataset_root
    feature_root = root / "features"
    if not (args.checkpoint_dir / "open_clip_config.json").is_file() or not (args.checkpoint_dir / "open_clip_model.safetensors").is_file():
        raise FileNotFoundError(
            f"Required local SigLIP-384 checkpoint is incomplete at {args.checkpoint_dir}; "
            "no 224/context-16 fallback is attempted."
        )
    model, tokenizer, model_info = load_siglip_text_tower(args.checkpoint_dir, str(device))
    output_dim = int(model_info["text_output_dim"])
    context_length = int(model_info["text_context_length"])
    model.to(device).eval()
    feature_root.mkdir(parents=True, exist_ok=True)
    for split in (("train", "val") if args.split == "all" else (args.split,)):
        path = root / "annotations2014" / f"captions_{split}2014.json"
        payload = json.loads(path.read_text())
        records = payload["annotations"][: args.limit or None]
        ids = np.asarray([[int(r["id"]), int(r["image_id"])] for r in records], dtype=np.int64)
        texts_path = feature_root / f"caption_ids_{split}2014.npy"
        if texts_path.exists():
            if not np.array_equal(np.load(texts_path, mmap_mode="r"), ids):
                raise RuntimeError(f"Existing {split} caption index differs from annotations.")
        else:
            np.save(texts_path, ids, allow_pickle=False)
        values_path = feature_root / f"text_features_{split}2014_fp16.npy"
        trunc_path = feature_root / f"caption_truncated_{split}2014.npy"
        done_path = feature_root / f"text_done_{split}2014.npy"
        values_new, trunc_new, done_new = not values_path.exists(), not trunc_path.exists(), not done_path.exists()
        values = np.lib.format.open_memmap(values_path, mode="r+" if not values_new else "w+", dtype=np.float16, shape=(len(ids), output_dim))
        truncated = np.lib.format.open_memmap(trunc_path, mode="r+" if not trunc_new else "w+", dtype=np.uint8, shape=(len(ids),))
        done = np.lib.format.open_memmap(done_path, mode="r+" if not done_new else "w+", dtype=np.uint8, shape=(len(ids),))
        if done_new:
            done[:] = 0
        if trunc_new:
            truncated[:] = 0
        pending = np.flatnonzero(np.asarray(done) == 0).tolist()
        for offset in range(0, len(pending), args.batch_size):
            batch_rows = pending[offset : offset + args.batch_size]
            batch_text = [records[i]["caption"] for i in batch_rows]
            untruncated = tokenizer.tokenize_untruncated(batch_text)
            batch_truncated = np.asarray([len(ids_) > context_length for ids_ in untruncated], dtype=np.uint8)
            token_ids = tokenizer(batch_text).to(device)
            embeddings = encode_text_embeddings(model, token_ids, expected_dim=output_dim)
            if tuple(embeddings.shape) != (len(batch_rows), output_dim):
                raise AssertionError(f"Unexpected text feature tensor {tuple(embeddings.shape)}.")
            values[batch_rows] = embeddings.float().cpu().numpy().astype(np.float16)
            truncated[batch_rows] = batch_truncated
            done[batch_rows] = 1
            values.flush(); truncated.flush(); done.flush()
            complete = int(np.asarray(done).sum())
            completed_truncations = int(np.asarray(truncated)[np.asarray(done) == 1].sum())
            atomic_json(feature_root / f"text_{split}2014_manifest.json", {
                "split": split,
                "checkpoint_repository": "timm/ViT-SO400M-14-SigLIP-384",
                "checkpoint_revision": SIGLIP_TEXT_MODEL_REVISION,
                "checkpoint_dir": str(args.checkpoint_dir),
                "model_info": model_info,
                "caption_count": len(ids), "completed": complete,
                "feature_shape": [len(ids), output_dim], "dtype": "float16",
                "context_length": context_length,
                "tokenizer": tokenizer.metadata(),
                "truncated_count": completed_truncations,
                "truncated_fraction_completed": completed_truncations / complete if complete else 0.0,
                "caption_id_index_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                "updated_at_unix": time.time(),
            })
            print(f"SigLIP {split} captions: {complete}/{len(ids)}; truncated={int(np.asarray(truncated).sum())}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/opt/openvla/cache/datasets/talk2dino/coco2014"))
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="bounded smoke validation; 0 means all captions")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
