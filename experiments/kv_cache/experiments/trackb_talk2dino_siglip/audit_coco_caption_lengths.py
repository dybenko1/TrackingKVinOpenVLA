#!/usr/bin/env python3
"""Audit COCO caption lengths using the shared checkpoint-configured tokenizer."""
from __future__ import annotations

import argparse
import heapq
import json
import math
import statistics
from pathlib import Path
from typing import Any

import sys
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.tokenizer import load_siglip_tokenizer


def percentile_nearest_rank(sorted_values: list[int], q: float) -> int:
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def summarize_split(path: Path, tokenizer: Any, batch_size: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload["annotations"]
    lengths: list[int] = []
    examples: list[tuple[int, int, str, list[int]]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [str(row["caption"]) for row in batch]
        tokenized = tokenizer.tokenize_untruncated(texts)
        for offset, (row, ids) in enumerate(zip(batch, tokenized)):
            length = len(ids)
            lengths.append(length)
            if length > tokenizer.context_length:
                item = (length, start + offset, str(row["caption"]), ids)
                if len(examples) < 8:
                    heapq.heappush(examples, item)
                elif item[0] > examples[0][0]:
                    heapq.heapreplace(examples, item)
    ordered = sorted(lengths)
    truncated = sum(length > tokenizer.context_length for length in lengths)
    inner = tokenizer.wrapper.tokenizer
    return {
        "annotation_file": str(path),
        "caption_count": len(lengths),
        "within_context_count": len(lengths) - truncated,
        "over_context_count": truncated,
        "over_context_percent": 100.0 * truncated / len(lengths) if lengths else 0.0,
        "context_length": tokenizer.context_length,
        "token_length_including_special_tokens": {
            "mean": statistics.fmean(lengths) if lengths else None,
            "median": statistics.median(ordered) if ordered else None,
            "p90_nearest_rank": percentile_nearest_rank(ordered, .90) if ordered else None,
            "p99_nearest_rank": percentile_nearest_rank(ordered, .99) if ordered else None,
            "maximum": ordered[-1] if ordered else None,
        },
        "longest_over_context_examples": [
            {"annotation_row": row, "token_count": length, "caption": caption,
             "tokens": inner.convert_ids_to_tokens(ids)}
            for length, row, caption, ids in sorted(examples, reverse=True)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/opt/openvla/cache/datasets/talk2dino/coco2014"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384"))
    parser.add_argument("--output", type=Path, default=Path("/opt/openvla/experiments/kv_cache/outputs/trackb_talk2dino_siglip/caption_audit_siglip384.json"))
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.checkpoint_dir / "open_clip_config.json"
    if not config_path.is_file():
        result = {
            "status": "blocked_missing_required_checkpoint",
            "required_checkpoint": "timm/ViT-SO400M-14-SigLIP-384",
            "checkpoint_dir": str(args.checkpoint_dir),
            "reason": f"Missing local checkpoint config: {config_path}",
            "note": "No 224/context-16 substitute was used; caption counts for context 64 remain pending.",
        }
    else:
        tokenizer = load_siglip_tokenizer(args.checkpoint_dir)
        result = {
            "status": "complete",
            "tokenizer": tokenizer.metadata(),
            "special_tokens_included": True,
            "truncated_rule": "raw tokenized length including special tokens > checkpoint context length",
            "percentile_method": "nearest rank: sorted[ceil(q*n)-1]",
            "splits": {
                split: summarize_split(args.dataset_root / "annotations2014" / f"captions_{split}2014.json", tokenizer, args.batch_size)
                for split in ("train", "val")
            },
        }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
