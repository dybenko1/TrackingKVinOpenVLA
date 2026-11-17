#!/usr/bin/env python3
"""Small post-hoc metrics for Talk2DINO patch masks and semantic label maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True, help="NPZ created by segment.py")
    parser.add_argument("--target-fg", type=Path, required=True, help="post-hoc binary reference mask, never used at inference")
    args = parser.parse_args()
    with np.load(args.prediction) as pack:
        pred = np.asarray(pack["union_mask"], dtype=bool)
    target = np.asarray(np.load(args.target_fg), dtype=bool)
    if pred.shape != (16, 16) or target.shape != (16, 16):
        raise ValueError(f"Expected aligned 16x16 masks, got {pred.shape} and {target.shape}.")
    tp = int((pred & target).sum()); fp = int((pred & ~target).sum()); fn = int((~pred & target).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); iou = tp / max(tp + fp + fn, 1)
    print(json.dumps({"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "iou": iou, "reference_role": "post-hoc diagnostic only"}, indent=2))


if __name__ == "__main__":
    main()
