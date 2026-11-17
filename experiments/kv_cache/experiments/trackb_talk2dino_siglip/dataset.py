"""Annotation-level COCO feature joins for the original Talk2DINO protocol."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def load_coco_records(data_root: Path, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported COCO split: {split}")
    path = Path(data_root) / "annotations2014" / f"captions_{split}2014.json"
    with path.open() as handle:
        payload = json.load(handle)
    return payload["images"], payload["annotations"]


class CocoAnnotationDataset(Dataset):
    """One training row per caption annotation, preserving duplicate image IDs.

    This matches Talk2DINO's annotation-level sampling convention. Multiple
    captions for one image may occur in one batch; the trainer reports these
    possible diagonal-label false negatives rather than silently changing the
    contrastive objective or deduplicating the rows.
    """

    def __init__(self, image_ids_path: Path, image_regions_path: Path,
                 caption_ids_path: Path, text_features_path: Path):
        self.image_ids = np.load(image_ids_path, mmap_mode="r")
        self.regions = np.load(image_regions_path, mmap_mode="r")
        self.caption_ids = np.load(caption_ids_path, mmap_mode="r")
        self.text_features = np.load(text_features_path, mmap_mode="r")
        if self.caption_ids.ndim != 2 or self.caption_ids.shape[1] != 2:
            raise ValueError("caption_ids must contain [annotation_id, image_id] rows")
        if len(self.caption_ids) != len(self.text_features):
            raise ValueError("Caption ID and text feature row counts differ")
        if len(self.image_ids) != len(self.regions):
            raise ValueError("Image ID and region feature row counts differ")
        self.image_row_by_id = {int(image_id): row for row, image_id in enumerate(self.image_ids.tolist())}
        self.caption_rows: list[tuple[int, int, int]] = []
        for row, (annotation_id, image_id) in enumerate(self.caption_ids.tolist()):
            image_row = self.image_row_by_id.get(int(image_id))
            if image_row is not None:
                self.caption_rows.append((int(annotation_id), int(image_id), image_row))
        self.dropped_missing_images = len(self.caption_ids) - len(self.caption_rows)
        if not self.caption_rows:
            raise ValueError("No caption annotations have matching image-level features")

    def __len__(self) -> int:
        return len(self.caption_rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        annotation_id, image_id, image_row = self.caption_rows[index]
        region = np.array(self.regions[image_row], copy=True)
        text = np.array(self.text_features[index], copy=True)
        return (torch.from_numpy(region).float(), torch.from_numpy(text).float(), image_id, annotation_id)


def repeated_image_id_count(image_ids: torch.Tensor | list[int]) -> int:
    values = image_ids.detach().cpu().tolist() if isinstance(image_ids, torch.Tensor) else list(image_ids)
    return sum(count - 1 for count in Counter(map(int, values)).values() if count > 1)


# Backward-compatible import name; semantics are deliberately annotation-level.
UniqueImageCaptionDataset = CocoAnnotationDataset
