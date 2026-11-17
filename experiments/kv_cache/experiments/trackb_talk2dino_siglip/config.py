"""Shared dimensions and compatibility checks for Talk2DINO feature caches."""
from __future__ import annotations

from typing import Any


EXPECTED_DINO_FEATURE_DIM = 1024
EXPECTED_SIGLIP_TEXT_DIM = 1152
EXPECTED_SIGLIP_CONTEXT_LENGTH = 64
SIGLIP_TEXT_MODEL_REVISION = "ac16108d567c4389e6cd2b11c9b8585f7474435b"
EXPECTED_REGION_HEADS = 16
EXPECTED_PATCH_GRID = (16, 16)


def validate_feature_manifests(train: dict[str, Any], validation: dict[str, Any]) -> None:
    """Require train and validation to use the same visual feature definition."""
    keys = ("feature_definition", "preprocessing", "dino_model", "dino_revision")
    mismatches = {
        key: (train.get(key), validation.get(key))
        for key in keys
        if train.get(key) != validation.get(key)
    }
    if mismatches:
        raise ValueError(f"Train/validation visual feature manifests differ: {mismatches}")
    train_shape, val_shape = train.get("region_shape"), validation.get("region_shape")
    if not train_shape or not val_shape or train_shape[1:] != val_shape[1:]:
        raise ValueError(f"Train/validation region feature dimensions differ: {train_shape}, {val_shape}")
    expected = train.get("feature_definition")
    if expected != "talk2dino_final_layer_cls_attention_weighted_patch_mean_v1":
        raise ValueError(f"Unexpected visual feature definition: {expected!r}")
