"""Load only the frozen SigLIP text tower from a local OpenCLIP safetensors file."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.tokenizer import (
    load_siglip_tokenizer,
    read_siglip_config,
    validate_text_config,
)
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.config import (
    EXPECTED_SIGLIP_CONTEXT_LENGTH,
    EXPECTED_SIGLIP_TEXT_DIM,
    SIGLIP_TEXT_MODEL_REVISION,
)


DEFAULT_SIGLIP_CHECKPOINT = Path(
    "/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384"
)
DEFAULT_EXPECTED_CONTEXT = EXPECTED_SIGLIP_CONTEXT_LENGTH
DEFAULT_EXPECTED_OUTPUT_DIM = EXPECTED_SIGLIP_TEXT_DIM


def load_siglip_text_tower(
    checkpoint_dir: Path = DEFAULT_SIGLIP_CHECKPOINT,
    device: str | None = None,
    *,
    expected_context_length: int = DEFAULT_EXPECTED_CONTEXT,
    expected_output_dim: int = DEFAULT_EXPECTED_OUTPUT_DIM,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a text-only OpenCLIP ``TextTransformer`` without a visual tower.

    The checkpoint directory must be supplied locally. This function never
    downloads a model and never falls back to the 224px / 16-position checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    config = read_siglip_config(checkpoint_dir)
    weights = checkpoint_dir / "open_clip_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(
            f"Required SigLIP checkpoint weights are missing: {weights}. "
            "Expected the paired SO400M-14-SigLIP-384 checkpoint; no fallback is attempted."
        )
    model_cfg = config["model_cfg"]
    text_cfg_raw = model_cfg["text_cfg"]
    output_dim = int(model_cfg.get("embed_dim", text_cfg_raw["width"]))
    with _safetensors_header(weights) as header:
        pos_shape = header.get_slice("text.positional_embedding").get_shape()
    dims = validate_text_config(config, positional_shape=pos_shape)
    if dims["context_length"] != int(expected_context_length):
        raise ValueError(
            f"Checkpoint context is {dims['context_length']}, expected {expected_context_length}; "
            "context length is architectural and is not extended or silently substituted."
        )
    if output_dim != int(expected_output_dim) or dims["output_dim"] != int(expected_output_dim):
        raise ValueError(f"Checkpoint output dimension {output_dim} does not match expected {expected_output_dim}.")

    try:
        import open_clip
        import torch
        from open_clip.model import CLIPTextCfg, _build_text_tower
    except ImportError as exc:
        raise RuntimeError(f"The isolated Talk2DINO image needs its pinned OpenCLIP runtime: {exc}") from exc

    # Use OpenCLIP's own text-tower builder, but omit CustomTextCLIP's vision
    # tower entirely. Keep only fields understood by this installed release.
    cfg_names = {field.name for field in fields(CLIPTextCfg)}
    text_cfg = CLIPTextCfg(**{key: value for key, value in text_cfg_raw.items() if key in cfg_names})
    text_tower = _build_text_tower(output_dim, text_cfg)
    from safetensors import safe_open

    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        text_state = {
            key[len("text."):]: handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith("text.")
        }
    if not text_state:
        raise ValueError(f"No `text.*` parameters found in {weights}")
    incompatible = text_tower.load_state_dict(text_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"SigLIP text weight mismatch: {incompatible}")

    if device is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    text_tower.to(device).eval()
    for parameter in text_tower.parameters():
        parameter.requires_grad_(False)
    tokenizer = load_siglip_tokenizer(checkpoint_dir)
    metadata = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_repository_expected": "timm/ViT-SO400M-14-SigLIP-384",
        "checkpoint_revision": SIGLIP_TEXT_MODEL_REVISION,
        "weights_path": str(weights),
        "text_context_length": dims["context_length"],
        "text_width": dims["text_width"],
        "text_output_dim": output_dim,
        "text_layers": int(text_cfg_raw["layers"]),
        "text_heads": int(text_cfg_raw["heads"]),
        "text_positional_embedding_shape": list(pos_shape),
        "text_state_tensor_count": len(text_state),
        "strict_text_state_load": True,
        "missing_text_keys": list(incompatible.missing_keys),
        "unexpected_text_keys": list(incompatible.unexpected_keys),
        "open_clip_version": open_clip.__version__,
        "torch_version": torch.__version__,
        "tokenizer": tokenizer.metadata(),
        "visual_tower_loaded": False,
        "frozen": all(not parameter.requires_grad for parameter in text_tower.parameters()),
    }
    return text_tower, tokenizer, metadata


def _safetensors_header(path: Path):
    from safetensors import safe_open

    return safe_open(str(path), framework="pt", device="cpu")


def encode_text_embeddings(text_tower: Any, token_ids: Any, expected_dim: int = DEFAULT_EXPECTED_OUTPUT_DIM):
    """Encode with a frozen text tower and validate `[batch, output_dim]`."""
    import torch

    with torch.inference_mode():
        features = text_tower(token_ids)
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[-1] != expected_dim:
        raise AssertionError(f"Expected text features [B,{expected_dim}], got {getattr(features, 'shape', None)}")
    if not torch.isfinite(features).all():
        raise FloatingPointError("SigLIP text encoder returned non-finite features")
    return features
