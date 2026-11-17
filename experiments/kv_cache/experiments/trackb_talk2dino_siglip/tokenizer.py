"""Shared checkpoint-configured OpenCLIP/Hugging Face SigLIP tokenizer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class SigLIPTokenizerBundle:
    wrapper: Any
    context_length: int
    checkpoint_dir: Path
    config: dict[str, Any]

    def __call__(self, texts: str | Sequence[str]):
        return self.wrapper(texts)

    def tokenize_untruncated(self, texts: str | Sequence[str]) -> list[list[int]]:
        """Tokenize using the same OpenCLIP ``clean_fn`` and special tokens."""
        if isinstance(texts, str):
            texts = [texts]
        cleaned = [self.wrapper.clean_fn(str(text)) for text in texts]
        encoded = self.wrapper.tokenizer(
            cleaned,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        return [list(map(int, ids)) for ids in encoded["input_ids"]]

    def truncation_flags(self, texts: str | Sequence[str]) -> list[bool]:
        return [len(ids) > self.context_length for ids in self.tokenize_untruncated(texts)]

    def metadata(self) -> dict[str, Any]:
        inner = self.wrapper.tokenizer
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "tokenizer_class": inner.__class__.__name__,
            "clean": self.config["model_cfg"]["text_cfg"].get("tokenizer_kwargs", {}).get("clean"),
            "canonicalization": "OpenCLIP HFTokenizer.clean_fn from checkpoint configuration",
            "context_length": self.context_length,
            "pad_token_id": getattr(inner, "pad_token_id", None),
            "eos_token_id": getattr(inner, "eos_token_id", None),
        }


def read_siglip_config(checkpoint_dir: Path) -> dict[str, Any]:
    config_path = checkpoint_dir / "open_clip_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"SigLIP checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if "model_cfg" not in config or "text_cfg" not in config["model_cfg"]:
        raise ValueError(f"Malformed OpenCLIP text configuration in {config_path}")
    return config


def validate_text_config(config: dict[str, Any], positional_shape: Sequence[int] | None = None) -> dict[str, int]:
    text_cfg = config["model_cfg"]["text_cfg"]
    context = int(text_cfg["context_length"])
    width = int(text_cfg["width"])
    output_dim = int(config["model_cfg"].get("embed_dim", width))
    if context <= 0 or width <= 0 or output_dim <= 0:
        raise ValueError(f"Invalid SigLIP dimensions: context={context}, width={width}, output={output_dim}")
    if positional_shape is not None and tuple(map(int, positional_shape)) != (context, width):
        raise ValueError(
            f"Text positional embedding shape {tuple(positional_shape)} does not match config {(context, width)}"
        )
    return {"context_length": context, "text_width": width, "output_dim": output_dim}


def load_siglip_tokenizer(checkpoint_dir: Path) -> SigLIPTokenizerBundle:
    """Load only local tokenizer assets; never downloads or picks a fallback."""
    checkpoint_dir = Path(checkpoint_dir)
    config = read_siglip_config(checkpoint_dir)
    dims = validate_text_config(config)
    tokenizer_name = config["model_cfg"]["text_cfg"].get("hf_tokenizer_name", "")
    if tokenizer_name not in {
        "timm/ViT-SO400M-14-SigLIP",
        "timm/ViT-SO400M-14-SigLIP-384",
    }:
        raise ValueError("Unexpected tokenizer identity in SigLIP checkpoint configuration")
    from open_clip.tokenizer import HFTokenizer

    tokenizer = HFTokenizer(
        str(checkpoint_dir),
        context_length=dims["context_length"],
        cache_dir=None,
        **config["model_cfg"]["text_cfg"].get("tokenizer_kwargs", {}),
        local_files_only=True,
    )
    return SigLIPTokenizerBundle(tokenizer, dims["context_length"], checkpoint_dir, config)
