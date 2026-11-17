#!/usr/bin/env python3
"""Optional Talk2DINO category-map inference; not connected to OpenVLA actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.extract_visual import load_vision
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.inference_adapter import Talk2DINOAdapter
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import load_siglip_text_tower


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("categories", nargs="+", help='e.g. "red cube" "black bowl"')
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--siglip-dir", type=Path, default=Path("/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384"))
    parser.add_argument("--output-dir", type=Path, default=Path("talk2dino_output"))
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    model, _processor, device, _torch = load_vision()
    dino = model.vision_backbone.featurizer
    text_tower, tokenizer, text_meta = load_siglip_text_tower(args.siglip_dir, str(device))
    checkpoint = torch.load(args.projection, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {}).get("model", {})
    projection = RegionTextProjection(
        text_dim=int(cfg.get("clip_embed_dim", 1152)),
        hidden_dim=int(cfg.get("dino_embed_dim", 1024)),
        region_dim=int(cfg.get("dino_embed_dim", 1024)),
    )
    projection.load_state_dict(checkpoint.get("state_dict", checkpoint))
    adapter = Talk2DINOAdapter(dino, text_tower, tokenizer, projection, device)
    image = Image.open(args.image).convert("RGB")
    maps = adapter.score(image, args.categories)
    scores = maps["patch_similarity"].float().cpu().numpy()
    guided = maps["attention_guided_similarity"].float().cpu().numpy()
    labels = scores.argmax(axis=0).astype(np.uint8)
    union = scores.max(axis=0) >= args.threshold
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "patch_maps.npz", scores=scores, attention_guided=guided,
                        label_map=labels, union_mask=union.astype(np.uint8), categories=np.asarray(args.categories))

    # Match the Talk2DINO 224 transform's square output for the visualization.
    rgb = image.resize((224, 224), Image.Resampling.BICUBIC)
    palette = np.asarray([[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48],
                          [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60]], dtype=np.uint8)
    label_img = Image.fromarray(palette[labels % len(palette)].repeat(14, 0).repeat(14, 1)).resize(rgb.size, Image.Resampling.NEAREST)
    panel = Image.new("RGB", (448, 224 * (len(args.categories) + 1)), "white")
    panel.paste(rgb, (0, 0)); panel.paste(Image.blend(rgb, label_img, .45), (224, 0))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 4), "RGB", fill="white"); draw.text((228, 4), "Category argmax", fill="white")
    for idx, category in enumerate(args.categories):
        values = scores[idx]
        lo, hi = float(values.min()), float(values.max())
        gray = Image.fromarray(np.uint8(np.clip((values - lo) / max(hi - lo, 1e-8), 0, 1) * 255)).resize((224, 224), Image.Resampling.NEAREST)
        row_y = (idx + 1) * 224
        panel.paste(gray.convert("RGB"), (0, row_y))
        panel.paste(Image.blend(rgb, gray.convert("RGB"), .5), (224, row_y))
        draw.text((4, row_y + 4), f"{category}: patch similarity", fill="white")
        draw.text((228, row_y + 4), f"{category}: attention guided", fill="white")
    panel.save(args.output_dir / "overlay.png")
    metadata = {
        "image": str(args.image), "categories": args.categories, "grid": [16, 16],
        "patch_similarity_shape": list(scores.shape), "attention_guided_shape": list(guided.shape),
        "text_checkpoint": text_meta, "threshold": args.threshold,
        "score_semantics": "cosine similarity, not calibrated probability",
        "policy_action_path_modified": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
