"""Optional Talk2DINO sidecar; does not alter OpenVLA's standard forward path."""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.dino_features import final_patch_and_region_features
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.preprocessing import preprocess_talk2dino_image
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import encode_text_embeddings


class Talk2DINOAdapter:
    """Score category phrases against one image using supplied frozen towers.

    `dino` must be the already initialized OpenVLA TIMM DINOv2 module. The
    auxiliary text tower and projection are explicit inputs; this class never
    loads a second vision model or monkey-patches policy methods.
    """

    def __init__(self, dino: Any, text_tower: Any, tokenizer: Any,
                 projection: RegionTextProjection, device: str | torch.device | None = None):
        self.dino = dino
        self.text_tower = text_tower
        self.tokenizer = tokenizer
        self.projection = projection
        self.device = torch.device(device) if device is not None else next(dino.parameters()).device
        self.dino.eval()
        self.text_tower.eval()
        for tower in (self.dino, self.text_tower):
            for parameter in tower.parameters():
                parameter.requires_grad_(False)
        self.projection.eval().to(self.device)

    @torch.inference_mode()
    def score(self, image: Any, categories: Sequence[str]) -> dict[str, torch.Tensor]:
        if not categories:
            raise ValueError("At least one category phrase is required")
        token_ids = self.tokenizer(list(categories)).to(self.device)
        text = encode_text_embeddings(
            self.text_tower, token_ids,
            expected_dim=self.projection.text_dim,
        ).float()
        pixels = preprocess_talk2dino_image(image).unsqueeze(0).to(
            self.device, dtype=next(self.dino.parameters()).dtype
        )
        patches, attention, regions = final_patch_and_region_features(self.dino, pixels)
        projected = F.normalize(self.projection.text_projection(text.float()), dim=-1)
        patch_features = F.normalize(patches.float(), dim=-1)
        region_features = F.normalize(regions.float(), dim=-1)
        patch_scores = torch.einsum("bpd,cd->bcp", patch_features, projected)
        head_scores = torch.einsum("bhd,cd->bch", region_features, projected)
        winning_heads = head_scores.argmax(dim=-1)
        guided = torch.stack([
            patch_scores[0, category_idx] * attention[0, int(winning_heads[0, category_idx])].float()
            for category_idx in range(len(categories))
        ])
        if not torch.isfinite(patch_scores).all() or not torch.isfinite(guided).all():
            raise FloatingPointError("Talk2DINO produced non-finite category maps")
        return {
            "patch_similarity": patch_scores[0].reshape(len(categories), 16, 16),
            "attention_guided_similarity": guided.reshape(len(categories), 16, 16),
            "winning_head_indices": winning_heads[0],
            "text_embeddings": text,
            "patch_embeddings": patches,
            "attention": attention,
        }
