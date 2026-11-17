import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.inference_adapter import Talk2DINOAdapter
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import encode_text_embeddings


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12, bias=False)
        self.num_heads = 2
        self.q_norm = None
        self.k_norm = None
        self.scale = 2 ** -0.5


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_prefix_tokens, self.num_reg_tokens, self.num_features = 5, 4, 4
        self.patch_embed = type("PatchEmbed", (), {"grid_size": (16, 16)})()
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[-1].attn = FakeAttention()
        self.norm = nn.LayerNorm(4)

    def get_intermediate_layers(self, pixels, n, norm, return_prefix_tokens):
        b = pixels.shape[0]
        x = torch.randn(b, 261, 4, dtype=pixels.dtype, device=pixels.device)
        self.blocks[-1].attn.qkv(x)
        x = self.norm(x)
        return ((x[:, 5:], x[:, :5]),)


class FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, token_ids):
        return torch.ones(token_ids.shape[0], 1152, device=token_ids.device) * self.weight


class FakeTokenizer:
    def __call__(self, texts):
        return torch.ones(len(texts), 64, dtype=torch.long)


class TextAdapterTests(unittest.TestCase):
    def test_siglip_text_dimension_finite_and_no_grad(self):
        encoder = FakeText()
        result = encode_text_embeddings(encoder, torch.ones(2, 64, dtype=torch.long), expected_dim=1152)
        self.assertEqual(tuple(result.shape), (2, 1152))
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(encoder.weight.requires_grad)
        self.assertIsNone(encoder.weight.grad)

    def test_adapter_emits_category_maps_without_policy_hooks(self):
        dino, text = FakeDino(), FakeText()
        projection = RegionTextProjection(text_dim=1152, hidden_dim=8, region_dim=4)
        adapter = Talk2DINOAdapter(dino, text, FakeTokenizer(), projection, "cpu")
        with patch("experiments.kv_cache.experiments.trackb_talk2dino_siglip.inference_adapter.preprocess_talk2dino_image", return_value=torch.zeros(3, 224, 224)):
            result = adapter.score(object(), ["cup", "bowl"])
        self.assertEqual(tuple(result["patch_similarity"].shape), (2, 16, 16))
        self.assertEqual(tuple(result["attention_guided_similarity"].shape), (2, 16, 16))
        self.assertEqual(tuple(result["winning_head_indices"].shape), (2,))
        self.assertEqual(len(dino.blocks[-1].attn.qkv._forward_hooks), 0)
        self.assertTrue(all(p.grad is None for p in dino.parameters()))
        self.assertTrue(all(p.grad is None for p in text.parameters()))

    def test_openvla_standard_backbone_does_not_import_sidecar(self):
        from pathlib import Path
        source = Path(__file__).resolve().parents[5] / "prismatic/models/backbones/vision/dinosiglip_vit.py"
        text = source.read_text()
        self.assertNotIn("trackb_talk2dino_siglip", text)
        self.assertIn("def forward", text)


if __name__ == "__main__":
    unittest.main()
