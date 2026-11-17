import unittest

import torch
from torch import nn

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.dino_features import final_patch_and_region_features


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(1024, 3072, bias=False)
        self.num_heads = 16
        self.q_norm = nn.LayerNorm(64)
        self.k_norm = nn.LayerNorm(64)
        self.scale = 64 ** -0.5


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_prefix_tokens = 5
        self.num_reg_tokens = 4
        self.num_features = 1024
        self.patch_embed = type("PatchEmbed", (), {"grid_size": (16, 16)})()
        self.blocks = nn.ModuleList([FakeBlock()])
        self.norm = nn.LayerNorm(1024)

    def get_intermediate_layers(self, pixels, n, norm, return_prefix_tokens):
        b = pixels.shape[0]
        tokens = torch.arange(b * 261 * 1024, dtype=pixels.dtype, device=pixels.device).reshape(b, 261, 1024) / 1000
        self.blocks[-1].attn.qkv(tokens)
        tokens = self.norm(tokens)
        if return_prefix_tokens:
            return ((tokens[:, 5:, :], tokens[:, :5, :]),)
        return (tokens[:, 5:, :],)


class DinoFeatureTests(unittest.TestCase):
    def test_excludes_cls_and_four_registers_and_returns_heads(self):
        model = FakeDino()
        pixels = torch.zeros(1, 3, 224, 224)
        patches, attention, regions = final_patch_and_region_features(model, pixels)
        self.assertEqual(tuple(patches.shape), (1, 256, 1024))
        self.assertEqual(tuple(attention.shape), (1, 16, 256))
        self.assertEqual(tuple(regions.shape), (1, 16, 1024))
        self.assertTrue(torch.allclose(attention.sum(-1), torch.ones(1, 16), atol=1e-6))
        expected = (patches.unsqueeze(1) * attention.to(patches.dtype).unsqueeze(-1)).mean(2)
        self.assertTrue(torch.equal(regions, expected))
        self.assertEqual(len(model.blocks[-1].attn.qkv._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
