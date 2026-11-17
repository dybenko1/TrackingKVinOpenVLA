import unittest

import torch

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.losses import symmetric_infonce
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection


class ProjectionLossTests(unittest.TestCase):
    def test_projection_shape_parameter_count_max_head_and_grad(self):
        model = RegionTextProjection()
        self.assertEqual(model.trainable_parameter_count, 2_230_272)
        self.assertEqual(tuple(model.text_projection(torch.randn(2, 1152)).shape), (2, 1024))
        self.assertEqual([type(layer).__name__ for layer in model.text_projection], ["Linear", "Tanh", "Linear"])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimizer_ids, {id(p) for p in model.parameters()})
        regions = torch.randn(4, 16, 1024)
        text = torch.randn(4, 1152)
        scores, heads = model(regions, text, return_head_indices=True)
        self.assertEqual(tuple(scores.shape), (4, 4))
        self.assertEqual(tuple(heads.shape), (4, 4))
        self.assertTrue(torch.all((heads >= 0) & (heads < 16)))
        self.assertTrue(torch.allclose(torch.nn.functional.normalize(model.text_projection(text), dim=-1).norm(dim=-1), torch.ones(4), atol=1e-5))
        loss = symmetric_infonce(scores)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertFalse(text.requires_grad)

    def test_canonical_and_upstream_legacy_loss_scaling(self):
        scores = torch.tensor([[2.0, 0.1], [0.1, 2.0]], requires_grad=True)
        canonical = symmetric_infonce(scores, temperature=0.07)
        legacy = symmetric_infonce(scores, temperature=0.07, legacy_extra_batch_square=True)
        self.assertTrue(torch.isfinite(canonical) and torch.isfinite(legacy))
        self.assertAlmostEqual(float(legacy.detach()), float(canonical.detach()) / 4, places=6)
        legacy.backward()
        self.assertIsNotNone(scores.grad)


if __name__ == "__main__":
    unittest.main()
