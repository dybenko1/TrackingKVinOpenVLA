import torch
import torch.nn.functional as F

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase3_spatial import weighted_background_cleaning
from experiments.kv_cache.experiments.trackb_talk2dino_native.model import (
    Talk2DINOProjection, semantic_labels, talk2dino_maps,
)


def test_official_cleaning_matches_phase3_reference():
    torch.manual_seed(13)
    projection = Talk2DINOProjection().eval()
    patches = torch.randn(256, 1024)
    attention = torch.randn(16, 256)
    text = torch.randn(4, 512)
    outputs = talk2dino_maps(patches, attention, text, projection)
    projected = F.normalize(projection(text), dim=-1)
    reference, _ = weighted_background_cleaning(
        outputs["raw_soft"].reshape(1, 4, 16, 16),
        patches.T.reshape(1, 1024, 16, 16),
        attention.reshape(1, 16, 256), projected, lambda_bg=0.2,
    )
    assert torch.allclose(outputs["cleaned"], reference.reshape(4, 256), atol=1e-6, rtol=1e-6)
    assert outputs["attention_head_weights"].shape == (4, 16)
    assert torch.allclose(outputs["attention_head_weights"].sum(-1), torch.ones(4), atol=1e-6)


def test_semantic_background_rejection_uses_global_threshold():
    scores = torch.zeros(2, 256)
    scores[:, 0] = torch.tensor([0.60, 0.30])
    scores[:, 1] = torch.tensor([0.20, 0.53])
    labels = semantic_labels(scores, 0.54)
    assert torch.equal(labels[:2], torch.tensor([1, 0]))
    assert torch.equal(labels[2:], torch.zeros(254, dtype=torch.long))


def test_projection_checkpoint_architecture_dimensions():
    projection = Talk2DINOProjection()
    assert projection(torch.zeros(3, 512)).shape == (3, 1024)
    assert sum(p.numel() for p in projection.parameters()) == 1024 * 512 + 1024 + 1024 * 1024 + 1024

