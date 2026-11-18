import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.full_val2014_retrieval import run_split
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection


def test_chunked_multi_positive_retrieval_matches_brute_force():
    rng = np.random.default_rng(31)
    torch.manual_seed(19)
    regions = rng.normal(size=(5, 16, 1024)).astype(np.float16)
    texts = rng.normal(size=(10, 1152)).astype(np.float16)
    image_ids = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    captions = np.array([[100 + i, 10 + i // 2] for i in range(10)], dtype=np.int64)
    model = RegionTextProjection().eval()
    model._epoch = 20
    with tempfile.TemporaryDirectory() as directory:
        result = run_split(model, regions, image_ids, texts, captions, torch.device("cpu"),
                           "synthetic", Path(directory), "manifest-sha", Path(directory),
                           image_chunk=2, caption_chunk=3)
    with torch.inference_mode():
        region_t = F.normalize(torch.from_numpy(regions.astype(np.float32)), dim=-1)
        text_t = F.normalize(model.text_projection(torch.from_numpy(texts.astype(np.float32))), dim=-1)
        scores = torch.stack([torch.einsum("hd,nd->nh", region_t[i], text_t).max(dim=-1).values
                              for i in range(len(image_ids))])
        positive = torch.as_tensor(image_ids)[:, None] == torch.as_tensor(captions[:, 1])[None, :]
        image_ranks = torch.stack([(scores[i] > scores[i][positive[i]].max()).sum() + 1
                                   for i in range(len(image_ids))])
        image_rows = {int(value): i for i, value in enumerate(image_ids)}
        text_ranks = torch.stack([(scores[:, j] > scores[image_rows[int(captions[j, 1])], j]).sum() + 1
                                  for j in range(len(captions))])
        logits = scores / .07
        loss_i = torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
        loss_t = torch.logsumexp(logits, dim=0) - logits[torch.as_tensor([image_rows[int(x)] for x in captions[:, 1]]), torch.arange(len(captions))]
    assert result["candidate_pairs"] == len(image_ids) * len(captions)
    np.testing.assert_allclose(result["image_to_text"]["rank_mean"], float(image_ranks.float().mean()), rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["text_to_image"]["rank_mean"], float(text_ranks.float().mean()), rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["multi_positive_infonce"]["image_to_text"], float(loss_i.mean()), rtol=1e-6)
    np.testing.assert_allclose(result["multi_positive_infonce"]["text_to_image"], float(loss_t.mean()), rtol=1e-6)
