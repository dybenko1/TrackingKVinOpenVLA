import ast
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase3_spatial import (
    binary_mask_from_polygons,
    center_crop_offsets,
    decode_compressed_coco_rle,
    decode_coco_rle,
    sigmoid_masks,
    short_edge_resize_size,
    transform_mask_to_policy,
    weighted_background_cleaning,
)
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase3_evaluate import (
    extract_upstream_weighted_method,
    make_category_masks,
)


def encode_compressed_counts(counts):
    result = []
    for index, original in enumerate(counts):
        value = original - counts[index - 2] if index > 2 else original
        more = True
        while more:
            code = value & 0x1F
            value >>= 5
            if code & 0x10:
                more = value != -1
            else:
                more = value != 0
            if more:
                code |= 0x20
            result.append(chr(code + 48))
    return "".join(result)


class Phase3SpatialTests(unittest.TestCase):
    def test_sigmoid_matches_upstream_soft_mask(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        self.assertTrue(torch.equal(sigmoid_masks(x), torch.sigmoid(x)))

    def test_weighted_cleaning_matches_upstream_source(self):
        torch.manual_seed(17)
        b, n, c, h, w = 1, 3, 8, 4, 4
        mask = torch.sigmoid(torch.randn(b, n, h, w))
        features = F.normalize(torch.randn(b, c, h, w), dim=1)
        attention = torch.softmax(torch.randn(b, 16, h * w), dim=-1)
        text = F.normalize(torch.randn(n, c), dim=-1)
        upstream = extract_upstream_weighted_method()
        class Dummy: pass
        expected = upstream(Dummy(), mask, features, attention, text, 0.2)
        actual, prior = weighted_background_cleaning(mask, features, attention, text, 0.2)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))
        self.assertEqual(tuple(prior.shape), (b, n, h, w))
        self.assertTrue(torch.allclose(attention.sum(-1), torch.ones((b, 16)), atol=1e-6))

    def test_category_order_is_preserved(self):
        torch.manual_seed(23)
        mask = torch.sigmoid(torch.randn(1, 3, 4, 4))
        features = F.normalize(torch.randn(1, 8, 4, 4), dim=1)
        attn = torch.softmax(torch.randn(1, 16, 16), dim=-1)
        text = F.normalize(torch.randn(3, 8), dim=-1)
        expected, _ = weighted_background_cleaning(mask, features, attn, text)
        permutation = torch.tensor([2, 0, 1])
        actual, _ = weighted_background_cleaning(mask[:, permutation], features, attn, text[permutation])
        inverse = torch.argsort(permutation)
        self.assertTrue(torch.allclose(actual[:, inverse], expected, atol=1e-7, rtol=1e-6))

    def test_polygon_union_and_policy_alignment(self):
        first = binary_mask_from_polygons([[10, 10, 60, 10, 60, 50, 10, 50]], 120, 240)
        second = binary_mask_from_polygons([[100, 20, 150, 20, 150, 70, 100, 70]], 120, 240)
        union = first | second
        transformed = transform_mask_to_policy(union)
        self.assertEqual(tuple(transformed.shape), (1, 1, 224, 224))
        self.assertGreater(float(transformed.sum()), 0)
        # The aspect-ratio crop removes the far-left/right strips symmetrically;
        # both known interior objects must remain visible and separated.
        grid = F.interpolate(transformed, (16, 16), mode="nearest")[0, 0] > 0.5
        occupied_rows = torch.where(grid.any(dim=1))[0]
        occupied_cols = torch.where(grid.any(dim=0))[0]
        self.assertGreaterEqual(len(occupied_rows), 2)
        self.assertGreaterEqual(len(occupied_cols), 2)

    def test_resize_and_center_crop_match_torchvision_integer_geometry(self):
        cases = [(427, 640), (640, 427), (480, 640), (640, 480), (224, 224)]
        for h, w in cases:
            nh, nw = short_edge_resize_size(h, w, 224)
            self.assertEqual(min(nh, nw), 224)
            top, left = center_crop_offsets(nh, nw, 224)
            self.assertEqual((top, left), (round((nh - 224) / 2), round((nw - 224) / 2)))

    def test_category_instance_union(self):
        meta = {"height": 100, "width": 100}
        annotations = [
            {"category_id": 1, "segmentation": [[5, 5, 25, 5, 25, 25, 5, 25]]},
            {"category_id": 1, "segmentation": [[60, 60, 80, 60, 80, 80, 60, 80]]},
        ]
        masks = make_category_masks(meta, annotations)
        self.assertEqual(set(masks), {1})
        self.assertGreater(int(masks[1].sum()), 800)
        self.assertTrue(masks[1][10, 10] and masks[1][70, 70])

    def test_uncompressed_and_compressed_rle(self):
        counts = [3, 2, 4, 1]
        encoded = encode_compressed_counts(counts)
        self.assertEqual(decode_compressed_coco_rle(encoded), counts)
        mask = decode_coco_rle({"size": [2, 5], "counts": counts}, 2, 5)
        self.assertEqual(tuple(mask.shape), (2, 5))
        self.assertEqual(int(mask.sum()), counts[1] + counts[3])
        mask_c = decode_coco_rle({"size": [2, 5], "counts": encoded}, 2, 5)
        self.assertTrue(torch.equal(mask, mask_c))


if __name__ == "__main__":
    unittest.main()
