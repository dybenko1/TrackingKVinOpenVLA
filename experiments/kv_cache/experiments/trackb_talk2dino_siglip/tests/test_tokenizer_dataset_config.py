import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.config import validate_feature_manifests
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.dataset import CocoAnnotationDataset, repeated_image_id_count
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.tokenizer import SigLIPTokenizerBundle, validate_text_config


class FakeInnerTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[1] + [ord(char) % 50 + 3 for char in text] + [2] for text in texts]}

    def convert_ids_to_tokens(self, ids):
        return [str(token) for token in ids]


class FakeWrapper:
    def __init__(self):
        self.tokenizer = FakeInnerTokenizer()
        self.clean_calls = []

    def clean_fn(self, text):
        self.clean_calls.append(text)
        return text.replace("_", " ").lower()

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        clean = [self.clean_fn(text) for text in texts]
        ids = self.tokenizer(clean)["input_ids"]
        return torch.tensor([row + [0] * (12 - len(row)) for row in ids])


class TokenizerDatasetTests(unittest.TestCase):
    def test_context_and_positional_length_are_checkpoint_derived(self):
        config = {"model_cfg": {"embed_dim": 1152, "text_cfg": {"context_length": 64, "width": 1152}}}
        self.assertEqual(validate_text_config(config, (64, 1152)), {"context_length": 64, "text_width": 1152, "output_dim": 1152})
        with self.assertRaises(ValueError):
            validate_text_config(config, (16, 1152))

    def test_audit_and_feature_tokenization_share_clean_fn(self):
        wrapper = FakeWrapper()
        bundle = SigLIPTokenizerBundle(wrapper, 12, Path("fake"), {"model_cfg": {"text_cfg": {"tokenizer_kwargs": {"clean": "canonicalize"}}}})
        text = "RED_CUP!"
        audit_ids = bundle.tokenize_untruncated(text)[0]
        feature_ids = bundle(text).tolist()[0]
        self.assertEqual(wrapper.clean_calls, [text, text])
        self.assertEqual(feature_ids[:len(audit_ids)], audit_ids)
        self.assertEqual(bundle.truncation_flags(text), [len(audit_ids) > 12])

    def test_annotation_level_join_preserves_captions_and_logs_duplicate_image_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "image_ids.npy", np.array([10, 20]))
            np.save(root / "regions.npy", np.zeros((2, 16, 1024), dtype=np.float16))
            np.save(root / "captions.npy", np.array([[101, 10], [102, 10], [201, 20]]))
            np.save(root / "texts.npy", np.ones((3, 1152), dtype=np.float16))
            data = CocoAnnotationDataset(root / "image_ids.npy", root / "regions.npy", root / "captions.npy", root / "texts.npy")
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0][2:], (10, 101))
            self.assertEqual(data[1][2:], (10, 102))
            self.assertEqual(repeated_image_id_count([10, 10, 20]), 1)

    def test_train_validation_feature_contract_matches_except_row_count(self):
        train = {"feature_definition": "talk2dino_final_layer_cls_attention_weighted_patch_mean_v1", "preprocessing": "imagenet224", "dino_model": "vitl14", "dino_revision": "abc", "region_shape": [100, 16, 1024]}
        val = {**train, "region_shape": [20, 16, 1024]}
        validate_feature_manifests(train, val)
        with self.assertRaises(ValueError):
            validate_feature_manifests(train, {**val, "feature_definition": "different"})


if __name__ == "__main__":
    unittest.main()
