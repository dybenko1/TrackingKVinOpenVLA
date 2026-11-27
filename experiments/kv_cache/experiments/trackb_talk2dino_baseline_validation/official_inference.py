"""Minimal adapter that invokes the cached upstream Talk2DINO inference code.

Only unused distributed/MMCV registration APIs are shimmed because the
validated OpenVLA runtime has PyTorch 2.6/Python 3.12 and no compatible
MMCV/MMsegmentation build. The upstream DINOText methods, CLIP checkpoint,
projection MLP, DINO model, tokenization, map computation, and PAMR are used.
"""
from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms


def _install_import_shims(official_repo: Path):
    """Provide optional infra symbols needed only to import original model code."""
    source = official_repo / "src/open_vocabulary_segmentation"
    for item in (official_repo, source):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))

    class Registry:
        def __init__(self, *_args, **_kwargs): self.items = {}
        def register_module(self, *_args, **_kwargs):
            def decorate(obj): self.items[obj.__name__] = obj; return obj
            return decorate
        def build(self, config):
            return self.items[config["type"]](**{k:v for k,v in config.items() if k != "type"})

    mmcv = types.ModuleType("mmcv"); mmcv.__path__ = []
    mmcv_utils = types.ModuleType("mmcv.utils"); mmcv_utils.Registry = Registry; mmcv.utils = mmcv_utils
    mmcv_engine = types.ModuleType("mmcv.engine")
    mmcv_engine.collect_results_cpu = lambda *_a, **_k: None
    mmcv_engine.collect_results_gpu = lambda *_a, **_k: None
    mmcv_image = types.ModuleType("mmcv.image"); mmcv_image.tensor2imgs = lambda *_a, **_k: None
    mmcv_runner = types.ModuleType("mmcv.runner"); mmcv_runner.get_dist_info = lambda: (0, 1)
    mmcv.engine, mmcv.image, mmcv.runner = mmcv_engine, mmcv_image, mmcv_runner
    sys.modules.update({"mmcv":mmcv, "mmcv.utils":mmcv_utils, "mmcv.engine":mmcv_engine,
                        "mmcv.image":mmcv_image, "mmcv.runner":mmcv_runner})

    mmseg = types.ModuleType("mmseg"); mmseg.__path__ = []
    apis = types.ModuleType("mmseg.apis"); apis.__path__ = []
    apis_test = types.ModuleType("mmseg.apis.test"); apis_test.np2tmp = lambda *_a, **_k: None
    sys.modules.update({"mmseg":mmseg, "mmseg.apis":apis, "mmseg.apis.test":apis_test})
    # The upstream inference class only asks for this logger during object
    # construction; distributed utilities are not called by our single-process
    # model evaluation.
    utils = types.ModuleType("utils")
    utils.get_logger = lambda *_a, **_k: logging.getLogger("talk2dino-upstream")
    sys.modules["utils"] = utils
    omega = types.ModuleType("omegaconf"); omega.OmegaConf = type("OmegaConf", (), {})
    sys.modules["omegaconf"] = omega
    clip = types.ModuleType("clip")
    import open_clip
    clip.tokenize = open_clip.get_tokenizer("ViT-B-16")
    sys.modules["clip"] = clip

    # Re-import the source module so it binds the current shim registry.
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "src" or name.startswith("src."):
            del sys.modules[name]
    return importlib.import_module("models.dinotext.dinotext")


class OfficialTalk2DINO:
    """Original DINOText methods with local, verified pretrained components."""
    def __init__(self, official_repo: Path, dino, clip_weights: Path, mlp_weights: Path,
                 resize_dim: int = 448, device: str = "cuda:0"):
        from experiments.kv_cache.experiments.trackb_talk2dino_native.model import OpenAIClipTextTower
        import yaml
        dino_module = _install_import_shims(official_repo)
        source = official_repo / "src/open_vocabulary_segmentation"
        sys.path.insert(0, str(official_repo))
        from src.model import ProjectionLayer

        clip_model = OpenAIClipTextTower().eval()
        clip_meta = clip_model.load_openai_state(clip_weights)
        clip_model.to(device)
        projection_cfg = yaml.safe_load((official_repo / "configs/vitl_mlp_infonce.yaml").read_text())["model"]
        projection = ProjectionLayer.from_config(projection_cfg)
        state = torch.load(mlp_weights, map_location="cpu", weights_only=True)
        # The upstream ProjectionLayer in this pinned checkout overrides
        # load_state_dict and returns None even after a strict load. PyTorch's
        # strict=True raises on any missing/unexpected parameter, so let that
        # exception be the integrity gate instead of inspecting a return value.
        projection.load_state_dict(state, strict=True)
        projection.eval().to(device)
        for module in (dino, clip_model, projection):
            module.eval(); module.requires_grad_(False)

        model = dino_module.DINOText.__new__(dino_module.DINOText)
        nn.Module.__init__(model)
        model.model = dino
        model.model_name = "dinov2_vitl14_reg"
        model.resize_dim = resize_dim
        model.clip_model_name = "ViT-B/16"
        model.clip_model = clip_model
        model.proj = projection
        model.feats = {}
        model.num_global_tokens = 5
        model.num_attn_heads = int(dino.num_heads)
        model.scale = 0.125
        model.avg_self_attn_token = False
        model.disentangled_self_attn_token = True
        model.use_avg_text_token = False
        model.keep_cls = False
        model.keep_end_seq = False
        model.with_bg_clean = True
        model.pamr = None
        model.image_transforms = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            lambda x: transforms.ToTensor()(x) if not isinstance(x, torch.Tensor) else x / 255.0,
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        model.masker = dino_module.DINOTextMasker(similarity_type="cosine").eval().to(device)
        self.model, self.clip_model, self.projection = model, clip_model, projection
        self.dino_module, self.clip_metadata = dino_module, clip_meta
        self.resize_dim = resize_dim
        self._qkv_hook = dino.blocks[-1].attn.qkv.register_forward_hook(model.get_self_attention)

    def encode_categories(self, classnames: list[str], template_set: str = "sub_imagenet_template"):
        """Use original build_dataset_class_tokens/build_text_embedding methods."""
        tokens = self.model.build_dataset_class_tokens(template_set, classnames)
        return self.model.build_text_embedding(tokens)

    @torch.inference_mode()
    def score_bgr_crop(self, crop_bgr, text_embedding, clean: bool = True, pamr: bool = False):
        """Call upstream DINOText.generate_masks on one MMCV-style BGR crop."""
        import numpy as np
        import torch
        model = self.model
        arr = np.asarray(crop_bgr)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected HWC BGR crop, got {arr.shape}")
        image = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(
            device=next(model.parameters()).device, dtype=torch.float32)
        model.feats.clear()
        model.with_bg_clean = bool(clean)
        masks, cosine = model.generate_masks(image, None, text_embedding,
                                             ["background"] + [str(i) for i in range(text_embedding.shape[0])],
                                             apply_pamr=False, lambda_bg=0.2)
        rgb = image[:, [2, 1, 0], :, :]
        return {"scores": masks[0], "cosine": cosine[0], "rgb": rgb[0],
                "attention_qkv_captured": "self_attn" in model.feats}

    def close(self):
        self._qkv_hook.remove()
        self._qkv_hook = None

