"""Text-only OpenAI CLIP and Talk2DINO inference primitives.

The DINO adapters capture the existing model pass. They never call the backbone
twice: the regular OpenVLA forward returns its normalized block-23 output while
hooks also retain the block-24 output and QKV projection for Talk2DINO.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class _CLIPResidualBlock(nn.Module):
    def __init__(self, width: int, heads: int, causal_mask: torch.Tensor):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", nn.Identity()),
            ("c_proj", nn.Linear(width * 4, width)),
        ]))
        self.ln_2 = nn.LayerNorm(width)
        self.register_buffer("attn_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match OpenAI CLIP's ResidualAttentionBlock ordering and causal mask.
        q = self.ln_1(x)
        x = x + self.attn(q, q, q, need_weights=False, attn_mask=self.attn_mask)[0]
        y = self.ln_2(x)
        y = self.mlp.c_proj(_quick_gelu(self.mlp.c_fc(y)))
        return x + y


class _CLIPTransformer(nn.Module):
    def __init__(self, width: int, heads: int, layers: int, context: int):
        super().__init__()
        causal = torch.empty(context, context).fill_(float("-inf")).triu_(1)
        self.resblocks = nn.ModuleList([_CLIPResidualBlock(width, heads, causal) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class OpenAIClipTextTower(nn.Module):
    """OpenAI CLIP ViT-B/16 text tower only; no CLIP visual module is created."""

    def __init__(self, context_length: int = 77, width: int = 512, layers: int = 12, heads: int = 8,
                 vocab_size: int = 49408, output_dim: int = 512):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.transformer = _CLIPTransformer(width, heads, layers, context_length)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def load_openai_state(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        # OpenAI's published .pt is a TorchScript archive. Its state dictionary
        # has the canonical OpenAI key names.
        try:
            module = torch.jit.load(str(path), map_location="cpu")
            raw = module.state_dict()
        except Exception:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        # TorchScript's archived CLIP state also carries these scalar metadata
        # entries; they are not text-tower parameters/buffers.
        metadata_keys = {"logit_scale", "context_length", "input_resolution", "vocab_size"}
        state = {k: v for k, v in raw.items()
                 if not k.startswith("visual.") and k not in metadata_keys}
        expected = set(self.state_dict())
        got = set(state)
        missing, unexpected = sorted(expected - got), sorted(got - expected)
        if missing or unexpected:
            raise RuntimeError(f"CLIP text state mismatch: missing={missing}, unexpected={unexpected}")
        self.load_state_dict(state, strict=True)
        return {"source": str(path), "text_keys": len(state), "missing": missing, "unexpected": unexpected,
                "visual_tower_loaded": False}

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.context_length:
            raise ValueError(f"Expected token ids [B,{self.context_length}], got {tuple(tokens.shape)}")
        x = self.token_embedding(tokens).to(self.positional_embedding.dtype)
        x = x + self.positional_embedding
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        # OpenAI CLIP uses the position of the largest token id (EOT=49407).
        x = x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)]
        return x @ self.text_projection


class Talk2DINOProjection(nn.Module):
    """Exact ViT-L projection checkpoint structure from Talk2DINO."""

    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(512, 1024)
        self.hidden_layers = nn.ModuleList([nn.Linear(1024, 1024)])

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.linear_layer(text.float()))
        return self.hidden_layers[0](x)

    def load_official(self, path: str | Path) -> dict[str, Any]:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=True)
        return {"source": str(path), "sha256_required_by_experiment":
                "4e28d24ecb905592c902b04d9c077484346ebafac8547b968bd5af16291d8d69",
                "strict": True, "keys": sorted(state)}


def open_clip_tokenize(texts: list[str]) -> torch.Tensor:
    """Use OpenCLIP's OpenAI-BPE-compatible tokenizer with CLIP's 77 context."""
    import open_clip
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    tokens = tokenizer(texts)
    if tuple(tokens.shape) != (len(texts), 77):
        raise RuntimeError(f"OpenAI CLIP ViT-B/16 tokenizer returned {tuple(tokens.shape)}")
    return tokens


def _qkv_logits(qkv_module: nn.Module, qkv_output: torch.Tensor, batch: int, heads: int,
                head_dim: int, scale: float, *, q_norm=None, k_norm=None, num_global_tokens: int = 5):
    # Implements the official Talk2DINO process_self_attention(): scale Q,
    # compute QK^T, take CLS query and remove CLS + four register key positions.
    b, n, three_d = qkv_output.shape
    if b != batch or three_d != 3 * heads * head_dim:
        raise RuntimeError(f"Unexpected final-block QKV shape {tuple(qkv_output.shape)}")
    qkv = qkv_output.reshape(batch, n, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    if q_norm is not None:
        q = q_norm(q)
    if k_norm is not None:
        k = k_norm(k)
    logits = (q.float() * float(scale)) @ k.float().transpose(-2, -1)
    patch_logits = logits[:, :, 0, num_global_tokens:]
    if patch_logits.shape[-1] != 256:
        raise RuntimeError(f"Expected 256 CLS-to-patch attention logits, got {tuple(patch_logits.shape)}")
    return patch_logits[0]


class TimmTalk2DINOHook:
    """Capture OpenVLA's penultimate and final states from one normal DINO pass."""

    def __init__(self, dino: nn.Module):
        self.dino = dino
        self.records: dict[str, torch.Tensor] = {}
        self.handles = []
        self.module_calls = 0
        self.block_calls = [0] * 24
        self.num_heads = None

    def __enter__(self):
        blocks = self.dino.blocks
        self.num_heads = int(getattr(self.dino, "num_heads", blocks[-1].attn.num_heads))
        if len(blocks) != 24 or int(self.dino.embed_dim) != 1024 or self.num_heads != 16:
            raise RuntimeError(f"Expected OpenVLA DINOv2 ViT-L/14-register: 24 blocks, 1024 width, 16 heads; got {len(blocks)}/{self.dino.embed_dim}/{self.num_heads}")
        def module_hook(_m, _i, out):
            self.module_calls += 1
            self.records["policy"] = out[0] if isinstance(out, (tuple, list)) else out
        self.handles = [self.dino.register_forward_hook(module_hook)]
        for idx, block in enumerate(blocks):
            def count_hook(_m, _i, out, block_index=idx):
                self.block_calls[block_index] += 1
                if block_index == 22: self.records["block23"] = out
                if block_index == 23: self.records["block24"] = out
            self.handles.append(block.register_forward_hook(count_hook))
        self.handles.append(blocks[23].attn.qkv.register_forward_hook(
            lambda _m, _i, out: self.records.__setitem__("qkv24", out)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def extract_from_one_normal_pass(self, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        """Call the existing OpenVLA DINO forward once and capture its intermediate states."""
        if not self.handles:
            raise RuntimeError("Hooks must be installed with a context manager")
        self.records.clear()
        self.module_calls = 0
        self.block_calls = [0] * len(self.dino.blocks)
        self.dino(pixels)
        return self.build_from_captured(pixels)

    def build_from_captured(self, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        """Derive sidecar features from the already-completed normal DINO pass."""
        required = {"policy", "block23", "block24", "qkv24"}
        if not required <= self.records.keys():
            raise RuntimeError(f"Normal OpenVLA DINO pass did not fire hooks: keys={list(self.records)}")
        policy_penultimate = self.records["policy"]
        block23 = self.records["block23"]
        block24 = self.records["block24"]
        # TIMM 0.9.16 _intermediate_layers() records the requested penultimate
        # output but continues through the remaining block before returning.
        # Consequently block 24 and its QKV already exist in this exact pass.
        if block23.ndim != 3 or block23.shape[1] != 261 or block24.shape != block23.shape:
            raise RuntimeError(f"Expected [B,261,1024] CLS+register+patch tokens: {block23.shape}/{block24.shape}")
        norm = self.dino.norm
        # OpenVLA calls get_intermediate_layers(norm=False), so its policy path
        # receives the raw block-23 patch states. LayerNorm is applied only to
        # the block-24 sidecar representation required by Talk2DINO.
        reconstructed_policy = block23[:, 5:, :]
        policy_penultimate = policy_penultimate
        if policy_penultimate.shape != reconstructed_policy.shape:
            raise RuntimeError("OpenVLA policy feature layout differs from hooked block 23")
        difference = (policy_penultimate.float() - reconstructed_policy.float()).abs().max()
        tolerance = 2e-3 if policy_penultimate.dtype in (torch.float16, torch.bfloat16) else 1e-5
        if float(difference) > tolerance:
            raise RuntimeError(f"Hooked block23 does not reproduce policy output: max abs diff={float(difference)}")
        expected_calls = [1] * 24
        if self.module_calls != 1 or self.block_calls != expected_calls:
            raise RuntimeError(f"Expected one normal OpenVLA DINO pass through all 24 blocks; module={self.module_calls} blocks={self.block_calls}")
        final = norm(block24)[:, 5:, :]
        attention = _qkv_logits(self.dino.blocks[23].attn.qkv, self.records["qkv24"],
                                pixels.shape[0], self.num_heads,
                                self.dino.embed_dim // self.num_heads,
                                self.dino.blocks[23].attn.scale,
                                q_norm=getattr(self.dino.blocks[23].attn, "q_norm", None),
                                k_norm=getattr(self.dino.blocks[23].attn, "k_norm", None))
        if tuple(final.shape) != (pixels.shape[0], 256, 1024):
            raise RuntimeError(f"Malformed Talk2DINO final features {tuple(final.shape)}")
        return {"policy_penultimate": policy_penultimate, "block23_norm": reconstructed_policy,
                "final_patch_features": final, "attention_logits": attention,
                "block23_tokens": block23, "block24_tokens": block24,
                "qkv_calls": torch.tensor(1, device=pixels.device)}


class OfficialDINOv2Hook:
    """Capture final normalized features and final-block attention from Meta DINOv2."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.records: dict[str, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        blocks = self.model.blocks
        if len(blocks) != 24 or self.model.embed_dim != 1024 or self.model.num_heads != 16:
            raise RuntimeError("Expected official DINOv2 ViT-L/14 with four register tokens")
        self.handles = [
            blocks[-1].register_forward_hook(lambda _m, _i, out: self.records.__setitem__("block24", out)),
            blocks[-1].attn.qkv.register_forward_hook(lambda _m, _i, out: self.records.__setitem__("qkv24", out)),
        ]
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def forward(self, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        self.records.clear()
        out = self.model.forward_features(pixels)
        if "block24" not in self.records or "qkv24" not in self.records:
            raise RuntimeError("Official DINO pass did not expose final block/QKV")
        patches = out.get("x_norm_patchtokens") if isinstance(out, dict) else None
        if patches is None or tuple(patches.shape) != (pixels.shape[0], 256, 1024):
            raise RuntimeError(f"Official DINO returned incompatible patch layout: {getattr(patches,'shape',None)}")
        block = self.model.blocks[-1]
        attn = _qkv_logits(block.attn.qkv, self.records["qkv24"], pixels.shape[0], self.model.num_heads,
                           self.model.embed_dim // self.model.num_heads, block.attn.scale,
                           q_norm=getattr(block.attn, "q_norm", None),
                           k_norm=getattr(block.attn, "k_norm", None))
        return {"final_patch_features": patches, "attention_logits": attn,
                "block24_tokens": self.records["block24"]}


def talk2dino_maps(patches: torch.Tensor, attention_logits: torch.Tensor,
                   text_features: torch.Tensor, projection: Talk2DINOProjection,
                   lambda_bg: float = 0.2) -> dict[str, torch.Tensor]:
    """Original Talk2DINO cosine→sigmoid maps and attention-weighted cleaning."""
    if patches.ndim != 2 or patches.shape != (256, 1024):
        raise ValueError(f"patches must be [256,1024], got {tuple(patches.shape)}")
    if attention_logits.shape != (16, 256):
        raise ValueError(f"attention logits must be [16,256], got {tuple(attention_logits.shape)}")
    if text_features.ndim != 2 or text_features.shape[1] != 512:
        raise ValueError(f"CLIP text features must be [C,512], got {tuple(text_features.shape)}")
    projected = F.normalize(projection(text_features.float()), dim=-1)
    patch_unit = F.normalize(patches.float(), dim=-1)
    cosine = projected @ patch_unit.T
    raw_soft = cosine.sigmoid()
    # Direct equivalent to DINOText.similarity_assignment_weighted.
    feature_chw = patches.float().T  # [C,P]
    head_regions = (attention_logits.float().unsqueeze(1) * feature_chw.unsqueeze(0)).mean(dim=-1)
    head_regions = F.normalize(head_regions, dim=-1).T  # [C,heads]
    weights = (projected @ head_regions).softmax(dim=-1)  # per category, across heads
    attn_class = (attention_logits.float().unsqueeze(0) * weights.unsqueeze(-1)).sum(dim=1)
    attn_class = attn_class / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    attn_class = attn_class.softmax(dim=-1)
    amin, amax = attn_class.min(), attn_class.max()
    denom = torch.maximum(amax, amax - amin).clamp_min(1e-12)
    attn_scaled = (attn_class - amin) / denom
    mmin, mmax = raw_soft.min(), raw_soft.max()
    attn_scaled = attn_scaled * (mmax - mmin) + mmin
    cleaned = (raw_soft + lambda_bg * attn_scaled) / (1.0 + lambda_bg)
    return {"cosine": cosine, "raw_soft": raw_soft, "cleaned": cleaned,
            "attention_head_weights": weights, "attention_by_category": attn_class}


def semantic_labels(category_scores: torch.Tensor, threshold: float = 0.54) -> torch.Tensor:
    """Original global background rejection: max category score, else background."""
    if category_scores.ndim != 2 or category_scores.shape[1] != 256:
        raise ValueError("category_scores must be [C,256]")
    values, labels = category_scores.max(dim=0)
    return torch.where(values >= threshold, labels + 1, torch.zeros_like(labels))

