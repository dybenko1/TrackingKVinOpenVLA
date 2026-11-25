#!/usr/bin/env python3
"""Adaptive V2 + task-relevance selective K/V reuse with rolling token state.

One configuration only: V2 p-threshold .20, attention protects 10% of current
V2 background, previous-action confidence margin gates full refresh at .65,
previous-step layer-15 attention, and foreground-to-background transitions are
refreshed. There is no periodic age cap. This is an experimental runner, not a
change to the normal OpenVLA policy path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
import traceback
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image, ImageDraw
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, repeat_kv

REPO = Path(__file__).resolve().parents[3]
import experiments.kv_cache as _kv_package
_kv_package.__path__.insert(0, str(REPO / "experiments/kv_cache"))
import experiments.kv_cache.experiments as _exp_package
_exp_package.__path__.insert(0, str(REPO / "experiments/kv_cache/experiments"))

from experiments.kv_cache.experiments.adaptive_hybrid_kv_policy import (
    TokenAgeTracker, compose_adaptive_mask, confidence_requires_refresh, rolling_merge_rows,
)
from experiments.kv_cache.experiments.stage3b_oracle_full_kv_matrix import (
    HORIZON, LAYERS, SEED, STATES, TASKS,
)
from experiments.kv_cache.experiments.stage3c_oracle_task_relevance import RefreshAttentionSidecar
from experiments.kv_cache.experiments.stage4a_oracle_free_attention import (
    ProjectionRowMonitor, _action_for_env, _policy_call, make_env,
)
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import (
    policy_image_from_observation, seed_everything,
)
from experiments.kv_cache.experiments.static_reuse_experiment import (
    DEFAULT_MODEL_ID, cache_layers, load_model_and_processor, resolve_unnorm_key,
)

BASE_OUT = REPO / "experiments/kv_cache/outputs/adaptive_hybrid_kv_cache"
V2_CKPT = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/best_projection.pt"
SIGLIP = Path("/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384")
V2_SHA256 = "8232ec8f657da47645d73741b16560a1140bfea89cce6826948b9457ce4854d4"
LAYOUT_VISUAL = (1, 256)
FG_THRESHOLD = 0.20
COSINE_OFFSET = 0.20
COSINE_SCALE = 10.0
ATTENTION_FRAC = 0.10
CONFIDENCE_THRESHOLD = 0.65


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=str)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def vector_stats(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(x.size), "mean": float(x.mean()) if x.size else None,
            "p50": float(np.percentile(x, 50)) if x.size else None,
            "p95": float(np.percentile(x, 95)) if x.size else None,
            "max": float(x.max()) if x.size else None}


def cache_identity_signature(cache):
    """Storage identity valid for PyTorch inference tensors without version counters."""
    return [(int(k.data_ptr()), int(v.data_ptr()), tuple(k.shape), tuple(v.shape), str(k.dtype), str(v.dtype))
            for k, v in cache_layers(cache)]


class DINOFeatureTap:
    """Passively capture the existing OpenVLA penultimate DINO feature tensor."""
    def __init__(self, model, callback=None):
        self.dino = getattr(model.vision_backbone, "dino_featurizer",
                            getattr(model.vision_backbone, "featurizer", None))
        if self.dino is None:
            raise RuntimeError("OpenVLA DINO featurizer is unavailable")
        self.callback = callback; self.features = None; self.calls = 0; self.handle = None

    def __enter__(self):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)): output = output[0]
            self.calls += 1; self.features = output.detach()
            if self.callback: self.callback(self.features)
        self.handle = self.dino.register_forward_hook(hook)
        return self

    def __exit__(self, *_):
        if self.handle is not None: self.handle.remove()


def v2_probabilities(features: torch.Tensor, text_vectors: torch.Tensor) -> tuple[np.ndarray, dict]:
    if tuple(features.shape) != (1, 256, 1024):
        raise RuntimeError(f"Native DINO feature shape mismatch: {tuple(features.shape)}")
    t0 = time.perf_counter(); ev0 = ev1 = None
    if features.is_cuda:
        ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True); ev0.record()
    patches = F.normalize(features.float(), dim=-1)[0]
    text = F.normalize(text_vectors.float(), dim=-1)
    probs = torch.sigmoid(COSINE_SCALE * (patches @ text.T - COSINE_OFFSET)).T
    if tuple(probs.shape) != (text.shape[0], 256) or not torch.isfinite(probs).all():
        raise RuntimeError("V2 probabilities have invalid dimensions or nonfinite values")
    if ev1 is not None: ev1.record()
    host_ms = (time.perf_counter() - t0) * 1000
    gpu_ms = float(ev0.elapsed_time(ev1)) if ev0 is not None else 0.0
    return probs.detach().cpu().numpy(), {"v2_gpu_ms": gpu_ms, "v2_scoring_cpu_transfer_wall_ms": host_ms}


def task_text(description, text_model, tokenizer, projection, device):
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.transfer_validation.libero_demo_ground_truth import instruction_phrases
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import encode_text_embeddings
    phrases = instruction_phrases(description.lower())
    prompts = [f"a photo of a {p['phrase']}" for p in phrases]
    if not prompts: raise RuntimeError(f"No object prompts found for task instruction {description!r}")
    ids = tokenizer(prompts).to(device)
    with torch.inference_mode():
        emb = encode_text_embeddings(text_model, ids).float()
        vectors = F.normalize(projection.text_projection(emb), dim=-1)
    return phrases, vectors


def save_adaptive_visual(path: Path, rgb, selection, ages, confidence, reason):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB").resize((320, 320), Image.Resampling.BICUBIC)
    panels = [("RGB", image)]
    for name, field, color in [
        ("V2 foreground", "foreground", (30, 175, 65)),
        ("Attention protect", "attention_protected", (240, 145, 15)),
        ("FG→BG transition", "transition_protected", (40, 130, 220)),
        ("Fresh union", "fresh", (35, 140, 60)),
        ("Reuse candidates", "reused", (65, 70, 80)),
    ]:
        grid = np.zeros((16, 16, 3), np.uint8); grid[:] = (35, 35, 35)
        grid.reshape(256, 3)[selection[field]] = color
        panels.append((f"{name}: {int(selection[field].sum())}/256", Image.fromarray(grid).resize((320, 320), Image.Resampling.NEAREST)))
    agegrid = np.zeros((16, 16, 3), np.uint8)
    ageflat = agegrid.reshape(256, 3)
    ageflat[:] = np.stack([np.clip(25 + ages*40, 0, 255), np.clip(170 - ages*25, 0, 255), np.clip(240 - ages*35, 0, 255)], axis=1).astype(np.uint8)
    panels.append(("Token age after decision", Image.fromarray(agegrid).resize((320, 320), Image.Resampling.NEAREST)))
    canvas = Image.new("RGB", (1280, 2 * 366), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), f"{reason} · prior margin={confidence if confidence is not None else 'n/a'}", fill="black")
    for i, (label, panel) in enumerate(panels):
        row, col = divmod(i, 4); x, y = col * 320, row * 366 + 35
        canvas.paste(panel, (x, y)); draw.text((x + 7, y - 18), label, fill="black")
    canvas.save(path)


def attention_scores_from_assembled(module, q_rot, assembled_k, attention_mask, seq_len):
    """Stage 3C score from current Q and actual full assembled (RoPE) K."""
    if q_rot.shape[-2] != seq_len or assembled_k.shape[-2] != seq_len:
        raise RuntimeError("Attention sidecar Q/K sequence length mismatch")
    if q_rot.shape[0] != 1:
        raise RuntimeError("Adaptive attention currently requires batch size 1")
    q_start = 257
    if q_start >= seq_len: raise RuntimeError("No text query positions after 256 image keys")
    q_pos = torch.arange(q_start, seq_len, dtype=torch.long, device=q_rot.device)
    q_heads = q_rot.shape[1]
    kv_groups = q_heads // assembled_k.shape[1]
    q = q_rot.index_select(-2, q_pos).float()
    k = repeat_kv(assembled_k, kv_groups).float()
    logits = torch.matmul(q, k.transpose(-1, -2)) * float(module.scaling)
    key_positions = torch.arange(seq_len, device=q.device)
    valid = key_positions[None, :] <= q_pos[:, None]
    valid = valid[None, None, :, :]
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            if attention_mask.ndim == 4:
                valid = valid & attention_mask.index_select(-2, q_pos)
            elif attention_mask.ndim == 2:
                valid = valid & attention_mask[:, None, None, :]
            else: raise RuntimeError(f"Unsupported boolean mask shape {tuple(attention_mask.shape)}")
        elif attention_mask.ndim == 4:
            valid = valid & torch.isfinite(attention_mask.index_select(-2, q_pos))
        else: raise RuntimeError(f"Unsupported additive mask shape {tuple(attention_mask.shape)}")
    probs = torch.softmax(logits.masked_fill(~valid, torch.finfo(logits.dtype).min), dim=-1)
    scores = probs[..., 1:257].mean(dim=(0, 1, 2)).detach()
    if tuple(scores.shape) != (256,) or not torch.isfinite(scores).all():
        raise RuntimeError("Assembled text-to-visual relevance must contain 256 finite values")
    row_error = float((probs.sum(-1) - 1).abs().max().item())
    return scores.cpu().numpy().astype(np.float32), row_error


class AdaptiveRollingKVSkip:
    """True full-layer selective K/V projection skipping over rolling cache rows."""
    def __init__(self, model, rolling_cache, layout, on_attention=None):
        self.model = model; self.source = rolling_cache; self.layout = dict(layout)
        self.length = int(layout["prefill_length"]); self.mask = None; self.scores = None
        self.capture_ms = None; self.row_sum_max_error = None; self.on_attention = on_attention
        self.original_forwards = {}; self.diag = {"layers": [], "rows_by_layer": [], "decode_rows_by_layer": [], "unexpected": []}

    def set_mask(self, mask):
        m = np.asarray(mask, dtype=bool)
        if m.shape != (256,): raise ValueError(f"Need 256 refresh decisions, got {m.shape}")
        self.mask = m.copy()

    def __enter__(self):
        if self.source is None: raise RuntimeError("Rolling cache is absent")
        vs, ve = self.layout["visual"]
        if (vs, ve) != LAYOUT_VISUAL: raise RuntimeError(f"Unsupported visual layout {vs,ve}")
        source_layers = cache_layers(self.source)
        if len(source_layers) != 32 or any(k.shape[-2] != self.length or v.shape[-2] != self.length for k, v in source_layers):
            raise RuntimeError("Rolling cache layer/sequence layout is incompatible")
        self.source_versions = cache_identity_signature(self.source)
        layers = self.model.language_model.model.layers
        for li in LAYERS:
            attn = layers[li].self_attn; self.original_forwards[li] = attn.forward
            def forward(module, hidden_states, position_embeddings, attention_mask,
                        past_key_value=None, cache_position=None, **kwargs):
                if hidden_states.shape[-2] != self.length or past_key_value is None:
                    if hidden_states.shape[-2] == 1:
                        self.diag["decode_rows_by_layer"].append({"layer": module.layer_idx, "rows": 1})
                    else: self.diag["unexpected"].append({"layer": module.layer_idx, "rows": int(hidden_states.shape[-2])})
                    return self.original_forwards[module.layer_idx](hidden_states, position_embeddings,
                        attention_mask, past_key_value, cache_position, **kwargs)
                if self.mask is None: raise RuntimeError("V2 mask callback did not run before Llama prefill")
                b, n, _ = hidden_states.shape; device = hidden_states.device
                vs, ve = self.layout["visual"]; ls, le = self.layout["language"]
                visual_fresh_idx = torch.as_tensor(np.flatnonzero(self.mask) + vs, dtype=torch.long, device=device)
                language_idx = torch.arange(ls, le + 1, dtype=torch.long, device=device)
                fresh_idx = torch.cat((torch.zeros(1, dtype=torch.long, device=device), visual_fresh_idx, language_idx))
                q_heads = module.q_proj.out_features // module.head_dim
                kv_heads = module.k_proj.out_features // module.head_dim
                query_raw = module.q_proj(hidden_states).view(b, n, q_heads, module.head_dim).transpose(1, 2)
                fresh_hidden = hidden_states.index_select(1, fresh_idx)
                fresh_k_raw = module.k_proj(fresh_hidden).view(b, fresh_idx.numel(), kv_heads, module.head_dim).transpose(1, 2)
                fresh_v = module.v_proj(fresh_hidden).view(b, fresh_idx.numel(), kv_heads, module.head_dim).transpose(1, 2)
                cos, sin = position_embeddings
                query, _ = apply_rotary_pos_emb(query_raw, query_raw, cos, sin)
                _, fresh_k = apply_rotary_pos_emb(fresh_k_raw, fresh_k_raw,
                    cos.index_select(1, fresh_idx), sin.index_select(1, fresh_idx))
                old_k, old_v = source_layers[module.layer_idx]
                key = rolling_merge_rows(old_k, fresh_k, fresh_idx, axis=2)
                value = rolling_merge_rows(old_v, fresh_v, fresh_idx, axis=2)
                projected_rows = int(fresh_idx.numel())
                reused_visual_idx = torch.as_tensor(np.flatnonzero(~self.mask) + vs,
                    dtype=torch.long, device=device)
                if not torch.equal(key.index_select(2, fresh_idx), fresh_k):
                    raise AssertionError(f"Layer {module.layer_idx}: freshly projected K rows mismatch")
                if not torch.equal(value.index_select(2, fresh_idx), fresh_v):
                    raise AssertionError(f"Layer {module.layer_idx}: freshly projected V rows mismatch")
                if not torch.equal(key.index_select(2, reused_visual_idx), old_k.index_select(2, reused_visual_idx)):
                    raise AssertionError(f"Layer {module.layer_idx}: rolling K rows changed before attention")
                if not torch.equal(value.index_select(2, reused_visual_idx), old_v.index_select(2, reused_visual_idx)):
                    raise AssertionError(f"Layer {module.layer_idx}: rolling V rows changed before attention")
                if module.layer_idx == 15:
                    t0 = time.perf_counter(); torch.cuda.synchronize(device)
                    self.scores, self.row_sum_max_error = attention_scores_from_assembled(
                        module, query, key, attention_mask, n)
                    torch.cuda.synchronize(device); self.capture_ms = (time.perf_counter() - t0) * 1000
                    if self.on_attention: self.on_attention(self.scores)
                key, value = past_key_value.update(key, value, module.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position})
                backend = getattr(module.config, "_attn_implementation", None)
                if backend != "sdpa": raise RuntimeError(f"Expected SDPA backend, got {backend!r}")
                interface = eager_attention_forward if backend == "eager" else ALL_ATTENTION_FUNCTIONS[backend]
                out, weights = interface(module, query, key, value, attention_mask,
                    dropout=0.0 if not module.training else module.attention_dropout,
                    scaling=module.scaling, **kwargs)
                self.diag["layers"].append(module.layer_idx)
                self.diag["rows_by_layer"].append({"layer": module.layer_idx,
                    "q_projected_rows": n, "k_projected_rows": projected_rows,
                    "v_projected_rows": projected_rows,
                    "visual_reused_rows": 256 - int(self.mask.sum()),
                    "full_q": True, "fresh_k_original_rope_positions": True,
                    "attention_uses_assembled_k": True})
                return module.o_proj(out.reshape(b, n, -1).contiguous()), weights
            attn.forward = types.MethodType(forward, attn)
        return self

    def __exit__(self, *_):
        for li, original in self.original_forwards.items():
            self.model.language_model.model.layers[li].self_attn.forward = original

    def validate(self, cache):
        if sorted(self.diag["layers"]) != list(LAYERS) or self.diag["unexpected"]:
            raise AssertionError("Selective prefill did not run exactly the 32 expected layers")
        if len(self.diag["decode_rows_by_layer"]) != 32 * 6:
            raise AssertionError("Autoregressive one-token decode was not delegated normally")
        rows = self.diag["rows_by_layer"]
        ls, le = self.layout["language"]
        expected = 1 + int(self.mask.sum()) + (le - ls + 1)
        if len(rows) != 32 or any(r["q_projected_rows"] != self.length or
                                  r["k_projected_rows"] != expected or
                                  r["v_projected_rows"] != expected for r in rows):
            raise AssertionError("Actual projection rows differ from selected mask")
        if self.scores is None or self.scores.shape != (256,): raise AssertionError("Missing adaptive attention scores")
        new_layers, old_layers = cache_layers(cache), cache_layers(self.source)
        reused = torch.as_tensor(np.flatnonzero(~self.mask) + 1, dtype=torch.long, device=new_layers[0][0].device)
        fresh = torch.as_tensor(np.flatnonzero(self.mask) + 1, dtype=torch.long, device=new_layers[0][0].device)
        for li, ((nk, nv), (ok, ov)) in enumerate(zip(new_layers, old_layers)):
            if not torch.equal(nk.index_select(2, reused), ok.index_select(2, reused)):
                raise AssertionError(f"Layer {li}: retained K rows differ from latest rolling source")
            if not torch.equal(nv.index_select(2, reused), ov.index_select(2, reused)):
                raise AssertionError(f"Layer {li}: retained V rows differ from latest rolling source")
            if not torch.equal(nk.index_select(2, fresh), cache_layers(cache)[li][0].index_select(2, fresh)):
                raise AssertionError(f"Layer {li}: fresh K rows were not retained")
        if cache_identity_signature(self.source) != self.source_versions:
            raise AssertionError("Previous rolling-source storage identity changed during inference")


def get_components(repo_root, siglip_path, checkpoint):
    actual = sha256(checkpoint)
    if actual != V2_SHA256: raise RuntimeError(f"V2 checkpoint SHA mismatch: {actual}")
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import load_siglip_text_tower
    model, processor, device = load_model_and_processor(DEFAULT_MODEL_ID)
    if device.type != "cuda": raise RuntimeError(f"CUDA unavailable: {device}")
    if getattr(model.language_model.config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError("Adaptive method requires the validated SDPA backend")
    key = resolve_unnorm_key(model, None)
    text_model, tokenizer, text_meta = load_siglip_text_tower(siglip_path, device=str(device))
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    projection = RegionTextProjection().to(device).eval(); projection.load_state_dict(raw["model"], strict=True)
    for mod in (model, text_model, projection): mod.eval()
    for mod in (text_model, projection):
        for p in mod.parameters(): p.requires_grad_(False)
    model_revision = getattr(model.config, "_commit_hash", None)
    if not model_revision:
        raise RuntimeError("Cannot establish the loaded OpenVLA Hugging Face revision from config._commit_hash")
    dino = getattr(model.vision_backbone, "dino_featurizer",
                   getattr(model.vision_backbone, "featurizer", None))
    model_meta = {"repository": DEFAULT_MODEL_ID, "revision": model_revision,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "attention_backend": model.language_model.config._attn_implementation,
        "policy_checkpoint_config_class": type(model.config).__name__,
        "native_dino_module_class": type(dino).__name__ if dino is not None else None,
        "native_dino_expected_features": [1, 256, 1024],
        "fused_visual_features_reference_shape": [1, 256, 2176],
        "fused_visual_features_source": "previous E3 live OpenVLA integration profile; default forward unchanged",
        "image_transform": "existing OpenVLA processor and policy_image_from_observation; no preprocessing changes"}
    return model, processor, device, key, text_model, tokenizer, projection, text_meta, model_meta


def run_episode(task_id, state_id, model, processor, device, key, text_model, tokenizer, projection,
                out_dir: Path, smoke_steps: int | None = None, *, fg_threshold=FG_THRESHOLD,
                confidence_threshold=CONFIDENCE_THRESHOLD, force_full_refresh_steps=(),
                validate_fresh_parity=False):
    seed_everything(SEED)
    suite, task, states, env, description = make_env(task_id)
    prompt = f"In: What action should the robot take to {description.lower()}?\nOut:"
    phrases, text_vectors = task_text(description, text_model, tokenizer, projection, device)
    records = []; rolling = None; layout = None; last_attn = None; previous_fg = None; previous_margin = None
    ages = TokenAgeTracker(); last_feature = None; done = False
    try:
        partial_path = out_dir / "partial" / f"task{task_id}_state{state_id}.json"
        if partial_path.exists():
            archive = out_dir / "interrupted_partials"
            archive.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            shutil.move(str(partial_path), str(archive / f"task{task_id}_state{state_id}_{stamp}.json"))
        env.reset(); obs = env.set_init_state(states[state_id])
        from experiments.kv_analysis.experiments.extract_libero_image import DEFAULT_STEP_ACTION
        for _ in range(10):
            obs, _, done, _ = env.step(DEFAULT_STEP_ACTION)
            if done: raise RuntimeError("Environment ended during settling")
        horizon = int(smoke_steps or HORIZON)
        for step in range(horizon):
            selection_attention_source_step = step - 1 if last_attn is not None else None
            attention_used_for_selection = last_attn.copy() if last_attn is not None else None
            trigger = confidence_requires_refresh(previous_margin, confidence_threshold)
            if rolling is None: refresh_reason = "initial_full_refresh"
            elif step in force_full_refresh_steps: refresh_reason = "validation_forced_full_refresh"
            elif trigger: refresh_reason = f"previous_action_margin_below_{confidence_threshold:g}"
            else: refresh_reason = "selective_rolling_reuse"
            refresh = refresh_reason != "selective_rolling_reuse"
            if not refresh and (last_attn is None or previous_fg is None or layout is None):
                refresh = True; refresh_reason = "missing_selector_state_fail_safe_refresh"
            image = policy_image_from_observation(obs, True)
            selected = {}; selection_times = {}; manager = None; sidecar = None
            if refresh:
                sidecar = RefreshAttentionSidecar(model, layout_hint=layout)
                tap = DINOFeatureTap(model)
                intervention = sidecar
            else:
                manager = AdaptiveRollingKVSkip(model, rolling, layout)
                def select_during_vision(feat):
                    probs, times = v2_probabilities(feat, text_vectors)
                    t = time.perf_counter()
                    choice = compose_adaptive_mask(probs, last_attn, previous_fg,
                        probability_threshold=fg_threshold,
                        attention_background_fraction=ATTENTION_FRAC)
                    compose_ms = (time.perf_counter() - t) * 1000
                    selected["choice"] = choice; selected["probabilities"] = probs
                    selection_times.update(times); selection_times["mask_union_transition_union_ms"] = compose_ms
                    selection_times["complete_v2_selection_ms"] = times["v2_scoring_cpu_transfer_wall_ms"] + compose_ms
                    manager.set_mask(choice["fresh"])
                tap = DINOFeatureTap(model, select_during_vision); intervention = manager
            source_versions = cache_identity_signature(rolling) if rolling is not None else None
            with tap:
                pred, call_ms, row_monitor = _policy_call(model, processor, image, prompt, device, key,
                    intervention, capture_action_margins=True)
            if tap.calls != 1 or tap.features is None: raise RuntimeError("Native DINO forward was not captured exactly once")
            if tuple(tap.features.shape) != (1, 256, 1024): raise RuntimeError(f"Wrong DINO sidecar tensor {tuple(tap.features.shape)}")
            if list(pred["layout"]["visual"]) != [1, 256]: raise RuntimeError(f"Wrong LLM visual positions {pred['layout']}")
            if row_monitor is None: raise RuntimeError("Projection-row monitor is missing")
            if refresh:
                if sidecar.scores is None or tuple(sidecar.scores.shape) != (256,) or not np.isfinite(sidecar.scores).all():
                    raise RuntimeError("Fresh layer-15 attention sidecar returned invalid scores")
                if sidecar.backend != "sdpa" or sidecar.sequence_length != pred["prefill_length"]:
                    raise RuntimeError("Fresh attention sidecar backend/layout mismatch")
                probs, selection_times = v2_probabilities(tap.features, text_vectors)
                current_fg = probs.max(axis=0) >= fg_threshold
                if last_attn is not None and previous_fg is not None:
                    compose_start = time.perf_counter()
                    choice = compose_adaptive_mask(probs, last_attn, previous_fg,
                        probability_threshold=fg_threshold,
                        attention_background_fraction=ATTENTION_FRAC)
                    selection_times["mask_union_transition_union_ms"] = (time.perf_counter() - compose_start) * 1000
                    selection_times["complete_v2_selection_ms"] = (
                        selection_times["v2_scoring_cpu_transfer_wall_ms"] + selection_times["mask_union_transition_union_ms"])
                else:
                    choice = {"foreground": current_fg, "attention_protected": np.zeros(256, bool),
                              "transition_protected": np.zeros(256, bool), "fresh": current_fg.copy(),
                              "reused": ~current_fg, "counts": {"foreground": int(current_fg.sum()),
                              "attention_protected": 0, "transition_protected": 0,
                              "attention_transition_overlap": 0, "fresh_union": int(current_fg.sum()),
                              "reused": int((~current_fg).sum())}}
                    selection_times["mask_union_transition_union_ms"] = 0.0
                    selection_times["complete_v2_selection_ms"] = selection_times["v2_scoring_cpu_transfer_wall_ms"]
                # Confidence-gated refresh recomputes every visual row regardless of the selector union.
                choice["adaptive_candidate_fresh"] = choice["fresh"].copy()
                choice["adaptive_candidate_reused"] = choice["reused"].copy()
                choice["fresh"] = np.ones(256, dtype=bool); choice["reused"] = np.zeros(256, dtype=bool)
                choice["counts"]["fresh_union"] = 256; choice["counts"]["reused"] = 0
                ages_before = ages.ages(step) if ages.last_recomputed_step is not None else np.zeros(256, dtype=np.int64)
                row_monitor.validate(prefill_length=pred["prefill_length"], projected_kv_rows=pred["prefill_length"])
                if step == 0 and validate_fresh_parity:
                    # Additional full-inference check is done once using same exact image/prompt.
                    base, _, _ = _policy_call(model, processor, image, prompt, device, key,
                                              monitor=False, capture_action_margins=True)
                    if base["action_token_ids"] != pred["action_token_ids"] or not torch.equal(base["logits"], pred["logits"]):
                        raise AssertionError("Sidecar instrumentation changed fresh policy tokens/logits")
                new_layout = dict(pred["layout"])
                cache_shape = [(list(k.shape), list(v.shape)) for k, v in cache_layers(pred["cache"])]
                layout = new_layout
                last_attn = sidecar.scores.copy()
                selection_times = dict(selection_times)
                selection_times.update({"attention_capture_ms": sidecar.capture_ms or 0.0,
                                        "attention_capture_peak_delta_bytes": sidecar.peak_memory_delta_bytes or 0})
                attention_row_error = sidecar.row_sum_max_error
                cache_update_start = time.perf_counter()
                ages.commit_successful_prefill(step, np.ones(256, bool), full_refresh=True)
                rolling = pred["cache"]
                cache_update_ms = (time.perf_counter() - cache_update_start) * 1000.0
                if not cache_identity_signature(rolling): raise AssertionError("No rolling cache storage snapshot")
                ages_after = np.zeros(256, dtype=np.int64)
            else:
                if "choice" not in selected: raise AssertionError("Adaptive selector did not run before the Llama prefill")
                choice = selected["choice"]; probs = selected["probabilities"]
                manager.validate(pred["cache"])
                row_monitor.validate(prefill_length=pred["prefill_length"],
                    projected_kv_rows=1 + int(choice["fresh"].sum()) + (layout["language"][1]-layout["language"][0]+1))
                ages_before = ages.ages(step)
                cache_update_start = time.perf_counter()
                ages_after = ages.commit_successful_prefill(step, choice["fresh"])
                last_attn = manager.scores.copy()
                attention_row_error = manager.row_sum_max_error
                selection_times.update({"attention_capture_ms": manager.capture_ms or 0.0,
                    "attention_capture_peak_delta_bytes": 0})
                cache_shape = [(list(k.shape), list(v.shape)) for k, v in cache_layers(pred["cache"])]
                rolling = pred["cache"]  # New independent PrefillSnapshot clone with row-wise latest provenance.
                cache_update_ms = (time.perf_counter() - cache_update_start) * 1000.0
                if cache_identity_signature(rolling) == source_versions:
                    raise AssertionError("Rolling cache was not replaced after successful prefill")
            if not np.isfinite(pred["policy_action"]).all(): raise RuntimeError("Nonfinite action")
            if pred["action_confidence_margin"] is None or not np.isfinite(pred["action_confidence_margin"]):
                raise RuntimeError("Action confidence margin missing or nonfinite")
            if choice["fresh"].shape != (256,) or choice["reused"].shape != (256,):
                raise AssertionError("Hybrid selection does not align to 256 visual positions")
            if not np.all(choice["fresh"] | choice["reused"]) or np.any(choice["fresh"] & choice["reused"]):
                raise AssertionError("Refresh/reuse masks are not a partition")
            if last_feature is None:
                feature_delta = np.zeros(256, dtype=np.float32)
            else:
                cur = F.normalize(tap.features.float(), dim=-1)[0].detach().cpu().numpy()
                feature_delta = 1.0 - np.sum(cur * last_feature, axis=-1)
            current_features = F.normalize(tap.features.float(), dim=-1)[0].detach().cpu().numpy()
            # Visual-feature change against each token's last K/V recomputation source is diagnostic only.
            last_feature_next = current_features.copy() if last_feature is None else last_feature.copy()
            if refresh: last_feature_next[:] = current_features
            else: last_feature_next[choice["fresh"]] = current_features[choice["fresh"]]
            last_feature = last_feature_next
            if step in (0, 1, 2) or refresh_reason != "selective_rolling_reuse":
                save_adaptive_visual(out_dir / "mask_visualizations" / f"task{task_id}_state{state_id}_step{step:03d}.png",
                    image, choice, ages_after, previous_margin, refresh_reason)
            skipped_per_layer = 0 if refresh else int(choice["reused"].sum())
            record = {
                "step": step, "task_id": task_id, "state_id": state_id,
                "refresh": bool(refresh), "refresh_reason": refresh_reason,
                "source_policy": "per-token rolling latest-computed K/V; no rollback to older refresh",
                "previous_action_margin": previous_margin,
                "action_confidence_margin": pred["action_confidence_margin"],
                "action_token_probability_margins": pred["action_probability_margins"],
                "confidence_threshold": confidence_threshold,
                "confidence_compute_ms": pred["confidence_compute_ms"],
                "action_token_ids": pred["action_token_ids"],
                "policy_action": pred["policy_action"].tolist(),
                "prefill_logits_shape": list(pred["logits"].shape),
                "visual_positions": pred["layout"]["visual"], "prefill_length": pred["prefill_length"],
                "dino_shape": list(tap.features.shape), "prompt_phrases": phrases,
                "selection_attention_source_step": selection_attention_source_step,
                "score_captured_step": step,
                "score_age_decisions_at_selection": 1 if attention_used_for_selection is not None else None,
                "attention_scores_used_for_selection": attention_used_for_selection.tolist() if attention_used_for_selection is not None else None,
                "attention_scores_captured_current_step": last_attn.tolist(),
                "attention_capture_ms": selection_times.get("attention_capture_ms", 0.0),
                "attention_row_normalization_error": attention_row_error,
                "v2_probability_min": float(probs.min()), "v2_probability_max": float(probs.max()),
                "v2_fg_count": choice["counts"]["foreground"],
                "adaptive_candidate_fresh_count": int(choice.get("adaptive_candidate_fresh", choice["fresh"]).sum()),
                "adaptive_candidate_reused_count": int(choice.get("adaptive_candidate_reused", choice["reused"]).sum()),
                "attention_protection_count": choice["counts"]["attention_protected"],
                "transition_protection_count": choice["counts"]["transition_protected"],
                "attention_transition_overlap": choice["counts"]["attention_transition_overlap"],
                "fresh_union_count": int(choice["fresh"].sum()), "reused_candidate_count": int(choice["reused"].sum()),
                "actual_reused_tokens": 0 if refresh else int(choice["reused"].sum()),
                "actual_reuse_coverage": 0.0 if refresh else float(choice["reused"].sum()/256),
                "k_skipped_rows_per_layer": skipped_per_layer,
                "v_skipped_rows_per_layer": skipped_per_layer,
                "total_k_skipped_rows": 32 * skipped_per_layer,
                "total_v_skipped_rows": 32 * skipped_per_layer,
                "token_age_before": ages_before.tolist(), "token_age_after": ages_after.tolist(),
                "token_age_max_after": int(ages_after.max()),
                "token_last_recomputed_step": ages.last_recomputed_step.tolist(),
                "feature_change_from_last_recomputation_mean": float(feature_delta.mean()),
                "feature_change_reused_mean": float(feature_delta[choice["reused"]].mean()) if choice["reused"].any() else 0.0,
                "rolling_cache_update_ms": cache_update_ms, "cache_layer_shapes": cache_shape,
                "policy_call_ms": call_ms,
                # Include the full per-frame DINO-score/host-transfer/mask-selection
                # cost on selective decisions as well as full refresh decisions.
                "end_to_end_decision_ms": (call_ms + selection_times.get("complete_v2_selection_ms", 0.0)
                                           + cache_update_ms),
                "projection_rows_by_layer": row_monitor.per_layer_summary(),
                "cuda_allocated_bytes_after_call": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes_after_call": int(torch.cuda.memory_reserved(device)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                **selection_times,
                "all_layers_projected_or_skipped": True,
                "rolling_source_immutable_during_inference": True,
                "roPE_original_position_semantics": True,
                "normal_sdpa_backend": "sdpa",
            }
            records.append(record)
            atomic_json(out_dir / "partial" / f"task{task_id}_state{state_id}.json",
                {"status": "running", "task_id": task_id, "state_id": state_id,
                 "steps_completed": len(records), "records": records})
            previous_fg = choice["foreground"].copy()
            previous_margin = float(pred["action_confidence_margin"])
            obs, _, done, _ = env.step(_action_for_env(pred).tolist())
            if done: break
        return {"task_id": task_id, "state_id": state_id, "success": bool(done),
                "steps": len(records), "records": records}
    finally:
        env.close()


def summarize(out_dir, config, episodes):
    rows = [r for ep in episodes for r in ep["records"]]
    decisions = len(rows); reuse = [r for r in rows if not r["refresh"]]
    refreshes = [r for r in rows if r["refresh"]]
    sum_k = sum(r["total_k_skipped_rows"] for r in rows); sum_v = sum(r["total_v_skipped_rows"] for r in rows)
    all_ages = np.asarray([x for r in rows for x in r["token_age_after"]], dtype=np.int64)
    confidence = np.asarray([r["action_confidence_margin"] for r in rows], dtype=float)
    aggregate = {
        "status": "complete", "scenario_count": len(episodes), "successes": sum(x["success"] for x in episodes),
        "total_policy_decisions": decisions,
        "actual_reused_tokens_per_decision_including_refreshes": float(np.mean([r["actual_reused_tokens"] for r in rows])),
        "actual_reuse_coverage_including_full_refreshes": float(np.mean([r["actual_reuse_coverage"] for r in rows])),
        "mean_reuse_coverage_on_selective_decisions": float(np.mean([r["actual_reuse_coverage"] for r in reuse])) if reuse else 0,
        "mean_skipped_k_rows_per_decision": sum_k / decisions,
        "mean_skipped_v_rows_per_decision": sum_v / decisions,
        "total_k_projection_rows_skipped": sum_k, "total_v_projection_rows_skipped": sum_v,
        "projection_rows_skipped_per_layer": [
            {"layer": li,
             "k_rows_skipped": sum(max(0, int(r["prefill_length"]) - int(r["projection_rows_by_layer"][li]["k_prefill_rows"])) for r in rows),
             "v_rows_skipped": sum(max(0, int(r["prefill_length"]) - int(r["projection_rows_by_layer"][li]["v_prefill_rows"])) for r in rows),
             "q_rows_skipped": 0}
            for li in range(32)],
        "refresh_count": len(refreshes), "refresh_frequency": len(refreshes) / decisions,
        "refresh_reasons": {reason: sum(r["refresh_reason"] == reason for r in rows) for reason in sorted({r["refresh_reason"] for r in rows})},
        "action_confidence": vector_stats(confidence),
        "confidence_threshold": config["confidence_threshold"],
        "token_age_after_decision": vector_stats(all_ages),
        "max_token_age": int(all_ages.max()),
        "attention_protection_mean_on_selective_steps": float(np.mean([r["attention_protection_count"] for r in reuse])) if reuse else 0,
        "transition_protection_mean_on_selective_steps": float(np.mean([r["transition_protection_count"] for r in reuse])) if reuse else 0,
        "scenario_summary": [{"task_id": e["task_id"], "state_id": e["state_id"], "success": e["success"], "steps": e["steps"]} for e in episodes],
        "configuration": config,
    }
    timings = {
        "selection_complete_ms": vector_stats(np.asarray([r.get("complete_v2_selection_ms", 0) for r in rows if "complete_v2_selection_ms" in r])),
        "attention_refresh_ms": vector_stats(np.asarray([r["attention_capture_ms"] for r in refreshes])),
        "attention_selective_ms": vector_stats(np.asarray([r["attention_capture_ms"] for r in reuse])),
        "attention_amortized_per_decision_ms": float(np.mean([r["attention_capture_ms"] for r in rows])),
        "confidence_compute_ms": vector_stats(np.asarray([r["confidence_compute_ms"] for r in rows])),
        "rolling_cache_update_ms": vector_stats(np.asarray([r["rolling_cache_update_ms"] for r in rows])),
        "policy_refresh_ms": vector_stats(np.asarray([r["policy_call_ms"] for r in refreshes])),
        "policy_selective_ms": vector_stats(np.asarray([r["policy_call_ms"] for r in reuse])),
        "end_to_end_refresh_decision_ms": vector_stats(np.asarray([r["end_to_end_decision_ms"] for r in refreshes])),
        "end_to_end_selective_decision_ms": vector_stats(np.asarray([r["end_to_end_decision_ms"] for r in reuse])),
        "cuda_allocated_bytes": vector_stats(np.asarray([r["cuda_allocated_bytes_after_call"] for r in rows])),
        "cuda_reserved_bytes": vector_stats(np.asarray([r["cuda_reserved_bytes_after_call"] for r in rows])),
        "cuda_peak_allocated_bytes": int(max(r["cuda_peak_allocated_bytes"] for r in rows)),
    }
    aggregate["timing"] = timings
    atomic_json(out_dir / "summary.json", aggregate)
    with (out_dir / "per_scenario_metrics.csv").open("w", newline="") as f:
        fields = ["task_id", "state_id", "success", "steps", "refresh_count", "selective_count",
                  "reuse_coverage_all_decisions", "reuse_coverage_selective_decisions", "k_rows_skipped", "v_rows_skipped",
                  "refresh_ms_mean", "selective_ms_mean", "end_to_end_refresh_mean", "end_to_end_refresh_p50",
                  "end_to_end_refresh_p95", "end_to_end_selective_mean", "end_to_end_selective_p50",
                  "end_to_end_selective_p95", "refreshes_low_confidence", "refreshes_other"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for ep in episodes:
            rr = ep["records"]; sl = [r for r in rr if not r["refresh"]]; rf = [r for r in rr if r["refresh"]]
            w.writerow({"task_id": ep["task_id"], "state_id": ep["state_id"], "success": ep["success"], "steps": ep["steps"],
                "refresh_count": len(rf), "selective_count": len(sl),
                "reuse_coverage_all_decisions": np.mean([r["actual_reuse_coverage"] for r in rr]),
                "reuse_coverage_selective_decisions": np.mean([r["actual_reuse_coverage"] for r in sl]) if sl else 0,
                "k_rows_skipped": sum(r["total_k_skipped_rows"] for r in rr),
                "v_rows_skipped": sum(r["total_v_skipped_rows"] for r in rr),
                "refresh_ms_mean": np.mean([r["policy_call_ms"] for r in rf]),
                "selective_ms_mean": np.mean([r["policy_call_ms"] for r in sl]) if sl else "",
                "end_to_end_refresh_mean": np.mean([r["end_to_end_decision_ms"] for r in rf]),
                "end_to_end_refresh_p50": np.percentile([r["end_to_end_decision_ms"] for r in rf], 50),
                "end_to_end_refresh_p95": np.percentile([r["end_to_end_decision_ms"] for r in rf], 95),
                "end_to_end_selective_mean": np.mean([r["end_to_end_decision_ms"] for r in sl]) if sl else "",
                "end_to_end_selective_p50": np.percentile([r["end_to_end_decision_ms"] for r in sl], 50) if sl else "",
                "end_to_end_selective_p95": np.percentile([r["end_to_end_decision_ms"] for r in sl], 95) if sl else "",
                "refreshes_low_confidence": sum(r["refresh_reason"].startswith("previous_action_margin_below_") for r in rf),
                "refreshes_other": sum(r["refresh_reason"] != "previous_action_margin_below_0.65" for r in rf)})
    report = ["# Adaptive Hybrid KV Cache — V2 + Task Relevance", "",
        f"Completed scenarios: **{aggregate['successes']}/{len(episodes)}**. This evaluates the combined policy; it does not isolate the effects of confidence gating, transition protection, or rolling storage.", "",
        "## Runtime model identity", "",
        f"Model: `{config.get('model_runtime', {}).get('repository', DEFAULT_MODEL_ID)}` at revision `{config.get('model_runtime', {}).get('revision', 'not recorded')}`; Transformers `{config.get('model_runtime', {}).get('transformers_version', 'not recorded')}`, PyTorch `{config.get('model_runtime', {}).get('torch_version', 'not recorded')}`, attention backend `{config.get('model_runtime', {}).get('attention_backend', 'not recorded')}`. Native DINO capture is `[1,256,1024]`; fused `[1,256,2176]` is cross-checked against the prior E3 live profile. V2 projection SHA-256 `{config['v2_checkpoint_sha256']}`. The auxiliary SigLIP tower identity and context are in `configuration.json`.", "",
        "## Policy and action-confidence definition", "",
        "Every observation uses passive native penultimate OpenVLA DINO features to compute `sigmoid(10 × (cosine − 0.20))`; foreground is probability ≥ 0.20. During autoregressive decoding, for each of the seven generated action tokens, probability distributions are softmax-normalized over the full model vocabulary, matching greedy GenerationMixin token selection. The confidence is the mean top-1 minus top-2 probability margin. Step t uses step t−1's confidence to decide full refresh; margin < 0.65 triggers a full refresh. This matches the Gated VLA-Cache OpenVLA definition, so the provisional threshold was used as-is without calibration on these nine scenarios.", "",
        "Layer-15 relevance is captured during each prefill. On selective steps it is computed from the same full Q and assembled post-RoPE K tensor used by SDPA, including both retained rolling rows and newly projected rows; skipped K projections are not recomputed. That attention only controls the next decision. At t, top 10% (nearest integer, half up) of current predicted background are attention-protected using scores from t−1. Foreground-to-background transitions are also refreshed. All K/V source entries update per token after successful prefill; no periodic refresh or maximum age is imposed.", "",
        "## Scenario outcomes", "",
        "| Task | State | Adaptive | Fresh historical | Stage 3C oracle historical | V2 hybrid R5 historical |", "|---:|---:|---|---|---|",
    ]
    refs = {"4:0": ("success / 140", "success / 129", "success / 149"),
            "4:1": ("success / 145", "success / 143", "failure / 220"),
            "4:2": ("success / 130", "success / 131", "success / 135"),
            "5:0": ("success / 105", "success / 111", "success / 110"),
            "5:1": ("success / 89", "success / 100", "success / 96"),
            "5:2": ("success / 90", "success / 88", "success / 183"),
            "7:0": ("success / 111", "success / 152", "failure / 220"),
            "7:1": ("success / 120", "success / 124", "success / 118"),
            "7:2": ("failure / 220", "success / 126", "success / 123")}
    for e in episodes:
        key = f"{e['task_id']}:{e['state_id']}"; hist = refs[key]
        report.append(f"| {e['task_id']} | {e['state_id']} | {'success' if e['success'] else 'failure'} / {e['steps']} | {hist[0]} | {hist[1]} | {hist[2]} |")
    report += ["", "Historical trajectories are not paired: different policies change subsequent states. The nine-scenario matrix is small and does not establish generalization.", "",
        "## Cache use and confidence gate", "",
        f"Across {decisions} decisions including full refreshes, realized reuse was {aggregate['actual_reuse_coverage_including_full_refreshes']*100:.1f}% ({aggregate['actual_reused_tokens_per_decision_including_refreshes']:.1f}/256 tokens per decision). Selective steps reused {aggregate['mean_reuse_coverage_on_selective_decisions']*100:.1f}% on average. Refreshes: {len(refreshes)}/{decisions} ({aggregate['refresh_frequency']*100:.1f}%). Mean skipped projection rows per decision: K {aggregate['mean_skipped_k_rows_per_decision']:.0f}, V {aggregate['mean_skipped_v_rows_per_decision']:.0f}. Total omitted rows across all layers: K {sum_k:,}, V {sum_v:,}.", "",
        f"Refresh reasons: `{json.dumps(aggregate['refresh_reasons'], sort_keys=True)}`. Action confidence margin p50/p95: {np.percentile(confidence,50):.4f}/{np.percentile(confidence,95):.4f}; threshold .65. Maximum visual-token age was {aggregate['max_token_age']} decisions (there is intentionally no age cap). Mean attention-protected tokens on selective steps: {aggregate['attention_protection_mean_on_selective_steps']:.1f}; mean transition-protected tokens: {aggregate['transition_protection_mean_on_selective_steps']:.1f}.", "",
        "## Runtime and interpretation", "",
        f"V2 selection mean: {timings['selection_complete_ms']['mean'] or 0:.3f} ms. Attention capture mean: full refresh {timings['attention_refresh_ms']['mean'] or 0:.3f} ms; selective step {timings['attention_selective_ms']['mean'] or 0:.3f} ms; amortized {timings['attention_amortized_per_decision_ms']:.3f} ms/decision. Confidence margin calculation mean: {timings['confidence_compute_ms']['mean'] or 0:.3f} ms. Rolling cache update mean: {timings['rolling_cache_update_ms']['mean'] or 0:.3f} ms. Synchronized policy-call latency: refresh mean {timings['policy_refresh_ms']['mean'] or 0:.2f} ms, selective mean {timings['policy_selective_ms']['mean'] or 0:.2f} ms.", "",
        "Skipped-row counts are structural and do not demonstrate acceleration. End-to-end decision latency includes V2 scoring, device-to-host transfer and mask composition on both refresh and selective decisions; attention extraction is inside the synchronized policy-call interval and is separately reported as a component, not double-counted. Policy-call latency includes the already-existing greedy decode verification pass. Historical timing is not directly matched; no net speedup is claimed. Selective K/V provenance tracks last recomputation but deeper-layer values still depend on the reused contextual hidden states that produced them; it is not equivalent to fully fresh inference.", "",
        "## Threshold follow-up protocol", "",
        "For a later, separate threshold study (not run here), freeze development tasks 0–3, 6, 8, 9 with states 0–2 and hold out tasks 4, 5, 7 with states 0–2. Compare τFG ∈ {0.10, 0.15, 0.20} only on development, rank by task success, completion efficiency, actual reuse, then measured inference cost, and freeze the choice before a single held-out evaluation. The current nine comparison scenarios were not used to tune this run.", "",
        "## Artifacts", "", "- `summary.json` and `per_scenario_metrics.csv` contain aggregate/scenario outcomes.",
        "- `scenario_results/` contains per-step confidence, refresh reason, token ages, K/V row counts, and attention.",
        "- `mask_visualizations/` shows semantic, attention, transition, union, reuse, and age maps.",
        "- `configuration.json`, `validation.json`, and `RUN_STATUS.json` preserve run identity and validation.", ""]
    (out_dir / "ADAPTIVE_HYBRID_KV_REPORT.md").write_text("\n".join(report))
    return aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("validation", "run-all"), required=True)
    ap.add_argument("--output-dir", type=Path, default=BASE_OUT / "primary_20261125")
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--siglip-checkpoint", type=Path, default=SIGLIP)
    ap.add_argument("--v2-checkpoint", type=Path, default=V2_CKPT)
    ap.add_argument("--foreground-threshold", type=float, default=FG_THRESHOLD)
    ap.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = ap.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    cfg = {"name": "adaptive hybrid rolling V2 + task relevance", "branch": "thesis/trackb-dense-grounding-v2",
        "v2_checkpoint_sha256": V2_SHA256, "v2_probability_threshold": FG_THRESHOLD,
        "sigmoid_cosine_scale": COSINE_SCALE, "sigmoid_cosine_offset": COSINE_OFFSET,
        "attention_fraction_of_current_background": ATTENTION_FRAC, "attention_layer": 15,
        "foreground_probability_threshold": args.foreground_threshold,
        "attention_source_lag": 1, "transition_rule": "FG_(t-1) AND NOT FG_t",
        "rolling_token_cache": True, "periodic_refresh": False, "maximum_age": None,
        "confidence_metric": "mean per-action-token (top1-top2) full-vocabulary softmax probability margin",
        "confidence_threshold": args.confidence_threshold, "confidence_trigger_lag": 1,
        "checkpoint": str(args.v2_checkpoint), "tasks": list(TASKS), "states": list(STATES),
        "seed": SEED, "horizon": HORIZON, "mode": args.mode}
    cfg["configuration_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    atomic_json(out / "configuration.json", cfg)
    status = {"status": "starting", "phase": "preflight", "completed": [], "configuration_sha256": cfg["configuration_sha256"]}
    atomic_json(out / "RUN_STATUS.json", status)
    try:
        model, processor, device, key, text_model, tokenizer, projection, text_meta, model_meta = get_components(args.repo_root,args.siglip_checkpoint,args.v2_checkpoint)
        cfg["model_runtime"] = model_meta
        cfg["siglip_runtime"] = text_meta
        cfg["configuration_sha256"] = hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k!="configuration_sha256"}, sort_keys=True).encode()).hexdigest()
        atomic_json(out / "configuration.json", cfg)
        status.update(status="running", phase="model_loaded", device=str(device), gpu=torch.cuda.get_device_name(device),
                      text_encoder=text_meta, model_runtime=model_meta, configuration_sha256=cfg["configuration_sha256"])
        atomic_json(out / "RUN_STATUS.json", status)
        if args.mode == "validation":
            # One no-trigger trace demonstrates rolling ages beyond five; a
            # second short trace demonstrates a live confidence-trigger reset.
            rolling_dir = out / "validation_rolling_age"
            roll = run_episode(4, 0, model, processor, device, key, text_model, tokenizer, projection,
                rolling_dir, smoke_steps=8, fg_threshold=args.foreground_threshold,
                confidence_threshold=0.0, force_full_refresh_steps=(6,), validate_fresh_parity=True)
            roll_record = roll["records"]
            if len(roll_record) != 8 or not all(not r["refresh"] for r in roll_record[1:6]):
                raise AssertionError("No-trigger validation failed to demonstrate rolling reuse through age >4")
            if roll_record[5]["token_age_max_after"] <= 4:
                raise AssertionError("Rolling age validation did not exceed four decisions")
            if roll_record[6]["refresh_reason"] != "validation_forced_full_refresh" or roll_record[6]["token_age_max_after"] != 0:
                raise AssertionError("Forced full-refresh validation did not reset all token ages")
            gate_dir = out / "validation_confidence_gate"
            gate = run_episode(4, 0, model, processor, device, key, text_model, tokenizer, projection,
                gate_dir, smoke_steps=2, fg_threshold=args.foreground_threshold,
                confidence_threshold=0.99, validate_fresh_parity=False)
            if len(gate["records"]) != 2 or not gate["records"][1]["refresh_reason"].startswith("previous_action_margin_below_"):
                raise AssertionError("Live confidence-triggered full-refresh test did not fire")
            validation = {"status": "passed", "exact_fresh_action_and_logits_parity": True,
                "native_dino_one_pass_per_observation": True, "visual_layout_1_to_256_verified": True,
                "selective_steps_through_age_5": 5, "token_age_greater_than_four": True,
                "explicit_full_refresh_reset_at_step_6": True,
                "live_confidence_trigger_on_step_1": True,
                "full_q_and_layerwise_kv_skip": "checked at every selective forward across 32 layers",
                "rolling_retained_rows_equal_latest_cache": True,
                "source_cache_immutable_during_each_prefill": True,
                "r5_periodic_schedule_used": False,
                "confidence_gate_validation_threshold": 0.99,
                "production_confidence_threshold": args.confidence_threshold,
                "rolling_trace": str(rolling_dir / "partial/task4_state0.json"),
                "confidence_trace": str(gate_dir / "partial/task4_state0.json")}
            atomic_json(out / "validation.json", validation)
            status.update(status="validation_passed",phase="validation_passed",validation=validation)
            atomic_json(out / "RUN_STATUS.json", status)
            return
        scenarios = [(t,s) for t in TASKS for s in STATES]
        episodes = []
        for task_id,state_id in scenarios:
            result_path=out/"scenario_results"/f"task{task_id}_state{state_id}.json"
            if result_path.is_file():
                existing=json.loads(result_path.read_text())
                valid=(existing.get("configuration_sha256")==cfg["configuration_sha256"]
                       and existing.get("task_id")==task_id and existing.get("state_id")==state_id
                       and isinstance(existing.get("records"),list)
                       and existing.get("steps")==len(existing["records"]))
                if not valid:
                    raise RuntimeError(f"Refusing to overwrite scenario result with different or invalid provenance: {result_path}")
                episodes.append(existing)
                if not any(x.get("task_id")==task_id and x.get("state_id")==state_id for x in status["completed"]):
                    status["completed"].append({"task_id":task_id,"state_id":state_id,
                        "success":existing["success"],"steps":existing["steps"],"resumed":True})
                atomic_json(out/"RUN_STATUS.json",status)
                continue
            status["current_scenario"]={"task_id":task_id,"state_id":state_id}; status["phase"]="running"
            atomic_json(out/"RUN_STATUS.json",status)
            result=run_episode(task_id,state_id,model,processor,device,key,text_model,tokenizer,projection,out,
                               fg_threshold=args.foreground_threshold,
                               confidence_threshold=args.confidence_threshold)
            result["configuration_sha256"]=cfg["configuration_sha256"]
            atomic_json(result_path,result); episodes.append(result)
            status["completed"].append({"task_id":task_id,"state_id":state_id,"success":result["success"],"steps":result["steps"]})
            status.pop("current_scenario",None); atomic_json(out/"RUN_STATUS.json",status)
        if args.mode == "validation":
            status.update(status="validation_passed",phase="validation_passed",validation_scenarios=len(episodes))
            atomic_json(out/"RUN_STATUS.json",status)
        else:
            aggregate=summarize(out,cfg,episodes)
            status.update(status="complete",phase="complete",finished_unix=time.time(),summary=aggregate)
            atomic_json(out/"RUN_STATUS.json",status)
    except Exception as exc:
        status.update(status="failed",phase="failed",error=repr(exc),traceback=traceback.format_exc(),failed_unix=time.time())
        atomic_json(out/"RUN_STATUS.json",status); raise


if __name__ == "__main__": main()
