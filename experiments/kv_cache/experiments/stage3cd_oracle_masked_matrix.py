"""Stage 3C/3D oracle-masked visual K/V projection-skip factorial cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

from experiments.kv_analysis.experiments.extract_libero_image import (
    DEFAULT_STEP_ACTION, TASK_SUITE_NAME, normalize_and_invert_openvla_gripper,
)
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import (
    assert_independent_cache_clone, capture_prediction, policy_image_from_observation, seed_everything,
)
from experiments.kv_cache.experiments.oracle_foreground_mask import rendered_label_mapping
from experiments.kv_cache.experiments.stage3a_masked_visual_k_validation import mask_from_obs, seg_env
from experiments.kv_cache.experiments.static_reuse_experiment import (
    DEFAULT_MODEL_ID, cache_layers, clone_prefill_cache, load_model_and_processor, resolve_unnorm_key,
)
from experiments.kv_cache.experiments.stage3b_oracle_full_kv_matrix import make_env_and_task

ROOT = Path(__file__).resolve().parents[1]
STAGE3B = ROOT / "outputs" / "stage3b_oracle_full_kv_matrix"
R, HORIZON, SEED = 5, 220, 7
ALL_LAYERS = tuple(range(32))
VARIANTS = {
    "stage3c": {"name": "stage3c_oracle_k_all_layers", "layers": ALL_LAYERS, "components": ("K",)},
    "stage3d": {"name": "stage3d_oracle_kv_first14", "layers": tuple(range(14)), "components": ("K", "V")},
}
NUMERICAL_LOGIT_REL_L2_TOLERANCE = 0.02


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _indices(foreground: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    fg = torch.as_tensor(np.flatnonzero(foreground) + 1, dtype=torch.long, device=device)
    bg = torch.as_tensor(np.flatnonzero(~foreground) + 1, dtype=torch.long, device=device)
    return fg, bg


class MaskedOverwrite:
    """Reference path: full projections, followed by stale-background insertion."""
    def __init__(self, source: DynamicCache, layout: dict[str, Any], foreground: np.ndarray,
                 layers: tuple[int, ...], components: tuple[str, ...]):
        self.source, self.layout, self.fg = source, layout, np.asarray(foreground, dtype=bool)
        self.layers, self.components = layers, components
        self.length = int(layout["prefill_length"])
        self.original = None
        self.diag = {"prefill_layers": [], "decode_calls": 0, "unexpected_calls": []}

    def __enter__(self):
        vs, ve = self.layout["visual"]
        assert (vs, ve) == (1, 256) and self.fg.shape == (256,)
        prior = cache_layers(self.source)
        self.original = DynamicCache.update
        layers, components, length, fg = self.layers, self.components, self.length, self.fg

        def update(cache, key, value, layer_idx, cache_kwargs=None):
            if layer_idx in layers and key.shape[-2] == length:
                old_k, old_v = prior[layer_idx]
                assert key.shape == old_k.shape and value.shape == old_v.shape
                bg, fresh_fg = _indices(fg, key.device)
                current_k, current_v = key.clone(), value.clone()
                if "K" in components:
                    key.index_copy_(2, bg, old_k.index_select(2, bg))
                    assert torch.equal(key.index_select(2, bg), old_k.index_select(2, bg))
                    assert torch.equal(key.index_select(2, fresh_fg), current_k.index_select(2, fresh_fg))
                if "V" in components:
                    value.index_copy_(2, bg, old_v.index_select(2, bg))
                    assert torch.equal(value.index_select(2, bg), old_v.index_select(2, bg))
                    assert torch.equal(value.index_select(2, fresh_fg), current_v.index_select(2, fresh_fg))
                assert torch.equal(key[:, :, 0, :], current_k[:, :, 0, :])
                assert torch.equal(value[:, :, 0, :], current_v[:, :, 0, :])
                ls, le = self.layout["language"]
                assert torch.equal(key[:, :, ls:le + 1, :], current_k[:, :, ls:le + 1, :])
                assert torch.equal(value[:, :, ls:le + 1, :], current_v[:, :, ls:le + 1, :])
                self.diag["prefill_layers"].append(layer_idx)
            elif layer_idx in layers and key.shape[-2] == 1:
                self.diag["decode_calls"] += 1
            elif layer_idx in layers:
                self.diag["unexpected_calls"].append({"layer": layer_idx, "rows": int(key.shape[-2])})
            return self.original(cache, key, value, layer_idx, cache_kwargs)

        DynamicCache.update = update
        return self

    def __exit__(self, *_):
        DynamicCache.update = self.original

    def validate(self, cache: DynamicCache):
        assert sorted(self.diag["prefill_layers"]) == list(self.layers)
        assert self.diag["decode_calls"] == len(self.layers) * 6
        assert not self.diag["unexpected_calls"]
        for layer, (key, value) in enumerate(cache_layers(cache)):
            old_k, old_v = cache_layers(self.source)[layer]
            assert key.shape == old_k.shape and value.shape == old_v.shape
            if layer in self.layers:
                _, bg = _indices(self.fg, key.device)
                if "K" in self.components:
                    assert torch.equal(key.index_select(2, bg), old_k.index_select(2, bg))
                if "V" in self.components:
                    assert torch.equal(value.index_select(2, bg), old_v.index_select(2, bg))


class ProjectionAudit:
    """Non-invasive row-count hooks for every Q/K/V linear in the decoder."""
    def __init__(self, model: Any):
        self.calls: dict[str, dict[int, list[int]]] = {name: {i: [] for i in ALL_LAYERS} for name in ("q", "k", "v")}
        self.handles = []
        for i, layer in enumerate(model.language_model.model.layers):
            for name in ("q", "k", "v"):
                module = getattr(layer.self_attn, f"{name}_proj")
                self.handles.append(module.register_forward_pre_hook(self._hook(name, i)))

    def _hook(self, name: str, layer: int):
        def hook(_module, args):
            self.calls[name][layer].append(int(args[0].shape[-2]))
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    def summary(self, prefill_length: int) -> dict[str, list[int]]:
        out = {}
        for name in ("q", "k", "v"):
            rows = []
            for layer in ALL_LAYERS:
                calls = self.calls[name][layer]
                assert len(calls) == 7, (name, layer, calls)
                assert calls[-6:] == [1] * 6, (name, layer, calls)
                rows.append(calls[0])
            out[f"{name}_projected_rows_by_layer"] = rows
        assert out["q_projected_rows_by_layer"] == [prefill_length] * 32
        return out


class MaskedProjectionSkip:
    """True skip: K and optionally V projection omit only oracle-background visual rows."""
    def __init__(self, model: Any, source: DynamicCache, layout: dict[str, Any], foreground: np.ndarray,
                 layers: tuple[int, ...], components: tuple[str, ...]):
        self.model, self.source, self.layout = model, source, layout
        self.fg = np.asarray(foreground, dtype=bool)
        self.layers, self.components = layers, components
        self.length = int(layout["prefill_length"])
        self.original_forwards: dict[int, Any] = {}
        self.diag: dict[str, Any] = {"prefill_layers": [], "rows_by_layer": [], "decode_rows_by_layer": [], "unexpected_fallbacks": []}

    def __enter__(self):
        vs, ve = self.layout["visual"]
        ls, le = self.layout["language"]
        source_layers = cache_layers(self.source)
        assert (vs, ve) == (1, 256) and self.fg.shape == (256,) and len(source_layers) == 32
        assert all(k.shape[-2] == self.length and v.shape[-2] == self.length for k, v in source_layers)
        for layer_idx in self.layers:
            attention = self.model.language_model.model.layers[layer_idx].self_attn
            self.original_forwards[layer_idx] = attention.forward

            def forward(module, hidden_states, position_embeddings, attention_mask,
                        past_key_value=None, cache_position=None, **kwargs):
                h = hidden_states
                if h.shape[-2] == 1:
                    self.diag["decode_rows_by_layer"].append({"layer": module.layer_idx, "rows": 1})
                    return self.original_forwards[module.layer_idx](
                        h, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
                if h.shape[-2] != self.length or past_key_value is None:
                    self.diag["unexpected_fallbacks"].append({"layer": module.layer_idx, "rows": int(h.shape[-2])})
                    raise RuntimeError("Unexpected non-prefill call reached the visual-mask skip path.")
                batch, seq_len, _ = h.shape
                device = h.device
                fg_idx, bg_idx = _indices(self.fg, device)
                language_idx = torch.arange(ls, seq_len, dtype=torch.long, device=device)
                fresh_idx = torch.cat((torch.zeros(1, dtype=torch.long, device=device), fg_idx, language_idx))
                shape = (batch, seq_len, -1, module.head_dim)
                query = module.q_proj(h).view(shape).transpose(1, 2)
                cos, sin = position_embeddings
                query, _ = apply_rotary_pos_emb(query, query, cos, sin)

                if "K" in self.components:
                    fresh_k = module.k_proj(h.index_select(1, fresh_idx))
                    fresh_k = fresh_k.view(batch, fresh_idx.numel(), -1, module.head_dim).transpose(1, 2)
                    _, fresh_k = apply_rotary_pos_emb(
                        fresh_k, fresh_k, cos.index_select(1, fresh_idx), sin.index_select(1, fresh_idx))
                    prior_k, _ = source_layers[module.layer_idx]
                    key = prior_k.clone()
                    key.index_copy_(2, fresh_idx, fresh_k)
                    assert torch.equal(key.index_select(2, fresh_idx), fresh_k)
                    k_rows = int(fresh_idx.numel())
                else:
                    key = module.k_proj(h).view(shape).transpose(1, 2)
                    # The original implementation uses the unrotated Q here;
                    # this branch is retained for completeness, though both
                    # Stage 3C/3D variants reuse K.
                    q_unrotated = module.q_proj(h).view(shape).transpose(1, 2)
                    _, key = apply_rotary_pos_emb(q_unrotated, key, cos, sin)
                    k_rows = seq_len

                if "V" in self.components:
                    fresh_v = module.v_proj(h.index_select(1, fresh_idx))
                    fresh_v = fresh_v.view(batch, fresh_idx.numel(), -1, module.head_dim).transpose(1, 2)
                    _, prior_v = source_layers[module.layer_idx]
                    value = prior_v.clone()
                    value.index_copy_(2, fresh_idx, fresh_v)
                    assert torch.equal(value.index_select(2, fresh_idx), fresh_v)
                    v_rows = int(fresh_idx.numel())
                else:
                    value = module.v_proj(h).view(shape).transpose(1, 2)
                    v_rows = seq_len

                assert key.shape[-2] == seq_len and value.shape[-2] == seq_len
                assert key.shape[-2] == source_layers[module.layer_idx][0].shape[-2]
                self.diag["rows_by_layer"].append({
                    "layer": module.layer_idx, "sequence_rows": seq_len, "q_projected_rows": seq_len,
                    "k_projected_rows": k_rows, "v_projected_rows": v_rows,
                    "k_reused_background_rows": int(bg_idx.numel()) if "K" in self.components else 0,
                    "v_reused_background_rows": int(bg_idx.numel()) if "V" in self.components else 0,
                    "k_skipped_background_rows": int(bg_idx.numel()) if "K" in self.components else 0,
                    "v_skipped_background_rows": int(bg_idx.numel()) if "V" in self.components else 0,
                    "fresh_indices_sha256": hashlib.sha256(fresh_idx.detach().cpu().numpy().tobytes()).hexdigest()[:16],
                    "fresh_k_rope_original_positions": True, "v_rope_applied": False,
                })
                key, value = past_key_value.update(
                    key, value, module.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
                interface = eager_attention_forward if module.config._attn_implementation == "eager" else ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
                out, weights = interface(
                    module, query, key, value, attention_mask,
                    dropout=0.0 if not module.training else module.attention_dropout,
                    scaling=module.scaling, **kwargs)
                self.diag["prefill_layers"].append(module.layer_idx)
                return module.o_proj(out.reshape(batch, seq_len, -1).contiguous()), weights

            attention.forward = types.MethodType(forward, attention)
        return self

    def __exit__(self, *_):
        for layer_idx, original in self.original_forwards.items():
            self.model.language_model.model.layers[layer_idx].self_attn.forward = original

    def validate(self, cache: DynamicCache):
        assert sorted(self.diag["prefill_layers"]) == list(self.layers)
        assert not self.diag["unexpected_fallbacks"]
        assert len(self.diag["decode_rows_by_layer"]) == len(self.layers) * 6
        expected_kv_rows = 1 + int(self.fg.sum()) + (self.layout["language"][1] - self.layout["language"][0] + 1)
        assert len(self.diag["rows_by_layer"]) == len(self.layers)
        for row in self.diag["rows_by_layer"]:
            assert row["q_projected_rows"] == self.length
            assert row["k_projected_rows"] == (expected_kv_rows if "K" in self.components else self.length)
            assert row["v_projected_rows"] == (expected_kv_rows if "V" in self.components else self.length)
        for layer, (key, value) in enumerate(cache_layers(cache)):
            old_k, old_v = cache_layers(self.source)[layer]
            assert key.shape == old_k.shape and value.shape == old_v.shape
            if layer in self.layers:
                _, bg_idx = _indices(self.fg, key.device)
                if "K" in self.components:
                    assert torch.equal(key.index_select(2, bg_idx), old_k.index_select(2, bg_idx))
                if "V" in self.components:
                    assert torch.equal(value.index_select(2, bg_idx), old_v.index_select(2, bg_idx))


def _metric(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    x, y = a.float().cpu(), b.float().cpu()
    diff = x - y
    return {"max_abs": float(diff.abs().max()), "mean_abs": float(diff.abs().mean()),
            "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(x).clamp_min(1e-20))}


def _run_prediction(mode: str, model: Any, processor: Any, device: Any, key: str, task: Any,
                    image: np.ndarray, foreground: np.ndarray, step: int, source: dict[str, Any],
                    audit: ProjectionAudit | None = None):
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    if step % R == 0:
        try:
            pred = capture_prediction(model, processor, image, prompt, device, key)
        finally:
            if audit is not None:
                audit.close()
        return pred, None
    expected_source = step - step % R
    assert source and source["step"] == expected_source and source["fresh"]
    cls = MaskedOverwrite if mode == "overwrite" else MaskedProjectionSkip
    if mode == "overwrite":
        ctx = cls(source["cache"], source["layout"], foreground, ACTIVE_LAYERS, COMPONENTS)
    else:
        ctx = cls(model, source["cache"], source["layout"], foreground, ACTIVE_LAYERS, COMPONENTS)
    try:
        pred = capture_prediction(model, processor, image, prompt, device, key, ctx)
    finally:
        if audit is not None:
            audit.close()
    ctx.validate(pred["cache"])
    return pred, ctx


def run_validation(model: Any, processor: Any, device: Any, key: str) -> dict[str, Any]:
    """Age-1..4 same-observation/source overwrite-vs-skip check before matrix."""
    seed_everything(SEED)
    task, states, env = make_env_and_task(4)
    rows, source, done = [], None, False
    try:
        env.reset(); obs = env.set_init_state(states[0])
        mapping = rendered_label_mapping(env.env.model.instances_to_ids)
        for _ in range(10): obs, _, _, _ = env.step(DEFAULT_STEP_ACTION)
        for step in range(6):
            foreground = mask_from_obs(obs, mapping)
            image = policy_image_from_observation(obs, True)
            if step % R == 0:
                fresh, _ = _run_prediction("skip", model, processor, device, key, task, image, foreground, step, source)
                cloned = clone_prefill_cache(fresh["cache"])
                assert_independent_cache_clone(cloned, fresh["cache"])
                source = {"cache": cloned, "layout": dict(fresh["layout"]), "step": step, "fresh": True}
                skip_pred = overwrite_pred = fresh
                delta = {"k": None, "v": None, "logits": {"max_abs": 0.0, "mean_abs": 0.0, "relative_l2": 0.0}}
                row_audit = {"k_projected_rows_by_layer": [fresh["prefill_length"]] * 32,
                             "v_projected_rows_by_layer": [fresh["prefill_length"]] * 32,
                             "q_projected_rows_by_layer": [fresh["prefill_length"]] * 32}
            else:
                audit_ow = ProjectionAudit(model)
                overwrite_pred, ow = _run_prediction("overwrite", model, processor, device, key, task, image,
                                                      foreground, step, source, audit_ow)
                # Recreate hooks for a clean per-mode count.
                audit_sk = ProjectionAudit(model)
                skip_pred, sk = _run_prediction("skip", model, processor, device, key, task, image,
                                                foreground, step, source, audit_sk)
                ow_cache, sk_cache = cache_layers(overwrite_pred["cache"]), cache_layers(skip_pred["cache"])
                for layer in ACTIVE_LAYERS:
                    assert ow_cache[layer][0].shape == sk_cache[layer][0].shape
                    assert ow_cache[layer][1].shape == sk_cache[layer][1].shape
                    _, bg = _indices(foreground, ow_cache[layer][0].device)
                    for component_index, component in ((0, "K"), (1, "V")):
                        if component in COMPONENTS:
                            assert torch.equal(ow_cache[layer][component_index].index_select(2, bg), sk_cache[layer][component_index].index_select(2, bg))
                assert overwrite_pred["layout"] == skip_pred["layout"]
                assert overwrite_pred["prefill_length"] == skip_pred["prefill_length"]
                overwrite_rows = audit_ow.summary(overwrite_pred["prefill_length"])
                assert overwrite_rows["q_projected_rows_by_layer"] == [overwrite_pred["prefill_length"]] * 32
                assert overwrite_rows["k_projected_rows_by_layer"] == [overwrite_pred["prefill_length"]] * 32
                assert overwrite_rows["v_projected_rows_by_layer"] == [overwrite_pred["prefill_length"]] * 32
                row_audit = audit_sk.summary(skip_pred["prefill_length"])
                delta = {"k": None, "v": None, "logits": _metric(overwrite_pred["logits"], skip_pred["logits"])}
                for component_index, component in ((0, "K"), (1, "V")):
                    if component in COMPONENTS:
                        values = [_metric(ow_cache[layer][component_index], sk_cache[layer][component_index]) for layer in ACTIVE_LAYERS]
                        delta[component.lower()] = {"max_abs": max(x["max_abs"] for x in values),
                                                    "mean_abs": float(np.mean([x["mean_abs"] for x in values])),
                                                    "relative_l2_by_layer": [x["relative_l2"] for x in values]}
                numerical_gate_passed = delta["logits"]["relative_l2"] <= NUMERICAL_LOGIT_REL_L2_TOLERANCE
            if step % R == 0:
                numerical_gate_passed = True
            assert np.isfinite(overwrite_pred["policy_action"]).all() and np.isfinite(skip_pred["policy_action"]).all()
            policy_delta = np.asarray(overwrite_pred["policy_action"]) - np.asarray(skip_pred["policy_action"])
            row = {"step": step, "type": "refresh" if step % R == 0 else "reuse",
                   "source_step": source["step"] if step % R else None, "source_age": step - source["step"] if step % R else 0,
                   "foreground_tokens": int(foreground.sum()), "background_tokens": int((~foreground).sum()),
                   "selected_layer_set_verified": list(ACTIVE_LAYERS) == CONFIG["layers"],
                   "all_32_layers_patched": len(ACTIVE_LAYERS) == 32,
                   "patched_layers": list(ACTIVE_LAYERS), "components": list(COMPONENTS),
                   "projection_rows_by_layer": row_audit, "numerical_differences": delta,
                   "numerical_gate_passed": numerical_gate_passed,
                   "action_token_ids_equal": overwrite_pred["action_token_ids"] == skip_pred["action_token_ids"],
                   "policy_action_max_abs_difference": float(np.max(np.abs(policy_delta))),
                   "cache_shapes_and_stale_source_provenance_passed": True,
                   "all_actions_finite": True, "unexpected_fallbacks": 0, "decode_delegated": True}
            rows.append(row)
            action = normalize_and_invert_openvla_gripper(skip_pred["policy_action"])
            obs, _, done, _ = env.step(action.tolist())
            if done: break
    finally:
        env.close()
    result = {"stage": STAGE, "task_id": 4, "state_id": 0, "seed": SEED, "steps": len(rows),
              "layers": list(ACTIVE_LAYERS), "components": list(COMPONENTS), "refresh_interval": R,
              "max_source_age": max(x["source_age"] for x in rows), "records": rows,
              "all_valid": len(rows) == 6 and all(x["all_actions_finite"] and x["cache_shapes_and_stale_source_provenance_passed"] and x["numerical_gate_passed"] for x in rows),
              "logit_relative_l2_tolerance": NUMERICAL_LOGIT_REL_L2_TOLERANCE}
    assert result["all_valid"] and result["max_source_age"] == 4
    atomic_json(OUT / "validation.json", result)
    print(f"[VALIDATION_PASS] {STAGE}; layers={len(ACTIVE_LAYERS)}; components={COMPONENTS}; Q full; source ages 1-4; logit rel-L2 <= {NUMERICAL_LOGIT_REL_L2_TOLERANCE}", flush=True)
    assert result["all_valid"], f"Numerical/structural validation failed; see {OUT / 'validation.json'}"
    return result


def _fresh_reference(task: int, state: int) -> dict[str, Any]:
    path = STAGE3B / f"task{task}_state{state}" / "results.json"
    data = json.loads(path.read_text())
    assert data["config"]["seed"] == SEED and data["config"]["horizon"] == HORIZON
    assert data["config"]["R"] == R and data["config"]["layers"] == list(ALL_LAYERS)
    assert data["config"]["visual_positions"] == [1, 256]
    fresh = data["fresh"]
    assert fresh["seed"] == SEED and fresh["task_id"] == task and fresh["state_id"] == state
    assert fresh["all_valid"] and fresh["termination"] in ("success", "horizon")
    p = path.relative_to(ROOT)
    return {"reference_path": str(p), "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_commit": "c9b05be688ab0ad5d474665300b8504f4c077ebb",
            **{k: fresh[k] for k in ("condition", "task_id", "task_description", "state_id", "seed", "success", "steps", "termination", "refresh_count", "reuse_count", "reuse_fraction", "all_valid")}}


def run_reuse_rollout(task_id: int, state_id: int, model: Any, processor: Any, device: Any, key: str) -> dict[str, Any]:
    seed_everything(SEED)
    task, states, env = make_env_and_task(task_id)
    records, source, done = [], None, False
    try:
        env.reset(); obs = env.set_init_state(states[state_id])
        mapping = rendered_label_mapping(env.env.model.instances_to_ids)
        for _ in range(10): obs, _, _, _ = env.step(DEFAULT_STEP_ACTION)
        for step in range(HORIZON):
            foreground = mask_from_obs(obs, mapping)
            image = policy_image_from_observation(obs, True)
            audit = ProjectionAudit(model)
            pred, context = _run_prediction("skip", model, processor, device, key, task, image, foreground, step, source, audit)
            row_counts = audit.summary(pred["prefill_length"])
            if step % R == 0:
                fresh_cache = clone_prefill_cache(pred["cache"])
                assert_independent_cache_clone(fresh_cache, pred["cache"])
                assert all(k.shape[-2] == pred["prefill_length"] and v.shape[-2] == pred["prefill_length"] for k, v in cache_layers(fresh_cache))
                source = {"cache": fresh_cache, "layout": dict(pred["layout"]), "step": step, "fresh": True}
                expected_source, age = step, 0
                skipped_k = [0] * 32; skipped_v = [0] * 32
            else:
                expected_source = step - step % R
                assert source is not None and source["step"] == expected_source and source["fresh"]
                age = step - source["step"]
                skipped_k = [int((~foreground).sum()) if i in ACTIVE_LAYERS and "K" in COMPONENTS else 0 for i in ALL_LAYERS]
                skipped_v = [int((~foreground).sum()) if i in ACTIVE_LAYERS and "V" in COMPONENTS else 0 for i in ALL_LAYERS]
                expected_fresh = 1 + int(foreground.sum()) + int(pred["layout"]["language"][1] - pred["layout"]["language"][0] + 1)
                for i in ACTIVE_LAYERS:
                    assert row_counts["k_projected_rows_by_layer"][i] == (expected_fresh if "K" in COMPONENTS else pred["prefill_length"])
                    assert row_counts["v_projected_rows_by_layer"][i] == (expected_fresh if "V" in COMPONENTS else pred["prefill_length"])
                    assert row_counts["q_projected_rows_by_layer"][i] == pred["prefill_length"]
                for i in ALL_LAYERS:
                    if i not in ACTIVE_LAYERS:
                        assert row_counts["k_projected_rows_by_layer"][i] == pred["prefill_length"]
                        assert row_counts["v_projected_rows_by_layer"][i] == pred["prefill_length"]
                        assert row_counts["q_projected_rows_by_layer"][i] == pred["prefill_length"]
            assert pred["layout"]["visual"] == [1, 256] and pred["layout"]["visual_token_count"] == 256
            assert np.isfinite(pred["policy_action"]).all()
            action = normalize_and_invert_openvla_gripper(pred["policy_action"])
            obs, _, done, _ = env.step(action.tolist())
            records.append({
                "step": step, "type": "refresh" if step % R == 0 else "reuse", "refresh_interval": R,
                "source_step": expected_source, "source_age": age, "source_is_fresh_refresh_snapshot": bool(source and source["fresh"]),
                "prefill_length": pred["prefill_length"], "language_tokens": pred["layout"]["language"][1] - pred["layout"]["language"][0] + 1,
                "foreground_tokens": int(foreground.sum()), "background_tokens": int((~foreground).sum()),
                "mask_sha256": hashlib.sha256(foreground.tobytes()).hexdigest(),
                "q_projected_rows_by_layer": row_counts["q_projected_rows_by_layer"],
                "k_projected_rows_by_layer": row_counts["k_projected_rows_by_layer"],
                "v_projected_rows_by_layer": row_counts["v_projected_rows_by_layer"],
                "skipped_k_rows_by_layer": skipped_k, "skipped_v_rows_by_layer": skipped_v,
                "policy_action": pred["policy_action"].tolist(), "environment_action": action.tolist(),
                "action_token_ids": pred["action_token_ids"], "done": bool(done),
                "validation": {"passed": True, "all_layers_projected_as_configured": True,
                               "foreground_fresh_background_source": True, "q_full": True,
                               "rope_original_position": True, "v_unrotated": True,
                               "decode_normal": True, "nonfinite_action": False,
                               "unexpected_fallbacks": 0},
            })
            if done: break
    finally:
        env.close()
    reuse = [r for r in records if r["type"] == "reuse"]
    total_normal = sum(r["prefill_length"] * 32 for r in records)
    skipped_k = [sum(r["skipped_k_rows_by_layer"][i] for r in records) for i in ALL_LAYERS]
    skipped_v = [sum(r["skipped_v_rows_by_layer"][i] for r in records) for i in ALL_LAYERS]
    result = {"condition": f"{STAGE}_true_skip", "task_id": task_id, "task_description": task.language,
              "state_id": state_id, "seed": SEED, "success": bool(done), "steps": len(records),
              "termination": "success" if done else "horizon",
              "refresh_count": sum(r["type"] == "refresh" for r in records),
              "reuse_count": len(reuse), "reuse_fraction": len(reuse) / len(records),
              "source_age_counts": {str(age): sum(r["source_age"] == age for r in records) for age in range(R)},
              "max_source_age": max(r["source_age"] for r in records), "all_valid": all(r["validation"]["passed"] for r in records),
              "skipped_k_rows_by_layer_aggregate": skipped_k, "skipped_v_rows_by_layer_aggregate": skipped_v,
              "total_skipped_k_projection_rows": sum(skipped_k), "total_skipped_v_projection_rows": sum(skipped_v),
              "total_skipped_kv_projection_rows": sum(skipped_k) + sum(skipped_v),
              "normal_k_rows_all_prefills": total_normal, "normal_v_rows_all_prefills": total_normal,
              "effective_k_row_skip_fraction": sum(skipped_k) / total_normal if total_normal else 0.0,
              "effective_v_row_skip_fraction": sum(skipped_v) / total_normal if total_normal else 0.0,
              "effective_combined_kv_row_skip_fraction": (sum(skipped_k) + sum(skipped_v)) / (2 * total_normal) if total_normal else 0.0,
              "reuse_spatial_background_fraction": sum(r["background_tokens"] for r in reuse) / (256 * len(reuse)) if reuse else 0.0,
              "latency_measured": False, "records": records}
    return result


def validate_summary(path: Path):
    data = json.loads(path.read_text())
    assert data["status"] == "complete" and len(data["scenarios"]) == 9
    for compact in data["scenarios"]:
        scenario_path = path.parent / f"task{compact['task_id']}_state{compact['state_id']}" / "results.json"
        scenario = json.loads(scenario_path.read_text())
        assert scenario["config"] == CONFIG
        rollout = scenario["true_skip"]
        assert rollout["all_valid"] and rollout["steps"] == len(rollout["records"])
        for row in rollout["records"]:
            assert row["foreground_tokens"] + row["background_tokens"] == 256
            assert row["source_age"] == row["step"] % R
            assert len(row["skipped_k_rows_by_layer"]) == len(row["skipped_v_rows_by_layer"]) == 32
            if row["step"] % R:
                assert row["source_step"] == row["step"] - row["step"] % R
                expected_fresh = 1 + row["foreground_tokens"] + row["language_tokens"]
                for layer in ALL_LAYERS:
                    expected_k = expected_fresh if layer in ACTIVE_LAYERS and "K" in COMPONENTS else row["prefill_length"]
                    expected_v = expected_fresh if layer in ACTIVE_LAYERS and "V" in COMPONENTS else row["prefill_length"]
                    expected_skip_k = row["background_tokens"] if layer in ACTIVE_LAYERS and "K" in COMPONENTS else 0
                    expected_skip_v = row["background_tokens"] if layer in ACTIVE_LAYERS and "V" in COMPONENTS else 0
                    assert row["k_projected_rows_by_layer"][layer] == expected_k
                    assert row["v_projected_rows_by_layer"][layer] == expected_v
                    assert row["q_projected_rows_by_layer"][layer] == row["prefill_length"]
                    assert row["skipped_k_rows_by_layer"][layer] == expected_skip_k
                    assert row["skipped_v_rows_by_layer"][layer] == expected_skip_v
    return data


def _compact_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    reuse = scenario["true_skip"]
    records = reuse["records"]
    return {"task_id": scenario["task_id"], "state_id": scenario["state_id"],
            "fresh_reference": {k: scenario["fresh_reference"][k] for k in
                                ("reference_path", "reference_sha256", "source_commit", "success", "steps", "termination")},
            "true_skip": {k: reuse[k] for k in ("success", "steps", "termination", "refresh_count", "reuse_count",
                                                "reuse_fraction", "source_age_counts", "max_source_age", "all_valid",
                                                "skipped_k_rows_by_layer_aggregate", "skipped_v_rows_by_layer_aggregate",
                                                "total_skipped_k_projection_rows", "total_skipped_v_projection_rows",
                                                "total_skipped_kv_projection_rows", "normal_k_rows_all_prefills",
                                                "normal_v_rows_all_prefills", "effective_k_row_skip_fraction",
                                                "effective_v_row_skip_fraction", "effective_combined_kv_row_skip_fraction",
                                                "reuse_spatial_background_fraction")},
            "foreground_tokens_min": min(x["foreground_tokens"] for x in records),
            "foreground_tokens_max": max(x["foreground_tokens"] for x in records),
            "background_tokens_min": min(x["background_tokens"] for x in records),
            "background_tokens_max": max(x["background_tokens"] for x in records),
            "prefill_lengths_observed": sorted({x["prefill_length"] for x in records}),
            "language_tokens_observed": sorted({x["language_tokens"] for x in records})}


def run_matrix(model: Any, processor: Any, device: Any, key: str):
    fresh_refs = []
    for task_id in (4, 5, 7):
        for state_id in (0, 1, 2):
            fresh_refs.append(_fresh_reference(task_id, state_id))
    scenarios = []
    (OUT / "summary.json").write_text(json.dumps({"status": "running", "completed": 0}, indent=2))
    for fresh_ref in fresh_refs:
        task_id, state_id = fresh_ref["task_id"], fresh_ref["state_id"]
        scenario_dir = OUT / f"task{task_id}_state{state_id}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        result_path = scenario_dir / "results.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if existing.get("config") == CONFIG and existing.get("true_skip", {}).get("all_valid"):
                scenarios.append(existing)
                print(f"[SCENARIO_RESUME] {STAGE} task={task_id} state={state_id}; verified saved result", flush=True)
                continue
            raise RuntimeError(f"Refusing to overwrite existing incompatible/invalid scenario output: {result_path}")
        true_skip = run_reuse_rollout(task_id, state_id, model, processor, device, key)
        scenario = {"task_id": task_id, "state_id": state_id, "config": CONFIG,
                    "fresh_reference": fresh_ref, "true_skip": true_skip}
        atomic_json(result_path, scenario)
        scenarios.append(scenario)
        partial = {"status": "running", "completed": len(scenarios),
                   "scenarios": [{"task_id": x["task_id"], "state_id": x["state_id"],
                                  "fresh_success": x["fresh_reference"]["success"], "fresh_steps": x["fresh_reference"]["steps"],
                                  "success": x["true_skip"]["success"], "steps": x["true_skip"]["steps"],
                                  "refresh": x["true_skip"]["refresh_count"], "reuse": x["true_skip"]["reuse_count"],
                                  "all_valid": x["true_skip"]["all_valid"]} for x in scenarios]}
        atomic_json(OUT / "summary.json", partial)
        print(f"[SCENARIO_COMPLETE] {STAGE} task={task_id} state={state_id} success={true_skip['success']}/{true_skip['steps']}", flush=True)
    fresh_success = sum(x["fresh_reference"]["success"] for x in scenarios)
    reuse_success = sum(x["true_skip"]["success"] for x in scenarios)
    final = {"status": "complete", "config": CONFIG,
             "aggregate": {"fresh_successes": fresh_success, "fresh_total": 9,
                           "true_skip_successes": reuse_success, "true_skip_total": 9,
                           "true_skip_successes_among_fresh_successes": sum(x["true_skip"]["success"] for x in scenarios if x["fresh_reference"]["success"]),
                           "fresh_success_scenarios": fresh_success,
                           "total_skipped_k_rows": sum(x["true_skip"]["total_skipped_k_projection_rows"] for x in scenarios),
                           "total_skipped_v_rows": sum(x["true_skip"]["total_skipped_v_projection_rows"] for x in scenarios),
                           "latency_measured": False},
             "scenarios": [_compact_scenario(x) for x in scenarios],
             "detail_files": [f"task{x['task_id']}_state{x['state_id']}/results.json" for x in scenarios]}
    path = OUT / "summary.json"
    atomic_json(path, final)
    validate_summary(path)
    return final


def main():
    global STAGE, ACTIVE_LAYERS, COMPONENTS, OUT, CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(VARIANTS), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--matrix", action="store_true")
    args = parser.parse_args()
    STAGE = args.stage
    settings = VARIANTS[STAGE]
    ACTIVE_LAYERS, COMPONENTS = settings["layers"], settings["components"]
    OUT = ROOT / "outputs" / settings["name"]
    CONFIG = {"stage": STAGE, "suite": TASK_SUITE_NAME, "tasks": [4, 5, 7], "states": [0, 1, 2],
              "seed": SEED, "horizon": HORIZON, "settling_steps": 10, "R": R,
              "layers": list(ACTIVE_LAYERS), "visual_positions": [1, 256], "components": list(COMPONENTS),
              "mechanism": "oracle masked visual background projection skip", "fresh_reference_commit": "c9b05be688ab0ad5d474665300b8504f4c077ebb",
              "latency_measured": False}
    OUT.mkdir(parents=True, exist_ok=True)
    model, processor, device = load_model_and_processor(DEFAULT_MODEL_ID)
    key = resolve_unnorm_key(model, None)
    if args.validate_only:
        run_validation(model, processor, device, key)
    else:
        run_matrix(model, processor, device, key)


if __name__ == "__main__":
    main()
