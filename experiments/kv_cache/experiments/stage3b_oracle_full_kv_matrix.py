"""Stage 3B: full-layer oracle-masked visual K/V projection reuse, R=5."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from libero.libero import benchmark
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage3b_oracle_full_kv_matrix"
LAYERS = tuple(range(32))
TASKS, STATES, R, HORIZON, SEED = (4, 5, 7), (0, 1, 2), 5, 220, 7


class MaskedKVOverwrite:
    """Reference: full fresh projections, then replace background visual K/V."""
    def __init__(self, source: DynamicCache, layout: dict[str, Any], foreground: np.ndarray):
        self.source, self.layout = source, layout
        self.fg = np.asarray(foreground, dtype=bool); self.bg = ~self.fg
        self.length = int(layout["prefill_length"]); self.original = None
        self.diag: dict[str, Any] = {"layers": [], "decode_calls": 0, "unexpected_calls": []}

    def __enter__(self):
        vs, ve = self.layout["visual"]
        assert (vs, ve) == (1, 256) and self.fg.shape == (256,)
        self.original = DynamicCache.update
        source_layers = cache_layers(self.source)
        bg = torch.as_tensor(np.flatnonzero(self.bg) + 1, dtype=torch.long)
        fg = torch.as_tensor(np.flatnonzero(self.fg) + 1, dtype=torch.long)

        def update(cache, key, value, layer_idx, cache_kwargs=None):
            if layer_idx in LAYERS and key.shape[-2] == self.length:
                old_k, old_v = source_layers[layer_idx]
                assert key.shape == old_k.shape and value.shape == old_v.shape
                fresh_k, fresh_v = key.clone(), value.clone()
                bi, fi = bg.to(key.device), fg.to(key.device)
                key[:, :, bi, :].copy_(old_k[:, :, bi, :])
                value[:, :, bi, :].copy_(old_v[:, :, bi, :])
                assert torch.equal(key[:, :, bi, :], old_k[:, :, bi, :])
                assert torch.equal(value[:, :, bi, :], old_v[:, :, bi, :])
                assert torch.equal(key[:, :, fi, :], fresh_k[:, :, fi, :])
                assert torch.equal(value[:, :, fi, :], fresh_v[:, :, fi, :])
                assert torch.equal(key[:, :, 0, :], fresh_k[:, :, 0, :])
                assert torch.equal(value[:, :, 0, :], fresh_v[:, :, 0, :])
                ls, le = self.layout["language"]
                assert torch.equal(key[:, :, ls:le+1, :], fresh_k[:, :, ls:le+1, :])
                assert torch.equal(value[:, :, ls:le+1, :], fresh_v[:, :, ls:le+1, :])
                self.diag["layers"].append(layer_idx)
            elif layer_idx in LAYERS and key.shape[-2] == 1:
                self.diag["decode_calls"] += 1
            elif layer_idx in LAYERS:
                self.diag["unexpected_calls"].append({"layer": layer_idx, "rows": int(key.shape[-2])})
            return self.original(cache, key, value, layer_idx, cache_kwargs)

        DynamicCache.update = update
        return self

    def __exit__(self, *_):
        DynamicCache.update = self.original

    def validate(self, cache):
        assert sorted(self.diag["layers"]) == list(LAYERS)
        assert self.diag["decode_calls"] == len(LAYERS) * 6
        assert not self.diag["unexpected_calls"]
        self._validate_background(cache)

    def _validate_background(self, cache):
        bg = torch.as_tensor(np.flatnonzero(self.bg) + 1, dtype=torch.long, device=cache_layers(cache)[0][0].device)
        for layer, (key, value) in enumerate(cache_layers(cache)):
            old_k, old_v = cache_layers(self.source)[layer]
            assert key.shape == old_k.shape and value.shape == old_v.shape
            assert torch.equal(key[:, :, bg, :], old_k[:, :, bg, :])
            assert torch.equal(value[:, :, bg, :], old_v[:, :, bg, :])


class MaskedKVProjectionSkip:
    """True skip: project K/V only for BOS, foreground visual, and language rows."""
    def __init__(self, model: Any, source: DynamicCache, layout: dict[str, Any], foreground: np.ndarray):
        self.model, self.source, self.layout = model, source, layout
        self.fg = np.asarray(foreground, dtype=bool); self.bg = ~self.fg
        self.length = int(layout["prefill_length"]); self.original_forwards: dict[int, Any] = {}
        self.diag: dict[str, Any] = {"layers": [], "rows_by_layer": [], "decode_rows_by_layer": [], "unexpected_fallbacks": []}

    def __enter__(self):
        vs, ve = self.layout["visual"]; ls, le = self.layout["language"]
        assert (vs, ve) == (1, 256) and self.fg.shape == (256,)
        source_layers = cache_layers(self.source)
        assert len(source_layers) == 32
        assert all(k.shape[-2] == self.length and v.shape[-2] == self.length for k, v in source_layers)
        layers = self.model.language_model.model.layers
        for layer_idx in LAYERS:
            attention = layers[layer_idx].self_attn
            self.original_forwards[layer_idx] = attention.forward

            def forward(module, hidden_states, position_embeddings, attention_mask,
                        past_key_value=None, cache_position=None, **kwargs):
                if hidden_states.shape[-2] != self.length or past_key_value is None:
                    if hidden_states.shape[-2] == 1:
                        self.diag["decode_rows_by_layer"].append({"layer": module.layer_idx, "rows": 1})
                    else:
                        self.diag["unexpected_fallbacks"].append({"layer": module.layer_idx, "rows": int(hidden_states.shape[-2])})
                    return self.original_forwards[module.layer_idx](
                        hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs
                    )
                batch, seq_len, _ = hidden_states.shape
                device = hidden_states.device
                foreground_idx = torch.as_tensor(np.flatnonzero(self.fg) + 1, dtype=torch.long, device=device)
                language_idx = torch.arange(ls, seq_len, dtype=torch.long, device=device)
                fresh_idx = torch.cat((torch.zeros(1, dtype=torch.long, device=device), foreground_idx, language_idx))
                shape = (batch, seq_len, -1, module.head_dim)
                query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
                # Both K and V use the same selected fresh rows. Q stays full length.
                fresh_hidden = hidden_states.index_select(1, fresh_idx)
                fresh_k = module.k_proj(fresh_hidden).view(batch, fresh_idx.numel(), -1, module.head_dim).transpose(1, 2)
                fresh_v = module.v_proj(fresh_hidden).view(batch, fresh_idx.numel(), -1, module.head_dim).transpose(1, 2)
                cos, sin = position_embeddings
                query, _ = apply_rotary_pos_emb(query, query, cos, sin)
                _, fresh_k = apply_rotary_pos_emb(
                    fresh_k, fresh_k, cos.index_select(1, fresh_idx), sin.index_select(1, fresh_idx)
                )
                prior_k, prior_v = source_layers[module.layer_idx]
                # Source K is already post-RoPE; source V is unrotated. Clone then
                # write every fresh position, leaving only background visual rows stale.
                key = prior_k.clone(); value = prior_v.clone()
                key.index_copy_(2, fresh_idx, fresh_k)
                value.index_copy_(2, fresh_idx, fresh_v)
                assert key.shape[-2] == seq_len and value.shape[-2] == seq_len
                rows = int(fresh_idx.numel())
                assert rows == 1 + int(self.fg.sum()) + (le - ls + 1)
                # Fresh K/V rows are inserted verbatim; background rows remain source values.
                assert torch.equal(key.index_select(2, fresh_idx), fresh_k)
                assert torch.equal(value.index_select(2, fresh_idx), fresh_v)
                self.diag["rows_by_layer"].append({
                    "layer": module.layer_idx, "sequence_rows": seq_len,
                    "k_projected_rows": rows, "v_projected_rows": rows,
                    "k_reused_background_rows": int(self.bg.sum()),
                    "v_reused_background_rows": int(self.bg.sum()),
                    "k_skipped_background_rows": int(self.bg.sum()),
                    "v_skipped_background_rows": int(self.bg.sum()),
                    "fresh_indices_sha256": hashlib.sha256(fresh_idx.detach().cpu().numpy().tobytes()).hexdigest()[:16],
                    "fresh_k_rope_original_positions": True, "v_rope_applied": False,
                })
                key, value = past_key_value.update(
                    key, value, module.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )
                interface = eager_attention_forward if module.config._attn_implementation == "eager" else ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
                out, weights = interface(
                    module, query, key, value, attention_mask,
                    dropout=0.0 if not module.training else module.attention_dropout,
                    scaling=module.scaling, **kwargs,
                )
                self.diag["layers"].append(module.layer_idx)
                return module.o_proj(out.reshape(batch, seq_len, -1).contiguous()), weights

            attention.forward = types.MethodType(forward, attention)
        return self

    def __exit__(self, *_):
        for layer_idx, original in self.original_forwards.items():
            self.model.language_model.model.layers[layer_idx].self_attn.forward = original

    def validate(self, cache):
        assert sorted(self.diag["layers"]) == list(LAYERS)
        assert not self.diag["unexpected_fallbacks"]
        expected_decode = len(LAYERS) * 6
        assert len(self.diag["decode_rows_by_layer"]) == expected_decode
        expected_kv_rows = 1 + int(self.fg.sum()) + (self.layout["language"][1] - self.layout["language"][0] + 1)
        assert len(self.diag["rows_by_layer"]) == len(LAYERS)
        assert all(x["k_projected_rows"] == expected_kv_rows and x["v_projected_rows"] == expected_kv_rows for x in self.diag["rows_by_layer"])
        assert all(x["sequence_rows"] == self.length for x in self.diag["rows_by_layer"])
        self.validate_cache(cache)

    def validate_cache(self, cache):
        bg = torch.as_tensor(np.flatnonzero(self.bg) + 1, dtype=torch.long, device=cache_layers(cache)[0][0].device)
        for layer, (key, value) in enumerate(cache_layers(cache)):
            old_k, old_v = cache_layers(self.source)[layer]
            assert key.shape == old_k.shape and value.shape == old_v.shape
            assert torch.equal(key[:, :, bg, :], old_k[:, :, bg, :])
            assert torch.equal(value[:, :, bg, :], old_v[:, :, bg, :])


def make_env_and_task(task_id: int):
    suite = benchmark.get_benchmark_dict()[TASK_SUITE_NAME]()
    task = suite.get_task(task_id)
    from experiments.kv_cache.experiments.stage3a_masked_visual_k_validation import seg_env
    env = seg_env(task)
    return task, suite.get_task_init_states(task_id), env


def predict_step(condition: str, obs, task, mapping, step, source, model, processor, device, key):
    foreground = mask_from_obs(obs, mapping)
    reused = condition == "stage3b_true_skip" and step % R != 0
    image = policy_image_from_observation(obs, True)
    if reused:
        expected_source = step - step % R
        assert source is not None and source["step"] == expected_source and source["fresh"]
        context = MaskedKVProjectionSkip(model, source["cache"], source["layout"], foreground)
        pred = capture_prediction(model, processor, image, f"In: What action should the robot take to {task.language.lower()}?\nOut:", device, key, context)
        context.validate(pred["cache"])
        projection_by_layer = context.diag["rows_by_layer"]
        validation = {"passed": True, "all_32_layers_patched": True, "q_full_prefill": True,
                      "k_v_fresh_rows": projection_by_layer[0]["k_projected_rows"],
                      "k_v_reused_background_rows": int((~foreground).sum()),
                      "unexpected_fallbacks": 0, "decode_delegated_calls": len(context.diag["decode_rows_by_layer"])}
        source_step, age = source["step"], step - source["step"]
    else:
        pred = capture_prediction(model, processor, image, f"In: What action should the robot take to {task.language.lower()}?\nOut:", device, key)
        projection_by_layer = [{"layer": layer, "sequence_rows": pred["prefill_length"],
                                "k_projected_rows": pred["prefill_length"], "v_projected_rows": pred["prefill_length"],
                                "k_reused_background_rows": 0, "v_reused_background_rows": 0,
                                "k_skipped_background_rows": 0, "v_skipped_background_rows": 0}
                               for layer in LAYERS]
        validation = {"passed": True, "refresh_full_prefill": True, "all_32_layers_normal": True}
        source_step, age = None, 0
    assert pred["layout"]["visual"] == [1, 256] and pred["layout"]["visual_token_count"] == 256
    return foreground, reused, pred, source_step, age, projection_by_layer, validation


def run_rollout(condition: str, task_id: int, state_id: int, model, processor, device, key, horizon=HORIZON):
    seed_everything(SEED)
    task, states, env = make_env_and_task(task_id)
    records, source, done = [], None, False
    try:
        env.reset(); obs = env.set_init_state(states[state_id])
        mapping = rendered_label_mapping(env.env.model.instances_to_ids)
        for _ in range(10): obs, _, _, _ = env.step(DEFAULT_STEP_ACTION)
        for step in range(horizon):
            fg, reuse, pred, source_step, age, rows_by_layer, validation = predict_step(
                condition, obs, task, mapping, step, source, model, processor, device, key
            )
            if condition == "stage3b_true_skip" and not reuse:
                fresh_clone = clone_prefill_cache(pred["cache"])
                assert_independent_cache_clone(fresh_clone, pred["cache"])
                source = {"cache": fresh_clone, "layout": dict(pred["layout"]), "step": step, "fresh": True}
            action = normalize_and_invert_openvla_gripper(pred["policy_action"])
            obs, _, done, _ = env.step(action.tolist())
            records.append({
                "step": step, "type": "reuse" if reuse else "refresh", "source_step": source_step,
                "source_age": age, "prefill_length": pred["prefill_length"],
                "language_tokens": pred["layout"]["language"][1] - pred["layout"]["language"][0] + 1,
                "foreground_tokens": int(fg.sum()), "background_tokens": int((~fg).sum()),
                "mask_sha256": hashlib.sha256(fg.tobytes()).hexdigest(),
                "mask_foreground_indices": np.flatnonzero(fg).tolist(),
                "action_token_ids": pred["action_token_ids"], "policy_action": pred["policy_action"].tolist(),
                "environment_action": action.tolist(), "done": bool(done),
                "projected_k_rows_by_layer": [x["k_projected_rows"] for x in rows_by_layer],
                "projected_v_rows_by_layer": [x["v_projected_rows"] for x in rows_by_layer],
                "reused_k_rows_by_layer": [x["k_reused_background_rows"] for x in rows_by_layer],
                "reused_v_rows_by_layer": [x["v_reused_background_rows"] for x in rows_by_layer],
                "skipped_k_rows_by_layer": [x["k_skipped_background_rows"] for x in rows_by_layer],
                "skipped_v_rows_by_layer": [x["v_skipped_background_rows"] for x in rows_by_layer],
                "layer_instrumentation": rows_by_layer, "validation": validation,
            })
            if done: break
    finally:
        env.close()
    reuse_rows = [x for x in records if x["type"] == "reuse"]
    refresh_rows = [x for x in records if x["type"] == "refresh"]
    per_layer_skipped_k = [sum(x["skipped_k_rows_by_layer"][li] for x in records) for li in LAYERS]
    per_layer_skipped_v = [sum(x["skipped_v_rows_by_layer"][li] for x in records) for li in LAYERS]
    total_normal_rows = sum(x["prefill_length"] * len(LAYERS) for x in records)
    return {
        "condition": condition, "task_id": task_id, "task_description": task.language,
        "state_id": state_id, "seed": SEED, "success": bool(done), "steps": len(records),
        "termination": "success" if done else "horizon", "refresh_count": len(refresh_rows),
        "reuse_count": len(reuse_rows), "reuse_fraction": len(reuse_rows) / len(records),
        "source_age_counts": {str(age): sum(x["source_age"] == age for x in records) for age in range(R)},
        "max_source_age": max(x["source_age"] for x in records),
        "all_valid": all(x["validation"]["passed"] for x in records),
        "skipped_k_rows_by_layer_aggregate": per_layer_skipped_k,
        "skipped_v_rows_by_layer_aggregate": per_layer_skipped_v,
        "total_skipped_k_projection_rows": sum(per_layer_skipped_k),
        "total_skipped_v_projection_rows": sum(per_layer_skipped_v),
        "total_skipped_kv_projection_rows": sum(per_layer_skipped_k) + sum(per_layer_skipped_v),
        "normal_k_rows_all_prefills": total_normal_rows,
        "normal_v_rows_all_prefills": total_normal_rows,
        "effective_k_row_skip_fraction": sum(per_layer_skipped_k) / total_normal_rows if total_normal_rows else 0.0,
        "effective_v_row_skip_fraction": sum(per_layer_skipped_v) / total_normal_rows if total_normal_rows else 0.0,
        "effective_combined_kv_row_skip_fraction": (sum(per_layer_skipped_k) + sum(per_layer_skipped_v)) / (2 * total_normal_rows) if total_normal_rows else 0.0,
        "reuse_spatial_background_fraction": sum(x["background_tokens"] for x in reuse_rows) / (256 * len(reuse_rows)) if reuse_rows else 0.0,
        "latency_measured": False, "records": records,
    }


def run_validation(model, processor, device, key):
    """Six-step task4/state0 trace; compare overwrite and skip on each same input/source."""
    seed_everything(SEED)
    task, states, env = make_env_and_task(4)
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    rows, source, done = [], None, False
    try:
        env.reset(); obs = env.set_init_state(states[0])
        mapping = rendered_label_mapping(env.env.model.instances_to_ids)
        for _ in range(10): obs, _, _, _ = env.step(DEFAULT_STEP_ACTION)
        for step in range(6):
            fg = mask_from_obs(obs, mapping)
            image = policy_image_from_observation(obs, True)
            if step % R == 0:
                fresh = capture_prediction(model, processor, image, prompt, device, key)
                source_cache = clone_prefill_cache(fresh["cache"])
                assert_independent_cache_clone(source_cache, fresh["cache"])
                source = {"cache": source_cache, "layout": dict(fresh["layout"]), "step": step, "fresh": True}
                skip_pred = fresh; overwrite_pred = fresh
                full_rows = fresh["prefill_length"]
                row_detail = [{"layer": l, "k_projected_rows": full_rows, "v_projected_rows": full_rows,
                               "k_reused_background_rows": 0, "v_reused_background_rows": 0} for l in LAYERS]
                age = 0
            else:
                assert source is not None and source["step"] == step - step % R
                ow = MaskedKVOverwrite(source["cache"], source["layout"], fg)
                overwrite_pred = capture_prediction(model, processor, image, prompt, device, key, ow)
                ow.validate(overwrite_pred["cache"])
                sk = MaskedKVProjectionSkip(model, source["cache"], source["layout"], fg)
                skip_pred = capture_prediction(model, processor, image, prompt, device, key, sk)
                sk.validate(skip_pred["cache"])
                row_detail = sk.diag["rows_by_layer"]
                full_rows = overwrite_pred["prefill_length"]
                age = step - source["step"]
                # Both paths must have the same cache structure and exactly the same stale BG provenance.
                ow_layers, sk_layers = cache_layers(overwrite_pred["cache"]), cache_layers(skip_pred["cache"])
                assert len(ow_layers) == len(sk_layers) == 32
                for layer in LAYERS:
                    assert ow_layers[layer][0].shape == sk_layers[layer][0].shape
                    assert ow_layers[layer][1].shape == sk_layers[layer][1].shape
                    bgidx = torch.as_tensor(np.flatnonzero(~fg) + 1, device=ow_layers[layer][0].device)
                    assert torch.equal(ow_layers[layer][0][:, :, bgidx, :], sk_layers[layer][0][:, :, bgidx, :])
                    assert torch.equal(ow_layers[layer][1][:, :, bgidx, :], sk_layers[layer][1][:, :, bgidx, :])
                assert overwrite_pred["layout"] == skip_pred["layout"]
                assert overwrite_pred["prefill_length"] == skip_pred["prefill_length"]
            for z in (overwrite_pred, skip_pred):
                assert np.all(np.isfinite(z["policy_action"]))
            action_delta = np.asarray(overwrite_pred["policy_action"]) - np.asarray(skip_pred["policy_action"])
            logit_delta = overwrite_pred["logits"].float().cpu() - skip_pred["logits"].float().cpu()
            rows.append({
                "step": step, "type": "refresh" if step % R == 0 else "reuse",
                "source_step": source["step"] if step % R else None, "source_age": age,
                "foreground_tokens": int(fg.sum()), "background_tokens": int((~fg).sum()),
                "mask_sha256": hashlib.sha256(fg.tobytes()).hexdigest(),
                "overwrite_k_rows_per_layer": full_rows,
                "overwrite_v_rows_per_layer": full_rows,
                "true_skip_k_rows_per_layer": [x["k_projected_rows"] for x in row_detail],
                "true_skip_v_rows_per_layer": [x["v_projected_rows"] for x in row_detail],
                "true_skip_reused_k_rows_per_layer": [x["k_reused_background_rows"] for x in row_detail],
                "true_skip_reused_v_rows_per_layer": [x["v_reused_background_rows"] for x in row_detail],
                "all_32_layers_patched": step % R == 0 or sorted(sk.diag["layers"]) == list(LAYERS),
                "decode_calls_delegated": 0 if step % R == 0 else len(sk.diag["decode_rows_by_layer"]),
                "unanticipated_fallback_count": 0 if step % R == 0 else len(sk.diag["unexpected_fallbacks"]),
                "same_cache_shapes": True, "background_kv_provenance_matches": True,
                "all_actions_finite": True,
                "action_token_ids_equal": overwrite_pred["action_token_ids"] == skip_pred["action_token_ids"],
                "max_abs_policy_action_difference": float(np.max(np.abs(action_delta))),
                "policy_action_l2_difference": float(np.linalg.norm(action_delta)),
                "logits_max_abs_difference": float(logit_delta.abs().max()),
                "logits_mean_abs_difference": float(logit_delta.abs().mean()),
                "logits_relative_l2_difference": float(torch.linalg.vector_norm(logit_delta) / torch.linalg.vector_norm(overwrite_pred["logits"].float().cpu()).clamp_min(1e-20)),
                "validation_passed": True,
            })
            # The skip result drives the short validation environment trajectory.
            action = normalize_and_invert_openvla_gripper(skip_pred["policy_action"])
            obs, _, done, _ = env.step(action.tolist())
            if done: break
    finally:
        env.close()
    result = {"task_id": 4, "state_id": 0, "seed": SEED, "steps": len(rows), "records": rows,
              "all_valid": len(rows) == 6 and all(x["validation_passed"] for x in rows),
              "layers": list(LAYERS), "refresh_interval": R, "max_source_age": max(x["source_age"] for x in rows),
              "validation_type": "same observation/source compute-overwrite vs K/V true skip"}
    assert result["all_valid"] and result["max_source_age"] == 4
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(result, indent=2))
    print("[VALIDATION_PASS] six steps; 32 layers; Q full; foreground K/V fresh; background K/V sourced from fresh refresh; exact source provenance; decode delegated", flush=True)
    return result


def validate_artifact(path: Path):
    result = json.loads(path.read_text())
    assert result["config"]["R"] == R and result["config"]["layers"] == list(LAYERS)
    assert len(result["scenarios"]) == 9
    for scenario in result["scenarios"]:
        for condition in ("fresh", "stage3b_true_skip"):
            rollout = scenario[condition]
            assert rollout["all_valid"] and rollout["steps"] == len(rollout["records"])
            for row in rollout["records"]:
                assert row["foreground_tokens"] + row["background_tokens"] == 256
                assert np.all(np.isfinite(row["policy_action"]))
            if condition == "stage3b_true_skip":
                for row in rollout["records"]:
                    reuse = row["step"] % R != 0
                    assert (row["type"] == "reuse") == reuse
                    assert row["source_age"] == row["step"] % R
                    if reuse:
                        assert row["source_step"] == row["step"] - row["step"] % R
                        assert all(r == 1 + row["foreground_tokens"] + row["language_tokens"] for r in row["projected_k_rows_by_layer"])
                        assert row["projected_k_rows_by_layer"] == row["projected_v_rows_by_layer"]
                        assert all(r == row["background_tokens"] for r in row["skipped_k_rows_by_layer"] + row["skipped_v_rows_by_layer"])
    return result


def run_matrix(model, processor, device, key):
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = []
    (OUT / "summary.json").write_text(json.dumps({"status": "running", "scenarios_completed": 0}, indent=2))
    for task in TASKS:
        for state in STATES:
            scenario = {"task_id": task, "state_id": state,
                        "config": {"suite": TASK_SUITE_NAME, "seed": SEED, "horizon": HORIZON, "R": R,
                                   "layers": list(LAYERS), "visual_positions": [1, 256], "components": ["K", "V"],
                                   "compute_skip": "background visual K and V only", "preprocessing": "Stage 3A exact path", "settling_steps": 10},
                        "fresh": run_rollout("fresh", task, state, model, processor, device, key),
                        "stage3b_true_skip": run_rollout("stage3b_true_skip", task, state, model, processor, device, key)}
            (OUT / f"task{task}_state{state}").mkdir(parents=True, exist_ok=True)
            (OUT / f"task{task}_state{state}" / "results.json").write_text(json.dumps(scenario, indent=2))
            scenarios.append(scenario)
            summary = {"status": "running", "scenarios_completed": len(scenarios),
                       "results": [{"task": x["task_id"], "state": x["state_id"],
                                    "fresh_success": x["fresh"]["success"], "fresh_steps": x["fresh"]["steps"],
                                    "stage3b_success": x["stage3b_true_skip"]["success"], "stage3b_steps": x["stage3b_true_skip"]["steps"],
                                    "refresh": x["stage3b_true_skip"]["refresh_count"], "reuse": x["stage3b_true_skip"]["reuse_count"],
                                    "reuse_fraction": x["stage3b_true_skip"]["reuse_fraction"],
                                    "skipped_k_rows": x["stage3b_true_skip"]["total_skipped_k_projection_rows"],
                                    "skipped_v_rows": x["stage3b_true_skip"]["total_skipped_v_projection_rows"],
                                    "valid": x["stage3b_true_skip"]["all_valid"]} for x in scenarios]}
            (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"[SCENARIO_COMPLETE] task={task} state={state} fresh={scenario['fresh']['success']}/{scenario['fresh']['steps']} stage3b={scenario['stage3b_true_skip']['success']}/{scenario['stage3b_true_skip']['steps']}", flush=True)
    fresh_successes = sum(bool(x["fresh"]["success"]) for x in scenarios)
    reuse_successes = sum(bool(x["stage3b_true_skip"]["success"]) for x in scenarios)
    paired_fresh = [x for x in scenarios if x["fresh"]["success"]]
    final = {"status": "complete", "config": {"suite": TASK_SUITE_NAME, "tasks": list(TASKS), "states": list(STATES),
             "seed": SEED, "horizon": HORIZON, "R": R, "layers": list(LAYERS), "visual_positions": [1, 256],
             "components": ["K", "V"], "latency_measured": False},
             "aggregate": {"fresh_successes": fresh_successes, "fresh_total": 9,
                           "stage3b_successes": reuse_successes, "stage3b_total": 9,
                           "stage3b_successes_among_fresh_successes": sum(bool(x["stage3b_true_skip"]["success"]) for x in paired_fresh),
                           "fresh_success_scenarios": len(paired_fresh),
                           "total_skipped_k_rows": sum(x["stage3b_true_skip"]["total_skipped_k_projection_rows"] for x in scenarios),
                           "total_skipped_v_rows": sum(x["stage3b_true_skip"]["total_skipped_v_projection_rows"] for x in scenarios),
                           "latency_measured": False},
             "scenarios": scenarios}
    (OUT / "summary.json").write_text(json.dumps(final, indent=2))
    validate_artifact(OUT / "summary.json")
    return final


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--matrix", action="store_true")
    args = parser.parse_args()
    model, processor, device = load_model_and_processor(DEFAULT_MODEL_ID)
    key = resolve_unnorm_key(model, None)
    if args.validate_only:
        run_validation(model, processor, device, key)
    else:
        run_matrix(model, processor, device, key)

if __name__ == "__main__":
    main()
