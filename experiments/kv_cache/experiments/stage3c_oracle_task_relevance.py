"""Stage 3C: Stage-3B oracle background reuse with task-relevance protection.

Attention relevance is captured as a read-only Q/K sidecar during the normal,
fully-fresh layer-15 refresh prefill. The model's configured attention backend
and policy output are not changed. Relevance scores are held for the following
R=5 window and rank only current oracle-background patches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from experiments.kv_analysis.experiments.extract_libero_image import (
    DEFAULT_STEP_ACTION, normalize_and_invert_openvla_gripper,
)
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import (
    assert_independent_cache_clone, capture_prediction, policy_image_from_observation, seed_everything,
)
from experiments.kv_cache.experiments.oracle_foreground_mask import rendered_label_mapping
from experiments.kv_cache.experiments.stage3a_masked_visual_k_validation import mask_from_obs
from experiments.kv_cache.experiments.stage3b_oracle_full_kv_matrix import (
    LAYERS, HORIZON, R, SEED, TASKS, STATES, MaskedKVProjectionSkip, make_env_and_task,
)
from experiments.kv_cache.experiments.stage3e_attention_relevance import protect_top_background, text_to_visual_scores
from experiments.kv_cache.experiments.static_reuse_experiment import (
    DEFAULT_MODEL_ID, clone_prefill_cache, load_model_and_processor, resolve_unnorm_key,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage3c_oracle_task_relevance"
FRACTIONS = (0.10, 0.25, 0.50)
RELEVANCE_LAYER = 15  # The VLA-Cache OpenVLA utility's default scoring layer.


class RefreshAttentionSidecar:
    """Capture layer-15 text-query -> visual-key attention without changing inference.

    Existing q_proj/k_proj module outputs are captured by hooks. After the
    original LlamaAttention.forward has completed, this sidecar applies the
    same RoPE position embeddings, the module's attention scaling, and the
    causal/additive mask to compute an FP32 diagnostic attention distribution.
    It is not fed back into the model's normal SDPA attention output.
    """

    def __init__(self, model: Any, layout_hint: dict[str, Any] | None = None):
        self.model = model
        self.layer = model.language_model.model.layers[RELEVANCE_LAYER].self_attn
        self.layout_hint = layout_hint
        self.original_forward = None
        self.q_handle = self.k_handle = None
        self.scores: np.ndarray | None = None
        self.query_positions: list[int] = []
        self.sequence_length: int | None = None
        self.capture_ms: float | None = None
        self.peak_memory_delta_bytes: int | None = None
        self.axis_shape: list[int] | None = None
        self.row_sum_max_error: float | None = None
        self.backend = getattr(self.layer.config, "_attn_implementation", "unknown")

    def __enter__(self):
        if self.backend != "sdpa":
            raise RuntimeError(f"Expected unchanged SDPA backend, found {self.backend!r}.")
        if self.layer.training:
            raise RuntimeError("Attention sidecar requires eval mode.")
        self.original_forward = self.layer.forward
        captured: dict[str, torch.Tensor] = {}

        def q_hook(_module, _inputs, output):
            captured["q"] = output

        def k_hook(_module, _inputs, output):
            captured["k"] = output

        self.q_handle = self.layer.q_proj.register_forward_hook(q_hook)
        self.k_handle = self.layer.k_proj.register_forward_hook(k_hook)

        def wrapped(module, hidden_states, position_embeddings, attention_mask,
                    past_key_value=None, cache_position=None, **kwargs):
            if hidden_states.shape[-2] <= 1:
                return self.original_forward(hidden_states, position_embeddings, attention_mask,
                                             past_key_value, cache_position, **kwargs)
            if past_key_value is not None:
                get_length = getattr(past_key_value, "get_seq_length", None)
                # DynamicCache grows layer-by-layer; by layer 15 its default
                # (layer-0) length already reflects earlier layers. Inspect
                # this attention module's own slot to distinguish a fresh
                # prefill from an appended decode call.
                try:
                    incoming_length = None if get_length is None else int(get_length(RELEVANCE_LAYER))
                except TypeError:
                    incoming_length = None if get_length is None else int(get_length())
                if incoming_length not in (0,):
                    raise RuntimeError(
                        "Task-relevance scores must be captured only on fresh refresh prefills; "
                        f"layer-{RELEVANCE_LAYER} cache sequence_length={incoming_length}."
                    )
            captured.clear()
            result = self.original_forward(hidden_states, position_embeddings, attention_mask,
                                           past_key_value, cache_position, **kwargs)
            if set(captured) != {"q", "k"}:
                raise RuntimeError("Normal attention forward did not invoke both q_proj and k_proj hooks.")
            torch.cuda.synchronize(hidden_states.device)
            baseline_memory = torch.cuda.memory_allocated(hidden_states.device)
            torch.cuda.reset_peak_memory_stats(hidden_states.device)
            start_wall = time.perf_counter()
            self.scores = self._compute(captured["q"], captured["k"], position_embeddings,
                                        attention_mask, hidden_states.shape[-2])
            torch.cuda.synchronize(hidden_states.device)
            self.capture_ms = (time.perf_counter() - start_wall) * 1000.0
            self.peak_memory_delta_bytes = max(
                int(torch.cuda.max_memory_allocated(hidden_states.device) - baseline_memory), 0
            )
            return result

        self.layer.forward = types.MethodType(wrapped, self.layer)
        return self

    def __exit__(self, *_):
        self.layer.forward = self.original_forward
        if self.q_handle is not None:
            self.q_handle.remove()
        if self.k_handle is not None:
            self.k_handle.remove()

    @staticmethod
    def _mask_rows(attention_mask, query_positions: torch.Tensor, seq_len: int, device: torch.device):
        key_positions = torch.arange(seq_len, device=device)
        causal = key_positions[None, :] <= query_positions[:, None]
        allowed = causal[None, None, :, :]
        if attention_mask is None:
            return allowed
        mask = attention_mask
        if mask.dtype == torch.bool:
            if mask.ndim == 4:
                selected = mask.index_select(-2, query_positions)
                if selected.shape[-1] != seq_len:
                    raise RuntimeError(f"Boolean attention mask key dimension {selected.shape[-1]} != {seq_len}.")
                return allowed & selected
            if mask.ndim == 2:
                key_allowed = mask[:, None, None, :]
                return allowed & key_allowed
            raise RuntimeError(f"Unsupported boolean attention-mask shape {tuple(mask.shape)}.")
        if not mask.is_floating_point():
            raise RuntimeError(f"Unsupported attention-mask dtype {mask.dtype}.")
        if mask.ndim != 4:
            raise RuntimeError(f"Expected additive 4D attention mask, got {tuple(mask.shape)}.")
        selected = mask.index_select(-2, query_positions)
        if selected.shape[-1] != seq_len:
            raise RuntimeError(f"Additive attention mask key dimension {selected.shape[-1]} != {seq_len}.")
        return allowed & torch.isfinite(selected)

    def _compute(self, q_raw: torch.Tensor, k_raw: torch.Tensor, position_embeddings,
                 attention_mask, seq_len: int) -> np.ndarray:
        batch, q_len, _ = q_raw.shape
        if q_len != seq_len or k_raw.shape[1] != seq_len:
            raise RuntimeError("Captured Q/K rows do not match full multimodal prefill length.")
        # The actual prompt layout is checked by Stage 3B's dynamic layout
        # derivation; text query rows must follow the complete 256-key block.
        text_start, text_end = 257, seq_len - 1
        if text_start > text_end:
            raise RuntimeError(f"No prompt query positions after visual block in length {seq_len}.")
        q_pos = torch.arange(text_start, text_end + 1, dtype=torch.long, device=q_raw.device)
        self.query_positions = q_pos.detach().cpu().tolist()
        head_dim = self.layer.head_dim
        num_q_heads = q_raw.shape[-1] // head_dim
        num_kv_heads = k_raw.shape[-1] // head_dim
        if num_q_heads % num_kv_heads:
            raise RuntimeError("Q/K head counts are incompatible for grouped-query attention.")
        q = q_raw.view(batch, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k = k_raw.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, num_q_heads // num_kv_heads)
        q = q.index_select(-2, q_pos).float()
        k = k.float()
        logits = torch.matmul(q, k.transpose(-1, -2)) * float(self.layer.scaling)
        valid = self._mask_rows(attention_mask, q_pos, seq_len, q.device)
        # causal mask remains explicit even when the normal SDPA path uses
        # is_causal=True with no materialized mask.
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits, dim=-1)
        row_sums = probabilities.sum(dim=-1)
        self.row_sum_max_error = float((row_sums - 1.0).abs().max().item())
        if not torch.isfinite(probabilities).all() or self.row_sum_max_error > 1e-5:
            raise RuntimeError(f"Invalid attention sidecar probabilities; max row-sum error={self.row_sum_max_error}.")
        self.sequence_length = seq_len
        self.axis_shape = list(probabilities.shape)
        compact_queries = np.arange(len(self.query_positions), dtype=np.int64)
        cpu_probs = probabilities.detach().to(device="cpu", dtype=torch.float32).numpy()
        scores = text_to_visual_scores(
            [cpu_probs], compact_queries, query_token_positions=self.query_positions,
            visual_start=1, visual_count=256,
        )
        if scores.shape != (256,) or not np.isfinite(scores).all():
            raise RuntimeError("Attention aggregation did not produce 256 finite visual scores.")
        return scores


def save_score_visualization(path: Path, image: np.ndarray, oracle_fg: np.ndarray,
                             protected_bg: np.ndarray, reusable_bg: np.ndarray, scores: np.ndarray):
    """Save a real observation plus oracle/protected/reusable 16x16 patch maps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = Image.fromarray(image.astype(np.uint8)).convert("RGB").resize((256, 256))
    panels = [rgb]
    arrays = []
    for mask, fg_color, bg_color in ((oracle_fg, (220, 50, 50), (45, 45, 45)),
                                     (protected_bg, (255, 180, 0), (30, 30, 30)),
                                     (reusable_bg, (50, 200, 90), (30, 30, 30))):
        grid = np.zeros((16, 16, 3), dtype=np.uint8)
        flat = grid.reshape(256, 3)
        flat[:] = np.where(mask[:, None], np.array(fg_color, dtype=np.uint8), np.array(bg_color, dtype=np.uint8))
        arrays.append(grid)
    score_norm = (scores - scores.min()) / max(float(scores.max() - scores.min()), 1e-12)
    heat = np.zeros((16, 16, 3), dtype=np.uint8)
    heat[..., 0] = (255 * score_norm.reshape(16, 16)).astype(np.uint8)
    heat[..., 1] = (180 * (1.0 - score_norm.reshape(16, 16))).astype(np.uint8)
    heat[..., 2] = 40
    arrays.append(heat)
    labels = ["RGB", "Oracle foreground", "Protected background", "Reusable background", "Text-to-visual relevance"]
    for grid in arrays:
        panels.append(Image.fromarray(grid).resize((256, 256), Image.Resampling.NEAREST))
    canvas = Image.new("RGB", (len(panels) * 256, 286), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 256, 26))
        draw.text((index * 256 + 4, 5), labels[index], fill=(0, 0, 0))
    canvas.save(path)


def run_rollout(task_id: int, state_id: int, fraction: float, model, processor, device, key):
    seed_everything(SEED)
    task, states, env = make_env_and_task(task_id)
    outdir = OUT / f"protect_{int(fraction * 100):02d}" / f"task{task_id}_state{state_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    records, source, done, visualization_written = [], None, False, False
    try:
        env.reset()
        obs = env.set_init_state(states[state_id])
        mapping = rendered_label_mapping(env.env.model.instances_to_ids)
        for _ in range(10):
            obs, _, _, _ = env.step(DEFAULT_STEP_ACTION)
        for step in range(HORIZON):
            oracle_fg = mask_from_obs(obs, mapping)
            image = policy_image_from_observation(obs, True)
            refresh = step % R == 0
            if refresh:
                sidecar = RefreshAttentionSidecar(model)
                pred = capture_prediction(model, processor, image, prompt, device, key, sidecar)
                if sidecar.scores is None or sidecar.sequence_length != pred["prefill_length"]:
                    raise RuntimeError("Refresh failed to capture aligned text-to-visual attention scores.")
                protected_bg, reusable_bg = protect_top_background(sidecar.scores, oracle_fg, fraction)
                source_cache = clone_prefill_cache(pred["cache"])
                assert_independent_cache_clone(source_cache, pred["cache"])
                source = {"cache": source_cache, "layout": dict(pred["layout"]), "step": step,
                          "fresh": True, "scores": sidecar.scores.copy(), "score_age": 0}
                projected_k = projected_v = pred["prefill_length"]
                reused_tokens = 0
                projection_by_layer = [{"layer": layer, "k_projected_rows": pred["prefill_length"],
                                        "v_projected_rows": pred["prefill_length"],
                                        "k_reused_background_rows": 0, "v_reused_background_rows": 0,
                                        "k_skipped_background_rows": 0, "v_skipped_background_rows": 0}
                                       for layer in LAYERS]
                source_step, age = step, 0
                mask_selection_ms = 0.0
                cache_selection_ms = 0.0
                attention_capture_ms = sidecar.capture_ms
                attention_peak_bytes = sidecar.peak_memory_delta_bytes
                score_shape, query_positions = sidecar.axis_shape, sidecar.query_positions
                if not visualization_written:
                    save_score_visualization(outdir / "refresh_step_000_relevance.png", image, oracle_fg,
                                             protected_bg, reusable_bg, sidecar.scores)
                    visualization_written = True
            else:
                expected_source = step - step % R
                if source is None or source["step"] != expected_source or not source["fresh"]:
                    raise AssertionError("Reuse source is not the latest fully fresh R5 refresh snapshot.")
                selection_start = time.perf_counter()
                protected_bg, reusable_bg = protect_top_background(source["scores"], oracle_fg, fraction)
                mask_selection_ms = (time.perf_counter() - selection_start) * 1000.0
                cache_start = time.perf_counter()
                effective_fg = oracle_fg | protected_bg
                cache_selection_ms = (time.perf_counter() - cache_start) * 1000.0
                context = MaskedKVProjectionSkip(model, source["cache"], source["layout"], effective_fg)
                pred = capture_prediction(model, processor, image, prompt, device, key, context)
                context.validate(pred["cache"])
                projection_by_layer = context.diag["rows_by_layer"]
                reused_tokens = int(reusable_bg.sum())
                projected_k = projection_by_layer[0]["k_projected_rows"]
                projected_v = projection_by_layer[0]["v_projected_rows"]
                source_step, age = source["step"], step - source["step"]
                if age not in (1, 2, 3, 4):
                    raise AssertionError(f"R5 source age out of range: {age}.")
                attention_capture_ms = None
                attention_peak_bytes = None
                score_shape, query_positions = None, None
            expected_projected = 1 + int((oracle_fg | protected_bg).sum()) + (pred["layout"]["language"][1] - pred["layout"]["language"][0] + 1)
            if refresh:
                expected_projected = pred["prefill_length"]
            if any(row["k_projected_rows"] != expected_projected or row["v_projected_rows"] != expected_projected
                   for row in projection_by_layer):
                raise AssertionError("Observed projected row count disagrees with combined oracle/relevance mask.")
            if not np.all(np.isfinite(pred["policy_action"])):
                raise AssertionError("Non-finite policy action.")
            action = normalize_and_invert_openvla_gripper(pred["policy_action"])
            skipped_each_layer = reused_tokens if not refresh else 0
            record = {
                "step": step, "type": "refresh" if refresh else "reuse", "source_step": source_step,
                "source_age": age, "attention_score_age": age,
                "prefill_length": pred["prefill_length"],
                "language_tokens": pred["layout"]["language"][1] - pred["layout"]["language"][0] + 1,
                "foreground_tokens": int(oracle_fg.sum()), "background_tokens": int((~oracle_fg).sum()),
                "task_relevant_background_protected": int(protected_bg.sum()),
                "background_tokens_reused": reused_tokens,
                "protected_background_indices": np.flatnonzero(protected_bg).tolist(),
                "reusable_background_indices": np.flatnonzero(reusable_bg).tolist(),
                "relevance_score_mean": float(source["scores"].mean()),
                "relevance_score_min": float(source["scores"].min()),
                "relevance_score_max": float(source["scores"].max()),
                "relevance_scores": source["scores"].tolist() if refresh else None,
                "relevance_attention_shape": score_shape,
                "relevance_query_positions": query_positions,
                "attention_capture_ms": attention_capture_ms,
                "attention_sidecar_peak_memory_bytes": attention_peak_bytes,
                "mask_selection_ms": mask_selection_ms,
                "cache_selection_ms": cache_selection_ms,
                "effective_fresh_visual_tokens": int((oracle_fg | protected_bg).sum()),
                "projected_k_rows_per_layer": [x["k_projected_rows"] for x in projection_by_layer],
                "projected_v_rows_per_layer": [x["v_projected_rows"] for x in projection_by_layer],
                "skipped_k_rows_per_layer": [x["k_skipped_background_rows"] for x in projection_by_layer],
                "skipped_v_rows_per_layer": [x["v_skipped_background_rows"] for x in projection_by_layer],
                "total_k_rows_skipped": skipped_each_layer * len(LAYERS),
                "total_v_rows_skipped": skipped_each_layer * len(LAYERS),
                "all_32_layers_validated": refresh or sorted(context.diag["layers"]) == list(LAYERS),
                "q_full_prefill": True, "foreground_fresh": True, "protected_background_fresh": True,
                "unprotected_background_reused": True, "source_is_fresh_refresh": bool(source["fresh"]),
                "decode_normal": refresh or len(context.diag["decode_rows_by_layer"]) == 32 * 6,
                "finite_action": True, "action_token_ids": pred["action_token_ids"],
                "policy_action": pred["policy_action"].tolist(), "environment_action": action.tolist(),
            }
            if not all((record["all_32_layers_validated"], record["q_full_prefill"], record["foreground_fresh"],
                        record["protected_background_fresh"], record["unprotected_background_reused"],
                        record["source_is_fresh_refresh"], record["decode_normal"], record["finite_action"])):
                raise AssertionError("Runtime validation failed.")
            obs, _, done, _ = env.step(action.tolist())
            record["done"] = bool(done)
            records.append(record)
            partial = {
                "status": "running", "task_id": task_id, "state_id": state_id, "protection_fraction": fraction,
                "steps_completed": len(records), "records": records,
            }
            (outdir / "results.partial.json").write_text(json.dumps(partial, indent=2))
            if done:
                break
    finally:
        env.close()
    refresh_rows = [r for r in records if r["type"] == "refresh"]
    reuse_rows = [r for r in records if r["type"] == "reuse"]
    final = {
        "status": "complete", "condition": "oracle_bg_plus_task_relevance_full_layer_kv_true_skip",
        "task_id": task_id, "task_description": task.language, "state_id": state_id, "seed": SEED,
        "horizon": HORIZON, "success": bool(done), "steps": len(records),
        "termination": "success" if done else "horizon", "protection_fraction": fraction,
        "layers": list(LAYERS), "visual_positions": [1, 256], "components": ["K", "V"], "R": R,
        "refresh_count": len(refresh_rows), "reuse_count": len(reuse_rows),
        "reuse_fraction_decisions": len(reuse_rows) / len(records) if records else 0.0,
        "max_source_age": max((r["source_age"] for r in records), default=0),
        "source_age_counts": {str(i): sum(r["source_age"] == i for r in records) for i in range(R)},
        "all_valid": len(records) > 0 and all(r["all_32_layers_validated"] and r["decode_normal"] and r["finite_action"] for r in records),
        "total_k_rows_skipped": sum(r["total_k_rows_skipped"] for r in records),
        "total_v_rows_skipped": sum(r["total_v_rows_skipped"] for r in records),
        "mean_protected_background_tokens": float(np.mean([r["task_relevant_background_protected"] for r in records])) if records else 0.0,
        "mean_reused_background_tokens": float(np.mean([r["background_tokens_reused"] for r in records])) if records else 0.0,
        "mean_attention_capture_ms_refresh": float(np.mean([r["attention_capture_ms"] for r in refresh_rows if r["attention_capture_ms"] is not None])) if refresh_rows else None,
        "mean_mask_selection_ms_reuse": float(np.mean([r["mask_selection_ms"] for r in reuse_rows])) if reuse_rows else None,
        "mean_cache_selection_ms_reuse": float(np.mean([r["cache_selection_ms"] for r in reuse_rows])) if reuse_rows else None,
        "max_attention_sidecar_peak_memory_bytes": max((r["attention_sidecar_peak_memory_bytes"] or 0 for r in records), default=0),
        "latency_measured": False, "records": records,
    }
    (outdir / "results.json").write_text(json.dumps(final, indent=2))
    partial_path = outdir / "results.partial.json"
    if partial_path.exists():
        partial_path.unlink()
    return final


def validate_action_invariance(model, processor, device, key):
    """Check that installing/removing the scoring sidecar does not change output."""
    seed_everything(SEED)
    task, states, env = make_env_and_task(4)
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    try:
        env.reset(); obs = env.set_init_state(states[0])
        mapping = rendered_label_mapping(env.env.model.instances_to_ids)
        for _ in range(10): obs, _, _, _ = env.step(DEFAULT_STEP_ACTION)
        image = policy_image_from_observation(obs, True)
        base = capture_prediction(model, processor, image, prompt, device, key)
        sidecar = RefreshAttentionSidecar(model)
        scored = capture_prediction(model, processor, image, prompt, device, key, sidecar)
        assert sidecar.scores is not None and sidecar.scores.shape == (256,)
        assert sidecar.backend == "sdpa" and sidecar.row_sum_max_error is not None and sidecar.row_sum_max_error <= 1e-5
        if base["action_token_ids"] != scored["action_token_ids"]:
            raise AssertionError("Attention capture changed greedy action-token IDs.")
        if not np.array_equal(base["policy_action"], scored["policy_action"]):
            raise AssertionError("Attention capture changed decoded policy actions.")
        if not torch.equal(base["logits"], scored["logits"]):
            raise AssertionError("Attention capture changed prefill logits.")
        source_cache = clone_prefill_cache(scored["cache"])
        assert_independent_cache_clone(source_cache, scored["cache"])
        obs, _, _, _ = env.step(normalize_and_invert_openvla_gripper(scored["policy_action"]).tolist())
        current_fg = mask_from_obs(obs, mapping)
        protected_bg, reusable_bg = protect_top_background(sidecar.scores, current_fg, 0.25)
        if np.any(protected_bg & current_fg) or not np.all((current_fg | protected_bg)[current_fg]):
            raise AssertionError("Foreground protection mask overlaps/omits oracle foreground.")
        if int(protected_bg.sum()) == 0 or int(reusable_bg.sum()) == 0:
            raise AssertionError("Validation frame did not exercise both protected and reused background tokens.")
        context = MaskedKVProjectionSkip(model, source_cache, scored["layout"], current_fg | protected_bg)
        masked = capture_prediction(model, processor, policy_image_from_observation(obs, True), prompt,
                                    device, key, context)
        context.validate(masked["cache"])
        expected_rows = 1 + int((current_fg | protected_bg).sum()) + (masked["layout"]["language"][1] - masked["layout"]["language"][0] + 1)
        if any(row["k_projected_rows"] != expected_rows or row["v_projected_rows"] != expected_rows
               or row["k_reused_background_rows"] != int(reusable_bg.sum())
               or row["v_reused_background_rows"] != int(reusable_bg.sum())
               for row in context.diag["rows_by_layer"]):
            raise AssertionError("Masked true-skip row counts do not match foreground/relevance mask.")
        if len(context.diag["layers"]) != 32 or len(context.diag["decode_rows_by_layer"]) != 32 * 6:
            raise AssertionError("Not all layers or one-token decode calls were validated.")
        if not np.all(np.isfinite(masked["policy_action"])):
            raise AssertionError("Masked reuse produced a non-finite action.")
        return {"passed": True, "task_id": 4, "state_id": 0, "prefill_length": scored["prefill_length"],
                "visual_scores": int(sidecar.scores.size), "attention_shape": sidecar.axis_shape,
                "query_positions": [sidecar.query_positions[0], sidecar.query_positions[-1]],
                "backend": sidecar.backend, "probability_row_sum_max_error": sidecar.row_sum_max_error,
                "token_ids_equal": True, "policy_action_exact": True, "prefill_logits_exact": True,
                "attention_capture_ms": sidecar.capture_ms,
                "masked_reuse_validation": {"source_step": 0, "source_age": 1,
                    "oracle_foreground_tokens": int(current_fg.sum()), "oracle_background_tokens": int((~current_fg).sum()),
                    "task_relevant_background_protected": int(protected_bg.sum()),
                    "background_tokens_reused": int(reusable_bg.sum()), "layers_patched": len(context.diag["layers"]),
                    "k_projected_rows_per_layer": expected_rows,
                    "k_rows_reused_per_layer": int(reusable_bg.sum()),
                    "v_projected_rows_per_layer": expected_rows,
                    "v_rows_reused_per_layer": int(reusable_bg.sum()),
                    "decode_calls_delegated": len(context.diag["decode_rows_by_layer"]),
                    "cache_validation": "pass", "finite_action": True},
                "caveat": "FP32 sidecar probabilities reconstructed from normal-pass Q/K; SDPA outputs are unchanged."}
    finally:
        env.close()


def validate_summary(path: Path):
    summary = json.loads(path.read_text())
    if summary.get("status") != "complete" or len(summary.get("scenarios", [])) != 9:
        raise ValueError("Threshold summary incomplete.")
    for result in summary["scenarios"]:
        if result["steps"] != len(result["records"]) or not result["all_valid"]:
            raise ValueError("Scenario record counts or validation status invalid.")
        for r in result["records"]:
            if r["foreground_tokens"] + r["background_tokens"] != 256:
                raise ValueError("Oracle mask does not cover 256 visual tokens.")
            if r["type"] == "reuse" and (r["source_age"] not in (1, 2, 3, 4) or not r["source_is_fresh_refresh"]):
                raise ValueError("Reuse source age/provenance invalid.")
    return summary


def run_threshold(model, processor, device, key, fraction: float):
    outdir = OUT / f"protect_{int(fraction * 100):02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps({"status": "running", "fraction": fraction, "completed": 0}, indent=2))
    scenarios = []
    for task_id in TASKS:
        for state_id in STATES:
            scenario_path = outdir / f"task{task_id}_state{state_id}" / "results.json"
            if scenario_path.exists():
                candidate = json.loads(scenario_path.read_text())
                if candidate.get("status") == "complete" and candidate.get("all_valid"):
                    scenario = candidate
                else:
                    scenario = run_rollout(task_id, state_id, fraction, model, processor, device, key)
            else:
                scenario = run_rollout(task_id, state_id, fraction, model, processor, device, key)
            # Attach matched existing Stage 3B and Fresh outcomes, without reruns.
            stage3b_path = ROOT / "outputs" / "stage3b_oracle_full_kv_matrix" / f"task{task_id}_state{state_id}" / "results.json"
            if not stage3b_path.exists():
                raise FileNotFoundError(f"Missing validated Stage 3B scenario: {stage3b_path}")
            stage3b = json.loads(stage3b_path.read_text())
            scenario["stage3b_reference"] = {"success": stage3b["stage3b_true_skip"]["success"],
                                               "steps": stage3b["stage3b_true_skip"]["steps"]}
            scenario["fresh_reference"] = {"success": stage3b["fresh"]["success"], "steps": stage3b["fresh"]["steps"]}
            scenario_path.write_text(json.dumps(scenario, indent=2))
            scenarios.append(scenario)
            (outdir / "summary.json").write_text(json.dumps({
                "status": "running", "fraction": fraction, "completed": len(scenarios),
                "scenarios": [{"task_id": x["task_id"], "state_id": x["state_id"],
                               "success": x["success"], "steps": x["steps"], "all_valid": x["all_valid"]}
                              for x in scenarios]}, indent=2))
            print(f"[SCENARIO_COMPLETE] fraction={fraction:.2f} task={task_id} state={state_id} success={scenario['success']} steps={scenario['steps']}", flush=True)
    final = {
        "status": "complete", "experiment": "Stage 3C — Oracle Background + Task Relevance, Full-Layer K+V, R5",
        "fraction": fraction, "score_layer": RELEVANCE_LAYER,
        "scoring_method": "FP32 sidecar attention from layer-15 normal-pass RoPE-applied Q/K; mean heads and prompt-text queries; visual key positions 1..256; previous fresh refresh scores used for reuse window",
        "custom_protection_budget": True,
        "config": {"tasks": list(TASKS), "states": list(STATES), "seed": SEED, "horizon": HORIZON,
                   "R": R, "layers": list(LAYERS), "components": ["K", "V"], "visual_positions": [1, 256],
                   "attention_backend": "SDPA unchanged", "latency_measured": False},
        "aggregate": {"successes": sum(bool(x["success"]) for x in scenarios), "total": 9,
                      "successes_among_stage3b_successes": sum(bool(x["success"]) for x in scenarios if x["stage3b_reference"]["success"]),
                      "stage3b_successes": sum(bool(x["stage3b_reference"]["success"]) for x in scenarios),
                      "total_skipped_k_rows": sum(x["total_k_rows_skipped"] for x in scenarios),
                      "total_skipped_v_rows": sum(x["total_v_rows_skipped"] for x in scenarios),
                      "latency_measured": False},
        "validation": {"action_invariance": ACTION_INVARIANCE, "all_scenarios_valid": all(x["all_valid"] for x in scenarios)},
        "scenarios": scenarios,
    }
    path = outdir / "summary.json"
    path.write_text(json.dumps(final, indent=2))
    validate_summary(path)
    return final


def finalize_outputs() -> dict[str, Any]:
    """Create cross-threshold CSVs, plots and a concise report from saved JSON."""
    threshold_summaries = []
    for fraction in FRACTIONS:
        path = OUT / f"protect_{int(fraction * 100):02d}" / "summary.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("status") == "complete":
                validate_summary(path)
                threshold_summaries.append(data)
    complete = len(threshold_summaries) == len(FRACTIONS)
    scenario_rows, relevance_rows, projection_rows = [], [], []
    threshold_rows = []
    for threshold in threshold_summaries:
        pct = int(threshold["fraction"] * 100)
        scenario_rows.extend({
            "protection_percent": pct, "task_id": x["task_id"], "state_id": x["state_id"],
            "fresh_success": x["fresh_reference"]["success"], "fresh_steps": x["fresh_reference"]["steps"],
            "stage3b_success": x["stage3b_reference"]["success"], "stage3b_steps": x["stage3b_reference"]["steps"],
            "stage3c_success": x["success"], "stage3c_steps": x["steps"],
            "refresh_count": x["refresh_count"], "reuse_count": x["reuse_count"],
            "reuse_fraction_decisions": x["reuse_fraction_decisions"],
            "mean_protected_background_tokens": x["mean_protected_background_tokens"],
            "mean_reused_background_tokens": x["mean_reused_background_tokens"],
            "max_source_age": x["max_source_age"], "all_valid": x["all_valid"],
            "skipped_k_rows": x["total_k_rows_skipped"], "skipped_v_rows": x["total_v_rows_skipped"],
        } for x in threshold["scenarios"])
        for scenario in threshold["scenarios"]:
            projection_rows.append({"protection_percent": pct, "task_id": scenario["task_id"],
                                    "state_id": scenario["state_id"],
                                    "skipped_k_rows": scenario["total_k_rows_skipped"],
                                    "skipped_v_rows": scenario["total_v_rows_skipped"],
                                    "combined_kv_rows_skipped": scenario["total_k_rows_skipped"] + scenario["total_v_rows_skipped"],
                                    "mean_protected_background_tokens_per_step": scenario["mean_protected_background_tokens"],
                                    "mean_reused_background_tokens_per_step": scenario["mean_reused_background_tokens"]})
            for record in scenario["records"]:
                if record["type"] == "refresh":
                    relevance_rows.append({"protection_percent": pct, "task_id": scenario["task_id"],
                                          "state_id": scenario["state_id"], "step": record["step"],
                                          "score_age": record["attention_score_age"],
                                          "score_mean": record["relevance_score_mean"],
                                          "score_min": record["relevance_score_min"],
                                          "score_max": record["relevance_score_max"],
                                          "capture_ms": record["attention_capture_ms"],
                                          "protected_background_tokens": record["task_relevant_background_protected"]})
        threshold_rows.append({"protection_percent": pct, "successes": threshold["aggregate"]["successes"],
                               "total": 9,
                               "mean_skipped_k_rows": float(np.mean([x["total_k_rows_skipped"] for x in threshold["scenarios"]])),
                               "mean_skipped_v_rows": float(np.mean([x["total_v_rows_skipped"] for x in threshold["scenarios"]]))})

    def write_csv(filename: str, rows: list[dict[str, Any]]):
        if not rows:
            (OUT / filename).write_text("status,no_complete_thresholds\n")
        else:
            with (OUT / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)

    write_csv("per_scenario_metrics.csv", scenario_rows)
    write_csv("relevance_scores_summary.csv", relevance_rows)
    write_csv("projection_savings.csv", projection_rows)
    summary = {"status": "complete" if complete else "incomplete",
               "experiment": "Stage 3C — Oracle Background + Task Relevance, Full-Layer K+V, R5",
               "required_protection_budgets": [int(x * 100) for x in FRACTIONS],
               "completed_protection_budgets": [x["protection_percent"] for x in threshold_rows],
               "threshold_results": threshold_rows,
               "references": {"fresh_successes": 8, "fresh_total": 9,
                              "stage3a_successes": 8, "stage3a_total": 9,
                              "stage3b_successes": 5, "stage3b_total": 9},
               "latency_measured": False,
               "method_caveat": "Adapted VLA-Cache-style task relevance; this is not a reproduction of the complete VLA-Cache algorithm."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    report = [
        "# Stage 3C — Oracle Background + Task Relevance, Full-Layer K+V, R5", "",
        f"Status: **{summary['status']}**; completed protection budgets: {summary['completed_protection_budgets']}.", "",
        "For each fresh R5 refresh, a read-only layer-15 sidecar computes text-query to visual-key attention from the already-produced Q/K projections after RoPE, applies causal masking and averages heads and prompt text queries for the 256 visual key positions. Its scores are reused for the following four reuse decisions. All oracle foreground patches and the selected top-ranked fraction of oracle-background patches remain fresh; only remaining background K/V rows are skipped and sourced from the last independent fresh prefill.", "",
        "The policy attention backend remains SDPA, and its output is not replaced. The sidecar uses FP32 QK/softmax for ranking, so it is an attention-derived relevance estimate rather than a dump of SDPA's internal probabilities. Its measured refresh-sidecar and mask-ranking costs are recorded separately. No full inference latency benchmark was run.", "",
        "| Protected background budget | Success | Mean K rows skipped / rollout | Mean V rows skipped / rollout |", "|---:|---:|---:|---:|",
    ]
    report.extend(f"| {x['protection_percent']}% | {x['successes']}/{x['total']} | {x['mean_skipped_k_rows']:.0f} | {x['mean_skipped_v_rows']:.0f} |" for x in threshold_rows)
    report.extend(["", "Reference success counts on the same nine scenarios: Fresh 8/9, Stage 3A 8/9, Stage 3B 5/9. Exact scenario pairs and structural row counts are in the CSV files and threshold subdirectories. Completion-step differences are descriptive only.", "", "This tests one task-relevance safety filter inside the existing oracle-masked full-layer K+V projection-skipping policy. It does not test the complete VLA-Cache method, and skipped projection rows are not measured latency savings.", ""])
    (OUT / "REPORT.md").write_text("\n".join(report))
    if threshold_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        labels = ["Fresh", "Stage 3A", "Stage 3B"] + [f"Stage 3C {x['protection_percent']}%" for x in threshold_rows]
        values = [8, 8, 5] + [x["successes"] for x in threshold_rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color=["#555555", "#4477aa", "#cc6677"] + ["#228833"] * len(threshold_rows))
        ax.set(ylim=(0, 9), ylabel="Successful scenarios (of 9)", xlabel="Condition",
               title="Fresh and oracle-masked reuse success comparison")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout(); fig.savefig(OUT / "success_by_protection_budget.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([x["protection_percent"] for x in threshold_rows], [x["mean_skipped_k_rows"] for x in threshold_rows], marker="o", label="K rows skipped")
        ax.plot([x["protection_percent"] for x in threshold_rows], [x["mean_skipped_v_rows"] for x in threshold_rows], marker="s", label="V rows skipped")
        ax.set(xlabel="Protection budget (%)", ylabel="Mean skipped projection rows / rollout",
               title="Structural projection savings (not measured latency)")
        ax.legend(); fig.tight_layout(); fig.savefig(OUT / "projection_rows_by_protection_budget.png", dpi=160); plt.close(fig)
        scenario_ids = [f"T{x['task_id']}/S{x['state_id']}" for x in threshold_summaries[0]["scenarios"]]
        matrix = [[int(s["fresh_reference"]["success"]) for s in threshold_summaries[0]["scenarios"]],
                  [int(s["stage3b_reference"]["success"]) for s in threshold_summaries[0]["scenarios"]]]
        ylabels = ["Fresh", "Stage 3B"]
        for threshold in threshold_summaries:
            matrix.append([int(s["success"]) for s in threshold["scenarios"]])
            ylabels.append(f"Stage 3C {int(threshold['fraction'] * 100)}%")
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.imshow(matrix, cmap=ListedColormap(["#cc6677", "#228833"]), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(scenario_ids)), scenario_ids, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(ylabels)), ylabels)
        ax.set_title("Scenario-level success outcomes (green=success, red=failure)")
        fig.tight_layout(); fig.savefig(OUT / "scenario_outcomes.png", dpi=160); plt.close(fig)
    return summary


ACTION_INVARIANCE: dict[str, Any] = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--threshold", type=float, choices=FRACTIONS)
    args = parser.parse_args()
    if args.finalize_only:
        print(json.dumps(finalize_outputs(), indent=2), flush=True)
        return
    if not args.validate_only and args.threshold is None:
        parser.error("--threshold is required unless --validate-only is set")
    model, processor, device = load_model_and_processor(DEFAULT_MODEL_ID)
    key = resolve_unnorm_key(model, None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configuration.json").write_text(json.dumps({
        "experiment": "Stage 3C — Oracle Background + Task Relevance, Full-Layer K+V, R5",
        "model": DEFAULT_MODEL_ID, "suite": "libero_spatial", "tasks": list(TASKS),
        "states": list(STATES), "seed": SEED, "horizon": HORIZON, "settling_steps": 10,
        "refresh_interval": R, "layers": list(LAYERS), "visual_positions": [1, 256],
        "components": ["K", "V"], "budgets": [0.10, 0.25, 0.50],
        "relevance_layer": RELEVANCE_LAYER, "attention_backend": "SDPA, unchanged",
        "score_method": "FP32 read-only sidecar from normal-pass RoPE-applied Q/K; mean heads and actual prompt-text queries over visual keys",
        "relevance_score_refresh_policy": "scores captured on fully fresh refresh and reused until next refresh",
        "source_policy": "most recent independent fresh prefill only; reuse outputs never promoted",
        "compute_skipping": "K/V for oracle foreground and protected background are fresh; only remaining oracle-background rows are skipped",
        "latency_benchmark": False,
    }, indent=2))
    (OUT / "run_metadata.json").write_text(json.dumps({
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "transformers_runtime": "4.49.0 (Stage 3B container)",
    }, indent=2))
    global ACTION_INVARIANCE
    ACTION_INVARIANCE = validate_action_invariance(model, processor, device, key)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(ACTION_INVARIANCE, indent=2))
    print("[ATTENTION_VALIDATION_PASS] " + json.dumps(ACTION_INVARIANCE), flush=True)
    if args.validate_only:
        return
    result = run_threshold(model, processor, device, key, args.threshold)
    print(f"[THRESHOLD_COMPLETE] fraction={args.threshold:.2f} successes={result['aggregate']['successes']}/9", flush=True)


if __name__ == "__main__":
    main()
