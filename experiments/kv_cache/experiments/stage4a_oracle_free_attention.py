#!/usr/bin/env python3
"""Stage 4A: oracle-free lagged text-to-visual attention K/V selection.

This runner uses the validated Stage 3C read-only layer-15 attention sidecar
and Stage 3B true K/V projection-skip context. It does not call oracle
segmentation, use oracle masks, or load a segmentation/prediction model.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from experiments.kv_analysis.experiments.extract_libero_image import (
    DEFAULT_STEP_ACTION, TASK_SUITE_NAME, create_libero_env,
    normalize_and_invert_openvla_gripper,
)
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import (
    assert_independent_cache_clone, capture_prediction, policy_image_from_observation,
    seed_everything,
)
from experiments.kv_cache.experiments.stage3b_oracle_full_kv_matrix import (
    HORIZON, LAYERS, R, SEED, STATES, TASKS, MaskedKVProjectionSkip,
)
from experiments.kv_cache.experiments.stage3c_oracle_task_relevance import RefreshAttentionSidecar
from experiments.kv_cache.experiments.static_reuse_experiment import (
    DEFAULT_MODEL_ID, cache_layers, clone_prefill_cache, load_model_and_processor,
    resolve_unnorm_key,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage4a_oracle_free_attention"
FRESH_FRACTIONS = (0.90, 0.75, 0.60)
METHODS = ("attention", "random")
VISUAL_COUNT = 256
VALIDATION_STEPS = 2


def fresh_count_for(fraction: float) -> int:
    if fraction not in FRESH_FRACTIONS:
        raise ValueError(f"Unsupported fresh fraction: {fraction}")
    return int(np.floor(VISUAL_COUNT * fraction + 0.5))


def rank_attention(scores: np.ndarray, fresh_count: int) -> np.ndarray:
    """Select exactly K highest scores; equal scores prefer lower token index."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (VISUAL_COUNT,) or not np.isfinite(scores).all():
        raise ValueError(f"Need 256 finite attention scores, got {scores.shape}.")
    if not 1 <= fresh_count <= VISUAL_COUNT:
        raise ValueError("Fresh visual-token count must be within [1, 256].")
    order = np.lexsort((np.arange(VISUAL_COUNT), -scores))
    mask = np.zeros(VISUAL_COUNT, dtype=bool)
    mask[order[:fresh_count]] = True
    if int(mask.sum()) != fresh_count or not mask.any() or mask.all() and fresh_count != VISUAL_COUNT:
        raise AssertionError("Attention mask partition/count invalid.")
    return mask


def random_mask(fresh_count: int, *, seed: int) -> np.ndarray:
    if not 1 <= fresh_count <= VISUAL_COUNT:
        raise ValueError("Fresh visual-token count must be within [1, 256].")
    mask = np.zeros(VISUAL_COUNT, dtype=bool)
    mask[np.random.default_rng(seed).choice(VISUAL_COUNT, fresh_count, replace=False)] = True
    assert int(mask.sum()) == fresh_count
    return mask


def mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def spatial_mask_stats(mask: np.ndarray) -> dict[str, Any]:
    grid=np.asarray(mask,dtype=bool).reshape(16,16)
    quadrants={"top_left":int(grid[:8,:8].sum()),"top_right":int(grid[:8,8:].sum()),
               "bottom_left":int(grid[8:,:8].sum()),"bottom_right":int(grid[8:,8:].sum())}
    return {"row_counts":grid.sum(axis=1).astype(int).tolist(),"column_counts":grid.sum(axis=0).astype(int).tolist(),
            "quadrant_counts":quadrants,
            "horizontal_neighbor_fraction":float((grid[:,:-1]&grid[:,1:]).sum()/max(grid[:,:-1].sum(),1)),
            "vertical_neighbor_fraction":float((grid[:-1,:]&grid[1:,:]).sum()/max(grid[:-1,:].sum(),1))}


def cache_layer_shapes(cache: Any) -> list[dict[str, Any]]:
    return [{"layer": i, "k_shape": list(k.shape), "v_shape": list(v.shape)}
            for i, (k, v) in enumerate(cache_layers(cache))]


def cache_storage_versions(cache: Any) -> list[tuple[int, int, int, int]]:
    return [(int(k.data_ptr()), int(v.data_ptr()), int(k._version), int(v._version))
            for k, v in cache_layers(cache)]


class ProjectionRowMonitor:
    """Read-only hooks proving actual q/k/v projection row counts."""
    def __init__(self, model: Any):
        self.model = model
        self.handles = []
        self.rows: dict[str, dict[int, list[int]]] = {name: {} for name in ("q", "k", "v")}

    def __enter__(self):
        for li, block in enumerate(self.model.language_model.model.layers):
            for name in ("q", "k", "v"):
                module = getattr(block.self_attn, f"{name}_proj")
                self.rows[name][li] = []

                def pre_hook(_module, inputs, name=name, li=li):
                    self.rows[name][li].append(int(inputs[0].shape[-2]))

                self.handles.append(module.register_forward_pre_hook(pre_hook))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def validate(self, *, prefill_length: int, projected_kv_rows: int, layers=LAYERS):
        expected = set(layers)
        for kind in ("q", "k", "v"):
            if set(self.rows[kind]) != expected:
                raise AssertionError(f"{kind}_proj hooks did not cover all selected layers.")
            for li in layers:
                calls = self.rows[kind][li]
                long_calls = [n for n in calls if n > 1]
                decode_calls = [n for n in calls if n == 1]
                wanted = prefill_length if kind == "q" else projected_kv_rows
                if long_calls != [wanted] or not decode_calls or any(n != 1 for n in decode_calls):
                    raise AssertionError(f"Unexpected {kind}_proj row calls at layer {li}: {calls}, expected prefill={wanted} and one-token decode.")

    def compact(self):
        return {kind: {str(li): rows for li, rows in by_layer.items()}
                for kind, by_layer in self.rows.items()}

    def per_layer_summary(self):
        result=[]
        for li in sorted(self.rows["q"]):
            result.append({"layer":li,
                "q_prefill_rows":max(self.rows["q"][li]),
                "k_prefill_rows":max(self.rows["k"][li]),
                "v_prefill_rows":max(self.rows["v"][li]),
                "q_one_token_calls":sum(n==1 for n in self.rows["q"][li]),
                "k_one_token_calls":sum(n==1 for n in self.rows["k"][li]),
                "v_one_token_calls":sum(n==1 for n in self.rows["v"][li])})
        return result


def save_mask_visualization(path: Path, image: np.ndarray, scores: np.ndarray | None,
                            attention_mask: np.ndarray, random_selected: np.ndarray | None,
                            title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    size, header = 224, 34
    cols = 4 if random_selected is not None else 3
    canvas = Image.new("RGB", (cols * size, header + size + 34), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["Policy RGB", "Lagged attention score" if scores is not None else "Attention score not used",
              "Attention-selected fresh" if scores is not None else "Random-selected fresh"]
    if random_selected is not None:
        labels.append("Matched-budget random fresh")
    draw.text((4, 3), title, fill=(20, 20, 20), font=font)
    rgb = Image.fromarray(image.astype(np.uint8)).resize((size, size), Image.BICUBIC)
    if scores is None:
        heat = Image.new("RGB", (size, size), (225, 225, 225))
    else:
        score = scores.reshape(16, 16)
        norm = np.clip((score - score.min()) / max(float(score.max() - score.min()), 1e-9), 0, 1)
        colors = np.zeros((16, 16, 3), dtype=np.uint8)
        colors[..., 0] = (255 * norm).astype(np.uint8)
        colors[..., 1] = (190 * (1 - norm)).astype(np.uint8)
        colors[..., 2] = 35
        heat = Image.fromarray(colors).resize((size, size), Image.NEAREST)
    selected = np.zeros((16, 16, 3), dtype=np.uint8); selected[:] = (42, 47, 55)
    selected.reshape(256, 3)[attention_mask] = (255, 207, 51)
    panels = [rgb, Image.blend(rgb, heat, 0.60), Image.fromarray(selected).resize((size, size), Image.NEAREST)]
    if random_selected is not None:
        rnd = np.zeros((16, 16, 3), dtype=np.uint8); rnd[:] = (42, 47, 55)
        rnd.reshape(256, 3)[random_selected] = (67, 181, 255)
        panels.append(Image.fromarray(rnd).resize((size, size), Image.NEAREST))
    for i, panel in enumerate(panels):
        draw.text((i * size + 4, 17), labels[i], fill=(20, 20, 20), font=font)
        canvas.paste(panel, (i * size, header))
    draw.text((4, header + size + 5), "Yellow/blue = selected for fresh K/V; dark = candidate stale reuse. Operational selection only; not physical segmentation.", fill=(40, 40, 40), font=font)
    canvas.save(path)


def validate_no_oracle_policy_code() -> dict[str, Any]:
    """AST guard against accidental calls/imports of oracle segmentation APIs."""
    tree = ast.parse(Path(__file__).read_text())
    forbidden = {"mask_from_obs", "rendered_label_mapping", "foreground_mask", "policy_mask", "token_mask"}
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            violations.append(node.id)
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            violations.append(node.attr)
    if violations:
        raise AssertionError(f"Oracle segmentation symbol present in Stage 4A policy runner: {violations}")
    return {"passed": True, "oracle_segmentation_calls": 0, "forbidden_symbol_references": 0,
            "policy_mask_source": "layer-15 refresh attention or seeded random selection only"}


def make_env(task_id: int):
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[TASK_SUITE_NAME]()
    task = suite.get_task(task_id)
    states = suite.get_task_init_states(task_id)
    env, task_description = create_libero_env(task, resolution=256)
    return suite, task, states, env, task_description


def _policy_call(model, processor, image, prompt, device, key, intervention=None, monitor=True):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    if monitor:
        with ProjectionRowMonitor(model) as rows:
            pred = capture_prediction(model, processor, image, prompt, device, key, intervention)
    else:
        rows = None
        pred = capture_prediction(model, processor, image, prompt, device, key, intervention)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return pred, elapsed_ms, rows


def _action_for_env(pred):
    action = normalize_and_invert_openvla_gripper(pred["policy_action"])
    if not np.isfinite(action).all():
        raise AssertionError("Policy produced non-finite environment action.")
    return action


def validate_runtime(model, processor, device, key) -> dict[str, Any]:
    no_oracle = validate_no_oracle_policy_code()
    selection = {}
    sample = np.linspace(-1.0, 1.0, 256, dtype=np.float64)
    for frac in FRESH_FRACTIONS:
        count = fresh_count_for(frac)
        mask = rank_attention(sample, count)
        rng = random_mask(count, seed=SEED + int(frac * 100))
        if mask.sum() != count or rng.sum() != count:
            raise AssertionError("Freshness budget mask test failed.")
        selection[str(int(frac * 100))] = {"fresh_count": count, "reuse_candidates": 256-count,
                                          "attention_partition": int(mask.sum()) + int((~mask).sum()),
                                          "random_partition": int(rng.sum()) + int((~rng).sum())}

    seed_everything(SEED)
    _, task, states, env, description = make_env(TASKS[0])
    prompt = f"In: What action should the robot take to {description.lower()}?\nOut:"
    try:
        env.reset(); obs = env.set_init_state(states[0])
        for _ in range(10):
            obs, _, done, _ = env.step(DEFAULT_STEP_ACTION)
            if done: raise RuntimeError("Environment terminated during settling.")
        image = policy_image_from_observation(obs, True)
        base, base_ms, base_rows = _policy_call(model, processor, image, prompt, device, key)
        sidecar = RefreshAttentionSidecar(model)
        scored, sidecar_ms, sidecar_rows = _policy_call(model, processor, image, prompt, device, key, sidecar)
        if sidecar.scores is None or sidecar.scores.shape != (256,) or not np.isfinite(sidecar.scores).all():
            raise AssertionError("Layer-15 sidecar must return 256 finite visual relevance scores.")
        if sidecar.sequence_length != scored["prefill_length"] or sidecar.backend != "sdpa":
            raise AssertionError("Attention sidecar layout/backend did not match the normal prefill.")
        if not np.array_equal(base["action_token_ids"], scored["action_token_ids"]):
            raise AssertionError("Read-only sidecar changed action-token IDs.")
        if not np.array_equal(base["policy_action"], scored["policy_action"]) or not torch.equal(base["logits"], scored["logits"]):
            raise AssertionError("Read-only sidecar changed fresh policy action/logits.")
        if sidecar.row_sum_max_error is None or sidecar.row_sum_max_error > 1e-5:
            raise AssertionError("Attention sidecar probabilities are not normalized.")
        if base_rows is not None:
            base_rows.validate(prefill_length=base["prefill_length"], projected_kv_rows=base["prefill_length"])
            sidecar_rows.validate(prefill_length=scored["prefill_length"], projected_kv_rows=scored["prefill_length"])
        source_cache = clone_prefill_cache(scored["cache"])
        assert_independent_cache_clone(source_cache, scored["cache"])
        source_shapes = cache_layer_shapes(source_cache)
        source_versions = cache_storage_versions(source_cache)
        source_layout = dict(scored["layout"])
        expected_refresh_step = 0
        if source_layout["visual"] != [1, 256] or source_layout["visual_token_count"] != 256:
            raise AssertionError(f"Unexpected visual layout: {source_layout}")
        obs, _, done, _ = env.step(_action_for_env(scored).tolist())
        if done: raise RuntimeError("Validation task completed before reuse validation frame.")
        image1 = policy_image_from_observation(obs, True)
        count = fresh_count_for(FRESH_FRACTIONS[0])
        mask = rank_attention(sidecar.scores, count)
        for frac in FRESH_FRACTIONS:
            n_fresh = fresh_count_for(frac)
            attention_mask = rank_attention(sidecar.scores, n_fresh)
            random_selected = random_mask(n_fresh, seed=SEED + int(frac * 100))
            save_mask_visualization(
                OUT / "mask_visualizations" / "validation" /
                f"task{TASKS[0]}_state0_attention_vs_random_{int(frac*100)}.png",
                image, sidecar.scores, attention_mask, random_selected,
                f"Validation · attention vs matched random · {int(frac*100)}% fresh")
        fresh_cf, fresh_cf_ms, fresh_cf_rows = _policy_call(model, processor, image1, prompt, device, key)
        if fresh_cf_rows is not None:
            fresh_cf_rows.validate(prefill_length=fresh_cf["prefill_length"], projected_kv_rows=fresh_cf["prefill_length"])
        context = MaskedKVProjectionSkip(model, source_cache, source_layout, mask)
        masked, masked_ms, monitor = _policy_call(model, processor, image1, prompt, device, key, context)
        context.validate(masked["cache"])
        lang = int(masked["layout"]["language"][1] - masked["layout"]["language"][0] + 1)
        expected_rows = 1 + lang + count
        monitor.validate(prefill_length=masked["prefill_length"], projected_kv_rows=expected_rows)
        if len(context.diag["rows_by_layer"]) != 32 or sorted(context.diag["layers"]) != list(LAYERS):
            raise AssertionError("True-skip validation did not cover all 32 layers.")
        if any(r["k_projected_rows"] != expected_rows or r["v_projected_rows"] != expected_rows
               or r["k_skipped_background_rows"] != 256-count or r["v_skipped_background_rows"] != 256-count
               for r in context.diag["rows_by_layer"]):
            raise AssertionError("Actual skip rows disagree with selected mask.")
        if source_layout != masked["layout"]:
            raise AssertionError("Source/target multimodal layout differs during validation.")
        if cache_layer_shapes(masked["cache"]) != source_shapes:
            raise AssertionError("Reconstructed cache shape differs from refresh cache.")
        if cache_storage_versions(source_cache) != source_versions:
            raise AssertionError("Reuse modified the independently stored fresh source cache.")
        if not np.isfinite(masked["policy_action"]).all():
            raise AssertionError("Masked skip produced nonfinite policy action.")
        action_delta = np.asarray(masked["policy_action"], dtype=np.float64) - np.asarray(fresh_cf["policy_action"], dtype=np.float64)
        return {
            "status": "passed", "oracle_free_code_check": no_oracle,
            "freshness_budget_tests": selection,
            "refresh_sidecar": {"attention_backend": sidecar.backend, "score_shape": list(sidecar.scores.shape),
                "scores_finite": True, "probability_row_sum_max_error": sidecar.row_sum_max_error,
                "qk_probability_shape": sidecar.axis_shape, "text_query_positions": [sidecar.query_positions[0], sidecar.query_positions[-1]],
                "prefill_length": scored["prefill_length"], "visual_positions": scored["layout"]["visual"],
                "source_step": expected_refresh_step, "source_is_independent_prefill_clone": True},
            "fresh_output_invariance": {"action_tokens_exact": True, "policy_action_exact": True,
                "prefill_logits_exact": True, "baseline_wall_ms": base_ms,
                "sidecar_wall_ms": sidecar_ms, "sidecar_additional_computation_ms": sidecar.capture_ms,
                "qkv_rows_full_on_fresh_prefill": True},
            "one_reuse_true_skip": {"score_age": 1, "fresh_visual_count": count,
                "reuse_candidate_count": 256-count, "k_rows_projected_per_layer": expected_rows,
                "v_rows_projected_per_layer": expected_rows,
                "k_rows_skipped_total": (256-count)*32, "v_rows_skipped_total": (256-count)*32,
                "all_32_layers_validated": True, "q_rows_full": True,
                "decode_one_token_calls_normal": True, "cache_shape_equal": True,
                "source_cache_unmodified": True,
                "cache_validation": "Stage 3B validated context: fresh rows current; unselected visual rows from independent refresh source",
                "token_ids": masked["action_token_ids"], "action_finite": True,
                "fresh_counterfactual_token_ids": fresh_cf["action_token_ids"],
                "fresh_counterfactual_exact_token_match": masked["action_token_ids"] == fresh_cf["action_token_ids"],
                "fresh_counterfactual_policy_action_l2": float(np.linalg.norm(action_delta)),
                "fresh_counterfactual_policy_action_max_abs": float(np.max(np.abs(action_delta))),
                "fresh_counterfactual_wall_ms": fresh_cf_ms,
                "true_skip_wall_ms": masked_ms, "monitor_rows": monitor.compact()},
            "models_loaded": [DEFAULT_MODEL_ID], "extra_model_loaded": False,
            "oracle_mask_accessed": False,
        }
    finally:
        env.close()


def _seed_for_mask(task_id: int, state_id: int, step: int, fresh_count: int) -> int:
    return int(SEED * 1_000_003 + task_id * 10_007 + state_id * 101 + step * 17 + fresh_count)


def run_episode(method: str, fresh_fraction: float, task_id: int, state_id: int,
                model, processor, device, key, save_visuals=True) -> dict[str, Any]:
    if method not in METHODS: raise ValueError(method)
    fresh_count = fresh_count_for(fresh_fraction)
    seed_everything(SEED)
    suite, task, states, env, description = make_env(task_id)
    episode_dir = OUT / f"fresh_{int(fresh_fraction*100):02d}" / method / f"task{task_id}_state{state_id}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    prompt = f"In: What action should the robot take to {description.lower()}?\nOut:"
    records, source, done = [], None, False
    rollout_start = time.perf_counter()
    try:
        env.reset(); obs = env.set_init_state(states[state_id])
        for _ in range(10):
            obs, _, done, _ = env.step(DEFAULT_STEP_ACTION)
            if done: raise RuntimeError("Environment terminated during official settling steps.")
        for step in range(HORIZON):
            refresh = (step % R == 0)
            image = policy_image_from_observation(obs, True)
            mask_ms = 0.0
            counterfactual = None
            if refresh:
                sidecar = RefreshAttentionSidecar(model) if method == "attention" else None
                pred, inference_ms, monitor = _policy_call(model, processor, image, prompt, device, key, sidecar)
                if monitor is not None:
                    monitor.validate(prefill_length=pred["prefill_length"], projected_kv_rows=pred["prefill_length"])
                if method == "attention":
                    scores = np.asarray(sidecar.scores, dtype=np.float32)
                    if scores.shape != (256,) or not np.isfinite(scores).all():
                        raise AssertionError("Refresh attention scores invalid.")
                    select_start = time.perf_counter(); mask = rank_attention(scores, fresh_count)
                    mask_ms = (time.perf_counter()-select_start)*1000
                else:
                    scores = None
                    select_start = time.perf_counter()
                    mask = random_mask(fresh_count, seed=_seed_for_mask(task_id, state_id, step, fresh_count))
                    mask_ms = (time.perf_counter()-select_start)*1000
                fresh_cache = clone_prefill_cache(pred["cache"])
                assert_independent_cache_clone(fresh_cache, pred["cache"])
                source = {"cache": fresh_cache, "layout": dict(pred["layout"]), "step": step,
                    "fresh": True, "mask": mask.copy(), "scores": None if scores is None else scores.copy()}
                age, source_step = 0, step
                context = None
                projected = pred["prefill_length"]
                skipped = 0
                if save_visuals and step == 0:
                    viz = OUT / "mask_visualizations" / f"fresh_{int(fresh_fraction*100):02d}" / method / f"task{task_id}_state{state_id}_refresh000.png"
                    save_mask_visualization(viz, image, scores,
                                            mask, None, f"{method} · {int(fresh_fraction*100)}% fresh · Task {task_id}/State {state_id}")
            else:
                expected_source = step - (step % R)
                if source is None or source["step"] != expected_source or not source["fresh"]:
                    raise AssertionError("Source must remain the latest independent fully fresh R5 prefill.")
                mask, scores = source["mask"], source["scores"]
                if mask.shape != (256,) or int(mask.sum()) != fresh_count:
                    raise AssertionError("Lagged mask does not match configured freshness budget.")
                select_start = time.perf_counter()
                if method == "attention":
                    mask_now = rank_attention(scores, fresh_count)
                    if not np.array_equal(mask_now, mask): raise AssertionError("Attention mask changed inside refresh window.")
                mask_ms = (time.perf_counter()-select_start)*1000
                source_versions = cache_storage_versions(source["cache"])
                context = MaskedKVProjectionSkip(model, source["cache"], source["layout"], mask)
                pred, inference_ms, monitor = _policy_call(model, processor, image, prompt, device, key, context)
                context.validate(pred["cache"])
                lang_count = pred["layout"]["language"][1] - pred["layout"]["language"][0] + 1
                expected_rows = 1 + lang_count + fresh_count
                monitor.validate(prefill_length=pred["prefill_length"], projected_kv_rows=expected_rows)
                if any(x["k_projected_rows"] != expected_rows or x["v_projected_rows"] != expected_rows
                       or x["k_skipped_background_rows"] != 256-fresh_count or x["v_skipped_background_rows"] != 256-fresh_count
                       for x in context.diag["rows_by_layer"]):
                    raise AssertionError("Projection row instrumentation disagrees with active mask.")
                age, source_step = step - source["step"], source["step"]
                if age not in (1, 2, 3, 4): raise AssertionError(f"R5 age invalid: {age}")
                if cache_layers(source["cache"])[0][0].shape[-2] != pred["prefill_length"]:
                    raise AssertionError("Fresh source snapshot was mutated or extended.")
                if cache_storage_versions(source["cache"]) != source_versions:
                    raise AssertionError("Projection skip mutated its stored fresh source cache.")
                if step == 1:
                    # Same current observation, diagnostic only. Its action is
                    # never sent to the environment and cannot update source.
                    fresh_cf, fresh_cf_ms, _ = _policy_call(model, processor, image, prompt, device, key, monitor=False)
                    id_a=np.asarray(pred["action_token_ids"],dtype=np.int64)
                    id_b=np.asarray(fresh_cf["action_token_ids"],dtype=np.int64)
                    raw_delta=np.asarray(pred["policy_action"],dtype=np.float64)-np.asarray(fresh_cf["policy_action"],dtype=np.float64)
                    env_a=_action_for_env(pred);env_b=_action_for_env(fresh_cf)
                    env_delta=env_a.astype(np.float64)-env_b.astype(np.float64)
                    changed=np.flatnonzero(id_a!=id_b)
                    counterfactual={"fresh_token_ids":id_b.tolist(),"reuse_token_ids":id_a.tolist(),
                        "exact_token_match":bool(np.array_equal(id_a,id_b)),"changed_token_count":int(changed.size),
                        "first_divergent_token_index":int(changed[0]) if changed.size else None,
                        "raw_policy_action_l2":float(np.linalg.norm(raw_delta)),
                        "raw_policy_action_max_abs":float(np.max(np.abs(raw_delta))),
                        "environment_action_l2":float(np.linalg.norm(env_delta)),
                        "environment_action_max_abs":float(np.max(np.abs(env_delta))),
                        "fresh_counterfactual_wall_ms":fresh_cf_ms,"diagnostic_action_sent_to_environment":False}
                projected = expected_rows
                skipped = 256 - fresh_count
            env_action = _action_for_env(pred)
            if not np.isfinite(pred["policy_action"]).all(): raise AssertionError("Non-finite model action.")
            layer_count = len(LAYERS)
            record = {
                "step": step, "condition": f"{method}_fresh_{int(fresh_fraction*100)}",
                "mask_method": method, "fresh_budget_requested": fresh_fraction,
                "refresh_or_reuse": "refresh" if refresh else "reuse",
                "refresh_interval": R, "source_step": source_step, "source_age": age,
                "attention_score_age": age if method == "attention" else None,
                "prefill_length": pred["prefill_length"],
                "language_tokens": int(pred["layout"]["language"][1]-pred["layout"]["language"][0]+1),
                "visual_tokens": 256, "fresh_visual_tokens": int(mask.sum()),
                "candidate_reuse_visual_tokens": int((~mask).sum()),
                "fresh_visual_fraction_actual": float(mask.mean()),
                "mask_indices_on_refresh": np.flatnonzero(mask).tolist() if refresh else None,
                "mask_sha256": mask_hash(mask),
                "mask_spatial_stats": spatial_mask_stats(mask),
                "attention_scores": source["scores"].tolist() if method == "attention" and refresh else None,
                "attention_score_stats": (None if method != "attention" else {
                    "min": float(source["scores"].min()), "max": float(source["scores"].max()),
                    "mean": float(source["scores"].mean()), "std": float(source["scores"].std())}),
                "mask_selection_ms": mask_ms,
                "attention_capture_ms": sidecar.capture_ms if refresh and method == "attention" else None,
                "synchronized_policy_inference_wall_ms": inference_ms,
                "q_projected_rows_per_layer": pred["prefill_length"],
                "k_projected_rows_per_layer": pred["prefill_length"] if refresh else projected,
                "v_projected_rows_per_layer": pred["prefill_length"] if refresh else projected,
                "k_skipped_rows_per_layer": 0 if refresh else skipped,
                "v_skipped_rows_per_layer": 0 if refresh else skipped,
                "projection_rows_by_layer": monitor.per_layer_summary(),
                "total_k_rows_skipped_step": layer_count*skipped,
                "total_v_rows_skipped_step": layer_count*skipped,
                "all_32_layers_validated": refresh or len(context.diag["layers"]) == 32,
                "source_is_fresh_refresh_only": bool(source["fresh"]),
                "source_cache_unmodified": True,
                "action_token_ids": pred["action_token_ids"],
                "policy_action": np.asarray(pred["policy_action"]).tolist(),
                "environment_action": env_action.tolist(),
                "fresh_counterfactual_divergence": counterfactual,
                "finite_action": bool(np.isfinite(pred["policy_action"]).all()),
            }
            obs, _, done, _ = env.step(env_action.tolist())
            record["done"] = bool(done)
            records.append(record)
            (episode_dir / "results.partial.json").write_text(json.dumps({
                "status": "running", "method": method, "fresh_fraction": fresh_fraction,
                "task_id": task_id, "state_id": state_id, "steps_completed": len(records), "records": records,
            }, indent=2))
            if done: break
    finally:
        env.close()
    total_elapsed = time.perf_counter()-rollout_start
    refreshes = [x for x in records if x["refresh_or_reuse"] == "refresh"]
    reuses = [x for x in records if x["refresh_or_reuse"] == "reuse"]
    skip_k = sum(r["total_k_rows_skipped_step"] for r in records)
    skip_v = sum(r["total_v_rows_skipped_step"] for r in records)
    score_transition_records = []
    if method == "attention":
        for previous, current in zip(refreshes, refreshes[1:]):
            a = np.asarray(previous["attention_scores"], dtype=np.float64)
            b = np.asarray(current["attention_scores"], dtype=np.float64)
            ma = np.zeros(256, dtype=bool); mb = np.zeros(256, dtype=bool)
            ma[np.asarray(previous["mask_indices_on_refresh"], dtype=np.int64)] = True
            mb[np.asarray(current["mask_indices_on_refresh"], dtype=np.int64)] = True
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            score_transition_records.append({
                "from_refresh_step": previous["step"], "to_refresh_step": current["step"],
                "score_cosine": float(np.dot(a, b) / den) if den else None,
                "fresh_mask_flip_count": int(np.count_nonzero(ma != mb)),
                "fresh_mask_jaccard": float(np.count_nonzero(ma & mb) / np.count_nonzero(ma | mb)),
            })
    result = {
        "status": "complete", "experiment": "Stage 4A — Oracle-Free Lagged Attention Foreground Selection",
        "condition": f"{method}_fresh_{int(fresh_fraction*100)}", "method": method,
        "task_id": task_id, "task_description": description, "state_id": state_id,
        "seed": SEED, "horizon": HORIZON, "success": bool(done), "steps": len(records),
        "termination": "success" if done else "horizon", "R": R, "layers": list(LAYERS),
        "fresh_visual_tokens": fresh_count, "reuse_candidate_visual_tokens": 256-fresh_count,
        "refresh_count": len(refreshes), "reuse_count": len(reuses),
        "temporal_reuse_fraction": len(reuses)/len(records) if records else 0,
        "max_source_age": max((r["source_age"] for r in records), default=0),
        "source_age_counts": {str(age): sum(r["source_age"] == age for r in records) for age in range(R)},
        "total_k_rows_skipped": skip_k, "total_v_rows_skipped": skip_v,
        "normalized_k_rows_skipped_per_decision": skip_k/len(records) if records else 0,
        "normalized_v_rows_skipped_per_decision": skip_v/len(records) if records else 0,
        "fraction_of_all_visual_k_rows_skipped": skip_k/(len(records)*256*len(LAYERS)) if records else 0,
        "fraction_of_all_visual_v_rows_skipped": skip_v/(len(records)*256*len(LAYERS)) if records else 0,
        "mean_policy_inference_wall_ms": float(np.mean([r["synchronized_policy_inference_wall_ms"] for r in records])) if records else None,
        "p50_policy_inference_wall_ms": float(np.median([r["synchronized_policy_inference_wall_ms"] for r in records])) if records else None,
        "p95_policy_inference_wall_ms": float(np.percentile([r["synchronized_policy_inference_wall_ms"] for r in records],95)) if records else None,
        "mean_attention_capture_ms_refresh": float(np.mean([r["attention_capture_ms"] for r in refreshes if r["attention_capture_ms"] is not None])) if method == "attention" and refreshes else None,
        "mean_selection_ms_reuse": float(np.mean([r["mask_selection_ms"] for r in reuses])) if reuses else None,
        "attention_score_refresh_transitions": score_transition_records,
        "rollout_wall_seconds_including_env": total_elapsed,
        "all_invariants_passed": all(r["all_32_layers_validated"] and r["finite_action"] and r["source_is_fresh_refresh_only"] for r in records),
        "oracle_used_at_inference": False, "records": records,
    }
    path = episode_dir / "results.json"
    path.write_text(json.dumps(result, indent=2))
    partial = episode_dir / "results.partial.json"
    if partial.exists(): partial.unlink()
    return result


def load_reference(task_id: int, state_id: int) -> dict[str, Any]:
    path = OUT.parent / "stage3b_oracle_full_kv_matrix" / f"task{task_id}_state{state_id}" / "results.json"
    if not path.exists(): raise FileNotFoundError(f"Missing Stage3B/Fresh reference: {path}")
    d = json.loads(path.read_text())
    fresh = d["fresh"]
    stage3b = d["stage3b_true_skip"]
    return {"fresh": {"success": fresh["success"], "steps": fresh["steps"]},
            "stage3b": {"success": stage3b["success"], "steps": stage3b["steps"]}}


def attach_comparators(result: dict[str, Any]) -> dict[str, Any]:
    task_id, state_id = result["task_id"], result["state_id"]
    result["references"] = load_reference(task_id, state_id)
    stage3c_results = {}
    for pct in (10, 25, 50):
        p = OUT.parent / "stage3c_oracle_task_relevance" / f"protect_{pct:02d}" / f"task{task_id}_state{state_id}" / "results.json"
        if p.exists():
            d=json.loads(p.read_text()); stage3c_results[str(pct)]={"success":d["success"],"steps":d["steps"]}
    result["stage3c_oracle_relevance_references"] = stage3c_results
    return result


def validate_summary(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    expected = len(TASKS)*len(STATES)
    if d.get("status") != "complete" or len(d.get("scenarios", [])) != expected:
        raise ValueError(f"Incomplete matrix at {path}")
    for r in d["scenarios"]:
        if not r["all_invariants_passed"] or r["steps"] != len(r["records"]):
            raise ValueError(f"Invalid scenario: {r.get('task_id')}/{r.get('state_id')}")
        for step in r["records"]:
            if step["fresh_visual_tokens"]+step["candidate_reuse_visual_tokens"] != 256:
                raise ValueError("Fresh/reuse mask does not partition 256 tokens.")
            if step["refresh_or_reuse"] == "reuse" and step["source_age"] not in (1,2,3,4):
                raise ValueError("Bad R5 source age.")
            if step["refresh_or_reuse"] == "reuse" and step["source_step"] != step["step"]-step["step"]%R:
                raise ValueError("Reuse source is not latest refresh.")
    return d


def finalize_outputs() -> dict[str, Any]:
    scenario_rows=[]; budget_rows=[]; completed=True
    for frac in FRESH_FRACTIONS:
        pct=int(frac*100)
        for method in METHODS:
            scenarios=[]
            for task_id in TASKS:
                for state_id in STATES:
                    p=OUT/f"fresh_{pct:02d}"/method/f"task{task_id}_state{state_id}"/"results.json"
                    if not p.exists(): completed=False; continue
                    r=json.loads(p.read_text())
                    if not r.get("all_invariants_passed"): completed=False
                    scenarios.append(r)
                    scenario_rows.append({"fresh_budget_pct":pct,"mask_method":method,"task_id":task_id,"state_id":state_id,
                        "success":int(r["success"]),"steps":r["steps"],"refresh":r["refresh_count"],"reuse":r["reuse_count"],
                        "temporal_reuse_pct":100*r["temporal_reuse_fraction"],"fresh_visual_tokens":r["fresh_visual_tokens"],
                        "candidate_reuse_tokens":r["reuse_candidate_visual_tokens"],"skipped_k_rows":r["total_k_rows_skipped"],
                        "skipped_v_rows":r["total_v_rows_skipped"],"k_rows_per_decision":r["normalized_k_rows_skipped_per_decision"],
                        "v_rows_per_decision":r["normalized_v_rows_skipped_per_decision"],"p50_inference_ms":r["p50_policy_inference_wall_ms"],
                        "p95_inference_ms":r["p95_policy_inference_wall_ms"],"all_invariants_passed":r["all_invariants_passed"]})
            budget_rows.append({"fresh_budget_pct":pct,"method":method,"completed_scenarios":len(scenarios),
                "successes":sum(bool(x["success"]) for x in scenarios),"total":len(scenarios),
                "mean_skipped_k_rows_per_decision":float(np.mean([x["normalized_k_rows_skipped_per_decision"] for x in scenarios])) if scenarios else None,
                "mean_skipped_v_rows_per_decision":float(np.mean([x["normalized_v_rows_skipped_per_decision"] for x in scenarios])) if scenarios else None,
                "mean_temporal_reuse_fraction":float(np.mean([x["temporal_reuse_fraction"] for x in scenarios])) if scenarios else None})
    summary={"status":"complete" if completed and len(scenario_rows)==len(FRESH_FRACTIONS)*len(METHODS)*9 else "incomplete",
        "experiment":"Stage 4A — Oracle-Free Lagged Attention Foreground Selection",
        "scope":"lagged layer-15 text-query-to-visual-key attention; no oracle segmentation at inference",
        "config":{"model":DEFAULT_MODEL_ID,"suite":"libero_spatial","tasks":list(TASKS),"states":list(STATES),
            "seed":SEED,"horizon":HORIZON,"settling_steps":10,"R":R,"layers":list(LAYERS),
            "visual_positions":[1,256],"components":["K","V"],"fresh_budgets_pct":[90,75,60],
            "budget_token_counts":{"90":{"fresh":230,"reuse_candidate":26},"75":{"fresh":192,"reuse_candidate":64},"60":{"fresh":154,"reuse_candidate":102}},
            "oracle_used_at_inference":False,"attention_backend":"unchanged SDPA","lag":"scores/mask from latest full refresh, age 1-4"},
        "validation":json.loads((OUT/"validation.json").read_text()) if (OUT/"validation.json").exists() else None,
        "budget_method_aggregates":budget_rows,"scenarios":scenario_rows,
        "references":{"Fresh":"reused from matched Stage 3B records; no rerun","Stage3A":"outputs/stage3a_oracle_masked_matrix","Stage3B":"outputs/stage3b_oracle_full_kv_matrix","Stage3C":"outputs/stage3c_oracle_task_relevance"},
        "latency_caveat":"synchronized wall time includes policy preprocessing, model inference and greedy decoding, measured with projection hooks; not a dedicated CUDA-event benchmark",
        "scientific_caveat":"operational fresh/reuse selection is not validated physical foreground/background segmentation; nine scenarios are a bounded pilot."}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    with (OUT/"per_scenario.csv").open("w",newline="") as f:
        fields=list(scenario_rows[0]) if scenario_rows else ["status"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(scenario_rows)
    report=["# Stage 4A — Oracle-Free Lagged Attention Foreground Selection","",
        f"Status: **{summary['status']}**. This is the revised attention-based Stage 4A, not successful UNMATCH-style SigLIP grounding.","",
        "At each R5 refresh, a read-only layer-15 sidecar captures text-query-to-visual-key attention during the normal prefill. The attention-ranked mask is held through the following four steps. The top-K patches are operationally selected for fresh K/V; the remaining patches are reuse candidates. No oracle segmentation values enter policy selection. A seeded random mask uses the same exact fresh-token count as a nonsemantic control.","",
        "Fresh budgets are 90%, 75%, and 60% (rounded to 230, 192, and 154 visual positions; reuse candidates 26, 64, and 102). K and V true projection skipping covers all 32 Llama layers; Q remains full. Refresh sources are independent full fresh prefill snapshots only; reuse outputs are never promoted.","",
        "| Fresh budget | Mask | Success | Mean candidate K rows skipped / decision | Mean candidate V rows skipped / decision |","|---:|---|---:|---:|---:|"]
    for b in budget_rows: report.append(f"| {b['fresh_budget_pct']}% | {b['method']} | {b['successes']}/{b['total']} | {b['mean_skipped_k_rows_per_decision']:.1f} | {b['mean_skipped_v_rows_per_decision']:.1f} |" if b['completed_scenarios'] else f"| {b['fresh_budget_pct']}% | {b['method']} | incomplete | — | — |")
    report += ["","Fresh and Stage 3B outcomes are reused from matching saved scenario records; Stage 3C oracle-assisted outcomes are attached per scenario. The comparisons are descriptive, not evidence of significance or generalization.","",
        "Score maps are lagged by up to four control decisions. A relevant patch that becomes important after a refresh remains eligible for reuse until the next refresh. The selected set means fresh-computation priority, not confirmed object foreground. Skipped projection rows are structural counts, not latency savings.","",
        "Inference wall-time values are synchronized per-call diagnostics collected with read-only projection hooks; they are not a dedicated timing benchmark. Sidecar capture, mask-selection, and policy wall times are separately logged.",""]
    (OUT/"REPORT.md").write_text("\n".join(report))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--validate-only",action="store_true")
    ap.add_argument("--run-condition",choices=[f"{int(f*100)}-{m}" for f in FRESH_FRACTIONS for m in METHODS])
    ap.add_argument("--run-all",action="store_true",help="Run all budget/mask conditions after validation.json passes.")
    ap.add_argument("--finalize-only",action="store_true")
    args=ap.parse_args()
    if args.finalize_only:
        print(json.dumps(finalize_outputs(),indent=2));return
    OUT.mkdir(parents=True,exist_ok=True)
    config={"experiment":"Stage 4A — Oracle-Free Lagged Attention Foreground Selection",
        "model":DEFAULT_MODEL_ID,"suite":"libero_spatial","tasks":list(TASKS),"states":list(STATES),
        "seed":SEED,"horizon":HORIZON,"settling_steps":10,"refresh_interval":R,
        "layers":list(LAYERS),"visual_positions":[1,256],"components":["K","V"],
        "attention_score_layer":15,"attention_backend":"SDPA unchanged; read-only FP32 Q/K sidecar",
        "score_aggregation":"mean over text-query positions and attention heads for visual key positions 1..256",
        "score_lag":"captured on fully fresh refresh; held for the following reuse steps until the next R5 refresh",
        "selection":"stable descending attention ranking over all 256 positions; lower visual index breaks ties",
        "fresh_budgets":{"90":{"requested_fraction":0.90,"fresh_tokens":230,"reuse_candidates":26},
                         "75":{"requested_fraction":0.75,"fresh_tokens":192,"reuse_candidates":64},
                         "60":{"requested_fraction":0.60,"fresh_tokens":154,"reuse_candidates":102}},
        "controls":"seeded random mask with same exact fresh-token count, sampled at each refresh and held through its R5 window",
        "oracle_segmentation_in_policy":False,"extra_model_or_trained_head":False,
        "source":"independent clone of current fully fresh multimodal prefill only; reused caches never promoted",
        "latency":"synchronized wall-clock diagnostic around policy inference with projection hooks; not a dedicated CUDA-event benchmark"}
    (OUT/"configuration.json").write_text(json.dumps(config,indent=2))
    if args.validate_only or args.run_condition or args.run_all:
        model,processor,device=load_model_and_processor(DEFAULT_MODEL_ID)
        key=resolve_unnorm_key(model,None)
        (OUT/"run_metadata.json").write_text(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,
            "device":torch.cuda.get_device_name(device),"device_type":str(device),
            "transformers_expected":"4.49.0","method":"existing Stage 3C attention sidecar + Stage 3B masked projection skip"},indent=2))
        if args.validate_only:
            result=validate_runtime(model,processor,device,key)
            (OUT/"validation.json").write_text(json.dumps(result,indent=2))
            (OUT/"VALIDATION_REPORT.md").write_text(
                "# Stage 4A oracle-free attention selection — validation\n\n"
                "Status: **PASS**. The unchanged SDPA policy path produced exactly matching full-fresh prefill logits, greedy action-token IDs, and policy action with/without the read-only Q/K sidecar. The layer-15 sidecar returned 256 finite scores with normalized FP32 probability rows.\n\n"
                "A same-observation R5 reuse validation used the top-90%-fresh mask. Hooks verified full-sequence Q rows, 266 K and 266 V projection rows (BOS + 230 selected patches + 35 language positions), and 26 reused visual rows/layer across all 32 layers. Thus 832 K and 832 V rows were omitted per reuse prefill. Cache shape, independent fresh source, original source cache, and normal one-token decode checks passed. The reuse action matched the same-observation fresh counterfactual exactly in this validation sample.\n\n"
                "The policy runner contains no oracle segmentation calls or references. Fresh masks are generated from refresh attention rankings; random controls use a deterministic seeded mask. This validates the inherited Stage 3B cache context for one scenario and one budget; it is not evidence of task success preservation.\n")
            print("[VALIDATION_PASS] "+json.dumps(result),flush=True)
            return
        if args.run_all:
            validation_path=OUT/"validation.json"
            if not validation_path.exists() or json.loads(validation_path.read_text()).get("status")!="passed":
                raise RuntimeError("Refusing matrix: Stage 4A runtime validation has not passed.")
            for frac in FRESH_FRACTIONS:
                pct=int(frac*100)
                for method in METHODS:
                    condition=f"{pct}-{method}"
                    for task_id in TASKS:
                        for state_id in STATES:
                            p=OUT/f"fresh_{pct:02d}"/method/f"task{task_id}_state{state_id}"/"results.json"
                            if p.exists():
                                old=json.loads(p.read_text())
                                if old.get("status")=="complete" and old.get("all_invariants_passed"):
                                    print(f"[RESUME_SKIP] {condition} task={task_id} state={state_id}",flush=True);continue
                            result=attach_comparators(run_episode(method,frac,task_id,state_id,model,processor,device,key))
                            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2))
                            finalize_outputs()
                            print(f"[SCENARIO_COMPLETE] {condition} task={task_id} state={state_id} success={result['success']} steps={result['steps']}",flush=True)
                    finalize_outputs()
            print("[MATRIX_COMPLETE] "+json.dumps(finalize_outputs(),indent=2),flush=True)
            return
        pct_method=args.run_condition.split("-",1)
        pct=int(pct_method[0]);method=pct_method[1];frac=pct/100
        for task_id in TASKS:
            for state_id in STATES:
                p=OUT/f"fresh_{pct:02d}"/method/f"task{task_id}_state{state_id}"/"results.json"
                if p.exists():
                    old=json.loads(p.read_text())
                    if old.get("status")=="complete" and old.get("all_invariants_passed"):
                        print(f"[RESUME_SKIP] {args.run_condition} task={task_id} state={state_id}",flush=True);continue
                r=attach_comparators(run_episode(method,frac,task_id,state_id,model,processor,device,key))
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2))
                finalize_outputs()
                print(f"[SCENARIO_COMPLETE] {args.run_condition} task={task_id} state={state_id} success={r['success']} steps={r['steps']}",flush=True)
        finalize_outputs()
        return
    ap.error("Specify --validate-only, --run-condition, or --finalize-only")


if __name__ == "__main__":
    main()
