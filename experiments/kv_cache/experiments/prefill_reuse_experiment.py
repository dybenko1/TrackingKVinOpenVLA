"""During-prefill stale-language-KV intervention for OpenVLA.

This is intentionally different from ``static_reuse_experiment.py``.  It runs
the t+1 multimodal prefill normally up to each decoder layer's cache update,
then replaces selected *language* K and/or V positions with post-RoPE K/V saved
from observation t.  The modified values participate in that layer's attention,
so the t+1 prefill logits can change action token 1 and, consequently, tokens
2--7.

The intervention propagates through the transformer by design: changing layer
L attention changes its output hidden states, which changes the fresh Q/K/V
computed at L+1.  If L+1 is selected, its language K/V are then replaced again
immediately before its own attention.  This is not equivalent to editing a
completed cache independently per layer, and it is NOT compute-saving reuse:
fresh t+1 projections are still computed first.

The implementation temporarily wraps the live Transformers ``DynamicCache``
``update`` method only for an experimental forward.  In Transformers 4.49's
Llama attention path, ``update`` receives K after head reshaping and RoPE and V
after head reshaping, and returns the K/V consumed by the attention backend.
The wrapper restores the original method even if inference raises.

Run from the OpenVLA Docker environment, for example:

    python -m experiments.kv_cache.experiments.prefill_reuse_experiment \\
      --image-prev experiments/kv_analysis/inputs/trajectory_frames/step_000.png \\
      --image-current experiments/kv_analysis/inputs/trajectory_frames/step_001.png \\
      --instruction "pick up the black bowl between the plate and the ramekin and place it on the plate" \\
      --reuse-layers 0-13 --reuse-component kv

This is an offline counterfactual experiment.  It neither skips projections nor
applies an intervened action to LIBERO.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from PIL import Image
from transformers.cache_utils import DynamicCache

# Reuse read-only, generic utilities from the preserved post-prefill baseline.
from experiments.kv_cache.experiments.static_reuse_experiment import (
    DEFAULT_MODEL_ID,
    OUTPUT_DIR,
    PrefillSnapshot,
    cache_layers,
    cache_summary,
    clone_prefill_cache,
    decode_action_token_ids,
    greedy_decode_from_prefill,
    load_model_and_processor,
    parse_layer_spec,
    prepare_inputs,
    resolve_unnorm_key,
    validate_compatible_prefills,
)


PREFILL_OUTPUT_DIR = OUTPUT_DIR / "prefill_reuse"


def runtime_attention_report(model: Any) -> Dict[str, Any]:
    """Identify the live attention class and refuse an unrecognised update path.

    This is deliberately runtime inspection, not an assumption based on a
    pinned requirements file.  The source tokens are structural checks for the
    Llama path on which the DynamicCache intervention relies.
    """
    decoder = model.language_model.model
    layers = decoder.layers
    attention = layers[0].self_attn
    source = inspect.getsource(type(attention).forward)
    required = ("q_proj", "k_proj", "v_proj", "apply_rotary_pos_emb", "past_key_value.update")
    absent = [token for token in required if token not in source]
    if absent:
        raise RuntimeError(
            "The live attention implementation does not expose the expected Llama projection/RoPE/cache-update "
            f"path; refusing to patch a guessed implementation. Class={type(attention)!r}; missing={absent}."
        )
    return {
        "transformers_version": transformers.__version__,
        "language_model_class": f"{type(model.language_model).__module__}.{type(model.language_model).__qualname__}",
        "decoder_class": f"{type(decoder).__module__}.{type(decoder).__qualname__}",
        "attention_class": f"{type(attention).__module__}.{type(attention).__qualname__}",
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "text_attention_implementation": getattr(model.config.text_config, "_attn_implementation", None),
        "number_of_decoder_layers": len(layers),
        "verified_attention_operations": ["q_proj", "k_proj", "v_proj", "head reshape", "RoPE(Q,K)", "DynamicCache.update"],
    }


class DuringPrefillLanguageReuse:
    """Temporarily substitute selected post-RoPE K/V before attention consumes them.

    ``DynamicCache.update(key_states, value_states, layer_idx, ...)`` is called
    by each live attention module after RoPE and immediately before it receives
    the cached K/V return value.  During the full-length multimodal prefill we
    copy only ``[:, :, language_start:language_end+1, :]``.  Single-token action
    decoding updates do not match ``prefill_length`` and are deliberately left
    unchanged.
    """

    def __init__(
        self,
        previous_cache: DynamicCache,
        layout: Dict[str, List[int] | int],
        reuse_layers: Sequence[int],
        component: Optional[str],
    ) -> None:
        self.previous_cache = previous_cache
        self.prefill_length = int(layout["prefill_length"])
        self.language_start, self.language_end = layout["language"]  # inclusive
        self.reuse_layers = set(reuse_layers)
        self.component = component
        self.original_update: Optional[Any] = None
        self.diagnostics: Dict[str, Any] = {"inserted_layers": [], "layers": {}}

    def __enter__(self) -> "DuringPrefillLanguageReuse":
        self.original_update = DynamicCache.update
        previous_layers = cache_layers(self.previous_cache)

        def update(cache: DynamicCache, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
            # Action-token cache appends have sequence length 1.  Only the full
            # t+1 prefill is an intervention target.
            if self.component is not None and layer_idx in self.reuse_layers and key_states.shape[-2] == self.prefill_length:
                previous_k, previous_v = previous_layers[layer_idx]
                if key_states.shape != previous_k.shape or value_states.shape != previous_v.shape:
                    raise AssertionError(
                        f"Layer {layer_idx} update shape mismatch: current K/V {tuple(key_states.shape)}, "
                        f"{tuple(value_states.shape)} vs previous {tuple(previous_k.shape)}, {tuple(previous_v.shape)}"
                    )
                language = slice(self.language_start, self.language_end + 1)
                visual = slice(1, self.language_start)
                record: Dict[str, Any] = {}
                with torch.no_grad():
                    if self.component in {"k", "kv"}:
                        fresh_visual = key_states[:, :, visual, :].clone()
                        fresh_language = key_states[:, :, language, :].clone()
                        key_states[:, :, language, :].copy_(previous_k[:, :, language, :])
                        assert torch.equal(key_states[:, :, visual, :], fresh_visual), "K visual slots were modified."
                        assert torch.equal(key_states[:, :, language, :], previous_k[:, :, language, :]), "K insertion failed."
                        record["k_fresh_vs_stale_max_abs"] = float((fresh_language - previous_k[:, :, language, :]).abs().max())
                    if self.component in {"v", "kv"}:
                        fresh_visual = value_states[:, :, visual, :].clone()
                        fresh_language = value_states[:, :, language, :].clone()
                        value_states[:, :, language, :].copy_(previous_v[:, :, language, :])
                        assert torch.equal(value_states[:, :, visual, :], fresh_visual), "V visual slots were modified."
                        assert torch.equal(value_states[:, :, language, :], previous_v[:, :, language, :]), "V insertion failed."
                        record["v_fresh_vs_stale_max_abs"] = float((fresh_language - previous_v[:, :, language, :]).abs().max())
                self.diagnostics["inserted_layers"].append(layer_idx)
                self.diagnostics["layers"][str(layer_idx)] = record
            return self.original_update(cache, key_states, value_states, layer_idx, cache_kwargs)

        DynamicCache.update = update  # type: ignore[method-assign]
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        assert self.original_update is not None
        DynamicCache.update = self.original_update  # type: ignore[method-assign]

    def validate_completed_prefill(self, experimental_cache: DynamicCache) -> None:
        """Prove saved language values reached the cache, without saving tensors."""
        if self.component is None:
            assert not self.diagnostics["inserted_layers"]
            return
        assert sorted(self.diagnostics["inserted_layers"]) == sorted(self.reuse_layers), (
            "Not every selected layer was inserted exactly once during the full prefill: "
            f"{self.diagnostics['inserted_layers']}"
        )
        language = slice(self.language_start, self.language_end + 1)
        previous_layers = cache_layers(self.previous_cache)
        experimental_layers = cache_layers(experimental_cache)
        for layer in self.reuse_layers:
            previous_k, previous_v = previous_layers[layer]
            experimental_k, experimental_v = experimental_layers[layer]
            if self.component in {"k", "kv"}:
                assert torch.equal(experimental_k[:, :, language, :], previous_k[:, :, language, :]), (
                    f"Layer {layer} cached K does not contain the inserted stale language slots."
                )
            if self.component in {"v", "kv"}:
                assert torch.equal(experimental_v[:, :, language, :], previous_v[:, :, language, :]), (
                    f"Layer {layer} cached V does not contain the inserted stale language slots."
                )


class DuringPrefillLanguageKProjectionSkip:
    """Reuse post-RoPE language K without projecting those rows (Stage 2A)."""
    def __init__(self, model: Any, previous_cache: DynamicCache, layout: Dict[str, List[int] | int], reuse_layers: Sequence[int]) -> None:
        self.model, self.previous_cache, self.layout = model, previous_cache, layout
        self.prefill_length = int(layout["prefill_length"])
        self.language_start, self.language_end = layout["language"]
        self.reuse_layers = set(reuse_layers)
        self.original_forwards: Dict[int, Any] = {}
        self.diagnostics: Dict[str, Any] = {"skipped_layers": [], "k_proj_input_rows": [], "full_k_proj_calls": []}

    def __enter__(self) -> "DuringPrefillLanguageKProjectionSkip":
        import types
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

        previous_layers = cache_layers(self.previous_cache)
        modules = self.model.language_model.model.layers
        for layer_idx in self.reuse_layers:
            attention = modules[layer_idx].self_attn
            original = attention.forward
            self.original_forwards[layer_idx] = original

            def forward(module: Any, hidden_states: torch.Tensor, position_embeddings: Any, attention_mask: Any,
                        past_key_value: Optional[DynamicCache] = None, cache_position: Any = None, **kwargs: Any) -> Any:
                # Delegate refresh/full-normal and one-token decode paths untouched.
                if hidden_states.shape[-2] != self.prefill_length:
                    self.diagnostics["full_k_proj_calls"].append({"layer": module.layer_idx, "rows": int(hidden_states.shape[-2])})
                    return self.original_forwards[module.layer_idx](hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
                if past_key_value is None:
                    return self.original_forwards[module.layer_idx](hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)
                query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                prefix_hidden = hidden_states[:, :self.language_start, :]
                prefix_shape = (*prefix_hidden.shape[:-1], -1, module.head_dim)
                self.diagnostics["k_proj_input_rows"].append({"layer": module.layer_idx, "rows": int(prefix_hidden.shape[-2])})
                fresh_prefix_k = module.k_proj(prefix_hidden).view(prefix_shape).transpose(1, 2)
                cos, sin = position_embeddings
                query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
                _, fresh_prefix_k = apply_rotary_pos_emb(fresh_prefix_k, fresh_prefix_k, cos[:, :self.language_start], sin[:, :self.language_start])
                previous_k, _previous_v = previous_layers[module.layer_idx]
                if previous_k.shape[-2] != self.prefill_length:
                    raise AssertionError("Refresh-source prefill length mismatch.")
                stale_language_k = previous_k[:, :, self.language_start:self.language_end + 1, :]
                key_states = torch.cat((fresh_prefix_k, stale_language_k), dim=-2)
                if key_states.shape[-2] != self.prefill_length:
                    raise AssertionError("Reconstructed K has wrong prefill length.")
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, module.layer_idx, cache_kwargs)
                interface = eager_attention_forward if module.config._attn_implementation == "eager" else ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
                attn_output, attn_weights = interface(module, query_states, key_states, value_states, attention_mask,
                    dropout=0.0 if not module.training else module.attention_dropout, scaling=module.scaling, **kwargs)
                attn_output = module.o_proj(attn_output.reshape(*input_shape, -1).contiguous())
                self.diagnostics["skipped_layers"].append(module.layer_idx)
                return attn_output, attn_weights
            attention.forward = types.MethodType(forward, attention)
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        modules = self.model.language_model.model.layers
        for layer_idx, original in self.original_forwards.items():
            modules[layer_idx].self_attn.forward = original

    def validate_completed_prefill(self, cache: DynamicCache) -> None:
        assert sorted(self.diagnostics["skipped_layers"]) == sorted(self.reuse_layers)
        assert all(item["rows"] == self.language_start for item in self.diagnostics["k_proj_input_rows"])
        language = slice(self.language_start, self.language_end + 1)
        for layer in self.reuse_layers:
            current_k, _current_v = cache_layers(cache)[layer]
            previous_k, _previous_v = cache_layers(self.previous_cache)[layer]
            assert torch.equal(current_k[:, :, language, :], previous_k[:, :, language, :])


def run_prediction_with_prefill_snapshot(
    model: Any,
    inputs: Any,
    unnorm_key: str,
    intervention: Optional[DuringPrefillLanguageReuse],
) -> Dict[str, Any]:
    """Run full greedy predict_action and capture its initial prefill only.

    When ``intervention`` is provided, both the first action-token logits and
    later cached decoding belong to the modified prefill run.
    """
    context = intervention if intervention is not None else _NullContext()
    with context, PrefillSnapshot(model) as snapshot, torch.inference_mode():
        normal_action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    cache, logits, input_ids, layout = snapshot.require()
    return {"cache": cache, "logits": logits, "input_ids": input_ids, "layout": layout, "normal_action": normal_action}


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        return None


def capture_normal_observation(model: Any, processor: Any, prompt: str, image_path: Path, device: torch.device, unnorm_key: str) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    inputs = prepare_inputs(processor, prompt, image, device)
    return run_prediction_with_prefill_snapshot(model, inputs, unnorm_key, intervention=None)


def action_metrics(baseline_ids: torch.Tensor, reused_ids: torch.Tensor, baseline_action: np.ndarray, reused_action: np.ndarray) -> Dict[str, Any]:
    absolute = np.abs(reused_action - baseline_action)
    differing = (baseline_ids != reused_ids)[0]
    return {
        "baseline_action_token_ids": baseline_ids[0].detach().cpu().tolist(),
        "prefill_reuse_action_token_ids": reused_ids[0].detach().cpu().tolist(),
        "action_token_ids_exactly_equal": bool(torch.equal(baseline_ids, reused_ids)),
        "number_of_differing_action_tokens": int(differing.sum().item()),
        "differing_action_token_positions_1_indexed": (torch.nonzero(differing).flatten() + 1).detach().cpu().tolist(),
        "action_token_1_changed": bool(differing[0].item()),
        "baseline_action": np.asarray(baseline_action).tolist(),
        "prefill_reuse_action": np.asarray(reused_action).tolist(),
        "actions_exactly_equal": bool(np.array_equal(baseline_action, reused_action)),
        "action_difference_absolute_per_dimension": absolute.tolist(),
        "action_difference_l2": float(np.linalg.norm(reused_action - baseline_action)),
        "action_difference_max_absolute": float(absolute.max()),
    }


def evaluate_pair_from_snapshots(
    model: Any, processor: Any, device: torch.device, unnorm_key: str, action_dim: int,
    previous: Dict[str, Any], baseline: Dict[str, Any], current_image: Path, instruction: str,
    reuse_layers: Sequence[int], component: str,
) -> Dict[str, Any]:
    """Evaluate one pair using already-captured normal prefills.

    Batch mode passes the preceding pair's current baseline snapshot as
    ``previous``.  This is only an efficiency reuse of an already-normal
    prefill snapshot; it does not alter either the fresh baseline or the
    during-prefill intervention for the evaluated current image.
    """
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    validate_compatible_prefills(
        previous["cache"], baseline["cache"], previous["input_ids"], baseline["input_ids"],
        previous["layout"], baseline["layout"], reuse_layers,
    )

    # Exercise the same DynamicCache.update wrapper with no writes.  This proves
    # the new machinery itself does not perturb OpenVLA before interpretation.
    no_intervention = DuringPrefillLanguageReuse(previous["cache"], baseline["layout"], (), None)
    current = Image.open(current_image).convert("RGB")
    inputs = prepare_inputs(processor, prompt, current, device)
    no_intervention_run = run_prediction_with_prefill_snapshot(model, inputs, unnorm_key, no_intervention)
    no_intervention.validate_completed_prefill(no_intervention_run["cache"])
    if not np.array_equal(baseline["normal_action"], no_intervention_run["normal_action"]):
        raise AssertionError("No-intervention DynamicCache.update wrapper changed predict_action().")

    intervention = DuringPrefillLanguageReuse(previous["cache"], baseline["layout"], reuse_layers, component)
    inputs = prepare_inputs(processor, prompt, current, device)
    experimental = run_prediction_with_prefill_snapshot(model, inputs, unnorm_key, intervention)
    validate_compatible_prefills(
        previous["cache"], experimental["cache"], previous["input_ids"], experimental["input_ids"],
        previous["layout"], experimental["layout"], reuse_layers,
    )
    intervention.validate_completed_prefill(experimental["cache"])

    # Crucially these logits are captured from the *modified* prefill run, so
    # the first argmax below is action token 1 under the intervention.
    with torch.inference_mode():
        baseline_ids = greedy_decode_from_prefill(model, baseline["logits"], clone_prefill_cache(baseline["cache"]), action_dim)
        experimental_ids = greedy_decode_from_prefill(model, experimental["logits"], clone_prefill_cache(experimental["cache"]), action_dim)
    baseline_action = decode_action_token_ids(model, baseline_ids, unnorm_key)
    experimental_action = decode_action_token_ids(model, experimental_ids, unnorm_key)
    if not np.array_equal(baseline["normal_action"], baseline_action):
        raise AssertionError("Manual cached baseline does not reproduce normal predict_action().")
    if not np.array_equal(experimental["normal_action"], experimental_action):
        raise AssertionError("Manual decoder does not reproduce predict_action() from modified prefill logits/cache.")

    language_start, language_end = baseline["layout"]["language"]
    total_layers = len(cache_layers(baseline["cache"]))
    result = {
        "configuration": {
            "model_id": getattr(model.config, "_name_or_path", DEFAULT_MODEL_ID),
            "transformers_version": transformers.__version__,
            "instruction": instruction,
            "unnorm_key": unnorm_key,
            "reuse_layers": list(reuse_layers),
            "reuse_component": component,
            "intervention": "during_prefill_before_attention_via_DynamicCache_update",
        },
        "current_image": str(current_image),
        "layout": baseline["layout"],
        "cache": cache_summary(baseline["cache"]),
        "no_intervention_wrapper_matches_baseline": True,
        "normal_predict_action_matches_manual_baseline": True,
        "modified_prefill_predict_action_matches_manual_decoder": True,
        "insertion_diagnostics": intervention.diagnostics,
        "number_of_language_positions_reused": language_end - language_start + 1,
        "percentage_of_prefill_layer_position_slots_substituted": (
            100.0 * len(reuse_layers) * (language_end - language_start + 1) /
            (total_layers * int(baseline["layout"]["prefill_length"]))
        ),
    }
    result.update(action_metrics(baseline_ids, experimental_ids, baseline_action, experimental_action))
    return result


def evaluate_pair(
    model: Any, processor: Any, device: torch.device, unnorm_key: str, action_dim: int,
    previous_image: Path, current_image: Path, instruction: str, reuse_layers: Sequence[int], component: str,
) -> Dict[str, Any]:
    """Single-pair wrapper preserving the original CLI behavior."""
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    previous = capture_normal_observation(model, processor, prompt, previous_image, device, unnorm_key)
    baseline = capture_normal_observation(model, processor, prompt, current_image, device, unnorm_key)
    result = evaluate_pair_from_snapshots(
        model, processor, device, unnorm_key, action_dim, previous, baseline, current_image,
        instruction, reuse_layers, component,
    )
    result["previous_image"] = str(previous_image)
    return result


def discover_trajectory_frames(trajectory_dir: Path) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return contiguous ordered frames, preferring rollout metadata when present."""
    metadata_path = trajectory_dir / "trajectory_metadata.json"
    metadata: Optional[Dict[str, Any]] = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        frames = list(metadata.get("frames", []))
    else:
        frames = []
        for image_path in trajectory_dir.glob("step_*.png"):
            match = re.fullmatch(r"step_(\d+)\.png", image_path.name)
            if match:
                frames.append({"step": int(match.group(1)), "image": image_path.name})
    if len(frames) < 2:
        raise ValueError(f"Need at least two trajectory frames in {trajectory_dir}.")
    frames = sorted(frames, key=lambda frame: int(frame["step"]))
    for expected, frame in enumerate(frames, start=int(frames[0]["step"])):
        if int(frame["step"]) != expected:
            raise ValueError("Trajectory frame indices must be consecutive; refusing non-adjacent pairing.")
        image_path = trajectory_dir / frame["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Trajectory frame listed but absent: {image_path}")
    return frames, metadata


def summarize_trajectory(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate compact action-token and decoded-action sensitivity metrics."""
    if not results:
        raise ValueError("Cannot summarize an empty trajectory result set.")
    exact = np.asarray([item["action_token_ids_exactly_equal"] for item in results], dtype=bool)
    token_one = np.asarray([item["action_token_1_changed"] for item in results], dtype=bool)
    token_differences = np.asarray(
        [
            np.asarray(item["baseline_action_token_ids"])
            != np.asarray(item["prefill_reuse_action_token_ids"])
            for item in results
        ],
        dtype=bool,
    )
    expected_shape = (len(results), 7)
    assert token_differences.ndim == 2 and token_differences.shape == expected_shape, (
        f"Expected element-wise [pair, action-token] differences with shape {expected_shape}, "
        f"got {token_differences.shape}."
    )
    l2 = np.asarray([item["action_difference_l2"] for item in results], dtype=float)
    absolute = np.asarray([item["action_difference_absolute_per_dimension"] for item in results], dtype=float)
    max_index = int(np.argmax(l2))
    changed_pairs = [
        {
            "previous_step": item["previous_step"],
            "current_step": item["current_step"],
            "differing_action_token_positions_1_indexed": item["differing_action_token_positions_1_indexed"],
            "action_difference_l2": item["action_difference_l2"],
        }
        for item in results
        if not item["action_token_ids_exactly_equal"]
    ]
    count = len(results)
    return {
        "number_of_evaluated_pairs": count,
        "pairs_with_exact_7_token_equality": int(exact.sum()),
        "percentage_with_exact_7_token_equality": float(100.0 * exact.mean()),
        "pairs_with_any_action_token_change": int((~exact).sum()),
        "percentage_with_any_action_token_change": float(100.0 * (~exact).mean()),
        "pairs_with_action_token_1_change": int(token_one.sum()),
        "percentage_with_action_token_1_change": float(100.0 * token_one.mean()),
        "total_number_of_changed_action_tokens": int(token_differences.sum()),
        "changed_token_count_by_position_1_to_7": token_differences.sum(axis=0).astype(int).tolist(),
        "mean_decoded_action_l2_difference": float(l2.mean()),
        "maximum_decoded_action_l2_difference": float(l2.max()),
        "pair_with_maximum_l2_difference": {
            "previous_step": results[max_index]["previous_step"],
            "current_step": results[max_index]["current_step"],
            "action_difference_l2": float(l2[max_index]),
        },
        "mean_absolute_action_difference_per_dimension": absolute.mean(axis=0).tolist(),
        "maximum_absolute_action_difference_observed": float(absolute.max()),
        "changed_pairs": changed_pairs,
    }


def default_trajectory_output(reuse_layers_spec: str, component: str) -> Path:
    """Use configuration-specific filenames so K/V/KV runs never overwrite."""
    safe_layers = re.sub(r"[^0-9A-Za-z-]+", "_", reuse_layers_spec).strip("_")
    return PREFILL_OUTPUT_DIR / f"trajectory_layers_{safe_layers}_{component}.json"


def print_trajectory_summary(summary: Dict[str, Any]) -> None:
    print(f"Pairs evaluated: {summary['number_of_evaluated_pairs']}")
    print(
        "Exact 7-token matches: "
        f"{summary['pairs_with_exact_7_token_equality']} "
        f"({summary['percentage_with_exact_7_token_equality']:.2f}%)"
    )
    print(
        "Action-token-1 changes: "
        f"{summary['pairs_with_action_token_1_change']} "
        f"({summary['percentage_with_action_token_1_change']:.2f}%)"
    )
    print(f"Mean decoded-action L2: {summary['mean_decoded_action_l2_difference']:.8f}")
    print(f"Maximum decoded-action L2: {summary['maximum_decoded_action_l2_difference']:.8f}")
    for item in summary["changed_pairs"]:
        print(
            f"  {item['previous_step']:03d}->{item['current_step']:03d}: "
            f"changed tokens {item['differing_action_token_positions_1_indexed']}, "
            f"L2={item['action_difference_l2']:.8f}"
        )


def run_trajectory_batch(
    args: Any, model: Any, processor: Any, device: torch.device, unnorm_key: str,
    action_dim: int, reuse_layers: Sequence[int], report: Dict[str, Any],
) -> None:
    """Evaluate i->i+1 pairs offline; never steps or resets LIBERO."""
    frames, metadata = discover_trajectory_frames(args.trajectory_dir)
    instruction = args.instruction or (metadata or {}).get("task_description")
    if not instruction:
        raise ValueError("Trajectory mode needs --instruction when trajectory_metadata.json has no task_description.")
    if args.max_pairs is not None and args.max_pairs < 1:
        raise ValueError("--max-pairs must be >= 1.")
    pairs = list(zip(frames[:-1], frames[1:]))
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise ValueError("No adjacent trajectory pairs selected.")

    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    previous_frame = pairs[0][0]
    previous_snapshot = capture_normal_observation(
        model, processor, prompt, args.trajectory_dir / previous_frame["image"], device, unnorm_key
    )
    results: List[Dict[str, Any]] = []
    for previous_frame, current_frame in pairs:
        current_snapshot = capture_normal_observation(
            model, processor, prompt, args.trajectory_dir / current_frame["image"], device, unnorm_key
        )
        result = evaluate_pair_from_snapshots(
            model, processor, device, unnorm_key, action_dim, previous_snapshot, current_snapshot,
            args.trajectory_dir / current_frame["image"], instruction, reuse_layers, args.reuse_component,
        )
        result.update(
            {
                "previous_step": int(previous_frame["step"]),
                "current_step": int(current_frame["step"]),
                "previous_image": previous_frame["image"],
                "current_image": current_frame["image"],
            }
        )
        results.append(result)
        # This is the exact fresh baseline snapshot just computed for current;
        # retain it only because it becomes the next pair's previous observation.
        previous_snapshot = current_snapshot

    configuration = {
        "mode": "offline_adjacent_trajectory",
        "trajectory_dir": str(args.trajectory_dir),
        "instruction": instruction,
        "model_id": args.model_id,
        "unnorm_key": unnorm_key,
        "reuse_layers": list(reuse_layers),
        "reuse_component": args.reuse_component,
        "max_pairs": args.max_pairs,
        "pair_semantics": "Each result evaluates only adjacent i->i+1 recorded observations; no action is applied to LIBERO.",
        "intervention": "during_prefill_before_attention_via_DynamicCache_update",
    }
    summary = summarize_trajectory(results)
    output = args.batch_output or default_trajectory_output(args.reuse_layers, args.reuse_component)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "configuration": configuration,
                "runtime_attention_report": report,
                "per_pair_results": results,
                "batch_summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print_trajectory_summary(summary)
    print("Saved trajectory result:", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image-prev", type=Path, help="Previous image in single-pair mode.")
    parser.add_argument("--image-current", type=Path, help="Current image in single-pair mode.")
    parser.add_argument("--trajectory-dir", type=Path, help="Offline trajectory directory containing step_*.png frames.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Evaluate only the first N adjacent trajectory pairs.")
    parser.add_argument("--instruction", help="Required for single-pair mode; defaults to trajectory metadata in trajectory mode.")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--reuse-layers", default="0-13")
    parser.add_argument("--reuse-component", choices=("k", "v", "kv"), default="kv")
    parser.add_argument("--output", type=Path, default=PREFILL_OUTPUT_DIR / "prefill_reuse_result.json")
    parser.add_argument("--batch-output", type=Path, default=None, help="Optional trajectory JSON path; defaults to a configuration-specific filename.")
    args = parser.parse_args()
    single_pair = args.image_prev is not None or args.image_current is not None
    trajectory = args.trajectory_dir is not None
    if single_pair and trajectory:
        raise ValueError("Use either --image-prev/--image-current or --trajectory-dir, not both.")
    if not single_pair and not trajectory:
        raise ValueError("Specify either --image-prev plus --image-current, or --trajectory-dir.")
    if single_pair:
        if args.image_prev is None or args.image_current is None or not args.instruction:
            raise ValueError("Single-pair mode requires --image-prev, --image-current, and --instruction.")
        if not args.image_prev.is_file() or not args.image_current.is_file():
            raise FileNotFoundError("Both --image-prev and --image-current must be existing image files.")
    elif not args.trajectory_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {args.trajectory_dir}")

    model, processor, device = load_model_and_processor(args.model_id)
    report = runtime_attention_report(model)
    print("Installed transformers version:", report["transformers_version"])
    print("Active attention:", report["attention_class"], "(implementation:", report["attention_implementation"], ")")
    reuse_layers = parse_layer_spec(args.reuse_layers)
    unnorm_key = resolve_unnorm_key(model, args.unnorm_key)
    action_dim = model.get_action_dim(unnorm_key)
    if trajectory:
        run_trajectory_batch(args, model, processor, device, unnorm_key, action_dim, reuse_layers, report)
    else:
        result = evaluate_pair(
            model, processor, device, unnorm_key, action_dim, args.image_prev,
            args.image_current, args.instruction, reuse_layers, args.reuse_component,
        )
        result["runtime_attention_report"] = report
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("Baseline action-token IDs:", result["baseline_action_token_ids"])
        print("During-prefill-reuse action-token IDs:", result["prefill_reuse_action_token_ids"])
        print("Action token 1 changed:", result["action_token_1_changed"])
        print("Actions exactly equal:", result["actions_exactly_equal"])
        print(f"L2 action difference: {result['action_difference_l2']:.8f}")
        print("Inserted layers:", result["insertion_diagnostics"]["inserted_layers"])
        print("Saved compact result:", args.output)


if __name__ == "__main__":
    main()
