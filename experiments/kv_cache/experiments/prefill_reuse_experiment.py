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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def evaluate_pair(
    model: Any, processor: Any, device: torch.device, unnorm_key: str, action_dim: int,
    previous_image: Path, current_image: Path, instruction: str, reuse_layers: Sequence[int], component: str,
) -> Dict[str, Any]:
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    previous = capture_normal_observation(model, processor, prompt, previous_image, device, unnorm_key)
    baseline = capture_normal_observation(model, processor, prompt, current_image, device, unnorm_key)
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
        "previous_image": str(previous_image),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image-prev", required=True, type=Path)
    parser.add_argument("--image-current", required=True, type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--reuse-layers", default="0-13")
    parser.add_argument("--reuse-component", choices=("k", "v", "kv"), default="kv")
    parser.add_argument("--output", type=Path, default=PREFILL_OUTPUT_DIR / "prefill_reuse_result.json")
    args = parser.parse_args()
    if not args.image_prev.is_file() or not args.image_current.is_file():
        raise FileNotFoundError("Both --image-prev and --image-current must be existing image files.")

    model, processor, device = load_model_and_processor(args.model_id)
    report = runtime_attention_report(model)
    print("Installed transformers version:", report["transformers_version"])
    print("Active attention:", report["attention_class"], "(implementation:", report["attention_implementation"], ")")
    reuse_layers = parse_layer_spec(args.reuse_layers)
    unnorm_key = resolve_unnorm_key(model, args.unnorm_key)
    result = evaluate_pair(
        model, processor, device, unnorm_key, model.get_action_dim(unnorm_key), args.image_prev,
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
