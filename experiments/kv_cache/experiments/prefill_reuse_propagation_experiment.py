"""V3: measure layer-wise propagation from V2 during-prefill KV reuse.

This is an offline diagnostic layered on top of the validated V2 intervention:
``DuringPrefillLanguageReuse`` still replaces selected post-RoPE/head-shaped
language K/V in ``DynamicCache.update()`` before attention consumes them.  It
does not skip projections, change V2's intervention semantics, or execute an
action in LIBERO.

For a fresh baseline and an intervened prefill of the same current observation,
V3 compares, layer by layer: decoder-layer input/output hidden states and raw
``q_proj``/``k_proj``/``v_proj`` outputs.  Projection hooks observe outputs
immediately after their linear projections and before head reshape/RoPE/cache
insertion.  Thus selected layers retain their *fresh pre-replacement* Q/K/V;
unselected later layers reveal propagation from modified earlier hidden states.

The saved metrics are full-tensor and language-token-only cosine similarity,
relative L2, and maximum absolute difference.  Raw tensors are held only
temporarily on CPU between one baseline/intervention pair and are never saved.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from experiments.kv_cache.experiments.prefill_reuse_experiment import (
    DuringPrefillLanguageReuse,
    action_metrics,
    capture_normal_observation,
    discover_trajectory_frames,
    runtime_attention_report,
)
from experiments.kv_cache.experiments.static_reuse_experiment import (
    DEFAULT_MODEL_ID,
    OUTPUT_DIR,
    PrefillSnapshot,
    cache_layers,
    clone_prefill_cache,
    decode_action_token_ids,
    greedy_decode_from_prefill,
    load_model_and_processor,
    parse_layer_spec,
    prepare_inputs,
    resolve_unnorm_key,
    validate_compatible_prefills,
)


PROPAGATION_OUTPUT_DIR = OUTPUT_DIR / "prefill_reuse_propagation"
METRIC_NAMES = ("hidden_input", "hidden_output", "q", "k", "v")


def _hidden_from_layer_output(output: Any) -> torch.Tensor:
    """LlamaDecoderLayer returns a hidden-state tuple in the active HF path."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Cannot obtain decoder-layer hidden states from {type(output)!r}.")


def _hidden_from_layer_input(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> torch.Tensor:
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    hidden = kwargs.get("hidden_states")
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise TypeError("Decoder layer did not receive a tensor hidden_states argument.")


def tensor_divergence(baseline: torch.Tensor, intervention: torch.Tensor, language: slice) -> Dict[str, Dict[str, float]]:
    """Compute scalar metrics over all elements and the shared language tokens.

    Both tensors have the same shape and live temporarily on CPU.  We flatten
    each selected region; no incompatible axes are averaged independently.
    """
    if baseline.shape != intervention.shape:
        raise AssertionError(f"Instrumentation tensor shape changed: {baseline.shape} != {intervention.shape}")

    def calculate(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
        reference = reference.float().reshape(-1)
        candidate = candidate.float().reshape(-1)
        delta = candidate - reference
        ref_norm = torch.linalg.vector_norm(reference)
        candidate_norm = torch.linalg.vector_norm(candidate)
        denominator = (ref_norm * candidate_norm).clamp_min(torch.finfo(torch.float32).eps)
        return {
            "cosine_similarity": float(torch.dot(reference, candidate) / denominator),
            "relative_l2": float(torch.linalg.vector_norm(delta) / ref_norm.clamp_min(torch.finfo(torch.float32).eps)),
            "max_absolute_difference": float(delta.abs().max()),
        }

    return {
        "full_tensor": calculate(baseline, intervention),
        "language_tokens_only": calculate(baseline[:, language, :], intervention[:, language, :]),
    }


class PrefillRepresentationCapture:
    """Temporary hooks for one full multimodal prefill and its action decoding.

    Hooks filter on ``sequence_length == prefill_length``.  The following
    single-token action-token forwards are intentionally ignored.  In baseline
    mode compact CPU clones are retained only until the matching intervention
    run; in comparison mode metrics are immediately reduced to scalars.
    """

    def __init__(
        self,
        model: Any,
        prefill_length: Optional[int],
        language_range: Optional[Sequence[int]],
        baseline_tensors: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> None:
        self.model = model
        self.prefill_length = prefill_length
        self.language = None if language_range is None else slice(language_range[0], language_range[1] + 1)
        self.baseline_tensors = baseline_tensors
        self.tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        self.metrics: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}
        self.handles: List[Any] = []

    @property
    def comparing(self) -> bool:
        return self.baseline_tensors is not None

    def _record(self, layer: int, name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim < 3 or tensor.shape[1] <= 1:
            return
        if self.prefill_length is None:
            self.prefill_length = int(tensor.shape[1])
        if tensor.shape[1] != self.prefill_length:
            return
        # CPU temporary storage avoids retaining every layer's raw activations on GPU.
        captured = tensor.detach().to(device="cpu", copy=True)
        if self.comparing:
            expected = self.baseline_tensors[layer][name]
            assert self.language is not None
            self.metrics.setdefault(layer, {})[name] = tensor_divergence(expected, captured, self.language)
        else:
            self.tensors.setdefault(layer, {})[name] = captured

    def __enter__(self) -> "PrefillRepresentationCapture":
        layers = self.model.language_model.model.layers
        for layer_index, layer in enumerate(layers):
            self.handles.append(
                layer.register_forward_pre_hook(
                    lambda _module, args, kwargs, index=layer_index: self._record(
                        index, "hidden_input", _hidden_from_layer_input(args, kwargs)
                    ),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, _kwargs, output, index=layer_index: self._record(
                        index, "hidden_output", _hidden_from_layer_output(output)
                    ),
                    with_kwargs=True,
                )
            )
            for name in ("q", "k", "v"):
                projection = getattr(layer.self_attn, f"{name}_proj")
                self.handles.append(
                    projection.register_forward_hook(
                        lambda _module, _args, _kwargs, output, index=layer_index, metric=name: self._record(
                            index, metric, output
                        ),
                        with_kwargs=True,
                    )
                )
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def validate(self) -> None:
        if self.prefill_length is None:
            raise AssertionError("Instrumentation did not observe a full multimodal prefill.")
        expected_layers = set(range(len(self.model.language_model.model.layers)))
        source = self.metrics if self.comparing else self.tensors
        if set(source) != expected_layers:
            raise AssertionError(f"Incomplete layer instrumentation: captured {sorted(source)}, expected {sorted(expected_layers)}")
        for layer, values in source.items():
            if set(values) != set(METRIC_NAMES):
                raise AssertionError(f"Layer {layer} missing captured metrics: {set(METRIC_NAMES) - set(values)}")


def predict_with_capture(
    model: Any, inputs: Any, unnorm_key: str, capture: PrefillRepresentationCapture,
    intervention: Optional[DuringPrefillLanguageReuse],
) -> Dict[str, Any]:
    """Run full greedy predict_action while V2 intervention and V3 hooks coexist."""
    cache_context = intervention if intervention is not None else contextlib.nullcontext()
    with cache_context, capture, PrefillSnapshot(model) as snapshot, torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    cache, logits, input_ids, layout = snapshot.require()
    capture.validate()
    return {"cache": cache, "logits": logits, "input_ids": input_ids, "layout": layout, "normal_action": action}


def capture_baseline(
    model: Any, processor: Any, prompt: str, image_path: Path, device: torch.device, unnorm_key: str,
) -> Tuple[Dict[str, Any], PrefillRepresentationCapture]:
    image = Image.open(image_path).convert("RGB")
    inputs = prepare_inputs(processor, prompt, image, device)
    # This unhooked prediction is the instrumentation control.  It is retained
    # only long enough to prove that decoder/projection hooks do not alter the
    # normal policy action.
    uninstrumented = capture_normal_observation(model, processor, prompt, image_path, device, unnorm_key)
    # Layer 0 sees the actual multimodal sequence first.  Let the hook discover
    # that full length instead of issuing a separate vision forward.
    capture = PrefillRepresentationCapture(model, None, None)
    snapshot = predict_with_capture(model, inputs, unnorm_key, capture, intervention=None)
    if not np.array_equal(uninstrumented["normal_action"], snapshot["normal_action"]):
        raise AssertionError("Representation instrumentation changed normal predict_action().")
    del uninstrumented
    if snapshot["layout"]["prefill_length"] != capture.prefill_length:
        raise AssertionError("Hook prefill length disagrees with DynamicCache layout.")
    return snapshot, capture


def run_no_intervention_check(model: Any, processor: Any, prompt: str, image_path: Path, device: torch.device, unnorm_key: str, previous_cache: Any, layout: Dict[str, Any], baseline_action: np.ndarray) -> None:
    """Keep V2's no-write DynamicCache wrapper equivalence assertion per pair."""
    image = Image.open(image_path).convert("RGB")
    inputs = prepare_inputs(processor, prompt, image, device)
    no_write = DuringPrefillLanguageReuse(previous_cache, layout, (), None)
    with no_write, PrefillSnapshot(model) as snapshot, torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    no_write.validate_completed_prefill(snapshot.require()[0])
    if not np.array_equal(baseline_action, action):
        raise AssertionError("No-intervention DynamicCache.update wrapper changed predict_action().")


def evaluate_pair(
    model: Any, processor: Any, device: torch.device, unnorm_key: str, action_dim: int,
    previous: Dict[str, Any], baseline: Dict[str, Any], baseline_capture: PrefillRepresentationCapture,
    current_image: Path, instruction: str, reuse_layers: Sequence[int], component: str,
) -> Dict[str, Any]:
    """Use V2's cache mechanism, then reduce V3 propagation into scalars."""
    validate_compatible_prefills(
        previous["cache"], baseline["cache"], previous["input_ids"], baseline["input_ids"],
        previous["layout"], baseline["layout"], reuse_layers,
    )
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    run_no_intervention_check(
        model, processor, prompt, current_image, device, unnorm_key, previous["cache"],
        baseline["layout"], baseline["normal_action"],
    )

    image = Image.open(current_image).convert("RGB")
    inputs = prepare_inputs(processor, prompt, image, device)
    intervention = DuringPrefillLanguageReuse(previous["cache"], baseline["layout"], reuse_layers, component)
    experimental_capture = PrefillRepresentationCapture(
        model, int(baseline["layout"]["prefill_length"]), baseline["layout"]["language"], baseline_capture.tensors
    )
    experimental = predict_with_capture(model, inputs, unnorm_key, experimental_capture, intervention)
    validate_compatible_prefills(
        previous["cache"], experimental["cache"], previous["input_ids"], experimental["input_ids"],
        previous["layout"], experimental["layout"], reuse_layers,
    )
    intervention.validate_completed_prefill(experimental["cache"])

    with torch.inference_mode():
        baseline_ids = greedy_decode_from_prefill(model, baseline["logits"], clone_prefill_cache(baseline["cache"]), action_dim)
        experimental_ids = greedy_decode_from_prefill(model, experimental["logits"], clone_prefill_cache(experimental["cache"]), action_dim)
    baseline_action = decode_action_token_ids(model, baseline_ids, unnorm_key)
    experimental_action = decode_action_token_ids(model, experimental_ids, unnorm_key)
    if not np.array_equal(baseline["normal_action"], baseline_action):
        raise AssertionError("Manual cached baseline does not reproduce normal predict_action().")
    if not np.array_equal(experimental["normal_action"], experimental_action):
        raise AssertionError("Manual decoder does not reproduce modified predict_action().")

    result = action_metrics(baseline_ids, experimental_ids, baseline_action, experimental_action)
    result.update(
        {
            "layer_propagation_metrics": {str(layer): metrics for layer, metrics in experimental_capture.metrics.items()},
            "insertion_diagnostics": intervention.diagnostics,
            "direct_intervention_layers": list(reuse_layers),
            "indirect_propagation_layers": [layer for layer in range(len(cache_layers(baseline["cache"]))) if layer not in reuse_layers],
            "validation": {
                "no_intervention_wrapper_matches_baseline": True,
                "manual_baseline_matches_predict_action": True,
                "manual_modified_prefill_matches_predict_action": True,
                "all_requested_layers_inserted_once": True,
            },
        }
    )
    return result


def aggregate_layer_metrics(results: Sequence[Dict[str, Any]], filter_changed: Optional[bool] = None) -> Dict[str, Any]:
    selected = [item for item in results if filter_changed is None or (not item["action_token_ids_exactly_equal"]) == filter_changed]
    if not selected:
        return {"pair_count": 0, "layers": {}}
    aggregate: Dict[str, Any] = {"pair_count": len(selected), "layers": {}}
    layer_indices = sorted(int(layer) for layer in selected[0]["layer_propagation_metrics"])
    for layer in layer_indices:
        layer_key = str(layer)
        aggregate["layers"][layer_key] = {}
        for name in METRIC_NAMES:
            aggregate["layers"][layer_key][name] = {}
            for region in ("full_tensor", "language_tokens_only"):
                aggregate["layers"][layer_key][name][region] = {}
                for metric in ("cosine_similarity", "relative_l2", "max_absolute_difference"):
                    values = np.asarray([item["layer_propagation_metrics"][layer_key][name][region][metric] for item in selected])
                    aggregate["layers"][layer_key][name][region][metric] = {
                        "mean": float(values.mean()), "median": float(np.median(values)), "max": float(values.max())
                    }
    return aggregate


def segment_summary(aggregate: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Summarize mean relative-L2 trajectory within direct or indirect layer ranges."""
    result: Dict[str, Any] = {"layers": [start, end], "metrics": {}}
    for name in METRIC_NAMES:
        values = [aggregate["layers"][str(layer)][name]["full_tensor"]["relative_l2"]["mean"] for layer in range(start, end + 1)]
        maximum_layer = start + int(np.argmax(values))
        result["metrics"][name] = {"mean_over_layers": float(np.mean(values)), "maximum": float(np.max(values)), "maximum_layer": maximum_layer}
    return result


def make_plots(aggregate: Dict[str, Any], output_stem: Path) -> List[str]:
    """Create four simple PNGs if matplotlib is available; never require it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    layers = sorted(int(layer) for layer in aggregate["layers"])
    for name in ("hidden_output", "q", "k", "v"):
        values = [aggregate["layers"][str(layer)][name]["full_tensor"]["relative_l2"]["mean"] for layer in layers]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(layers, values, marker="o", markersize=3)
        axis.axvline(13.5, color="red", linestyle="--", label="direct/indirect boundary")
        axis.set(xlabel="Decoder layer", ylabel="Mean relative L2", title=f"During-prefill propagation: {name}")
        axis.legend()
        path = output_stem.with_name(f"{output_stem.name}_{name}_relative_l2.png")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    return paths


def default_output(reuse_layers: str, component: str) -> Path:
    safe_layers = re.sub(r"[^0-9A-Za-z-]+", "_", reuse_layers).strip("_")
    return PROPAGATION_OUTPUT_DIR / f"trajectory_layers_{safe_layers}_{component}.json"


def run_trajectory(args: Any, model: Any, processor: Any, device: torch.device, unnorm_key: str, action_dim: int, reuse_layers: Sequence[int], report: Dict[str, Any]) -> None:
    frames, metadata = discover_trajectory_frames(args.trajectory_dir)
    instruction = args.instruction or (metadata or {}).get("task_description")
    if not instruction:
        raise ValueError("Trajectory mode needs --instruction when metadata lacks task_description.")
    if args.max_pairs < 1:
        raise ValueError("--max-pairs must be >= 1.")
    pairs = list(zip(frames[:-1], frames[1:]))[: args.max_pairs]
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    previous = capture_normal_observation(model, processor, prompt, args.trajectory_dir / pairs[0][0]["image"], device, unnorm_key)
    results: List[Dict[str, Any]] = []
    for previous_frame, current_frame in pairs:
        baseline, baseline_capture = capture_baseline(model, processor, prompt, args.trajectory_dir / current_frame["image"], device, unnorm_key)
        result = evaluate_pair(
            model, processor, device, unnorm_key, action_dim, previous, baseline, baseline_capture,
            args.trajectory_dir / current_frame["image"], instruction, reuse_layers, args.reuse_component,
        )
        result.update({"previous_step": int(previous_frame["step"]), "current_step": int(current_frame["step"]), "previous_image": previous_frame["image"], "current_image": current_frame["image"]})
        results.append(result)
        previous = baseline

    aggregate = aggregate_layer_metrics(results)
    summary = {
        "direct_layers_0_to_13": segment_summary(aggregate, 0, 13),
        "indirect_layers_14_to_31": segment_summary(aggregate, 14, 31),
        "changed_action_pairs": aggregate_layer_metrics(results, True),
        "unchanged_action_pairs": aggregate_layer_metrics(results, False),
    }
    output = args.output or default_output(args.reuse_layers, args.reuse_component)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_paths = make_plots(aggregate, output.with_suffix("")) if args.plots else []
    payload = {
        "configuration": {"mode": "offline_adjacent_trajectory", "trajectory_dir": str(args.trajectory_dir), "instruction": instruction, "model_id": args.model_id, "unnorm_key": unnorm_key, "reuse_layers": list(reuse_layers), "reuse_component": args.reuse_component, "max_pairs": args.max_pairs, "offline_caution": "Intervened actions are not executed in LIBERO; this measures representation/action sensitivity only."},
        "runtime_attention_report": report,
        "per_pair_results": results,
        "aggregate_per_layer": aggregate,
        "aggregate_summaries": summary,
        "plots": plot_paths,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    indirect = summary["indirect_layers_14_to_31"]["metrics"]
    print(f"Pairs evaluated: {len(results)}; exact actions: {sum(item['action_token_ids_exactly_equal'] for item in results)}/{len(results)}")
    for name in METRIC_NAMES:
        print(f"Indirect layers 14-31 {name}: max mean relative L2={indirect[name]['maximum']:.8e} at layer {indirect[name]['maximum_layer']}")
    print("Saved propagation result:", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--reuse-layers", default="0-13")
    parser.add_argument("--reuse-component", choices=("k", "v", "kv"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.trajectory_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {args.trajectory_dir}")
    model, processor, device = load_model_and_processor(args.model_id)
    report = runtime_attention_report(model)
    reuse_layers = parse_layer_spec(args.reuse_layers)
    unnorm_key = resolve_unnorm_key(model, args.unnorm_key)
    print("Installed transformers version:", report["transformers_version"])
    print("Active attention:", report["attention_class"])
    run_trajectory(args, model, processor, device, unnorm_key, model.get_action_dim(unnorm_key), reuse_layers, report)


if __name__ == "__main__":
    main()
