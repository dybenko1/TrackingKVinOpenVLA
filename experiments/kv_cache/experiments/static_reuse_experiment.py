"""Static stale-language-KV intervention for two OpenVLA control observations.

This is deliberately *not* a compute-saving implementation.  It executes both
multimodal prefills normally, clones their immediate post-prefill DynamicCaches,
and substitutes selected language K/V slots from the previous observation into
the current observation's fresh cache before decoding action tokens.

Example (run inside the OpenVLA Docker environment):

    python -m experiments.kv_cache.experiments.static_reuse_experiment \
        --image-prev /path/to/step_t.png \
        --image-current /path/to/step_t_plus_1.png \
        --instruction "pick up the black bowl between the plate and the ramekin and place it on the plate" \
        --reuse-layers 0-13 --reuse-component kv

For an offline trajectory batch (no simulator interaction):

    python -m experiments.kv_cache.experiments.static_reuse_experiment \
        --trajectory-dir experiments/kv_analysis/inputs/trajectory_frames \
        --reuse-layers 0-13 --reuse-component kv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.cache_utils import DynamicCache


KV_CACHE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = KV_CACHE_ROOT / "outputs"
DEFAULT_MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
EMPTY_ACTION_PREFIX_TOKEN_ID = 29871


def parse_layer_spec(specification: str) -> List[int]:
    """Parse e.g. ``0-13,16,20-22`` into sorted unique layer indices."""
    layers = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending layer range: {item!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    if not layers:
        raise ValueError("--reuse-layers must select at least one layer.")
    return sorted(layers)


def cache_layers(cache: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if not isinstance(cache, DynamicCache):
        raise TypeError(
            "This experiment expects the live Transformers DynamicCache; got "
            f"{type(cache).__module__}.{type(cache).__qualname__}."
        )
    return list(zip(cache.key_cache, cache.value_cache))


def clone_prefill_cache(cache: DynamicCache) -> DynamicCache:
    """Create an independent cache snapshot before generate() appends action tokens."""
    cloned = DynamicCache()
    cloned.key_cache = [key.detach().clone() for key in cache.key_cache]
    cloned.value_cache = [value.detach().clone() for value in cache.value_cache]
    if hasattr(cloned, "_seen_tokens"):
        cloned._seen_tokens = cache.get_seq_length()  # type: ignore[attr-defined]
    return cloned


def cache_summary(cache: DynamicCache) -> Dict[str, Any]:
    layers = cache_layers(cache)
    if not layers:
        raise RuntimeError("Captured cache has no decoder layers.")
    return {
        "cache_type": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "num_layers": len(layers),
        "sequence_length": layers[0][0].shape[-2],
        "layers": [
            {
                "layer": index,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "key_dtype": str(key.dtype),
                "value_dtype": str(value.dtype),
                "key_device": str(key.device),
                "value_device": str(value.device),
            }
            for index, (key, value) in enumerate(layers)
        ],
    }


def layout_from_prefill(prefill_length: int, effective_prompt_length: int) -> Dict[str, List[int] | int]:
    visual_count = prefill_length - effective_prompt_length
    if visual_count < 0:
        raise AssertionError("Prefill cache is shorter than the effective prompt.")
    return {
        "prefill_length": prefill_length,
        "effective_prompt_length": effective_prompt_length,
        "visual_token_count": visual_count,
        "bos": [0, 0],
        "visual": [1, visual_count],
        "language": [visual_count + 1, prefill_length - 1],
    }


class PrefillSnapshot:
    """Capture a clone of exactly the first multimodal forward's cache and logits."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.handle: Optional[Any] = None
        self.cache: Optional[DynamicCache] = None
        self.last_prefill_logits: Optional[torch.Tensor] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.layout: Optional[Dict[str, List[int] | int]] = None

    def _hook(self, _module: Any, _args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> None:
        if self.cache is not None:
            return
        if kwargs.get("pixel_values") is None or kwargs.get("past_key_values") is not None:
            return
        input_ids = kwargs.get("input_ids")
        cache = getattr(output, "past_key_values", None)
        if input_ids is None or cache is None:
            raise RuntimeError("Multimodal prefill did not expose input_ids and past_key_values.")
        if not isinstance(cache, DynamicCache):
            raise TypeError(f"Expected DynamicCache from live model, got {type(cache)!r}.")

        # This hook runs before GenerationMixin consumes the prefill result and
        # extends its cache with action tokens.
        self.cache = clone_prefill_cache(cache)
        self.last_prefill_logits = output.logits[:, -1, :].detach().clone()
        self.input_ids = input_ids.detach().cpu().clone()
        self.layout = layout_from_prefill(cache.get_seq_length(), input_ids.shape[1])

    def __enter__(self) -> "PrefillSnapshot":
        self.handle = self.model.register_forward_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def require(self) -> Tuple[DynamicCache, torch.Tensor, torch.Tensor, Dict[str, List[int] | int]]:
        if self.cache is None or self.last_prefill_logits is None or self.input_ids is None or self.layout is None:
            raise RuntimeError("No multimodal prefill snapshot was captured.")
        return self.cache, self.last_prefill_logits, self.input_ids, self.layout


def prepare_inputs(processor: Any, prompt: str, image: Image.Image, device: torch.device) -> Any:
    inputs = processor(prompt, image)
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == EMPTY_ACTION_PREFIX_TOKEN_ID):
        empty_token = torch.full(
            (input_ids.shape[0], 1), EMPTY_ACTION_PREFIX_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device
        )
        inputs["input_ids"] = torch.cat((input_ids, empty_token), dim=1)
    if device.type == "cuda":
        inputs = inputs.to(device, dtype=torch.bfloat16)
    return inputs


def resolve_unnorm_key(model: Any, requested_key: Optional[str]) -> str:
    key = requested_key or "libero_spatial"
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    if key not in model.norm_stats:
        raise KeyError(f"Unknown action normalization key {key!r}; available: {sorted(model.norm_stats)}")
    return key


def validate_compatible_prefills(
    previous_cache: DynamicCache,
    current_cache: DynamicCache,
    previous_ids: torch.Tensor,
    current_ids: torch.Tensor,
    previous_layout: Dict[str, List[int] | int],
    current_layout: Dict[str, List[int] | int],
    reuse_layers: Sequence[int],
) -> None:
    assert torch.equal(previous_ids, current_ids), "Instruction tokenization differs between control steps."
    assert previous_layout == current_layout, f"Multimodal layouts differ: {previous_layout} != {current_layout}"
    previous_layers, current_layers = cache_layers(previous_cache), cache_layers(current_cache)
    assert len(previous_layers) == len(current_layers), "Cache layer counts differ."
    for layer in reuse_layers:
        assert 0 <= layer < len(current_layers), f"Requested layer {layer} is out of range."
    for layer, ((previous_k, previous_v), (current_k, current_v)) in enumerate(zip(previous_layers, current_layers)):
        assert previous_k.shape == current_k.shape, f"Layer {layer} K shapes differ."
        assert previous_v.shape == current_v.shape, f"Layer {layer} V shapes differ."
        assert previous_k.dtype == current_k.dtype and previous_v.dtype == current_v.dtype, f"Layer {layer} dtypes differ."
        assert previous_k.device == current_k.device and previous_v.device == current_v.device, f"Layer {layer} devices differ."


def substitute_language_slots(
    previous_cache: DynamicCache,
    fresh_current_cache: DynamicCache,
    language_range: Sequence[int],
    reuse_layers: Sequence[int],
    component: str,
) -> DynamicCache:
    """Return an independent current cache with only requested K/V language slots replaced."""
    modified = clone_prefill_cache(fresh_current_cache)
    language_start, language_end = language_range
    language_positions = slice(language_start, language_end + 1)
    previous_layers, modified_layers = cache_layers(previous_cache), cache_layers(modified)

    with torch.no_grad():
        for layer in reuse_layers:
            previous_k, previous_v = previous_layers[layer]
            modified_k, modified_v = modified_layers[layer]
            if component in {"k", "kv"}:
                modified_k[:, :, language_positions, :].copy_(previous_k[:, :, language_positions, :])
            if component in {"v", "kv"}:
                modified_v[:, :, language_positions, :].copy_(previous_v[:, :, language_positions, :])
    return modified


def greedy_decode_from_prefill(
    model: Any,
    prefill_last_logits: torch.Tensor,
    prefill_cache: DynamicCache,
    action_dim: int,
) -> torch.Tensor:
    """Continue ordinary cached, greedy one-token forwards from an existing prefill cache."""
    generated_ids: List[torch.Tensor] = []
    logits = prefill_last_logits
    for token_index in range(action_dim):
        next_token = torch.argmax(logits, dim=-1)
        generated_ids.append(next_token)
        if token_index + 1 == action_dim:
            break
        output = model(
            input_ids=next_token[:, None],
            past_key_values=prefill_cache,
            use_cache=True,
            return_dict=True,
        )
        logits = output.logits[:, -1, :]
    return torch.stack(generated_ids, dim=1)


def decode_action_token_ids(model: Any, token_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
    token_ids_np = token_ids[0].detach().cpu().numpy()
    discretized = model.vocab_size - token_ids_np
    discretized = np.clip(discretized - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized = model.bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


def load_model_and_processor(model_id: str) -> Tuple[Any, Any, torch.device]:
    print(f"Installed transformers version: {transformers.__version__}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not hasattr(model, "language_model"):
        raise TypeError("Expected the HF-exported OpenVLA model with language_model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), processor, device


def capture_observation(
    model: Any, processor: Any, prompt: str, image_path: Path, device: torch.device, unnorm_key: str
) -> Dict[str, Any]:
    """Run one normal prediction and retain only its action-free prefill snapshot."""
    image = Image.open(image_path).convert("RGB")
    inputs = prepare_inputs(processor, prompt, image, device)
    with PrefillSnapshot(model) as snapshot, torch.inference_mode():
        normal_action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    cache, logits, input_ids, layout = snapshot.require()
    return {"cache": cache, "logits": logits, "input_ids": input_ids, "layout": layout, "normal_action": normal_action}


def evaluate_pair(
    model: Any,
    previous: Dict[str, Any],
    current: Dict[str, Any],
    action_dim: int,
    unnorm_key: str,
    reuse_layers: Sequence[int],
    reuse_component: str,
    pair_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Perform one offline stale-language-KV counterfactual; never touches a simulator."""
    validate_compatible_prefills(
        previous["cache"], current["cache"], previous["input_ids"], current["input_ids"],
        previous["layout"], current["layout"], reuse_layers,
    )
    with torch.inference_mode():
        baseline_ids = greedy_decode_from_prefill(
            model, current["logits"], clone_prefill_cache(current["cache"]), action_dim
        )
        baseline_action = decode_action_token_ids(model, baseline_ids, unnorm_key)
        modified_cache = substitute_language_slots(
            previous["cache"], current["cache"], current["layout"]["language"], reuse_layers, reuse_component
        )
        reused_ids = greedy_decode_from_prefill(model, current["logits"], modified_cache, action_dim)
        reused_action = decode_action_token_ids(model, reused_ids, unnorm_key)

    if not np.array_equal(current["normal_action"], baseline_action):
        raise AssertionError("Manual cached baseline does not reproduce predict_action(); refusing intervention results.")

    absolute_difference = np.abs(reused_action - baseline_action)
    language_start, language_end = current["layout"]["language"]
    language_count = language_end - language_start + 1
    prefill_length = int(current["layout"]["prefill_length"])
    total_layers = len(cache_layers(current["cache"]))
    differing_tokens = int((baseline_ids != reused_ids).sum().item())
    result = {
        **pair_fields,
        "baseline_action_token_ids": baseline_ids[0].detach().cpu().tolist(),
        "reused_action_token_ids": reused_ids[0].detach().cpu().tolist(),
        "action_token_ids_exactly_equal": bool(torch.equal(baseline_ids, reused_ids)),
        "number_of_differing_action_tokens": differing_tokens,
        "baseline_action": np.asarray(baseline_action).tolist(),
        "reused_cache_action": np.asarray(reused_action).tolist(),
        "normal_predict_action_matches_manual_baseline": True,
        "actions_exactly_equal": bool(np.array_equal(baseline_action, reused_action)),
        "action_difference_absolute_per_dimension": absolute_difference.tolist(),
        "action_difference_l2": float(np.linalg.norm(reused_action - baseline_action)),
        "action_difference_max_absolute": float(absolute_difference.max()),
        "reused_layers": list(reuse_layers),
        "reuse_component": reuse_component,
        "number_of_language_positions_reused": language_count,
        "percentage_of_prefill_layer_position_slots_substituted": (
            100.0 * len(reuse_layers) * language_count / (total_layers * prefill_length)
        ),
        "layout": current["layout"],
    }
    return result


def select_adjacent_pairs(frames: Sequence[Dict[str, Any]], start: Optional[int], end: Optional[int], stride: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Select i->i+1 pairs; stride selects starting indices, never a temporal gap."""
    if stride < 1:
        raise ValueError("--stride must be >= 1.")
    by_step = {int(frame["step"]): frame for frame in frames}
    ordered_steps = sorted(by_step)
    if len(ordered_steps) < 2:
        raise ValueError("Trajectory metadata needs at least two frames.")
    first = ordered_steps[0] if start is None else start
    last = ordered_steps[-1] if end is None else end
    if first > last:
        raise ValueError("--start-step must be <= --end-step.")
    pairs = []
    for previous_step in range(first, last, stride):
        current_step = previous_step + 1
        if previous_step not in by_step or current_step not in by_step:
            raise ValueError(f"Missing consecutive trajectory frames for requested pair {previous_step}->{current_step}.")
        pairs.append((by_step[previous_step], by_step[current_step]))
    if not pairs:
        raise ValueError("No adjacent pairs selected; widen --start-step/--end-step.")
    return pairs


def summarize_batch(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    l2 = np.asarray([item["action_difference_l2"] for item in results])
    maximum = np.asarray([item["action_difference_max_absolute"] for item in results])
    absolute = np.asarray([item["action_difference_absolute_per_dimension"] for item in results])
    changed = (absolute > 0).sum(axis=0)
    differing_tokens = np.asarray([item["number_of_differing_action_tokens"] for item in results])
    exact_tokens = sum(item["action_token_ids_exactly_equal"] for item in results)
    return {
        "number_of_pairs_evaluated": len(results),
        "pairs_with_exact_action_token_sequences": int(exact_tokens),
        "percentage_with_exact_action_token_sequences": float(100.0 * exact_tokens / len(results)),
        "mean_action_l2_difference": float(l2.mean()),
        "median_action_l2_difference": float(np.median(l2)),
        "maximum_action_l2_difference": float(l2.max()),
        "mean_maximum_absolute_action_difference": float(maximum.mean()),
        "median_maximum_absolute_action_difference": float(np.median(maximum)),
        "maximum_observed_absolute_action_difference": float(maximum.max()),
        "mean_absolute_difference_per_action_dimension": absolute.mean(axis=0).tolist(),
        "pairs_where_each_action_dimension_changed": changed.astype(int).tolist(),
        "mean_number_of_differing_action_tokens": float(differing_tokens.mean()),
        "maximum_number_of_differing_action_tokens": int(differing_tokens.max()),
    }


def write_batch_outputs(
    output_dir: Path, configuration: Dict[str, Any], results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_batch(results)
    (output_dir / "results.json").write_text(json.dumps({"configuration": configuration, "results": results}, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps({"configuration": configuration, "summary": summary}, indent=2) + "\n")
    fieldnames = [
        "previous_step", "current_step", "previous_image", "current_image", "action_token_ids_exactly_equal",
        "number_of_differing_action_tokens", "actions_exactly_equal", "action_difference_l2",
        "action_difference_max_absolute", "number_of_language_positions_reused",
        "percentage_of_prefill_layer_position_slots_substituted", "reuse_component",
    ]
    action_dim = len(results[0]["baseline_action"])
    fieldnames += [f"absolute_action_difference_{index}" for index in range(action_dim)]
    with (output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {key: item[key] for key in fieldnames if key in item}
            row.update({f"absolute_action_difference_{index}": value for index, value in enumerate(item["action_difference_absolute_per_dimension"])})
            writer.writerow(row)
    return summary


def print_batch_summary(metadata: Dict[str, Any], results: Sequence[Dict[str, Any]], summary: Dict[str, Any], output_dir: Path) -> None:
    print(f"Trajectory: {metadata.get('task_suite')} / task {metadata.get('task_id')}")
    print(f"Pairs evaluated: {summary['number_of_pairs_evaluated']}")
    print(f"Reuse: layers {results[0]['reused_layers']}, component {results[0]['reuse_component']}")
    print(f"Exact token-sequence matches: {summary['pairs_with_exact_action_token_sequences']} ({summary['percentage_with_exact_action_token_sequences']:.2f}%)")
    print(f"Mean action L2: {summary['mean_action_l2_difference']:.8f}")
    print(f"Median action L2: {summary['median_action_l2_difference']:.8f}")
    print(f"Maximum action L2: {summary['maximum_action_l2_difference']:.8f}")
    print(f"Mean absolute difference by dimension: {summary['mean_absolute_difference_per_action_dimension']}")
    print("Largest action-L2 pairs:")
    for item in sorted(results, key=lambda result: result["action_difference_l2"], reverse=True)[:5]:
        print(f"  {item['previous_step']:03d}->{item['current_step']:03d}: {item['action_difference_l2']:.8f}")
    print(f"Results saved to: {output_dir.resolve()}")


def run_single_pair(args: Any, model: Any, processor: Any, device: torch.device, unnorm_key: str, action_dim: int, reuse_layers: Sequence[int]) -> None:
    if args.image_prev is None or args.image_current is None or args.instruction is None:
        raise ValueError("Single-pair mode requires --image-prev, --image-current, and --instruction.")
    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"
    previous = capture_observation(model, processor, prompt, args.image_prev, device, unnorm_key)
    current = capture_observation(model, processor, prompt, args.image_current, device, unnorm_key)
    result = evaluate_pair(
        model, previous, current, action_dim, unnorm_key, reuse_layers, args.reuse_component,
        {"previous_image": str(args.image_prev), "current_image": str(args.image_current)},
    )
    result.update(
        {
            "configuration": {
                "mode": "single_pair",
                "model_id": args.model_id,
                "transformers_version": transformers.__version__,
                "instruction": args.instruction,
                "unnorm_key": unnorm_key,
            },
            "step_t_normal_action": np.asarray(previous["normal_action"]).tolist(),
            "normal_current_predict_action": np.asarray(current["normal_action"]).tolist(),
            "normal_predict_action_matches_manual_baseline": True,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Baseline action: {result['baseline_action']}")
    print(f"Reused-cache action: {result['reused_cache_action']}")
    print(f"Actions exactly equal: {result['actions_exactly_equal']}")
    print(f"L2 action difference: {result['action_difference_l2']:.8f}")
    print(f"Saved compact result: {args.output}")


def run_trajectory_batch(args: Any, model: Any, processor: Any, device: torch.device, unnorm_key: str, action_dim: int, reuse_layers: Sequence[int]) -> None:
    metadata_path = args.trajectory_dir / "trajectory_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    instruction = metadata["task_description"]
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    pairs = select_adjacent_pairs(metadata["frames"], args.start_step, args.end_step, args.stride)
    results: List[Dict[str, Any]] = []
    cached_step: Optional[int] = None
    previous_snapshot: Optional[Dict[str, Any]] = None

    for previous_frame, current_frame in pairs:
        previous_step, current_step = int(previous_frame["step"]), int(current_frame["step"])
        if cached_step != previous_step:
            previous_snapshot = capture_observation(
                model, processor, prompt, args.trajectory_dir / previous_frame["image"], device, unnorm_key
            )
        assert previous_snapshot is not None
        current_snapshot = capture_observation(
            model, processor, prompt, args.trajectory_dir / current_frame["image"], device, unnorm_key
        )
        result = evaluate_pair(
            model, previous_snapshot, current_snapshot, action_dim, unnorm_key, reuse_layers, args.reuse_component,
            {
                "previous_step": previous_step,
                "current_step": current_step,
                "previous_image": previous_frame["image"],
                "current_image": current_frame["image"],
                "rollout_current_policy_action_before_gripper_conversion": current_frame.get("policy_action_before_gripper_conversion"),
                "rollout_current_environment_action": current_frame.get("action"),
            },
        )
        results.append(result)
        # Retain exactly this current snapshot only when it becomes the next pair's previous observation.
        previous_snapshot, cached_step = current_snapshot, current_step

    configuration = {
        "mode": "trajectory", "model_id": args.model_id, "transformers_version": transformers.__version__,
        "trajectory_dir": str(args.trajectory_dir), "instruction": instruction, "unnorm_key": unnorm_key,
        "reuse_layers": list(reuse_layers), "reuse_component": args.reuse_component,
        "start_step": args.start_step, "end_step": args.end_step, "stride": args.stride,
        "pair_semantics": "Each selected i evaluates i->i+1; stride selects i, never a temporal gap.",
    }
    summary = write_batch_outputs(args.batch_output_dir, configuration, results)
    print_batch_summary(metadata, results, summary, args.batch_output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image-prev", type=Path, help="Single-pair previous observation.")
    parser.add_argument("--image-current", type=Path, help="Single-pair current observation.")
    parser.add_argument("--instruction", help="Required only in single-pair mode.")
    parser.add_argument("--trajectory-dir", type=Path, help="Directory containing trajectory_metadata.json and step images.")
    parser.add_argument("--start-step", type=int, default=None, help="Inclusive previous-frame index in trajectory mode.")
    parser.add_argument("--end-step", type=int, default=None, help="Inclusive current-frame index in trajectory mode.")
    parser.add_argument("--stride", type=int, default=1, help="Selects pair starts: i->i+1, i+=stride.")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--reuse-layers", default="0-13")
    parser.add_argument("--reuse-component", choices=("kv", "k", "v"), default="kv")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "static_reuse_result.json")
    parser.add_argument("--batch-output-dir", type=Path, default=OUTPUT_DIR / "trajectory_batch")
    args = parser.parse_args()
    if args.trajectory_dir is not None and (args.image_prev is not None or args.image_current is not None):
        raise ValueError("Use either --trajectory-dir or the single-pair image arguments, not both.")
    if args.trajectory_dir is not None:
        metadata_path = args.trajectory_dir / "trajectory_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Trajectory metadata not found: {metadata_path}. Generate the continuous rollout first with "
                "`python -m experiments.kv_analysis.experiments.extract_libero_image --mode rollout "
                f"--output-dir {args.trajectory_dir}`."
            )

    model, processor, device = load_model_and_processor(args.model_id)
    unnorm_key = resolve_unnorm_key(model, args.unnorm_key)
    action_dim = model.get_action_dim(unnorm_key)
    reuse_layers = parse_layer_spec(args.reuse_layers)
    if args.trajectory_dir is not None:
        run_trajectory_batch(args, model, processor, device, unnorm_key, action_dim, reuse_layers)
    else:
        run_single_pair(args, model, processor, device, unnorm_key, action_dim, reuse_layers)


if __name__ == "__main__":
    main()
