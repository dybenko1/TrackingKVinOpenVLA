"""Capture the true OpenVLA multimodal-prefill attention cache.

Run from the repository root with the OpenVLA/LIBERO Python environment:

    python -m experiments.kv_analysis.experiments.inspect_cache \
        --image experiments/kv_analysis/inputs/imagenLIBERO.png \
        --instruction "pick up the black bowl between the plate and the ramekin and place it on the plate"

This is observation-only instrumentation.  It uses forward hooks to inspect the
first multimodal VLA forward performed by ``predict_action`` and does not alter
the model inputs, outputs, cache, or generation configuration.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
import transformers.cache_utils as cache_utils
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[3]
KV_ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = KV_ANALYSIS_ROOT / "outputs" / "data"
DEFAULT_MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
EMPTY_ACTION_PREFIX_TOKEN_ID = 29871


def _cache_layers(cache: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Normalize legacy tuples and Cache-like objects into per-layer K/V pairs."""
    if cache is None:
        raise RuntimeError("The multimodal prefill returned no past_key_values.")

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))

    return list(cache)


def _cache_description(cache: Any) -> Dict[str, Any]:
    layers = _cache_layers(cache)
    if not layers:
        raise RuntimeError("The multimodal prefill cache has no layers.")

    description = {
        "cache_type": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "num_layers": len(layers),
        "layers": [],
    }
    for layer_idx, (key, value) in enumerate(layers):
        description["layers"].append(
            {
                "layer": layer_idx,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "key_dtype": str(key.dtype),
                "value_dtype": str(value.dtype),
                "key_device": str(key.device),
                "value_device": str(value.device),
                "cached_sequence_length": key.shape[-2],
            }
        )
    return description


def _compact_sample(
    cache: Any, positions: Sequence[int]
) -> Dict[str, Any]:
    """Keep a tiny, CPU-resident sample rather than retaining a full cache."""
    key, value = _cache_layers(cache)[0]
    sample_positions = torch.tensor(positions, device=key.device, dtype=torch.long)
    feature_count = min(8, key.shape[-1])
    return {
        "layer": 0,
        "positions": list(positions),
        "key": key.index_select(-2, sample_positions)[..., :feature_count].detach().cpu(),
        "value": value.index_select(-2, sample_positions)[..., :feature_count].detach().cpu(),
    }


def _position_layout(prefill_length: int, prompt_length: int) -> Dict[str, Any]:
    visual_count = prefill_length - prompt_length
    if visual_count < 0:
        raise RuntimeError(
            f"Cache length ({prefill_length}) is shorter than prompt length ({prompt_length})."
        )

    language_start = visual_count + 1
    return {
        "prefill_cache_length": prefill_length,
        "effective_prompt_token_count": prompt_length,
        "visual_token_count": visual_count,
        "bos": [0, 0],
        "visual": [1, visual_count],
        "language": [language_start, prefill_length - 1],
        # The first generated action token is selected from prefill logits but is
        # not itself in this immediate post-prefill cache.
        "generated_action_tokens_in_prefill_cache": 0,
    }


class PrefillCacheObserver:
    """Observe the first multimodal VLA forward and one raw layer-0 K projection."""

    def __init__(self, model: Any, original_prompt_length: int) -> None:
        self.model = model
        self.original_prompt_length = original_prompt_length
        self.handles: List[Any] = []
        self.raw_layer0_k: Optional[torch.Tensor] = None
        self.raw_layer0_v: Optional[torch.Tensor] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.sample: Optional[Dict[str, Any]] = None
        self.rope_check: Optional[Dict[str, Any]] = None

    def _record_raw_k(self, _module: Any, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
        if self.raw_layer0_k is None:
            self.raw_layer0_k = output.detach()

    def _record_raw_v(self, _module: Any, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
        if self.raw_layer0_v is None:
            self.raw_layer0_v = output.detach()

    def _record_prefill(
        self, _module: Any, _args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any
    ) -> None:
        if self.metadata is not None:
            return
        if kwargs.get("past_key_values") is not None or kwargs.get("pixel_values") is None:
            return

        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Expected input_ids on the multimodal prefill forward.")
        cache = getattr(output, "past_key_values", None)
        description = _cache_description(cache)
        prefill_length = description["layers"][0]["cached_sequence_length"]
        effective_prompt_length = input_ids.shape[1]
        layout = _position_layout(prefill_length, effective_prompt_length)

        positions = sorted(
            {
                0,
                1,
                layout["visual"][1],
                layout["language"][0],
                prefill_length - 1,
            }
        )
        positions = [position for position in positions if 0 <= position < prefill_length]

        self.sample = _compact_sample(cache, positions)
        self.rope_check = self._verify_rope(cache)
        self.metadata = {
            "transformers_version": transformers.__version__,
            "model_type": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "language_model_type": (
                f"{type(self.model.language_model).__module__}."
                f"{type(self.model.language_model).__qualname__}"
            ),
            "language_model_source": inspect.getsourcefile(type(self.model.language_model)),
            "cache_utils_source": inspect.getsourcefile(cache_utils),
            "cache": description,
            "layout": layout,
            "original_processor_prompt_token_count": self.original_prompt_length,
            "extra_prompt_token_added_by_predict_action": effective_prompt_length - self.original_prompt_length,
            "representative_cache_positions": positions,
            "layer0_key_rope_check": self.rope_check,
        }

    def register(self) -> None:
        layers = self.model.language_model.model.layers
        layer0_attention = layers[0].self_attn
        self.handles.extend(
            [
                layer0_attention.k_proj.register_forward_hook(self._record_raw_k),
                layer0_attention.v_proj.register_forward_hook(self._record_raw_v),
                self.model.register_forward_hook(self._record_prefill, with_kwargs=True),
            ]
        )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _verify_rope(self, cache: Any) -> Dict[str, Any]:
        """Check layer-0 cached K against both raw and RoPE-rotated K."""
        if self.raw_layer0_k is None or self.raw_layer0_v is None:
            return {"verified": False, "reason": "layer-0 projection hook did not fire"}

        try:
            cached_key, _cached_value = _cache_layers(cache)[0]
            attention = self.model.language_model.model.layers[0].self_attn
            batch_size, query_length, _ = self.raw_layer0_k.shape
            num_kv_heads = cached_key.shape[1]
            head_dim = cached_key.shape[-1]
            raw_key = self.raw_layer0_k.view(batch_size, query_length, num_kv_heads, head_dim).transpose(1, 2)
            raw_value = self.raw_layer0_v.view(batch_size, query_length, num_kv_heads, head_dim).transpose(1, 2)
            position_ids = torch.arange(query_length, device=raw_key.device).unsqueeze(0)
            # Transformers <=4.40 stores RoPE on each attention module; 4.49
            # computes it once on LlamaModel and passes (cos, sin) to every layer.
            rotary_owner = attention
            if not hasattr(rotary_owner, "rotary_emb"):
                rotary_owner = self.model.language_model.model
            cos, sin = rotary_owner.rotary_emb(raw_value, position_ids)
            llama_module = inspect.getmodule(type(attention))
            apply_rope = getattr(llama_module, "apply_rotary_pos_emb", None)
            if apply_rope is None:
                return {
                    "verified": False,
                    "reason": "could not locate apply_rotary_pos_emb on the live attention module",
                }

            # The helper rotates Q and K; a zero Q is sufficient when only K is inspected.
            zero_query = torch.zeros(
                batch_size,
                self.model.language_model.config.num_attention_heads,
                query_length,
                head_dim,
                device=raw_key.device,
                dtype=raw_key.dtype,
            )
            _unused_query, rope_key = apply_rope(zero_query, raw_key, cos, sin)
            cached_prefill_key = cached_key[..., :query_length, :]
            raw_difference = (cached_prefill_key.float() - raw_key.float()).abs().max().item()
            rope_difference = (cached_prefill_key.float() - rope_key.float()).abs().max().item()
            return {
                "verified": bool(torch.allclose(cached_prefill_key, rope_key, rtol=1e-3, atol=1e-3)),
                "raw_k_proj_shape": list(self.raw_layer0_k.shape),
                "reshaped_raw_k_shape": list(raw_key.shape),
                "cached_k_shape": list(cached_prefill_key.shape),
                "max_abs_difference_raw_k_vs_cache": raw_difference,
                "max_abs_difference_rope_k_vs_cache": rope_difference,
                "explanation": (
                    "The hook sees [batch, sequence, num_kv_heads * head_dim]. "
                    "The cache stores [batch, num_kv_heads, sequence, head_dim] after RoPE on K."
                ),
            }
        except Exception as error:  # Keep inspection compatible with the live installed implementation.
            return {"verified": False, "reason": f"RoPE verification failed: {type(error).__name__}: {error}"}


def _prepare_inputs(processor: Any, prompt: str, image: Image.Image, device: torch.device) -> Tuple[Any, int]:
    inputs = processor(prompt, image)
    prompt_length = inputs["input_ids"].shape[1]
    if device.type == "cuda":
        inputs = inputs.to(device, dtype=torch.bfloat16)
    return inputs, prompt_length


def _resolve_unnorm_key(model: Any, requested_key: Optional[str]) -> str:
    key = requested_key or "libero_spatial"
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    if key not in model.norm_stats:
        raise KeyError(f"Unknown action normalization key {key!r}; available: {sorted(model.norm_stats)}")
    return key


def _print_cache_report(metadata: Dict[str, Any]) -> None:
    cache = metadata["cache"]
    print(f"Transformers: {metadata['transformers_version']}")
    print(f"Language model: {metadata['language_model_type']}")
    print(f"Language-model source: {metadata['language_model_source']}")
    print(f"Cache utilities source: {metadata['cache_utils_source']}")
    print(f"Cache: {cache['cache_type']} ({cache['num_layers']} layers)")
    for layer in cache["layers"]:
        print(
            f"  layer {layer['layer']:2d}: K={layer['key_shape']} {layer['key_dtype']} {layer['key_device']}; "
            f"V={layer['value_shape']} {layer['value_dtype']} {layer['value_device']}; "
            f"length={layer['cached_sequence_length']}"
        )
    layout = metadata["layout"]
    print("Multimodal cache layout:")
    print(
        f"  BOS={layout['bos']}; visual={layout['visual']}; language={layout['language']}; "
        f"generated action tokens in prefill cache={layout['generated_action_tokens_in_prefill_cache']}"
    )
    print("Layer-0 K RoPE check:", metadata["layer0_key_rope_check"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--image",
        type=Path,
        default=KV_ANALYSIS_ROOT / "inputs" / "imagenLIBERO.png",
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--unnorm-key", default=None)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"

    print(f"Installed transformers version: {transformers.__version__}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if not hasattr(model, "language_model"):
        raise TypeError("Expected the HF-exported OpenVLA model with a language_model attribute.")
    unnorm_key = _resolve_unnorm_key(model, args.unnorm_key)

    # Baseline and instrumented calls use independently prepared but identical inputs.
    baseline_inputs, _ = _prepare_inputs(processor, prompt, image, device)
    with torch.inference_mode():
        baseline_action = model.predict_action(**baseline_inputs, unnorm_key=unnorm_key, do_sample=False)

    instrumented_inputs, original_prompt_length = _prepare_inputs(processor, prompt, image, device)
    observer = PrefillCacheObserver(model, original_prompt_length)
    observer.register()
    try:
        with torch.inference_mode():
            instrumented_action = model.predict_action(**instrumented_inputs, unnorm_key=unnorm_key, do_sample=False)
    finally:
        observer.remove()

    if observer.metadata is None or observer.sample is None:
        raise RuntimeError("The observer did not capture a multimodal prefill cache.")
    action_equal = np.array_equal(baseline_action, instrumented_action)
    action_close = np.allclose(baseline_action, instrumented_action)
    observer.metadata.update(
        {
            "model_id": args.model_id,
            "image": str(args.image),
            "instruction": args.instruction,
            "unnorm_key": unnorm_key,
            "baseline_action": np.asarray(baseline_action).tolist(),
            "instrumented_action": np.asarray(instrumented_action).tolist(),
            "actions_exactly_equal": bool(action_equal),
            "actions_allclose": bool(action_close),
        }
    )
    _print_cache_report(observer.metadata)
    if not observer.metadata["layer0_key_rope_check"].get("verified", False):
        raise AssertionError("Could not verify that the captured layer-0 cache K is post-RoPE.")
    print(f"Baseline action: {baseline_action}")
    print(f"Instrumented action: {instrumented_action}")
    print(f"Actions exactly equal: {action_equal}; allclose: {action_close}")
    if not action_equal:
        raise AssertionError("Instrumentation changed the greedy action output.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / "prefill_cache_metadata.json"
    sample_path = OUTPUT_DIR / "prefill_cache_sample.pt"
    metadata_path.write_text(json.dumps(observer.metadata, indent=2) + "\n")
    torch.save(observer.sample, sample_path)
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved small layer-0 cache sample: {sample_path}")


if __name__ == "__main__":
    main()
