"""Stage 2D-0: adjacent-step visual-token K/V temporal stability.

This is observation-only: it drives a normal fresh OpenVLA policy, snapshots
the post-prefill DynamicCache, and never changes model inputs, cache contents,
or actions.  Cache K is post-RoPE, as established by inspect_cache.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from libero.libero import benchmark

from experiments.kv_analysis.experiments.extract_libero_image import (
    DEFAULT_MODEL_ID, DEFAULT_STEP_ACTION, TASK_SUITE_NAME, create_libero_env,
    extract_libero_image, normalize_and_invert_openvla_gripper,
    prepare_openvla_policy_image,
)
from experiments.kv_cache.experiments.static_reuse_experiment import (
    PrefillSnapshot, cache_layers, load_model_and_processor, prepare_inputs,
    resolve_unnorm_key,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "visual_kv_stability"
LAYERS = tuple(range(32))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exactly the primary cosine definition used by analyze_language_kv.py."""
    return F.cosine_similarity(a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)).item()


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exactly the relative-L2 definition used by analyze_language_kv.py."""
    return (torch.norm(b.float() - a.float()) / torch.norm(a.float())).item()


def stats(values: np.ndarray) -> Dict[str, float | int]:
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "std": float(np.std(values)), "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)), "min": float(np.min(values)),
        "max": float(np.max(values)), "count": int(values.size),
    }


def policy_image(observation: Dict[str, Any], center_crop: bool) -> np.ndarray:
    # Same evaluator-compatible resize/crop path used by validated rollouts.
    import tensorflow as tf
    image = tf.image.encode_jpeg(extract_libero_image(observation))
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (224, 224), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
    return prepare_openvla_policy_image(image, center_crop=center_crop)


def capture_fresh_prefill(model: Any, processor: Any, image: np.ndarray, prompt: str,
                          device: torch.device, unnorm_key: str) -> Dict[str, Any]:
    """Run untouched fresh greedy policy inference and snapshot only its prefill."""
    inputs = prepare_inputs(processor, prompt, Image.fromarray(image).convert("RGB"), device)
    with PrefillSnapshot(model) as snapshot, torch.inference_mode():
        action = np.asarray(model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False))
    cache, _logits, input_ids, layout = snapshot.require()
    if len(cache_layers(cache)) != 32:
        raise AssertionError("Expected 32 decoder layers in the prefill cache.")
    if int(layout["prefill_length"]) != int(cache.get_seq_length()):
        raise AssertionError("Derived prefill layout disagrees with DynamicCache length.")
    return {"cache": cache, "layout": layout, "input_ids": input_ids, "policy_action": action}


def visual_tensors(cache: Any, layout: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
    """Copy only BOS-excluded visual cache positions to CPU, preserving [S,H,D]."""
    visual_count = int(layout["visual_token_count"])
    if visual_count <= 0:
        raise AssertionError("Derived visual token count must be positive.")
    visual_start, visual_end = layout["visual"]
    if (visual_start, visual_end) != (1, visual_count):
        raise AssertionError(f"Unexpected BOS|visual layout: {layout['visual']!r}")
    tensors: List[Dict[str, torch.Tensor]] = []
    for layer, (key, value) in enumerate(cache_layers(cache)):
        if key.shape[-2] != int(layout["prefill_length"]) or value.shape[-2] != key.shape[-2]:
            raise AssertionError(f"Layer {layer}: inconsistent cache sequence length.")
        # DynamicCache: [batch, kv_heads, sequence, head_dim]; select [1:1+V].
        k = key[:, :, visual_start:visual_end + 1, :].permute(0, 2, 1, 3).detach().float().cpu()
        v = value[:, :, visual_start:visual_end + 1, :].permute(0, 2, 1, 3).detach().float().cpu()
        if k.shape[1] != visual_count or v.shape != k.shape:
            raise AssertionError(f"Layer {layer}: visual K/V shape mismatch: {tuple(k.shape)}, {tuple(v.shape)}")
        tensors.append({"k": k, "v": v})
    return tensors


def plot_layer_metric(layer_stats: List[Dict[str, Any]], component: str, output: Path) -> None:
    layers = np.arange(32)
    means = np.array([row[component]["cosine"]["mean"] for row in layer_stats])
    p10 = np.array([row[component]["cosine"]["p10"] for row in layer_stats])
    p90 = np.array([row[component]["cosine"]["p90"] for row in layer_stats])
    plt.figure(figsize=(9, 5))
    plt.plot(layers, means, marker="o", markersize=3, label=f"Visual {component.upper()}")
    plt.fill_between(layers, p10, p90, alpha=.22, label="adjacent-pair p10–p90")
    plt.xlabel("Transformer layer")
    plt.ylabel("Cosine similarity (flattened visual block)")
    plt.title(f"Visual {component.upper()} adjacent-step temporal stability")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(output, dpi=250); plt.close()


def plot_kv(layer_stats: List[Dict[str, Any]], output: Path) -> None:
    layers = np.arange(32)
    plt.figure(figsize=(9, 5))
    for component in ("k", "v"):
        mean = np.array([row[component]["cosine"]["mean"] for row in layer_stats])
        plt.plot(layers, mean, marker="o", markersize=3, label=f"Visual {component.upper()}")
    plt.xlabel("Transformer layer"); plt.ylabel("Mean cosine similarity (adjacent pairs)")
    plt.title("Visual K vs V adjacent-step temporal stability")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(output, dpi=250); plt.close()


def plot_heatmap(matrix: np.ndarray, component: str, output: Path) -> None:
    plt.figure(figsize=(13, 7))
    im = plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.xlabel("Visual token position (cache positions 1–256; no semantic identity assumed)")
    plt.ylabel("Transformer layer")
    plt.title(f"Visual {component.upper()} mean adjacent-step cosine by layer and token")
    plt.colorbar(im, label="Mean cosine similarity across adjacent control-step pairs")
    plt.tight_layout(); plt.savefig(output, dpi=250); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=220)
    parser.add_argument("--settling-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.horizon < 1 or args.settling_steps < 0:
        raise ValueError("horizon must be positive and settling-steps nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"; figures.mkdir(exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    model, processor, device = load_model_and_processor(DEFAULT_MODEL_ID)
    unnorm_key = resolve_unnorm_key(model, None)
    suite = benchmark.get_benchmark_dict()[TASK_SUITE_NAME](); task = suite.get_task(0)
    initial_states = suite.get_task_init_states(0); env, description = create_libero_env(task)
    prompt = f"In: What action should the robot take to {description.lower()}?\nOut:"

    previous = None; raw_pairs: List[Dict[str, Any]] = []
    # [transition, layer, visual-position], kept as float32 in an NPZ sidecar.
    token_metrics: Dict[str, List[np.ndarray]] = {name: [] for name in ("k_cosine", "k_relative_l2", "v_cosine", "v_relative_l2")}
    actions: List[List[float]] = []; layouts: List[Dict[str, Any]] = []
    done = False; termination_reason = "horizon_reached"
    try:
        env.reset(); observation = env.set_init_state(initial_states[0])
        for _ in range(args.settling_steps):
            observation, _reward, done, _info = env.step(DEFAULT_STEP_ACTION)
            if done: raise RuntimeError("Environment terminated during settling; no policy trajectory exists.")
        for step in range(args.horizon):
            image = policy_image(observation, args.center_crop)
            prediction = capture_fresh_prefill(model, processor, image, prompt, device, unnorm_key)
            layout = prediction["layout"]
            current = visual_tensors(prediction["cache"], layout)
            if len(current) != 32: raise AssertionError("Did not capture all 32 layers.")
            layouts.append({k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in layout.items()})
            if previous is not None:
                if layout != prior_layout:
                    raise AssertionError(f"Visual layout changed at transition {step - 1}->{step}: {prior_layout} != {layout}")
                row: Dict[str, Any] = {"from_control_step": step - 1, "to_control_step": step, "layers": []}
                per = {name: np.empty((32, int(layout["visual_token_count"])), dtype=np.float32) for name in token_metrics}
                for layer in LAYERS:
                    layer_row: Dict[str, Any] = {"layer": layer}
                    for component in ("k", "v"):
                        a, b = previous[layer][component], current[layer][component]
                        if a.shape != b.shape: raise AssertionError(f"Layer {layer}: adjacent visual shape mismatch.")
                        layer_row[component] = {"cosine": cosine_similarity(a, b), "relative_l2": relative_l2(a, b)}
                        # Flatten KV-head and head-dim per visual position, preserving token identity.
                        av, bv = a[0].reshape(a.shape[1], -1), b[0].reshape(b.shape[1], -1)
                        per[f"{component}_cosine"][layer] = F.cosine_similarity(av, bv, dim=1).numpy()
                        per[f"{component}_relative_l2"][layer] = (torch.norm(bv - av, dim=1) / torch.norm(av, dim=1)).numpy()
                    row["layers"].append(layer_row)
                raw_pairs.append(row)
                for name in token_metrics: token_metrics[name].append(per[name])
            previous, prior_layout = current, dict(layout)
            actions.append(prediction["policy_action"].tolist())
            environment_action = normalize_and_invert_openvla_gripper(prediction["policy_action"])
            observation, _reward, done, _info = env.step(environment_action.tolist())
            print(f"control step {step}: action={prediction['policy_action'].tolist()}")
            if done:
                termination_reason = "environment_done_success"; break
    finally:
        env.close()

    if not raw_pairs: raise RuntimeError("Need at least two policy control steps for temporal analysis.")
    arrays = {name: np.stack(values, axis=0) for name, values in token_metrics.items()}
    np.savez_compressed(args.output_dir / "per_visual_token_adjacent_metrics.npz", **arrays)
    layer_stats: List[Dict[str, Any]] = []
    for layer in LAYERS:
        out: Dict[str, Any] = {"layer": layer}
        for component in ("k", "v"):
            block_cos = np.array([pair["layers"][layer][component]["cosine"] for pair in raw_pairs])
            block_l2 = np.array([pair["layers"][layer][component]["relative_l2"] for pair in raw_pairs])
            token_cos = arrays[f"{component}_cosine"][:, layer, :]
            token_l2 = arrays[f"{component}_relative_l2"][:, layer, :]
            out[component] = {
                "cosine": stats(block_cos), "relative_l2": stats(block_l2),
                "per_visual_position_cosine_distribution": stats(token_cos.mean(axis=0)),
                "per_visual_position_relative_l2_distribution": stats(token_l2.mean(axis=0)),
            }
        layer_stats.append(out)
    token_aggregate = {name: np.mean(values, axis=0).tolist() for name, values in arrays.items()}
    metadata = {
        "experiment": "stage2d_0_visual_token_temporal_stability", "model_id": DEFAULT_MODEL_ID,
        "task_suite": TASK_SUITE_NAME, "task_id": 0, "task_description": description,
        "initial_state_id": 0, "seed": args.seed, "settling_steps_excluded": args.settling_steps,
        "policy_control_steps": len(actions), "adjacent_control_step_pairs": len(raw_pairs),
        "success": bool(done), "termination_reason": termination_reason,
        "layout": layouts[0], "layout_identical_at_all_policy_steps": all(x == layouts[0] for x in layouts),
        "cache_k_is_post_rope": True,
        "metric_definition": {"cosine": "F.cosine_similarity(a.float().flatten(), b.float().flatten())", "relative_l2": "norm(b.float()-a.float()) / norm(a.float())"},
        "sanity_checks": {"layers_captured": 32, "bos_excluded": True, "language_excluded": True,
                          "generated_action_tokens_excluded": True, "settling_observations_excluded": True,
                          "comparisons_are_adjacent_same_layer": True, "matching_shapes_verified": True,
                          "fresh_policy_unmodified": True, "k_v_not_swapped": True},
        "layer_statistics": layer_stats, "per_visual_position_mean_metrics": token_aggregate,
        "language_comparison": {"created": False, "reason": "Existing language K results were captured at raw k_proj before RoPE, whereas this analysis uses post-RoPE cache K. A K comparison would not be representation-equivalent; no combined figure was manufactured."},
    }
    with (args.output_dir / "visual_kv_stability_results.json").open("w") as f: json.dump(metadata, f, indent=2)
    with (args.output_dir / "adjacent_pair_metrics.json").open("w") as f: json.dump(raw_pairs, f)
    with (args.output_dir / "policy_actions.json").open("w") as f: json.dump(actions, f)
    plot_layer_metric(layer_stats, "k", figures / "visual_k_stability_by_layer.png")
    plot_layer_metric(layer_stats, "v", figures / "visual_v_stability_by_layer.png")
    plot_kv(layer_stats, figures / "visual_k_vs_v_stability_by_layer.png")
    plot_heatmap(np.mean(arrays["k_cosine"], axis=0), "k", figures / "visual_k_layer_token_heatmap.png")
    plot_heatmap(np.mean(arrays["v_cosine"], axis=0), "v", figures / "visual_v_layer_token_heatmap.png")
    print(json.dumps({"output_dir": str(args.output_dir), "steps": len(actions), "pairs": len(raw_pairs), "success": bool(done), "layout": layouts[0]}, indent=2))


if __name__ == "__main__":
    main()
