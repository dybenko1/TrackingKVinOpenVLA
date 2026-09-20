"""Stage-1 closed-loop LIBERO deployment of fixed-schedule language-KV reuse.

This script executes three policies from the same LIBERO spatial/task-0/initial
state: all-fresh baseline, K-only reuse, and K+V reuse.  For the two reuse
conditions, control steps divisible by the fixed refresh interval are fresh
REFRESH steps; all other steps reuse the independent prefill-only cache
captured at the most recent scheduled refresh step.
The reuse action itself is sent to ``env.step()``.  This is behavioral
deployment, not an offline counterfactual and not compute-saving: fresh current
projections are still calculated before V2 overwrites selected post-RoPE K/V.

The V2 ``DuringPrefillLanguageReuse`` context is deliberately reused unchanged.
It modifies only language positions in requested layers during the full
multimodal prefill; one-token action decoding is excluded by its sequence-length
guard.  A refresh source is the ``PrefillSnapshot`` clone captured before
GenerationMixin appends action tokens, so it cannot alias the mutable generation
cache or a preceding reuse step.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from libero.libero import benchmark

from experiments.kv_analysis.experiments.extract_libero_image import (
    DEFAULT_MODEL_ID,
    DEFAULT_STEP_ACTION,
    INITIAL_STATE_ID,
    TASK_ID,
    TASK_SUITE_NAME,
    create_libero_env,
    extract_libero_image,
    normalize_and_invert_openvla_gripper,
    prepare_openvla_policy_image,
    save_image,
)
from experiments.kv_cache.experiments.prefill_reuse_experiment import (
    DuringPrefillLanguageReuse,
    runtime_attention_report,
)
from experiments.kv_cache.experiments.static_reuse_experiment import (
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


CLOSED_LOOP_OUTPUT_DIR = OUTPUT_DIR / "closed_loop_language_kv_reuse"
DEFAULT_REFRESH_INTERVAL = 2
REUSE_LAYERS = tuple(range(14))


def build_task_and_env_for(task_id: int) -> Tuple[Any, Any, Any, Any, str]:
    """Create one deterministic LIBERO Spatial task environment by ID."""
    task_suite = benchmark.get_benchmark_dict()[TASK_SUITE_NAME]()
    if not 0 <= task_id < task_suite.n_tasks:
        raise ValueError(f"task_id {task_id} is outside [0, {task_suite.n_tasks - 1}].")
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = create_libero_env(task)
    return task_suite, task, initial_states, env, task_description


def seed_everything(seed: int) -> None:
    """Match the existing rollout's deterministic seeding before every condition."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def policy_image_from_observation(observation: Dict[str, Any], center_crop: bool) -> np.ndarray:
    """Exact image path used by extract_libero_image.py's official-style rollout."""
    import tensorflow as tf

    image = tf.image.encode_jpeg(extract_libero_image(observation))
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (224, 224), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
    return prepare_openvla_policy_image(image, center_crop=center_crop)


def cache_length(cache: Any) -> int:
    layers = cache_layers(cache)
    if not layers:
        raise AssertionError("Refresh cache contains no decoder layers.")
    length = int(layers[0][0].shape[-2])
    if any(key.shape[-2] != length or value.shape[-2] != length for key, value in layers):
        raise AssertionError("Refresh cache has inconsistent per-layer sequence lengths.")
    return length


def capture_prediction(
    model: Any,
    processor: Any,
    policy_image: np.ndarray,
    prompt: str,
    device: torch.device,
    unnorm_key: str,
    intervention: Optional[DuringPrefillLanguageReuse] = None,
) -> Dict[str, Any]:
    """Run greedy policy inference and capture an independent prefill-only cache.

    ``PrefillSnapshot`` clones the cache from the top-level multimodal forward
    before GenerationMixin extends its separate working cache with seven action
    tokens.  The manual decode below always receives another clone, protecting
    this snapshot/source from mutation.
    """
    inputs = prepare_inputs(processor, prompt, Image.fromarray(policy_image).convert("RGB"), device)
    context = intervention if intervention is not None else _NullContext()
    with context, PrefillSnapshot(model) as snapshot, torch.inference_mode():
        policy_action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    cache, logits, input_ids, layout = snapshot.require()
    prefill_length = cache_length(cache)
    if prefill_length != int(layout["prefill_length"]):
        raise AssertionError("Snapshot cache length disagrees with derived multimodal layout.")
    with torch.inference_mode():
        action_ids = greedy_decode_from_prefill(
            model, logits, clone_prefill_cache(cache), model.get_action_dim(unnorm_key)
        )
    decoded_action = decode_action_token_ids(model, action_ids, unnorm_key)
    if not np.array_equal(np.asarray(policy_action), decoded_action):
        raise AssertionError("Manual action-token decode does not reproduce predict_action().")
    # The manual decode above must not append to the captured refresh candidate.
    if cache_length(cache) != prefill_length:
        raise AssertionError("Prefill snapshot was mutated during manual action decoding.")
    return {
        "cache": cache,
        "logits": logits,
        "input_ids": input_ids,
        "layout": layout,
        "policy_action": np.asarray(policy_action),
        "action_token_ids": action_ids[0].detach().cpu().tolist(),
        "prefill_length": prefill_length,
    }


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        return None


def assert_refresh_source(source: Dict[str, Any], expected_step: int) -> None:
    """Validate source ownership and prefill-only immutability at every reuse step."""
    if source["refresh_step"] != expected_step:
        raise AssertionError(f"Reuse source is from step {source['refresh_step']}, expected refresh step {expected_step}.")
    if source["intervened"]:
        raise AssertionError("A reuse-step/intervened cache was incorrectly retained as a refresh source.")
    if cache_length(source["cache"]) != source["prefill_length"]:
        raise AssertionError("Stored refresh source was extended after capture.")


def assert_independent_cache_clone(source_cache: Any, prefill_snapshot_cache: Any) -> None:
    """Prove R_t has independent tensor storage, not just a distinct object."""
    source_layers = cache_layers(source_cache)
    snapshot_layers = cache_layers(prefill_snapshot_cache)
    if len(source_layers) != len(snapshot_layers):
        raise AssertionError("Refresh source clone has a different layer count.")
    for layer, ((source_k, source_v), (snapshot_k, snapshot_v)) in enumerate(zip(source_layers, snapshot_layers)):
        if source_k.data_ptr() == snapshot_k.data_ptr() or source_v.data_ptr() == snapshot_v.data_ptr():
            raise AssertionError(f"Refresh source layer {layer} aliases the working prefill snapshot.")


def run_condition(
    condition: str,
    component: Optional[str],
    model: Any,
    processor: Any,
    device: torch.device,
    unnorm_key: str,
    task_id: int,
    task_description: str,
    initial_state_id: int,
    horizon: int,
    settling_steps: int,
    seed: int,
    center_crop: bool,
    save_frames: bool,
    output_dir: Path,
    refresh_interval: int,
) -> Dict[str, Any]:
    """Run one real closed-loop rollout with fresh/reuse state-machine checks."""
    if condition not in {"baseline", "k", "kv"}:
        raise ValueError(f"Unsupported closed-loop condition {condition!r}.")
    if condition == "baseline" and component is not None:
        raise AssertionError("Baseline must not have an intervention component.")
    if condition != "baseline" and component not in {"k", "kv"}:
        raise AssertionError("Reuse conditions must be K-only or K+V.")
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be positive.")

    seed_everything(seed)
    task_suite, _task, initial_states, env, task_from_env = build_task_and_env_for(task_id)
    if task_from_env != task_description:
        raise AssertionError("Condition environment does not match the requested LIBERO task.")
    if not 0 <= initial_state_id < len(initial_states):
        raise ValueError(f"initial_state_id {initial_state_id} is unavailable for task {task_id}.")
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    condition_dir = output_dir / condition
    if save_frames:
        condition_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    refresh_source: Optional[Dict[str, Any]] = None
    done = False
    termination_reason = "horizon_reached"
    try:
        env.reset()
        observation = env.set_init_state(initial_states[initial_state_id])
        for settle_index in range(settling_steps):
            observation, _reward, done, _info = env.step(DEFAULT_STEP_ACTION)
            if done:
                termination_reason = f"terminated_during_settling_step_{settle_index}"
                break

        for step in range(horizon):
            if done:
                break
            step_offset_from_refresh = step % refresh_interval
            is_reuse = condition != "baseline" and step_offset_from_refresh != 0
            step_type = "reuse" if is_reuse else "refresh"
            if step == 0 and step_type != "refresh":
                raise AssertionError("t0 must always be a refresh step.")
            if condition != "baseline":
                expected = "refresh" if step_offset_from_refresh == 0 else "reuse"
                if step_type != expected:
                    raise AssertionError(f"Refresh schedule violation at step {step}: {step_type} != {expected}.")

            image = policy_image_from_observation(observation, center_crop)
            image_filename = None
            if save_frames:
                image_filename = f"step_{step:03d}.png"
                save_image(image, condition_dir / image_filename)

            if not is_reuse:
                prediction = capture_prediction(model, processor, image, prompt, device, unnorm_key)
                # This object is already a PrefillSnapshot clone, captured before
                # generation.  Make a second explicit clone to establish that
                # R_t cannot alias either working generation or later manual decode.
                refresh_source = {
                    "cache": clone_prefill_cache(prediction["cache"]),
                    "input_ids": prediction["input_ids"].detach().clone(),
                    "layout": dict(prediction["layout"]),
                    "prefill_length": prediction["prefill_length"],
                    "refresh_step": step,
                    "intervened": False,
                    "independent_clone_verified": True,
                }
                assert_independent_cache_clone(refresh_source["cache"], prediction["cache"])
                assert_refresh_source(refresh_source, step)
                intervention_diagnostics: Dict[str, Any] = {"inserted_layers": [], "layers": {}}
                source_refresh_step = None
            else:
                # The source must remain the actual independent cache captured
                # at the latest scheduled refresh, never a cache from a reuse
                # step.  For interval 3: t1/t2 both source t0, t4/t5 source t3.
                expected_refresh_step = step - step_offset_from_refresh
                if refresh_source is None:
                    raise AssertionError("Reuse step has no stored refresh source.")
                assert_refresh_source(refresh_source, expected_refresh_step)
                intervention = DuringPrefillLanguageReuse(
                    refresh_source["cache"], refresh_source["layout"], REUSE_LAYERS, component
                )
                prediction = capture_prediction(
                    model, processor, image, prompt, device, unnorm_key, intervention=intervention
                )
                validate_compatible_prefills(
                    refresh_source["cache"], prediction["cache"], refresh_source["input_ids"], prediction["input_ids"],
                    refresh_source["layout"], prediction["layout"], REUSE_LAYERS,
                )
                intervention.validate_completed_prefill(prediction["cache"])
                assert_refresh_source(refresh_source, expected_refresh_step)
                intervention_diagnostics = intervention.diagnostics
                source_refresh_step = expected_refresh_step

            environment_action = normalize_and_invert_openvla_gripper(prediction["policy_action"])
            observation, _reward, done, _info = env.step(environment_action.tolist())
            records.append(
                {
                    "control_step": step,
                    "step_type": step_type,
                    "condition": condition,
                    "source_refresh_step": source_refresh_step,
                    "source_age": step - source_refresh_step if is_reuse else 0,
                    "intervention_applied": is_reuse,
                    "intervention_layers": list(REUSE_LAYERS) if is_reuse else [],
                    "reuse_component": component if is_reuse else None,
                    "prefill_sequence_length": prediction["prefill_length"],
                    "action_token_ids": prediction["action_token_ids"],
                    "policy_action_before_gripper_conversion": prediction["policy_action"].tolist(),
                    "environment_action": environment_action.tolist(),
                    "frame": image_filename,
                    "done": bool(done),
                    "success": bool(done),
                    "reuse_validation": {
                        "source_is_most_recent_scheduled_refresh": (
                            (not is_reuse) or source_refresh_step == step - (step % refresh_interval)
                        ),
                        "source_age_is_valid": (not is_reuse) or 1 <= step - source_refresh_step < refresh_interval,
                        "source_prefill_only_length": cache_length(refresh_source["cache"]),
                        "refresh_source_independent_clone": refresh_source["independent_clone_verified"],
                        "inserted_layers": intervention_diagnostics["inserted_layers"],
                        "insertion_layer_diagnostics": intervention_diagnostics["layers"],
                    },
                }
            )
            if done:
                termination_reason = "environment_done_success"
                break
    finally:
        env.close()

    return {
        "task_id": task_id,
        "task_description": task_description,
        "initial_state_id": initial_state_id,
        "condition": condition,
        "reuse_component": component,
        "success": bool(done),
        "control_steps_executed": len(records),
        "termination_reason": termination_reason,
        "refresh_interval": refresh_interval,
        "reuse_layers": list(REUSE_LAYERS) if condition != "baseline" else [],
        "steps": records,
        "validation": {
            "t0_refresh": not records or records[0]["step_type"] == "refresh",
            "refresh_reuse_schedule": all(
                record["step_type"] == (
                    "refresh"
                    if condition == "baseline" or record["control_step"] % refresh_interval == 0
                    else "reuse"
                )
                for record in records
            ),
            "reuses_only_most_recent_scheduled_refresh": all(
                not record["intervention_applied"]
                or record["source_refresh_step"] == record["control_step"] - (record["control_step"] % refresh_interval)
                for record in records
            ),
            "source_age_within_refresh_interval": all(
                not record["intervention_applied"] or 1 <= record["source_age"] < refresh_interval
                for record in records
            ),
            "baseline_has_zero_intervention": all(not record["intervention_applied"] for record in records) if condition == "baseline" else None,
        },
        "refresh_step_count": sum(record["step_type"] == "refresh" for record in records),
        "reuse_step_count": sum(record["step_type"] == "reuse" for record in records),
        "maximum_source_age": max((record["source_age"] for record in records), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--refresh-interval", type=int, default=DEFAULT_REFRESH_INTERVAL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settling-steps", type=int, default=10)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, default=CLOSED_LOOP_OUTPUT_DIR)
    parser.add_argument("--conditions", nargs="+", choices=("baseline", "k", "kv"), default=("baseline", "k", "kv"))
    parser.add_argument("--task-ids", nargs="+", type=int, default=(TASK_ID,))
    parser.add_argument("--initial-state-ids", nargs="+", type=int, default=(INITIAL_STATE_ID,))
    args = parser.parse_args()
    if args.horizon < 1 or args.settling_steps < 0 or args.refresh_interval < 1:
        raise ValueError("--horizon and --refresh-interval must be positive and --settling-steps non-negative.")
    if tuple(REUSE_LAYERS) != tuple(parse_layer_spec("0-13")):
        raise AssertionError("Closed-loop experiment must use layers 0-13 exactly.")

    task_descriptions: Dict[int, str] = {}
    for task_id in args.task_ids:
        _suite, _task, initial_states, _env, task_description = build_task_and_env_for(task_id)
        _env.close()
        if any(state_id < 0 or state_id >= len(initial_states) for state_id in args.initial_state_ids):
            raise ValueError(f"An initial-state ID is unavailable for task {task_id}.")
        task_descriptions[task_id] = task_description
    model, processor, device = load_model_and_processor(args.model_id)
    unnorm_key = resolve_unnorm_key(model, TASK_SUITE_NAME)
    report = runtime_attention_report(model)
    print("Installed transformers version:", report["transformers_version"])
    print("Active attention:", report["attention_class"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for task_id in args.task_ids:
        for initial_state_id in args.initial_state_ids:
            rollout_dir = args.output_dir / f"task_{task_id:02d}_state_{initial_state_id:02d}"
            for condition in args.conditions:
                component = None if condition == "baseline" else condition
                print(f"Running task={task_id}, state={initial_state_id}, condition={condition}")
                summary = run_condition(
                    condition, component, model, processor, device, unnorm_key, task_id,
                    task_descriptions[task_id], initial_state_id, args.horizon, args.settling_steps,
                    args.seed, args.center_crop, args.save_frames, rollout_dir, args.refresh_interval,
                )
                results.append(summary)
                print(
                    f"task={task_id}, state={initial_state_id}, {condition}: success={summary['success']}, "
                    f"steps={summary['control_steps_executed']}, reason={summary['termination_reason']}"
                )

    output = args.output_dir / f"closed_loop_interval_{args.refresh_interval}.json"
    output.write_text(
        json.dumps(
            {
                "configuration": {
                    "checkpoint": args.model_id,
                    "task_suite": TASK_SUITE_NAME,
                    "task_ids": args.task_ids,
                    "initial_state_ids": args.initial_state_ids,
                    "seed": args.seed,
                    "horizon": args.horizon,
                    "settling_steps": args.settling_steps,
                    "center_crop": args.center_crop,
                    "refresh_interval": args.refresh_interval,
                    "reuse_layers": list(REUSE_LAYERS),
                    "stage": "behavioral_deployment_not_compute_saving",
                },
                "runtime_attention_report": report,
                "rollouts": results,
            },
            indent=2,
        )
        + "\n"
    )
    print("Saved closed-loop result:", output)


if __name__ == "__main__":
    main()
