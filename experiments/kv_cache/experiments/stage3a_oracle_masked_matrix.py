"""Stage 3A: paired fresh vs oracle foreground-aware visual-K true skip, R=5."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from libero.libero import benchmark
from experiments.kv_analysis.experiments.extract_libero_image import DEFAULT_STEP_ACTION, TASK_SUITE_NAME, normalize_and_invert_openvla_gripper
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import capture_prediction, policy_image_from_observation, seed_everything, assert_independent_cache_clone
from experiments.kv_cache.experiments.oracle_foreground_mask import rendered_label_mapping
from experiments.kv_cache.experiments.stage3a_masked_visual_k_validation import LAYERS, MaskedSkip, mask_from_obs, seg_env
from experiments.kv_cache.experiments.static_reuse_experiment import DEFAULT_MODEL_ID, clone_prefill_cache, load_model_and_processor, resolve_unnorm_key

OUT = Path(__file__).resolve().parents[1] / "outputs" / "stage3a_oracle_masked_matrix"
TASKS, STATES, R, HORIZON = (4, 5, 7), (0, 1, 2), 5, 220

def rollout(condition, task_id, state_id, model, processor, device, key):
    seed_everything(7)
    suite = benchmark.get_benchmark_dict()[TASK_SUITE_NAME]()
    task, states = suite.get_task(task_id), suite.get_task_init_states(task_id)
    env = seg_env(task)
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    rows, source, done = [], None, False
    try:
        env.reset(); obs = env.set_init_state(states[state_id])
        mapping = rendered_label_mapping(env.env.model.instances_to_ids)
        for _ in range(10): obs, _, _, _ = env.step(DEFAULT_STEP_ACTION)
        for step in range(HORIZON):
            fg = mask_from_obs(obs, mapping)
            reuse = condition == "masked_true_skip" and step % R != 0
            image = policy_image_from_observation(obs, True)
            if reuse:
                expected = step - (step % R)
                assert source is not None and source["step"] == expected and source["fresh"]
                ctx = MaskedSkip(model, source["cache"], source["layout"], fg)
                pred = capture_prediction(model, processor, image, prompt, device, key, ctx)
                ctx.validate(pred["cache"])
                expected_k = 1 + int(fg.sum()) + (pred["layout"]["language"][1] - pred["layout"]["language"][0] + 1)
                assert len(ctx.d["q_rows"]) == len(LAYERS) and len(ctx.d["v_rows"]) == len(LAYERS)
                assert all(x["rows"] == pred["prefill_length"] for x in ctx.d["q_rows"] + ctx.d["v_rows"])
                assert all(x["rows"] == expected_k for x in ctx.d["k_rows"])
                assert len(ctx.d["decode_k_rows"]) == len(LAYERS) * 6
                source_step, age = source["step"], step - source["step"]
                validation = {"all_passed": True, "selected_layers": len(LAYERS), "q_rows_full": True, "v_rows_full": True,
                              "k_rows_per_selected_layer": expected_k, "reused_background_rows_per_selected_layer": int((~fg).sum()),
                              "source_is_fresh_refresh": True, "decode_untouched": True}
            else:
                pred = capture_prediction(model, processor, image, prompt, device, key)
                # Only fresh prefills become sources; fresh baseline does not need one.
                if condition == "masked_true_skip":
                    fresh_clone = clone_prefill_cache(pred["cache"])
                    assert_independent_cache_clone(fresh_clone, pred["cache"])
                    source = {"cache": fresh_clone, "layout": dict(pred["layout"]), "step": step, "fresh": True}
                source_step, age = None, 0
                validation = {"all_passed": True, "refresh_full_visual_k": True}
            assert pred["layout"]["visual"] == [1, 256] and pred["layout"]["visual_token_count"] == 256
            bg = ~fg
            act = normalize_and_invert_openvla_gripper(pred["policy_action"])
            obs, _, done, _ = env.step(act.tolist())
            rows.append({"step": step, "type": "reuse" if reuse else "refresh", "source_step": source_step,
                         "source_age": age, "prefill_length": pred["prefill_length"], "language_tokens": pred["layout"]["language"][1] - pred["layout"]["language"][0] + 1,
                         "foreground_tokens": int(fg.sum()), "background_tokens": int(bg.sum()), "projected_k_rows_per_selected_layer": validation.get("k_rows_per_selected_layer", pred["prefill_length"]),
                         "reused_k_rows_per_selected_layer": validation.get("reused_background_rows_per_selected_layer", 0),
                         "policy_action": pred["policy_action"].tolist(), "action_token_ids": pred["action_token_ids"], "done": bool(done), "validation": validation})
            if done: break
    finally:
        env.close()
    return {"condition": condition, "task_id": task_id, "task_description": task.language, "state_id": state_id, "seed": 7,
            "success": bool(done), "steps": len(rows), "termination": "success" if done else "horizon",
            "refresh_count": sum(r["type"] == "refresh" for r in rows), "reuse_count": sum(r["type"] == "reuse" for r in rows),
            "reuse_fraction": sum(r["type"] == "reuse" for r in rows) / len(rows), "max_source_age": max(r["source_age"] for r in rows),
            "records": rows, "all_valid": all(r["validation"]["all_passed"] for r in rows)}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, processor, device = load_model_and_processor(DEFAULT_MODEL_ID); key = resolve_unnorm_key(model, None)
    summary = []
    for task in TASKS:
        for state in STATES:
            sub = OUT / f"task{task}_state{state}"; sub.mkdir(parents=True, exist_ok=True)
            result = {"config": {"R": R, "layers": list(LAYERS), "visual_positions": [1,256], "mechanism": "oracle_foreground_masked_true_skip"},
                      "fresh": rollout("fresh", task, state, model, processor, device, key),
                      "masked_true_skip": rollout("masked_true_skip", task, state, model, processor, device, key)}
            (sub / "results.json").write_text(json.dumps(result, indent=2))
            summary.append({"task": task, "state": state, "fresh": {k: result["fresh"][k] for k in ("success","steps","termination")},
                            "masked_true_skip": {k: result["masked_true_skip"][k] for k in ("success","steps","termination","refresh_count","reuse_count","reuse_fraction","max_source_age","all_valid")}})
            (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary[-1]), flush=True)
if __name__ == "__main__": main()
