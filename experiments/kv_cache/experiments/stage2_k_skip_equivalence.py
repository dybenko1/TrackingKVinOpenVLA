D"""Stage-1 K-overwrite versus Stage-2 K-skip equivalence harness."""
import json
import numpy as np
import torch
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import build_task_and_env_for, policy_image_from_observation, capture_prediction, seed_everything, REUSE_LAYERS
from experiments.kv_cache.experiments.prefill_reuse_experiment import DuringPrefillLanguageReuse, DuringPrefillLanguageKProjectionSkip
from experiments.kv_cache.experiments.static_reuse_experiment import load_model_and_processor, resolve_unnorm_key, clone_prefill_cache, cache_layers
from experiments.kv_analysis.experiments.extract_libero_image import normalize_and_invert_openvla_gripper

def metrics(a, b):
    d = a.float() - b.float()
    return {"max_abs": float(d.abs().max()), "mean_abs": float(d.abs().mean()), "relative_l2": float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(a.float()).clamp_min(1e-12))}

def captured_attention(model, call):
    values, handles = {}, []
    for i in REUSE_LAYERS:
        def hook(_m, _a, out, i=i):
            if out[0].shape[1] > 1: values[i] = out[0].detach().clone()
        handles.append(model.language_model.model.layers[i].self_attn.register_forward_hook(hook))
    result = call()
    for h in handles: h.remove()
    return result, values

def main():
    seed_everything(7)
    model, processor, device = load_model_and_processor("openvla/openvla-7b-finetuned-libero-spatial")
    key = resolve_unnorm_key(model, "libero_spatial")
    results = []
    for task_id in (0, 4, 5, 7):
        seed_everything(7)
        _, _, states, env, desc = build_task_and_env_for(task_id)
        env.reset(); observation = env.set_init_state(states[0])
        image = policy_image_from_observation(observation, True); env.close()
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        fresh = capture_prediction(model, processor, image, prompt, device, key)
        source = clone_prefill_cache(fresh["cache"])
        s1 = DuringPrefillLanguageReuse(source, fresh["layout"], REUSE_LAYERS, "k")
        one, one_attention = captured_attention(model, lambda: capture_prediction(model, processor, image, prompt, device, key, s1))
        s1.validate_completed_prefill(one["cache"])
        s2 = DuringPrefillLanguageKProjectionSkip(model, source, fresh["layout"], REUSE_LAYERS)
        two, two_attention = captured_attention(model, lambda: capture_prediction(model, processor, image, prompt, device, key, s2))
        s2.validate_completed_prefill(two["cache"])
        ks, vs, ats = [], [], []
        for i in REUSE_LAYERS:
            k1, v1 = cache_layers(one["cache"])[i]; k2, v2 = cache_layers(two["cache"])[i]
            ks.append(metrics(k1, k2)); vs.append(metrics(v1, v2)); ats.append(metrics(one_attention[i], two_attention[i]))
        start, end = fresh["layout"]["language"]
        results.append({"task_id": task_id, "description": desc, "prefill_length": fresh["layout"]["prefill_length"], "prefix_length": start, "language_length": end-start+1, "k_proj_rows": [x["rows"] for x in s2.diagnostics["k_proj_input_rows"]], "decode_k_proj_rows": [x["rows"] for x in s2.diagnostics["full_k_proj_calls"]], "token_ids_equal": one["action_token_ids"] == two["action_token_ids"], "policy_action_metrics": metrics(torch.from_numpy(one["policy_action"]), torch.from_numpy(two["policy_action"])), "environment_action_metrics": metrics(torch.from_numpy(normalize_and_invert_openvla_gripper(one["policy_action"])), torch.from_numpy(normalize_and_invert_openvla_gripper(two["policy_action"]))), "k_metrics": ks, "v_metrics": vs, "attention_metrics": ats, "logit_metrics": metrics(one["logits"], two["logits"])})
    print(json.dumps(results))

if __name__ == "__main__": main()
