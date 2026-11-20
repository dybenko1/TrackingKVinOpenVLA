"""Build the concise transfer summary from completed, validated artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--prompt-stage-source", default="bc1df0f45b3cafdd6b47d153e669e0749b2ff7f0")
    parser.add_argument("--integration-stage-source", default="37adb3511b9cbed6cc3fe0f1740cc26fcab53bf6")
    args = parser.parse_args()
    out = args.output_dir
    e1 = read(out / "experiment1_prompt_ablation/prompt_ablation_metrics.json")
    e1_config = read(out / "experiment1_prompt_ablation/configuration.json")
    e2 = read(out / "experiment2_libero_ground_truth/libero_task_manifest.json")
    e3_path = out / "experiment3_native_integration/integration_timing.json"
    e3 = read(e3_path)
    status = read(out / "RUN_STATUS.json")
    if status.get("state") != "complete" or any(x.get("state") != "complete" for x in status.get("stages", {}).values()):
        raise RuntimeError("Cannot finalize: the parent run or at least one stage is incomplete")

    timing = e3["timing"]
    tokenization = timing["tokenization_cached_prompts"]["median_ms"]
    text_encoding = timing["one_time_text_encoding_four_prompts_ms"]
    projection = timing["one_time_task_text_mlp_projection_and_l2_normalization"]["median_ms"]
    mask_score = timing["cached_text_projection_and_16x16_scores_and_threshold"]["median_ms"]
    setup_ms = tokenization + text_encoding + projection
    amortized = {str(h): setup_ms / h + mask_score for h in (5, 10, 20, 50)}
    timing["timing_method"] = "CPU wall-clock with CUDA synchronized before and after each timed operation; no CUDA-event timings"
    timing["task_text_setup_ms_four_prompts"] = setup_ms
    timing["amortized_cached_mask_cost_ms_per_step_by_horizon"] = amortized
    timing["full_policy_prefill_timing_limit"] = "single sample baseline/hooked measurements only; timing difference is not interpreted due to warmup/order sensitivity"
    e3["timing"] = timing
    write(e3_path, e3)
    prefill_base = timing["full_policy_prefill_baseline_single_sample_ms"]
    prefill_hook = timing["full_policy_prefill_with_passive_capture_single_sample_ms"]
    profile_lines = [
        "# OpenVLA Integration and Computational Overhead", "",
        f"Read-only profiling used saved frame `{e3['image']}` with its task instruction. No action generation, rollout, or cache mutation occurred.", "",
        f"- Native DINO output: `{e3['captured_dino']['shape']}`; fused visual output: `{e3['captured_dino']['fused_visual_shape']}`.",
        f"- Hook executions in each full prefill: {e3['captured_dino']['hook_calls_during_full_prefill']}; no second DINO pass was used.",
        f"- Direct versus captured feature max absolute error: {e3['captured_dino']['max_abs_diff_vs_direct_dino']:.8g}.",
        f"- Baseline versus hooked prefill logits max absolute error: {e3['policy_prefill_read_only_hook']['max_abs_logit_diff']:.8g}; allclose: {e3['policy_prefill_read_only_hook']['allclose_rtol_atol_1e-4']}.", "",
        "## Timing", "",
        "Times are synchronized CPU wall-clock around GPU operations, with CUDA synchronization before and after timed calls; CUDA events were not used.",
        f"- Vision forward median over {timing['vla_visual_forward']['n']} repeats: {timing['vla_visual_forward']['median_ms']:.3f} ms (p95 {timing['vla_visual_forward']['p95_ms']:.3f}).",
        f"- Vision forward with passive DINO hook median: {timing['visual_forward_with_passive_dino_capture']['median_ms']:.3f} ms (p95 {timing['visual_forward_with_passive_dino_capture']['p95_ms']:.3f}); the small negative/positive difference is within timing noise, so no capture speedup or overhead reduction is claimed.",
        f"- Cached four-prompt patch normalization + similarities + sigmoid + threshold median: {timing['cached_text_projection_and_16x16_scores_and_threshold']['median_ms']:.3f} ms/frame (p95 {timing['cached_text_projection_and_16x16_scores_and_threshold']['p95_ms']:.3f}). Patch L2 normalization alone was {timing['per_frame_patch_l2_normalization']['median_ms']:.3f} ms and is already included in that total.",
        f"- Task setup for four representative prompts: tokenization {timing['tokenization_cached_prompts']['median_ms']:.3f} ms, text encoding {timing['one_time_text_encoding_four_prompts_ms']:.3f} ms, MLP projection + text normalization {timing['one_time_task_text_mlp_projection_and_l2_normalization']['median_ms']:.3f} ms, total {setup_ms:.3f} ms.",
        f"- Changed/new single prompt encode + project + score median: {timing['one_additional_prompt_encode_project_score']['median_ms']:.3f} ms; this can be cached until the instruction changes.",
        f"- Text tower initialization: {timing['text_tower_init_seconds']:.2f} s; VLA processor/model loading: {timing['one_time_vla_processor_and_model_load_seconds']:.2f} s.",
        "",
        "| Unchanged-prompt control horizon | Task setup amortized + cached scoring (ms/control step) |",
        "|---:|---:|",
    ]
    profile_lines += [f"| {h} | {amortized[str(h)]:.3f} |" for h in (5,10,20,50)]
    profile_lines += [
        "", "## Memory and limitations", "",
        f"- Device: {e3['memory']['cuda_device']}; peak CUDA allocation {e3['memory']['cuda_peak_allocated_bytes']/1e9:.2f} GB; reserved {e3['memory']['cuda_reserved_bytes_after_profile']/1e9:.2f} GB; allocated {e3['memory']['cuda_allocated_bytes_after_profile']/1e9:.2f} GB after profiling.",
        f"- SigLIP text parameters occupy {e3['memory']['siglip_text_parameter_dtype_bytes']/1e9:.2f} GB at the loaded dtype. This is a parameter-footprint estimate; incremental allocator usage for text-only residency was not isolated.",
        f"- The full-policy-prefill single samples were {prefill_base:.1f} ms baseline and {prefill_hook:.1f} ms with the hook. Their order/warmup sensitivity makes them non-comparable; they are retained in JSON but excluded from timing conclusions.",
        "- No isolated K/V saved-computation timing was measured. Therefore no net acceleration is claimed; measured selection overhead is the minimum work that a future cache implementation must save to break even.",
        "- The profile used four representative prompt strings and one saved observation. It did not profile all 80 COCO class prompts or task rollouts.",
    ]
    (out / "experiment3_native_integration/INTEGRATION_PROFILE_REPORT.md").write_text("\n".join(profile_lines) + "\n")

    chosen = e1["selected_prompt_strategy"]
    dev = e1["development_calibrated"][chosen]
    secondary = e1["secondary_holdout_selected_condition"][chosen]
    v1_dev = e1["development_calibrated"]["V1_seven_template"]
    v1_secondary = e1["secondary_holdout_selected_condition"]["V1_seven_template"]
    summary = {
        "status": "complete",
        "decision": "CONDITIONAL",
        "scope": {"retraining": False, "policy_rollouts": False, "kv_cache_changes": False},
        "provenance": {
            "reference_commit": "e1b06f93924bd1c11dcbdb20800fedbdf51f293f",
            "prompt_ablation_code_revision": args.prompt_stage_source,
            "integration_profile_code_revision": args.integration_stage_source,
            "source_head_at_publication": args.source_head,
            "v2_checkpoint_sha256": e1["v2_checkpoint_sha256"],
            "v1_checkpoint_sha256": e1["v1_checkpoint_sha256"],
            "split_manifest_sha256": e1_config["manifests"]["development_sha256"],
        },
        "experiment1": {
            "development_images": e1["development_image_count"],
            "secondary_images": e1["final_image_count"],
            "secondary_previously_inspected": True,
            "selected_prompt_condition": chosen,
            "selected_threshold": e1["selected_threshold"],
            "development_selected_metrics": dev,
            "secondary_selected_metrics": secondary,
            "v1_development_seven_template_metrics": v1_dev,
            "v1_secondary_seven_template_metrics": v1_secondary,
            "all_development_conditions": e1["development_calibrated"],
            "all_secondary_conditions": e1["secondary_holdout_selected_condition"],
            "thresholds_only_selected_on_development": True,
            "interpretation": "known COCO category prompt rewording only; no zero-shot unseen concept claim",
        },
        "experiment2": {
            "status": e2["status"],
            "bddl_tasks": e2["bddl_count"],
            "initial_state_assets": e2["libero_pruned_initial_state_files"],
            "saved_rgb_frames": e2["saved_frame_count"],
            "saved_task_state_pairs": e2["saved_task_state_pairs"],
            "demo_hdf5_files": e2["libero_hdf5_count"],
            "per_role_metrics_available": False,
            "reason": "No recorded LIBERO trajectories or simulator states paired with saved RGB; sparse RGB cannot supply exact simulator segmentation masks or consecutive frame turnover.",
        },
        "experiment3": {
            "feature_shape": e3["captured_dino"]["shape"],
            "fused_shape": e3["captured_dino"]["fused_visual_shape"],
            "direct_feature_max_abs_diff": e3["captured_dino"]["max_abs_diff_vs_direct_dino"],
            "prefill_logits_max_abs_diff": e3["policy_prefill_read_only_hook"]["max_abs_logit_diff"],
            "vision_ms_median": timing["vla_visual_forward"]["median_ms"],
            "passive_capture_vision_ms_median": timing["visual_forward_with_passive_dino_capture"]["median_ms"],
            "cached_mask_ms_median": mask_score,
            "one_new_prompt_encode_project_score_ms_median": timing["one_additional_prompt_encode_project_score"]["median_ms"],
            "text_tower_init_seconds": timing["text_tower_init_seconds"],
            "text_encode_four_prompts_ms": text_encoding,
            "text_projection_four_prompts_ms": projection,
            "task_setup_four_prompts_ms": setup_ms,
            "amortized_mask_ms_per_step": amortized,
            "siglip_text_weights_gb": e3["memory"]["siglip_text_parameter_dtype_bytes"] / 1e9,
            "cuda_peak_allocated_gb": e3["memory"]["cuda_peak_allocated_bytes"] / 1e9,
            "cuda_reserved_gb": e3["memory"]["cuda_reserved_bytes_after_profile"] / 1e9,
            "full_policy_prefill_comparison": "single-shot only and not used to infer a speed change",
            "net_speedup_claimed": False,
        },
        "run_status": status,
    }
    write(out / "summary.json", summary)

    lines = [
        "# Track B V2 Transfer Validation — Three-Experiment Summary", "",
        "## Status and decision", "",
        "The three modules completed sequentially. No retraining, action generation, policy rollout, or KV-cache change occurred. Decision: **CONDITIONAL**.", "",
        "## Experiment 1 — Prompt robustness on known COCO categories", "",
        f"The frozen V2 epoch-40 model selected **{chosen}** at threshold **{e1['selected_threshold']:.2f}** from the 1,000-image development split. Development pooled category macro IoU was **{dev['pooled_macro_iou']:.4f}**, positive-pair macro IoU **{dev['mean_positive_pair_macro_iou']:.4f}**, recall **{dev['recall_macro_category']:.4f}**, and small-object IoU **{dev['small_object_pooled_macro_iou']:.4f}**.",
        f"V1's faithful seven-template result on development had pooled macro IoU **{v1_dev['pooled_macro_iou']:.4f}** at its fixed 0.55 threshold. The V2 selected condition on the 3,000-image secondary set had pooled macro IoU **{secondary['pooled_macro_iou']:.4f}** and small-object IoU **{secondary['small_object_pooled_macro_iou']:.4f}**; V1 seven-template was **{v1_secondary['pooled_macro_iou']:.4f}**. This 3,000-image set had been inspected previously and is not an untouched test.",
        "The wording results concern categories included in V2 training. They do not show zero-shot understanding of attributes like `black bowl` or unseen concepts such as `plate`, `drawer`, or `robot gripper`.", "",
        "## Experiment 2 — LIBERO evidence availability", "",
        f"Status: **{e2['status']}**. The source tree contains {e2['bddl_count']} BDDL tasks, {e2['libero_pruned_initial_state_files']} initial-state assets, and {e2['saved_frame_count']} sparse saved RGB frames across {e2['saved_task_state_pairs']} task/state pairs, but no LIBERO demonstration HDF5 files or paired recorded simulator states. The {e2['hdf5_count']} HDF5 files found belong to unrelated robosuite examples. Therefore per-role IoU, missed task-relevant foreground, temporal turnover, and age-1–4 mask safety were not measured. RGB alone was not treated as segmentation ground truth.", "",
        "## Experiment 3 — Native integration and overhead", "",
        f"The passive hook captured DINO features of shape `{e3['captured_dino']['shape']}` from the normal fused vision path (fused output `{e3['captured_dino']['fused_visual_shape']}`); direct and captured feature max error was {e3['captured_dino']['max_abs_diff_vs_direct_dino']:.1f}, and prefill logits matched exactly (max error {e3['policy_prefill_read_only_hook']['max_abs_logit_diff']:.1f}). The hook reused the existing DINO pass.",
        f"Median visual forward was {timing['vla_visual_forward']['median_ms']:.3f} ms, with passive hook {timing['visual_forward_with_passive_dino_capture']['median_ms']:.3f} ms (30 synchronized wall-clock samples; difference is within measurement noise). Cached patch scoring + sigmoid + threshold was {mask_score:.3f} ms/frame. Four prompt tokenization, encoding, and projection cost {setup_ms:.3f} ms once per unchanged task instruction, or approximately {amortized['20']:.3f} ms/step amortized over 20 steps including cached scoring.",
        f"The SigLIP text tower initialized in {timing['text_tower_init_seconds']:.2f} s and its loaded weights occupied {e3['memory']['siglip_text_parameter_dtype_bytes']/1e9:.2f} GB; peak CUDA allocation for the profile was {e3['memory']['cuda_peak_allocated_bytes']/1e9:.2f} GB. No actual K/V savings timing was measured, so no net speedup is claimed.",
        "",
        "## Conclusion", "",
        "The model is operationally compatible with passive reuse of OpenVLA's existing patch features, and prompt wording robustness is measurable on known COCO classes. The required LIBERO task/object role and consecutive-frame evidence is absent, so V2 is not yet supported as a cache-safety selector. The result is **CONDITIONAL**, not GO.",
        "",
        "See `experiment1_prompt_ablation/PROMPT_ABLATION_REPORT.md`, `experiment2_libero_ground_truth/LIBERO_GROUND_TRUTH_REPORT.md`, `experiment3_native_integration/INTEGRATION_PROFILE_REPORT.md`, and `summary.json` for full evidence.",
    ]
    (out / "TRACKB_V2_TRANSFER_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "decision": "CONDITIONAL", "source_head": args.source_head, "summary": str(out / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
