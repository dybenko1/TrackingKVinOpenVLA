# Track B V2 Transfer Validation — Three-Experiment Summary

## Status and decision

The three modules completed sequentially. No retraining, action generation, policy rollout, or KV-cache change occurred. Decision: **CONDITIONAL**.

## Experiment 1 — Prompt robustness on known COCO categories

The frozen V2 epoch-40 model selected **A_training_prompt** at threshold **0.20** from the 1,000-image development split. Development pooled category macro IoU was **0.4928**, positive-pair macro IoU **0.3422**, recall **0.7237**, and small-object IoU **0.1441**.
V1's faithful seven-template result on development had pooled macro IoU **0.3987** at its fixed 0.55 threshold. The V2 selected condition on the 3,000-image secondary set had pooled macro IoU **0.4926** and small-object IoU **0.1325**; V1 seven-template was **0.3972**. This 3,000-image set had been inspected previously and is not an untouched test.
The wording results concern categories included in V2 training. They do not show zero-shot understanding of attributes like `black bowl` or unseen concepts such as `plate`, `drawer`, or `robot gripper`.

## Experiment 2 — LIBERO evidence availability

Status: **partial_initial_state_assets_only_no_recorded_sequences**. The source tree contains 130 BDDL tasks, 130 initial-state assets, and 27 sparse saved RGB frames across 9 task/state pairs, but no LIBERO demonstration HDF5 files or paired recorded simulator states. The 3 HDF5 files found belong to unrelated robosuite examples. Therefore per-role IoU, missed task-relevant foreground, temporal turnover, and age-1–4 mask safety were not measured. RGB alone was not treated as segmentation ground truth.

## Experiment 3 — Native integration and overhead

The passive hook captured DINO features of shape `[1, 256, 1024]` from the normal fused vision path (fused output `[1, 256, 2176]`); direct and captured feature max error was 0.0, and prefill logits matched exactly (max error 0.0). The hook reused the existing DINO pass.
Median visual forward was 60.245 ms, with passive hook 60.158 ms (30 synchronized wall-clock samples; difference is within measurement noise). Cached patch scoring + sigmoid + threshold was 0.415 ms/frame. Four prompt tokenization, encoding, and projection cost 186.212 ms once per unchanged task instruction, or approximately 9.726 ms/step amortized over 20 steps including cached scoring.
The SigLIP text tower initialized in 10.06 s and its loaded weights occupied 1.80 GB; peak CUDA allocation for the profile was 17.93 GB. No actual K/V savings timing was measured, so no net speedup is claimed.

## Conclusion

The model is operationally compatible with passive reuse of OpenVLA's existing patch features, and prompt wording robustness is measurable on known COCO classes. The required LIBERO task/object role and consecutive-frame evidence is absent, so V2 is not yet supported as a cache-safety selector. The result is **CONDITIONAL**, not GO.

See `experiment1_prompt_ablation/PROMPT_ABLATION_REPORT.md`, `experiment2_libero_ground_truth/LIBERO_GROUND_TRUTH_REPORT.md`, `experiment3_native_integration/INTEGRATION_PROFILE_REPORT.md`, and `summary.json` for full evidence.
