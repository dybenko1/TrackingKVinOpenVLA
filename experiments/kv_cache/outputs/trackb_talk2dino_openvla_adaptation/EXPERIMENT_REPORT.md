# Talk2DINO OpenVLA DINO Adaptation — Results

Status: **complete** · paired frames: 90 · tasks: 9

## Mean per-frame metrics on the frozen 60-frame evaluation split

| Condition | Map | Operating point | FG recall | BG precision | BG coverage | BG IoU |
|---|---|---|---:|---:|---:|---:|
| A_original_meta_final24 | RAW | fixed_0.54 | 0.997 | 0.999 | 0.453 | 0.533 |
| A_original_meta_final24 | RAW | dev_recall98 | 0.967 | 0.993 | 0.535 | 0.621 |
| A_original_meta_final24 | CLEAN | fixed_0.54 | 0.973 | 0.994 | 0.520 | 0.606 |
| A_original_meta_final24 | CLEAN | dev_recall98 | 0.963 | 0.992 | 0.537 | 0.623 |
| B_openvla_final24 | RAW | fixed_0.54 | 0.778 | 0.947 | 0.662 | 0.711 |
| B_openvla_final24 | RAW | dev_recall98 | 0.939 | 0.974 | 0.453 | 0.518 |
| B_openvla_final24 | CLEAN | fixed_0.54 | 0.628 | 0.929 | 0.795 | 0.813 |
| B_openvla_final24 | CLEAN | dev_recall98 | 0.931 | 0.975 | 0.487 | 0.556 |
| C_openvla_native23 | RAW | fixed_0.54 | 0.055 | 0.853 | 0.990 | 0.852 |
| C_openvla_native23 | RAW | dev_recall98 | 0.912 | 0.919 | 0.238 | 0.261 |
| C_openvla_native23 | CLEAN | fixed_0.54 | 0.011 | 0.848 | 0.998 | 0.848 |
| C_openvla_native23 | CLEAN | dev_recall98 | 0.908 | 0.940 | 0.295 | 0.327 |
| E2_openvla_final24_mlp_adapted | RAW | fixed_0.54 | 0.203 | 0.873 | 0.965 | 0.868 |
| E2_openvla_final24_mlp_adapted | RAW | dev_recall98 | 0.980 | 0.995 | 0.632 | 0.739 |
| E2_openvla_final24_mlp_adapted | CLEAN | fixed_0.54 | 0.014 | 0.848 | 0.998 | 0.848 |
| E2_openvla_final24_mlp_adapted | CLEAN | dev_recall98 | 0.982 | 0.996 | 0.628 | 0.735 |
| E3_openvla_final24_joint_adapted | RAW | fixed_0.54 | 0.025 | 0.850 | 0.996 | 0.849 |
| E3_openvla_final24_joint_adapted | RAW | dev_recall98 | 0.954 | 0.989 | 0.633 | 0.734 |
| E3_openvla_final24_joint_adapted | CLEAN | fixed_0.54 | 0.003 | 0.847 | 1.000 | 0.847 |
| E3_openvla_final24_joint_adapted | CLEAN | dev_recall98 | 0.955 | 0.989 | 0.632 | 0.733 |

## Task-role foreground recall on the 60-frame evaluation split

Role masks are simulator-derived. The table uses RAW union predictions at each model’s development-selected 98%-recall operating point; these are role protection diagnostics, not category instance IoU.

| Adapted model | Role | Mean recall | Positive role-frames |
|---|---|---:|---:|
| E2_openvla_final24_mlp_adapted | destination_objects | 0.773 | 60 |
| E2_openvla_final24_mlp_adapted | manipulated_objects | 0.982 | 60 |
| E2_openvla_final24_mlp_adapted | other_objects | 0.452 | 60 |
| E2_openvla_final24_mlp_adapted | robot_arm | 0.994 | 60 |
| E2_openvla_final24_mlp_adapted | robot_gripper | 0.997 | 60 |
| E2_openvla_final24_mlp_adapted | robot_mount | 0.781 | 50 |
| E2_openvla_final24_mlp_adapted | static_scene_background | 0.472 | 50 |
| E2_openvla_final24_mlp_adapted | task_relevant_fixtures | 0.307 | 30 |
| E2_openvla_final24_mlp_adapted | task_relevant_objects | 0.000 | 30 |
| E3_openvla_final24_joint_adapted | destination_objects | 0.776 | 60 |
| E3_openvla_final24_joint_adapted | manipulated_objects | 0.982 | 60 |
| E3_openvla_final24_joint_adapted | other_objects | 0.495 | 60 |
| E3_openvla_final24_joint_adapted | robot_arm | 0.993 | 60 |
| E3_openvla_final24_joint_adapted | robot_gripper | 0.997 | 60 |
| E3_openvla_final24_joint_adapted | robot_mount | 0.752 | 50 |
| E3_openvla_final24_joint_adapted | static_scene_background | 0.499 | 50 |
| E3_openvla_final24_joint_adapted | task_relevant_fixtures | 0.187 | 30 |
| E3_openvla_final24_joint_adapted | task_relevant_objects | 0.000 | 30 |

## E1 fine-grained calibration on the previously inspected 90-frame benchmark

The primary operating points use the predeclared 0.001 grid. Exact observed-score thresholds are a secondary diagnostic; all thresholds were selected on the 30 development frames, then scored on the 60-frame evaluation split. Evaluation misses of the development recall targets are reported rather than retuned.

| Condition | Map | Grid τ for dev 98% | Exact τ for dev 98% | Eval recall at exact τ | Eval BG coverage |
|---|---|---:|---:|---:|---:|
| A_original_meta_final24 | RAW | 0.551 | 0.551758 | 0.970 | 0.538 |
| A_original_meta_final24 | CLEAN | 0.542 | 0.542969 | 0.967 | 0.541 |
| B_openvla_final24 | RAW | 0.529 | 0.529785 | 0.931 | 0.462 |
| B_openvla_final24 | CLEAN | 0.524 | 0.524902 | 0.923 | 0.499 |
| C_openvla_native23 | RAW | 0.513 | 0.513184 | 0.915 | 0.238 |
| C_openvla_native23 | CLEAN | 0.510 | 0.510254 | 0.911 | 0.295 |

This confirms threshold choice affects the coverage/recall operating point, but fine calibration does not make the OpenVLA block-24 score distribution match the Meta-DINO baseline. For example, the fixed 0.54 OpenVLA block-24 baseline was high-coverage but low-recall; the development-selected high-recall point trades away background coverage. This is not a closed-loop cache-safety result.


## Incremental inference cost

Latency is a synchronized single-observation component benchmark using cached features. It measures MLP/raw scoring, RAW+CLEAN scoring, and the E3 sidecar block separately. It is not total OpenVLA policy inference latency.

| Component | Median wall ms | P95 wall ms | Median CUDA ms |
|---|---:|---:|---:|
| E2_raw_score_only_on_cached_features | 0.761 | 0.837 | 0.710 |
| E2_raw_plus_CLEAN_on_cached_features | 1.729 | 1.774 | 1.677 |
| E3_FP32_block24_plus_attention | 5.516 | 12.938 | 5.370 |
| E3_block24_attention_RAW_and_CLEAN_end_to_end | 4.630 | 5.840 | 4.553 |
- E2 CLEAN postprocessing increment (median wall): 0.968675478361547 ms.
- E3 CLEAN postprocessing increment (median wall): 0.9967550868168473 ms.
- Peak allocated/reserved CUDA memory during timing: 15456583168 / 15533604864 bytes.

## Training configuration and integrity

- E2 best epoch: 9 · checkpoint SHA-256: `236f6ab9a3456e544fee8a37e20f0593be914f4d7527033f907656cf38819bb9`.
- E3 best epoch: 1 · checkpoint SHA-256: `990bd4573b4568b952760dff6fd41006ac40f9a4ad2429071d94859ea5cf01b2`.
- E2 trains only the original Talk2DINO MLP; the OpenVLA DINO and CLIP text tower are frozen.
- E3 trains a FP32 sidecar copy of OpenVLA block 24 (initialized by exact BF16→FP32 cast) and the E2-selected MLP. OpenVLA policy DINO remains frozen; its block-23 outputs, final block, patch/position parameters, CLIP, projector and LLM are untouched.
- Loss: 0.70 asymmetric BCE-with-logits on cosine scores (positive weight 4, negative weight 1) plus 0.30 soft Dice; fractional task-role occupancy is the target; empty category observations are excluded.
- E2/E3 thresholds are calibrated on complete held-out demonstrations from the six development tasks, then frozen for the 60-frame paired evaluation split.
- A supplementary complete-demonstration holdout, never used for optimization or selection, contains 336 frames over 6 development-task identities. Its paired summary is in `new_heldout_summary.json`; it is demo-disjoint but not task-disjoint.
- Raw is sigmoid cosine; CLEAN applies the original attention-head weighting and lambda_bg=0.2 implementation.

## Preserved policy behavior

The final DINO block is sidecar-only: OpenVLA continues consuming its original block-23 representations. `policy_parity_validation.json` records action and fused-visual output comparisons on three saved observations; no actions were applied.

## Dataset and leakage control

- Training source: f98770d54528af01e9c1f4980606024ad5d94aefbba8f640722cbaa1db4218a4.
- Data consists of simulator-rendered RGB and instance masks from recorded LIBERO MuJoCo states. HDF5 actions were not applied.
- All demonstrations used by the previous 90-image benchmark were excluded. Only the six predeclared development tasks supplied train/development/new-demo holdout examples; complete demo IDs were split before rendering.
- The six predeclared held-out tasks and existing 60-frame evaluation set were excluded from optimization and checkpoint/threshold selection.

## Limitations

- The 90-frame paired benchmark has been inspected previously and remains a small regression set, not an untouched broad generalization benchmark.
- The new demonstration split reuses six task identities for training/development; the six prior held-out tasks provide the task-disjoint check only through the 60-frame benchmark.
- No closed-loop policy evaluation or KV-cache safety claim is made.

## Artifacts

- `E1_calibration.json`, `E1_threshold_sweep.csv`, `figures/E1_threshold_curves.png`, and `figures/E1_score_distributions.png` contain E1.
- `training_history_e2.json`/`.csv`, `training_history_e3.json`/`.csv`, and `training_configuration_*.json` contain optimization and development metrics.
- `per_frame_metrics.csv`, `role_metrics.csv`, `new_heldout_metrics.csv`, `new_heldout_role_metrics.csv`, `policy_parity_validation.json`, and `latency_metrics.json` contain paired evaluation, parity, and timing evidence.
- Model checkpoints and feature tensors are local ignored cache files and are not published.

## Execution and reproducibility notes

- The completed run required two implementation repairs before training: the training-config output path used an invalid `Path + str` operation; the batched attention capture retained only the first sample. Attention was reconstructed for every cached observation from its stored block-23 tokens through the frozen block 24. The adaptation-specific code now validates `[N,16,256]` attention alignment.
- E2 early-stopped after 13 epochs (best development checkpoint epoch 9); E3 early-stopped after 4 epochs (best development checkpoint epoch 1). The selected checkpoint epochs are development-selected, not final-evaluation-selected.
- The existing 60-frame evaluation split was examined in earlier experiments. It is a paired historical comparison, not an untouched test set. The additional holdout is demonstration-disjoint but not task-disjoint.
- Runtime retained Transformers 4.49.0 / tokenizers 0.21.0; OpenVLA remote code warns that it was authored for 4.40.1 / 0.19.1. No dependency or validated container changes were made.
