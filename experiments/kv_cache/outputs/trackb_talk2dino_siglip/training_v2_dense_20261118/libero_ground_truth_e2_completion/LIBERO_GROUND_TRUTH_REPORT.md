# LIBERO Ground-Truth Evaluation — Frozen Talk2DINO-SigLIP V2

## Outcome

This is an offline replay of official demonstration simulator states. No recorded actions were applied, no OpenVLA policy rollouts were run, no model was trained, and no KV-cache code was changed.

The development/held-out split is task-level, balanced within each suite and frozen in `libero_task_manifest.json` before prediction. The V2 threshold 0.20 is the frozen COCO development setting; the separate 98%-recall operating point is calibrated only on development tasks and then evaluated on held-out tasks.

## Main results

| Split | Model | Threshold | Task-FG recall | False-background rate | Mean candidate reused tokens / 256 | Mean background coverage |
|---|---|---:|---:|---:|---:|---:|
| development | V1_epoch50_seven_templates | 0.55 | 0.6319 | 0.3681 | 174.1 | 0.680 |
| development | V2_epoch40_COCO_prompt_0_20 | 0.20 | 0.5272 | 0.4728 | 185.2 | 0.723 |
| development | V2_epoch40_dev_98pct_recall | 0.08 | 0.9890 | 0.0110 | 37.7 | 0.147 |
| held_out | V1_epoch50_seven_templates | 0.55 | 0.6303 | 0.3697 | 176.8 | 0.690 |
| held_out | V2_epoch40_COCO_prompt_0_20 | 0.20 | 0.3958 | 0.6042 | 224.4 | 0.877 |
| held_out | V2_epoch40_dev_98pct_recall | 0.08 | 0.9813 | 0.0187 | 25.5 | 0.099 |

## Per-role foreground metrics

Counts are fractional 16×16 patch occupancy, evaluated only over patches mapped to known instances. Empty-mask frequency is among frames where that role is visible. `destination_or_fixture` is an additional combined target-role diagnostic; BDDL metadata is used only to associate that text role with object-vs-fixture ground truth after scoring.

| Split | Model | Role | Visible frames / small instance-frames | Recall | False-background rate | Pooled IoU | Empty-mask frequency |
|---|---|---|---:|---:|---:|---:|---:|
| development | V1_epoch50_seven_templates | task_relevant_fixtures | 96 | 0.2601 | 0.7399 | 0.1507 | 0.0104 |
| development | V1_epoch50_seven_templates | robot_arm | 576 | 0.5967 | 0.4033 | 0.2033 | 0.0000 |
| development | V1_epoch50_seven_templates | robot_gripper | 576 | 0.9301 | 0.0699 | 0.0889 | 0.0000 |
| development | V1_epoch50_seven_templates | robot_mount | 384 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| development | V1_epoch50_seven_templates | destination_or_fixture | 576 | 0.1373 | 0.8627 | 0.0999 | 0.7760 |
| development | V1_epoch50_seven_templates | task_relevant_foreground | 576 | 0.6319 | 0.3681 | 0.2880 | 0.0000 |
| development | V1_epoch50_seven_templates | manipulated_objects | 470 | 0.0000 | 1.0000 | 0.0000 | 0.9213 |
| development | V1_epoch50_seven_templates | destination_objects | 480 | 0.0094 | 0.9906 | 0.0093 | 0.9292 |
| development | V1_epoch50_seven_templates | small relevant instances (<4 patch-eq; union-mask diagnostic) | 694 | 0.6320565725427686 | — | 0.017424106082303163 | — |
| development | V2_epoch40_COCO_prompt_0_20 | task_relevant_fixtures | 96 | 0.2940 | 0.7060 | 0.0802 | 0.0000 |
| development | V2_epoch40_COCO_prompt_0_20 | robot_arm | 576 | 0.2367 | 0.7633 | 0.1800 | 0.0000 |
| development | V2_epoch40_COCO_prompt_0_20 | robot_gripper | 576 | 0.8649 | 0.1351 | 0.1679 | 0.0000 |
| development | V2_epoch40_COCO_prompt_0_20 | robot_mount | 384 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| development | V2_epoch40_COCO_prompt_0_20 | destination_or_fixture | 576 | 0.2094 | 0.7906 | 0.0479 | 0.6528 |
| development | V2_epoch40_COCO_prompt_0_20 | task_relevant_foreground | 576 | 0.5272 | 0.4728 | 0.2568 | 0.0000 |
| development | V2_epoch40_COCO_prompt_0_20 | manipulated_objects | 470 | 0.0932 | 0.9068 | 0.0696 | 0.6532 |
| development | V2_epoch40_COCO_prompt_0_20 | destination_objects | 480 | 0.0000 | 1.0000 | 0.0000 | 0.9771 |
| development | V2_epoch40_COCO_prompt_0_20 | small relevant instances (<4 patch-eq; union-mask diagnostic) | 694 | 0.5572570068115409 | — | 0.018450639611959554 | — |
| development | V2_epoch40_dev_98pct_recall | task_relevant_fixtures | 96 | 1.0000 | 0.0000 | 0.1502 | 0.0000 |
| development | V2_epoch40_dev_98pct_recall | robot_arm | 576 | 0.9770 | 0.0230 | 0.3197 | 0.0000 |
| development | V2_epoch40_dev_98pct_recall | robot_gripper | 576 | 0.9989 | 0.0011 | 0.0378 | 0.0000 |
| development | V2_epoch40_dev_98pct_recall | robot_mount | 384 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| development | V2_epoch40_dev_98pct_recall | destination_or_fixture | 576 | 0.7769 | 0.2231 | 0.0859 | 0.0000 |
| development | V2_epoch40_dev_98pct_recall | task_relevant_foreground | 576 | 0.9890 | 0.0110 | 0.2027 | 0.0000 |
| development | V2_epoch40_dev_98pct_recall | manipulated_objects | 470 | 0.9031 | 0.0969 | 0.0142 | 0.0000 |
| development | V2_epoch40_dev_98pct_recall | destination_objects | 480 | 0.5081 | 0.4919 | 0.0831 | 0.0000 |
| development | V2_epoch40_dev_98pct_recall | small relevant instances (<4 patch-eq; union-mask diagnostic) | 694 | 0.9856499705331998 | — | 0.010185977560145956 | — |
| held_out | V1_epoch50_seven_templates | manipulated_objects | 576 | 0.0000 | 1.0000 | 0.0000 | 0.9097 |
| held_out | V1_epoch50_seven_templates | task_relevant_fixtures | 192 | 0.3211 | 0.6789 | 0.1313 | 0.0000 |
| held_out | V1_epoch50_seven_templates | robot_arm | 576 | 0.5419 | 0.4581 | 0.1666 | 0.0000 |
| held_out | V1_epoch50_seven_templates | robot_gripper | 576 | 0.8696 | 0.1304 | 0.0999 | 0.0000 |
| held_out | V1_epoch50_seven_templates | robot_mount | 384 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| held_out | V1_epoch50_seven_templates | destination_or_fixture | 576 | 0.2166 | 0.7834 | 0.1110 | 0.5000 |
| held_out | V1_epoch50_seven_templates | task_relevant_foreground | 576 | 0.6303 | 0.3697 | 0.2862 | 0.0000 |
| held_out | V1_epoch50_seven_templates | destination_objects | 384 | 0.0221 | 0.9779 | 0.0218 | 0.8880 |
| held_out | V1_epoch50_seven_templates | small relevant instances (<4 patch-eq; union-mask diagnostic) | 848 | 0.646394729323589 | — | 0.01753600401979761 | — |
| held_out | V2_epoch40_COCO_prompt_0_20 | manipulated_objects | 576 | 0.1144 | 0.8856 | 0.0523 | 0.5608 |
| held_out | V2_epoch40_COCO_prompt_0_20 | task_relevant_fixtures | 192 | 0.0540 | 0.9460 | 0.0314 | 0.0000 |
| held_out | V2_epoch40_COCO_prompt_0_20 | robot_arm | 576 | 0.2231 | 0.7769 | 0.1695 | 0.0000 |
| held_out | V2_epoch40_COCO_prompt_0_20 | robot_gripper | 576 | 0.7740 | 0.2260 | 0.1988 | 0.0104 |
| held_out | V2_epoch40_COCO_prompt_0_20 | robot_mount | 384 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| held_out | V2_epoch40_COCO_prompt_0_20 | destination_or_fixture | 576 | 0.0351 | 0.9649 | 0.0225 | 0.4896 |
| held_out | V2_epoch40_COCO_prompt_0_20 | task_relevant_foreground | 576 | 0.3958 | 0.6042 | 0.2966 | 0.0000 |
| held_out | V2_epoch40_COCO_prompt_0_20 | destination_objects | 384 | 0.0000 | 1.0000 | 0.0000 | 0.9792 |
| held_out | V2_epoch40_COCO_prompt_0_20 | small relevant instances (<4 patch-eq; union-mask diagnostic) | 848 | 0.475383323684846 | — | 0.03156398701813132 | — |
| held_out | V2_epoch40_dev_98pct_recall | manipulated_objects | 576 | 0.7024 | 0.2976 | 0.0136 | 0.2292 |
| held_out | V2_epoch40_dev_98pct_recall | task_relevant_fixtures | 192 | 0.9501 | 0.0499 | 0.1260 | 0.0000 |
| held_out | V2_epoch40_dev_98pct_recall | robot_arm | 576 | 0.9746 | 0.0254 | 0.2936 | 0.0000 |
| held_out | V2_epoch40_dev_98pct_recall | robot_gripper | 576 | 0.9963 | 0.0037 | 0.0402 | 0.0000 |
| held_out | V2_epoch40_dev_98pct_recall | robot_mount | 384 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| held_out | V2_epoch40_dev_98pct_recall | destination_or_fixture | 576 | 0.8302 | 0.1698 | 0.0971 | 0.0000 |
| held_out | V2_epoch40_dev_98pct_recall | task_relevant_foreground | 576 | 0.9813 | 0.0187 | 0.1834 | 0.0000 |
| held_out | V2_epoch40_dev_98pct_recall | destination_objects | 384 | 0.5810 | 0.4190 | 0.0737 | 0.0000 |
| held_out | V2_epoch40_dev_98pct_recall | small relevant instances (<4 patch-eq; union-mask diagnostic) | 848 | 0.9981773069951712 | — | 0.009793929034859171 | — |

Development-only V2 conservative point: threshold **0.08**, task-relevant foreground recall **0.9920**, candidate background coverage **0.1313**. See `libero_ground_truth_metrics.json` for held-out metrics at this frozen threshold.

## Data and replay validation

- Official source: `yifengzhu-hf/LIBERO-datasets`, revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`.
- Selected tasks: 12 across `libero_goal, libero_object, libero_spatial`; task-level split: 6 development and 6 held-out.
- Consecutive replay states: 1152; selected demonstration sequences: 36.
- Actual frame count by split: development 6 tasks / 18 sequences / 576 frames; held-out 6 tasks / 18 sequences / 576 frames.

| Suite | Task | Split | HDF5 bytes | HDF5 SHA-256 | Sequences | Frames |
|---|---|---|---:|---|---:|---:|
| libero_spatial | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | development | 508779600 | `ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead` | 3 | 96 |
| libero_spatial | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | development | 589630943 | `75ede0cf5fbfc925093671b55032ad80f1b1f1cf35442ec6c663460d37d3e0b3` | 3 | 96 |
| libero_spatial | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | held_out | 748271833 | `7ff1ee60447875c152d8c7819f499d23dbc0eb4d4bf045225d5d87a2cdad99bd` | 3 | 96 |
| libero_spatial | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | held_out | 711715594 | `1fcb3990cf9fd6185f56531cc9f6dec0a77a59bfa0c7f39dff95b7e43cf6aeaa` | 3 | 96 |
| libero_object | pick_up_the_alphabet_soup_and_place_it_in_the_basket | development | 780145352 | `42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d` | 3 | 96 |
| libero_object | pick_up_the_butter_and_place_it_in_the_basket | development | 785532626 | `07f48f1c0c06b5e10bc219e63701f0240d8205f85ccd0daab3ff9bd27bc35fb9` | 3 | 96 |
| libero_object | pick_up_the_salad_dressing_and_place_it_in_the_basket | held_out | 664188039 | `45960606a85b9c68980eb8130e0fe16bf3621539cf5d4147f35aa8139573c5ac` | 3 | 96 |
| libero_object | pick_up_the_tomato_sauce_and_place_it_in_the_basket | held_out | 735593408 | `43c52bdaa78b4c6aff3afb54d932d6a0e065f5c30c8e30a6c17f78749d487ea9` | 3 | 96 |
| libero_goal | open_the_middle_drawer_of_the_cabinet | development | 702223367 | `20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737` | 3 | 96 |
| libero_goal | put_the_bowl_on_the_plate | development | 468246288 | `e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406` | 3 | 96 |
| libero_goal | open_the_top_drawer_and_put_the_bowl_inside | held_out | 1017886342 | `1b50d99a053a7e4547c9ae9ccd2e3cd762543d27be78f240f13440d7ae487ae8` | 3 | 96 |
| libero_goal | put_the_wine_bottle_on_the_rack | held_out | 878958730 | `f9092aa70734fc4083e97fc58c3ba25f87c614d18326182ddc7a455f0ab4da2e` | 3 | 96 |
- LIBERO source revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`; robosuite revision: `1b8b97ece22de6c7905adae88228c2ca1bcce3f2`.
- HDF5 `states` and `actions` were audited, but actions were never applied. Each episode's stored `model_file` was loaded, and the recorded flattened state was installed directly into MuJoCo.
- The RGB observation and instance labels were rendered from the same restored simulator state. Direct RGB sensor parity and raw geom-ID → `instances_to_ids` → rendered-label `index + 1` equality were checked in the two-task smoke test.
- The raw camera output follows the installed OpenGL/OpenCV image convention. The binary instance masks use nearest-neighbor image preprocessing and fractional area pooling to the 16×16 OpenVLA patch grid.
- Separate `Panda*`, `PandaGripper*`, and `RethinkMount*` labels were preserved from the low-level instance sensor. The wrapper's helper that collapses the robot IDs was not used to construct the per-role labels.

## Role definitions and prompts

Ground-truth manipulated/destination/fixture identities come from BDDL object/fixture declarations, `obj_of_interest`, and goal predicates. Region targets such as `basket_1_contain_region` are resolved to their owning object before role scoring; destination language masks are associated with object-vs-fixture roles post hoc using BDDL type names, without changing prompts or scores. Those simulator identities are scoring labels only. Model prompts are generated only from each HDF5 language instruction using the documented deterministic noun-phrase rules, plus the fixed `robot arm` and `robot gripper` prompts.

`Panda0`/mounted-Panda instances are arm, `PandaGripper*` instances are gripper, and `RethinkMount*` instances are robot mount. The fixed inference prompts include arm and gripper, but do not include a dedicated robot-mount prompt; mount recall is therefore reported as zero and should not be interpreted as validated safe background. Non-target object instances are other objects; non-target fixtures and world/background are structural background. Unmapped instance labels are marked unresolved and excluded from the valid scoring area rather than assumed background.

## Model comparisons and limitations

V2 epoch-40 checkpoint SHA-256 `8232ec8f657da47645d73741b16560a1140bfea89cce6826948b9457ce4854d4` uses penultimate OpenVLA DINO patches and the COCO-trained `a photo of a …` prompt form with threshold 0.20. V1 epoch-50 SHA-256 `f1c57d5e3a52fde5ed6d17b5fad47aadaa6ce25dc58cab72a3aba9a180d015b1` is a control using its published seven-template average and its established final-normalized DINO features with fixed probability threshold 0.55.

V1 and V2 therefore differ in visual feature layer and training objective as well as the projection checkpoint; the comparison is of the frozen systems, not an isolated training-loss ablation. Category prompts cannot distinguish two same-class instances reliably; per-instance target scores are consequently a stricter grounding diagnostic.

R5 age-1–4 tables compare simulator foreground unions and predicted current-frame masks only. They are offline mask-safety proxies, not a cache experiment and not proof that reusing K/V is safe. Motionless visible background is not treated as safe by definition.

The held-out scores were materialized before the BDDL region-role audit and were inspected while diagnosing the role-map issue. They were then recomputed with the corrected role mapping; no held-out scores influenced the development-only threshold calibration. Thus the held-out report is not a blind untouched test.

The task set is a 12-task selected subset, not the full LIBERO suite. A 98% development recall operating point may leave little reusable coverage; even a favorable mask metric would require a later, separately authorized cache-safety experiment.

## Consecutive-frame and R5-age diagnostic

Each selected sequence is treated as a fresh anchor at its first sampled state; rows then use ages 1–4 and refresh again every five sampled states. `stale_source_fg_among_reused` counts patches foreground in the anchor frame, background in the current frame, and therefore candidate-reused by the current predicted mask. These are offline mask/label overlap proxies only.

| Split | Condition | Age | Pairs | Mean source/current GT IoU | Mean source-FG→current-BG patches | Mean stale source-FG among reused | Mean current-GT-FG missed | Mean candidate reused tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| development | V2_epoch40_COCO_prompt_0_20 | 1 | 126 | 0.985 | 0.62 | 0.38 | 41.90 | 185.6 |
| development | V2_epoch40_COCO_prompt_0_20 | 2 | 108 | 0.973 | 1.04 | 0.65 | 41.02 | 184.9 |
| development | V2_epoch40_COCO_prompt_0_20 | 3 | 108 | 0.960 | 1.44 | 0.90 | 41.45 | 185.4 |
| development | V2_epoch40_COCO_prompt_0_20 | 4 | 108 | 0.948 | 1.77 | 1.14 | 41.67 | 185.2 |
| development | V2_epoch40_dev_98pct_recall | 1 | 126 | 0.985 | 0.62 | 0.00 | 1.57 | 37.7 |
| development | V2_epoch40_dev_98pct_recall | 2 | 108 | 0.973 | 1.04 | 0.01 | 1.53 | 38.1 |
| development | V2_epoch40_dev_98pct_recall | 3 | 108 | 0.960 | 1.44 | 0.00 | 1.59 | 37.5 |
| development | V2_epoch40_dev_98pct_recall | 4 | 108 | 0.948 | 1.77 | 0.00 | 1.57 | 37.3 |
| held_out | V2_epoch40_COCO_prompt_0_20 | 1 | 126 | 0.987 | 0.33 | 0.25 | 51.06 | 224.6 |
| held_out | V2_epoch40_COCO_prompt_0_20 | 2 | 108 | 0.973 | 0.87 | 0.66 | 50.90 | 224.7 |
| held_out | V2_epoch40_COCO_prompt_0_20 | 3 | 108 | 0.962 | 1.30 | 1.07 | 50.44 | 224.4 |
| held_out | V2_epoch40_COCO_prompt_0_20 | 4 | 108 | 0.950 | 1.68 | 1.45 | 50.66 | 224.5 |
| held_out | V2_epoch40_dev_98pct_recall | 1 | 126 | 0.987 | 0.33 | 0.00 | 1.65 | 25.2 |
| held_out | V2_epoch40_dev_98pct_recall | 2 | 108 | 0.973 | 0.87 | 0.01 | 1.71 | 25.4 |
| held_out | V2_epoch40_dev_98pct_recall | 3 | 108 | 0.962 | 1.30 | 0.02 | 1.67 | 25.7 |
| held_out | V2_epoch40_dev_98pct_recall | 4 | 108 | 0.950 | 1.68 | 0.04 | 1.69 | 25.5 |

## Artifacts

- `libero_task_manifest.json`: pinned source revision, task split, file hashes, HDF5 schema, episode ranges, and frame/state hashes.
- `libero_ground_truth_metrics.json`: per-role, per-task, per-model and R5-age summaries.
- `per_task_metrics.csv`, `per_frame_diagnostics.csv`, `temporal_r5_age_metrics.csv`.
- `visualizations/`: deterministic first-frame examples for each selected task.
- `smoke_validation.json` and `logs/`: exact mapping checks and execution record.
