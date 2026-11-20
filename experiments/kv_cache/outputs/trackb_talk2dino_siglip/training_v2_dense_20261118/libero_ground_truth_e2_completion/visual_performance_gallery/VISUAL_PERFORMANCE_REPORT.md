# Visual performance and reclassification gallery

Generated solely from corrected E2 score/occupancy caches and the matching recorded HDF5 `agentview_rgb` frames. No inference, replay, training, or policy execution was performed. The HDF5 files were SHA-256 checked against the frozen manifest. Role audit revision: `region-aware-bddl-v2`.

Binary token statistics classify a GT patch positive when its fractional occupancy is >0. TP/FP/FN/TN are counts over 256 cells. E2 aggregate recall in the report is occupancy-weighted, so these panel metrics are not numerically identical. Reclassification at .08 vs .20 labels switched foreground cells: green = relevant GT correction; orange = GT background newly protected; red = still-missed GT; dark = correctly reusable background.

## Dataset-level results

| Held-out method | Occupancy-weighted foreground recall | False-background | Mean candidate reusable tokens | Background coverage |
|---|---:|---:|---:|---:|
| V1_epoch50_seven_templates | 0.6303 | 0.3697 | 176.8/256 | 0.690 |
| V2_epoch40_COCO_prompt_0_20 | 0.3958 | 0.6042 | 224.4/256 | 0.877 |
| V2_epoch40_dev_98pct_recall | 0.9813 | 0.0187 | 25.5/256 | 0.099 |

## Selected frame gallery

### 1. task_median_fn — libero_goal / open_the_middle_drawer_of_the_cabinet / demo_10 / state 107

![Comparison panel](gallery/01_libero_goal_open_the_middle_drawer_of_the_cabinet_demo_10_s107.png)

### 2. task_median_fn — libero_goal / open_the_top_drawer_and_put_the_bowl_inside / demo_1 / state 95

![Comparison panel](gallery/02_libero_goal_open_the_top_drawer_and_put_the_bowl_inside_demo_1_s095.png)

### 3. task_median_fn — libero_goal / put_the_bowl_on_the_plate / demo_0 / state 24

![Comparison panel](gallery/03_libero_goal_put_the_bowl_on_the_plate_demo_0_s024.png)

### 4. task_median_fn — libero_goal / put_the_wine_bottle_on_the_rack / demo_0 / state 20

![Comparison panel](gallery/04_libero_goal_put_the_wine_bottle_on_the_rack_demo_0_s020.png)

### 5. task_median_fn — libero_object / pick_up_the_alphabet_soup_and_place_it_in_the_basket / demo_10 / state 147

![Comparison panel](gallery/05_libero_object_pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo_10_s147.png)

### 6. task_median_fn — libero_object / pick_up_the_butter_and_place_it_in_the_basket / demo_10 / state 140

![Comparison panel](gallery/06_libero_object_pick_up_the_butter_and_place_it_in_the_basket_demo_10_s140.png)

### 7. task_median_fn — libero_object / pick_up_the_salad_dressing_and_place_it_in_the_basket / demo_10 / state 143

![Comparison panel](gallery/07_libero_object_pick_up_the_salad_dressing_and_place_it_in_the_basket_demo_10_s143.png)

### 8. task_median_fn — libero_object / pick_up_the_tomato_sauce_and_place_it_in_the_basket / demo_1 / state 86

![Comparison panel](gallery/08_libero_object_pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo_1_s086.png)

### 9. task_median_fn — libero_spatial / pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate / demo_0 / state 20

![Comparison panel](gallery/09_libero_spatial_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo_0_s020.png)

### 10. task_median_fn — libero_spatial / pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate / demo_1 / state 69

![Comparison panel](gallery/10_libero_spatial_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo_1_s069.png)

### 11. task_median_fn — libero_spatial / pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate / demo_0 / state 15

![Comparison panel](gallery/11_libero_spatial_pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo_0_s015.png)

### 12. task_median_fn — libero_spatial / pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate / demo_0 / state 0

![Comparison panel](gallery/12_libero_spatial_pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo_0_s000.png)

### 13. worst_role_fn — libero_spatial / pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate / demo_1 / state 67

![Comparison panel](gallery/13_libero_spatial_pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo_1_s067.png)

### 14. worst_role_fn — libero_spatial / pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate / demo_1 / state 74

![Comparison panel](gallery/14_libero_spatial_pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo_1_s074.png)

### 15. worst_role_fn — libero_object / pick_up_the_butter_and_place_it_in_the_basket / demo_10 / state 120

![Comparison panel](gallery/15_libero_object_pick_up_the_butter_and_place_it_in_the_basket_demo_10_s120.png)

### 16. worst_role_fn — libero_object / pick_up_the_butter_and_place_it_in_the_basket / demo_10 / state 121

![Comparison panel](gallery/16_libero_object_pick_up_the_butter_and_place_it_in_the_basket_demo_10_s121.png)

### 17. v1_beats_v2 — libero_goal / put_the_wine_bottle_on_the_rack / demo_10 / state 142

![Comparison panel](gallery/17_libero_goal_put_the_wine_bottle_on_the_rack_demo_10_s142.png)

### 18. high_recall_low_reuse — libero_goal / open_the_top_drawer_and_put_the_bowl_inside / demo_1 / state 92

![Comparison panel](gallery/18_libero_goal_open_the_top_drawer_and_put_the_bowl_inside_demo_1_s092.png)

## Role-specific examples

These six panels pair each role occupancy mask with the union predictions. Roles without a dedicated inference prompt—including robot mount—are explicitly marked; union predictions do not claim role-specific classification. Aggregate held-out role recall is in the parent E2 report.

![Role diagnostics 1](gallery/role_01.png)

![Role diagnostics 2](gallery/role_02.png)

![Role diagnostics 3](gallery/role_03.png)

![Role diagnostics 4](gallery/role_04.png)

![Role diagnostics 5](gallery/role_05.png)

![Role diagnostics 6](gallery/role_06.png)

## Additional performance plots

![Score and false-background distributions](plots/score_and_fn_distributions.png)

![Per-task and switch accounting](plots/task_and_reclassification.png)

## Temporal strips

Colors: red = current GT foreground that threshold .20 would reuse; green = current GT foreground predicted fresh; cyan = GT foreground gained since the R5 source; orange = source GT foreground lost since source. The displayed predicted reuse is .20 only; summary CSV contains the E2 .08 vs .20 age tables. These are overlap proxies, not KV correctness. The panels visualize threshold 0.20; paired threshold 0.08 age-specific counts are tabulated in `../r5_age_summary.csv` and the full per-pair file.

![Temporal strip 1](temporal/1_libero_spatial.png)

![Temporal strip 2](temporal/2_libero_object.png)

![Temporal strip 3](temporal/3_libero_goal.png)

## Performance plots

![Recall versus reuse](plots/recall_vs_reuse.png)

![Per-role recall](plots/per_role_recall.png)

## Interpretation

The 0.20 cutoff misses substantial manipulated and destination foreground on held-out tasks. The development-calibrated 0.08 cutoff recovers much of the union foreground by making most patches fresh, sharply reducing reusable coverage. The robot mount has no dedicated prompt and is not validated as safe background. Small/thin objects remain coarse at 16×16; high occupancy recall must not be read as accurate boundaries. The held-out scores had already been inspected during the role-mapping audit, as disclosed in the E2 report; the 0.08 point was selected on development tasks only.

See `../LIBERO_GROUND_TRUTH_REPORT.md` for the complete role tables, source provenance, and R5 age summary. Per-example binary metrics are in `example_metrics.csv`; selected-frame and source hashes are in `gallery_manifest.json`.
