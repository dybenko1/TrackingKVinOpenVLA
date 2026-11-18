# LIBERO mask feasibility (offline saved frames)

Applied frozen epoch_050 Talk2DINO-SigLIP projection and the development-selected `pamr_raw` postprocessing (threshold 0.55) to 27 saved RGB frames across tasks 4, 5, 7 and states 0–2. Prompts per frame include the instruction-derived manipulated object, robot arm, robot gripper, table, and background scene.

This is a visualization and temporal-mask-stability probe only. It did not run policy inference, rollouts, or KV reuse. The saved frame set has no paired simulator oracle masks, so it cannot provide precision/recall/IoU against robot-object ground truth. Frame gaps are sparse early/middle/late snapshots; mask changes across them are not stepwise noise estimates. The score threshold is calibrated on COCO category prompts, not on these task-specific phrases. These masks are not yet validated as safe K/V reuse decisions.

Inspect `visualizations/` for RGB, score maps and binary 16×16 fresh-token masks; `temporal_stability.csv` reports sparse mask IoU and token flips. `per_frame_masks.csv` records prompt token counts.
