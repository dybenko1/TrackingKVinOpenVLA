# LIBERO Ground-Truth Availability Audit

## Outcome

The complete sequential segmentation evaluation cannot be reconstructed from the current saved artifacts alone. This audit did not run policy inference, alter simulator state, or derive labels from RGB.

## Installed source inventory

- LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- robosuite revision: `1b8b97ece22de6c7905adae88228c2ca1bcce3f2`
- BDDL task files: 130
- Available demonstration HDF5 files across LIBERO and robosuite source roots: 3
- LIBERO `*.pruned_init` task initial-state files: 130; these are reset variants, not recorded consecutive demonstrations.
- Saved RGB snapshots in the earlier visualization artifact: 27, spanning task IDs [4, 5, 7] and 9 task/state pairs.
- Simulator API indicators: `SegmentationRenderEnv=True`, `camera_segmentations=True`, `instances_to_ids=True`.

The simulator and robosuite source expose instance segmentation/mapping support, and LIBERO initial-state assets are available. However, no LIBERO demonstration trajectory is present to provide task-motion sequences; the only HDF5 files are unrelated robosuite lift/wipe/handover examples. The saved RGB snapshots are sparse and have no matching simulator state/camera provenance.

## Consequence

No per-role IoU, task-relevant recall, unseen-concept accuracy, consecutive task-motion turnover, or R5 age-1–4 mask-safety metric is reported. The minimum next input is recorded task trajectories or explicit authorization to generate clearly labeled non-policy simulator sequences from reset-state variants. This stage does not block the independent prompt and integration experiments.
