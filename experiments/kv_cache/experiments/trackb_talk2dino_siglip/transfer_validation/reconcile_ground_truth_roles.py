#!/usr/bin/env python3
"""Rebuild post-hoc LIBERO role metrics from saved same-state patch scores.

This is a metrics/label audit only: it does not load a model or recalculate
V1/V2 patch scores. Original scoring caches are read-only and preserved.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.transfer_validation import (
    libero_demo_ground_truth as e2,
)


def atomic_npz(path: Path, arrays: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def corrected_roles(task: dict) -> dict:
    roles = e2.parse_bddl_roles(Path(task["bddl_path"]))
    e2.add_instruction_named_fixtures(roles, task["instruction"] or "")
    return roles


def rebuild_one(source: Path, target: Path, manifest_hash: str, role_by_task: dict) -> tuple[dict, dict]:
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files if k != "metadata"}
        meta = json.loads(str(z["metadata"].item()))
    if meta.get("task_manifest_sha256") != manifest_hash:
        raise RuntimeError(f"Source cache has wrong task manifest: {source}")
    roles = role_by_task[meta["task_id"]]
    names = meta["instance_keys"]
    instance_values = arrays["instance_gt_occupancy"].astype(np.float32)
    per_frame = []
    keys = set()
    new_instance_roles = {name: e2.role_for_instance(name, roles) for name in names}
    for ti in range(instance_values.shape[0]):
        frame_roles = {}
        for name, occupancy in zip(names, instance_values[ti]):
            role = new_instance_roles[name]
            if role == "unresolved_instance":
                continue
            frame_roles.setdefault(role, np.zeros(e2.GRID * e2.GRID, dtype=np.float32))
            frame_roles[role] += occupancy
        for role in frame_roles:
            frame_roles[role] = np.clip(frame_roles[role], 0.0, 1.0)
        relevant = np.zeros(e2.GRID * e2.GRID, dtype=np.float32)
        for role in ("manipulated_objects", "destination_objects", "task_relevant_objects",
                     "task_relevant_fixtures", "robot_arm", "robot_gripper", "robot_mount"):
            if role in frame_roles:
                relevant = np.maximum(relevant, frame_roles[role])
        frame_roles["task_relevant_foreground"] = relevant
        keys.update(frame_roles)
        per_frame.append(frame_roles)
    role_keys = sorted(keys)
    role_array = np.zeros((len(per_frame), len(role_keys), e2.GRID * e2.GRID), dtype=np.float16)
    for ti, frame_roles in enumerate(per_frame):
        for ri, role in enumerate(role_keys):
            if role in frame_roles:
                role_array[ti, ri] = frame_roles[role]
    arrays["role_gt_occupancy"] = role_array
    meta["role_keys"] = role_keys
    meta["bddl_roles"] = roles
    meta["instance_roles"] = new_instance_roles
    meta["ground_truth_role_mapping_revision"] = "region-aware-bddl-v2"
    atomic_npz(target, arrays, meta)
    return arrays, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--libero-root", type=Path, required=True)
    p.add_argument("--robosuite-root", type=Path, required=True)
    p.add_argument("--task-manifest", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    manifest = json.loads(a.task_manifest.read_text())
    manifest_hash = manifest["manifest_sha256"]
    old_metrics = out / "libero_ground_truth_metrics.json"
    old_report = out / "LIBERO_GROUND_TRUTH_REPORT.md"
    if old_metrics.exists() and not (out / "initial_metrics_pre_role_audit.json").exists():
        shutil.copy2(old_metrics, out / "initial_metrics_pre_role_audit.json")
    if old_report.exists() and not (out / "initial_report_pre_role_audit.md").exists():
        shutil.copy2(old_report, out / "initial_report_pre_role_audit.md")

    task_by_id = {t["task_id"]: t for t in manifest["tasks"]}
    roles_by_task = {tid: corrected_roles(task) for tid, task in task_by_id.items()}
    role_audit = {
        "revision": "region-aware-bddl-v2",
        "method": "BDDL goal region symbols are resolved to their owning object/fixture instance; saved per-instance simulator occupancies are re-aggregated without changing any image, text embedding, V1/V2 patch score, or threshold rule.",
        "task_ground_truth_roles": {
            tid: role["role_instance_names"] for tid, role in roles_by_task.items()
        },
        "heldout_metrics_were_materialized_before_role_audit": True,
        "heldout_metrics_were_inspected_during_audit": True,
        "heldout_thresholds_selected_from": "development tasks only",
    }
    e2.atomic_json(out / "ground_truth_role_audit.json", role_audit)

    role_cache = out / "role_reconciled_scoring_cache"
    if role_cache.exists():
        shutil.rmtree(role_cache)
    role_cache.mkdir(parents=True)
    rebuilt_paths = []
    paths = sorted((a.cache_root / "scoring").glob("*/*.npz"))
    if len(paths) != int(manifest["total_sequences"]):
        raise RuntimeError(f"Expected {manifest['total_sequences']} sequence score caches, found {len(paths)}")
    for source in paths:
        # Preserve the original stable relative path structure.
        target = role_cache / source.relative_to(a.cache_root / "scoring")
        rebuild_one(source, target, manifest_hash, roles_by_task)
        rebuilt_paths.append(target)

    # Re-render one predeclared first state per task so the illustrative
    # simulator GT panels use the reconciled role mapping. Score maps come
    # unchanged from the saved first-frame inference cache.
    visual_dir = out / "visualizations"
    old_visual_dir = out / "initial_visualizations_pre_role_audit"
    if visual_dir.exists() and not old_visual_dir.exists():
        shutil.copytree(visual_dir, old_visual_dir)
    visual_dir.mkdir(parents=True, exist_ok=True)
    prep = json.loads((out / "model_and_preprocessing.json").read_text())
    geometry = prep["policy_image_processor_geometry"]
    galleries = []
    source_by_task = {}
    for source in paths:
        with np.load(source, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
        source_by_task.setdefault(meta["task_id"], source)
    for task in manifest["tasks"]:
        segment = task["selected_sequences"][0]
        source = source_by_task[task["task_id"]]
        with np.load(source, allow_pickle=False) as z:
            arrays = {k: np.array(z[k]) for k in z.files if k != "metadata"}
            score_meta = json.loads(str(z["metadata"].item()))
        rendered = e2.render_one_state(task, segment["demo"], int(segment["start"]),
                                       libero_root=a.libero_root, robosuite_root=a.robosuite_root,
                                       image_size=224, validate_mapping=False)
        role_masks, _, joint = e2.grid_masks(rendered["labels"], rendered["instance_names"],
                                             roles_by_task[task["task_id"]], geometry)
        role_masks["task_relevant_foreground"] = joint[0]
        role_keys = sorted(role_masks)
        galleries.append({
            "rgb": rendered["rgb"], "task": task, "prompt_specs": score_meta["prompt_specs"],
            "scores_v2": arrays["v2_probabilities"][0], "scores_v1": arrays["v1_probabilities"][0],
            "role_gt": np.stack([role_masks[key] for key in role_keys]),
            "role_keys": role_keys, "cache_meta": score_meta,
        })
    e2.render_deterministic_gallery(galleries, visual_dir, geometry)

    dev_rows = e2.aggregate_arrays(rebuilt_paths, "development", manifest_hash)
    held_rows = e2.aggregate_arrays(rebuilt_paths, "held_out", manifest_hash)
    calibration = e2.calibrate_dev(dev_rows)
    dev_metrics = {
        "V1_epoch50_seven_templates": e2.collect_metrics(dev_rows, "v1", e2.V1_THRESHOLD),
        "V2_epoch40_COCO_prompt_0_20": e2.collect_metrics(dev_rows, "v2", e2.V2_THRESHOLD),
        "V2_epoch40_dev_98pct_recall": e2.collect_metrics(dev_rows, "v2", calibration["threshold"]),
    }
    held_metrics = {
        "V1_epoch50_seven_templates": e2.collect_metrics(held_rows, "v1", e2.V1_THRESHOLD),
        "V2_epoch40_COCO_prompt_0_20": e2.collect_metrics(held_rows, "v2", e2.V2_THRESHOLD),
        "V2_epoch40_dev_98pct_recall": e2.collect_metrics(held_rows, "v2", calibration["threshold"]),
    }
    e2.atomic_json(out / "development_calibration.json", {
        "status": "recomputed_after_ground_truth_role_audit_before_reporting_heldout_metrics",
        "manifest_sha256": manifest_hash,
        "v2_threshold_0_20_metrics": dev_metrics["V2_epoch40_COCO_prompt_0_20"],
        "v1_threshold_0_55_metrics": dev_metrics["V1_epoch50_seven_templates"],
        "v2_98_percent_recall_point": calibration,
        "calibration_scope": "development tasks only",
        "ground_truth_role_mapping_revision": "region-aware-bddl-v2",
    })
    e2.atomic_json(out / "libero_ground_truth_metrics.json", {
        "status": "complete",
        "manifest_sha256": manifest_hash,
        "ground_truth_role_mapping_revision": "region-aware-bddl-v2",
        "development_task_metrics": dev_metrics,
        "held_out_task_metrics": held_metrics,
        "threshold_calibration": calibration,
        "threshold_calibration_used_only_development": True,
        "heldout_scores_materialized_before_role_audit": True,
        "heldout_threshold_selection_used": False,
        "model_weights_modified": False,
        "policy_rollouts": False,
        "recorded_actions_applied": False,
        "kv_cache_changed": False,
        "r5_is_offline_proxy_only": True,
    })
    task_split = {t["task_id"]: t["split"] for t in manifest["tasks"]}
    temporal_rows = []
    for split, metrics in (("development", dev_metrics), ("held_out", held_metrics)):
        for condition in ("V2_epoch40_COCO_prompt_0_20", "V2_epoch40_dev_98pct_recall"):
            result = metrics[condition]
            for row in result.get("r5_temporal_proxy", []):
                temporal_rows.append({**row, "split": split, "condition": condition,
                                      "threshold": result["threshold"]})
    e2.write_csv(out / "temporal_r5_age_metrics.csv", temporal_rows)
    grouped = defaultdict(list)
    for row in temporal_rows:
        grouped[(row["split"], row["condition"], row["source_age"])].append(row)
    age_summary = []
    summary_fields = ("source_current_gt_iou", "source_fg_current_bg_tokens",
                      "current_fg_source_bg_tokens", "current_gt_fg_not_pred_fg",
                      "candidate_reused_tokens", "reused_tokens_that_are_current_gt_fg",
                      "stale_source_fg_among_reused")
    for (split, condition, age), values in sorted(grouped.items()):
        row = {"split": split, "condition": condition, "threshold": values[0]["threshold"],
               "source_age": age, "comparison_count": len(values)}
        for field in summary_fields:
            row[f"mean_{field}"] = float(np.mean([x[field] for x in values]))
            row[f"sum_{field}"] = float(np.sum([x[field] for x in values]))
        age_summary.append(row)
    e2.write_csv(out / "r5_age_summary.csv", age_summary)
    task_csv = []
    for split, metrics in (("development", dev_metrics), ("held_out", held_metrics)):
        for model_name, result in metrics.items():
            for task_id, role_metrics in result["per_task_metrics"].items():
                for role, value in role_metrics.items():
                    task_csv.append({"split": split, "model": model_name, "task_id": task_id, "role": role, **value})
    e2.write_csv(out / "per_task_metrics.csv", task_csv)
    frame_rows = []
    for rows in (dev_rows, held_rows):
        for arrays, meta in rows:
            roles = {r: i for i, r in enumerate(meta["role_keys"])}
            for ti in range(arrays["v2_probabilities"].shape[0]):
                v2 = e2.get_union_mask(arrays["v2_probabilities"][ti], e2.V2_THRESHOLD)
                v1 = e2.get_union_mask(arrays["v1_probabilities"][ti], e2.V1_THRESHOLD)
                gt = np.zeros(256, np.float32)
                for role in ("manipulated_objects", "destination_objects", "task_relevant_fixtures",
                             "robot_arm", "robot_gripper", "robot_mount"):
                    if role in roles:
                        gt = np.maximum(gt, arrays["role_gt_occupancy"][ti, roles[role]].astype(np.float32))
                frame_rows.append({
                    "split": meta["split"], "task_id": meta["task_id"], "demo": meta["demo"],
                    "state_index": int(arrays["state_indices"][ti]),
                    "v2_fg_tokens_at_0_20": int(v2.sum()),
                    "v2_candidate_bg_tokens_at_0_20": int((~v2).sum()),
                    "v1_fg_tokens_at_0_55": int(v1.sum()),
                    "gt_task_relevant_patch_equivalents": float(gt.sum()),
                    "v2_missed_gt_occupancy": float((gt * (~v2)).sum()),
                    "v1_missed_gt_occupancy": float((gt * (~v1)).sum()),
                    "source_age_score": "current-frame rendering",
                })
    e2.write_csv(out / "per_frame_diagnostics.csv", frame_rows)
    e2.generate_report(out, manifest, dev_metrics, held_metrics, calibration,
                       {"libero": manifest["libero_revision"], "robosuite": manifest["robosuite_revision"]},
                       lambda message: print(message, flush=True))
    report_path = out / "LIBERO_GROUND_TRUTH_REPORT.md"
    report_text = report_path.read_text()
    report_text = report_text.replace("pixels/paches", "patches")
    report_text = report_text.replace(
        "Ground-truth manipulated/destination/fixture identities come from BDDL object/fixture declarations, `obj_of_interest`, and goal predicates.",
        "Ground-truth manipulated/destination/fixture identities come from BDDL object/fixture declarations, `obj_of_interest`, and goal predicates. Region targets such as `basket_1_contain_region` are resolved to their owning object before role scoring; destination language masks are associated with object-vs-fixture roles post hoc using BDDL type names, without changing prompts or scores."
    )
    report_text = report_text.replace(
        "`Panda0`/mounted-Panda instances are arm, `PandaGripper*` instances are gripper, and `RethinkMount*` instances are robot mount.",
        "`Panda0`/mounted-Panda instances are arm, `PandaGripper*` instances are gripper, and `RethinkMount*` instances are robot mount. The fixed inference prompts include arm and gripper, but do not include a dedicated robot-mount prompt; mount recall is therefore reported as zero and should not be interpreted as validated safe background."
    )
    report_text = report_text.replace(
        "The task set is a 12-task selected subset, not the full LIBERO suite.",
        "The held-out scores were materialized before the BDDL region-role audit and were inspected while diagnosing the role-map issue. They were then recomputed with the corrected role mapping; no held-out scores influenced the development-only threshold calibration. Thus the held-out report is not a blind untouched test.\n\nThe task set is a 12-task selected subset, not the full LIBERO suite."
    )
    age_lines = [
        "## Consecutive-frame and R5-age diagnostic",
        "",
        "Each selected sequence is treated as a fresh anchor at its first sampled state; rows then use ages 1–4 and refresh again every five sampled states. `stale_source_fg_among_reused` counts patches foreground in the anchor frame, background in the current frame, and therefore candidate-reused by the current predicted mask. These are offline mask/label overlap proxies only.",
        "",
        "| Split | Condition | Age | Pairs | Mean source/current GT IoU | Mean source-FG→current-BG patches | Mean stale source-FG among reused | Mean current-GT-FG missed | Mean candidate reused tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in age_summary:
        age_lines.append(
            f"| {row['split']} | {row['condition']} | {row['source_age']} | {row['comparison_count']} | "
            f"{row['mean_source_current_gt_iou']:.3f} | {row['mean_source_fg_current_bg_tokens']:.2f} | "
            f"{row['mean_stale_source_fg_among_reused']:.2f} | {row['mean_current_gt_fg_not_pred_fg']:.2f} | "
            f"{row['mean_candidate_reused_tokens']:.1f} |"
        )
    report_text = report_text.replace("## Artifacts", "\n".join(age_lines) + "\n\n## Artifacts")
    status_path = out / "RUN_STATUS.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        status["ground_truth_role_mapping_revision"] = "region-aware-bddl-v2"
        status["metrics_reconciled_from_saved_scores"] = True
        status["heldout_metrics_materialized_and_inspected_before_role_audit"] = True
        status["final_metrics_path"] = str(out / "libero_ground_truth_metrics.json")
        e2.atomic_json(status_path, status)
    report_path.write_text(report_text)
    print(json.dumps({"status": "role_metrics_reconciled", "manifest_sha256": manifest_hash,
                      "threshold_98_recall": calibration, "development_sequences": len(dev_rows),
                      "heldout_sequences": len(held_rows), "development_frames": sum(x[0]["v2_probabilities"].shape[0] for x in dev_rows),
                      "heldout_frames": sum(x[0]["v2_probabilities"].shape[0] for x in held_rows)}, indent=2))


if __name__ == "__main__":
    main()
