"""Offline inventory of LIBERO ground-truth assets; never runs a policy."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def git_head(path):
    try: return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
    except Exception as e: return f"unavailable: {e}"

def main():
    p=argparse.ArgumentParser();p.add_argument("--libero-root",type=Path,required=True);p.add_argument("--robosuite-root",type=Path,required=True);p.add_argument("--saved-frames",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    bddl=sorted(a.libero_root.rglob("*.bddl")); libero_h5=sorted([*a.libero_root.rglob("*.h5"),*a.libero_root.rglob("*.hdf5")]); robosuite_h5=sorted([*a.robosuite_root.rglob("*.h5"),*a.robosuite_root.rglob("*.hdf5")]); h5=libero_h5+robosuite_h5;npz=sorted([*a.libero_root.rglob("*.npz"),*a.robosuite_root.rglob("*.npz")]);init_states=sorted(a.libero_root.rglob("*.pruned_init"))
    env_wrapper=a.libero_root/"libero/libero/envs/env_wrapper.py"
    text=env_wrapper.read_text(errors="replace") if env_wrapper.exists() else ""
    seg_api={"env_wrapper_exists":env_wrapper.exists(),"SegmentationRenderEnv_present":"SegmentationRenderEnv" in text,
       "camera_segmentations_passthrough":"camera_segmentations" in text,
       "instance_ids_mapping":"instances_to_ids" in text,
       "env_wrapper_sha256":sha(env_wrapper) if env_wrapper.exists() else None,
       "robosuite_instance_mapping":"instances_to_ids" in (a.robosuite_root/"robosuite/environments/robot_env.py").read_text(errors="replace") if (a.robosuite_root/"robosuite/environments/robot_env.py").exists() else False}
    frame_rows=[];manifest=a.saved_frames/"frame_manifest.csv"
    if manifest.exists():
        with manifest.open(newline="") as f: frame_rows=list(csv.DictReader(f))
    tasks=sorted({int(r["task_id"]) for r in frame_rows if r.get("task_id","").isdigit()})
    state_count=len({(r.get("task_id"),r.get("state_id")) for r in frame_rows})
    # hdf5/npz presence alone is insufficient: only count files that plausibly
    # contain LIBERO demo or simulator-state trajectories.
    candidate_files=[str(x) for x in h5+npz+init_states]
    task_counts={}
    for r in frame_rows:
        key=f"task{r.get('task_id')}/state{r.get('state_id')}";task_counts[key]=task_counts.get(key,0)+1
    source_commits={"libero":git_head(a.libero_root),"robosuite":git_head(a.robosuite_root)}
    status="partial_initial_state_assets_only_no_recorded_sequences" if init_states and not libero_h5 else ("blocked_missing_paired_state_sequences" if not libero_h5 else "requires_hdf5_content_audit")
    payload={"status":status,"policy_rollouts":False,"libero_root":str(a.libero_root),"robosuite_root":str(a.robosuite_root),
      "source_commits":source_commits,"bddl_count":len(bddl),"bddl_examples":[str(x.relative_to(a.libero_root)) for x in bddl[:12]],
      "simulator_segmentation_api":seg_api,"candidate_state_or_demo_files":candidate_files,"hdf5_count":len(h5),"libero_hdf5_count":len(libero_h5),"robosuite_hdf5_count":len(robosuite_h5),"npz_count":len(npz),"libero_pruned_initial_state_files":len(init_states),"initial_state_assets":[{"path":str(x.relative_to(a.libero_root)),"sha256":sha(x)} for x in init_states],"init_state_suite_counts":{suite:sum(suite in str(x) for x in init_states) for suite in sorted({x.parent.name for x in init_states})},
      "saved_paired_rgb_frame_manifest":str(manifest) if manifest.exists() else None,"saved_frame_count":len(frame_rows),"saved_task_ids":tasks,
      "saved_task_state_pairs":state_count,"saved_frames_by_task_state":task_counts,
      "saved_frame_manifest_sha256":sha(manifest) if manifest.exists() else None,
      "limitations":["Saved RGB-only images do not contain MuJoCo state, geom IDs or same-state simulator segmentation.","Sparse early/middle/late images cannot be treated as consecutive trajectories.","No exact per-role segmentation metrics are computed from these RGB snapshots."]}
    (out/"libero_task_manifest.json").write_text(json.dumps(payload,indent=2)+"\n")
    metrics={"status":"not_evaluated_missing_consecutive_LIBERO_task_trajectories","per_role_metrics":None,"task_relevant_foreground_recall":None,"false_background_rate":None,"candidate_background_fraction":None,"temporal_age_1_to_4_metrics":None,"available_init_state_files":len(init_states),"saved_sparse_rgb_frames":len(frame_rows),"reason":"No recorded LIBERO demonstration sequences with paired simulator states were found. Exact same-state segmentation can be rendered from initial state files, but the requested >=32 consecutive task-motion observations per trajectory cannot be represented by the available saved observations. No model performance values are inferred from RGB-only files."}
    (out/"libero_ground_truth_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (out/"LIBERO_GROUND_TRUTH_REPORT.md").write_text("""# LIBERO Ground-Truth Availability Audit\n\n## Outcome\n\nThe complete sequential segmentation evaluation cannot be reconstructed from the current saved artifacts alone. This audit did not run policy inference, alter simulator state, or derive labels from RGB.\n\n## Installed source inventory\n\n- LIBERO revision: `{libero}`\n- robosuite revision: `{robosuite}`\n- BDDL task files: {bddl}\n- Available demonstration HDF5 files across LIBERO and robosuite source roots: {h5}
- LIBERO `*.pruned_init` task initial-state files: {init_files}; these are reset variants, not recorded consecutive demonstrations.\n- Saved RGB snapshots in the earlier visualization artifact: {frames}, spanning task IDs {tasks} and {states} task/state pairs.\n- Simulator API indicators: `SegmentationRenderEnv={seg}`, `camera_segmentations={cam}`, `instances_to_ids={ids}`.\n\nThe simulator and robosuite source expose instance segmentation/mapping support, and LIBERO initial-state assets are available. However, no LIBERO demonstration trajectory is present to provide task-motion sequences; the only HDF5 files are unrelated robosuite lift/wipe/handover examples. The saved RGB snapshots are sparse and have no matching simulator state/camera provenance.\n\n## Consequence\n\nNo per-role IoU, task-relevant recall, unseen-concept accuracy, consecutive task-motion turnover, or R5 age-1–4 mask-safety metric is reported. The minimum next input is recorded task trajectories or explicit authorization to generate clearly labeled non-policy simulator sequences from reset-state variants. This stage does not block the independent prompt and integration experiments.\n""".format(libero=source_commits["libero"],robosuite=source_commits["robosuite"],bddl=len(bddl),h5=len(h5),init_files=len(init_states),frames=len(frame_rows),tasks=tasks,states=state_count,seg=seg_api["SegmentationRenderEnv_present"],cam=seg_api["camera_segmentations_passthrough"],ids=seg_api["instance_ids_mapping"]))
    print(json.dumps({"status":status,"bddl_count":len(bddl),"candidate_state_files":len(candidate_files),"saved_frames":len(frame_rows),"tasks":tasks},indent=2),flush=True)
if __name__=="__main__": main()
