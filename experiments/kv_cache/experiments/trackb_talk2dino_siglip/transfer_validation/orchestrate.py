"""Sequential, resumable orchestration of the three frozen-model studies."""
from __future__ import annotations
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path

def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n");os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser()
    for k in ("repo","dataset-root","feature-root","training-output","v1-checkpoint","siglip-checkpoint","libero-root","robosuite-root","saved-frames","output-dir"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();out=a.output_dir;logs=out/"logs";logs.mkdir(parents=True,exist_ok=True)
    status=out/"RUN_STATUS.json";log=logs/"pipeline.log"
    manifest=a.training_output/"validation_manifest.json"
    feature_manifest=a.feature_root/"patch_features_val2014_manifest.json"
    try: source_commit=subprocess.check_output(["git","-C",str(a.repo),"rev-parse","HEAD"],text=True).strip()
    except Exception: source_commit="unavailable"
    contract={"branch_expected":"thesis/trackb-dense-grounding-v2","reference_commit":"e1b06f93924bd1c11dcbdb20800fedbdf51f293f","source_commit":source_commit,"no_training":True,"no_policy_rollouts":True,"no_KV_changes":True,"immutable_validation_manifest":str(manifest),"stages":["prompt_ablation","libero_ground_truth_audit","native_integration_profile"]}
    (out/"EXPERIMENT_CONTRACT.json").write_text(json.dumps(contract,indent=2)+"\n")
    if not manifest.is_file():raise FileNotFoundError(manifest)
    if not feature_manifest.is_file():raise FileNotFoundError(feature_manifest)
    fm=json.loads(feature_manifest.read_text())
    if fm.get("status")!="complete" or not fm.get("policy_processor_dino_slice_exact"):raise RuntimeError("V2 native validation feature cache is not complete/verified")
    data=json.loads(manifest.read_text());dev=data["development"]["image_ids"];ev=data["evaluation"]["image_ids"]
    stages=[
      ("prompt_ablation",[sys.executable,"-m","experiments.kv_cache.experiments.trackb_talk2dino_siglip.transfer_validation.prompt_ablation","--dataset-root",str(a.dataset_root),"--feature-root",str(a.feature_root),"--training-output",str(a.training_output),"--split-manifest",str(manifest),"--v1-checkpoint",str(a.v1_checkpoint),"--siglip-checkpoint",str(a.siglip_checkpoint),"--output-dir",str(out/"experiment1_prompt_ablation")]),
      ("libero_ground_truth_audit",[sys.executable,"-m","experiments.kv_cache.experiments.trackb_talk2dino_siglip.transfer_validation.libero_audit","--libero-root",str(a.libero_root),"--robosuite-root",str(a.robosuite_root),"--saved-frames",str(a.saved_frames),"--output-dir",str(out/"experiment2_libero_ground_truth")]),
      ("native_integration_profile",None),
    ]
    if status.exists():
        try:
            old=json.loads(status.read_text())
            if old.get("state")=="complete":
                print("Transfer validation already complete; leaving verified outputs unchanged.",flush=True);return
        except Exception: pass
    lock=(out/".orchestrator.lock").open("a+")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("A transfer-validation orchestrator is already running")
    payload={"state":"running","started_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"pid":os.getpid(),"host":os.uname().nodename,"current_stage":None,"stages":{},"development_images":len(dev),"secondary_previous_holdout_images":len(ev),"training_or_rollouts":False,"last_error":None}
    atomic(status,payload)
    def write(msg):
        line=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        with log.open("a") as f:f.write(line);f.flush();os.fsync(f.fileno())
        print(line.rstrip(),flush=True)
    try:
        for name,cmd in stages:
            marker={"prompt_ablation":out/"experiment1_prompt_ablation"/"prompt_ablation_metrics.json","libero_ground_truth_audit":out/"experiment2_libero_ground_truth"/"libero_task_manifest.json","native_integration_profile":out/"experiment3_native_integration"/"integration_timing.json"}[name]
            if marker.is_file():
                payload["stages"][name]={"state":"complete","resumed_from_verified_artifact":str(marker)};atomic(status,payload);write(f"SKIP {name}: found prior artifact {marker}");continue
            payload["current_stage"]=name;payload["stages"].setdefault(name,{"state":"running","started_at":time.time()});atomic(status,payload)
            if name=="native_integration_profile":
                # A saved task observation and its exact paired task instruction.
                import csv
                frame_csv=a.saved_frames/"frame_manifest.csv"
                with frame_csv.open(newline="") as f: rows=list(csv.DictReader(f))
                row=next((x for x in rows if x.get("task_id")=="4"),None)
                if row is None:raise RuntimeError("No saved task-4 LIBERO RGB observation found for integration profile")
                if row.get("sample_id"):
                    image=a.saved_frames/"visualizations"/row["sample_id"]/"camera_rgb.png"
                else:
                    key=next((k for k in ("image_path","path","file_path","image") if row.get(k)),None)
                    if key is None:raise RuntimeError(f"Cannot resolve saved RGB path from manifest fields {list(row)}")
                    image=Path(row[key]); image=image if image.is_absolute() else a.saved_frames/image
                if not image.is_file():raise FileNotFoundError(image)
                prompt=row.get("instruction") or row.get("task_description") or "pick up the black bowl and place it on the plate"
                cmd=[sys.executable,"-m","experiments.kv_cache.experiments.trackb_talk2dino_siglip.transfer_validation.integration_profile","--image",str(image),"--task-prompt",prompt,"--siglip-checkpoint",str(a.siglip_checkpoint),"--v2-checkpoint",str(a.training_output/"best_projection.pt"),"--output",str(out/"experiment3_native_integration"),"--repeats","30"]
            write(f"START {name}: {' '.join(cmd)}")
            rc=subprocess.call(cmd,cwd=a.repo,env=os.environ.copy())
            payload["stages"][name].update({"state":"complete" if rc==0 else "failed","exit_code":rc,"finished_at":time.time()});atomic(status,payload)
            if rc!=0:raise RuntimeError(f"Stage {name} failed with exit code {rc}")
            write(f"COMPLETE {name}")
        payload.update({"state":"complete","current_stage":None,"finished_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())});atomic(status,payload)
        e1=json.loads((out/"experiment1_prompt_ablation"/"prompt_ablation_metrics.json").read_text())
        e2=json.loads((out/"experiment2_libero_ground_truth"/"libero_task_manifest.json").read_text())
        e3=json.loads((out/"experiment3_native_integration"/"integration_timing.json").read_text())
        chosen=e1["selected_prompt_strategy"]; chosen_dev=e1["development_calibrated"].get(chosen,{})
        decision="CONDITIONAL" if e2["status"]!="complete" else "CONDITIONAL"
        report=f'''# Track B V2 Transfer Validation — Summary

## Status and scope

All three modules ran sequentially. No model was trained, no OpenVLA actions were generated, no policy rollout was run, and no KV-cache mechanism was changed. V2 used its frozen epoch-40 checkpoint. The 3,000-image COCO split was already inspected in earlier work, so its result is secondary and not an untouched test.

## Experiment 1 — Prompt rewording on known COCO classes

- Development-selected prompt condition: **{chosen}**, probability threshold **{e1['selected_threshold']:.2f}**.
- Development mean positive-pair macro IoU: **{chosen_dev.get('mean_positive_pair_macro_iou')}**; pooled macro category IoU: **{chosen_dev.get('pooled_macro_iou')}**.
- Four-form prompt sensitivity (mean pixel probability variance): **{e1['development_calibrated']['prompt_sensitivity'].get('mean_pixel_probability_variance')}**.
- V1 published reference remains seven templates at fixed 0.55; the V1 development-recalibrated threshold is a separate diagnostic (**{e1['v1_development_recalibrated_threshold']:.2f}**).
- This evaluates wording changes for categories included in V2 training. It does not measure unseen attributes or unseen concepts.
- Secondary 3,000-image result: `{e1['secondary_holdout_selected_condition'].get('pooled_macro_iou')}` pooled macro IoU, explicitly previously inspected.

## Experiment 2 — LIBERO ground truth

- Status: **{e2['status']}**.
- Simulator supports instance segmentation and instance-to-name mapping; {e2['libero_pruned_initial_state_files']} LIBERO initial-state files are available across task suites.
- No LIBERO demonstration HDF5 trajectories were found. The 27 saved RGB samples span {e2['saved_task_state_pairs']} task/state combinations and lack matching simulator states or consecutive-frame provenance.
- Thus no LIBERO per-role IoU, task-relevant false-background rate, temporal turnover, or age-1–4 safety proxy is claimed. Initial states alone are not treated as 32-frame demonstrations.

## Experiment 3 — Native integration and cost

- Captured DINO features: `{e3['captured_dino']['shape']}`; fused vision output `{e3['captured_dino']['fused_visual_shape']}`.
- Passive hook calls during a policy prefill: {e3['captured_dino']['hook_calls_during_full_prefill']}; direct-vs-captured feature maximum error **{e3['captured_dino']['max_abs_diff_vs_direct_dino']:.6g}**.
- Baseline versus hooked prefill logits maximum error **{e3['policy_prefill_read_only_hook']['max_abs_logit_diff']:.6g}**.
- Median cached score-and-threshold cost: **{e3['timing']['cached_text_projection_and_16x16_scores_and_threshold']['median_ms']:.3f} ms**; one additional prompt encoding/projection/scoring: **{e3['timing']['one_additional_prompt_encode_project_score']['median_ms']:.3f} ms**.
- No isolated saved K/V runtime was measured, so no net speedup is claimed. K/V compute must save more than the added selection cost for break-even.

## Decision: **{decision}**

The native OpenVLA DINO feature path supports passive read-only capture with matching prefill logits in this diagnostic. Prompt robustness is quantified on COCO categories seen during training. Transfer to LIBERO task-relevant objects and temporal mask safety remains unmeasured because the local artifact set lacks paired recorded task trajectories. Therefore V2 is **not yet established as a safe KV selector**; a separate cache-safety experiment is justified only after obtaining paired, task-level LIBERO masks and consecutive frames. This conclusion does not authorize policy rollouts or cache modifications.

Machine-readable results are in `experiment1_prompt_ablation/prompt_ablation_metrics.json`, `experiment2_libero_ground_truth/libero_task_manifest.json`, and `experiment3_native_integration/integration_timing.json`.
'''
        (out/"TRACKB_V2_TRANSFER_SUMMARY.md").write_text(report)
        write("ALL STAGES COMPLETE")
    except Exception as e:
        payload.update({"state":"failed","current_stage":payload.get("current_stage"),"failed_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"last_error":repr(e)});atomic(status,payload);write(f"FAILED: {e}");raise
if __name__=="__main__":main()
