#!/usr/bin/env python3
"""Official Talk2DINO scoring on the saved paired E2 LIBERO frames.

This is offline frame scoring, not policy inference or a rollout. It consumes
the authentic RGB/role-mask pairs from the previous E2 cache and checkpoints
each demonstration segment so it can resume safely.
"""
from __future__ import annotations
import csv, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

from experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.run_coco import (
    OUT as COCO_OUT, atomic_json, checkpoint_pair, load_model, sha256,
)

REPO=Path("/opt/openvla")
SOURCE=REPO/"cache/datasets/talk2dino_native_e2/scoring"
OUT=REPO/"cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_baseline_validation/libero_final"
ROLE_TO_GT={"manipulated_object":["manipulated_objects"],
            "destination_or_fixture":["destination_objects","task_relevant_fixtures"],
            "task_fixture":["task_relevant_fixtures"],
            "robot_arm":["robot_arm"],"robot_gripper":["robot_gripper"]}
THRESHOLD=.54

def pooled(pred, y, valid):
    pred=np.asarray(pred,dtype=bool); y=np.asarray(y,dtype=np.float64); v=np.asarray(valid,dtype=bool)
    y=np.clip(y,0,1)*v
    tp=float((pred*y).sum()); fp=float((pred*(1-y)*v).sum()); fn=float(((~pred)*y).sum())
    tn=float(((~pred)*(1-y)*v).sum())
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "precision":tp/max(tp+fp,1e-12),"recall":tp/max(tp+fn,1e-12),
            "iou":tp/max(tp+fp+fn,1e-12),"valid_patch_equivalents":float(v.sum())}

def run():
    import torch
    import torch.nn.functional as F
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"segments").mkdir(parents=True,exist_ok=True)
    files=sorted(SOURCE.glob("*/*.npz"))
    if len(files)!=36: raise RuntimeError(f"Expected 36 paired E2 segments, found {len(files)} at {SOURCE}")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    atomic_json(OUT/"RUN_STATUS.json",{"status":"running","stage":"load_model","segments":0,"frames":0,"total_segments":len(files),"threshold":THRESHOLD})
    model,names,_,identity=load_model(device)
    model.model.eval(); model.model.model.eval() if hasattr(model.model.model,"eval") else None
    role_totals={k:{m:0.0 for m in ("tp","fp","fn","tn")} for k in list(ROLE_TO_GT)+["task_relevant_foreground"]}
    rows=[]; frames=0; segment_metrics=[]
    for si,path in enumerate(files,1):
        target=OUT/"segments"/path.relative_to(SOURCE).with_suffix(".npz")
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            with np.load(target,allow_pickle=False) as z:
                if int(z["verified"])==1:
                    rows.extend(json.loads(str(z["metrics_json"].item())))
                    frames+=int(z["frame_count"]); continue
        with np.load(path,allow_pickle=False) as z:
            rgb=z["rgb"].copy(); gt=z["role_gt"].astype(np.float32); relevant_gt=z["relevant_gt"].astype(np.float32); valid=z["valid"].astype(np.float32)
            meta=json.loads(str(z["metadata"].item())); role_keys=meta["role_keys"]
            specs=meta["prompt_specs"]
        phrases=[s["phrase"] for s in specs]+[meta["instruction"]]
        role_names=[s["role"] for s in specs]+["task_relevant_foreground"]
        text=model.encode_categories(phrases,"sub_imagenet_template")
        seg_rows=[]
        with torch.inference_mode():
            for fi in range(len(rgb)):
                bgr=rgb[fi,...,::-1].copy()
                scores=model.score_bgr_crop(bgr,text,clean=True,pamr=False)
                # DINOText resizes these 224-square stored frames to 448
                # internally, so its ViT/14 maps are 32x32. E2 simulator
                # labels are aligned to the 224/OpenVLA 16x16 grid. Average
                # the 2x2 high-resolution score cells into each E2 patch
                # before applying the unchanged 0.54 reference threshold.
                clean=F.interpolate(scores["scores"].unsqueeze(0).float(),(16,16),mode="area")[0].flatten(1).cpu().numpy()
                raw=F.interpolate(scores["cosine"].sigmoid().unsqueeze(0).float(),(16,16),mode="area")[0].flatten(1).cpu().numpy()
                clean_pred=clean>=THRESHOLD; raw_pred=raw>=THRESHOLD
                frame_meta={"suite":meta["suite"],"task_id":meta["task_id"],"demo":meta["demo"],"frame_index":fi,
                            "instruction":meta["instruction"],"segment":path.stem}
                for qi,rn in enumerate(role_names):
                    names_gt=ROLE_TO_GT.get(rn,["task_relevant_objects"] if rn=="task_relevant_foreground" else [])
                    if not names_gt:
                        # A contextual phrase may not have an individually
                        # labeled instance in the preserved E2 ground truth.
                        frame_meta[f"{rn}_ground_truth_available"]=False
                        continue
                    if rn=="task_relevant_foreground":
                        # E2's union label is stored separately; the
                        # task_relevant_objects channel in role_gt is
                        # intentionally empty in this dataset schema.
                        target_mask=relevant_gt[fi].copy()
                    else:
                        target_mask=np.zeros(256,np.float32)
                        for gname in names_gt:
                            if gname in role_keys: target_mask=np.maximum(target_mask,gt[fi,role_keys.index(gname)])
                    if float(target_mask.sum()) <= 0:
                        frame_meta[f"{rn}_ground_truth_available"]=False
                        continue
                    frame_meta[f"{rn}_ground_truth_available"]=True
                    v=valid[fi]>0
                    for method,pred in (("raw",raw_pred[qi]),("clean",clean_pred[qi])):
                        m=pooled(pred,target_mask,v)
                        role_totals.setdefault(rn,{key:0.0 for key in ("tp","fp","fn","tn")})
                        for key in ("tp","fp","fn","tn"): role_totals[rn][key]+=m[key]
                        frame_meta[f"{rn}_{method}"]=m
                frame_meta["raw_scores_range"]=[float(raw.min()),float(raw.max())]
                frame_meta["clean_scores_range"]=[float(clean.min()),float(clean.max())]
                seg_rows.append(frame_meta)
        checkpoint_pair(target,{"metrics_json":np.asarray(json.dumps(seg_rows,separators=(",",":"))),
                               "frame_count":np.asarray(len(seg_rows),np.int32),"verified":np.asarray(1,np.uint8)})
        rows.extend(seg_rows); frames+=len(seg_rows)
        atomic_json(OUT/"RUN_STATUS.json",{"status":"running","stage":"score_libero_frames","segments":si,"frames":frames,
                    "total_segments":len(files),"total_frames":1152,"last_segment":path.name,"threshold":THRESHOLD,
                    "updated_unix":time.time()})
    for f in files:
        target=OUT/"segments"/f.relative_to(SOURCE).with_suffix(".npz")
        if not target.exists(): continue
        with np.load(target,allow_pickle=False) as z:
            for r in json.loads(str(z["metrics_json"].item())):
                # Rebuild aggregate totals from idempotent saved rows.
                for role in list(ROLE_TO_GT)+["task_relevant_foreground"]:
                    for method in ("raw","clean"):
                        mm=r.get(f"{role}_{method}",{})
                        for key in ("tp","fp","fn","tn"):
                            # rows are counted once below in fresh implementation; avoid double use here
                            pass
    aggregate={}
    for role,counts in role_totals.items():
        # For resumed runs, totals are recomputed from all verified segment rows.
        counts={k:0.0 for k in ("tp","fp","fn","tn")}
        for row in rows:
            for method in ("raw","clean"):
                m=row.get(f"{role}_{method}")
                if not m: continue
                for k in counts: counts[k]+=m[k]
        aggregate[role]={method:{**{k:counts[k] for k in counts},
            "precision":counts["tp"]/max(counts["tp"]+counts["fp"],1e-12),
            "recall":counts["tp"]/max(counts["tp"]+counts["fn"],1e-12),
            "iou":counts["tp"]/max(counts["tp"]+counts["fp"]+counts["fn"],1e-12)} for method in ("raw","clean")}
    # Rebuild correctly separated raw/clean aggregations.
    aggregate={}
    for role in list(ROLE_TO_GT)+["task_relevant_foreground"]:
        aggregate[role]={}
        for method in ("raw","clean"):
            cc={k:sum(float(r.get(f"{role}_{method}",{}).get(k,0)) for r in rows) for k in ("tp","fp","fn","tn")}
            aggregate[role][method]={**cc,"precision":cc["tp"]/max(cc["tp"]+cc["fp"],1e-12),
               "recall":cc["tp"]/max(cc["tp"]+cc["fn"],1e-12),"iou":cc["tp"]/max(cc["tp"]+cc["fp"]+cc["fn"],1e-12)}
    fields=["suite","task_id","demo","frame_index","instruction"]
    with (OUT/"per_frame_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields+sorted({k for r in rows for k in r if k not in fields}));w.writeheader();w.writerows(rows)
    atomic_json(OUT/"summary.json",{"frames":len(rows),"segments":len(files),"resolution":"224 RGB paired frames; upstream DINOText resizes square to 448 internally",
       "threshold":THRESHOLD,"threshold_is_official_COCO_setting_not_calibrated_for_LIBERO":True,
       "metrics_are_fractional_patch_occupancy_pooled_not_instance_iou":True,"metrics":aggregate,
       "official_model_identity":identity})
    model.close()
    atomic_json(OUT/"RUN_STATUS.json",{"status":"complete","stage":"complete","segments":len(files),"frames":len(rows),"updated_unix":time.time()})

if __name__=="__main__":
    try: run()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/"RUN_STATUS.json",{"status":"failed","stage":"exception","error":repr(exc),"traceback":traceback.format_exc(),"updated_unix":time.time()})
        raise
