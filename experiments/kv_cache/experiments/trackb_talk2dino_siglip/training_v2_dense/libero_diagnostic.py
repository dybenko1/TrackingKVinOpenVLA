"""Offline visualization on saved LIBERO RGB frames; no simulator or rollout."""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .extract import load_native_dino, process_image_batch
from ..projection import RegionTextProjection
from ..text_encoder import load_siglip_text_tower

TASK_OBJECTS={
    4:("black bowl","plate"),
    5:("black bowl","plate"),
    7:("black bowl","plate"),
}
EXTRA_PROMPTS=("robot arm","robot gripper","table")

def run(args):
    model,processor,vision,dino,geometry,torch=load_native_dino()
    device=next(dino.parameters()).device
    text_model,tokenizer,text_meta=load_siglip_text_tower(args.siglip_checkpoint,device=str(device))
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    projection=RegionTextProjection().to(device);projection.load_state_dict(ck["model"],strict=True);projection.eval()
    threshold=float(args.threshold)
    if args.threshold_file:
        calibration=json.loads(args.threshold_file.read_text())
        threshold=float(calibration["selected_thresholds"]["v2_raw"])
    rows=[]
    with args.frame_manifest.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["task_id"] not in {"4","5","7"}: continue
            image_id=r["sample_id"]
            image_path=args.visualizations/image_id/"camera_rgb.png"
            if image_path.is_file(): rows.append((r,image_path))
    if not rows: raise RuntimeError("No saved LIBERO RGB frames found in the requested task manifest")
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"visualizations").mkdir(exist_ok=True)
    by_episode={}
    metrics=[]
    for row,path in rows:
        task=int(row["task_id"]); state=int(row["state_id"]); step=int(row["time_step"])
        prompts=[*TASK_OBJECTS.get(task,()),*EXTRA_PROMPTS]
        image=Image.open(path).convert("RGB")
        _,pixels,x,patch=process_image_batch([(0,path)],processor,dino,device,torch)
        tokens=tokenizer([f"a photo of {p}" for p in prompts]).to(device)
        with torch.inference_mode():
            text=text_model(tokens)
            projected=F.normalize(projection.text_projection(text.float()),dim=-1)
            p=F.normalize(patch.float(),dim=-1)[0]
            scores=p@projected.T
            probs=torch.sigmoid(10*(scores-.20)).T.reshape(len(prompts),16,16)
        if probs.shape!=(len(prompts),16,16) or not torch.isfinite(probs).all(): raise RuntimeError("Malformed LIBERO relevance maps")
        masks=(probs>=threshold).cpu().numpy()
        record={"sample_id":row["sample_id"],"task_id":task,"state_id":state,"step":step,
                "task_description":row["task_description"],"prompts":prompts,"threshold":threshold,
                "mean_probability":probs.mean((1,2)).cpu().tolist(),"foreground_patches":masks.sum((1,2)).tolist(),
                "ground_truth_available":False}
        metrics.append(record); key=(task,state);by_episode.setdefault(key,[]).append((step,masks))
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(prompts)+1,figsize=(3.1*(len(prompts)+1),3.3))
        axes[0].imshow(image);axes[0].set_title(f"RGB\nTask {task}, state {state}, step {step}")
        arr=np.asarray(image.resize((224,224),Image.Resampling.BICUBIC))
        for j,prompt in enumerate(prompts):
            axes[j+1].imshow(arr)
            axes[j+1].imshow(probs[j].cpu().numpy(),cmap="magma",alpha=.60,interpolation="nearest",vmin=0,vmax=1)
            axes[j+1].contour(masks[j],levels=[.5],colors=["cyan"],linewidths=1.0,extent=(0,224,224,0))
            axes[j+1].set_title(f"{prompt}\nFG={int(masks[j].sum())}/256")
        for ax in axes: ax.axis("off")
        fig.suptitle("V2 dense SigLIP→DINO maps — offline diagnostic; no simulator labels",fontsize=11)
        fig.tight_layout();fig.savefig(args.output/"visualizations"/f"{row['sample_id']}.png",dpi=135);plt.close(fig)
    temporal=[]
    for (task,state),seq in by_episode.items():
        seq.sort(key=lambda x:x[0])
        for (s0,m0),(s1,m1) in zip(seq,seq[1:]):
            for j,prompt in enumerate([*TASK_OBJECTS.get(task,()),*EXTRA_PROMPTS]):
                a=m0[j];b=m1[j];u=np.logical_or(a,b).sum();i=np.logical_and(a,b).sum()
                temporal.append({"task_id":task,"state_id":state,"prompt":prompt,"source_step":s0,"current_step":s1,
                    "step_gap":s1-s0,"sampled_frame_mask_iou":float(i/u) if u else 1.0,
                    "source_fg_current_bg":int(np.logical_and(a,~b).sum()),"current_fg_source_bg":int(np.logical_and(~a,b).sum())})
    (args.output/"per_frame_metrics.json").write_text(json.dumps({"frames":metrics,"sparse_sample_mask_comparisons":temporal},indent=2)+"\n")
    summary={"status":"complete","frame_count":len(metrics),"tasks":sorted({x["task_id"] for x in metrics}),"prompts":list(EXTRA_PROMPTS),
        "object_prompts_by_task":{str(k):v for k,v in TASK_OBJECTS.items()},"mask_threshold":threshold,
        "no_oracle_labels":True,"no_policy_rollouts":True,"temporal_comparison_caveat":"These are sparse saved snapshots with large step gaps; mask overlap is descriptive, not frame-to-frame stability or ground-truth accuracy.",
        "mean_foreground_patches_by_prompt":{p:float(np.mean([r["foreground_patches"][i] for r in metrics for i,q in enumerate(r["prompts"]) if q==p])) for p in sorted({q for r in metrics for q in r["prompts"]})},
        "sampled_pair_count":len(temporal),"text_encoder":text_meta,"checkpoint":str(args.checkpoint)}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.output/"REPORT.md").write_text("# Offline LIBERO mask feasibility diagnostic\n\n"
        f"Generated {len(metrics)} saved RGB-frame visualizations from tasks 4, 5 and 7. No policy rollouts were run.\n\n"
        "The frozen V2 MLP scores prompts for the manipulated object, plate, robot arm, gripper and table on the existing OpenVLA 224×224 DINO input path. Cyan outlines show a visualization threshold selected on COCO development data. There are no corresponding LIBERO ground-truth masks in this diagnostic; do not interpret these figures as segmentation accuracy or cache-safety evidence.\n\n"
        "Sparse snapshots are compared only descriptively. Their mask IoU is not a frame-to-frame stability measure because snapshots may be dozens of control steps apart.\n\n"
        "See `summary.json`, `per_frame_metrics.json` and `visualizations/`.\n")
    print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--siglip-checkpoint",type=Path,required=True)
    p.add_argument("--frame-manifest",type=Path,required=True);p.add_argument("--visualizations",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--threshold",type=float,default=.5);p.add_argument("--threshold-file",type=Path);run(p.parse_args())
if __name__=="__main__":main()
