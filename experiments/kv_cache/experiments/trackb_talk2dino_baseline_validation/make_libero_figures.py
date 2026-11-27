#!/usr/bin/env python3
"""Create deterministic visual panels for the frozen official LIBERO scorer."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path("/opt/openvla/cache/worktrees/trackb-dense-grounding-v2")
SOURCE=Path("/opt/openvla/cache/datasets/talk2dino_native_e2/scoring")
OUT=ROOT/"experiments/kv_cache/outputs/trackb_talk2dino_baseline_validation/libero_final/figures"

def main():
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.run_coco import load_model
    OUT.mkdir(parents=True,exist_ok=True)
    model,_,_,_=load_model(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    chosen={}
    for path in sorted(SOURCE.glob("*/*.npz")):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(z["metadata"].item())
        if meta["suite"] in {"libero_spatial","libero_object","libero_goal"} and meta["suite"] not in chosen:
            chosen[meta["suite"]]=(path,meta)
    records=[]
    for suite,(path,meta) in chosen.items():
        with np.load(path,allow_pickle=False) as z:
            frame=16 if len(z["rgb"])>16 else len(z["rgb"])//2
            rgb=z["rgb"][frame].copy()
            roles=z["role_gt"][frame].astype(np.float32)
            relevant=z["relevant_gt"][frame].astype(np.float32)
            role_keys=meta["role_keys"]
        specs=meta["prompt_specs"]
        prompts=[s["phrase"] for s in specs]+[meta["instruction"]]
        labels=[s["role"] for s in specs]+["full instruction"]
        txt=model.encode_categories(prompts,"sub_imagenet_template")
        with torch.inference_mode():
            score=model.score_bgr_crop(rgb[...,::-1].copy(),txt,clean=True,pamr=False)
            clean=score["scores"].float().cpu().numpy()
            raw=F.interpolate(score["cosine"].sigmoid().unsqueeze(0).float(),(224,224),mode="bilinear",align_corners=True)[0].cpu().numpy()
        panels=[("RGB",rgb), ("GT task-relevant union",np.repeat(relevant.reshape(16,16),14,0).repeat(14,1))]
        for i,label in enumerate(labels): panels.append((f"{label}: cleaned",clean[i]))
        fig,axes=plt.subplots(1,len(panels),figsize=(3.3*len(panels),3.8),constrained_layout=True)
        for ax,(title,img) in zip(axes,panels):
            if title=="RGB": ax.imshow(img)
            elif title.startswith("GT"): ax.imshow(img,cmap="Greens",vmin=0,vmax=1)
            else: ax.imshow(img,cmap="magma",vmin=0,vmax=1)
            ax.set_title(title,fontsize=9); ax.axis("off")
        fig.suptitle(f"{suite} — {meta['task_id']} — demo {meta['demo']} frame {frame}\n{meta['instruction']}",fontsize=11)
        base=f"{suite}_{path.stem}_f{frame}"
        fig.savefig(OUT/f"{base}_cleaned.png",dpi=150); plt.close(fig)
        fig,axes=plt.subplots(2,len(labels),figsize=(3.0*len(labels),6),constrained_layout=True)
        for i,label in enumerate(labels):
            axes[0,i].imshow(raw[i],cmap="magma",vmin=0,vmax=1); axes[0,i].set_title(f"{label}: raw sigmoid",fontsize=8)
            axes[1,i].imshow(clean[i],cmap="magma",vmin=0,vmax=1); axes[1,i].set_title(f"{label}: cleaned",fontsize=8)
            axes[0,i].axis("off"); axes[1,i].axis("off")
        fig.suptitle(f"Raw vs. attention-cleaned maps — {suite}; fixed reference threshold 0.54",fontsize=11)
        fig.savefig(OUT/f"{base}_raw_vs_clean.png",dpi=150); plt.close(fig)
        records.append({"suite":suite,"source_npz":str(path),"source_frame_index":frame,"segment_metadata":meta,
                        "prompt_order":labels,"threshold":0.54,"figure_cleaned":f"{base}_cleaned.png","figure_raw_vs_clean":f"{base}_raw_vs_clean.png"})
    (OUT/"figure_manifest.json").write_text(json.dumps(records,indent=2)+"\n")
    model.close()
    print(json.dumps({"figures":len(records)*2,"manifest":str(OUT/"figure_manifest.json")},indent=2))
if __name__=="__main__": main()
