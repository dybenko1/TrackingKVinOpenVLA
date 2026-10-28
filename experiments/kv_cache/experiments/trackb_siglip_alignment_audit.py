"""Offline audit of paired SigLIP text/vision alignment using saved Stage 4A data.

This audit does not fit a model or modify Workstream A. It reuses saved Track B
patch features, Stage 4A affinity matrices, paired-checkpoint pooled scores, and
tokenization metadata. The optional CPU text pass only compares inclusion of
special/padding states with the original valid-token aggregation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"trackb_siglip_alignment_audit"
PHASE1=ROOT/"outputs"/"stage4a_siglip_foreground"/"phase1"
TRACK=ROOT/"outputs"/"track_b_segmentation"


def write_csv(path:Path, rows:list[dict[str,Any]]):
    path.parent.mkdir(parents=True,exist_ok=True)
    cols=list(rows[0]) if rows else []
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)


def auc(score,labels):
    s=np.asarray(score,dtype=np.float64).reshape(-1);y=np.asarray(labels,bool).reshape(-1)
    if y.sum()==0 or y.sum()==len(y):return None
    order=np.argsort(s,kind="mergesort");ranks=np.empty(len(s),np.float64);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and s[order[j]]==s[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2;i=j
    p=int(y.sum());return float((ranks[y].sum()-p*(p+1)/2)/(p*(len(y)-p)))


def _binary(scores,labels):
    # Otsu is used only as a descriptive post-hoc comparison, never in Workstream A.
    from experiments.kv_cache.experiments.trackb_unmatch_cache_safety import otsu_high_mask
    pred,meta=otsu_high_mask(scores);y=np.asarray(labels,bool)
    tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
    return {"otsu_threshold":meta.get("threshold"),"otsu_valid":meta["valid"],
            "precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),
            "iou":tp/max(tp+fp+fn,1),"predicted_fg_tokens":int(pred.sum())}


def _matrix(path,task_text,variant):
    import re
    key=re.sub(r"[^a-zA-Z0-9]+","_",task_text.lower()).strip("_")+"__"+variant
    with np.load(path) as z:return np.asarray(z[key],dtype=np.float32)


def _saved_audit():
    diag=json.loads((PHASE1/"matching_diagnostics.json").read_text())
    manifest=json.loads((TRACK/"dataset_manifest.json").read_text())
    track={r["sample_id"]:r for r in manifest["samples"]}
    token_rows=[];feature_rows=[];local_rows=[];phrase_rows=[]
    # Use all saved diagnostic frames and the exact existing patch matrices.
    for s in diag["samples"]:
        sid=s["sample_id"];image=TRACK/s["image_path"];oracle=np.load(TRACK/s["oracle_token_mask_path"]).astype(bool).reshape(-1)
        npz=PHASE1/s["affinity_npz"]
        pen=_matrix(npz,s["task_description"],"penultimate")
        fin=_matrix(npz,s["task_description"],"final")
        paired=_matrix(npz,s["task_description"],"paired_pretrained_final")
        for variant,m in [("OpenVLA fine-tuned penultimate",pen),("OpenVLA fine-tuned final",fin),("paired pretrained final",paired)]:
            scores=m.max(axis=0)
            metrics=_binary(scores,oracle)
            local_rows.append({"sample_id":sid,"task_id":s["task_id"],"feature_variant":variant,
                               "task_instruction_auroc":auc(scores,oracle),**metrics,
                               "matched_mismatched_map_corr":None})
        # Matched/mismatched spatial-map correlation uses the same variant and same image.
        import re
        with np.load(npz) as z:
            for variant in ("penultimate","final","paired_pretrained_final"):
                matched=np.asarray(z["black_bowl__"+variant],np.float64).max(axis=0)
                mismatched=np.asarray(z["red_cup__"+variant],np.float64).max(axis=0)
                corr=float(np.corrcoef(matched,mismatched)[0,1]) if np.std(matched)>0 and np.std(mismatched)>0 else None
                rowname={"penultimate":"OpenVLA fine-tuned penultimate","final":"OpenVLA fine-tuned final","paired_pretrained_final":"paired pretrained final"}[variant]
                for r in local_rows:
                    if r["sample_id"]==sid and r["feature_variant"]==rowname:r["matched_mismatched_map_corr"]=corr
        per_path=PHASE1/"affinity_matrices"/(sid+"_per_token.npz")
        keyprefix="penultimate_"
        # The original diagnostics saved token-wise arrays only for a subset of
        # frames. Preserve that sampling; do not synthesize missing evidence.
        if not per_path.exists():
            continue
        with np.load(per_path) as z:
            items=[]
            for k in z.files:
                if k.startswith(keyprefix):
                    tail=k[len(keyprefix):]
                    # saved field is index + formatted T5 token + 256 cosine values
                    idx_text,label=tail.split("_",1)
                    items.append((int(idx_text),label,np.asarray(z[k],np.float32)))
        items.sort()
        if items:
            stack=np.stack([x[2] for x in items]); winners=stack.argmax(axis=0)
            for i,(idx,label,values) in enumerate(items):
                token_rows.append({"sample_id":sid,"task_id":s["task_id"],"token_index":idx,"token":label,
                                   "winning_patch_count":int((winners==i).sum()),
                                   "winning_patch_fraction":float((winners==i).mean()),
                                   "mean_cosine_over_patches":float(values.mean()),
                                   "max_cosine_over_patches":float(values.max()),
                                   "max_patch_index":int(values.argmax())})
                if s["task_id"]==4:
                    phrase_rows.append({"sample_id":sid,"time_step":s["time_step"],"token_index":idx,
                                        "token":label,"r_j_max_visual_cosine":float(values.max())})
    # Paired model's ordinary pooled scores were computed with its intended MAP
    # image pooling, last-token text pooling, learned text projection and cosine.
    global_rows=[]
    for r in diag["global_pooled_scores"]:
        global_rows.append({"sample_id":r["sample_id"],"prompt":r["prompt"],
                            "paired_checkpoint_pooled_cosine":r["pooled_cosine"],"siglip_logit":r["siglip_logit"]})
    global_diffs=diag["matched_mismatched_pooled_score_differences"]
    return diag,local_rows,token_rows,phrase_rows,global_rows,global_diffs


def _text_special_control():
    """Recompute local scores with all 16 text states vs valid content tokens.

    Visual features come from saved native Track B `siglip` arrays. The pretrained
    paired text tower is loaded on CPU; no new vision pass or GPU workload is used.
    """
    import torch
    from experiments.kv_cache.experiments.stage4a_unmatch_siglip_diagnostics import _text_tokens,_tokenize
    from experiments.kv_cache.experiments.trackb_unmatch_cache_safety import _load_text_tower
    text_model,tokenizer=_load_text_tower(torch.device("cpu"))
    diag=json.loads((PHASE1/"matching_diagnostics.json").read_text())
    rows=[];sample_rows={x["sample_id"]:x for x in diag["samples"]}
    for sid,s in sample_rows.items():
        features=np.load(TRACK/"data"/"samples"/sid/"features.npz")["siglip"].astype(np.float32)
        image=torch.from_numpy(features).unsqueeze(0)
        ids,valid,tokens,truncated=_tokenize(tokenizer,s["task_description"],torch.device("cpu"))
        with torch.inference_mode(): _,states=_text_tokens(type("W",(),{"text":text_model})(),ids)
        t=F_normalize(states[0,valid[0]].float());v=F_normalize(image[0].float())
        valid_matrix=(t@v.T).numpy();valid_scores=valid_matrix.max(axis=0)
        all_matrix=(F_normalize(states[0].float())@v.T).numpy();all_scores=all_matrix.max(axis=0)
        npz=PHASE1/s["affinity_npz"]
        import re
        key=re.sub(r"[^a-zA-Z0-9]+","_",s["task_description"].lower()).strip("_")+"__penultimate"
        saved=_matrix(npz,s["task_description"],"penultimate")
        oracle=np.load(TRACK/s["oracle_token_mask_path"]).astype(bool).reshape(-1)
        rows.append({"sample_id":sid,"task_id":s["task_id"],"valid_text_tokens":int(valid.sum()),
                     "context_positions":int(ids.shape[1]),"truncated":bool(truncated),"tokens":" | ".join(tokens),
                     "special_ids_in_input":int(sum(int(x) in tokenizer.all_special_ids for x in ids[0].tolist())),
                     "max_abs_reproduction_vs_saved_valid_matrix":float(np.abs(valid_matrix-saved).max()),
                     "mean_abs_reproduction_vs_saved_valid_matrix":float(np.abs(valid_matrix-saved).mean()),
                     "valid_only_map_auroc":auc(valid_scores,oracle),"all_positions_map_auroc":auc(all_scores,oracle),
                     "valid_only_vs_all_map_corr":float(np.corrcoef(valid_scores,all_scores)[0,1]),
                     "valid_only_winner_map_entropy":_winner_entropy(valid_matrix),
                     "all_positions_winner_map_entropy":_winner_entropy(all_matrix),
                     "all_positions_vs_valid_score_mean_abs":float(np.abs(all_scores-valid_scores).mean())})
    return rows


def F_normalize(x):
    import torch.nn.functional as F
    return F.normalize(x,dim=-1)


def _winner_entropy(matrix):
    winners=np.asarray(matrix).argmax(axis=0);counts=np.bincount(winners,minlength=matrix.shape[0]).astype(np.float64);p=counts[counts>0]/counts.sum()
    return float(-(p*np.log2(p)).sum())


def _plots(diag,local_rows,token_rows,phrase_rows,global_diffs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True)
    # Overall image/text matched-vs-mismatched paired pooled scores.
    prompts={}
    for x in diag["global_pooled_scores"]:prompts.setdefault(x["sample_id"],{})[x["prompt"]]=x["pooled_cosine"]
    diff=np.asarray([prompts[x["sample_id"]].get(x["task_description"],np.nan)-prompts[x["sample_id"]].get("red cup",np.nan) for x in diag["samples"]])
    fig,ax=plt.subplots(figsize=(8,4));ax.axhline(0,color="black",lw=1);ax.bar(np.arange(len(diff)),diff,color=np.where(diff>0,"#2b8cbe","#d7301f"));ax.set_xticks(np.arange(len(diff)),[x["sample_id"].replace("task","T").replace("_state0_step","/") for x in diag["samples"]],rotation=45,ha="right");ax.set_ylabel("paired pooled cosine: task description − red cup");ax.set_title("Original paired SigLIP global image–text score control");fig.tight_layout();fig.savefig(OUT/"paired_global_matched_mismatch.png",dpi=160);plt.close(fig)
    variants=list(dict.fromkeys(r["feature_variant"] for r in local_rows));means=[np.mean([r["task_instruction_auroc"] for r in local_rows if r["feature_variant"]==v]) for v in variants]
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(variants,means,color=["#1b9e77","#d95f02","#7570b3"]);ax.axhline(.5,color="black",ls="--");ax.set_ylabel("broad oracle foreground AUROC");ax.set_title("Same task text, different saved SigLIP patch features");ax.tick_params(axis="x",rotation=15);fig.tight_layout();fig.savefig(OUT/"local_feature_variant_auc.png",dpi=160);plt.close(fig)
    # Same-frame matched/mismatched local maps, with a shared color scale.
    sample=next(x for x in diag["samples"] if x["sample_id"]=="task4_state0_step0000")
    rgb=np.asarray(Image.open(TRACK/sample["image_path"]).convert("RGB"))
    npz=PHASE1/sample["affinity_npz"]
    variants=("penultimate","final","paired_pretrained_final")
    labels=("OpenVLA penultimate","OpenVLA final","Paired pretrained final")
    maps=[]
    for variant in variants:
        for phrase in ("black bowl","red cup"):
            maps.append(_matrix(npz,phrase,variant).max(axis=0).reshape(16,16))
    lo=min(float(x.min()) for x in maps);hi=max(float(x.max()) for x in maps)
    fig,axes=plt.subplots(2,4,figsize=(15,7),gridspec_kw={"width_ratios":[1,1,1,1]})
    axes[0,0].imshow(rgb);axes[0,0].set_title("RGB: task 4/state 0")
    oracle=np.load(TRACK/sample["oracle_token_mask_path"]).astype(bool).reshape(16,16)
    axes[1,0].imshow(oracle,cmap="gray",vmin=0,vmax=1,interpolation="nearest");axes[1,0].set_title("Oracle FG (post-hoc)")
    for col,(variant,label) in enumerate(zip(variants,labels),start=1):
        for row,phrase in enumerate(("black bowl","red cup")):
            m=_matrix(npz,phrase,variant).max(axis=0).reshape(16,16)
            axes[row,col].imshow(rgb)
            axes[row,col].imshow(m,cmap="magma",alpha=.62,interpolation="bilinear",vmin=lo,vmax=hi)
            axes[row,col].set_title(f"{label}\n{phrase} · max valid-token cosine")
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("Local patch affinity: matched vs mismatched phrase (shared scale; not calibrated probabilities)")
    fig.tight_layout();fig.savefig(OUT/"local_matched_mismatched_examples.png",dpi=170);plt.close(fig)
    # Temporal inverse-reduction evidence for Task 4 state 0.
    by={}
    for r in phrase_rows:by.setdefault((r["token_index"],r["token"]),[]).append(r)
    fig,ax=plt.subplots(figsize=(10,5))
    for (i,tok),rs in by.items():
        rs=sorted(rs,key=lambda x:x["time_step"])
        if len(rs)>=2:ax.plot([r["time_step"] for r in rs],[r["r_j_max_visual_cosine"] for r in rs],marker="o",label=f"{i}:{tok}")
    ax.set_xlabel("saved task-4/state-0 trajectory step");ax.set_ylabel("r_j(t) = max visual patch cosine for token j");ax.set_title("Exploratory token evidence across the two frames with saved per-token arrays");ax.legend(ncol=3,fontsize=7);fig.tight_layout();fig.savefig(OUT/"task4_token_evidence_over_time.png",dpi=160);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    diag,local,token,phrase,global_rows,global_diffs=_saved_audit()
    write_csv(OUT/"paired_global_scores.csv",global_rows)
    write_csv(OUT/"local_feature_comparison.csv",local)
    write_csv(OUT/"text_token_dominance.csv",token)
    write_csv(OUT/"task4_token_evidence_over_time.csv",phrase)
    special=_text_special_control();write_csv(OUT/"valid_vs_special_pad_control.csv",special)
    _plots(diag,local,token,phrase,global_diffs)
    fd=diag["feature_variants"]["paired_vs_finetuned_openvla_visual_weight_comparison"]
    instructions=[x for x in diag["matched_mismatched_pooled_score_differences"]]
    summary={"experiment":"Track B SigLIP vision/text alignment audit","status":"complete",
             "paired_global_image_text":{"model":"timm/ViT-SO400M-14-SigLIP paired pretrained weights","matched_instruction_minus_red_cup_mean":float(np.mean([x["task_instruction_minus_red_cup_cosine"] for x in instructions])),"black_bowl_minus_red_cup_mean":float(np.mean([x["black_bowl_minus_red_cup_cosine"] for x in instructions])),"plate_minus_red_cup_mean":float(np.mean([x["plate_minus_red_cup_cosine"] for x in instructions]))},
             "vision_weight_drift":fd,
             "local_instruction_auc_mean_by_variant":{v:float(np.mean([x["task_instruction_auroc"] for x in local if x["feature_variant"]==v])) for v in dict.fromkeys(x["feature_variant"] for x in local)},
             "mean_matched_vs_mismatched_local_map_corr_by_variant":{v:float(np.mean([x["matched_mismatched_map_corr"] for x in local if x["feature_variant"]==v])) for v in dict.fromkeys(x["feature_variant"] for x in local)},
             "text_special_pad_control_mean_valid_auroc":float(np.mean([x["valid_only_map_auroc"] for x in special])),
             "text_special_pad_control_mean_all_auroc":float(np.mean([x["all_positions_map_auroc"] for x in special])),
             "special_pad_control_mean_map_corr":float(np.mean([x["valid_only_vs_all_map_corr"] for x in special])),
             "data_provenance":"saved Stage 4A NPZ/JSON and Track B saved OpenVLA SigLIP features; no model fitting or policy rollout"}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    config={"source_checkpoint":diag["checkpoint"],"openvla_snapshot":diag["feature_variants"]["A_penultimate"],
            "text_control":"all 16 encoder states (including special/padding EOS positions) vs original valid non-special states; CPU text tower pass",
            "sample_ids":[s["sample_id"] for s in diag["samples"]],"oracle_use":"post-hoc metrics only",
            "text_encoder_implementation":"open_clip_torch 2.31.0 loaded from isolated /tmp/stage4a_openclip_pkg; no main inference environment change",
            "reproduction_command":"python -u experiments/kv_cache/experiments/trackb_siglip_alignment_audit.py (inside openvla-cached:latest)",
            "audit_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "no_additional_vision_forward":True,"no_rollouts":True,"no_training":True}
    (OUT/"configuration.json").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n")
    report=f'''# Track B — SigLIP vision/text alignment audit\n\n## Evidence sources\n\nThis is a separate, offline diagnostic. It reuses the original Stage 4A saved affinity matrices and Track B native OpenVLA patch features. No policy inference, rollout, training, or mask selection occurred.\n\n## Paired pretrained image/text path\n\nThe saved original SigLIP dual-encoder run used the checkpoint's paired visual tower, MAP image pooling, final text-token pooling, learned text projection, and cosine scoring. Across nine same-image task-description vs `red cup` comparisons, mean task-description advantage was {summary['paired_global_image_text']['matched_instruction_minus_red_cup_mean']:.4f}; `black bowl` mean advantage was {summary['paired_global_image_text']['black_bowl_minus_red_cup_mean']:.4f}. This validates that the paired global path responds plausibly on average, not that local patches ground language.\n\n## Fine-tuned OpenVLA visual tower\n\nOnly {fd.get('same_after_bf16_cast_tensor_count')}/{fd.get('paired_tensors')} visual tensors matched the paired pretrained weights after BF16 casting; aggregate relative L2 was {fd.get('relative_l2'):.4f}. The saved checkpoint’s OpenVLA native penultimate SigLIP features are therefore materially drifted from the original image/text pairing. A global image/text score for those fine-tuned patches is not directly comparable: the native path exposes penultimate patch states and does not apply the original trained MAP pooler/paired visual weights.\n\n## Local patch affinity and token dominance\n\nMean broad-oracle instruction AUROC by saved patch variant: {json.dumps(summary['local_instruction_auc_mean_by_variant'])}. Same-image local maps for “black bowl” versus mismatched “red cup” remain correlated (means: {json.dumps(summary['mean_matched_vs_mismatched_local_map_corr_by_variant'])}). These broad oracle metrics are descriptive; the simulator labels robot, objects, and fixtures, not phrase-specific object identity. Per-token winners and cosine extrema are in `text_token_dominance.csv`.\n\nThe valid-token matrix was rederived from saved features and the paired text tower. Including all 16 text positions, including EOS/padding, changed maps; mean valid-only/all-position correlation was {summary['special_pad_control_mean_map_corr']:.3f}. The corresponding mean broad-FG AUROC was {summary['text_special_pad_control_mean_valid_auroc']:.3f} valid-only and {summary['text_special_pad_control_mean_all_auroc']:.3f} with special/padding positions included. The method correctly excludes special/padding states in its primary affinity.\n\n## Penultimate/final features and temporal evidence\n\n`local_feature_comparison.csv` compares the saved penultimate, final, and paired-pretrained final feature maps on the same images. Existing Stage 4A evidence found final-block OpenVLA features reduced the full-instruction broad-FG AUROC relative to penultimate features. Per-token maxima are available for two task-4/state-0 frames only; they are plotted in `task4_token_evidence_over_time.csv` and its figure as exploratory evidence, not evidence of task phase.\n\n## Conclusion\n\nThe original paired global encoder works plausibly on average, while the local max-cosine method remains weak for patch grounding. The fine-tuned OpenVLA visual tower has substantial weight drift relative to the external text tower, supporting alignment drift as a contributor. Generic/action-token dominance and high matched/mismatched local-map correlation support a separate local-aggregation failure. Preprocessing and dimensions were validated in the original saved report; these data do not isolate all possible alignment causes. No new segmentation method is recommended from this audit alone.\n'''
    (OUT/"REPORT.md").write_text(report)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
