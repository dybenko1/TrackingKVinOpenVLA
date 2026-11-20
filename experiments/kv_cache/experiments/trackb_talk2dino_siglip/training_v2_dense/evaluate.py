"""Development calibration and one-pass matched V1/V2 COCO val2014 evaluation."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ..projection import RegionTextProjection
from .masks import decode_segmentation,apply_mask_geometry,fractional_grid

THRESHOLDS=np.arange(.10,.901,.05,dtype=np.float32)
METHODS=("v1_raw_pamr","v2_raw","v2_pamr")

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n");os.replace(tmp,path)
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

def image_rgb224(image,geometry):
    arr=np.asarray(image.convert("RGB"));h,w=arr.shape[:2];size=geometry["resize_size"]
    if size is None: rh,rw=h,w
    elif isinstance(size,int):
        if w<h:rh,rw=int(size*h/w),size
        else:rh,rw=size,int(size*w/h)
    else:rh,rw=map(int,size)
    im=Image.fromarray(arr).resize((rw,rh),Image.Resampling.BICUBIC);ch,cw=geometry["crop_size"]
    top=round((rh-ch)/2);left=round((rw-cw)/2)
    return im.crop((left,top,left+cw,top+ch))

def category_targets(meta,annotations,category_order,geometry):
    h,w=int(meta["height"]),int(meta["width"]); unions={};ignore=np.zeros((h,w),np.uint8)
    for a in annotations:
        m=decode_segmentation(a.get("segmentation",[]),h,w)
        if a.get("iscrowd",0):ignore|=m
        else:
            cid=int(a["category_id"])
            if cid in category_order:
                if cid not in unions:unions[cid]=np.zeros((h,w),np.uint8)
                unions[cid]|=m
    valid224=apply_mask_geometry(1-ignore,geometry)>0
    valid16=fractional_grid(valid224.astype(np.uint8))
    out={}
    for cid,m in unions.items():
        m224=apply_mask_geometry(m,geometry)
        if m224.any():
            gt16=fractional_grid(m224)
            if float((gt16*valid16).sum())>0:
                out[cid]=(m224.astype(bool),gt16,valid224,valid16)
    return out

def confusion(score,target,valid,threshold):
    # Grid scores are 16x16 while fractional occupancy targets are stored as
    # 256 cells; flatten each in row-major order before elementwise metrics.
    p=(np.asarray(score).reshape(-1)>=threshold)
    v=np.asarray(valid,dtype=np.float64).reshape(-1)
    y=np.asarray(target,dtype=np.float64).reshape(-1)
    if p.shape!=v.shape or p.shape!=y.shape:
        raise ValueError(f"score/target/valid size mismatch: {p.shape}, {y.shape}, {v.shape}")
    tp=float((p*y*v).sum());fp=float((p*(1-y)*v).sum());fn=float(((~p)*y*v).sum());tn=float(((~p)*(1-y)*v).sum())
    return tp,fp,fn,tn

def auc_binary(scores, target, valid=None):
    x=np.asarray(scores,dtype=np.float64).ravel();y=np.asarray(target).ravel()>0
    if valid is not None:
        keep=np.asarray(valid).ravel()>0;x=x[keep];y=y[keep]
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:return None
    order=np.argsort(x,kind="mergesort");sx=x[order];ranks=np.empty(len(x),np.float64);i=0
    while i<len(x):
        j=i+1
        while j<len(x) and sx[j]==sx[i]:j+=1
        ranks[order[i:j]]=(i+1+j)/2;i=j
    return float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg))

def pointing_accuracy(score,target,valid):
    s=np.asarray(score,dtype=np.float64).reshape(-1)
    y=np.asarray(target).reshape(-1)>0
    v=np.asarray(valid).reshape(-1)>0
    if s.shape!=y.shape or s.shape!=v.shape:
        raise ValueError(f"pointing score/target/valid size mismatch: {s.shape}, {y.shape}, {v.shape}")
    if not v.any():return None
    best=int(np.argmax(np.where(v,s,-np.inf)))
    return float(y[best])

def thresholds_for_method(method, selected_thresholds=None):
    """Thresholds to materialize, keeping the V1 reference fixed."""
    if method=="v1_raw_pamr":return [.55]
    if selected_thresholds is None:return THRESHOLDS
    return [float(selected_thresholds[method])]

def load_pamr(device):
    from ..phase3_evaluate import load_pamr as load
    return load(device)

def _apply_pamr_batch(pamr, rgb224, maps):
    refined=[]
    for mask in maps:
        value=pamr(rgb224.unsqueeze(0),mask.unsqueeze(0))
        if value.ndim==4: value=value[0]
        if value.ndim==2: value=value.unsqueeze(0)
        if value.ndim!=3 or value.shape[0]!=1:
            raise RuntimeError(f"PAMR returned unexpected single-map shape {tuple(value.shape)}")
        refined.append(value)
    return torch.stack(refined,dim=0)

@torch.inference_mode()
def compute_maps(v1,v2,patch_v1,patch_v2,cat_emb_v1,cat_emb_v2,rgb224,pamr):
    txt1=F.normalize(v1.text_projection(cat_emb_v1.float()),dim=-1)
    txt2=F.normalize(v2.text_projection(cat_emb_v2.float()),dim=-1)
    x1=F.normalize(patch_v1.float(),dim=-1);x2=F.normalize(patch_v2.float(),dim=-1)
    s1=x1@txt1.T;s2=x2@txt2.T
    raw1=torch.sigmoid(s1.T.reshape(-1,1,16,16))
    raw2=torch.sigmoid(10*(s2.T.reshape(-1,1,16,16)-.20))
    up1=F.interpolate(raw1,(224,224),mode="bilinear",align_corners=True)
    up2=F.interpolate(raw2,(224,224),mode="bilinear",align_corners=True)
    p1=_apply_pamr_batch(pamr,rgb224,up1)
    p2=_apply_pamr_batch(pamr,rgb224,up2)
    return {"v1_raw_pamr":p1[:,0],"v2_raw":up2[:,0],"v2_pamr":p2[:,0]}, {"v1_raw_pamr":raw1[:,0],"v2_raw":raw2[:,0],"v2_pamr":F.interpolate(p2,size=(16,16),mode="nearest")[:,0]}

def image_records(ids,split):
    return [int(i) for i in split[split]["image_ids"]]

def process_subset(name,ids,*,split,root,cache,inst_json,catinfo,geometry,models,category_text_v1,category_text_v2,pamr,device,out,thresholds=None,checkpoint_hashes=None,make_gallery=False):
    captions=json.loads((root/"annotations2014/captions_val2014.json").read_text());inst=json.loads(inst_json.read_text())
    info={int(x["id"]):x for x in inst["images"]};by_image=defaultdict(list)
    for a in inst["annotations"]:by_image[int(a["image_id"])].append(a)
    cats={int(c["id"]):i for i,c in enumerate(catinfo)}
    ids_cache=np.load(cache/"image_ids_val2014.npy",mmap_mode="r");row_by_id={int(i):r for r,i in enumerate(ids_cache)}
    pen=np.load(cache/"patch_features_val2014_fp16.npy",mmap_mode="r");fin=np.load(cache/"patch_features_val2014_final_norm_fp16.npy",mmap_mode="r")
    saved=jsonl=out/f"{name}_per_image.jsonl";done=set()
    if saved.exists():
        for line in saved.read_text().splitlines():
            try:
                d=json.loads(line)
                if d.get("manifest_sha256")==sha256(out/"validation_manifest.json"):done.add(int(d["image_id"]))
            except Exception:pass
    mode="a" if done else "w"
    gallery_dir=out/"qualitative_results";gallery_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n,iid in enumerate(ids,1):
        if iid in done:continue
        if iid not in row_by_id or iid not in info:raise RuntimeError(f"Missing frozen evaluation image {iid}")
        meta=info[iid]; rgb=Image.open(root/"val2014"/meta["file_name"]).convert("RGB"); rgb224=image_rgb224(rgb,geometry)
        targets=category_targets(meta,by_image[iid],cats,geometry); present=[c for c in targets if c in cats]
        if not present:continue
        row=row_by_id[iid];p1=torch.from_numpy(np.array(fin[row],dtype=np.float32,copy=True)).to(device);p2=torch.from_numpy(np.array(pen[row],dtype=np.float32,copy=True)).to(device)
        category_rows=[cats[c] for c in present]
        txt1=torch.from_numpy(np.array(category_text_v1[category_rows],dtype=np.float32,copy=True)).to(device)
        txt2=torch.from_numpy(np.array(category_text_v2[category_rows],dtype=np.float32,copy=True)).to(device)
        rgb_tensor=torch.from_numpy(np.asarray(rgb224).copy()).permute(2,0,1).float().to(device)
        all224,all16=compute_maps(models["v1"],models["v2"],p1,p2,txt1,txt2,rgb_tensor,pamr)
        rec={"image_id":iid,"feature_row":row,"categories":{},"manifest_sha256":sha256(out/"validation_manifest.json")}
        for j,cid in enumerate(present):
            gt224,gt16,valid224,valid16=targets[cid]
            cat={"category_name":catinfo[cats[cid]]["name"],"occupied_patch_equivalents":float((gt16*valid16).sum()),"pixel_area":int((gt224*valid224).sum()),"methods":{}}
            for method in METHODS:
                m16=all16[method][j].detach().cpu().numpy();m224=all224[method][j].detach().cpu().numpy()
                th=.55 if method=="v1_raw_pamr" else None if thresholds is None else float(thresholds[method])
                metric={}
                for scale,score,target,valid in (("grid16",m16,gt16,valid16),("224",m224,gt224.astype(np.float32),valid224.astype(np.float32))):
                    ts=thresholds_for_method(method,thresholds)
                    metric[scale]={}
                    for t in ts:
                        tp,fp,fn,tn=confusion(score,target,valid,float(t))
                        metric[scale][f"{float(t):.2f}"]={"tp":tp,"fp":fp,"fn":fn,"tn":tn,"iou":tp/max(1e-12,tp+fp+fn),"precision":tp/max(1e-12,tp+fp),"recall":tp/max(1e-12,tp+fn)}
                cat["methods"][method]={"threshold":th,"grid16":metric["grid16"],"224":metric["224"],
                    "grid16_auroc":auc_binary(m16,gt16,valid16),"pointing_grid16":pointing_accuracy(m16,gt16,valid16),
                    "pointing_224":pointing_accuracy(m224,gt224,valid224)}
            rec["categories"][str(cid)]=cat
        with saved.open(mode) as f:f.write(json.dumps(rec,separators=(",",":"))+"\n");f.flush();os.fsync(f.fileno())
        mode="a"
        rows.append(rec)
        if make_gallery and n<=16:render_gallery(rgb224,rec,all224,targets,present,catinfo,cats,gallery_dir)
        if n%100==0 or n==len(ids): print(f"{name}: {n}/{len(ids)} images processed",flush=True)
    if saved.exists():
        rows=[json.loads(x) for x in saved.read_text().splitlines() if x.strip() and json.loads(x).get("manifest_sha256")==sha256(out/"validation_manifest.json")]
    if len(rows)!=len(ids):raise RuntimeError(f"{name}: expected one record per manifest image ({len(ids)}), got {len(rows)}")
    return rows

def render_gallery(rgb,rec,maps,targets,present,catinfo,cats,directory):
    import matplotlib.pyplot as plt
    cid=present[0];gt=targets[cid][0];fig,ax=plt.subplots(1,5,figsize=(16,3))
    ax[0].imshow(rgb);ax[0].set_title(f"RGB / {catinfo[cats[cid]]['name']}")
    ax[1].imshow(gt,cmap="gray");ax[1].set_title("Ground truth")
    for a,m,name in zip(ax[2:],(maps["v1_raw_pamr"][0].cpu().numpy(),maps["v2_raw"][0].cpu().numpy(),maps["v2_pamr"][0].cpu().numpy()),("V1 epoch50 + PAMR","V2 raw","V2 + PAMR")):
        a.imshow(rgb);a.imshow(m,alpha=.55,cmap="magma");a.set_title(name)
    for a in ax:a.axis("off")
    fig.suptitle(f"COCO val2014 image {rec['image_id']} — same frozen category prompts")
    fig.tight_layout();fig.savefig(directory/f"image_{rec['image_id']}.png",dpi=140);plt.close(fig)

def aggregate(records,thresholds=None):
    agg=defaultdict(lambda:defaultdict(lambda:defaultdict(float))); percat=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    for rec in records:
        for cid,item in rec["categories"].items():
            area=item["occupied_patch_equivalents"]
            for method,m in item["methods"].items():
                t=.55 if method=="v1_raw_pamr" else (thresholds or {}).get(method)
                for scale in ("grid16","224"):
                    if t is None:continue
                    key=f"{float(t):.2f}";v=m[scale].get(key)
                    if not v:continue
                    for stat in ("tp","fp","fn","tn"):agg[method][scale][stat]+=v[stat];percat[cid][method][scale+stat]+=v[stat]
                    agg[method][scale]["object_pairs"]+=1
                    if 0<area<4:
                        for stat in ("tp","fp","fn","tn"):agg[method][scale+"_small"][stat]+=v[stat]
                    percat[cid][method][scale+"pair_iou_sum"]+=v["iou"]
                    percat[cid][method][scale+"pair_iou_count"]+=1
                for key in ("grid16_auroc","pointing_grid16","pointing_224"):
                    value=item["methods"][method].get(key)
                    if value is not None:
                        agg[method]["diagnostics"][key+"_sum"]+=float(value)
                        agg[method]["diagnostics"][key+"_count"]+=1
    rows=[]
    for cid,methods in percat.items():
        for method,scales in methods.items():
            for scale in ("grid16","224"):
                v=scales;tp=v[scale+"tp"];fp=v[scale+"fp"];fn=v[scale+"fn"]
                rows.append({"category_id":int(cid),"method":method,"scale":scale,"iou":tp/max(1e-12,tp+fp+fn),"mean_positive_image_iou":v[scale+"pair_iou_sum"]/max(1,v[scale+"pair_iou_count"]),"positive_image_category_pairs":int(v[scale+"pair_iou_count"]),"precision":tp/max(1e-12,tp+fp),"recall":tp/max(1e-12,tp+fn),"tp":tp,"fp":fp,"fn":fn})
    summary={}
    for method,scales in agg.items():
        summary[method]={}
        for scale in ("grid16","224"):
            v=scales[scale];tp=v["tp"];fp=v["fp"];fn=v["fn"];tn=v["tn"]
            cat_rows=[r for r in rows if r["method"]==method and r["scale"]==scale]
            summary[method][scale]={"micro_iou":tp/max(1e-12,tp+fp+fn),"macro_category_miou_pooled":float(np.mean([r["iou"] for r in cat_rows])) if cat_rows else None,"macro_mean_positive_image_category_iou":float(np.mean([r["mean_positive_image_iou"] for r in cat_rows])) if cat_rows else None,"precision":tp/max(1e-12,tp+fp),"recall":tp/max(1e-12,tp+fn),"false_negative_foreground_rate":fn/max(1e-12,tp+fn),"candidate_background_coverage":1.0-(tp+fp)/max(1e-12,tp+fp+fn+tn),"positive_image_category_pairs":int(v["object_pairs"])}
            sm=scales[scale+"_small"];summary[method][scale]["small_object_iou"]=sm["tp"]/max(1e-12,sm["tp"]+sm["fp"]+sm["fn"])
        d=scales["diagnostics"];summary[method]["diagnostics"]={k.removesuffix("_sum"):d[k]/max(1,d[k.removesuffix("_sum")+"_count"]) for k in d if k.endswith("_sum")}
    return summary,rows

@torch.inference_mode()
def retrieval_diagnostic(models,cache,caption_cache,split,patch_ids,penultimate,final,device,out):
    dev_ids=list(map(int,split["development"]["image_ids"])); pos_by_id={iid:i for i,iid in enumerate(dev_ids)}
    cap_ids=np.load(caption_cache/"caption_ids_val2014.npy",mmap_mode="r"); text=np.load(caption_cache/"text_features_val2014_fp16.npy",mmap_mode="r")
    cap_rows=np.asarray([r for r,(_,iid) in enumerate(cap_ids) if int(iid) in pos_by_id],dtype=np.int64)
    cap_iids=np.asarray([int(cap_ids[r,1]) for r in cap_rows],dtype=np.int64)
    cap_text=np.asarray(text[cap_rows],dtype=np.float16)
    img_rows={int(i):r for r,i in enumerate(patch_ids)}
    selected_rows=np.asarray([img_rows[i] for i in dev_ids],dtype=np.int64)
    output={}
    for name,model,patches in (("V1_epoch50",models["v1"],final),("V2_best",models["v2"],penultimate)):
        proj=F.normalize(model.text_projection(torch.from_numpy(cap_text.astype(np.float32)).to(device)),dim=-1)
        matrix=np.empty((len(dev_ids),len(cap_rows)),dtype=np.float32)
        for ib in range(0,len(dev_ids),8):
            rows=selected_rows[ib:ib+8]; x=F.normalize(torch.from_numpy(np.array(patches[rows],dtype=np.float32,copy=True)).to(device),dim=-1)
            for cb in range(0,len(cap_rows),128):
                t=proj[cb:cb+128];s=torch.einsum("bpd,cd->bcp",x,t)
                score=.1*(torch.logsumexp(s/.1,dim=-1)-np.log(256.0))
                matrix[ib:ib+len(rows),cb:cb+len(t)]=score.cpu().numpy()
            if ib%128==0:print(f"retrieval {name}: images={min(ib+8,len(dev_ids))}/{len(dev_ids)} captions={len(cap_rows)}",flush=True)
        image_pos=np.full(len(dev_ids),-np.inf,np.float32)
        caption_pos=np.empty(len(cap_rows),np.float32)
        for r,iid in enumerate(dev_ids):
            js=np.flatnonzero(cap_iids==iid)
            if len(js)==0:raise RuntimeError(f"Dev image {iid} has no captions")
            image_pos[r]=matrix[r,js].max();caption_pos[js]=matrix[r,js]
        ri=1+(matrix>image_pos[:,None]).sum(1);rt=1+(matrix>caption_pos[None,:]).sum(0)
        def rec(rank):return {f"R@{k}":float(np.mean(rank<=k)) for k in (1,5,10)}
        logits=matrix.astype(np.float64)/.07
        li=[];lt=[]
        for r,iid in enumerate(dev_ids):
            js=np.flatnonzero(cap_iids==iid);li.append(np.logaddexp.reduce(logits[r,js])-np.logaddexp.reduce(logits[r]))
        for c,iid in enumerate(cap_iids):
            r=pos_by_id[int(iid)];lt.append(logits[r,c]-np.logaddexp.reduce(logits[:,c]))
        output[name]={"images":len(dev_ids),"captions":len(cap_rows),"image_to_text":rec(ri),"text_to_image":rec(rt),"multi_positive_symmetric_infonce":float(-(np.mean(li)+np.mean(lt))/2),"score_mean":float(matrix.mean()),"score_std":float(matrix.std()),"pooling":"tau=0.10 patch log-mean-exp","contrastive_temperature":.07}
    atomic_json(out/"development_image_caption_retrieval.json",{"status":"complete","split":"development only","metrics":output})
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset-root",type=Path,required=True);p.add_argument("--cache-root",type=Path,required=True);p.add_argument("--text-cache-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--split-manifest",type=Path,required=True);p.add_argument("--train-output",type=Path,required=True);p.add_argument("--checkpoint-dir",type=Path,required=True);p.add_argument("--v1-checkpoint",type=Path,required=True);p.add_argument("--resume",action="store_true");a=p.parse_args()
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");a.output_dir.mkdir(parents=True,exist_ok=True)
    split=json.loads(a.split_manifest.read_text());atomic_json(a.output_dir/"validation_manifest.json",split)
    # Categories are emitted by prepare.py at the experiment root, while this
    # evaluator writes its outputs to the nested evaluation/ directory.
    # Prefer that canonical file and retain support for a local copy.
    categories_path=a.output_dir/"categories.json"
    if not categories_path.is_file():
        categories_path=a.train_output/"categories.json"
    cats=json.loads(categories_path.read_text());category_text_v2=np.load(a.cache_root/"category_text_features_fp16.npy",mmap_mode="r")
    geom=json.loads((a.cache_root/"patch_features_train2014_manifest.json").read_text())["preprocessing_geometry"]
    import sys
    repo=Path(__file__).resolve().parents[5];sys.path.insert(0,str(repo))
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase3_evaluate import load_pamr, encode_class_templates
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase3_spatial import decode_coco_rle, binary_mask_from_polygons
    v1_ids,v1_names,v1_text_tensor,prompt_metadata=encode_class_templates(dev)
    if [int(x) for x in v1_ids]!=[int(x["id"]) for x in cats] or v1_names!=[x["name"] for x in cats]:
        raise RuntimeError("V1 seven-template category order does not match cached category order")
    category_text_v1=v1_text_tensor.detach().float().cpu().numpy()
    atomic_json(a.output_dir/"evaluation_metadata.json",{"v1_prompt_templates":["itap of a {}.","a bad photo of a {}.","a origami {}.","a photo of the large {}.","a {} in a video game.","art of the {}.","a photo of the small {}."],"v1_prompt_source":"existing Phase 3 encode_class_templates implementation","v1_embedding_aggregation":"mean raw text embedding across seven templates before V1 MLP projection and L2 normalization","v1_visual_features":"final-layer normalized DINO patch features from the cached V1-compatible extraction","v2_prompt_source":"single training prompt: a photo of a <category>","v2_visual_features":"OpenVLA penultimate DINO patch features from the standard 224x224 path","ground_truth":"COCO val2014 instance masks; non-crowd instances unioned by category; crowd pixels excluded; fractional 16x16 occupancy and aligned binary 224 targets","category_aggregation":"pooled per-category IoU sums TP/FP/FN over all positive image/category pairs for each category; absent-category images are excluded; separately report macro mean per-positive-image/category IoU to match the V1 Phase 3 aggregation","split_manifest_sha256":sha256(a.split_manifest),"v1_encoder_metadata":prompt_metadata})
    pamr=load_pamr(dev)
    ckroot=a.checkpoint_dir; best=torch.load(a.train_output/"best_projection.pt",map_location="cpu",weights_only=False)
    v2=RegionTextProjection().to(dev);v2.load_state_dict(best["model"],strict=True);v2.eval()
    v1_path=a.v1_checkpoint
    if not v1_path.is_file(): raise FileNotFoundError(f"V1 epoch50 checkpoint missing: {v1_path}")
    v1ck=torch.load(v1_path,map_location="cpu",weights_only=False);v1=RegionTextProjection().to(dev);v1.load_state_dict(v1ck["model"],strict=True);v1.eval()
    patch_ids=np.load(a.cache_root/"image_ids_val2014.npy",mmap_mode="r")
    penultimate=np.load(a.cache_root/"patch_features_val2014_fp16.npy",mmap_mode="r")
    final=np.load(a.cache_root/"patch_features_val2014_final_norm_fp16.npy",mmap_mode="r")
    retrieval_diagnostic({"v1":v1,"v2":v2},a.cache_root,a.text_cache_root,split,patch_ids,penultimate,final,dev,a.output_dir)
    allrows={}
    for name in ("development","evaluation"):
        ids=split[name]["image_ids"]
        th=None
        if name=="evaluation":
            devresult=json.loads((a.output_dir/"development_thresholds.json").read_text());th=devresult["selected_thresholds"]
        rows=process_subset(name,ids,split=split,root=a.dataset_root,cache=a.cache_root,inst_json=a.dataset_root/"annotations2014/instances_val2014.json",catinfo=cats,geometry=geom,models={"v1":v1,"v2":v2},category_text_v1=category_text_v1,category_text_v2=category_text_v2,pamr=pamr,device=dev,out=a.output_dir,thresholds=th,make_gallery=name=="evaluation")
        if name=="development":
            calibration={}
            for method in METHODS:
                candidates=[]
                for t in ([.55] if method=="v1_raw_pamr" else THRESHOLDS):
                    tot=defaultdict(float);bycat=defaultdict(lambda:defaultdict(float))
                    for rec in rows:
                        for cid,item in rec["categories"].items():
                            v=item["methods"][method]["grid16"][f"{float(t):.2f}"]
                            for s in ("tp","fp","fn"):tot[s]+=v[s];bycat[cid][s]+=v[s]
                    miou=np.mean([q["tp"]/max(1e-12,q["tp"]+q["fp"]+q["fn"]) for q in bycat.values()])
                    candidates.append((float(miou),float(t)))
                best=max(candidates,key=lambda q:(q[0],-abs(q[1]-.5),-q[1]))
                calibration[method]={"threshold":best[1],"development_macro_category_miou":best[0],"tie_break":"closest to 0.50, then lower threshold"}
            selected={k:v["threshold"] for k,v in calibration.items()}
            th=selected
            atomic_json(a.output_dir/"development_thresholds.json",{"manifest_sha256":sha256(a.split_manifest),"selected_thresholds":selected,"calibration":calibration,"rule":"maximize native-grid pooled per-category IoU on the fixed 1,000-image development split; V1 threshold fixed at 0.55; ties prefer threshold nearest 0.50 then lower threshold"})
        summary,percat=aggregate(rows,th)
        atomic_json(a.output_dir/f"{name}_metrics.json",{"summary":summary,"per_category":percat,"image_count":len(rows)})
        allrows[name]=len(rows)
    expected={"development":1000,"evaluation":3000}
    if allrows!=expected:raise RuntimeError(f"Split coverage failed: expected {expected}, got {allrows}")
    atomic_json(a.output_dir/"evaluation_complete.json",{"status":"complete","images":allrows,"v1_checkpoint":str(v1_path),"v1_checkpoint_sha256":sha256(v1_path),"v1_prompt_template_count":7,"v2_prompt_template_count":1,"v2_checkpoint":str(a.train_output/"best_projection.pt"),"v2_checkpoint_sha256":sha256(a.train_output/"best_projection.pt"),"device":str(dev)})

if __name__=="__main__":main()
