#!/usr/bin/env python3
"""Restartable COCO-Object run using upstream Talk2DINO model inference.

The upstream DINOText methods are used directly. This script supplies the
single-process equivalent of the upstream MMSeg slide/test/evaluate wrapper;
the validated runtime lacks a compatible MMCV/MMseg build (Torch 2.6/Python
3.12). Full per-image confusion matrices are atomically checkpointed.
"""
from __future__ import annotations

import argparse, ast, csv, hashlib, importlib.util, json, os, sys, time, zipfile
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = Path("/opt/openvla")
WORKTREE = REPO / "cache/worktrees/trackb-dense-grounding-v2"
UPSTREAM = REPO / "cache/references/Talk2DINO"
REF = UPSTREAM / "src/open_vocabulary_segmentation"
DATA = REPO / "cache/datasets/coco_stuff164k"
DINOV2_REPO = REPO / "cache/references/dinov2"
DINO_WEIGHT = REPO / "cache/models/dinov2/facebook_dinov2_large/dinov2_vitl14_reg4_pretrain.pth"
CLIP_WEIGHT = REPO / "cache/models/openai/clip/ViT-B-16.pt"
MLP_WEIGHT = UPSTREAM / "weights/vitl_mlp_infonce.pth"
OUT = WORKTREE / "experiments/kv_cache/outputs/trackb_talk2dino_baseline_validation/coco_object"
IMAGE_SHA = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
MLP_SHA = "4e28d24ecb905592c902b04d9c077484346ebafac8547b968bd5af16291d8d69"
DINO_SHA = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
NCLASS, IGNORE, BG_THRESHOLD, LAMBDA_BG = 81, 255, 0.54, 0.2
SCALE, CROP, STRIDE = (2048, 448), 448, 224


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n")
    with tmp.open("rb") as f: os.fsync(f.fileno())
    os.replace(tmp,path)


def class_names():
    source=REF/"segmentation/datasets/coco_object.py"
    tree=ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.ClassDef) and node.name=="COCOObjectDataset":
            for item in node.body:
                if isinstance(item,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CLASSES" for t in item.targets):
                    names=ast.literal_eval(item.value)
                    assert len(names)==NCLASS and names[0]=="background"
                    return list(names)
    raise RuntimeError("Cannot find upstream COCOObjectDataset.CLASSES")


def label_lut():
    tree=ast.parse((REF/"convert_dataset/convert_coco_object.py").read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="clsID_to_trID" for t in n.targets))
    mapping=ast.literal_eval(node.value); lut=np.zeros(256,dtype=np.uint8)
    for src,dst in mapping.items():
        # Upstream conversion maps source labels >90 to background, except
        # 255 which remains the dataset ignore label.
        lut[int(src)] = 255 if int(src) == 255 else (0 if int(src) > 90 else int(dst)+1)
    return lut


def resize_keep_ratio(image, max_size=SCALE):
    import cv2
    h,w=image.shape[:2]; maxlong,minshort=max_size
    factor=min(maxlong/max(h,w),minshort/min(h,w))
    return cv2.resize(image,(int(w*factor+0.5),int(h*factor+0.5)),interpolation=cv2.INTER_LINEAR)


def windows(height,width):
    def starts(length):
        grids=max(length-CROP+STRIDE-1,0)//STRIDE+1; out=[]
        for i in range(grids):
            hi=min(i*STRIDE+CROP,length); lo=max(hi-CROP,0); out.append((lo,hi))
        return out
    for y1,y2 in starts(height):
        for x1,x2 in starts(width): yield y1,y2,x1,x2


def confusion(pred,gt):
    pred=np.asarray(pred,dtype=np.int64).ravel(); gt=np.asarray(gt,dtype=np.int64).ravel()
    valid=(gt!=IGNORE)&(gt>=0)&(gt<NCLASS)&(pred>=0)&(pred<NCLASS)
    return np.bincount(NCLASS*gt[valid]+pred[valid],minlength=NCLASS*NCLASS).reshape(NCLASS,NCLASS).astype(np.uint64)


def metrics(cm):
    x=cm.astype(np.float64); tp=np.diag(x); g=x.sum(1); p=x.sum(0); u=g+p-tp
    iou=np.divide(tp,u,out=np.full(NCLASS,np.nan),where=u>0)
    return {"miou_including_background":float(np.nanmean(iou)),"foreground_miou":float(np.nanmean(iou[1:])),
            "background_iou":float(iou[0]),"pixel_accuracy":float(tp.sum()/max(1,x.sum())),
            "foreground_precision_micro":float(tp[1:].sum()/max(1,p[1:].sum())),
            "foreground_recall_micro":float(tp[1:].sum()/max(1,g[1:].sum())),"valid_pixels":int(x.sum()),
            "classes_with_ground_truth":int((g>0).sum()),"classes_nonempty_union":int(np.isfinite(iou).sum())}, iou


def checkpoint_pair(path, values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**values); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def _json_status(stage,done,total,**kw):
    atomic_json(OUT/"RUN_STATUS.json",{"status":"running","stage":stage,"completed":done,"total":total,
        "updated_unix":time.time(),"container":"trackb-v1-coco-object-20261126",**kw})


def load_model(device):
    import torch
    from experiments.kv_cache.experiments.trackb_talk2dino_native.model import OpenAIClipTextTower
    from experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.official_inference import OfficialTalk2DINO
    from experiments.kv_cache.experiments.trackb_talk2dino_native.run_libero_e2 import load_official_dino
    for p,want,label in ((DINO_WEIGHT,DINO_SHA,"DINOv2"),(CLIP_WEIGHT,IMAGE_SHA,"OpenAI CLIP"),(MLP_WEIGHT,MLP_SHA,"Talk2DINO MLP")):
        got=sha256(p)
        if got!=want: raise RuntimeError(f"{label} checksum mismatch {got} != {want}")
    dino,meta=load_official_dino(device,torch.float32)
    official=OfficialTalk2DINO(UPSTREAM,dino,CLIP_WEIGHT,MLP_WEIGHT,resize_dim=448,device=str(device))
    names=class_names()
    embeddings=official.encode_categories(names[1:], "sub_imagenet_template")
    if tuple(embeddings.shape)!=(80,1024) or not torch.isfinite(embeddings).all(): raise RuntimeError("Bad official COCO class-text embeddings")
    info={"official_repo_revision":subprocess_git(UPSTREAM),"official_dino":meta,
          "clip_sha256":sha256(CLIP_WEIGHT),"mlp_sha256":sha256(MLP_WEIGHT),"templates":"upstream sub_imagenet_template",
          "class_count":len(names),"classnames_sha256":hashlib.sha256("\n".join(names).encode()).hexdigest(),
          "image_size":448,"test_scale":list(SCALE),"slide_crop":CROP,"slide_stride":STRIDE,
          "background_threshold":BG_THRESHOLD,"lambda_bg":LAMBDA_BG,"pamr":{"iter":10,"dilations":[1,2,4,8,12,24]},
          "dino_transform":"upstream DINOText Resize(448,448), ImageNet mean/std; MMCV FloatImage crop in BGR then upstream BGR→RGB",
          "eval_metric":"global 81-class confusion matrix; nanmean over nonempty class unions including background"}
    return official,names,embeddings,info


def subprocess_git(repo):
    import subprocess
    try: return subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    except Exception: return "unavailable"


def infer_image(official,text_embed,image_path,gt_path,lut,pamr,debug=False):
    import cv2,torch,torch.nn.functional as F
    image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(image_path)
    H0,W0=image.shape[:2]; gt_raw=np.asarray(Image.open(gt_path),dtype=np.uint8)
    if gt_raw.shape!=(H0,W0): raise RuntimeError(f"GT shape mismatch {image_path.stem}: {gt_raw.shape}/{image.shape}")
    gt=lut[gt_raw]; scaled=resize_keep_ratio(image); h,w=scaled.shape[:2]
    sums={k:np.zeros((NCLASS,h,w),np.float32) for k in ("raw","clean","clean_pamr")}; counts=np.zeros((h,w),np.float32)
    start=time.perf_counter()
    for y1,y2,x1,x2 in windows(h,w):
        crop=scaled[y1:y2,x1:x2]
        if crop.shape[:2]!=(448,448): raise RuntimeError(f"Official slide crop not 448: {crop.shape}")
        result=official.score_bgr_crop(crop,text_embed,clean=True,pamr=False)
        clean=result["scores"].float()
        raw=F.interpolate(result["cosine"].unsqueeze(0).sigmoid(),(448,448),mode="bilinear",align_corners=True)[0]
        rgb=result["rgb"].unsqueeze(0)
        clean_pamr=official.model.apply_pamr(rgb,clean.unsqueeze(0))[0].float()
        bg=torch.full((1,448,448),BG_THRESHOLD,device=clean.device,dtype=torch.float32)
        triplets={"raw":torch.cat((bg,raw),0),"clean":torch.cat((bg,clean),0),
                  "clean_pamr":torch.cat((bg,clean_pamr),0)}
        for k,score in triplets.items():
            sums[k][:,y1:y2,x1:x2]+=score.detach().cpu().numpy()[:,:y2-y1,:x2-x1]
        counts[y1:y2,x1:x2]+=1
    result_cm={}
    for k,total in sums.items():
        total/=np.maximum(counts[None],1)
        ten=torch.from_numpy(total).unsqueeze(0)
        full=F.interpolate(ten,size=(H0,W0),mode="bilinear",align_corners=False)[0]
        pred=full.argmax(0).byte().cpu().numpy()
        result_cm[k]=confusion(pred,gt)
    return result_cm,(time.perf_counter()-start), (image,gt,scaled)


def write_metrics(cm_by_name,names):
    rows=[]; summary={}
    for key,cm in cm_by_name.items():
        overall,iou=metrics(cm); summary[key]=overall
        for i,name in enumerate(names):
            rows.append({"condition":key,"class_id":i,"class_name":name,
                         "iou":None if not np.isfinite(iou[i]) else float(iou[i]),
                         "ground_truth_pixels":int(cm[i].sum()),"predicted_pixels":int(cm[:,i].sum()),
                         "true_positive_pixels":int(cm[i,i])})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"per_category_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    atomic_json(OUT/"summary.json",summary)


def run(limit=None,smoke=False):
    import torch
    image_dir=DATA/"images/val2017"; gt_dir=DATA/"annotations_raw/val2017"
    images=sorted(image_dir.glob("*.jpg")); expected=5000
    if len(images)!=expected: raise RuntimeError(f"Expected 5000 val2017 images, found {len(images)}")
    if any(not (gt_dir/(p.stem+".png")).is_file() for p in images): raise RuntimeError("Missing image/GT pair")
    if smoke: images=images[:5]
    elif limit: images=images[:limit]
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True,exist_ok=True); cache=OUT/"cache/per_image"; cache.mkdir(parents=True,exist_ok=True)
    lut=label_lut(); names=class_names(); _json_status("load_models",0,len(images),smoke=smoke)
    official,classnames,embeddings,identity=load_model(device)
    import torch
    official.model.eval(); official.model.model.eval() if hasattr(official.model.model,"eval") else None
    from experiments.kv_cache.experiments.trackb_talk2dino_baseline_validation.official_inference import OfficialTalk2DINO
    with torch.inference_mode():
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
        latencies=[]; done=[]
        for i,p in enumerate(images,1):
            target=cache/f"{p.stem}.npz"
            if target.exists():
                with np.load(target) as z: values={k:np.array(z[k]) for k in z.files}
                if int(values.get("verified",0))==1:
                    done.append(p.stem); latencies.append(float(values.get("seconds",0))); continue
            cms,secs,_=infer_image(official,embeddings,p,gt_dir/f"{p.stem}.png",lut,None)
            checkpoint_pair(target,{**cms,"seconds":np.asarray(secs,np.float32),"verified":np.asarray(1,np.uint8)})
            done.append(p.stem); latencies.append(secs)
            if i%10==0 or i==len(images):
                _json_status("evaluate_images",len(done),len(images),last_image=p.stem,seconds_per_image=float(np.mean(latencies)),smoke=smoke)
        cms={k:np.zeros((NCLASS,NCLASS),np.uint64) for k in ("raw","clean","clean_pamr")}
        for image_id in done:
            with np.load(cache/f"{image_id}.npz") as z:
                for k in cms: cms[k]+=z[k]
        for key, value in cms.items():
            np.save(OUT/f"confusion_{key}.npy", value)
        write_metrics(cms,names)
        summary=json.loads((OUT/"summary.json").read_text())
        for key,val in summary.items():
            val["published_reference_miou"]=0.376 if key=="clean" else 0.423 if key=="clean_pamr" else None
            if val["published_reference_miou"] is not None: val["delta_from_published"]=val["miou_including_background"]-val["published_reference_miou"]
        atomic_json(OUT/"summary.json",summary)
        identity.update({"evaluated_images":len(done),"expected_images":5000,"smoke":smoke,
                         "latency_s_per_image":{"mean":float(np.mean(latencies)),"median":float(np.median(latencies)),
                                                 "p95":float(np.percentile(latencies,95))},
                         "peak_cuda_allocated_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else None,
                         "official_miou_note":"The Talk2DINO source model inference functions are used unchanged; single-process code implements the pinned MMSeg slide and confusion-matrix semantics because compatible MMCV/MMseg packages are absent.",
                         "source_config_files":["configs/coco_object/dinotext_coco_object_vitl_mlp_infonce.yml","src/open_vocabulary_segmentation/configs/coco_object/eval_coco_object_vitl.yml","src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/coco.py"]})
        atomic_json(OUT/"configuration.json",identity)
    official.close()
    final_status = "smoke_complete" if smoke else ("complete" if len(done) == expected else "partial")
    atomic_json(OUT/"RUN_STATUS.json",{"status":final_status,"stage":"complete",
        "completed":len(done),"total":len(images),"updated_unix":time.time(),"smoke":smoke})
    return json.loads((OUT/"summary.json").read_text())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--limit",type=int)
    args=ap.parse_args()
    try:
        print(json.dumps(run(args.limit,args.smoke),indent=2),flush=True)
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/"RUN_STATUS.json",{"status":"failed","stage":"exception",
            "error":repr(exc),"traceback":traceback.format_exc(),"updated_unix":time.time()})
        raise

if __name__=="__main__": main()
