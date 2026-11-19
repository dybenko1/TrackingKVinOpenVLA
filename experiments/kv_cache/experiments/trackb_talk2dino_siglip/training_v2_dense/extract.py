"""Resumable native OpenVLA penultimate DINO patch extraction and mask caching."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
from PIL import Image

from .masks import geometry_from_transform, extract_instances_cache

VLA_ID="openvla/openvla-7b-finetuned-libero-spatial"

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def load_native_dino():
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor=AutoProcessor.from_pretrained(VLA_ID,trust_remote_code=True)
    model=AutoModelForVision2Seq.from_pretrained(VLA_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True)
    vision=model.vision_backbone
    dino=getattr(vision,"dino_featurizer",getattr(vision,"featurizer",None))
    if dino is None: raise RuntimeError("OpenVLA vision backbone exposes no DINO featurizer")
    dino.eval().to("cuda:0" if torch.cuda.is_available() else "cpu")
    for p in dino.parameters(): p.requires_grad_(False)
    geometry=geometry_from_transform(processor.image_processor)
    if geometry["resize_size"] not in ([224,224],(224,224)) or tuple(geometry["crop_size"])!=(224,224):
        raise RuntimeError(f"Unexpected OpenVLA policy DINO geometry: {geometry}")
    return model, processor, vision, dino, geometry, torch

def process_image_batch(entries, processor, dino, device, torch):
    images=[Image.open(path).convert("RGB") for _,path in entries]
    pixels=processor.image_processor(images=images,return_tensors="pt")["pixel_values"]
    if pixels.ndim!=4 or pixels.shape[1]<3 or tuple(pixels.shape[-2:])!=(224,224):
        raise RuntimeError(f"OpenVLA policy processor produced unexpected pixels {tuple(pixels.shape)}")
    x=pixels[:,:3].to(device=device,dtype=next(dino.parameters()).dtype)
    with torch.inference_mode(): y=dino(x)
    if isinstance(y,(tuple,list)): y=y[0]
    if y.ndim!=3 or tuple(y.shape[1:])!=(256,1024): raise RuntimeError(f"Native OpenVLA DINO output is {tuple(y.shape)}, expected [B,256,1024]")
    if not torch.isfinite(y).all(): raise FloatingPointError("Nonfinite native DINO features")
    return images,pixels,x,y

def _run_batch_with_oom_split(entries, processor, dino, device, torch):
    """Run every entry, splitting batches on OOM without dropping their tails."""
    try:
        _, _, _, y = process_image_batch(entries, processor, dino, device, torch)
        return y
    except torch.cuda.OutOfMemoryError:
        if len(entries) <= 1:
            raise
        torch.cuda.empty_cache()
        mid = len(entries) // 2
        left = _run_batch_with_oom_split(entries[:mid], processor, dino, device, torch)
        right = _run_batch_with_oom_split(entries[mid:], processor, dino, device, torch)
        return torch.cat((left, right), dim=0)

def _run_final_with_oom_split(entries, processor, dino, device, torch):
    """Extract final normalized patch states for every entry, safely on OOM."""
    try:
        _, _, x, _ = process_image_batch(entries, processor, dino, device, torch)
        with torch.inference_mode():
            return dino.get_intermediate_layers(x, n={len(dino.blocks)-1}, norm=True)[0]
    except torch.cuda.OutOfMemoryError:
        if len(entries) <= 1:
            raise
        torch.cuda.empty_cache()
        mid = len(entries) // 2
        left = _run_final_with_oom_split(entries[:mid], processor, dino, device, torch)
        right = _run_final_with_oom_split(entries[mid:], processor, dino, device, torch)
        return torch.cat((left, right), dim=0)

def extract_patches(dataset_root:Path, feature_root:Path, split:str="train", batch_size:int=4, status_path:Path|None=None, selected_ids=None, keep_v1_final=False):
    cap=json.loads((dataset_root/"annotations2014"/f"captions_{split}2014.json").read_text())
    selected_set=None if selected_ids is None else set(map(int,selected_ids))
    entries=[(int(x["id"]),dataset_root/f"{split}2014"/x["file_name"]) for x in cap["images"] if selected_set is None or int(x["id"]) in selected_set]
    ids=np.asarray([x[0] for x in entries],dtype=np.int64)
    for iid,p in entries[:min(16,len(entries))]:
        if not p.is_file(): raise FileNotFoundError(p)
    feature_root.mkdir(parents=True,exist_ok=True)
    ids_path=feature_root/f"image_ids_{split}2014.npy"; feat_path=feature_root/f"patch_features_{split}2014_fp16.npy"; done_path=feature_root/f"patch_done_{split}2014.npy"
    if ids_path.exists() and not np.array_equal(np.load(ids_path,mmap_mode="r"),ids): raise RuntimeError("Native DINO cache image-ID order differs from COCO caption metadata")
    if not ids_path.exists(): np.save(ids_path,ids,allow_pickle=False)
    manifest_path=feature_root/f"patch_features_{split}2014_manifest.json"
    final_path=feature_root/f"patch_features_{split}2014_final_norm_fp16.npy" if keep_v1_final else None
    if done_path.exists() and manifest_path.exists() and feat_path.exists():
        prior=json.loads(manifest_path.read_text())
        done_existing=np.load(done_path,mmap_mode="r")
        cached=np.load(feat_path,mmap_mode="r")
        valid_cache=(prior.get("status")=="complete" and prior.get("count")==len(ids)
            and np.all(done_existing==1) and cached.shape==(len(ids),256,1024)
            and prior.get("image_ids_sha256")==hashlib.sha256(ids.tobytes()).hexdigest())
        if keep_v1_final:
            valid_cache=valid_cache and final_path.exists() and np.load(final_path,mmap_mode="r").shape==(len(ids),256,1024)
        if valid_cache:
            return prior
    model,processor,vision,dino,geometry,torch=load_native_dino(); device=next(dino.parameters()).device
    # Audit the actual policy processor output; no approximate transform is substituted.
    first=Image.open(entries[0][1]).convert("RGB")
    _,policy_pixels,policy_dino_pixels,_=process_image_batch([(ids[0],entries[0][1])],processor,dino,device,torch)
    policy_slice=policy_pixels[:,:3].to(device=device,dtype=policy_dino_pixels.dtype)
    if policy_dino_pixels.shape!=(1,3,224,224) or not torch.equal(policy_slice,policy_dino_pixels):
        raise RuntimeError(f"Policy processor DINO input split mismatch: {tuple(policy_slice.shape)} vs {tuple(policy_dino_pixels.shape)}")
    expected=(len(ids),256,1024)
    feat=np.lib.format.open_memmap(feat_path,mode="r+" if feat_path.exists() else "w+",dtype=np.float16,shape=expected)
    final_feat=np.lib.format.open_memmap(final_path,mode="r+" if final_path.exists() else "w+",dtype=np.float16,shape=expected) if keep_v1_final else None
    done=np.lib.format.open_memmap(done_path,mode="r+" if done_path.exists() else "w+",dtype=np.uint8,shape=(len(ids),))
    if not done_path.exists(): done[:]=0; done.flush()
    todo=np.flatnonzero(np.asarray(done)==0); t0=time.time(); completed=int(np.asarray(done).sum())
    step=max(1,batch_size)
    for off in range(0,len(todo),step):
        rows=todo[off:off+step]; batch=[entries[int(i)] for i in rows]
        y=_run_batch_with_oom_split(batch,processor,dino,device,torch)
        if y.shape[0] != len(rows): raise RuntimeError("OOM retry did not return every requested image")
        out=y.float().cpu().numpy().astype(np.float16)
        feat[rows]=out; feat.flush()
        if final_feat is not None:
            # V1's segmentation path used final normalized patch states. Keep this
            # separate from V2's exact standard-policy penultimate representation.
            final=_run_final_with_oom_split(batch,processor,dino,device,torch)
            if tuple(final.shape[1:])!=(256,1024) or not torch.isfinite(final).all(): raise RuntimeError("V1 final-layer validation features malformed")
            final_feat[rows]=final.float().cpu().numpy().astype(np.float16); final_feat.flush()
        done[rows]=1; done.flush(); completed+=len(rows)
        if status_path and (completed%128<len(rows) or completed==len(ids)):
            atomic_json(status_path,{"stage":f"extract_{split}_native_dino","state":"running","completed":completed,"total":len(ids),"feature_shape":list(expected),"feature_dtype":"float16","elapsed_seconds":time.time()-t0,"geometry":geometry})
        if completed%1024==0 or completed==len(ids): print(f"native DINO {split}: {completed}/{len(ids)}",flush=True)
    feat.flush(); del feat,done
    if final_feat is not None: final_feat.flush(); del final_feat
    manifest={"status":"complete","split":split,"count":len(ids),"patch_features":str(feat_path),"shape":list(expected),"dtype":"float16","definition":"OpenVLA DinoSigLIPViTBackbone.dino_featurizer forward; standard penultimate block output (len(blocks)-2), exact PrismaticImageProcessor DINO channel slice, pre-normalization output as returned by policy backbone","model_id":VLA_ID,"timm_dino_id":"vit_large_patch14_reg4_dinov2.lvd142m","grid":[16,16],"feature_dim":1024,"image_ids_sha256":hashlib.sha256(ids.tobytes()).hexdigest(),"preprocessing_geometry":geometry,"policy_processor_dino_slice_exact":True,"completed_rows":len(ids)}
    manifest["selected_ids"]=selected_ids is not None
    manifest["v1_final_norm_features"]=str(final_path) if final_path else None
    atomic_json(manifest_path,manifest)
    return manifest

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset-root",type=Path,required=True);p.add_argument("--feature-root",type=Path,required=True);p.add_argument("--split",choices=["train","val"],required=True);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--status",type=Path)
    a=p.parse_args(); print(json.dumps(extract_patches(a.dataset_root,a.feature_root,a.split,a.batch_size,a.status),indent=2),flush=True)
if __name__=="__main__":main()
