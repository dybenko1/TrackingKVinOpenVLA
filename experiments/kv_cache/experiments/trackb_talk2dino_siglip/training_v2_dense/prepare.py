"""Idempotent extraction/target preparation for the full dense V2 experiment."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
from .extract import extract_patches, atomic_json
from .manifests import ensure_instance_annotations, create_split_manifest
from .masks import geometry_from_transform, extract_instances_cache
from ..text_encoder import load_siglip_text_tower
from ..config import SIGLIP_TEXT_MODEL_REVISION

def prepare(a):
    annotations=ensure_instance_annotations(a.dataset_root)
    train_caps=json.loads((a.dataset_root/"annotations2014/captions_train2014.json").read_text())
    val_caps=json.loads((a.dataset_root/"annotations2014/captions_val2014.json").read_text())
    train_inst=json.loads(annotations["annotations/instances_train2014.json"].read_text())
    categories=sorted(train_inst["categories"],key=lambda x:int(x["id"]))
    cat_path=a.output_dir/"categories.json"; atomic_json(cat_path,[{"id":int(c["id"]),"name":c["name"]} for c in categories])
    split_path=a.output_dir/"validation_manifest.json"
    split=create_split_manifest(a.dataset_root/"annotations2014/captions_val2014.json",annotations["annotations/instances_val2014.json"],a.previous_output,split_path)
    # Validate train caption mappings against the pre-existing SigLIP feature cache.
    cache=a.cache_root; source=a.train_text_cache_root
    ids=np.load(source/"image_ids_train2014.npy",mmap_mode="r"); capids=np.load(source/"caption_ids_train2014.npy",mmap_mode="r"); text=np.load(source/"text_features_train2014_fp16.npy",mmap_mode="r")
    if len(ids)!=82783 or len(capids)!=414113 or text.shape!=(414113,1152): raise RuntimeError("Reusable V1 text cache does not match full COCO train2014")
    wanted=np.asarray([int(x["id"]) for x in train_caps["images"]],dtype=np.int64)
    if not np.array_equal(ids,wanted): raise RuntimeError("Existing image IDs do not match caption annotation order")
    ann_map={int(x["id"]):int(x["image_id"]) for x in train_caps["annotations"]}
    if len(ann_map)!=414113 or any(ann_map.get(int(aid))!=int(iid) for aid,iid in capids): raise RuntimeError("Existing SigLIP text feature joins are not valid")
    status=a.status
    atomic_json(status,{"stage":"train_native_dino","state":"running","pid":os.getpid(),"progress":0,"total":82783})
    train_manifest=extract_patches(a.dataset_root,cache,"train",a.batch_size,status_path=status)
    # Extract the fixed 1k+3k image-only validation set; preserve final-layer features for the V1 reference.
    val_ids=split["development"]["image_ids"]+split["evaluation"]["image_ids"]
    atomic_json(status,{"stage":"validation_native_dino","state":"running","pid":os.getpid(),"progress":0,"total":len(val_ids)})
    val_manifest=extract_patches(a.dataset_root,cache,"val",a.batch_size,status_path=status,selected_ids=val_ids,keep_v1_final=True)
    # Exact mask geometry comes from the actual OpenVLA transform recorded by extraction.
    geometry=train_manifest["preprocessing_geometry"]
    train_infos={int(x["id"]):x for x in train_inst["images"]}
    val_inst=json.loads(annotations["annotations/instances_val2014.json"].read_text())
    val_infos={int(x["id"]):x for x in val_inst["images"]}
    val_row_ids=np.load(cache/"image_ids_val2014.npy",mmap_mode="r")
    atomic_json(status,{"stage":"train_fractional_masks","state":"running","pid":os.getpid(),"progress":0,"total":len(ids)})
    def progress(n,total): atomic_json(status,{"stage":"fractional_masks","state":"running","pid":os.getpid(),"progress":n,"total":total})
    train_mask=extract_instances_cache(annotations["annotations/instances_train2014.json"],ids,train_infos,geometry,cache/"masks",progress=progress,split="train2014")
    val_mask=extract_instances_cache(annotations["annotations/instances_val2014.json"],val_row_ids,val_infos,geometry,cache/"masks",progress=progress,split="val2014")
    # Encode the fixed 80 category prompts with the exact frozen paired SigLIP text tower.
    category_path=cache/"category_text_features_fp16.npy"
    if not category_path.exists():
        model,tokenizer,meta=load_siglip_text_tower(device="cuda:0")
        prompts=[f"a photo of a {c['name']}" for c in categories]
        tokens=tokenizer(prompts)
        import torch
        with torch.inference_mode(): emb=model(tokens)
        if emb.shape!=(80,1152) or not torch.isfinite(emb).all(): raise RuntimeError(f"Category embeddings malformed: {tuple(emb.shape)}")
        np.save(category_path,emb.float().cpu().numpy().astype(np.float16),allow_pickle=False)
        atomic_json(a.output_dir/"category_prompt_metadata.json",{"prompts":prompts,"checkpoint_revision":SIGLIP_TEXT_MODEL_REVISION,"encoder":meta})
    else:
        mat=np.load(category_path,mmap_mode="r")
        if mat.shape!=(80,1152) or not np.isfinite(mat).all(): raise RuntimeError("Cached category embeddings invalid")
    atomic_json(a.status,{"stage":"preparation_complete","state":"complete","pid":os.getpid(),"train_images":len(ids),"train_captions":len(capids),"validation_images":len(val_row_ids),"development_images":1000,"evaluation_images":3000,"patch_manifests":{"train":train_manifest,"validation":val_manifest},"mask_manifests":{"train":train_mask,"validation":val_mask},"split_manifest":str(split_path),"category_text_features":str(category_path)})
    return split_path,cat_path

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset-root",type=Path,required=True);p.add_argument("--cache-root",type=Path,required=True);p.add_argument("--train-text-cache-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--previous-output",type=Path,required=True);p.add_argument("--status",type=Path,required=True);p.add_argument("--batch-size",type=int,default=4);prepare(p.parse_args())
if __name__=="__main__":main()
