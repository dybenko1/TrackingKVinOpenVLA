"""Frozen, metadata-only train/development/final split construction."""
from __future__ import annotations
import hashlib, json, random, zipfile
from pathlib import Path

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def ensure_instance_annotations(dataset_root:Path):
    ann=dataset_root/"annotations2014"; ann.mkdir(parents=True,exist_ok=True)
    zpath=dataset_root/"archives"/"annotations_trainval2014.zip"
    wanted=["annotations/instances_train2014.json","annotations/instances_val2014.json"]
    with zipfile.ZipFile(zpath) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f"COCO annotation archive CRC failed at {bad}")
        for member in wanted:
            target=ann/Path(member).name
            if not target.exists():
                tmp=target.with_suffix(target.suffix+".partial")
                with z.open(member) as src, tmp.open("wb") as dst:
                    while b:=src.read(8<<20): dst.write(b)
                tmp.replace(target)
            if target.stat().st_size != z.getinfo(member).file_size: raise RuntimeError(f"Bad extracted annotation size: {target}")
    return {x:ann/Path(x).name for x in wanted}

def _prior_ids(previous_output:Path):
    excluded=set(); hashes={}
    sp=previous_output/"postprocessing_optimization/full_val2014_holdout/segmentation_manifest.json"
    if sp.exists():
        data=json.loads(sp.read_text()); excluded|={int(x["image_id"]) for x in data["images"]}; hashes["previous_2500_segmentation_manifest"]=sha256_file(sp)
    p3=previous_output/"spatial_evaluation/evaluation_manifest.json"
    if p3.exists():
        data=json.loads(p3.read_text()); excluded|={int(x["image_id"]) for x in data.get("images",[])}; hashes["phase3_manifest"]=sha256_file(p3)
    return excluded,hashes

def create_split_manifest(captions_json:Path,instances_json:Path,previous_output:Path,out_path:Path,seed=20261118,dev_n=1000,eval_n=3000):
    if out_path.exists():
        return json.loads(out_path.read_text())
    caps=json.loads(captions_json.read_text()); inst=json.loads(instances_json.read_text())
    excluded,prior_hashes=_prior_ids(previous_output)
    infos={int(x["id"]):x for x in caps["images"]}
    strata={}; eligible=set()
    for a in inst["annotations"]:
        if a.get("iscrowd",0): continue
        iid=int(a["image_id"])
        if iid not in infos or iid in excluded: continue
        area=float(a.get("area",0)); image_area=max(1,int(infos[iid]["width"])*int(infos[iid]["height"]))
        if area<=0: continue
        ratio=area/image_area; size="small" if ratio<0.01 else "medium" if ratio<0.10 else "large"
        key=(int(a["category_id"]),size); strata.setdefault(key,set()).add(iid); eligible.add(iid)
    rng=random.Random(seed); queues={k:list(v) for k,v in strata.items()}
    for vals in queues.values(): rng.shuffle(vals)
    selected=[]; selected_set=set()
    keys=sorted(queues,key=lambda x:(x[0],x[1]))
    while len(selected)<dev_n+eval_n:
        changed=False
        for key in keys:
            q=queues[key]
            while q and q[-1] in selected_set: q.pop()
            if q:
                chosen=q.pop();selected.append(chosen);selected_set.add(chosen);changed=True
                if len(selected)>=dev_n+eval_n: break
        if not changed: break
    if len(selected)<dev_n+eval_n:
        rest=list(eligible-selected_set); rng.shuffle(rest); selected.extend(rest[:dev_n+eval_n-len(selected)])
    if len(selected)!=dev_n+eval_n: raise RuntimeError(f"Insufficient eligible val images: {len(selected)}")
    # Select according to annotation metadata only. Freeze before feature extraction or scoring.
    dev=selected[:dev_n]; evaluation=selected[dev_n:]
    cats=sorted(inst["categories"],key=lambda x:int(x["id"]))
    idx={"dataset":"COCO val2014 instance annotations","seed":seed,"selection_rule":"deterministic category x annotation-area-stratum round robin; previous Phase 3 and 2,500-image segmentation manifests excluded; no predictions/features inspected","excluded_ids":len(excluded),"excluded_manifest_sha256":prior_hashes,
         "annotation_sha256":{"captions":sha256_file(captions_json),"instances":sha256_file(instances_json)},
         "categories":[{"id":int(c["id"]),"name":c["name"]} for c in cats],
         "development":{"image_ids":dev,"image_count":len(dev)},"evaluation":{"image_ids":evaluation,"image_count":len(evaluation)},
         "disjoint":not bool(set(dev)&set(evaluation)),"all_eligible":len(eligible)}
    out_path.parent.mkdir(parents=True,exist_ok=True); tmp=out_path.with_suffix(".tmp"); tmp.write_text(json.dumps(idx,indent=2,sort_keys=True)+"\n"); tmp.replace(out_path)
    return idx
