"""Recompute corrected dev metrics for the saved V2 checkpoint; never trains."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch

from ..projection import RegionTextProjection
from .train import eval_dev


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--train-output",type=Path,required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--split-manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint=a.train_output/"best_projection.pt"
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=RegionTextProjection().to(device)
    model.load_state_dict(state["model"],strict=True)
    category_text=np.load(a.cache_root/"category_text_features_fp16.npy",mmap_mode="r")
    model._category_text=torch.from_numpy(np.array(category_text,dtype=np.float32,copy=True)).to(device)
    model.eval()
    split=json.loads(a.split_manifest.read_text())
    ids=np.load(a.cache_root/"image_ids_val2014.npy",mmap_mode="r")
    rows={int(i):r for r,i in enumerate(ids)}
    dev_rows=np.asarray([rows[int(i)] for i in split["development"]["image_ids"]],dtype=np.int64)
    result=eval_dev(model,
        np.load(a.cache_root/"patch_features_val2014_fp16.npy",mmap_mode="r"),
        np.load(a.cache_root/"masks/occupancy_val2014_fp16.npy",mmap_mode="r"),
        np.load(a.cache_root/"masks/valid_val2014_fp16.npy",mmap_mode="r"),
        dev_rows,device)
    checkpoints=sorted(x.name for x in a.checkpoint_dir.glob("*.pt"))
    completion=json.loads((a.train_output/"training_completion.json").read_text())
    result.update({"status":"complete","checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),
        "checkpoint_epoch":completion.get("best",{}).get("epoch"),"development_manifest_sha256":sha256(a.split_manifest),
        "development_images":len(dev_rows),"small_object_definition":"effective valid fractional occupancy > 0 and < 4 patch equivalents; absent categories excluded",
        "macro_definition":"pooled intersection/union per category over development images, then macro-averaged over categories with positive valid ground truth",
        "selection_caveat":"This recomputes corrected metrics for the saved checkpoint only. Earlier epoch model weights are unavailable, so it cannot retroactively reselect the checkpoint.",
        "checkpoint_files_available":checkpoints,"device":str(device)})
    atomic_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("thresholds","macro_category_miou","small_object_iou")},indent=2))
    print("best_threshold",result["best_threshold"],"best_macro_category_miou",result["best_macro_category_miou"],"corrected_small_object_iou",result["best_small_object_iou"])


if __name__=="__main__":
    main()
