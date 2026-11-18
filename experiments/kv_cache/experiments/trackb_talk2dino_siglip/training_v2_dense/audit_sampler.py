"""Full-scale preflight audit of the actual 414,113-caption round sampler."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .sampler import CaptionRoundSampler

def audit(path,batch_size=128,seed=123):
    ids=np.load(path,mmap_mode="r")
    if ids.shape!=(414113,2):raise RuntimeError(f"Expected [414113,2] caption IDs, got {ids.shape}")
    image_ids=ids[:,1].astype(np.int64,copy=False)
    sampler=CaptionRoundSampler(image_ids,batch_size,seed)
    hashes=[];seen=np.zeros(len(ids),dtype=np.uint8);batch_n=0;max_dupes=0
    for epoch in (0,1):
        sampler.set_epoch(epoch);h=hashlib.sha256();local_seen=np.zeros(len(ids),dtype=np.uint8);n=0
        for batch in sampler:
            if len({int(image_ids[i]) for i in batch})!=len(batch):raise RuntimeError(f"Repeated image ID at epoch {epoch}, batch {n}")
            local_seen[batch]+=1;h.update(np.asarray(batch,dtype=np.int32).tobytes());n+=1
        if not np.all(local_seen==1):raise RuntimeError(f"Caption coverage failed at epoch {epoch}")
        hashes.append(h.hexdigest());batch_n=n
    if hashes[0]==hashes[1]:raise RuntimeError("Epoch permutation did not change")
    sampler.set_epoch(0);h=hashlib.sha256()
    for batch in sampler:h.update(np.asarray(batch,dtype=np.int32).tobytes())
    if h.hexdigest()!=hashes[0]:raise RuntimeError("Sampler is not reproducible")
    return {"status":"passed","annotations":len(ids),"unique_images":int(np.unique(image_ids).size),"batch_size":batch_size,"batches_per_epoch":batch_n,"all_captions_once":True,"no_same_image_collision":True,"epoch_order_changes":True,"same_seed_reproducible":True,"epoch0_order_sha256":hashes[0],"epoch1_order_sha256":hashes[1]}

def main():
    p=argparse.ArgumentParser();p.add_argument("--caption-ids",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    out=audit(a.caption_ids);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
