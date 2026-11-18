"""Resumable full-COCO dense grounding training; only projection MLP is trainable."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from ..projection import RegionTextProjection
from .config import DenseV2Config
from .losses import dense_pair_losses, patch_logmeanexp_scores, symmetric_infonce
from .sampler import CaptionRoundSampler, choose_categories

def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)

def atomic_torch(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp); os.replace(tmp,path)

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

@torch.inference_mode()
def eval_dev(model, patches, occupancy, valid, rows, device, batch=16):
    thresholds=np.linspace(.05,.95,19,dtype=np.float32); c=occupancy.shape[1]
    inter=np.zeros((len(thresholds),c),np.float64); union=np.zeros_like(inter); gt_area=np.zeros(c,np.float64)
    small_i=np.zeros(len(thresholds),np.float64); small_u=np.zeros_like(small_i)
    cat_text=F.normalize(model.text_projection(model._category_text.float()),dim=-1)
    for st in range(0,len(rows),batch):
        rr=rows[st:st+batch]
        x=F.normalize(torch.from_numpy(np.array(patches[rr],dtype=np.float32,copy=True)).to(device),dim=-1)
        y=np.array(occupancy[rr],dtype=np.float32,copy=True); v=np.array(valid[rr],dtype=np.float32,copy=True)
        prob=torch.sigmoid(10*(torch.einsum("bpd,cd->bcp",x,cat_text)-.20)).cpu().numpy()
        y*=v[:,None,:]; gt_area+=y.sum((0,2)); small=(y.sum(-1)<4)
        for ti,t in enumerate(thresholds):
            pred=(prob>=t).astype(np.float32)*v[:,None,:]
            inter[ti]+=(pred*y).sum((0,2)); union[ti]+=(pred+y-pred*y).sum((0,2))
            sm=small[:,:,None]; small_i[ti]+=(pred*y*sm).sum(); small_u[ti]+=((pred+y-pred*y)*sm).sum()
    per=np.divide(inter,union,out=np.zeros_like(inter),where=union>0); active=gt_area>0
    macro=per[:,active].mean(1) if active.any() else np.zeros(len(thresholds))
    small=np.divide(small_i,small_u,out=np.zeros_like(small_i),where=small_u>0)
    best=max(range(len(thresholds)),key=lambda i:(macro[i],small[i],-abs(float(thresholds[i])-.5)))
    return {"thresholds":thresholds.tolist(),"macro_category_miou":macro.tolist(),"small_object_iou":small.tolist(),
            "best_threshold":float(thresholds[best]),"best_macro_category_miou":float(macro[best]),
            "best_small_object_iou":float(small[best]),"active_categories":int(active.sum()),
            "image_count":len(rows),"metric":"native 16x16 fractional-occupancy IoU; binary predictions at development-calibrated threshold"}

def run_training(a):
    cfg=DenseV2Config(batch_size=a.batch_size); seed=cfg.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root,out,ckdir=a.cache_root,a.output_dir,a.checkpoint_dir; out.mkdir(parents=True,exist_ok=True); ckdir.mkdir(parents=True,exist_ok=True)
    ids=np.load(root/"image_ids_train2014.npy",mmap_mode="r")
    capids=np.load(root/"caption_ids_train2014.npy",mmap_mode="r")
    text=np.load(root/"text_features_train2014_fp16.npy",mmap_mode="r")
    patches=np.load(root/"patch_features_train2014_fp16.npy",mmap_mode="r")
    occ=np.load(root/"masks/occupancy_train2014_fp16.npy",mmap_mode="r")
    valid=np.load(root/"masks/valid_train2014_fp16.npy",mmap_mode="r")
    vids=np.load(root/"image_ids_val2014.npy",mmap_mode="r")
    vpatch=np.load(root/"patch_features_val2014_fp16.npy",mmap_mode="r")
    vocc=np.load(root/"masks/occupancy_val2014_fp16.npy",mmap_mode="r")
    vvalid=np.load(root/"masks/valid_val2014_fp16.npy",mmap_mode="r")
    split=json.loads(a.split_manifest.read_text()); vrow={int(i):r for r,i in enumerate(vids)}
    dev_rows=np.asarray([vrow[int(i)] for i in split["development"]["image_ids"]],dtype=np.int64)
    if len(capids)!=414113 or len(ids)!=82783 or patches.shape!=(82783,256,1024) or vpatch.shape!=(4000,256,1024):
        raise RuntimeError("Full COCO feature counts/shapes do not match the frozen configuration")
    if not np.asarray(np.load(root/"patch_done_train2014.npy",mmap_mode="r")).all() or not np.asarray(np.load(root/"masks/mask_done_train2014.npy",mmap_mode="r")).all(): raise RuntimeError("Training features/masks incomplete")
    if not np.asarray(np.load(root/"patch_done_val2014.npy",mmap_mode="r")).all() or not np.asarray(np.load(root/"masks/mask_done_val2014.npy",mmap_mode="r")).all(): raise RuntimeError("Validation features/masks incomplete")
    row_by_id={int(i):r for r,i in enumerate(ids)}
    cap_payload=json.loads(a.captions_train.read_text()); ann_by_id={int(x["id"]):x for x in cap_payload["annotations"]}
    cap_iids=[]; cap_txtrows=[]
    for aid,iid in capids:
        aid=int(aid); iid=int(iid)
        if aid not in ann_by_id or int(ann_by_id[aid]["image_id"])!=iid: raise RuntimeError(f"Bad caption/image feature join for annotation {aid}")
        cap_iids.append(iid); cap_txtrows.append(aid)
    text_row_by_aid={int(aid):r for r,(aid,_) in enumerate(capids)}
    cap_txtrows=[text_row_by_aid[int(aid)] for aid,_ in capids]
    if len(cap_iids)!=414113: raise RuntimeError("Caption coverage failure")
    catinfo=json.loads(a.categories_json.read_text()); names=[c["name"] for c in catinfo]
    cat_text=np.load(root/"category_text_features_fp16.npy",mmap_mode="r")
    if cat_text.shape!=(len(names),1152) or occ.shape[1]!=len(names): raise RuntimeError("Category text/GT category order mismatch")
    freq=np.count_nonzero(np.asarray(occ).sum(-1)>0,axis=0)
    model=RegionTextProjection().to(device); model._category_text=torch.from_numpy(np.array(cat_text,dtype=np.float32,copy=True)).to(device)
    for p in model.parameters(): p.requires_grad_(True)
    init=ckdir/"initial_mlp.pt"
    if not init.exists(): atomic_torch(init,{"model":model.state_dict(),"seed":seed,"config":cfg.to_dict()})
    init_hash=sha256(init); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    per_image_counts=Counter(cap_iids)
    batches_per_epoch=sum((sum(n>r for n in per_image_counts.values())+cfg.batch_size-1)//cfg.batch_size for r in range(max(per_image_counts.values())))
    total_steps=batches_per_epoch*cfg.max_epochs; warmup=max(1,int(total_steps*cfg.warmup_fraction))
    epoch0=0; step=0; best={"miou":-1.,"small":-1.,"epoch":None,"threshold":.5}; stale=0; history=[]
    latest=ckdir/"latest.pt"
    if a.resume and latest.exists():
        q=torch.load(latest,map_location="cpu",weights_only=False); model.load_state_dict(q["model"],strict=True); opt.load_state_dict(q["optimizer"])
        epoch0=int(q["epoch"]);step=int(q["step"]);best=q["best"];stale=int(q["stale"]);history=q["history"]
        random.setstate(q["python_rng"]);np.random.set_state(q["numpy_rng"]);torch.set_rng_state(q["torch_rng"])
        if q["initial_sha256"]!=init_hash: raise RuntimeError("Resume initialization hash mismatch")
        if torch.cuda.is_available() and q.get("cuda_rng"): torch.cuda.set_rng_state_all(q["cuda_rng"])
    sampler=CaptionRoundSampler(cap_iids,cfg.batch_size,seed); start=time.time()
    if len(sampler)!=batches_per_epoch:raise RuntimeError(f"Sampler epoch length mismatch: {len(sampler)} vs {batches_per_epoch}")
    for epoch in range(epoch0,cfg.max_epochs):
        sampler.set_epoch(epoch); sums=Counter(); n_batches=0; visits=Counter(); batch_count=len(sampler); dense_pair_count=0; category_counts=np.zeros(len(names),dtype=np.int64)
        model.train()
        for batch_rows in sampler:
            iids=[cap_iids[i] for i in batch_rows]
            if len(set(iids))!=len(iids): raise RuntimeError("Repeated image ID inside InfoNCE batch")
            ir=np.asarray([row_by_id[i] for i in iids],dtype=np.int64)
            tr=np.asarray([cap_txtrows[i] for i in batch_rows],dtype=np.int64)
            x=F.normalize(torch.from_numpy(np.array(patches[ir],dtype=np.float32,copy=True)).to(device),dim=-1)
            txt=torch.from_numpy(np.array(text[tr],dtype=np.float32,copy=True)).to(device)
            caption_proj=model.text_projection(txt)
            scores=patch_logmeanexp_scores(x,caption_proj,cfg.pool_temperature)
            li=symmetric_infonce(scores,cfg.temperature)
            areas=np.array(occ[ir],dtype=np.float32,copy=True).sum(-1); candidates=areas>0; pairs=[]
            for bi,iid in enumerate(iids):
                visit=visits[iid];visits[iid]+=1
                selected=choose_categories(iid,np.flatnonzero(candidates[bi]),areas[bi],freq,epoch,visit,cfg.masks_per_image_visit,seed)
                pairs.extend((bi,int(ci)) for ci in selected)
                for ci in selected:category_counts[int(ci)]+=1
            dense_pair_count+=len(pairs)
            if pairs:
                bi=np.asarray([q[0] for q in pairs],np.int64);ci=np.asarray([q[1] for q in pairs],np.int64)
                ti=torch.as_tensor(ci,device=device)
                et=F.normalize(model.text_projection(model._category_text[ti]),dim=-1)
                logits=(x[torch.as_tensor(bi,device=device)]*et[:,None,:]).sum(-1)*10.-2.
                yy=np.stack([occ[ir[b],c].astype(np.float32) for b,c in pairs])
                vv=np.stack([valid[ir[b]].astype(np.float32) for b,c in pairs])
                dense=dense_pair_losses(logits[:,None,:],torch.from_numpy(yy).to(device)[:,None,:],torch.from_numpy(vv).to(device)[:,None,:],torch.from_numpy(yy.sum(-1)).to(device)[:,None],cfg.hard_negatives,cfg.occupancy_negative_cutoff)
            else:
                z=x.sum()*0;dense={"bce":z,"dice":z,"hard_negative":z}
            loss=.50*dense["bce"]+.50*dense["dice"]+.10*dense["hard_negative"]+.20*li
            if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite objective at epoch {epoch+1}, step {step}")
            lr=cfg.lr*(step+1)/warmup if step<warmup else cfg.min_lr+(cfg.lr-cfg.min_lr)*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total_steps-warmup))))
            for group in opt.param_groups: group["lr"]=lr
            opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step();step+=1
            for k,val in (("total",loss),("bce",dense["bce"]),("dice",dense["dice"]),("hard_negative",dense["hard_negative"]),("infonce",li),("grad_norm",gn)): sums[k]+=float(val.detach().cpu())
            n_batches+=1
            if n_batches%250==0: print(f"epoch={epoch+1} batches={n_batches}/{batch_count} loss={sums['total']/n_batches:.4f} lr={lr:.3g}",flush=True)
        metrics={k:sums[k]/max(1,n_batches) for k in ("total","bce","dice","hard_negative","infonce","grad_norm")}
        metrics.update({"epoch":epoch+1,"batches":n_batches,"captions_processed":len(cap_iids),"same_image_collisions":0,"duration_seconds":time.time()-start,"lr_end":lr,"dense_pairs":dense_pair_count,"category_coverage":int(np.count_nonzero(category_counts)),"category_pair_counts":{names[i]:int(x) for i,x in enumerate(category_counts) if x}})
        history.append(metrics); evaluation=None
        if epoch+1>=cfg.min_epochs and (epoch+1)%cfg.evaluate_every==0:
            model.eval();evaluation=eval_dev(model,vpatch,vocc,vvalid,dev_rows,device);model.train()
            score=evaluation["best_macro_category_miou"]; small=evaluation["best_small_object_iou"]
            improved=score>best["miou"]+cfg.improvement_tolerance
            tie=abs(score-best["miou"])<=cfg.improvement_tolerance and small>best["small"]
            if improved or tie:
                best={"miou":score,"small":small,"threshold":evaluation["best_threshold"],"epoch":epoch+1};stale=0
                atomic_torch(ckdir/"best_dev.pt",{"model":model.state_dict(),"epoch":epoch+1,"best":best,"config":cfg.to_dict(),"evaluation":evaluation,"initial_sha256":init_hash})
            else: stale+=1
        state={"epoch":epoch+1,"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step,"best":best,"stale":stale,"history":history,"config":cfg.to_dict(),"initial_sha256":init_hash,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        atomic_torch(latest,state);atomic_json(out/"training_history.json",{"config":cfg.to_dict(),"history":history,"best":best,"last_development_evaluation":evaluation})
        print(f"epoch={epoch+1}/{cfg.max_epochs} metrics={json.dumps(metrics)} dev={json.dumps(evaluation) if evaluation else None} best={best} stale={stale}",flush=True)
        if epoch+1>=cfg.min_epochs and stale>=cfg.patience_events: print("early stop: development patience reached",flush=True);break
    if not (ckdir/"best_dev.pt").exists(): raise RuntimeError("No checkpoint selected on development mIoU")
    (out/"best_projection.pt").write_bytes((ckdir/"best_dev.pt").read_bytes())
    atomic_json(out/"training_completion.json",{"status":"complete","epochs_completed":len(history),"best":best,"initial_sha256":init_hash,"best_checkpoint_sha256":sha256(out/"best_projection.pt"),"checkpoint":str(out/"best_projection.pt"),"elapsed_seconds":time.time()-start,"device":str(device)})

def main():
    p=argparse.ArgumentParser();p.add_argument("--cache-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--checkpoint-dir",type=Path,required=True);p.add_argument("--split-manifest",type=Path,required=True);p.add_argument("--categories-json",type=Path,required=True);p.add_argument("--captions-train",type=Path,required=True);p.add_argument("--batch-size",type=int,default=128);p.add_argument("--resume",action="store_true");run_training(p.parse_args())
if __name__=="__main__":main()
