#!/usr/bin/env python3
"""Exhaustive chunked multi-positive retrieval on disjoint COCO val2014 splits."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sys
REPO=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(REPO))
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
      for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

@torch.inference_mode()
def run_split(model, regions, image_ids, text, caption_ids, device, name, output, config_hash,
              scratch, image_chunk=32, caption_chunk=4096, temperature=.07):
    image_ids=np.asarray(image_ids,dtype=np.int64); caption_ids=np.asarray(caption_ids,dtype=np.int64)
    row_by_id={int(v):i for i,v in enumerate(image_ids)}
    cap_rows_by_image={}
    for j,iid in enumerate(caption_ids[:,1]): cap_rows_by_image.setdefault(int(iid),[]).append(j)
    if set(map(int,image_ids))!=set(cap_rows_by_image): raise RuntimeError(f'{name}: image/caption IDs not a complete one-to-many mapping')
    ni,nc=len(image_ids),len(caption_ids)
    # Project and normalize the text tower outputs once; save as a resumable cache.
    proj_path=scratch/f'projected_text_{name}_epoch{model._epoch}_fp32.npy'
    scratch.mkdir(parents=True,exist_ok=True)
    proj=np.lib.format.open_memmap(proj_path,mode='w+',dtype=np.float32,shape=(nc,1024))
    for s in range(0,nc,4096):
        t=torch.from_numpy(np.array(text[s:s+4096],dtype=np.float32,copy=True)).to(device)
        proj[s:s+len(t)]=F.normalize(model.text_projection(t),dim=-1).cpu().numpy()
    proj.flush()
    image_pos=np.empty(ni,np.float32); image_pos_logsum=np.empty(ni,np.float32)
    cap_pos=np.empty(nc,np.float32)
    # Positive image-caption scores; at most ~5 captions/image, so compute directly.
    for i,iid in enumerate(image_ids):
        rr=torch.from_numpy(np.array(regions[row_by_id[int(iid)]],dtype=np.float32,copy=True)).to(device)
        rr=F.normalize(rr,dim=-1)
        js=cap_rows_by_image[int(iid)]
        tt=torch.from_numpy(np.array(proj[js],dtype=np.float32,copy=True)).to(device)
        ps=(rr@tt.T).max(dim=0).values
        image_pos[i]=float(ps.max().cpu()); image_pos_logsum[i]=float(torch.logsumexp(ps/temperature,dim=0).cpu())
        cap_pos[js]=ps.cpu().numpy()
    rank_i=np.ones(ni,dtype=np.int64); rank_t=np.ones(nc,dtype=np.int64)
    den_i=torch.full((ni,),-torch.inf,dtype=torch.float64,device=device)
    den_t=torch.full((nc,),-torch.inf,dtype=torch.float64,device=device)
    sum_score=0.0; sum_sq=0.0; count=0; start=time.time()
    for ia in range(0,ni,image_chunk):
      ib=min(ni,ia+image_chunk)
      rr=torch.from_numpy(np.array(regions[ia:ib],dtype=np.float32,copy=True)).to(device)
      rr=F.normalize(rr,dim=-1)
      pi=torch.from_numpy(image_pos[ia:ib]).to(device)
      for ca in range(0,nc,caption_chunk):
        cb=min(nc,ca+caption_chunk)
        tt=torch.from_numpy(np.array(proj[ca:cb],dtype=np.float32,copy=True)).to(device)
        raw=torch.matmul(rr.reshape(-1,1024),tt.T).reshape(ib-ia,16,cb-ca)
        score=raw.max(dim=1).values
        if not torch.isfinite(score).all(): raise FloatingPointError(f'{name} nonfinite retrieval scores')
        rank_i[ia:ib]+=(score>pi[:,None]).sum(dim=1).cpu().numpy()
        pc=torch.from_numpy(cap_pos[ca:cb]).to(device)
        rank_t[ca:cb]+=(score>pc[None,:]).sum(dim=0).cpu().numpy()
        logits=score.double()/temperature
        den_i[ia:ib]=torch.logaddexp(den_i[ia:ib],torch.logsumexp(logits,dim=1))
        den_t[ca:cb]=torch.logaddexp(den_t[ca:cb],torch.logsumexp(logits,dim=0))
        sum_score+=float(score.double().sum().cpu()); sum_sq+=float(score.double().square().sum().cpu()); count+=score.numel()
      if (ia//image_chunk)%25==0 or ib==ni:
        atomic_json(output/f'retrieval_{name}_epoch{model._epoch}_progress.json',{'status':'running','image_rows_completed':ib,'images':ni,'caption_candidates_per_image':nc,'elapsed_seconds':time.time()-start})
        print(f'retrieval epoch={model._epoch} split={name} images={ib}/{ni} captions={nc}',flush=True)
    numerator_i=torch.as_tensor(image_pos_logsum,device=device,dtype=torch.float64)
    numerator_t=torch.as_tensor(cap_pos,device=device,dtype=torch.float64)/temperature
    loss_i=(den_i-numerator_i).mean(); loss_t=(den_t-numerator_t).mean()
    def recalls(rank): return {f'recall_at_{k}':float(np.mean(rank<=k)) for k in (1,5,10)}
    result={'status':'complete','split':name,'images':ni,'captions':nc,'candidate_pairs':ni*nc,
      'split_manifest_sha256':config_hash,'checkpoint_epoch':model._epoch,
      'image_to_text':{**recalls(rank_i),'rank_mean':float(rank_i.mean()),'rank_median':float(np.median(rank_i))},
      'text_to_image':{**recalls(rank_t),'rank_mean':float(rank_t.mean()),'rank_median':float(np.median(rank_t))},
      'multi_positive_infonce':{'image_to_text':float(loss_i.cpu()),'text_to_image':float(loss_t.cpu()),'symmetric_mean':float(((loss_i+loss_t)/2).cpu()),'temperature':temperature},
      'score_matrix_statistics':{'mean':sum_score/count,'std':max(0.0,sum_sq/count-(sum_score/count)**2)**.5,'pair_count':count},
      'rank_definition':'1 + number of all candidates with score strictly greater than the best positive (image->text), or matching-image positive (text->image); ties are treated as not outranking positive',
      'elapsed_seconds':time.time()-start,'projected_text_sha256':sha256(proj_path)}
    atomic_json(output/f'retrieval_{name}_epoch{model._epoch}.json',result)
    (output/f'retrieval_{name}_epoch{model._epoch}_progress.json').unlink(missing_ok=True)
    del proj
    proj_path.unlink(missing_ok=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--feature-root',type=Path,required=True); ap.add_argument('--split-manifest',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--scratch-dir',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--image-chunk',type=int,default=32); ap.add_argument('--caption-chunk',type=int,default=4096)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); dev=torch.device(a.device)
    if dev.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    split=json.loads(a.split_manifest.read_text()); image_ids=np.load(a.feature_root/'image_ids_full_val2014.npy',mmap_mode='r')
    regions=np.load(a.feature_root/'region_features_full_val2014_fp16.npy',mmap_mode='r')
    capids=np.load(a.feature_root/'caption_ids_val2014.npy',mmap_mode='r'); texts=np.load(a.feature_root/'text_features_val2014_fp16.npy',mmap_mode='r')
    if len(image_ids)!=40504 or len(capids)!=202654 or regions.shape!=(40504,16,1024) or texts.shape!=(202654,1152): raise RuntimeError('Full val cache shape/count integrity gate failed')
    if not np.asarray(np.load(a.feature_root/'region_done_full_val2014.npy',mmap_mode='r')).all() or not np.asarray(np.load(a.feature_root/'text_done_val2014.npy',mmap_mode='r')).all(): raise RuntimeError('Feature completion flags are incomplete')
    if not np.isfinite(regions).all() or not np.isfinite(texts).all(): raise FloatingPointError('Nonfinite full val features')
    splits={name:set(map(int,split[name]['image_ids'])) for name in ('development','evaluation')}
    if splits['development']&splits['evaluation'] or len(splits['development']|splits['evaluation'])!=40504: raise RuntimeError('Retrieval split is not a disjoint full partition')
    checkpoints={20:Path('/opt/openvla/experiments/kv_cache/outputs/trackb_talk2dino_siglip/full_coco_training_20261117/checkpoints/epoch_020.pt'),
      50:Path('/opt/openvla/experiments/kv_cache/outputs/trackb_talk2dino_siglip/full_coco_training_20261117/checkpoints/epoch_050.pt'),
      100:Path('/opt/openvla/experiments/kv_cache/outputs/trackb_talk2dino_siglip/full_coco_training_20261117/checkpoints/epoch_100.pt')}
    summaries={}; t=time.time()
    for epoch,path in checkpoints.items():
      digest=sha256(path); ck=torch.load(path,map_location='cpu',weights_only=False)
      if ck.get('completed_epoch')!=epoch: raise RuntimeError(f'Checkpoint epoch mismatch: {path}')
      model=RegionTextProjection(); model.load_state_dict(ck['model'],strict=True); model.to(dev).eval(); model._epoch=epoch
      cp={}
      for name,ids in splits.items():
        result_path=a.output_dir/f'retrieval_{name}_epoch{epoch}.json'
        if result_path.is_file():
          prior=json.loads(result_path.read_text())
          if prior.get('status')=='complete' and prior.get('split_manifest_sha256')==hashlib.sha256(a.split_manifest.read_bytes()).hexdigest() and prior.get('checkpoint_sha256')==digest:
            cp[name]=prior
            print(f'retrieval epoch={epoch} split={name}: verified saved result, skipping',flush=True)
            continue
        irows=np.flatnonzero(np.isin(image_ids, list(ids))); crows=np.flatnonzero(np.isin(capids[:,1],list(ids)))
        cp[name]=run_split(model,regions[irows],image_ids[irows],texts[crows],capids[crows],dev,name,a.output_dir,hashlib.sha256(a.split_manifest.read_bytes()).hexdigest(),a.scratch_dir,a.image_chunk,a.caption_chunk)
        cp[name]['checkpoint_sha256']=digest
        atomic_json(result_path,cp[name])
      summaries[str(epoch)]={'checkpoint':str(path),'sha256':digest,'completed_epoch':epoch,'development':cp['development'],'evaluation':cp['evaluation']}
      atomic_json(a.output_dir/'full_val2014_retrieval_summary.json',{'status':'running','epochs':summaries,'elapsed_seconds':time.time()-t})
    atomic_json(a.output_dir/'full_val2014_retrieval_summary.json',{'status':'complete','epochs':summaries,'images':40504,'captions':202654,'development_image_count':len(splits['development']),'evaluation_image_count':len(splits['evaluation']),'score_chunks':{'image':a.image_chunk,'caption':a.caption_chunk},'elapsed_seconds':time.time()-t,'device':str(dev),'torch':torch.__version__,'cuda':torch.version.cuda})
if __name__=='__main__': main()
