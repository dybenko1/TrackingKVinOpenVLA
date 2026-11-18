#!/usr/bin/env python3
"""Restartable full COCO MLP-only training with epoch-atomic checkpoints."""
from __future__ import annotations
import argparse, json, os, random, resource, time, hashlib
from pathlib import Path
import sys
import shutil
from types import SimpleNamespace
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.dataset import CocoAnnotationDataset, repeated_image_id_count
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.caption_round_sampler import CaptionRoundBatchSampler
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.losses import symmetric_infonce

def atomic_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2)+'\n'); os.replace(t,p)
def save_atomic(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp')
    with t.open('wb') as f: torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    os.replace(t,p)
def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch']);
    if torch.cuda.is_available() and s.get('cuda') is not None: torch.cuda.set_rng_state_all(s['cuda'])

@torch.inference_mode()
def retrieval(model, root, device):
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase2_pilot import multi_positive_metrics
    names=['image_ids_val2014.npy','region_features_val2014.npy','caption_ids_val2014.npy','text_features_val2014.npy']
    paths=[root/n for n in names]
    if not all(p.is_file() for p in paths): return {'status':'cached_validation_features_unavailable'}
    ids,regions,caps,texts=[np.load(p,mmap_mode='r') for p in paths]
    return multi_positive_metrics(model,regions,ids,texts,caps,device,temperature=.07)

def phase3_spatial_comparison(output: Path, full_checkpoint: Path):
    """Run the fixed historical Phase-3 manifest in a new, isolated folder."""
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip import phase3_evaluate as p3
    historical = p3.OUT
    manifest = historical / 'evaluation_manifest.json'
    if not manifest.is_file():
        raise RuntimeError(f'Required frozen Phase-3 manifest is missing: {manifest}')
    spatial = output / 'spatial_evaluation'
    spatial.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, spatial / 'evaluation_manifest.json')
    p3.OUT = spatial
    p3.CHECKPOINTS = {
      'untrained': p3.INITIAL,
      'original_pilot': p3.ORIGINAL,
      'phase2_corrected': p3.CORRECTED,
      'full_coco': full_checkpoint,
    }
    metrics = p3.evaluate(SimpleNamespace(device='cuda:0', pamr=False, max_eval_images=None))
    p3.atomic_json(spatial / 'full_coco_comparison.json', metrics)
    return {'status':'complete','output':str(spatial),'manifest_source':str(manifest),'checkpoints':list(p3.CHECKPOINTS)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--feature-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--pilot-validation-features',type=Path,required=True); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--seed',type=int,default=123); ap.add_argument('--lr',type=float,default=1e-4)
    a=ap.parse_args(); dev=torch.device('cuda')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    d=a.feature_root; ds=CocoAnnotationDataset(d/'image_ids_train2014.npy',d/'region_features_train2014.npy',d/'caption_ids_train2014.npy',d/'text_features_train2014_fp16.npy')
    if len(ds)!=414113 or ds.dropped_missing_images: raise RuntimeError(f'Expected all captions aligned; got {len(ds)}, missing={ds.dropped_missing_images}')
    image_ids=[row[1] for row in ds.caption_rows]; sampler=CaptionRoundBatchSampler(image_ids,a.batch_size,a.seed)
    model=RegionTextProjection(1152,1024,1024).to(dev); opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    if {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in model.parameters()}: raise RuntimeError('Optimizer includes non-MLP parameters')
    latest=a.output/'checkpoint_latest.pt'; history=[]; start_epoch=0
    if latest.exists():
      ck=torch.load(latest,map_location='cpu',weights_only=False)
      expected={'epochs':a.epochs,'batch_size':a.batch_size,'seed':a.seed,'lr':a.lr,'feature_root':str(d)}
      saved=ck.get('config',{})
      if any(saved.get(k)!=v for k,v in expected.items()): raise RuntimeError(f'Resume configuration mismatch: expected={expected}, saved={saved}')
      model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); start_epoch=int(ck['completed_epoch']); history=ck['history']; restore_rng(ck['rng'])
      print(f'Resuming at epoch {start_epoch+1} from {latest}',flush=True)
    else:
      initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; save_atomic(a.output/'initial_projection.pt',initial)
    config={'epochs':a.epochs,'batch_size':a.batch_size,'seed':a.seed,'lr':a.lr,'feature_root':str(d),'optimizer':'Adam','loss':'symmetric_diagonal_infonce','temperature':.07,'sampler':'caption_round_no_same_image_collisions_v1','training_rows':len(ds)}
    atomic_json(a.output/'FULL_TRAINING_CONFIG.json',config)
    for epoch in range(start_epoch,a.epochs):
      sampler.set_epoch(epoch); model.train(); total=0.; seen=0; batches=0; duplicates=0; t=time.time()
      loader=DataLoader(ds,batch_sampler=sampler,num_workers=0,pin_memory=True)
      for regions,texts,ids,_anns in loader:
        duplicates+=repeated_image_id_count(ids)
        regions=regions.to(dev,non_blocking=True); texts=texts.to(dev,non_blocking=True)
        opt.zero_grad(set_to_none=True); loss=symmetric_infonce(model(regions,texts),temperature=.07,legacy_extra_batch_square=False)
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at epoch {epoch+1}, batch {batches}')
        loss.backward(); opt.step(); total+=float(loss.detach())*len(ids); seen+=len(ids); batches+=1
        if batches%100==0:
          atomic_json(a.output/'status.json',{'status':'training','stage':'training','epoch':epoch+1,'epochs_total':a.epochs,'batches_completed':batches,'batches_estimate':len(sampler),'captions_processed_epoch':seen,'captions_total':len(ds),'latest_checkpoint':str(latest),'updated_unix':time.time()})
      if seen!=len(ds) or duplicates: raise RuntimeError(f'Sampler invariant failed: seen={seen}, expected={len(ds)}, duplicates={duplicates}')
      model.eval(); val=retrieval(model,a.pilot_validation_features,dev) if (epoch+1)%5==0 else None
      rec={'epoch':epoch+1,'train_loss':total/seen,'captions':seen,'batches':batches,'same_image_collisions':duplicates,'duration_seconds':time.time()-t,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(dev),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(dev),'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'validation_multi_positive':val}
      history.append(rec); model_cpu={k:v.detach().cpu() for k,v in model.state_dict().items()}
      ck={'model':model_cpu,'optimizer':opt.state_dict(),'completed_epoch':epoch+1,'rng':rng_state(),'config':config,'history':history,'torch':torch.__version__,'cuda':torch.version.cuda}
      save_atomic(latest,ck)
      if (epoch+1)%10==0: save_atomic(a.output/'checkpoints'/f'epoch_{epoch+1:03d}.pt',ck)
      atomic_json(a.output/'training_history.json',{'completed_epochs':epoch+1,'history':history})
      atomic_json(a.output/'status.json',{'status':'training','stage':'training','epoch':epoch+1,'epochs_total':a.epochs,'batches_completed':batches,'captions_processed_epoch':seen,'latest_checkpoint':str(latest),'last_train_loss':rec['train_loss'],'updated_unix':time.time()})
      print(json.dumps(rec),flush=True)
    final_path=a.output/'final_projection.pt'
    save_atomic(final_path,{k:v.detach().cpu() for k,v in model.state_dict().items()})
    del model, opt
    torch.cuda.empty_cache()
    spatial=phase3_spatial_comparison(a.output,final_path)
    atomic_json(a.output/'post_training_evaluation.json',spatial)
    print('FULL_TRAINING_AND_PHASE3_EVALUATION_COMPLETE',flush=True)
if __name__=='__main__': main()
