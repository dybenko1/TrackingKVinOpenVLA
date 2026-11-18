#!/usr/bin/env python3
"""Resumable full COCO feature and projection-training pipeline.

Intended to run inside the validated persistent OpenVLA Docker image. Each
extractor already resumes from per-row completion flags; this supervisor
validates outputs and advances stages without touching prior pilot artifacts.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WORKTREE = Path(__file__).resolve().parents[4]
DATA = Path('/opt/openvla/cache/datasets/talk2dino/coco2014')
OUT = WORKTREE / 'experiments/kv_cache/outputs/trackb_talk2dino_siglip/full_coco_training_20261117'
FEATURES = Path('/opt/openvla/cache/datasets/talk2dino/coco2014/features_full_coco_20261117')
CHECKPOINT = Path('/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384/open_clip_model.safetensors')
LOCK = OUT / '.pipeline.lock'

def atomic(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)

def sha(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def stage(status, name, detail=None):
    status.update(stage=name, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), detail=detail)
    atomic(OUT/'status.json', status)
    print(f'PIPELINE_STAGE {name}: {detail or ""}', flush=True)

def run(cmd, log):
    print('RUN '+' '.join(map(str,cmd)), flush=True)
    with log.open('a') as f:
        p=subprocess.Popen(list(map(str,cmd)), cwd='/opt/openvla', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end='', flush=True); f.write(line); f.flush()
        code=p.wait()
    if code: raise RuntimeError(f'Command exit {code}: {cmd}')

def validate_features():
    import numpy as np
    expected={
      'image_ids_train2014.npy':(82783,),
      'region_features_train2014.npy':(82783,16,1024),
      'caption_ids_train2014.npy':(414113,2),
      'text_features_train2014_fp16.npy':(414113,1152),
    }
    report={'checked':{},'all_finite':True,'complete':True}
    for fn,shape in expected.items():
        p=FEATURES/fn
        if not p.is_file(): raise RuntimeError(f'Missing feature array {p}')
        a=np.load(p,mmap_mode='r')
        if tuple(a.shape)!=shape: raise RuntimeError(f'{fn} shape {a.shape}, expected {shape}')
        # Chunked finite check; do not load the full multi-GB arrays.
        finite=True
        for start in range(0,len(a),2048):
            if not np.isfinite(np.asarray(a[start:start+2048])).all(): finite=False; break
        report['checked'][fn]={'shape':list(a.shape),'dtype':str(a.dtype),'bytes':p.stat().st_size,'finite':finite}
        if not finite: raise RuntimeError(f'Non-finite values in {fn}')
    for fn,n in [('region_done_train2014.npy',82783),('text_done_train2014.npy',414113)]:
        p=FEATURES/fn
        a=np.load(p,mmap_mode='r')
        count=int(np.asarray(a).sum())
        report['checked'][fn]={'completed':count,'expected':n}
        if len(a)!=n or count!=n: raise RuntimeError(f'Incomplete flags: {fn} {count}/{n}')
    image_ids=np.load(FEATURES/'image_ids_train2014.npy',mmap_mode='r')
    if len(np.unique(np.asarray(image_ids)))!=82783: raise RuntimeError('Training image IDs are not unique')
    caps=np.load(FEATURES/'caption_ids_train2014.npy',mmap_mode='r')
    if len(np.unique(np.asarray(caps[:,0])))!=414113: raise RuntimeError('Caption annotation IDs are not unique')
    report.update(train_images=82783,train_captions=414113,integrity='passed')
    atomic(OUT/'FULL_FEATURE_VALIDATION.json',report)

def main():
    OUT.mkdir(parents=True,exist_ok=True); FEATURES.mkdir(parents=True,exist_ok=True)
    lockfd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(lockfd,str(os.getpid()).encode()); os.close(lockfd)
    status={'job':'trackb_talk2dino_full_20261117','status':'running','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'pid':os.getpid(),'host':'container'}
    atomic(OUT/'status.json',status)
    log=OUT/'logs/pipeline.log'; log.parent.mkdir(parents=True,exist_ok=True)
    try:
      stage(status,'preflight','CUDA, dataset, checkpoint, disk, and configuration')
      import torch
      if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in designated container')
      if not CHECKPOINT.is_file() or sha(CHECKPOINT)!='97faeafa0576c1ba4c92bae739ca06ef5f5d0c2db92f86a6086653d82a262043': raise RuntimeError('SigLIP-384 checkpoint checksum mismatch')
      train=json.loads((DATA/'annotations2014/captions_train2014.json').read_text())
      if len(train['images'])!=82783 or len(train['annotations'])!=414113: raise RuntimeError('COCO train counts do not match required full split')
      free=os.statvfs('/opt/openvla'); free_bytes=free.f_bavail*free.f_frsize
      if free_bytes<20*1024**3: raise RuntimeError(f'Insufficient free disk: {free_bytes}')
      atomic(OUT/'configuration.json',{'dataset':'COCO Captions 2014 train','images':82783,'captions':414113,'seed':123,'epochs':100,'batch_size':128,'optimizer':'Adam','lr':1e-4,'loss':'symmetric diagonal InfoNCE','temperature':0.07,'visual_backbone':'vit_large_patch14_reg4_dinov2.lvd142m','text_checkpoint':str(CHECKPOINT),'text_checkpoint_sha256':sha(CHECKPOINT),'feature_root':str(FEATURES),'container_image':'openvla-talk2dino-siglip:20261116'})
      py=Path('/opt/openvla/cache/worktrees/trackb-talk2dino-siglip/experiments/kv_cache/experiments/trackb_talk2dino_siglip')
      atomic(OUT/'logs/environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'cuda_free_bytes':torch.cuda.mem_get_info()[0],'disk_free_bytes':free_bytes,'train_images':len(train['images']),'train_captions':len(train['annotations'])})
      stage(status,'visual_features','resumable DINOv2 extraction for all unique train images')
      run([sys.executable,py/'extract_visual.py','--dataset-root',DATA,'--feature-root',FEATURES,'--split','train','--batch-size','1'],log)
      stage(status,'text_features','resumable SigLIP text extraction for all train captions')
      run([sys.executable,py/'extract_text.py','--dataset-root',DATA,'--feature-root',FEATURES,'--split','train','--batch-size','128'],log)
      stage(status,'feature_validation','validating dimensions, completion flags, IDs, and finiteness')
      validate_features()
      stage(status,'training','100-epoch caption-round training; atomic epoch checkpoints')
      run([sys.executable,py/'full_train.py','--feature-root',FEATURES,'--output',OUT,'--pilot-validation-features',WORKTREE/'experiments/kv_cache/outputs/trackb_talk2dino_siglip/pilot_1000/features','--epochs','100','--batch-size','128','--seed','123','--lr','0.0001'],log)
      status.update(status='complete',exit_code=0,completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
      stage(status,'complete','training and final checkpoint validation complete')
    except Exception as e:
      traceback.print_exc()
      status.update(status='failed',exit_code=1,error=str(e),failed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
      atomic(OUT/'status.json',status); print(f'PIPELINE_FAILED: {e}',flush=True)
      raise
    finally:
      try: LOCK.unlink()
      except FileNotFoundError: pass

if __name__=='__main__': main()
