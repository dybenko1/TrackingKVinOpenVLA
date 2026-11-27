#!/usr/bin/env python3
"""Cache native OpenVLA DINO states needed for frozen E2 / trainable E3."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

WORKTREE = Path("/opt/openvla/cache/worktrees/trackb-dense-grounding-v2")
REPO = Path("/opt/openvla")
OUT = Path(os.environ.get("TRACKB_ADAPT_OUT", str(WORKTREE / "experiments/kv_cache/outputs/trackb_talk2dino_openvla_adaptation")))
CACHE = Path(os.environ.get("TRACKB_ADAPT_CACHE", "/opt/openvla/cache/datasets/talk2dino_openvla_adaptation_v1"))
MANIFEST = CACHE / "dataset_manifest.json"


def sha_bytes(x: bytes): return hashlib.sha256(x).hexdigest()


def atomic_npz(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f: np.savez_compressed(f,**arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp")
    with t.open('w') as f: json.dump(obj,f,indent=2,sort_keys=True,default=str); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(t,p)


def setup_imports():
    sys.path.insert(0,str(WORKTREE));sys.path.insert(0,str(REPO))
    import experiments.kv_cache
    experiments.kv_cache.__path__.append(str(WORKTREE/'experiments/kv_cache'))
    import experiments.kv_cache.experiments
    experiments.kv_cache.experiments.__path__.append(str(WORKTREE/'experiments/kv_cache/experiments'))
    from experiments.kv_cache.experiments.trackb_talk2dino_native import run_libero_e2 as native
    return native


def build_model():
    import torch
    native=setup_imports()
    model,processor,vision,dino,geometry,meta=native.openvla_model(torch.device('cuda:0'))
    model.eval();dino.eval()
    return native,model,processor,vision,dino,geometry,meta,torch


def process_batch(images,processor,dino,device,torch):
    pixels=processor.image_processor(images=[Image.fromarray(np.asarray(x,dtype=np.uint8),mode='RGB') for x in images],return_tensors='pt')['pixel_values']
    x=pixels[:,:3].to(device=device,dtype=next(dino.parameters()).dtype)
    if tuple(x.shape[1:])!=(3,224,224): raise RuntimeError(f'Unexpected OpenVLA DINO tensor {tuple(x.shape)}')
    return x


def hook_features(dino,pixels,native,torch):
    from experiments.kv_cache.experiments.trackb_talk2dino_native.model import TimmTalk2DINOHook
    with torch.inference_mode(), TimmTalk2DINOHook(dino) as hook:
        result=hook.extract_from_one_normal_pass(pixels)
    if hook.module_calls!=1 or hook.block_calls != [1]*24:
        raise RuntimeError(f'DINO execution counts invalid: {hook.module_calls}/{hook.block_calls}')
    # The shared native helper intentionally returns [heads, patches] for its
    # single-frame evaluator. For batched extraction, reconstruct the attention
    # for every image from the QKV captured during this same normal pass.
    result['attention_logits']=batched_attention_from_qkv(dino,hook.records['qkv24'],pixels.shape[0],torch)
    return result


def batched_attention_from_qkv(dino,qkv,batch,torch):
    attn=dino.blocks[-1].attn
    heads=int(attn.num_heads);head_dim=int(dino.embed_dim)//heads
    b,n,width=qkv.shape
    if b!=batch or width!=3*heads*head_dim: raise RuntimeError(f'Unexpected block24 QKV shape {tuple(qkv.shape)}')
    q,k=qkv.reshape(b,n,3,heads,head_dim).permute(2,0,3,1,4)[:2]
    if getattr(attn,'q_norm',None) is not None:q=attn.q_norm(q)
    if getattr(attn,'k_norm',None) is not None:k=attn.k_norm(k)
    logits=(q.float()*float(attn.scale))@k.float().transpose(-2,-1)
    patches=logits[:,:,0,5:]
    if tuple(patches.shape)!=(batch,16,256):raise RuntimeError(f'Unexpected batched attention shape {tuple(patches.shape)}')
    return patches


def rebuild_cached_attention(dino,block23,torch,batch_size=8):
    """Recompute only block-24 attention from saved exact block-23 sequences."""
    block=dino.blocks[-1];device=next(dino.parameters()).device;dtype=next(dino.parameters()).dtype
    captures=[]
    for start in range(0,len(block23),batch_size):
        seq=torch.as_tensor(block23[start:start+batch_size],device=device,dtype=dtype)
        qkv=[];handle=block.attn.qkv.register_forward_hook(lambda _m,_i,o:qkv.append(o))
        try:
            with torch.inference_mode():block(seq)
        finally:handle.remove()
        if len(qkv)!=1:raise RuntimeError(f'Expected one QKV capture while repairing cache, got {len(qkv)}')
        captures.append(batched_attention_from_qkv(dino,qkv[0],len(seq),torch).cpu().numpy().astype(np.float16))
    out=np.concatenate(captures,axis=0) if captures else np.empty((0,16,256),np.float16)
    if out.shape!=(len(block23),16,256):raise RuntimeError(f'Repaired attention cache has shape {out.shape}')
    return out


def feature_file_for(record):
    task=record.get('task_id','benchmark')
    slug=re.sub(r'[^A-Za-z0-9_-]','_',task)
    split=record.get('split','benchmark')
    demo=record.get('demo','segment')
    suffix=f"{demo}_{record.get('start','')}_{record.get('end','')}".strip('_')
    return CACHE/'features'/split/slug/f'{suffix}.npz'


def cache_prepared_dataset(native,processor,dino,device,torch,manifest,smoke=False):
    rows=[(task,split,rec) for task in manifest['tasks'] for split,items in task['splits'].items() for rec in items]
    if smoke: rows=rows[:1]
    total=sum(rec['frame_count'] for _,_,rec in rows)
    done=0
    status_path=OUT/'stage_progress.json'
    for task,split,rec in rows:
        src=Path(rec['cache']); out=CACHE/'features'/split/re.sub(r'[^A-Za-z0-9_-]','_',task['task_id'])/f"{rec['demo']}.npz"
        if out.exists():
            try:
                with np.load(out,allow_pickle=False) as z:
                    if z['block23_tokens'].shape==(rec['frame_count'],261,1024) and str(z['source_hdf5_sha256'].item())==task['hdf5_sha256']:
                        if z['attention_logits'].shape!=(rec['frame_count'],16,256):
                            arrays={k:np.asarray(z[k]) for k in z.files}
                            arrays['attention_logits']=rebuild_cached_attention(dino,arrays['block23_tokens'],torch)
                            atomic_npz(out,**arrays)
                            print(f'[ATTENTION_REPAIR] {task["task_id"]}:{rec["demo"]} rows={len(arrays["attention_logits"])}',flush=True)
                        done+=rec['frame_count']; print(f'[FEATURE_CACHE] {done}/{total} {task["task_id"]}:{rec["demo"]}',flush=True); continue
            except Exception: pass
        with np.load(src,allow_pickle=False) as z: rgbs=np.asarray(z['rgb'],dtype=np.uint8)
        if smoke: rgbs=rgbs[:1]
        final=[];block23=[];attn=[]
        for offset in range(0,len(rgbs),2):
            images=rgbs[offset:offset+2]; x=process_batch(images,processor,dino,device,torch)
            result=hook_features(dino,x,native,torch)
            final.extend(result['final_patch_features'].float().cpu().numpy().astype(np.float16))
            block23.extend(result['block23_tokens'].float().cpu().numpy().astype(np.float16))
            attn.extend(result['attention_logits'].float().cpu().numpy().astype(np.float16))
        atomic_npz(out,final_features=np.asarray(final,np.float16),block23_tokens=np.asarray(block23,np.float16),
                   attention_logits=np.asarray(attn,np.float16),source_hdf5_sha256=np.asarray(task['hdf5_sha256']),
                   task_id=np.asarray(task['task_id']),demo=np.asarray(rec['demo']),split=np.asarray(split),
                   openvla_revision=np.asarray('962318cec55ac10993ff0f5f43eda9a270b4c873'),
                   feature_definition=np.asarray('one normal OpenVLA TIMM DINO pass; block23 full 261-token sequence; block24 norm patch features; block24 CLS-to-patch QK'))
        done+=len(rgbs);print(f'[FEATURES] {done}/{total} {task["task_id"]}:{rec["demo"]}',flush=True)
        atomic_json(status_path,{'status':'running','stage':'feature_cache_training_demos','completed_frames':done,'total_frames':total,
            'current_task':task['task_id'],'current_demo':rec['demo'],'updated_unix':time.time(),'error':None})


def cache_benchmark(native,processor,dino,device,torch,smoke=False):
    runner_path=WORKTREE/'experiments/kv_cache/experiments/trackb_talk2dino_openvla_multicategory/run_experiment.py'
    spec=importlib.util.spec_from_file_location('adaptation_baseline_runner',runner_path)
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    segments,manifest,manifest_sha=runner.input_segments()
    if smoke: segments=segments[:1]
    outroot=CACHE/'features'/'benchmark';outroot.mkdir(parents=True,exist_ok=True)
    for seg_i,seg in enumerate(segments):
        out=outroot/f"{re.sub(r'[^A-Za-z0-9_-]','_',seg['task_id'])}.npz"
        if out.exists():
            try:
                with np.load(out,allow_pickle=False) as z:
                    if z['block23_tokens'].shape==(10,261,1024) and str(z['source_score_sha256'].item())==seg['sha256']:
                        if z['attention_logits'].shape!=(10,16,256):
                            arrays={k:np.asarray(z[k]) for k in z.files}
                            arrays['attention_logits']=rebuild_cached_attention(dino,arrays['block23_tokens'],torch)
                            atomic_npz(out,**arrays)
                            print(f'[ATTENTION_REPAIR] benchmark:{seg["task_id"]} rows={len(arrays["attention_logits"])}',flush=True)
                        print(f'[BENCH_CACHE] {seg["task_id"]}',flush=True);continue
            except Exception: pass
        rgb=seg['arrays']['rgb']
        if smoke: rgb=rgb[:1]
        final=[];block23=[];attn=[]
        for offset in range(0,len(rgb),2):
            x=process_batch(rgb[offset:offset+2],processor,dino,device,torch)
            result=hook_features(dino,x,native,torch)
            final.extend(result['final_patch_features'].float().cpu().numpy().astype(np.float16))
            block23.extend(result['block23_tokens'].float().cpu().numpy().astype(np.float16))
            attn.extend(result['attention_logits'].float().cpu().numpy().astype(np.float16))
        atomic_npz(out,final_features=np.asarray(final,np.float16),block23_tokens=np.asarray(block23,np.float16),
                   attention_logits=np.asarray(attn,np.float16),raw=seg['arrays']['raw'][:len(rgb)],clean=seg['arrays']['clean'][:len(rgb)],
                   relevant_gt=seg['arrays']['relevant_gt'][:len(rgb)],role_gt=seg['arrays']['role_gt'][:len(rgb)],valid=seg['arrays']['valid'][:len(rgb)],
                   rgb=rgb,queries_json=np.asarray(json.dumps(seg['queries'],sort_keys=True)),role_keys_json=np.asarray(json.dumps(seg['role_keys'])),
                   task_id=np.asarray(seg['task_id']),suite=np.asarray(seg['suite']),demo=np.asarray(seg['demo']),split=np.asarray(seg['split']),
                   source_score_sha256=np.asarray(seg['sha256']),source_manifest_sha256=np.asarray(manifest_sha))
        print(f'[BENCH_FEATURES] {seg["task_id"]} frames={len(rgb)}',flush=True)
        atomic_json(OUT/'stage_progress.json',{'status':'running','stage':'feature_cache_benchmark','completed_tasks':seg_i+1,
            'total_tasks':len(segments),'current_task':seg['task_id'],'updated_unix':time.time(),'error':None})


def run(smoke=False):
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; refusing CPU fallback')
    if not MANIFEST.exists(): raise FileNotFoundError(MANIFEST)
    manifest=json.loads(MANIFEST.read_text())
    if manifest.get('status')!='complete': raise RuntimeError('Simulator label manifest is incomplete')
    native,model,processor,vision,dino,geometry,meta,torch=build_model()
    device=torch.device('cuda:0'); torch.cuda.reset_peak_memory_stats(device)
    # One-image integration smoke: standard output and captured block23 are
    # bit-identical and cached block23 -> block24 reproduces final patch output.
    first_task=manifest['tasks'][0]; first_rec=first_task['splits']['train'][0]
    with np.load(first_rec['cache'],allow_pickle=False) as z: sample=z['rgb'][:1]
    x=process_batch(sample,processor,dino,device,torch)
    with torch.inference_mode():
        plain=dino(x)
        captured=hook_features(dino,x,native,torch)
        final_again=dino.norm(dino.blocks[-1](captured['block23_tokens']))[:,5:,:]
    plain=plain[0] if isinstance(plain,(tuple,list)) else plain
    delta_policy=float((plain.float()-captured['policy_penultimate'].float()).abs().max())
    delta_final=float((final_again.float()-captured['final_patch_features'].float()).abs().max())
    tol=2e-3 if x.dtype in (torch.float16,torch.bfloat16) else 1e-5
    if tuple(captured['block23_tokens'].shape)!=(1,261,1024) or tuple(captured['final_patch_features'].shape)!=(1,256,1024): raise RuntimeError('Feature shape smoke failed')
    if delta_policy>tol or delta_final>tol: raise RuntimeError(f'Feature parity smoke failed policy={delta_policy}, final={delta_final}, tol={tol}')
    atomic_json(OUT/'feature_extraction_smoke.json',{'status':'passed','native_dino_revision':meta.get('revision'),'policy_block23_shape':list(plain.shape),
        'block23_sequence_shape':list(captured['block23_tokens'].shape),'final_patch_shape':list(captured['final_patch_features'].shape),
        'attention_shape':list(captured['attention_logits'].shape),'policy_max_abs_delta':delta_policy,'block23_to_block24_max_abs_delta':delta_final,
        'tolerance':tol,'one_forward_24_block_calls':True,'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device))})
    print(f'[SMOKE_PASS] policy_delta={delta_policy:.3g} block23_to_block24_delta={delta_final:.3g}',flush=True)
    cache_prepared_dataset(native,processor,dino,device,torch,manifest,smoke=smoke)
    cache_benchmark(native,processor,dino,device,torch,smoke=smoke)
    atomic_json(OUT/'feature_cache_validation.json',{'status':'complete','dataset_manifest_sha256':manifest['manifest_sha256'],
        'openvla_revision':meta.get('revision'),'feature_arrays':['block23_tokens [B,261,1024] fp16','final_features [B,256,1024] fp16','attention_logits [B,16,256] fp16'],
        'source_processing':'OpenVLA processor.image_processor, first three DINO channels, 224x224','benchmark_manifest_sha256':json.loads((WORKTREE/'experiments/kv_cache/outputs/trackb_talk2dino_openvla_multicategory/frame_manifest.json').read_text())['source_manifest_sha256'],
        'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device))})


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.smoke)
if __name__=='__main__':main()
