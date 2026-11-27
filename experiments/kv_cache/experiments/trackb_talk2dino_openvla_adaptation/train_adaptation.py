#!/usr/bin/env python3
"""E2 frozen-feature MLP and E3 final-block + MLP task-relevance adaptation."""
from __future__ import annotations

import argparse, copy, csv, hashlib, importlib.util, json, os, random, re, sys, time
from pathlib import Path

import numpy as np

WORKTREE=Path('/opt/openvla/cache/worktrees/trackb-dense-grounding-v2')
REPO=Path('/opt/openvla')
OUT=Path(os.environ.get('TRACKB_ADAPT_OUT',str(WORKTREE/'experiments/kv_cache/outputs/trackb_talk2dino_openvla_adaptation')))
CACHE=Path(os.environ.get('TRACKB_ADAPT_CACHE','/opt/openvla/cache/datasets/talk2dino_openvla_adaptation_v1'))
DATA_MANIFEST=CACHE/'dataset_manifest.json'
BENCH_CACHE=CACHE/'features/benchmark'
TEMPLATES=('itap of a {}.','a bad photo of a {}.','a origami {}.','a photo of the large {}.','a {} in a video game.','art of the {}.','a photo of the small {}.')
SEED=123


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(obj,f,indent=2,sort_keys=True,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def atomic_torch_save(torch,path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    torch.save(obj,t)
    with t.open('rb') as f:os.fsync(f.fileno())
    os.replace(t,p)

def write_history_csv(path,history):
    import csv
    rows=[]
    for item in history:
        row={k:v for k,v in item.items() if k!='dev_metrics'}
        for post,metrics in item.get('dev_metrics',{}).items():
            op=metrics.get('operating_points',{}).get('0.98',{})
            vals=op.get('metrics',{})
            row.update({f'dev_{post}_tau98':op.get('threshold'),f'dev_{post}_recall98':vals.get('fg_recall'),
                        f'dev_{post}_bgcoverage98':vals.get('bg_coverage'),f'dev_{post}_bgprecision98':vals.get('bg_precision')})
        rows.append(row)
    if not rows:return
    p=Path(path);tmp=p.with_suffix('.csv.tmp');fields=list(rows[0])
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def setup_imports():
    sys.path.insert(0,str(WORKTREE));sys.path.insert(0,str(REPO))
    import experiments.kv_cache
    experiments.kv_cache.__path__.append(str(WORKTREE/'experiments/kv_cache'))
    import experiments.kv_cache.experiments
    experiments.kv_cache.experiments.__path__.append(str(WORKTREE/'experiments/kv_cache/experiments'))
    from experiments.kv_cache.experiments.trackb_talk2dino_native import run_libero_e2 as native
    return native

def load_base():
    import torch
    native=setup_imports()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU substitution')
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);torch.cuda.manual_seed_all(SEED)
    model,processor,vision,dino,geometry,meta=native.openvla_model(torch.device('cuda:0'))
    clip,clip_meta=native.load_clip_tower(torch.device('cuda:0'))
    projection,projection_meta=native.load_projection(torch.device('cuda:0'))
    for p in clip.parameters():p.requires_grad_(False)
    model.eval();dino.eval();clip.eval()
    return native,model,processor,vision,dino,clip,projection,meta,clip_meta,projection_meta,torch

def query_embeddings(task,queries,clip,native,device,torch):
    phrases=[q['phrase'] for q in queries]
    prompts=[t.format(s) for s in phrases for t in TEMPLATES]
    tokens=native.official_clip_tokens(prompts).to(device)
    with torch.inference_mode():
        e=clip.encode_text(tokens).float()
        if e.shape!=(len(phrases)*len(TEMPLATES),512) or not torch.isfinite(e).all():raise RuntimeError('Invalid CLIP text features')
        e=e.view(len(phrases),len(TEMPLATES),512).mean(1)
    # Inference tensors cannot be saved by the trainable projection's Linear
    # layer for backward, even though the text encoder itself is frozen.
    return e.clone()

def slug(s):return re.sub(r'[^A-Za-z0-9_-]','_',s)

def load_groups(split,torch):
    manifest=json.loads(DATA_MANIFEST.read_text())
    groups=[]
    for task in manifest['tasks']:
        records=task['splits'].get(split,[])
        if not records:continue
        data_rows=[];features=[];attention=[];blocks=[];queries=None
        for rec in records:
            with np.load(rec['cache'],allow_pickle=False) as z:
                data_rows.append({k:np.asarray(z[k]) for k in ('gt','relevant','valid')})
                if queries is None:queries=json.loads(str(z['queries_json'].item()))
                elif queries!=json.loads(str(z['queries_json'].item())):raise RuntimeError('Task prompt list changed across demos')
            fp=CACHE/'features'/split/slug(task['task_id'])/f"{rec['demo']}.npz"
            if not fp.is_file():raise FileNotFoundError(f'Feature cache missing: {fp}')
            with np.load(fp,allow_pickle=False) as z:
                features.append(np.asarray(z['final_features'],np.float16));attention.append(np.asarray(z['attention_logits'],np.float16));blocks.append(np.asarray(z['block23_tokens'],np.float16))
        group={'task_id':task['task_id'],'suite':task['suite'],'instruction':task['instruction'],'queries':queries,
               'text_base':None,'gt':torch.from_numpy(np.concatenate([r['gt'] for r in data_rows]).astype(np.float32)),
               'relevant':torch.from_numpy(np.concatenate([r['relevant'] for r in data_rows]).astype(np.float32)),
               'valid':torch.from_numpy(np.concatenate([r['valid'] for r in data_rows]).astype(np.float32)),
               'features':torch.from_numpy(np.concatenate(features).astype(np.float16)),
               'attention':torch.from_numpy(np.concatenate(attention).astype(np.float16)),
               'block23':torch.from_numpy(np.concatenate(blocks).astype(np.float16))}
        if len(group['features'])!=len(group['gt']):raise RuntimeError('Label and feature row counts differ')
        n=len(group['gt'])
        if tuple(group['features'].shape)!=(n,256,1024) or tuple(group['attention'].shape)!=(n,16,256) or tuple(group['block23'].shape)!=(n,261,1024):
            raise RuntimeError(f"Cached feature alignment invalid for {group['task_id']}: features={tuple(group['features'].shape)} attention={tuple(group['attention'].shape)} block23={tuple(group['block23'].shape)} labels={n}")
        groups.append(group)
    if not groups:raise RuntimeError(f'No groups in split={split}')
    return groups,manifest

def encode_group_text(groups,clip,native,device,torch):
    for g in groups:
        path=CACHE/'text_features'/f"{slug(g['task_id'])}.npz"
        prompts=[t.format(q['phrase']) for q in g['queries'] for t in TEMPLATES]
        if path.exists():
            try:
                with np.load(path,allow_pickle=False) as z:
                    if str(z['task_id'].item())==g['task_id'] and json.loads(str(z['queries_json'].item()))==g['queries']:
                        g['text_base']=torch.from_numpy(np.asarray(z['features'],np.float32));continue
            except Exception:pass
        emb=query_embeddings(g['task_id'],g['queries'],clip,native,device,torch)
        arr=emb.detach().float().cpu().numpy().astype(np.float32)
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,features=arr,task_id=np.asarray(g['task_id']),queries_json=np.asarray(json.dumps(g['queries'],sort_keys=True)),
            prompts_json=np.asarray(json.dumps(prompts)),clip_sha256=np.asarray('5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'));f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);g['text_base']=torch.from_numpy(arr)

def final_from_block23(block24,norm,seq):
    # OpenVLA loads DINO in BF16. Keep a detached FP32 sidecar copy for E3 so
    # the 1e-6 block learning rate is represented by the optimizer instead of
    # being rounded away in BF16 parameter storage. Initial values are exact
    # BF16->FP32 casts; policy DINO parameters remain untouched.
    h=block24(seq.float())
    return norm(h)[:,5:,:]

def raw_logits(projection,text_base,visual,torch):
    import torch.nn.functional as F
    t=F.normalize(projection(text_base.float()).float(),dim=-1)
    v=F.normalize(visual.float(),dim=-1)
    return torch.einsum('qd,bpd->bqp',t,v)

def dense_loss(logits,y,valid,torch):
    import torch.nn.functional as F
    y=y.float();v=valid.float().unsqueeze(1).expand_as(y)
    present=((y*v).sum(-1)>1e-6)
    if not present.any():return logits.sum()*0.0,{'bce':0.0,'dice':0.0,'positive_cosine':float('nan'),'negative_cosine':float(logits.detach().mean())}
    bce=F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none')
    weight=v*(1.0+3.0*y)
    numerator=(bce*weight*present.unsqueeze(-1)).sum()
    denominator=(weight*present.unsqueeze(-1)).sum().clamp_min(1.0)
    bce_loss=numerator/denominator
    prob=logits.float().sigmoid()
    inter=(prob*y*v).sum(-1)
    denom=((prob+y)*v).sum(-1)
    dice=1.0-(2.0*inter+1.0)/(denom+1.0)
    dice_loss=(dice*present).sum()/present.sum().clamp_min(1)
    loss=0.7*bce_loss+0.3*dice_loss
    positive=(logits.detach()[y>0.5].mean() if (y>0.5).any() else logits.detach().new_tensor(float('nan')))
    negative=(logits.detach()[y<0.02].mean() if (y<0.02).any() else logits.detach().mean())
    return loss,{'bce':float(bce_loss.detach()),'dice':float(dice_loss.detach()),'positive_cosine':float(positive),'negative_cosine':float(negative)}

def calc_metrics(score,y,valid,tau):
    p=np.asarray(score)>=float(tau);y=np.asarray(y,dtype=np.float64);v=np.asarray(valid,dtype=np.float64)
    tp=float((p*y*v).sum());fp=float((p*(1-y)*v).sum());fn=float(((~p)*y*v).sum());tn=float(((~p)*(1-y)*v).sum())
    d=lambda a,b:a/b if b else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'fg_precision':d(tp,tp+fp),'fg_recall':d(tp,tp+fn),'fg_iou':d(tp,tp+fp+fn),
        'bg_precision':d(tn,tn+fn),'bg_recall':d(tn,tn+fp),'bg_iou':d(tn,tn+fp+fn),
        'bg_coverage':d(tn+fn,tp+fp+fn+tn),'fg_coverage':d(tp+fp,tp+fp+fn+tn),'foreground_as_background':d(fn,tp+fn)}

def choose_tau(score,y,valid,target):
    s=np.asarray(score,dtype=np.float32).ravel(); yy=(np.asarray(y,dtype=np.float64)*np.asarray(valid,dtype=np.float64)).ravel(); vv=np.asarray(valid,dtype=np.float64).ravel()
    mask=vv>0; s=s[mask]; yy=yy[mask]; total=float(yy.sum())
    if total<=0:return {'threshold':0.5,'constraint_met':False,'metrics':calc_metrics(score,y,valid,.5)}
    order=np.argsort(s)[::-1]; cs=np.cumsum(yy[order]); need=target*total
    idx=int(np.searchsorted(cs,need,side='left'))
    if idx>=len(order):tau=float(np.min(s)); met=False
    else:tau=float(s[order[idx]]);met=True
    return {'threshold':tau,'constraint_met':met,'metrics':calc_metrics(score,y,valid,tau)}

def aggregate_score(groups,scores_by_task,key,threshold):
    a=np.zeros(4,np.float64)
    for g in groups:
        score=scores_by_task[g['task_id']][key]
        y=g['relevant'].numpy();v=g['valid'].numpy();p=score>=threshold
        a+=np.asarray([(p*y*v).sum(),(p*(1-y)*v).sum(),((~p)*y*v).sum(),((~p)*(1-y)*v).sum()])
    tp,fp,fn,tn=a;d=lambda n,x:n/x if x else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'fg_recall':d(tp,tp+fn),'fg_precision':d(tp,tp+fp),'bg_precision':d(tn,tn+fn),
            'bg_recall':d(tn,tn+fp),'bg_iou':d(tn,tn+fp+fn),'fg_iou':d(tp,tp+fp+fn),'bg_coverage':d(tn+fn,tp+fp+fn+tn),
            'fg_coverage':d(tp+fp,tp+fp+fn+tn),'foreground_as_background':d(fn,tp+fn)}

def score_clean_batch(projection,text_base,visual,attention,torch):
    import torch.nn.functional as F
    projected=F.normalize(projection(text_base.float()).float(),dim=-1)
    v=visual.float();vn=F.normalize(v,dim=-1)
    cosine=torch.einsum('qd,bpd->bqp',projected,vn);raw=cosine.sigmoid()
    att=attention.float()
    regions=(att.unsqueeze(-1)*v.unsqueeze(1)).mean(dim=2)
    regions=F.normalize(regions,dim=-1)
    weights=torch.einsum('qd,bhd->bqh',projected,regions).softmax(dim=-1)
    a=torch.einsum('bqh,bhp->bqp',weights,att)
    a=a.softmax(dim=-1)
    amin,amax=a.amin(dim=(1,2),keepdim=True),a.amax(dim=(1,2),keepdim=True)
    scaled=(a-amin)/torch.maximum(amax,amax-amin).clamp_min(1e-12)
    rmin,rmax=raw.amin(dim=(1,2),keepdim=True),raw.amax(dim=(1,2),keepdim=True)
    scaled=scaled*(rmax-rmin)+rmin
    clean=(raw+0.2*scaled)/1.2
    return raw,clean

def score_dev(groups,projection,dino,mode,device,torch,batch_size=8,block24_sidecar=None,norm_sidecar=None):
    result={};
    for g in groups:
        raws=[];cleans=[];n=len(g['gt'])
        for st in range(0,n,batch_size):
            ed=min(st+batch_size,n);q=g['text_base'].to(device)
            if mode=='e2':
                vis=g['features'][st:ed].to(device=device,dtype=torch.float32)
                att=g['attention'][st:ed].to(device=device,dtype=torch.float32)
            else:
                tokens=g['block23'][st:ed].to(device=device,dtype=torch.float32)
                vis,att=run_block24_sidecar(block24_sidecar,norm_sidecar,tokens,torch,grad=False)
            r,c=score_clean_batch(projection,q,vis,att,torch)
            raws.extend(r.max(dim=1).values.detach().cpu().numpy());cleans.extend(c.max(dim=1).values.detach().cpu().numpy())
        result[g['task_id']]={'raw':np.asarray(raws,dtype=np.float32),'clean':np.asarray(cleans,dtype=np.float32)}
    return result

def attention_logits_from_qkv(block24,qkv):
    b,n,_=qkv.shape;attn_module=block24.attn;heads=int(attn_module.num_heads);hd=int(qkv.shape[-1]//(3*heads))
    qk=qkv.reshape(b,n,3,heads,hd).permute(2,0,3,1,4);q,k=qk[0],qk[1]
    if getattr(attn_module,'q_norm',None) is not None:q=attn_module.q_norm(q)
    if getattr(attn_module,'k_norm',None) is not None:k=attn_module.k_norm(k)
    attn=(q.float()*float(attn_module.scale))@k.float().transpose(-2,-1)
    return attn[:,:,0,5:]

def run_block24_sidecar(block24,norm,seq,torch,grad):
    qkv=[];handle=block24.attn.qkv.register_forward_hook(lambda _m,_i,o:qkv.append(o))
    try:
        if grad:
            h=block24(seq.float());patches=norm(h)[:,5:,:]
        else:
            with torch.no_grad():
                h=block24(seq.float());patches=norm(h)[:,5:,:]
    finally:
        handle.remove()
    if len(qkv)!=1:raise RuntimeError(f'Expected one block24 QKV capture, got {len(qkv)}')
    return patches,attention_logits_from_qkv(block24,qkv[0])

def dev_selection(groups,scores):
    aggregate={}
    for key in ('raw','clean'):
        pooled=np.concatenate([scores[g['task_id']][key] for g in groups])
        y=np.concatenate([g['relevant'].numpy() for g in groups]);v=np.concatenate([g['valid'].numpy() for g in groups])
        operating={str(t):choose_tau(pooled,y,v,t) for t in (.95,.98,.99)}
        aggregate[key]={'operating_points':operating,'selection_98_bg_coverage':operating['0.98']['metrics']['bg_coverage']}
    objective=(aggregate['raw']['selection_98_bg_coverage']+aggregate['clean']['selection_98_bg_coverage'])/2
    return objective,aggregate

def train_stage(stage,native,model,processor,vision,dino,clip,projection,meta,clip_meta,proj_meta,torch,max_epochs=None):
    import torch.nn.functional as F
    device=torch.device('cuda:0');feature_mode='e2' if stage=='e2' else 'e3'
    block24_sidecar=None;norm_sidecar=None
    train_groups,manifest=load_groups('train',torch);dev_groups,_=load_groups('development',torch)
    encode_group_text(train_groups,clip,native,device,torch);encode_group_text(dev_groups,clip,native,device,torch)
    manifest_sha=manifest['manifest_sha256']
    if stage=='e2':
        for p in dino.parameters():p.requires_grad_(False)
        for p in projection.parameters():p.requires_grad_(True)
        opt=torch.optim.Adam(projection.parameters(),lr=5e-5)
        epochs=max_epochs or 20;batch_size=32;patience=4;min_epochs=5
        params={'projection_lr':5e-5,'dino_lr':0.0,'batch_size':batch_size,'max_epochs':epochs,'early_stopping_patience':patience}
    else:
        e2_best=CACHE/'checkpoints/e2/best_projection.pt'
        if not e2_best.exists():raise FileNotFoundError('E3 requires best E2 MLP checkpoint')
        state=torch.load(e2_best,map_location='cpu',weights_only=False);projection.load_state_dict(state['projection'],strict=True)
        for p in projection.parameters():p.requires_grad_(True)
        for p in dino.parameters():p.requires_grad_(False)
        block24_sidecar=copy.deepcopy(dino.blocks[-1]).to(device=device,dtype=torch.float32)
        norm_sidecar=copy.deepcopy(dino.norm).to(device=device,dtype=torch.float32)
        for p in block24_sidecar.parameters():p.requires_grad_(True)
        for p in norm_sidecar.parameters():p.requires_grad_(False)
        if any(p.requires_grad for p in dino.parameters()):raise RuntimeError('Policy DINO must remain frozen during E3')
        opt=torch.optim.AdamW([{'params':block24_sidecar.parameters(),'lr':1e-6,'weight_decay':0.01},
                               {'params':projection.parameters(),'lr':5e-5,'weight_decay':0.0}])
        epochs=max_epochs or 12;batch_size=8;patience=3;min_epochs=4
        params={'projection_lr':5e-5,'dino_block24_lr':1e-6,'batch_size':batch_size,'max_epochs':epochs,'early_stopping_patience':patience,
                'weight_decay_block24':0.01,'weight_decay_projection':0.0,
                'block24_train_precision':'FP32 sidecar initialized by exact OpenVLA BF16-to-FP32 cast',
                'final_layernorm':'frozen FP32 copy of original OpenVLA BF16 values; policy module untouched'}
        # Frozen input tower and CLIP must have no trainable parameters.
        if any(p.requires_grad for p in clip.parameters()) or any(p.requires_grad for p in dino.patch_embed.parameters()):raise RuntimeError('Frozen backbone/text parameters unexpectedly trainable')
    if stage=='e2':epochs=max_epochs or 20
    else:epochs=max_epochs or 12
    for p in clip.parameters():p.requires_grad_(False)
    for p in model.parameters():
        if id(p) not in {id(q) for q in projection.parameters()} and not (stage=='e3' and id(p) in {id(q) for q in dino.blocks[-1].parameters()}):p.requires_grad_(False)
    run_dir=CACHE/'checkpoints'/stage;run_dir.mkdir(parents=True,exist_ok=True)
    latest=run_dir/'latest.pt';best_path=run_dir/('best_projection.pt' if stage=='e2' else 'best_block24_projection.pt')
    history_path=OUT/f'training_history_{stage}.json'
    config={'stage':stage,'loss':'0.70 weighted BCEWithLogits(cosine, fractional role occupancy; positive weight 4, negative 1) + 0.30 soft Dice; empty query masks skipped','optimizer':type(opt).__name__,
            'learning_rate_schedule':'cosine over epochs, 5% terminal floor; multiplicative factor shared by optimizer groups',**params,
            'seed':SEED,'data_manifest_sha256':manifest_sha,'input_features':'cached exact OpenVLA block23 261-token sequence; FP32 block24 sidecar' if stage=='e3' else 'cached OpenVLA final block24 LayerNorm spatial patches',
            'text_encoder':'frozen OpenAI CLIP ViT-B/16 seven-template average; raw 512-dimensional embeddings cached per task','inference':'sigmoid cosine RAW; original lambda_bg=0.2 CLEAN','checkpoint_selection':'mean RAW/CLEAN dev BG coverage at >=98% FG recall; threshold calibrated on dev only'}
    if latest.exists():
        ck=torch.load(latest,map_location=device,weights_only=False)
        if ck.get('data_manifest_sha256')!=manifest_sha or ck.get('stage')!=stage:raise RuntimeError('Resume checkpoint experiment/data mismatch')
        projection.load_state_dict(ck['projection'],strict=True)
        if stage=='e3':
            block24_sidecar.load_state_dict(ck['block24'],strict=True)
            norm_sidecar.load_state_dict(ck['block24_norm'],strict=True)
        opt.load_state_dict(ck['optimizer']);start_epoch=int(ck['epoch'])+1;history=ck.get('history',[]);best=float(ck.get('best_objective',-1));best_epoch=ck.get('best_epoch');bad=int(ck.get('bad_evals',0))
        if 'torch_rng' in ck:torch.set_rng_state(ck['torch_rng'].cpu())
        if 'cuda_rng' in ck:torch.cuda.set_rng_state_all(ck['cuda_rng'])
        if 'numpy_rng' in ck:np.random.set_state(ck['numpy_rng'])
        if 'python_rng' in ck:random.setstate(ck['python_rng'])
    else:start_epoch=1;history=[];best=-1.;best_epoch=None;bad=0
    n_train=sum(len(g['gt']) for g in train_groups)
    if stage=='e3':
        trainable=sum(p.numel() for p in block24_sidecar.parameters() if p.requires_grad)+sum(p.numel() for p in projection.parameters() if p.requires_grad)
    else:trainable=sum(p.numel() for p in projection.parameters())
    atomic_json(OUT/f'training_configuration_{stage}.json',{**config,'train_images':n_train,'trainable_parameter_count':trainable,
        'projection_parameter_count':sum(p.numel() for p in projection.parameters()),'block24_parameter_count':sum(p.numel() for p in (block24_sidecar.parameters() if stage=='e3' else dino.blocks[-1].parameters())),
        'dino_revision':meta.get('revision'),'clip_sha256':clip_meta.get('sha256'),'initial_mlp_sha256':proj_meta.get('sha256') if stage=='e2' else sha_file(e2_best)})
    projection.train()
    if stage=='e3':block24_sidecar.train()
    all_batches=sum((len(g['gt'])+batch_size-1)//batch_size for g in train_groups)
    base_lrs=[5e-5] if stage=='e2' else [1e-6,5e-5]
    for epoch in range(start_epoch,epochs+1):
        factor=0.05+0.95*0.5*(1.0+np.cos(np.pi*(epoch-1)/max(epochs-1,1)))
        for group,base_lr in zip(opt.param_groups,base_lrs):group['lr']=base_lr*float(factor)
        epoch_t=time.time();loss_sum=0.;parts={'bce':0.,'dice':0.};steps=0;gradnorms=[]
        for g in train_groups:
            rng=np.random.default_rng(SEED+epoch+sum(g['task_id'].encode()))
            indices=rng.permutation(len(g['gt']))
            text=g['text_base'].to(device)
            for st in range(0,len(indices),batch_size):
                ix=indices[st:st+batch_size];ix_t=torch.as_tensor(ix,dtype=torch.long)
                gt=g['gt'][ix_t].to(device=device,dtype=torch.float32);valid=g['valid'][ix_t].to(device=device,dtype=torch.float32)
                opt.zero_grad(set_to_none=True)
                if stage=='e2':
                    visual=g['features'][ix_t].to(device=device,dtype=torch.float32)
                else:
                    seq=g['block23'][ix_t].to(device=device,dtype=torch.float32)
                    visual,_=run_block24_sidecar(block24_sidecar,norm_sidecar,seq,torch,grad=True)
                logits=raw_logits(projection,text,visual,torch)
                loss,component=dense_loss(logits,gt,valid,torch)
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite {stage} loss at epoch={epoch}')
                loss.backward()
                params_to_clip=list(projection.parameters())+(list(block24_sidecar.parameters()) if stage=='e3' else [])
                gradnorm=torch.nn.utils.clip_grad_norm_(params_to_clip,1.0)
                if not torch.isfinite(gradnorm):raise FloatingPointError(f'Nonfinite gradient norm in {stage}')
                opt.step();loss_sum+=float(loss.detach());parts['bce']+=component['bce'];parts['dice']+=component['dice'];gradnorms.append(float(gradnorm));steps+=1
        projection.eval();
        if stage=='e3':block24_sidecar.eval()
        scores=score_dev(dev_groups,projection,dino,stage,device,torch,batch_size=4 if stage=='e3' else 16,
                         block24_sidecar=block24_sidecar,norm_sidecar=norm_sidecar)
        objective,metrics=dev_selection(dev_groups,scores)
        row={'epoch':epoch,'train_loss':loss_sum/max(steps,1),'train_bce':parts['bce']/max(steps,1),'train_dice':parts['dice']/max(steps,1),
             'mean_gradient_norm':float(np.mean(gradnorms)) if gradnorms else 0.,'optimizer_steps':steps,'dev_selection_objective_mean_raw_clean_bg_coverage_at_98_recall':objective,
             'learning_rates':[float(group['lr']) for group in opt.param_groups],'dev_metrics':metrics,'epoch_seconds':time.time()-epoch_t,'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device))}
        history.append(row)
        improved=objective>best+0.001
        if improved:
            best=objective;best_epoch=epoch;bad=0
            state={'stage':stage,'epoch':epoch,'projection':{k:v.detach().cpu() for k,v in projection.state_dict().items()},
                   'block24':{k:v.detach().cpu() for k,v in block24_sidecar.state_dict().items()} if stage=='e3' else None,
                   'block24_norm':{k:v.detach().cpu() for k,v in norm_sidecar.state_dict().items()} if stage=='e3' else None,
                   'data_manifest_sha256':manifest_sha,'history':history,'best_objective':best,'best_epoch':best_epoch,
                   'dev_metrics':metrics,'training_configuration':config,'optimizer':opt.state_dict()}
            atomic_torch_save(torch,best_path,state)
        else:bad+=1
        checkpoint={'stage':stage,'epoch':epoch,'projection':{k:v.detach().cpu() for k,v in projection.state_dict().items()},
                    'block24':{k:v.detach().cpu() for k,v in block24_sidecar.state_dict().items()} if stage=='e3' else None,
                    'block24_norm':{k:v.detach().cpu() for k,v in norm_sidecar.state_dict().items()} if stage=='e3' else None,
                    'data_manifest_sha256':manifest_sha,'history':history,'best_objective':best,'best_epoch':best_epoch,
                    'bad_evals':bad,'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                    'training_configuration':config}
        atomic_torch_save(torch,latest,checkpoint)
        atomic_json(history_path,history)
        write_history_csv(OUT/f'training_history_{stage}.csv',history)
        print(f'[{stage.upper()}] epoch={epoch}/{epochs} loss={row["train_loss"]:.5f} dev98cov={objective:.4f} best={best:.4f}@{best_epoch} patience={bad}/{patience} sec={row["epoch_seconds"]:.1f}',flush=True)
        projection.train();
        if stage=='e3':block24_sidecar.train()
        if epoch>=min_epochs and bad>=patience:break
    # Restore development-selected best model for the next stage/evaluation.
    chosen=torch.load(best_path,map_location='cpu',weights_only=False)
    projection.load_state_dict(chosen['projection'],strict=True)
    if stage=='e3':
        block24_sidecar.load_state_dict(chosen['block24'],strict=True)
        norm_sidecar.load_state_dict(chosen['block24_norm'],strict=True)
    projection.eval();dino.eval()
    atomic_json(OUT/f'{stage}_training_summary.json',{'status':'complete','best_epoch':chosen['best_epoch'],'best_objective':chosen['best_objective'],
        'best_checkpoint':str(best_path),'checkpoint_sha256':sha_file(best_path),'epochs_completed':len(history),'configuration':config,
        'history_file':str(history_path),'final_loss':history[-1]['train_loss'] if history else None,'stopped_early':len(history)<epochs})

def run_train(stage,max_epochs=None):
    native,model,processor,vision,dino,clip,projection,meta,clip_meta,proj_meta,torch=load_base()
    train_stage(stage,native,model,processor,vision,dino,clip,projection,meta,clip_meta,proj_meta,torch,max_epochs=max_epochs)

def smoke_train():
    import torch
    import torch.nn.functional as F
    native,model,processor,vision,dino,clip,projection,meta,clip_meta,proj_meta,torch=load_base()
    for p in projection.parameters():p.requires_grad_(True)
    manifest=json.loads(DATA_MANIFEST.read_text())
    task=manifest['tasks'][0]; rec=task['splits']['train'][0]
    with np.load(rec['cache'],allow_pickle=False) as z:
        gt=torch.from_numpy(np.asarray(z['gt'][:1],np.float32)).to('cuda')
        valid=torch.from_numpy(np.asarray(z['valid'][:1],np.float32)).to('cuda')
        queries=json.loads(str(z['queries_json'].item()))
    fp=CACHE/'features/train'/slug(task['task_id'])/f"{rec['demo']}.npz"
    with np.load(fp,allow_pickle=False) as z:
        vis=torch.from_numpy(np.asarray(z['final_features'][:1],np.float32)).to('cuda')
        seq=torch.from_numpy(np.asarray(z['block23_tokens'][:1],np.float32)).to(device='cuda',dtype=next(dino.parameters()).dtype)
    tb=query_embeddings(task['task_id'],queries,clip,native,torch.device('cuda'),torch)
    opt=torch.optim.Adam(projection.parameters(),lr=5e-5)
    logits=raw_logits(projection,tb,vis,torch);loss,parts=dense_loss(logits,gt,valid,torch)
    loss.backward();gn=torch.nn.utils.clip_grad_norm_(projection.parameters(),1.);opt.step()
    if not torch.isfinite(loss) or not torch.isfinite(gn):raise FloatingPointError('E2 smoke loss/gradient nonfinite')
    temp=CACHE/'smoke_e2_checkpoint.pt'; atomic_torch_save(torch,temp,{'projection':projection.state_dict(),'optimizer':opt.state_dict(),'epoch':1})
    restored=type(projection)().to('cuda');saved=torch.load(temp,map_location='cuda',weights_only=False);restored.load_state_dict(saved['projection'],strict=True)
    with torch.no_grad():
        if not torch.equal(projection(tb),restored(tb)):raise RuntimeError('E2 checkpoint restore changed MLP outputs')
    for p in dino.parameters():p.requires_grad_(False)
    block24_sidecar=copy.deepcopy(dino.blocks[-1]).to(device='cuda',dtype=torch.float32)
    norm_sidecar=copy.deepcopy(dino.norm).to(device='cuda',dtype=torch.float32)
    for p in block24_sidecar.parameters():p.requires_grad_(True)
    for p in norm_sidecar.parameters():p.requires_grad_(False)
    opt3=torch.optim.AdamW([{'params':block24_sidecar.parameters(),'lr':1e-6},{'params':projection.parameters(),'lr':5e-5}])
    seq=seq.float();block_before=next(block24_sidecar.parameters()).detach().clone()
    vis3,att3=run_block24_sidecar(block24_sidecar,norm_sidecar,seq,torch,grad=True)
    logits3=raw_logits(projection,tb,vis3,torch);loss3,parts3=dense_loss(logits3,gt,valid,torch)
    loss3.backward();g3=torch.nn.utils.clip_grad_norm_(list(block24_sidecar.parameters())+list(projection.parameters()),1.);opt3.step()
    if not torch.isfinite(loss3) or not torch.isfinite(g3):raise FloatingPointError('E3 smoke loss/gradient nonfinite')
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in dino.parameters()):raise FloatingPointError('Nonfinite DINO gradient')
    if any(p.grad is not None for p in dino.parameters()):raise RuntimeError('E3 smoke leaked gradients into frozen policy DINO')
    if torch.equal(block_before,next(block24_sidecar.parameters()).detach()):raise RuntimeError('E3 FP32 block sidecar did not update')
    with torch.no_grad():
        test_raw,test_clean=score_clean_batch(projection,tb,vis3,att3,torch)
    if test_raw.shape!=(1,len(queries),256) or test_clean.shape!=test_raw.shape or not torch.isfinite(test_clean).all():raise RuntimeError('Scoring smoke output malformed')
    e3temp=CACHE/'smoke_e3_checkpoint.pt';atomic_torch_save(torch,e3temp,{'projection':projection.state_dict(),'block24':block24_sidecar.state_dict(),'block24_norm':norm_sidecar.state_dict(),'optimizer':opt3.state_dict(),'epoch':1})
    ck=torch.load(e3temp,map_location='cuda',weights_only=False);block24_sidecar.load_state_dict(ck['block24'],strict=True);norm_sidecar.load_state_dict(ck['block24_norm'],strict=True)
    atomic_json(OUT/'training_smoke_validation.json',{'status':'passed','E2':{'loss':float(loss.detach()),'bce':parts['bce'],'dice':parts['dice'],'gradient_norm':float(gn),'atomic_checkpoint_restore':True},
        'E3':{'loss':float(loss3.detach()),'bce':parts3['bce'],'dice':parts3['dice'],'gradient_norm':float(g3),'block24_fp32_sidecar_trainable':True,
              'initial_block24_values_exact_bf16_cast':True,'policy_dino_frozen_no_grad':True,'block24_parameter_changed':True,'atomic_checkpoint_written':True},
        'score_shapes':{'raw':list(test_raw.shape),'clean':list(test_clean.shape)},'data_rows':1,'no_policy_actions_applied':True})
    for p in (temp,e3temp):
        try:p.unlink()
        except FileNotFoundError:pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['e2','e3','smoke'],required=True);p.add_argument('--max-epochs',type=int);a=p.parse_args()
    if a.stage=='smoke':smoke_train()
    else:run_train(a.stage,a.max_epochs)
if __name__=='__main__':main()
