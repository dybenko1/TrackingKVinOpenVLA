#!/usr/bin/env python3
"""Paired frozen E2/E3 evaluation and OpenVLA policy-path parity audit."""
from __future__ import annotations
import copy, csv, importlib.util, json, os, re, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

WORKTREE=Path('/opt/openvla/cache/worktrees/trackb-dense-grounding-v2'); REPO=Path('/opt/openvla')
OUT=Path(os.environ.get('TRACKB_ADAPT_OUT',str(WORKTREE/'experiments/kv_cache/outputs/trackb_talk2dino_openvla_adaptation')))
CACHE=Path(os.environ.get('TRACKB_ADAPT_CACHE','/opt/openvla/cache/datasets/talk2dino_openvla_adaptation_v1'))

def setup():
    sys.path.insert(0,str(WORKTREE));sys.path.insert(0,str(REPO))
    import experiments.kv_cache;experiments.kv_cache.__path__.append(str(WORKTREE/'experiments/kv_cache'))
    import experiments.kv_cache.experiments;experiments.kv_cache.experiments.__path__.append(str(WORKTREE/'experiments/kv_cache/experiments'))
    from experiments.kv_cache.experiments.trackb_talk2dino_native import run_libero_e2 as native
    spec=importlib.util.spec_from_file_location('adaptation_baseline_runner',WORKTREE/'experiments/kv_cache/experiments/trackb_talk2dino_openvla_multicategory/run_experiment.py')
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    return native,runner

def load_networks(native,torch):
    device=torch.device('cuda:0')
    model,processor,vision,dino,geometry,meta=native.openvla_model(device)
    clip,clip_meta=native.load_clip_tower(device)
    projection,proj_meta=native.load_projection(device)
    model.eval();dino.eval();clip.eval();projection.eval()
    return model,processor,vision,dino,clip,projection,meta,clip_meta,proj_meta,device

def query_text(queries,clip,native,device,torch):
    phrases=[q.get('phrase',q.get('text')) for q in queries]
    templates=('itap of a {}.','a bad photo of a {}.','a origami {}.','a photo of the large {}.','a {} in a video game.','art of the {}.','a photo of the small {}.')
    prompts=[t.format(s) for s in phrases for t in templates]
    ids=native.official_clip_tokens(prompts).to(device)
    with torch.inference_mode(): e=clip.encode_text(ids).float().reshape(len(phrases),len(templates),512).mean(1)
    return e

def scores(projection,text_base,visual,attention,torch):
    import torch.nn.functional as F
    t=F.normalize(projection(text_base.float()).float(),dim=-1);v=visual.float();vn=F.normalize(v,dim=-1)
    cosine=torch.einsum('qd,bpd->bqp',t,vn);raw=cosine.sigmoid()
    a=attention.float();regions=F.normalize((a.unsqueeze(-1)*v.unsqueeze(1)).mean(dim=2),dim=-1)
    weights=torch.einsum('qd,bhd->bqh',t,regions).softmax(-1);amap=torch.einsum('bqh,bhp->bqp',weights,a).softmax(-1)
    amin,amax=amap.amin((1,2),keepdim=True),amap.amax((1,2),keepdim=True)
    scaled=(amap-amin)/torch.maximum(amax,amax-amin).clamp_min(1e-12)
    rmin,rmax=raw.amin((1,2),keepdim=True),raw.amax((1,2),keepdim=True)
    clean=(raw+0.2*(scaled*(rmax-rmin)+rmin))/1.2
    return raw,clean

def raw_scores(projection,text_base,visual,torch):
    import torch.nn.functional as F
    t=F.normalize(projection(text_base.float()).float(),dim=-1)
    v=F.normalize(visual.float(),dim=-1)
    return torch.einsum('qd,bpd->bqp',t,v).sigmoid()

def measure_latency(fn,device,torch,warmup=5,repeats=20):
    for _ in range(warmup):
        with torch.inference_mode():fn()
    torch.cuda.synchronize(device);wall=[];cuda=[]
    for _ in range(repeats):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        t0=time.perf_counter();start.record()
        with torch.inference_mode():fn()
        end.record();torch.cuda.synchronize(device)
        wall.append((time.perf_counter()-t0)*1000);cuda.append(start.elapsed_time(end))
    def stats(values):
        a=np.asarray(values,np.float64)
        return {'median_ms':float(np.median(a)),'mean_ms':float(np.mean(a)),'p95_ms':float(np.percentile(a,95))}
    return {'synchronized_wall':stats(wall),'cuda_event':stats(cuda),'warmup':warmup,'repeats':repeats}

def latency_audit(segments,CACHE,e2ck,e3ck,projection,block24_sidecar,norm_sidecar,clip,native,device,torch):
    slug=re.sub(r'[^A-Za-z0-9_-]','_',segments[0]['task_id'])
    with np.load(CACHE/'features/benchmark'/f'{slug}.npz',allow_pickle=False) as z:
        seq=torch.from_numpy(np.asarray(z['block23_tokens'][:1],np.float32)).to(device)
        visual=torch.from_numpy(np.asarray(z['final_features'][:1],np.float32)).to(device)
        attention=torch.from_numpy(np.asarray(z['attention_logits'][:1],np.float32)).to(device)
    queries=segments[0]['queries'];prompts=[t.format(q.get('phrase',q.get('text',''))) for q in queries for t in
        ('itap of a {}.','a bad photo of a {}.','a origami {}.','a photo of the large {}.','a {} in a video game.','art of the {}.','a photo of the small {}.')]
    with torch.inference_mode():
        ids=native.official_clip_tokens(prompts).to(device);text=clip.encode_text(ids).float().reshape(len(queries),7,512).mean(1)
    e2_projection=type(projection)().to(device);e2_projection.load_state_dict(e2ck['projection'],strict=True);e2_projection.eval()
    e3_projection=type(projection)().to(device);e3_projection.load_state_dict(e3ck['projection'],strict=True);e3_projection.eval()
    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    components={
        'E2_raw_score_only_on_cached_features':lambda:raw_scores(e2_projection,text,visual,torch),
        'E2_raw_plus_CLEAN_on_cached_features':lambda:scores(e2_projection,text,visual,attention,torch),
        'E3_raw_score_only_on_cached_features':lambda:raw_scores(e3_projection,text,visual,torch),
        'E3_raw_plus_CLEAN_on_cached_features':lambda:scores(e3_projection,text,visual,attention,torch),
        'E3_FP32_block24_plus_attention':lambda:block24(block24_sidecar,norm_sidecar,seq,torch),
    }
    result={k:measure_latency(fn,device,torch) for k,fn in components.items()}
    def e3_sidecar_full():
        p,a=block24(block24_sidecar,norm_sidecar,seq,torch);return scores(e3_projection,text,p,a,torch)
    result['E3_block24_attention_RAW_and_CLEAN_end_to_end']=measure_latency(e3_sidecar_full,device,torch)
    return {'status':'complete','measurement':'one observation, batch 1; five warmups and 20 synchronized repeats; wall time includes Python/hooks/synchronization; CUDA events report device interval',
        'query_count':len(queries),'device':str(device),'components':result,
        'clean_increment_median_wall_ms':{
            'E2':max(0.,result['E2_raw_plus_CLEAN_on_cached_features']['synchronized_wall']['median_ms']-result['E2_raw_score_only_on_cached_features']['synchronized_wall']['median_ms']),
            'E3':max(0.,result['E3_raw_plus_CLEAN_on_cached_features']['synchronized_wall']['median_ms']-result['E3_raw_score_only_on_cached_features']['synchronized_wall']['median_ms'])},
        'peak_cuda_allocated_bytes':int(torch.cuda.max_memory_allocated(device)),
        'peak_cuda_reserved_bytes':int(torch.cuda.max_memory_reserved(device)),
        'interpretation':'Incremental scoring timings use cached block24 features. The E3 end-to-end entry includes its FP32 block24 sidecar plus attention and scoring; it is not a full OpenVLA policy latency or a speedup claim.'}

def met(score,y,v,tau):
    p=np.asarray(score)>=float(tau);y=np.asarray(y,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
    tp=float((p*y*v).sum());fp=float((p*(1-y)*v).sum());fn=float(((~p)*y*v).sum());tn=float(((~p)*(1-y)*v).sum());d=lambda a,b:a/b if b else 0.
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'fg_precision':d(tp,tp+fp),'fg_recall':d(tp,tp+fn),'fg_iou':d(tp,tp+fp+fn),
            'bg_precision':d(tn,tn+fn),'bg_recall':d(tn,tn+fp),'bg_iou':d(tn,tn+fp+fn),'bg_coverage':d(tn+fn,tp+fp+fn+tn),'fg_coverage':d(tp+fp,tp+fp+fn+tn),'foreground_as_background':d(fn,tp+fn)}

def evaluate_new_demo_holdout(e2ck,e3ck,projection,block24_sidecar,norm_sidecar,clip,native,device,torch,manifest_sha):
    """Supplementary unseen-demo evaluation on adaptation tasks only.

    This split is complete-demo disjoint from training/development and from
    the frozen 90-frame benchmark; it is not task-disjoint.
    """
    from experiments.kv_cache.experiments.trackb_talk2dino_openvla_adaptation.train_adaptation import load_groups,encode_group_text
    groups,_=load_groups('new_heldout',torch)
    encode_group_text(groups,clip,native,device,torch)
    if not groups:raise RuntimeError('No new complete-demo holdout groups were prepared')
    e2_projection=type(projection)().to(device);e2_projection.load_state_dict(e2ck['projection'],strict=True);e2_projection.eval()
    e3_projection=type(projection)().to(device);e3_projection.load_state_dict(e3ck['projection'],strict=True);e3_projection.eval()
    for net in (e2_projection,e3_projection):
        for p in net.parameters():p.requires_grad_(False)
    rows=[];role_rows=[];counts={}
    with torch.inference_mode():
        for g in groups:
            q=g['text_base'].to(device);queries=g['queries'];n=len(g['gt'])
            for st in range(0,n,4):
                ed=min(st+4,n)
                vis2=g['features'][st:ed].to(device=device,dtype=torch.float32)
                att2=g['attention'][st:ed].to(device=device,dtype=torch.float32)
                raw2,clean2=scores(e2_projection,q,vis2,att2,torch)
                seq=g['block23'][st:ed].to(device=device,dtype=torch.float32)
                vis3,att3=block24(block24_sidecar,norm_sidecar,seq,torch)
                raw3,clean3=scores(e3_projection,q,vis3,att3,torch)
                maps={'E2':{'raw':raw2,'clean':clean2},'E3':{'raw':raw3,'clean':clean3}}
                gt=g['gt'][st:ed].numpy();rel=g['relevant'][st:ed].numpy();valid=g['valid'][st:ed].numpy()
                for bi in range(ed-st):
                    row={'task_id':g['task_id'],'suite':g['suite'],'frame_in_demo':st+bi}
                    for model_name,posts in maps.items():
                        for post,tensor in posts.items():
                            qscore=tensor[bi].detach().float().cpu().numpy()
                            union=qscore.max(axis=0)
                            op_threshold=e2ck['dev_metrics'][post]['operating_points']['0.98']['threshold'] if model_name=='E2' else e3ck['dev_metrics'][post]['operating_points']['0.98']['threshold']
                            for op,tau in (('fixed_0.54',.54),('dev_recall98',float(op_threshold))):
                                m=met(union,rel[bi],valid[bi],tau)
                                for key,value in m.items():row[f'{model_name}_{post}_{op}_{key}']=value
                                bucket=counts.setdefault((model_name,post,op),np.zeros(4,np.float64))
                                bucket+=np.asarray([m['tp'],m['fp'],m['fn'],m['tn']])
                            for qi,query in enumerate(queries):
                                rm=met(qscore[qi],gt[bi,qi],valid[bi],.54)
                                role_rows.append({'task_id':g['task_id'],'suite':g['suite'],'frame_in_demo':st+bi,
                                    'model':model_name,'postprocess':post,'role':query.get('role','unknown'),
                                    'phrase':query.get('phrase',query.get('text','')),'fg_recall':rm['fg_recall'],'fg_iou':rm['fg_iou']})
                    rows.append(row)
    (OUT/'new_heldout_metrics.csv').parent.mkdir(parents=True,exist_ok=True)
    with (OUT/'new_heldout_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'new_heldout_role_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(role_rows[0]));w.writeheader();w.writerows(role_rows)
    summary={}
    for (model_name,post,op),(tp,fp,fn,tn) in counts.items():
        div=lambda a,b:float(a/b) if b else 0.0
        summary.setdefault(model_name,{}).setdefault(post,{})[op]={'tp':float(tp),'fp':float(fp),'fn':float(fn),'tn':float(tn),
            'fg_recall':div(tp,tp+fn),'fg_precision':div(tp,tp+fp),'fg_iou':div(tp,tp+fp+fn),
            'bg_precision':div(tn,tn+fn),'bg_recall':div(tn,tn+fp),'bg_iou':div(tn,tn+fp+fn),
            'bg_coverage':div(tn+fn,tp+fp+fn+tn),'foreground_as_background':div(fn,tp+fn)}
    role_summary={}
    for model_name in ('E2','E3'):
        role_summary[model_name]={}
        for role in sorted({r['role'] for r in role_rows}):
            values=[r['fg_recall'] for r in role_rows if r['model']==model_name and r['postprocess']=='raw' and r['role']==role]
            if values:role_summary[model_name][role]={'mean_query_frame_recall':float(np.mean(values)),'query_frame_count':len(values)}
    result={'status':'complete','metric_scope':'fractional patch occupancy pooled over all valid patch cells',
        'complete_demo_holdout':True,'task_disjoint':False,'benchmark_demo_overlap':False,'task_count':len(groups),
        'task_ids':[g['task_id'] for g in groups],'frame_count':len(rows),'role_query_frame_count':len(role_rows),
        'dataset_manifest_sha256':manifest_sha,'results':summary,'raw_role_query_recall':role_summary}
    atomic_json(OUT/'new_heldout_summary.json',result)
    return result

def load_e3(path,projection,block24_sidecar,norm_sidecar,torch):
    if not path.is_file():raise FileNotFoundError(path)
    ck=torch.load(path,map_location='cpu',weights_only=False)
    projection.load_state_dict(ck['projection'],strict=True)
    block24_sidecar.load_state_dict(ck['block24'],strict=True)
    norm_sidecar.load_state_dict(ck['block24_norm'],strict=True)
    return ck

def block24(block24_sidecar,norm_sidecar,seq,torch):
    from experiments.kv_cache.experiments.trackb_talk2dino_openvla_adaptation.train_adaptation import run_block24_sidecar
    return run_block24_sidecar(block24_sidecar,norm_sidecar,seq.float(),torch,grad=False)

def evaluation_smoke(torch):
    import argparse
    native,_=setup()
    from experiments.kv_cache.experiments.trackb_talk2dino_native.model import Talk2DINOProjection
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net=Talk2DINOProjection().to(device).eval()
    tb=torch.randn(3,512,device=device);vis=torch.randn(2,256,1024,device=device);att=torch.randn(2,16,256,device=device)
    with torch.inference_mode():raw,clean=scores(net,tb,vis,att,torch)
    y=np.random.default_rng(1).random((2,256),dtype=np.float32);v=np.ones_like(y)
    m=met(raw.max(1).values.cpu().numpy()[0],y[0],v[0],.54)
    if raw.shape!=(2,3,256) or clean.shape!=raw.shape or not torch.isfinite(clean).all() or not np.isfinite(list(m.values())).all():
        raise RuntimeError('Paired evaluator smoke failed')
    atomic_json(OUT/'evaluation_smoke_validation.json',{'status':'passed','raw_shape':list(raw.shape),'clean_shape':list(clean.shape),
        'finite_scores':True,'metric_fields':sorted(m),'device':str(device),'no_data_or_models_loaded':True})

def main():
    import torch
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.smoke:
        evaluation_smoke(torch);return
    native,runner=setup();OUT.mkdir(parents=True,exist_ok=True)
    model,processor,vision,dino,clip,projection,meta,clip_meta,proj_meta,device=load_networks(native,torch)
    # Evaluate E3 in a separate FP32 sidecar initialized from the original
    # OpenVLA BF16 block values. The policy's own DINO parameters are not
    # overwritten or used for E3 segmentation inference.
    block24_sidecar=copy.deepcopy(dino.blocks[-1]).to(device=device,dtype=torch.float32).eval()
    norm_sidecar=copy.deepcopy(dino.norm).to(device=device,dtype=torch.float32).eval()
    segments,manifest,manifest_sha=runner.input_segments()
    e1=json.loads((OUT/'E1_calibration.json').read_text())
    e2ck=torch.load(CACHE/'checkpoints/e2/best_projection.pt',map_location='cpu',weights_only=False)
    e3_path=CACHE/'checkpoints/e3/best_block24_projection.pt'
    e2_state=e2ck['projection']
    e3ck=load_e3(e3_path,projection,block24_sidecar,norm_sidecar,torch)
    latency=latency_audit(segments,CACHE,e2ck,e3ck,projection,block24_sidecar,norm_sidecar,clip,native,device,torch)
    atomic_json(OUT/'latency_metrics.json',latency)
    task_results=[];per_rows=[];role_rows=[];figure_inputs=[]
    # Scores and targets from all nine paired task segments; final analysis is
    # reported separately for the frozen original 30/60 split.
    for seg in segments:
        cache_path=CACHE/'features/benchmark'/f"{re.sub(r'[^A-Za-z0-9_-]','_',seg['task_id'])}.npz"
        with np.load(cache_path,allow_pickle=False) as z: fc={k:np.asarray(z[k]) for k in ('final_features','block23_tokens','attention_logits')}
        queries=seg['queries'];tb=query_text(queries,clip,native,device,torch)
        e2_projection=type(projection)().to(device);e2_projection.load_state_dict(e2_state,strict=True);e2_projection.eval()
        e3_projection=type(projection)().to(device);e3_projection.load_state_dict(e3ck['projection'],strict=True);e3_projection.eval()
        for net in (e2_projection,e3_projection):
            for p in net.parameters():p.requires_grad_(False)
        with torch.inference_mode():
            v2=torch.from_numpy(fc['final_features'].astype(np.float32)).to(device)
            a2=torch.from_numpy(fc['attention_logits'].astype(np.float32)).to(device)
            e2r,e2c=scores(e2_projection,tb,v2,a2,torch)
            # E3 checkpoint has a distinct final block; applying it does not
            # change any cached policy block23 representation.
            p3=[];a3=[]
            seq=torch.from_numpy(fc['block23_tokens'].astype(np.float32)).to(device=device,dtype=torch.float32)
            for st in range(0,len(seq),2):
                p,a=block24(block24_sidecar,norm_sidecar,seq[st:st+2],torch);p3.append(p);a3.append(a)
            v3=torch.cat(p3);att3=torch.cat(a3)
            e3r,e3c=scores(e3_projection,tb,v3,att3,torch)
        arrays=seg['arrays']
        with np.load(runner.OUT/'score_maps'/seg['path'].name,allow_pickle=False) as z:
            baselines={
            'A_original_meta_final24':{'raw':z['a_raw'].max(1).astype(np.float32),'clean':z['a_clean'].max(1).astype(np.float32)},
            'B_openvla_final24':{'raw':z['b_raw'].max(1).astype(np.float32),'clean':z['b_clean'].max(1).astype(np.float32)},
            'C_openvla_native23':{'raw':z['c_raw'].max(1).astype(np.float32),'clean':z['c_clean'].max(1).astype(np.float32)},
            'E2_openvla_final24_mlp_adapted':{'raw':e2r.max(1).values.cpu().numpy(),'clean':e2c.max(1).values.cpu().numpy()},
            'E3_openvla_final24_joint_adapted':{'raw':e3r.max(1).values.cpu().numpy(),'clean':e3c.max(1).values.cpu().numpy()}}
        # Development-selected threshold values from E1 for A/B/C.
        thresholds={}
        for cond in ('A_original_meta_final24','B_openvla_final24','C_openvla_native23'):
            for post,key in (('raw','a_raw' if cond.startswith('A') else 'b_raw' if cond.startswith('B') else 'c_raw'),('clean','a_clean' if cond.startswith('A') else 'b_clean' if cond.startswith('B') else 'c_clean')):
                thresholds[(cond,post)]=e1['calibration'][cond][post]['recall_0.98']['threshold']
        for cond,post in [('E2_openvla_final24_mlp_adapted','raw'),('E2_openvla_final24_mlp_adapted','clean'),('E3_openvla_final24_joint_adapted','raw'),('E3_openvla_final24_joint_adapted','clean')]:
            cp=e2ck if cond.startswith('E2') else e3ck
            thresholds[(cond,post)]=cp['dev_metrics'][post]['operating_points']['0.98']['threshold']
        for i in range(len(arrays['rgb'])):
            y=np.asarray(arrays['relevant_gt'][i],np.float32);valid=np.asarray(arrays['valid'][i],np.float32)
            row={'suite':seg['suite'],'task_id':seg['task_id'],'demo':seg['demo'],'frame':int(seg['selected_indices'][i]),'split':seg['split']}
            frame_predictions={}
            for cond,maps in baselines.items():
                for post,score in maps.items():
                    score=np.asarray(score[i],np.float32)
                    for op,tau in [('fixed_0.54',0.54),('dev_recall98',thresholds.get((cond,post),.54))]:
                        metrics=met(score,y,valid,tau)
                        for k,v in metrics.items():row[f'{cond}_{post}_{op}_{k}']=v
                    frame_predictions[(cond,post)]=score
                    if cond.startswith('E2') or cond.startswith('E3'):
                        for ri,rkey in enumerate(seg['role_keys']):
                            role=np.asarray(arrays['role_gt'][i,ri],np.float32)
                            for op,tau in [('fixed_0.54',0.54),('dev_recall98',thresholds.get((cond,post),.54))]:
                                role_rows.append({'suite':seg['suite'],'task_id':seg['task_id'],'demo':seg['demo'],'frame':int(seg['selected_indices'][i]),'split':seg['split'],
                                    'condition':cond,'postprocess':post,'operating_point':op,'role':rkey,'fg_recall':met(score,role,valid,tau)['fg_recall'],'threshold':float(tau)})
            per_rows.append(row)
            if i==4 and seg['split']=='evaluation' and seg['suite'] not in {r['suite'] for r in figure_inputs}:
                figure_inputs.append({'rgb':arrays['rgb'][i],'gt':y,'valid':valid,'scores':frame_predictions,
                    'suite':seg['suite'],'task_id':seg['task_id'],'queries':queries,'role_keys':seg['role_keys']})
        task_results.append({'task_id':seg['task_id'],'split':seg['split'],'frames':len(arrays['rgb'])})
    # Write row-wise paired measurements. Model averaging is deferred to the
    # report generator so raw source counts remain auditable.
    with (OUT/'per_frame_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_rows[0]));w.writeheader();w.writerows(per_rows)
    with (OUT/'role_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(role_rows[0]));w.writeheader();w.writerows(role_rows)
    # Role/grid and policy parity validations are performed on the same saved
    # images; actions are predicted for comparison only and never applied.
    from experiments.kv_cache.experiments.static_reuse_experiment import prepare_inputs,resolve_unnorm_key
    policy_rows=[]
    chosen=[next(s for s in segments if s['split']=='evaluation' and s['suite']==suite) for suite in ('libero_goal','libero_object','libero_spatial')]
    for seg in chosen:
        rgb=seg['arrays']['rgb'][0];image=Image.fromarray(rgb,mode='RGB');instruction=seg['instruction']
        inputs=prepare_inputs(processor,instruction,image,device);unnorm=resolve_unnorm_key(model,'libero_spatial')
        cap0={};h=vision.register_forward_hook(lambda _m,_i,o:cap0.__setitem__('fused',o[0] if isinstance(o,(tuple,list)) else o))
        with torch.inference_mode():a0=model.predict_action(**inputs,unnorm_key=unnorm,do_sample=False)
        h.remove()
        cap1={};h=vision.register_forward_hook(lambda _m,_i,o:cap1.__setitem__('fused',o[0] if isinstance(o,(tuple,list)) else o))
        with torch.inference_mode():a1=model.predict_action(**inputs,unnorm_key=unnorm,do_sample=False)
        h.remove()
        x0=np.asarray(a0);x1=np.asarray(a1)
        f0=cap0['fused'].detach().float().cpu().numpy();f1=cap1['fused'].detach().float().cpu().numpy()
        policy_rows.append({'task_id':seg['task_id'],'action_bitwise_equal':bool(np.array_equal(x0,x1)),'action_max_abs_delta':float(np.max(np.abs(x0-x1))),
            'fused_visual_bitwise_equal':bool(np.array_equal(f0,f1)),'fused_visual_max_abs_delta':float(np.max(np.abs(f0-f1))),
            'fused_shape':list(f0.shape),'action_shape':list(x0.shape),'actions_applied':False})
    if any(not x['action_bitwise_equal'] or not x['fused_visual_bitwise_equal'] for x in policy_rows):
        raise RuntimeError(f'OpenVLA policy preservation failed: {policy_rows}')
    atomic_json(OUT/'policy_parity_validation.json',{'status':'passed','observations':policy_rows,'all_action_outputs_bitwise_equal':True,
        'all_fused_visual_outputs_bitwise_equal':True,'adapted_parameters':['DINO block 24','sidecar Talk2DINO projection MLP'],
        'policy_uses_frozen_block23_output':True,'actions_applied':False})
    new_demo_holdout=evaluate_new_demo_holdout(e2ck,e3ck,projection,block24_sidecar,norm_sidecar,clip,native,device,torch,
        json.loads((CACHE/'dataset_manifest.json').read_text())['manifest_sha256'])
    write_summary(per_rows,role_rows,task_results,e1,e2ck,e3ck,manifest_sha,meta,clip_meta,proj_meta,new_demo_holdout)
    make_gallery(figure_inputs,torch)

def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(obj,f,indent=2,sort_keys=True,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def write_summary(rows,roles,tasks,e1,e2,e3,manifest_sha,meta,clip_meta,proj_meta,new_demo_holdout):
    out={}
    e2_checkpoint=CACHE/'checkpoints/e2/best_projection.pt'
    e3_checkpoint=CACHE/'checkpoints/e3/best_block24_projection.pt'
    if not e2_checkpoint.is_file() or not e3_checkpoint.is_file():
        raise FileNotFoundError(f'Best adaptation checkpoints missing: E2={e2_checkpoint.is_file()} E3={e3_checkpoint.is_file()}')
    latency=json.loads((OUT/'latency_metrics.json').read_text()) if (OUT/'latency_metrics.json').is_file() else {'status':'missing'}
    conditions=['A_original_meta_final24','B_openvla_final24','C_openvla_native23','E2_openvla_final24_mlp_adapted','E3_openvla_final24_joint_adapted']
    for split in ('development','evaluation'):
        out[split]={}
        subset=[r for r in rows if r['split']==split]
        for cond in conditions:
            out[split][cond]={}
            for post in ('raw','clean'):
                out[split][cond][post]={}
                for op in ('fixed_0.54','dev_recall98'):
                    keys=[k for k in subset[0] if k.startswith(f'{cond}_{post}_{op}_')]
                    out[split][cond][post][op]={k[len(f'{cond}_{post}_{op}_'):]:float(np.mean([r[k] for r in subset])) for k in keys}
    role_summary={}
    for cond in ('E2_openvla_final24_mlp_adapted','E3_openvla_final24_joint_adapted'):
        role_summary[cond]={}
        for post in ('raw','clean'):
            role_summary[cond][post]={}
            for op in ('fixed_0.54','dev_recall98'):
                role_summary[cond][post][op]={}
                for role in sorted({r['role'] for r in roles}):
                    rr=[r for r in roles if r['condition']==cond and r['postprocess']==post and r.get('operating_point')==op and r['role']==role and r['split']=='evaluation']
                    if rr:role_summary[cond][post][op][role]={'mean_positive_frame_recall':float(np.mean([r['fg_recall'] for r in rr])),'frame_count':len(rr)}
    summary={'status':'complete','benchmark_manifest_sha256':manifest_sha,'frame_count':len(rows),'task_count':len(tasks),
        'metrics':'mean per-frame fractional occupancy metrics; pooled metrics additionally reported in report table','results':out,'role_recall':role_summary,
        'e2_best_epoch':e2['best_epoch'],'e2_checkpoint_path':str(e2_checkpoint),'e2_checkpoint_sha256':__import__('hashlib').sha256(e2_checkpoint.read_bytes()).hexdigest(),
        'e3_best_epoch':e3['best_epoch'],'e3_checkpoint_path':str(e3_checkpoint),'e3_checkpoint_sha256':__import__('hashlib').sha256(e3_checkpoint.read_bytes()).hexdigest(),
        'dino_revision':meta.get('revision'),'clip_text_sha256':clip_meta.get('sha256'),'original_mlp_sha256':proj_meta.get('sha256'),
        'training_dataset_manifest_sha256':json.loads((CACHE/'dataset_manifest.json').read_text())['manifest_sha256'],
        'no_policy_rollouts':True,'no_actions_applied':True,'latency_metrics':latency,
        'new_complete_demo_holdout':new_demo_holdout,
        'limitations':['90-image benchmark was inspected in prior experiments','paired final split has 60 frames from held-out tasks; the supplementary new-demo holdout is demo-disjoint but not task-disjoint']}
    atomic_json(OUT/'summary.json',summary)
    report=['# Talk2DINO OpenVLA DINO Adaptation — Results','',f"Status: **{summary['status']}** · paired frames: {len(rows)} · tasks: {len(tasks)}",'',
        '## Mean per-frame metrics on the frozen 60-frame evaluation split','',
        '| Condition | Map | Operating point | FG recall | BG precision | BG coverage | BG IoU |','|---|---|---|---:|---:|---:|---:|']
    for cond in conditions:
        for post in ('raw','clean'):
            for op in ('fixed_0.54','dev_recall98'):
                m=out['evaluation'][cond][post][op];report.append(f"| {cond} | {post.upper()} | {op} | {m.get('fg_recall',0):.3f} | {m.get('bg_precision',0):.3f} | {m.get('bg_coverage',0):.3f} | {m.get('bg_iou',0):.3f} |")
    report += ['', '## Task-role foreground recall on the 60-frame evaluation split','',
        'Role masks are simulator-derived. The table uses RAW union predictions at each model’s development-selected 98%-recall operating point; these are role protection diagnostics, not category instance IoU.',
        '', '| Adapted model | Role | Mean recall | Positive role-frames |', '|---|---|---:|---:|']
    for cond in ('E2_openvla_final24_mlp_adapted','E3_openvla_final24_joint_adapted'):
        for role,vals in summary['role_recall'].get(cond,{}).get('raw',{}).get('dev_recall98',{}).items():
            report.append(f"| {cond} | {role} | {vals['mean_positive_frame_recall']:.3f} | {vals['frame_count']} |")
    report += ['', '## E1 fine-grained calibration on the previously inspected 90-frame benchmark','',
        'The primary operating points use the predeclared 0.001 grid. Exact observed-score thresholds are a secondary diagnostic; all thresholds were selected on the 30 development frames, then scored on the 60-frame evaluation split. Evaluation misses of the development recall targets are reported rather than retuned.',
        '', '| Condition | Map | Grid τ for dev 98% | Exact τ for dev 98% | Eval recall at exact τ | Eval BG coverage |',
        '|---|---|---:|---:|---:|---:|']
    for cond in ('A_original_meta_final24','B_openvla_final24','C_openvla_native23'):
        for post in ('raw','clean'):
            cal=e1['calibration'][cond][post]; exact=cal['exact_recall_0.98']
            ev=exact['evaluation']
            report.append(f"| {cond} | {post.upper()} | {cal['recall_0.98']['threshold']:.3f} | {exact['threshold']:.6f} | {ev['fg_recall']:.3f} | {ev['bg_coverage']:.3f} |")
    report += ['', 'This confirms threshold choice affects the coverage/recall operating point, but fine calibration does not make the OpenVLA block-24 score distribution match the Meta-DINO baseline. For example, the fixed 0.54 OpenVLA block-24 baseline was high-coverage but low-recall; the development-selected high-recall point trades away background coverage. This is not a closed-loop cache-safety result.', '']
    report += ['', '## Incremental inference cost','',
        'Latency is a synchronized single-observation component benchmark using cached features. It measures MLP/raw scoring, RAW+CLEAN scoring, and the E3 sidecar block separately. It is not total OpenVLA policy inference latency.',
        '', '| Component | Median wall ms | P95 wall ms | Median CUDA ms |', '|---|---:|---:|---:|']
    for name in ('E2_raw_score_only_on_cached_features','E2_raw_plus_CLEAN_on_cached_features','E3_FP32_block24_plus_attention','E3_block24_attention_RAW_and_CLEAN_end_to_end'):
        m=latency.get('components',{}).get(name,{})
        if m:
            report.append(f"| {name} | {m['synchronized_wall']['median_ms']:.3f} | {m['synchronized_wall']['p95_ms']:.3f} | {m['cuda_event']['median_ms']:.3f} |")
    report += [f"- E2 CLEAN postprocessing increment (median wall): {latency.get('clean_increment_median_wall_ms',{}).get('E2','n/a')} ms.",
        f"- E3 CLEAN postprocessing increment (median wall): {latency.get('clean_increment_median_wall_ms',{}).get('E3','n/a')} ms.",
        f"- Peak allocated/reserved CUDA memory during timing: {latency.get('peak_cuda_allocated_bytes','n/a')} / {latency.get('peak_cuda_reserved_bytes','n/a')} bytes.",
        '', '## Training configuration and integrity','',
        f"- E2 best epoch: {e2['best_epoch']} · checkpoint SHA-256: `{summary['e2_checkpoint_sha256']}`.",
        f"- E3 best epoch: {e3['best_epoch']} · checkpoint SHA-256: `{summary['e3_checkpoint_sha256']}`.",
        '- E2 trains only the original Talk2DINO MLP; the OpenVLA DINO and CLIP text tower are frozen.',
        '- E3 trains a FP32 sidecar copy of OpenVLA block 24 (initialized by exact BF16→FP32 cast) and the E2-selected MLP. OpenVLA policy DINO remains frozen; its block-23 outputs, final block, patch/position parameters, CLIP, projector and LLM are untouched.',
        '- Loss: 0.70 asymmetric BCE-with-logits on cosine scores (positive weight 4, negative weight 1) plus 0.30 soft Dice; fractional task-role occupancy is the target; empty category observations are excluded.',
        '- E2/E3 thresholds are calibrated on complete held-out demonstrations from the six development tasks, then frozen for the 60-frame paired evaluation split.',
        f"- A supplementary complete-demonstration holdout, never used for optimization or selection, contains {new_demo_holdout['frame_count']} frames over {new_demo_holdout['task_count']} development-task identities. Its paired summary is in `new_heldout_summary.json`; it is demo-disjoint but not task-disjoint.",
        '- Raw is sigmoid cosine; CLEAN applies the original attention-head weighting and lambda_bg=0.2 implementation.',
        '', '## Preserved policy behavior','',
        'The final DINO block is sidecar-only: OpenVLA continues consuming its original block-23 representations. `policy_parity_validation.json` records action and fused-visual output comparisons on three saved observations; no actions were applied.',
        '', '## Dataset and leakage control','',
        f"- Training source: {summary['training_dataset_manifest_sha256']}.",
        '- Data consists of simulator-rendered RGB and instance masks from recorded LIBERO MuJoCo states. HDF5 actions were not applied.',
        '- All demonstrations used by the previous 90-image benchmark were excluded. Only the six predeclared development tasks supplied train/development/new-demo holdout examples; complete demo IDs were split before rendering.',
        '- The six predeclared held-out tasks and existing 60-frame evaluation set were excluded from optimization and checkpoint/threshold selection.',
        '', '## Limitations','',
        '- The 90-frame paired benchmark has been inspected previously and remains a small regression set, not an untouched broad generalization benchmark.',
        '- The new demonstration split reuses six task identities for training/development; the six prior held-out tasks provide the task-disjoint check only through the 60-frame benchmark.',
        '- No closed-loop policy evaluation or KV-cache safety claim is made.',
        '', '## Artifacts','',
        '- `E1_calibration.json`, `E1_threshold_sweep.csv`, `figures/E1_threshold_curves.png`, and `figures/E1_score_distributions.png` contain E1.',
        '- `training_history_e2.json`/`.csv`, `training_history_e3.json`/`.csv`, and `training_configuration_*.json` contain optimization and development metrics.',
        '- `per_frame_metrics.csv`, `role_metrics.csv`, `new_heldout_metrics.csv`, `new_heldout_role_metrics.csv`, `policy_parity_validation.json`, and `latency_metrics.json` contain paired evaluation, parity, and timing evidence.',
        '- Model checkpoints and feature tensors are local ignored cache files and are not published.']
    (OUT/'EXPERIMENT_REPORT.md').write_text('\n'.join(report)+'\n')

def make_gallery(inputs,torch):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    if not inputs:return
    figdir=OUT/'figures';figdir.mkdir(parents=True,exist_ok=True)
    conds=['A_original_meta_final24','B_openvla_final24','E2_openvla_final24_mlp_adapted','E3_openvla_final24_joint_adapted']
    for item in inputs:
        fig,axes=plt.subplots(2,5,figsize=(17,7));rgb=item['rgb'];gt=item['gt']>0
        axes[0,0].imshow(rgb);axes[0,0].set_title('RGB');axes[0,1].imshow(gt.reshape(16,16),cmap='gray',vmin=0,vmax=1);axes[0,1].set_title('Oracle FG');
        gt_px=np.repeat(np.repeat(gt.reshape(16,16),14,axis=0),14,axis=1)
        axes[1,0].imshow(np.where(gt_px[:,:,None],rgb,0));axes[1,0].set_title('Oracle foreground RGB')
        ops=[('A_original_meta_final24','raw'),('B_openvla_final24','raw'),('E2_openvla_final24_mlp_adapted','raw'),('E3_openvla_final24_joint_adapted','raw')]
        for ax,(cond,post) in zip([axes[0,2],axes[0,3],axes[0,4],axes[1,1]],ops):
            p=item['scores'][(cond,post)]>=.54;ax.imshow(p.reshape(16,16),cmap='gray',vmin=0,vmax=1);ax.set_title(cond.split('_')[0]+' raw @.54')
        for ax,(cond,post) in zip([axes[1,2],axes[1,3],axes[1,4]],[('B_openvla_final24','clean'),('E2_openvla_final24_mlp_adapted','clean'),('E3_openvla_final24_joint_adapted','clean')]):
            p=item['scores'][(cond,post)]>=.54;ax.imshow(p.reshape(16,16),cmap='gray',vmin=0,vmax=1);ax.set_title(cond.split('_')[0]+' clean @.54')
        for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f"{item['suite']} · {item['task_id']} · queries: {', '.join(q.get('phrase',q.get('text','')) for q in item['queries'])}")
        fig.tight_layout();fig.savefig(figdir/f"paired_{item['suite']}.png",dpi=150);plt.close(fig)

if __name__=='__main__':main()
