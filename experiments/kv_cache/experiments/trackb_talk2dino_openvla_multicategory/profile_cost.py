#!/usr/bin/env python3
"""Synchronized sidecar cost profile on six already-saved E2 observations."""
from __future__ import annotations
import json, statistics, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
import run_experiment as exp

def sync(torch,device):
    if device.type=='cuda': torch.cuda.synchronize(device)

def timed_summary(values):
    return {'count':len(values),'mean_ms':float(np.mean(values)),'median_ms':float(np.median(values)),
            'p95_ms':float(np.percentile(values,95))}

def main():
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for synchronized profile')
    torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(exp.WORKTREE));sys.path.insert(0,str(exp.REPO))
    import experiments.kv_cache
    experiments.kv_cache.__path__.append(str(exp.WORKTREE/'experiments/kv_cache'))
    import experiments.kv_cache.experiments
    experiments.kv_cache.experiments.__path__.append(str(exp.WORKTREE/'experiments/kv_cache/experiments'))
    from experiments.kv_cache.experiments.trackb_talk2dino_native import run_libero_e2 as native
    from experiments.kv_cache.experiments.trackb_talk2dino_native.model import TimmTalk2DINOHook,_qkv_logits
    from experiments.kv_cache.experiments.static_reuse_experiment import prepare_inputs,resolve_unnorm_key
    segments,_,_=exp.input_segments()
    device=torch.device('cuda:0')
    model,processor,_,dino,_,vla_meta=native.openvla_model(device)
    dtype=next(dino.parameters()).dtype
    clip,clip_meta=native.load_clip_tower(device);projection,mlp_meta=native.load_projection(device)
    summaries={'device':torch.cuda.get_device_name(device),'samples':[],'model_identity':{
        'openvla_revision':vla_meta.get('revision'),'clip_sha256':clip_meta['sha256'],'mlp_sha256':mlp_meta['sha256']}}
    # Deterministic two evaluation frames from each LIBERO suite.
    selected=[]
    for suite in ('libero_goal','libero_object','libero_spatial'):
        seg=next(s for s in segments if s['suite']==suite and s['split']=='evaluation')
        selected.extend((seg,i) for i in (0,5))
    baseline_dino=[];hook_dino=[];unpack=[];attn23=[];attn24=[];raw_ms={'B':[],'C':[]};clean_ms={'B':[],'C':[]}
    per_frame=[];first_action=None
    for seg,i in selected:
        rgb=seg['arrays']['rgb'][i]; pil=Image.fromarray(rgb)
        pixels,_=native.make_official_pixels(rgb,processor);x=pixels.to(device=device,dtype=dtype)
        def plain_pass():
            sync(torch,device);t=time.perf_counter()
            with torch.inference_mode(): dino(x)
            sync(torch,device);return (time.perf_counter()-t)*1000
        def hooked_pass():
            qkv23={};block23=dino.blocks[22]
            with torch.inference_mode(),TimmTalk2DINOHook(dino) as hook:
                h=block23.attn.qkv.register_forward_hook(lambda _m,_i,o:qkv23.__setitem__('out',o))
                sync(torch,device);t=time.perf_counter();dino(x);sync(torch,device)
                elapsed=(time.perf_counter()-t)*1000
                if hook.block_calls != [1]*24 or hook.module_calls != 1: raise RuntimeError('Instrumented DINO traversal did not execute each block exactly once')
                captured={'block24':hook.records['block24'],'policy':hook.records['policy'],
                          'qkv24':hook.records['qkv24'],'qkv23':qkv23['out']}
                h.remove()
            return elapsed,captured
        # Warm both paths and alternate their order to reduce Orin clock/cache bias.
        with torch.inference_mode():
            for _ in range(2): plain_pass();hooked_pass()
        base_reps=[];hook_reps=[];captured=None
        for rep in range(5):
            if rep%2==0:
                base_reps.append(plain_pass());ht,captured=hooked_pass();hook_reps.append(ht)
            else:
                ht,captured=hooked_pass();hook_reps.append(ht);base_reps.append(plain_pass())
        base=float(np.median(base_reps));hooked=float(np.median(hook_reps))
        baseline_dino.append(base);hook_dino.append(hooked)
        # All 24 blocks were executed once in the ordinary hooked pass; block 24
        # is not launched a second time for this sidecar.
        sync(torch,device);t=time.perf_counter()
        bpatch=F.normalize(dino.norm(captured['block24'])[:,5:,:].float(),dim=-1)[0]
        cpatch=captured['policy'][0]
        sync(torch,device);unpack_ms=(time.perf_counter()-t)*1000;unpack.append(unpack_ms)
        bq=captured['qkv24'];cq=captured['qkv23']
        bblock=dino.blocks[23];cblock=dino.blocks[22]
        def get_b_attention():
            return _qkv_logits(bblock.attn.qkv,bq,1,16,64,bblock.attn.scale,
                q_norm=getattr(bblock.attn,'q_norm',None),k_norm=getattr(bblock.attn,'k_norm',None))
        def get_c_attention():
            return _qkv_logits(cblock.attn.qkv,cq,1,16,64,cblock.attn.scale,
                q_norm=getattr(cblock.attn,'q_norm',None),k_norm=getattr(cblock.attn,'k_norm',None))
        get_b_attention();get_c_attention()
        def measure(fn):
            vals=[];value=None
            for _ in range(5):
                sync(torch,device);t=time.perf_counter();value=fn();sync(torch,device);vals.append((time.perf_counter()-t)*1000)
            return value,float(np.median(vals))
        ba,bt=measure(get_b_attention);attn24.append(bt)
        ca,ct=measure(get_c_attention);attn23.append(ct)
        phrases=[q['text'] for q in seg['queries']]
        templates=("itap of a {}.","a bad photo of a {}.","a origami {}.","a photo of the large {}.","a {} in a video game.","art of the {}.","a photo of the small {}.")
        tok=native.official_clip_tokens([t.format(q) for q in phrases for t in templates]).to(device)
        with torch.inference_mode():
            te=clip.encode_text(tok).float().reshape(len(phrases),len(templates),512).mean(1)
            text=F.normalize(projection(te),dim=-1)
        for name,patch,attn in (('B',bpatch,ba),('C',cpatch,ca)):
            def raw_score():
                cosine=text@F.normalize(patch.float(),dim=-1).T
                return cosine.sigmoid()
            raw=raw_score();raw,raw_time=measure(raw_score);raw_ms[name].append(raw_time)
            def clean_score():
                regions=(attn.float().unsqueeze(1)*patch.float().T.unsqueeze(0)).mean(dim=-1)
                regions=F.normalize(regions,dim=-1).T
                weights=(text@regions).softmax(dim=-1)
                attn_class=(attn.float().unsqueeze(0)*weights.unsqueeze(-1)).sum(dim=1)
                attn_class=attn_class/weights.sum(dim=-1,keepdim=True).clamp_min(1e-12)
                attn_class=attn_class.softmax(dim=-1)
                amin,amax=attn_class.min(),attn_class.max()
                scaled=(attn_class-amin)/torch.maximum(amax,amax-amin).clamp_min(1e-12)
                rmin,rmax=raw.min(),raw.max();scaled=scaled*(rmax-rmin)+rmin
                return (raw+exp.LAMBDA_BG*scaled)/(1+exp.LAMBDA_BG)
            clean_score();_clean,clean_time=measure(clean_score);clean_ms[name].append(clean_time)
            # The complete decomposition must reproduce the validated scorer.
            reference=exp.projected_scoring(patch,attn,text,torch)
            if (reference['raw']-raw).abs().max().item()>2e-5 or (reference['clean']-_clean).abs().max().item()>2e-5:
                raise RuntimeError(f'{name} cost decomposition changed Talk2DINO scores')
        per_frame.append({'task_id':seg['task_id'],'frame_index':seg['selected_indices'][i],
            'baseline_native_dino_ms':base,'hooked_native_dino_ms':hooked,'feature_unpack_ms':unpack_ms,
            'block24_attention_logits_ms':bt,'block23_attention_logits_ms':ct,
            'B_raw_scoring_ms':raw_ms['B'][-1],'B_cleaning_ms':clean_ms['B'][-1],
            'C_raw_scoring_ms':raw_ms['C'][-1],'C_cleaning_ms':clean_ms['C'][-1]})
        if first_action is None: first_action=(seg,i,x,pil)
    # A small matched deterministic action-prediction timing probe. No action is
    # applied to an environment; sidecar output is computed on the captured pass.
    seg,i,x,pil=first_action
    action_inputs=prepare_inputs(processor,seg['instruction'],pil,device)
    key=resolve_unnorm_key(model,'libero_spatial')
    def action_call(instrumented):
        with torch.inference_mode():
            if not instrumented:
                sync(torch,device);t=time.perf_counter();out=model.predict_action(**action_inputs,unnorm_key=key,do_sample=False);sync(torch,device)
                return out,(time.perf_counter()-t)*1000
            with TimmTalk2DINOHook(dino) as hook:
                h=dino.blocks[22].attn.qkv.register_forward_hook(lambda _m,_i,o:None)
                try:
                    sync(torch,device);t=time.perf_counter();out=model.predict_action(**action_inputs,unnorm_key=key,do_sample=False);sync(torch,device)
                finally:h.remove()
                return out,(time.perf_counter()-t)*1000
    baseline_action=[];hooked_action=[]
    for _ in range(2): action_call(False);action_call(True)
    for _ in range(5):
        out0,v=action_call(False);baseline_action.append(v)
        out1,v=action_call(True);hooked_action.append(v)
        if not np.array_equal(np.asarray(out0),np.asarray(out1)):
            raise RuntimeError('Passive timing hooks changed the deterministic action output')
    summaries['native_dino_baseline']=timed_summary(baseline_dino)
    summaries['instrumented_native_dino_forward']=timed_summary(hook_dino)
    summaries['feature_unwrap_and_final_norm']=timed_summary(unpack)
    summaries['additional_block24_execution_count']=0
    summaries['attention_logits_extraction']={'block23':timed_summary(attn23),'block24':timed_summary(attn24)}
    summaries['scoring_ms_per_frame']={'B_raw':timed_summary(raw_ms['B']),'B_cleaning':timed_summary(clean_ms['B']),
        'C_raw':timed_summary(raw_ms['C']),'C_cleaning':timed_summary(clean_ms['C'])}
    summaries['deterministic_action_latency_probe']={'repeats':5,'warmup_pairs':2,'baseline_ms':timed_summary(baseline_action),
        'passive_hook_ms':timed_summary(hooked_action),'action_differences_applied_to_environment':False,
        'mean_instrumentation_delta_ms':float(np.mean(hooked_action)-np.mean(baseline_action))}
    summaries['incremental_sidecar_estimate_ms']={
        'B_final24':float(np.median(np.array(hook_dino)-np.array(baseline_dino))+np.median(unpack)+np.median(attn24)+np.median(raw_ms['B'])+np.median(clean_ms['B'])),
        'C_native23':float(np.median(np.array(hook_dino)-np.array(baseline_dino))+np.median(unpack)+np.median(attn23)+np.median(raw_ms['C'])+np.median(clean_ms['C']))}
    summaries['incremental_sidecar_estimate_definition']='paired instrumented-minus-uninstrumented DINO traversal + feature unwrap/final norm + separately synchronized attention-logit extraction + RAW score + CLEAN score; component estimate, not a closed-loop speedup claim'
    summaries['per_frame']=per_frame
    summaries['peak_cuda_allocated_bytes']=int(torch.cuda.max_memory_allocated(device));summaries['peak_cuda_reserved_bytes']=int(torch.cuda.max_memory_reserved(device))
    path=exp.OUT/'latency_profile.json';exp.atomic_json(path,summaries)
    # Merge profiled components into the existing main latency artifact.
    main_path=exp.OUT/'latency_metrics.json';main=json.loads(main_path.read_text())
    main['component_profile']=summaries
    exp.atomic_json(main_path,main)
    report=exp.OUT/'EXPERIMENT_REPORT.md'
    if report.exists():
        report_text=report.read_text()
        section=("\n## Synchronized sidecar cost profile\n\n"
            f"A deterministic component profile used {len(selected)} already-saved evaluation frames (two per suite), plus five paired action-decode timing repeats on one saved image after two warmup pairs. The profile used CUDA synchronization around measured operations. The standard DINO forward already traversed all 24 blocks exactly once; the segmentation sidecar added **zero** block-24 executions. The B/C pass measured {summaries['native_dino_baseline']['median_ms']:.3f} ms median without hooks and {summaries['instrumented_native_dino_forward']['median_ms']:.3f} ms with passive captures.\n\n"
            f"| Component | Median ms | P95 ms |\n|---|---:|---:|\n"
            f"| Native DINO forward, uninstrumented | {summaries['native_dino_baseline']['median_ms']:.3f} | {summaries['native_dino_baseline']['p95_ms']:.3f} |\n"
            f"| Native DINO forward with passive hooks | {summaries['instrumented_native_dino_forward']['median_ms']:.3f} | {summaries['instrumented_native_dino_forward']['p95_ms']:.3f} |\n"
            f"| Feature unwrap + final LayerNorm sidecar | {summaries['feature_unwrap_and_final_norm']['median_ms']:.3f} | {summaries['feature_unwrap_and_final_norm']['p95_ms']:.3f} |\n"
            f"| Block-23 QK attention extraction | {summaries['attention_logits_extraction']['block23']['median_ms']:.3f} | {summaries['attention_logits_extraction']['block23']['p95_ms']:.3f} |\n"
            f"| Block-24 QK attention extraction | {summaries['attention_logits_extraction']['block24']['median_ms']:.3f} | {summaries['attention_logits_extraction']['block24']['p95_ms']:.3f} |\n"
            f"| B RAW score | {summaries['scoring_ms_per_frame']['B_raw']['median_ms']:.3f} | {summaries['scoring_ms_per_frame']['B_raw']['p95_ms']:.3f} |\n"
            f"| B CLEAN calculation after RAW | {summaries['scoring_ms_per_frame']['B_cleaning']['median_ms']:.3f} | {summaries['scoring_ms_per_frame']['B_cleaning']['p95_ms']:.3f} |\n"
            f"| C RAW score | {summaries['scoring_ms_per_frame']['C_raw']['median_ms']:.3f} | {summaries['scoring_ms_per_frame']['C_raw']['p95_ms']:.3f} |\n"
            f"| C CLEAN calculation after RAW | {summaries['scoring_ms_per_frame']['C_cleaning']['median_ms']:.3f} | {summaries['scoring_ms_per_frame']['C_cleaning']['p95_ms']:.3f} |\n\n"
            f"The paired action-decode probe measured {summaries['deterministic_action_latency_probe']['baseline_ms']['median_ms']:.3f} ms median without hooks and {summaries['deterministic_action_latency_probe']['passive_hook_ms']['median_ms']:.3f} ms with hooks; action outputs were exactly equal. The estimated per-frame sidecar work (hooked-minus-unhooked DINO time + feature unwrap/final norm + QK extraction + RAW + CLEAN) was {summaries['incremental_sidecar_estimate_ms']['B_final24']:.3f} ms for B and {summaries['incremental_sidecar_estimate_ms']['C_native23']:.3f} ms for C. This is a component estimate, not a policy speedup; text encoding is task-cached and its first-use cost is excluded. Full measurements and per-frame values are in `latency_metrics.json` and `latency_profile.json`.\n")
        if '## Synchronized sidecar cost profile' in report_text:
            report_text=report_text.split('\n## Synchronized sidecar cost profile',1)[0]
        report.write_text(report_text+section)
    print(json.dumps({'status':'complete','path':str(path),'incremental_sidecar_estimate_ms':summaries['incremental_sidecar_estimate_ms']},indent=2),flush=True)

if __name__=='__main__':main()
