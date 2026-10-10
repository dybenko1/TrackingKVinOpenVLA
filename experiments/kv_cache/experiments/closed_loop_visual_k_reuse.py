"""Stage 2D-2 closed-loop visual-K fixed-refresh reuse."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from libero.libero import benchmark
from experiments.kv_analysis.experiments.extract_libero_image import DEFAULT_STEP_ACTION,TASK_SUITE_NAME,create_libero_env,normalize_and_invert_openvla_gripper
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import capture_prediction,policy_image_from_observation,seed_everything
from experiments.kv_cache.experiments.prefill_reuse_experiment import DuringPrefillVisualKReuse,runtime_attention_report
from experiments.kv_cache.experiments.visual_k_projection_skip import DuringPrefillVisualKProjectionSkip
from experiments.kv_cache.experiments.static_reuse_experiment import DEFAULT_MODEL_ID,cache_layers,clone_prefill_cache,load_model_and_processor,resolve_unnorm_key,validate_compatible_prefills
OUT=Path(__file__).resolve().parents[1]/'outputs'; LAYERS=tuple(range(14))
def run(name,interval,model,processor,device,key,true_skip=False,task_id=0,state_id=0):
 seed_everything(7); suite=benchmark.get_benchmark_dict()[TASK_SUITE_NAME](); task=suite.get_task(task_id); states=suite.get_task_init_states(task_id); env,desc=create_libero_env(task); prompt=f'In: What action should the robot take to {desc.lower()}?\nOut:'; src=None; rows=[]; done=False
 try:
  env.reset();obs=env.set_init_state(states[state_id])
  for _ in range(10):obs,_,done,_=env.step(DEFAULT_STEP_ACTION)
  for step in range(220):
   reuse=name!='baseline' and step%interval!=0; image=policy_image_from_observation(obs,True)
   if reuse:
    assert src and src['step']==step-(step%interval)
    ctx=(DuringPrefillVisualKProjectionSkip(model,src['cache'],src['layout'],LAYERS) if true_skip else DuringPrefillVisualKReuse(src['cache'],src['layout'],LAYERS)); pred=capture_prediction(model,processor,image,prompt,device,key,ctx); validate_compatible_prefills(src['cache'],pred['cache'],src['input_ids'],pred['input_ids'],src['layout'],pred['layout'],LAYERS)
    if true_skip: ctx.validate(pred['cache']); val={'inserted':sorted(ctx.diagnostics['layers'])==list(LAYERS),'skip_rows':all(x['rows']==35 for x in ctx.diagnostics['k_rows']),'qv_rows':all(x['rows']==pred['prefill_length'] for x in ctx.diagnostics['q_rows']+ctx.diagnostics['v_rows']),'rows_omitted':256*len(LAYERS)}
    else: ctx.validate_completed_prefill(pred['cache']); val={'inserted':sorted(ctx.diagnostics['inserted_layers'])==list(LAYERS),'decode_untouched':ctx.diagnostics['decode_updates_untouched']==84}
   else:
    pred=capture_prediction(model,processor,image,prompt,device,key); src={'cache':clone_prefill_cache(pred['cache']),'input_ids':pred['input_ids'].clone(),'layout':dict(pred['layout']),'step':step};val={'inserted':True}
   assert pred['layout']['visual']==[1,256] and pred['layout']['visual_token_count']==256
   age=step-src['step'] if reuse else 0; assert not reuse or 1<=age<interval
   act=normalize_and_invert_openvla_gripper(pred['policy_action']);obs,_,done,_=env.step(act.tolist());rows.append({'step':step,'refresh_interval':interval,'type':'reuse' if reuse else 'refresh','source_step':src['step'] if reuse else None,'source_age':age,'prefill_length':pred['prefill_length'],'visual_token_count':256,'layers':list(LAYERS) if reuse else [],'policy_action':pred['policy_action'].tolist(),'environment_action':act.tolist(),'done':bool(done),'validation':val})
   if done:break
 finally:env.close()
 return {'condition':name,'success':bool(done),'steps':len(rows),'termination':'success' if done else 'horizon','records':rows,'refresh_count':sum(x['type']=='refresh' for x in rows),'reuse_count':sum(x['type']=='reuse' for x in rows),'reuse_fraction':sum(x['type']=='reuse' for x in rows)/len(rows),'max_source_age':max(x['source_age'] for x in rows),'valid':all(x['validation'].get('inserted',True) and x['validation'].get('decode_untouched',True) for x in rows)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--interval',type=int,required=True);p.add_argument('--baseline',action='store_true');p.add_argument('--true-skip',action='store_true');p.add_argument('--task-id',type=int,default=0);p.add_argument('--state-id',type=int,default=0);a=p.parse_args();assert a.interval in (2,3,5)
 out=OUT/(f'stage2d_visual_k_skip_generalization/task{a.task_id}_state{a.state_id}' if a.true_skip else f'stage2d_closed_loop_visual_k_r{a.interval}');out.mkdir(parents=True,exist_ok=True);model,processor,device=load_model_and_processor(DEFAULT_MODEL_ID);key=resolve_unnorm_key(model,None);result={'config':{'R':a.interval,'task_id':a.task_id,'state_id':a.state_id,'layers':list(LAYERS),'visual_positions':[1,256],'compute_skipping':a.true_skip},'attention':runtime_attention_report(model),'visual_k':run(f'visual_k_r{a.interval}',a.interval,model,processor,device,key,a.true_skip,a.task_id,a.state_id)}
 if a.baseline: result['baseline']=run('baseline',a.interval,model,processor,device,key,False,a.task_id,a.state_id)
 (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['visual_k'],indent=2))
if __name__=='__main__':main()
