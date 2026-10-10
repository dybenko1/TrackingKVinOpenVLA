"""Stage 2D-1 offline age-1 visual-K substitution; no compute skipping or env feedback."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch
from libero.libero import benchmark
from experiments.kv_analysis.experiments.extract_libero_image import DEFAULT_STEP_ACTION, TASK_SUITE_NAME, create_libero_env, normalize_and_invert_openvla_gripper
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import capture_prediction, policy_image_from_observation, seed_everything
from experiments.kv_cache.experiments.prefill_reuse_experiment import DuringPrefillVisualKReuse, action_metrics, runtime_attention_report
from experiments.kv_cache.experiments.static_reuse_experiment import DEFAULT_MODEL_ID, cache_layers, clone_prefill_cache, load_model_and_processor, resolve_unnorm_key, validate_compatible_prefills

OUT=Path(__file__).resolve().parents[1]/'outputs'/'stage2d_visual_k_age1'; LAYERS=tuple(range(14))
def stats(x):
 x=np.asarray(x,float); return {'mean':float(x.mean()),'median':float(np.median(x)),'std':float(x.std()),'p10':float(np.percentile(x,10)),'p90':float(np.percentile(x,90)),'min':float(x.min()),'max':float(x.max()),'count':int(x.size)}
def plot(x,y,path,title,xlabel,ylabel,kind='line'):
 plt.figure(figsize=(10,5)); plt.scatter(x,y,s=16) if kind=='scatter' else plt.plot(x,y,marker='o',ms=3); plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path,dpi=220); plt.close()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=OUT); p.add_argument('--horizon',type=int,default=220); p.add_argument('--seed',type=int,default=7); p.add_argument('--settling-steps',type=int,default=10); p.add_argument('--center-crop',action=argparse.BooleanOptionalAction,default=True); a=p.parse_args()
 out=a.output_dir; figs=out/'figures'; figs.mkdir(parents=True,exist_ok=True)
 seed_everything(a.seed); model,processor,device=load_model_and_processor(DEFAULT_MODEL_ID); key=resolve_unnorm_key(model,None); attention=runtime_attention_report(model)
 comp_root=Path('experiments/kv_analysis/outputs/language_visual_comparison'); stability=json.loads((comp_root/'adjacent_pair_metrics.json').read_text()); reference_actions=json.loads((comp_root/'policy_actions.json').read_text())
 suite=benchmark.get_benchmark_dict()[TASK_SUITE_NAME](); task=suite.get_task(0); states=suite.get_task_init_states(0); env,desc=create_libero_env(task); prompt=f'In: What action should the robot take to {desc.lower()}?\nOut:'
 previous=None; rows=[]; fresh_actions=[]; done=False
 try:
  env.reset(); obs=env.set_init_state(states[0])
  for _ in range(a.settling_steps): obs,_,done,_=env.step(DEFAULT_STEP_ACTION)
  for step in range(a.horizon):
   image=policy_image_from_observation(obs,a.center_crop)
   baseline=capture_prediction(model,processor,image,prompt,device,key); fresh_actions.append(baseline['policy_action'].tolist())
   if step < len(reference_actions) and not np.array_equal(baseline['policy_action'],np.asarray(reference_actions[step])): raise AssertionError(f'Fresh trajectory diverged from Stage 2D-0 at step {step}.')
   if previous is not None:
    validate_compatible_prefills(previous['cache'],baseline['cache'],previous['input_ids'],baseline['input_ids'],previous['layout'],baseline['layout'],LAYERS)
    layout=baseline['layout']
    if int(layout['visual_token_count'])!=256 or layout['visual']!=[1,256]: raise AssertionError(f'Visual layout invalid: {layout}')
    source=clone_prefill_cache(previous['cache'])
    context=DuringPrefillVisualKReuse(source,layout,LAYERS)
    experimental=capture_prediction(model,processor,image,prompt,device,key,intervention=context)
    context.validate_completed_prefill(experimental['cache'])
    metrics=action_metrics(torch.tensor([baseline['action_token_ids']]),torch.tensor([experimental['action_token_ids']]),baseline['policy_action'],experimental['policy_action'])
    diff=np.asarray(metrics['baseline_action_token_ids'])!=np.asarray(metrics['prefill_reuse_action_token_ids']); first=(np.flatnonzero(diff)[0]+1).item() if diff.any() else None
    st=stability[step-1]; ks=[st['blocks']['visual'][l]['k'] for l in LAYERS]
    rows.append({'source_step':step-1,'target_step':step,'source_age':1,'prefill_length':baseline['prefill_length'],'visual_token_count':256,'intervention_layers':list(LAYERS),'first_divergent_token_1_indexed':first,'visual_k_stability_layers_0_13':{'mean_cosine':float(np.mean([x['cosine'] for x in ks])),'mean_relative_l2':float(np.mean([x['relative_l2'] for x in ks])),'per_layer':ks},'intervention_validation':{'all_layers_inserted':sorted(context.diagnostics['inserted_layers'])==list(LAYERS),'decode_calls_untouched':context.diagnostics['decode_updates_untouched']==len(LAYERS)*6,'source_prefill_independent_clone':all(sk.data_ptr()!=pk.data_ptr() and sv.data_ptr()!=pv.data_ptr() for (sk,sv),(pk,pv) in zip(cache_layers(source),cache_layers(previous['cache']))),'diagnostics':context.diagnostics},**metrics})
   previous=baseline
   obs,_,done,_=env.step(normalize_and_invert_openvla_gripper(baseline['policy_action']).tolist())
   print(f'control step {step}')
   if done: break
 finally: env.close()
 if len(fresh_actions)!=89 or len(rows)!=88 or not done: raise RuntimeError(f'Fresh trajectory discrepancy: steps={len(fresh_actions)}, pairs={len(rows)}, success={done}; expected 89/88/success.')
 l2=np.array([r['action_difference_l2'] for r in rows]); exact=np.array([r['action_token_ids_exactly_equal'] for r in rows]); token1=np.array([r['action_token_1_changed'] for r in rows]); changed=np.array([r['number_of_differing_action_tokens'] for r in rows]); cos=np.array([r['visual_k_stability_layers_0_13']['mean_cosine'] for r in rows]); rel=np.array([r['visual_k_stability_layers_0_13']['mean_relative_l2'] for r in rows])
 aggregate={'pairs':88,'exact_7_token_match_count':int(exact.sum()),'exact_7_token_match_rate':float(exact.mean()),'token_1_divergence_count':int(token1.sum()),'token_1_divergence_rate':float(token1.mean()),'total_changed_action_tokens':int(changed.sum()),'first_divergent_token_distribution_1_indexed':{str(i):int(sum(r['first_divergent_token_1_indexed']==i for r in rows)) for i in range(1,8)},'action_l2':stats(l2),'action_max_absolute':stats([r['action_difference_max_absolute'] for r in rows]),'stability_action_l2_pearson':{'cosine':float(np.corrcoef(cos,l2)[0,1]),'relative_l2':float(np.corrcoef(rel,l2)[0,1])},'worst_by_action_l2':sorted(rows,key=lambda r:r['action_difference_l2'],reverse=True)[:10],'lowest_by_visual_k_cosine':sorted(rows,key=lambda r:r['visual_k_stability_layers_0_13']['mean_cosine'])[:10]}
 result={'configuration':{'stage':'2D-1','model_id':DEFAULT_MODEL_ID,'task_suite':TASK_SUITE_NAME,'task_id':0,'initial_state_id':0,'seed':a.seed,'settling_steps':a.settling_steps,'horizon':a.horizon,'reuse':'visual K only, age 1, compute-then-overwrite during prefill','layers':list(LAYERS),'source':'independent fresh prefill snapshot at t','target':'same fresh-policy observation at t+1'},'attention_runtime':attention,'trajectory_validation':{'expected_steps':89,'actual_steps':len(fresh_actions),'expected_pairs':88,'actual_pairs':len(rows),'success':bool(done),'stage2d0_fresh_actions_match':True},'aggregate':aggregate,'pairs':rows}
 (out/'visual_k_age1_results.json').write_text(json.dumps(result,indent=2)); (out/'per_pair_summary.json').write_text(json.dumps(rows,indent=2))
 x=np.arange(88); plot(x,l2,figs/'visual_k_action_l2_by_transition.png','Visual-K age-1 action deviation','Transition t → t+1','Action L2'); plot(x,(~exact).astype(int),figs/'visual_k_exact_match_by_transition.png','Visual-K age-1 discrete changes','Transition t → t+1','Any action-token change (1=yes)'); plot(cos,l2,figs/'visual_k_cosine_vs_action_l2.png','Visual-K similarity vs action deviation','Mean visual-K cosine, layers 0–13','Action L2','scatter'); plot(rel,l2,figs/'visual_k_relative_l2_vs_action_l2.png','Visual-K change vs action deviation','Mean visual-K relative L2, layers 0–13','Action L2','scatter')
 print(json.dumps(aggregate,indent=2))
if __name__=='__main__': main()
