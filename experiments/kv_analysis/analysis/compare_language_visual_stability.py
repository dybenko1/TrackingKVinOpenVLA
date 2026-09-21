"""Aligned Stage 2D-0 language-versus-visual adjacent-step KV reanalysis.

One untouched fresh OpenVLA trajectory captures both raw k_proj K and the
post-RoPE DynamicCache K/V in the same multimodal prefill.  It does not alter
policy outputs, cache entries, or environment actions.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from libero.libero import benchmark
from experiments.kv_analysis.experiments.extract_libero_image import (DEFAULT_MODEL_ID, DEFAULT_STEP_ACTION, TASK_SUITE_NAME, create_libero_env, extract_libero_image, normalize_and_invert_openvla_gripper, prepare_openvla_policy_image)
from experiments.kv_cache.experiments.static_reuse_experiment import PrefillSnapshot, cache_layers, load_model_and_processor, prepare_inputs, resolve_unnorm_key

OUT = Path(__file__).resolve().parents[1] / "outputs" / "language_visual_comparison"
SELECTED = (0, 5, 10, 15, 20, 25, 31)

def cos(a,b): return F.cosine_similarity(a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)).item()
def l2(a,b): return (torch.norm(b.float()-a.float()) / torch.norm(a.float())).item()
def stat(x):
 x=np.asarray(x); return {"mean":float(x.mean()),"median":float(np.median(x)),"std":float(x.std()),"p10":float(np.percentile(x,10)),"p90":float(np.percentile(x,90)),"min":float(x.min()),"max":float(x.max()),"count":int(x.size)}
def image(obs, crop):
 import tensorflow as tf
 x=tf.image.encode_jpeg(extract_libero_image(obs)); x=tf.io.decode_image(x,expand_animations=False,dtype=tf.uint8); x=tf.image.resize(x,(224,224),method="lanczos3",antialias=True); x=tf.cast(tf.clip_by_value(tf.round(x),0,255),tf.uint8).numpy()
 return prepare_openvla_policy_image(x,center_crop=crop)

class RawKObserver:
 def __init__(self, model): self.model=model; self.handles=[]; self.outputs={}
 def __enter__(self):
  for i,layer in enumerate(self.model.language_model.model.layers):
   def hook(_m,_a,out,i=i):
    if i not in self.outputs and out.shape[1] > 1: self.outputs[i]=out.detach().cpu().clone()
   self.handles.append(layer.self_attn.k_proj.register_forward_hook(hook))
  return self
 def __exit__(self,*_):
  for h in self.handles: h.remove()
  if len(self.outputs)!=32: raise RuntimeError(f"Expected raw prefill K for 32 layers, got {len(self.outputs)}")

def blocks(cache, raw, layout):
 v=int(layout['visual_token_count']); plen=int(layout['prefill_length']); ls=v+1
 if (v,plen,ls)!=(256,291,257): raise AssertionError(f"Unexpected validated task-0 layout: {layout}")
 ans={"language":[],"visual":[],"raw_language_k":[]}
 for i,((k,val)) in enumerate(cache_layers(cache)):
  # cache [B,H,S,D] -> [B,S,H,D]; raw K [B,S,H*D] -> equivalent token layout.
  ck=k.permute(0,2,1,3).detach().float().cpu(); cv=val.permute(0,2,1,3).detach().float().cpu()
  rk=raw[i].view(1,plen,ck.shape[2],ck.shape[3]).float()
  if ck.shape[1]!=plen or raw[i].shape[1]!=plen: raise AssertionError("prefill shape mismatch")
  ans['language'].append({'k':ck[:,ls:,:], 'v':cv[:,ls:,:]})
  ans['visual'].append({'k':ck[:,1:ls,:], 'v':cv[:,1:ls,:]})
  ans['raw_language_k'].append(rk[:,ls:,:])
 return ans

def plot_temporal(pairs, block, component, metric, path):
 plt.figure(figsize=(12,6))
 for layer in SELECTED:
  y=[p['blocks'][block][layer][component][metric] for p in pairs]
  plt.plot(range(len(y)),y,label=f"Layer {layer}")
 plt.xlabel("Control-step transition t → t+1"); plt.ylabel("Cosine similarity" if metric=='cosine' else "Relative L2 change")
 plt.title(f"{block.title()} {component.upper()} temporal stability"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=220); plt.close()
def plot_compare(summary, comp, metric, path):
 x=np.arange(32); plt.figure(figsize=(9,5))
 for block in ('language','visual'): plt.plot(x,[r['blocks'][block][comp][metric]['mean'] for r in summary],marker='o',ms=3,label=f"{block.title()} {comp.upper()}")
 plt.xlabel("Transformer layer"); plt.ylabel("Mean adjacent-step cosine" if metric=='cosine' else "Mean adjacent-step relative L2 (lower = more stable)"); plt.title(f"Language vs visual {comp.upper()} {metric.replace('_',' ')}"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=220); plt.close()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=OUT); ap.add_argument('--horizon',type=int,default=220); ap.add_argument('--settling-steps',type=int,default=10); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--center-crop',action=argparse.BooleanOptionalAction,default=True); a=ap.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=True); figs=a.output_dir/'figures'; figs.mkdir(exist_ok=True)
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
 model,processor,device=load_model_and_processor(DEFAULT_MODEL_ID); key=resolve_unnorm_key(model,None)
 suite=benchmark.get_benchmark_dict()[TASK_SUITE_NAME](); task=suite.get_task(0); states=suite.get_task_init_states(0); env,desc=create_libero_env(task); prompt=f"In: What action should the robot take to {desc.lower()}?\nOut:"
 prev=None; pairs=[]; actions=[]; done=False
 try:
  env.reset(); obs=env.set_init_state(states[0])
  for _ in range(a.settling_steps): obs,_,done,_=env.step(DEFAULT_STEP_ACTION)
  for step in range(a.horizon):
   inputs=prepare_inputs(processor,prompt,Image.fromarray(image(obs,a.center_crop)).convert('RGB'),device)
   with RawKObserver(model) as raw, PrefillSnapshot(model) as snap, torch.inference_mode(): action=np.asarray(model.predict_action(**inputs,unnorm_key=key,do_sample=False))
   cache,_logits,_ids,layout=snap.require(); now=blocks(cache,raw.outputs,layout); actions.append(action.tolist())
   if prev is not None:
    row={'from_control_step':step-1,'to_control_step':step,'blocks':{'language':[],'visual':[]},'rope':[]}
    for layer in range(32):
     for block in ('language','visual'):
      rr={}
      for c in ('k','v'): rr[c]={'cosine':cos(prev[block][layer][c],now[block][layer][c]),'relative_l2':l2(prev[block][layer][c],now[block][layer][c])}
      row['blocks'][block].append(rr)
     rawc={'cosine':cos(prev['raw_language_k'][layer],now['raw_language_k'][layer]),'relative_l2':l2(prev['raw_language_k'][layer],now['raw_language_k'][layer])}
     post=row['blocks']['language'][layer]['k']; row['rope'].append({'raw':rawc,'post_rope':post,'abs_difference':{'cosine':abs(rawc['cosine']-post['cosine']),'relative_l2':abs(rawc['relative_l2']-post['relative_l2'])}})
    pairs.append(row)
   prev=now; obs,_,done,_=env.step(normalize_and_invert_openvla_gripper(action).tolist()); print(f'control step {step}')
   if done: break
 finally: env.close()
 if not pairs: raise RuntimeError('Need two control steps')
 summary=[]
 for layer in range(32):
  r={'layer':layer,'blocks':{}}
  for block in ('language','visual'):
   r['blocks'][block]={c:{m:stat([p['blocks'][block][layer][c][m] for p in pairs]) for m in ('cosine','relative_l2')} for c in ('k','v')}
  for c in ('k','v'):
   r['differences_visual_minus_language_'+c]={m:stat([p['blocks']['visual'][layer][c][m]-p['blocks']['language'][layer][c][m] for p in pairs]) for m in ('cosine','relative_l2')}
  r['rope_raw_vs_post_rope_language_k']={m:stat([p['rope'][layer]['abs_difference'][m] for p in pairs]) for m in ('cosine','relative_l2')}; summary.append(r)
 rope={m:stat([p['rope'][l]['abs_difference'][m] for p in pairs for l in range(32)]) for m in ('cosine','relative_l2')}
 result={'trajectory':{'task_suite':TASK_SUITE_NAME,'task_id':0,'initial_state_id':0,'seed':a.seed,'settling_steps_excluded':a.settling_steps,'policy_control_steps':len(actions),'adjacent_pairs':len(pairs),'success':bool(done),'instruction':desc,'layout':{'prefill_length':291,'visual_positions':[1,256],'language_positions':[257,290],'bos_excluded':True,'generated_action_tokens_excluded':True},'preprocessing':'official-style extract/resize/center-crop path; fresh greedy policy'},'metric_definition':{'cosine':'cosine(flatten(A.float()), flatten(B.float()))','relative_l2':'norm(B.float()-A.float()) / norm(A.float())'},'shapes':{'cached_k_v':'[batch, tokens, kv_heads, head_dim] after cache transpose','raw_k_proj':'[batch, tokens, kv_heads, head_dim] after reshape'},'layer_summary':summary,'rope_raw_vs_post_rope_absolute_difference_all_layers_pairs':rope}
 (a.output_dir/'comparison_results.json').write_text(json.dumps(result,indent=2)); (a.output_dir/'adjacent_pair_metrics.json').write_text(json.dumps(pairs)); (a.output_dir/'policy_actions.json').write_text(json.dumps(actions))
 with (a.output_dir/'layer_summary.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['layer','language_k_cosine','visual_k_cosine','visual_minus_language_k_cosine','language_k_relative_l2','visual_k_relative_l2','visual_minus_language_k_relative_l2','language_v_cosine','visual_v_cosine','visual_minus_language_v_cosine','language_v_relative_l2','visual_v_relative_l2','visual_minus_language_v_relative_l2'])
  for r in summary:
   g=lambda b,c,m:r['blocks'][b][c][m]['mean']; d=lambda c,m:r['differences_visual_minus_language_'+c][m]['mean']
   w.writerow([r['layer'],g('language','k','cosine'),g('visual','k','cosine'),d('k','cosine'),g('language','k','relative_l2'),g('visual','k','relative_l2'),d('k','relative_l2'),g('language','v','cosine'),g('visual','v','cosine'),d('v','cosine'),g('language','v','relative_l2'),g('visual','v','relative_l2'),d('v','relative_l2')])
 for b in ('language','visual'):
  for c in ('k','v'):
   for m in ('cosine','relative_l2'): plot_temporal(pairs,b,c,m,figs/f'{b}_temporal_{c}_{m}.png')
 for c in ('k','v'):
  for m in ('cosine','relative_l2'): plot_compare(summary,c,m,figs/f'language_vs_visual_{c}_{m}_by_layer.png')
 print(json.dumps({'steps':len(actions),'pairs':len(pairs),'success':bool(done),'rope':rope,'output':str(a.output_dir)},indent=2))
if __name__=='__main__': main()
