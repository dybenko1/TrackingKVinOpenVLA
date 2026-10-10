"""Stage 3A oracle foreground/background visual-K validation (no matrix)."""
from __future__ import annotations
import hashlib, json, os, types
from pathlib import Path
from typing import Any
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import SegmentationRenderEnv
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
from experiments.kv_analysis.experiments.extract_libero_image import DEFAULT_STEP_ACTION, TASK_SUITE_NAME, normalize_and_invert_openvla_gripper
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import capture_prediction, policy_image_from_observation, seed_everything
from experiments.kv_cache.experiments.oracle_foreground_mask import rendered_label_mapping, foreground_mask, policy_mask, token_mask
from experiments.kv_cache.experiments.static_reuse_experiment import DEFAULT_MODEL_ID, cache_layers, clone_prefill_cache, load_model_and_processor, resolve_unnorm_key

LAYERS=tuple(range(14)); OUT=Path(__file__).resolve().parents[1]/'outputs/stage3a_masked_visual_k_validation'

def seg_env(task):
 bddl=os.path.join(get_libero_path('bddl_files'),task.problem_folder,task.bddl_file)
 e=SegmentationRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,camera_segmentations='instance'); e.seed(0); return e

def mask_from_obs(obs,mapping):
 seg=np.asarray(obs['agentview_segmentation_instance'])
 if seg.ndim==3: seg=seg[...,0]
 fg=token_mask(policy_mask(foreground_mask(seg,mapping)[::-1,::-1],True))
 return fg

class MaskedOverwrite:
 def __init__(self,source,layout,fg):
  self.source,self.layout,self.fg=source,layout,np.asarray(fg,dtype=bool); self.bg=~self.fg; self.length=int(layout['prefill_length']);self.original=None
  self.d={'layers':[],'decode':0,'projected_rows':[],'reused_rows':[],'provenance':{}}
 def __enter__(self):
  self.original=torch.classes # placeholder replaced below
  from transformers.cache_utils import DynamicCache
  self.original=DynamicCache.update; prior=cache_layers(self.source); vs,ve=self.layout['visual']; ls,le=self.layout['language']
  assert (vs,ve)==(1,256) and self.fg.size==256
  bg=torch.as_tensor(np.flatnonzero(self.bg)+1,dtype=torch.long)
  fg=torch.as_tensor(np.flatnonzero(self.fg)+1,dtype=torch.long)
  def update(cache,k,v,layer,kwargs=None):
   if layer in LAYERS and k.shape[-2]==self.length:
    p,_=prior[layer]; assert k.shape==p.shape
    fresh=k.clone(); vv=v.clone(); inds=bg.to(k.device)
    k[:,:,inds,:]=p[:,:,inds,:]
    assert torch.equal(k[:,:,inds,:],p[:,:,inds,:])
    assert torch.equal(k[:,:,fg.to(k.device),:],fresh[:,:,fg.to(k.device),:])
    assert torch.equal(v,vv) and torch.equal(k[:,:,:1,:],fresh[:,:,:1,:]) and torch.equal(k[:,:,ls:le+1,:],fresh[:,:,ls:le+1,:])
    self.d['layers'].append(layer);self.d['projected_rows'].append(self.length);self.d['reused_rows'].append(int(bg.numel()));self.d['provenance'][str(layer)]='fg=current,bg=refresh'
   elif layer in LAYERS and k.shape[-2]==1:self.d['decode']+=1
   return self.original(cache,k,v,layer,kwargs)
  DynamicCache.update=update; self._dc=DynamicCache; return self
 def __exit__(self,*a): self._dc.update=self.original
 def validate(self,cache):
  assert sorted(self.d['layers'])==list(LAYERS) and self.d['decode']==84
  bg=torch.as_tensor(np.flatnonzero(self.bg)+1,dtype=torch.long,device=cache_layers(cache)[0][0].device)
  for i in LAYERS: assert torch.equal(cache_layers(cache)[i][0][:,:,bg,:],cache_layers(self.source)[i][0][:,:,bg,:])

class MaskedSkip:
 def __init__(self,model,source,layout,fg):
  self.model,self.source,self.layout,self.fg=model,source,layout,np.asarray(fg,dtype=bool);self.bg=~self.fg;self.length=int(layout['prefill_length']);self.orig={}
  self.d={'layers':[],'q_rows':[],'v_rows':[],'k_rows':[],'reused_rows':[],'decode_k_rows':[],'provenance':{}}
 def __enter__(self):
  prior=cache_layers(self.source);vs,ve=self.layout['visual'];ls,le=self.layout['language'];assert (vs,ve)==(1,256) and self.fg.size==256
  for li in LAYERS:
   attn=self.model.language_model.model.layers[li].self_attn;self.orig[li]=attn.forward
   def forward(m,hidden_states,position_embeddings,attention_mask,past_key_value=None,cache_position=None,**kw):
    h=hidden_states
    if h.shape[-2]!=self.length or past_key_value is None:
     if h.shape[-2]==1:self.d['decode_k_rows'].append({'layer':m.layer_idx,'rows':1})
     return self.orig[m.layer_idx](h,position_embeddings,attention_mask,past_key_value,cache_position,**kw)
    B,S,_=h.shape; fg_idx=torch.as_tensor(np.flatnonzero(self.fg)+1,dtype=torch.long,device=h.device); lang=torch.arange(ls,S,device=h.device); idx=torch.cat((torch.zeros(1,dtype=torch.long,device=h.device),fg_idx,lang))
    shape=(B,S,-1,m.head_dim);q=m.q_proj(h).view(shape).transpose(1,2);v=m.v_proj(h).view(shape).transpose(1,2); fresh=m.k_proj(h.index_select(1,idx)).view(B,idx.numel(),-1,m.head_dim).transpose(1,2)
    self.d['q_rows'].append({'layer':m.layer_idx,'rows':S});self.d['v_rows'].append({'layer':m.layer_idx,'rows':S});self.d['k_rows'].append({'layer':m.layer_idx,'rows':int(idx.numel())});self.d['reused_rows'].append(int(self.bg.sum()))
    cos,sin=position_embeddings;q,_=apply_rotary_pos_emb(q,q,cos,sin);_,fresh=apply_rotary_pos_emb(fresh,fresh,cos.index_select(1,idx),sin.index_select(1,idx))
    p,_=prior[m.layer_idx];k=p.clone();k.index_copy_(2,idx,fresh);assert k.shape[-2]==S
    k,v=past_key_value.update(k,v,m.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position});fn=eager_attention_forward if m.config._attn_implementation=='eager' else ALL_ATTENTION_FUNCTIONS[m.config._attn_implementation];o,w=fn(m,q,k,v,attention_mask,dropout=0.0 if not m.training else m.attention_dropout,scaling=m.scaling,**kw);self.d['layers'].append(m.layer_idx);self.d['provenance'][str(m.layer_idx)]='fg=current,bg=refresh';return m.o_proj(o.reshape(B,S,-1).contiguous()),w
   attn.forward=types.MethodType(forward,attn)
  return self
 def __exit__(self,*a):
  for li,f in self.orig.items():self.model.language_model.model.layers[li].self_attn.forward=f
 def validate(self,cache):
  assert sorted(self.d['layers'])==list(LAYERS) and len(self.d['decode_k_rows'])==84
  lang=self.layout['language'][1]-self.layout['language'][0]+1; expected=1+int(self.fg.sum())+lang
  assert all(x['rows']==expected for x in self.d['k_rows'])
  bg=torch.as_tensor(np.flatnonzero(self.bg)+1,dtype=torch.long,device=cache_layers(cache)[0][0].device)
  for i in LAYERS:assert torch.equal(cache_layers(cache)[i][0][:,:,bg,:],cache_layers(self.source)[i][0][:,:,bg,:])

def metrics(a,b):
 x=a.float()-b.float();return {'max_abs':float(x.abs().max()),'mean_abs':float(x.abs().mean()),'relative_l2':float(torch.linalg.vector_norm(x)/torch.linalg.vector_norm(a.float()).clamp_min(1e-20))}
def rec(step,reuse,fg,ctx,src_step,pred):
 bg=~fg;return {'step':step,'type':'reuse' if reuse else 'refresh','source_step':src_step if reuse else None,'source_age':step-src_step if reuse else 0,'foreground_tokens':int(fg.sum()),'background_tokens':int(bg.sum()),'foreground_hash':hashlib.sha256(fg.tobytes()).hexdigest()[:16],'foreground_indices':np.flatnonzero(fg).tolist(),'projected_k_rows_per_selected_layer': (ctx.d['k_rows'][0]['rows'] if reuse and 'k_rows' in ctx.d else pred['prefill_length']),'reused_k_rows_per_selected_layer':int(bg.sum()) if reuse else 0,'provenance':'fg=current,bg=most_recent_fresh_refresh' if reuse else 'all_visual=current','prefill_length':pred['prefill_length'],'action_tokens':pred['action_token_ids'],'policy_action':pred['policy_action'].tolist(),'prefill_last_logits':pred['logits'][-1,:].float().cpu().tolist()}
def run_mode(mode,model,processor,device,key,steps=6):
 seed_everything(7);suite=benchmark.get_benchmark_dict()[TASK_SUITE_NAME]();task=suite.get_task(0);states=suite.get_task_init_states(0);e=seg_env(task);prompt=f'In: What action should the robot take to {task.language.lower()}?\nOut:'; rows=[];src=None
 try:
  e.reset();obs=e.set_init_state(states[0]);mapping=rendered_label_mapping(e.env.model.instances_to_ids)
  for _ in range(10):obs,_,_,_=e.step(DEFAULT_STEP_ACTION)
  for step in range(steps):
   fg=mask_from_obs(obs,mapping);reuse=step%5!=0; image=policy_image_from_observation(obs,True)
   if reuse:
    assert src is not None and src['step']==step-step%5
    ctx=MaskedOverwrite(src['cache'],src['layout'],fg) if mode=='overwrite' else MaskedSkip(model,src['cache'],src['layout'],fg)
    pred=capture_prediction(model,processor,image,prompt,device,key,ctx);ctx.validate(pred['cache'])
   else:
    pred=capture_prediction(model,processor,image,prompt,device,key);src={'cache':clone_prefill_cache(pred['cache']),'layout':dict(pred['layout']),'step':step};ctx=None
   rows.append(rec(step,reuse,fg,ctx,src['step'],pred));act=normalize_and_invert_openvla_gripper(pred['policy_action']);obs,_,done,_=e.step(act.tolist())
   if done:break
 finally:e.close()
 return rows
def main():
 OUT.mkdir(parents=True,exist_ok=True);model,processor,device=load_model_and_processor(DEFAULT_MODEL_ID);key=resolve_unnorm_key(model,None)
 ow=run_mode('overwrite',model,processor,device,key);sk=run_mode('skip',model,processor,device,key)
 paired=[]
 for a,b in zip(ow,sk):
  q={'step':a['step'],'same_schedule':a['type']==b['type'] and a['source_age']==b['source_age'],'same_mask':a['foreground_hash']==b['foreground_hash'],'same_row_provenance':a['foreground_tokens']==b['foreground_tokens'] and a['background_tokens']==b['background_tokens'] and a['reused_k_rows_per_selected_layer']==b['reused_k_rows_per_selected_layer'],'token_match':a['action_tokens']==b['action_tokens'],'action':{'max_abs':float(np.max(np.abs(np.asarray(a['policy_action'])-np.asarray(b['policy_action'])))),'l2':float(np.linalg.norm(np.asarray(a['policy_action'])-np.asarray(b['policy_action'])))},'prefill_last_logits':({'max_abs':float(np.max(np.abs(np.asarray(a['prefill_last_logits'])-np.asarray(b['prefill_last_logits'])))),'mean_abs':float(np.mean(np.abs(np.asarray(a['prefill_last_logits'])-np.asarray(b['prefill_last_logits'])))),'relative_l2':float(np.linalg.norm(np.asarray(a['prefill_last_logits'])-np.asarray(b['prefill_last_logits']))/max(np.linalg.norm(np.asarray(a['prefill_last_logits'])),1e-20))})}
  paired.append(q)
 out={'config':{'task':0,'state':0,'seed':7,'R':5,'layers':list(LAYERS),'visual_positions':[1,256],'steps_requested':6},'overwrite':ow,'true_skip':sk,'paired':paired,'valid':all(x['same_schedule'] and x['same_mask'] and x['same_row_provenance'] for x in paired)}
 (OUT/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
