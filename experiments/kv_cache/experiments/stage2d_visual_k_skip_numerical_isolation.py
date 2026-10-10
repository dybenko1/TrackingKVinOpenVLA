import json,torch
from pathlib import Path
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import build_task_and_env_for,policy_image_from_observation,capture_prediction,seed_everything
from experiments.kv_cache.experiments.static_reuse_experiment import load_model_and_processor,resolve_unnorm_key
def met(a,b):
 d=(a-b).float();return {'exact':bool(torch.equal(a,b)),'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'relative_l2':float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a.float()).clamp_min(1e-12)),'unequal_elements':int((a!=b).sum()),'fraction_unequal':float((a!=b).float().mean())}
def main():
 seed_everything(7);m,p,d=load_model_and_processor('openvla/openvla-7b-finetuned-libero-spatial');key=resolve_unnorm_key(m,None);_,_,states,e,desc=build_task_and_env_for(0);e.reset();o=e.set_init_state(states[0]);im=policy_image_from_observation(o,True);e.close();prompt=f'In: What action should the robot take to {desc.lower()}?\nOut:';box={};proj=m.language_model.model.layers[0].self_attn.k_proj
 def observe(_m,args):
  if 'h' not in box: box['h']=args[0].detach().clone()
 h=proj.register_forward_pre_hook(observe)
 capture_prediction(m,p,im,prompt,d,key);h.remove();H=box['h'];idx=torch.cat((torch.zeros(1,dtype=torch.long,device=d),torch.arange(257,H.shape[1],device=d)))
 with torch.inference_mode(): A=proj(H);B=proj(H.index_select(1,idx));S=A.index_select(1,idx);out={'shape':list(H.shape),'selected_rows':int(idx.numel()),'full_vs_gather':met(S,B),'bos':met(S[:,:1],B[:,:1]),'language':met(S[:,1:],B[:,1:]),'full_repeat':met(A,proj(H))}
 with torch.inference_mode():
  w=proj.weight.float();bias=proj.bias.float() if proj.bias is not None else None;Af=torch.nn.functional.linear(H.float(),w,bias).index_select(1,idx);Bf=torch.nn.functional.linear(H.index_select(1,idx).float(),w,bias);out['fp32_full_vs_gather']=met(Af,Bf)
 Path('experiments/kv_cache/outputs/stage2d_visual_k_skip_equivalence').mkdir(parents=True,exist_ok=True);Path('experiments/kv_cache/outputs/stage2d_visual_k_skip_equivalence/numerical_isolation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
