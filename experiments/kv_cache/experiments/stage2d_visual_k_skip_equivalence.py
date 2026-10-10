import json,torch
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import build_task_and_env_for,policy_image_from_observation,capture_prediction,seed_everything
from experiments.kv_cache.experiments.prefill_reuse_experiment import DuringPrefillVisualKReuse
from experiments.kv_cache.experiments.visual_k_projection_skip import DuringPrefillVisualKProjectionSkip
from experiments.kv_cache.experiments.static_reuse_experiment import load_model_and_processor,resolve_unnorm_key,clone_prefill_cache,cache_layers
def met(a,b):
 d=a.float()-b.float();return {'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'relative_l2':float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a.float()).clamp_min(1e-12))}
def main():
 seed_everything(7);m,p,d=load_model_and_processor('openvla/openvla-7b-finetuned-libero-spatial');key=resolve_unnorm_key(m,None);_,_,states,e,desc=build_task_and_env_for(0);e.reset();o=e.set_init_state(states[0]);im=policy_image_from_observation(o,True);e.close();prompt=f'In: What action should the robot take to {desc.lower()}?\nOut:';fresh=capture_prediction(m,p,im,prompt,d,key);src=clone_prefill_cache(fresh['cache']);layers=tuple(range(14));a=DuringPrefillVisualKReuse(src,fresh['layout'],layers);one=capture_prediction(m,p,im,prompt,d,key,a);a.validate_completed_prefill(one['cache']);b=DuringPrefillVisualKProjectionSkip(m,src,fresh['layout'],layers);two=capture_prediction(m,p,im,prompt,d,key,b);b.validate(two['cache']);res={'token_ids_equal':one['action_token_ids']==two['action_token_ids'],'policy_action':met(torch.tensor(one['policy_action']),torch.tensor(two['policy_action'])),'logits':met(one['logits'],two['logits']),'layers':[{'layer':i,'k':met(cache_layers(one['cache'])[i][0],cache_layers(two['cache'])[i][0]),'v':met(cache_layers(one['cache'])[i][1],cache_layers(two['cache'])[i][1])} for i in layers],'rows':b.diagnostics};print(json.dumps(res))
if __name__=='__main__':main()
