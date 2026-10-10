"""CUDA-event timing: Fresh vs visual overwrite vs genuine visual-K skip."""
import json,numpy as np,torch
from pathlib import Path
from PIL import Image
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import build_task_and_env_for,policy_image_from_observation,capture_prediction,seed_everything
from experiments.kv_cache.experiments.prefill_reuse_experiment import DuringPrefillVisualKReuse
from experiments.kv_cache.experiments.visual_k_projection_skip import DuringPrefillVisualKProjectionSkip
from experiments.kv_cache.experiments.static_reuse_experiment import load_model_and_processor,resolve_unnorm_key,clone_prefill_cache,prepare_inputs,validate_compatible_prefills
OUT=Path('experiments/kv_cache/outputs/stage2d_visual_k_skip_timing/timing_results.json');WARMUP=30;REPS=100;LAYERS=tuple(range(14))
class Null:
 def __enter__(self):return self
 def __exit__(self,*x):return None
def stats(x):
 a=np.asarray(x);return {'mean_ms':float(a.mean()),'median_ms':float(np.median(a)),'std_ms':float(a.std(ddof=1)),'p10_ms':float(np.percentile(a,10)),'p90_ms':float(np.percentile(a,90)),'n':len(x)}
def invoke(make,fn):
 with make(),torch.inference_mode():return fn()
def measure(make,fn):
 x=[]
 for _ in range(REPS):
  torch.cuda.synchronize();s,e=torch.cuda.Event(True),torch.cuda.Event(True);s.record();invoke(make,fn);e.record();e.synchronize();x.append(s.elapsed_time(e))
 return stats(x)
def main():
 seed_everything(7);model,processor,device=load_model_and_processor('openvla/openvla-7b-finetuned-libero-spatial');key=resolve_unnorm_key(model,None); rows=[]
 for task in (0,4,5,7):
  seed_everything(7);_,_,states,env,desc=build_task_and_env_for(task);env.reset();im=policy_image_from_observation(env.set_init_state(states[0]),True);env.close();prompt=f'In: What action should the robot take to {desc.lower()}?\nOut:';inputs=prepare_inputs(processor,prompt,Image.fromarray(im).convert('RGB'),device);fresh=capture_prediction(model,processor,im,prompt,device,key);source=clone_prefill_cache(fresh['cache']);layout=fresh['layout']
  makes={'fresh':lambda:Null(),'overwrite':lambda:DuringPrefillVisualKReuse(source,layout,LAYERS),'true_skip':lambda:DuringPrefillVisualKProjectionSkip(model,source,layout,LAYERS)}
  # Validate real intervention paths outside timed region.
  ow=makes['overwrite']();overwrite=capture_prediction(model,processor,im,prompt,device,key,ow);ow.validate_completed_prefill(overwrite['cache']);sk=makes['true_skip']();skip=capture_prediction(model,processor,im,prompt,device,key,sk);sk.validate(skip['cache']);validate_compatible_prefills(source,skip['cache'],fresh['input_ids'],skip['input_ids'],layout,skip['layout'],LAYERS)
  assert all(q['rows']==int(layout['prefill_length']) for q in sk.diagnostics['q_rows']+sk.diagnostics['v_rows']);assert all(q['rows']==1+(layout['language'][1]-layout['language'][0]+1) for q in sk.diagnostics['k_rows'])
  modes={}
  for name,make in makes.items():
   for _ in range(WARMUP):invoke(make,lambda:model(**inputs))
   pre=measure(make,lambda:model(**inputs))
   for _ in range(WARMUP):invoke(make,lambda:model.predict_action(**inputs,unnorm_key=key,do_sample=False))
   full=measure(make,lambda:model.predict_action(**inputs,unnorm_key=key,do_sample=False))
   modes[name]={'prefill':pre,'predict_action':full}
  L=int(layout['prefill_length']);rows.append({'task':task,'layout':layout,'language_tokens':layout['language'][1]-layout['language'][0]+1,'structural_validation':{'omitted_rows_per_reuse_prefill':3584,'skip_k_rows':sk.diagnostics['k_rows'],'qv_full_rows':sk.diagnostics['q_rows']+sk.diagnostics['v_rows']},'theoretical_k_row_fraction_removed':14*256/(32*L),'theoretical_qkv_fraction_removed':14*256/(32*L*3),'modes':modes})
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps({'warmup':WARMUP,'repetitions':REPS,'results':rows},indent=2))
if __name__=='__main__':main()
