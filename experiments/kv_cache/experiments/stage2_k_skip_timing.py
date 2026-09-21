"""CUDA-event microbenchmark for fresh, Stage-1 K overwrite, and Stage-2 K skip."""
import json, statistics, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import build_task_and_env_for, policy_image_from_observation, capture_prediction, seed_everything, REUSE_LAYERS
from experiments.kv_cache.experiments.prefill_reuse_experiment import DuringPrefillLanguageReuse, DuringPrefillLanguageKProjectionSkip
from experiments.kv_cache.experiments.static_reuse_experiment import load_model_and_processor, resolve_unnorm_key, clone_prefill_cache, prepare_inputs

OUT = Path("experiments/kv_cache/outputs/stage2_k_skip_timing.json")
WARMUP, REPS = 30, 100
def stats(xs):
 a=np.asarray(xs); return {"mean_ms":float(a.mean()),"median_ms":float(np.median(a)),"std_ms":float(a.std(ddof=1)),"p10_ms":float(np.percentile(a,10)),"p90_ms":float(np.percentile(a,90)),"n":len(xs)}
def timed(call, n):
 out=[]
 for _ in range(n):
  torch.cuda.synchronize(); s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record(); call(); e.record(); e.synchronize(); out.append(s.elapsed_time(e))
 return out
class Null:
 def __enter__(self): return self
 def __exit__(self,*x): return None
def main():
 seed_everything(7); model,processor,device=load_model_and_processor("openvla/openvla-7b-finetuned-libero-spatial"); key=resolve_unnorm_key(model,"libero_spatial"); rows=[]
 for task in (0,4,5,7):
  seed_everything(7); _,_,states,env,desc=build_task_and_env_for(task); env.reset(); im=policy_image_from_observation(env.set_init_state(states[0]),True); env.close(); prompt=f"In: What action should the robot take to {desc.lower()}?\\nOut:"
  inputs=prepare_inputs(processor,prompt,Image.fromarray(im).convert("RGB"),device); fresh=capture_prediction(model,processor,im,prompt,device,key); source=clone_prefill_cache(fresh["cache"]); layout=fresh["layout"]
  contexts={"fresh":lambda:Null(),"stage1":lambda:DuringPrefillLanguageReuse(source,layout,REUSE_LAYERS,"k"),"stage2":lambda:DuringPrefillLanguageKProjectionSkip(model,source,layout,REUSE_LAYERS)}
  correct={}
  for name,make in contexts.items():
   with make():
    with torch.inference_mode(): action=model.predict_action(**inputs,unnorm_key=key,do_sample=False)
   correct[name]=np.asarray(action)
  assert np.array_equal(correct["stage1"],correct["stage2"])
  modes={}
  for name,make in contexts.items():
   for _ in range(WARMUP):
    with make(),torch.inference_mode(): model(**inputs)
   pre=[]
   for _ in range(REPS):
    with make(),torch.inference_mode(): pre.extend(timed(lambda:model(**inputs),1))
   for _ in range(WARMUP):
    with make(),torch.inference_mode(): model.predict_action(**inputs,unnorm_key=key,do_sample=False)
   full=[]
   for _ in range(REPS):
    with make(),torch.inference_mode(): full.extend(timed(lambda:model.predict_action(**inputs,unnorm_key=key,do_sample=False),1))
   modes[name]={"prefill":stats(pre),"predict_action":stats(full)}
  ls,le=layout["language"]; rows.append({"task":task,"layout":layout,"language_rows":le-ls+1,"prefix_rows":ls,"modes":modes,"stage1_stage2_actions_equal":bool(np.array_equal(correct["stage1"],correct["stage2"]))})
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps({"warmup":WARMUP,"repetitions":REPS,"results":rows},indent=2)+"\n")
if __name__=="__main__": main()
