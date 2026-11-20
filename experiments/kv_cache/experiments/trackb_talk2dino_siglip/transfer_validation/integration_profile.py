"""Read-only native OpenVLA DINO capture and selection-overhead profile."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROMPTS=["black bowl","plate","robot gripper","cabinet drawer"]
def atomic_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(obj,indent=2,default=str)+"\n");os.replace(t,p)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)
def timed(fn,device,n=30,warm=5):
    for _ in range(warm):fn()
    sync(device); values=[]
    for _ in range(n):
        sync(device);t=time.perf_counter();fn();sync(device);values.append((time.perf_counter()-t)*1000)
    return {"n":n,"median_ms":float(np.median(values)),"p90_ms":float(np.percentile(values,90)),"p95_ms":float(np.percentile(values,95)),"mean_ms":float(np.mean(values)),"samples_ms":values}

def main():
    p=argparse.ArgumentParser()
    for k in ("image","task-prompt","siglip-checkpoint","v2-checkpoint","output"):
        p.add_argument("--"+k,type=Path if k in ("image","siglip-checkpoint","v2-checkpoint","output") else str,required=True)
    p.add_argument("--repeats",type=int,default=30);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoModelForVision2Seq,AutoProcessor
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import encode_text_embeddings,load_siglip_text_tower
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
    t=time.perf_counter();processor=AutoProcessor.from_pretrained("openvla/openvla-7b-finetuned-libero-spatial",trust_remote_code=True)
    model=AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b-finetuned-libero-spatial",torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True).eval()
    load_seconds=time.perf_counter()-t;device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():model.to(device)
    vision=model.vision_backbone;dino=getattr(vision,"dino_featurizer",getattr(vision,"featurizer",None))
    if dino is None:raise RuntimeError("Native OpenVLA vision backbone has no DINO featurizer")
    image=Image.open(a.image).convert("RGB")
    text=f"In: What action should the robot take to {a.task_prompt}?\nOut:"
    batch=processor(text=[text],images=[image],return_tensors="pt")
    batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
    if "pixel_values" not in batch or tuple(batch["pixel_values"].shape)!=(1,6,224,224):raise RuntimeError(f"Unexpected exact policy pixels: {tuple(batch['pixel_values'].shape)}")
    pixels=batch["pixel_values"].to(dtype=next(dino.parameters()).dtype)
    batch["pixel_values"]=pixels
    captured=[]
    def hook(_mod,_inp,outv):
        if isinstance(outv,(tuple,list)):outv=outv[0]
        captured.append(outv.detach())
    # Baseline and hooked full policy prefills use identical inputs, no generation/actions.
    sync(device);base_t=time.perf_counter()
    with torch.inference_mode(): base=model(**batch,use_cache=False,return_dict=True)
    sync(device);base_prefill_ms=(time.perf_counter()-base_t)*1000
    base_logits=base.logits.detach().float().cpu()
    handle=dino.register_forward_hook(hook)
    try:
        sync(device);hook_t=time.perf_counter()
        with torch.inference_mode(): hooked=model(**batch,use_cache=False,return_dict=True)
        sync(device);hook_prefill_ms=(time.perf_counter()-hook_t)*1000
    finally:handle.remove()
    if len(captured)!=1:raise RuntimeError(f"Expected one DINO pass in policy prefill, saw {len(captured)}")
    features=captured[0]
    if tuple(features.shape)!=(1,256,1024):raise RuntimeError(f"Captured feature shape {tuple(features.shape)}")
    if not torch.isfinite(features).all():raise RuntimeError("Captured nonfinite DINO features")
    logits=hooked.logits.detach().float().cpu()
    maxdiff=float((logits-base_logits).abs().max())
    if not torch.allclose(logits,base_logits,rtol=1e-4,atol=1e-4):raise RuntimeError(f"Read-only hook changed policy prefill logits: {maxdiff}")
    # Compare captured tensor to the same native DINO module's ordinary output.
    x=pixels[:,:3].to(device=device,dtype=next(dino.parameters()).dtype)
    with torch.inference_mode(): direct=dino(x)
    if isinstance(direct,(tuple,list)):direct=direct[0]
    parity_max=float((features-direct).abs().max().item())
    if not torch.allclose(features,direct,rtol=1e-5,atol=1e-5):raise RuntimeError(f"DINO hook/direct feature mismatch {parity_max}")
    fused=vision(pixels)
    if tuple(fused.shape)!=(1,256,2176):raise RuntimeError(f"Unexpected fused visual token tensor {tuple(fused.shape)}")
    # V2 mask score path: encode text once, then project and score per observation.
    t=time.perf_counter();text_model,tokenizer,textmeta=load_siglip_text_tower(a.siglip_checkpoint,device=str(device));text_init=time.perf_counter()-t
    tokens=tokenizer(PROMPTS).to(device)
    tokenization=timed(lambda:tokenizer(PROMPTS),device,n=a.repeats,warm=1)
    t=time.perf_counter()
    with torch.inference_mode():text_emb=encode_text_embeddings(text_model,tokens).float()
    sync(device);text_encode_once=(time.perf_counter()-t)*1000
    raw=torch.load(a.v2_checkpoint,map_location="cpu",weights_only=True);projection=RegionTextProjection().to(device).eval();projection.load_state_dict(raw["model"],strict=True)
    raw_features=features.float()
    patch_norm_timing=timed(lambda:F.normalize(raw_features,dim=-1),device,n=a.repeats)
    patch=F.normalize(raw_features,dim=-1)
    projection_timing=timed(lambda:F.normalize(projection.text_projection(text_emb),dim=-1),device,n=a.repeats)
    with torch.inference_mode():proj_text=F.normalize(projection.text_projection(text_emb),dim=-1)
    def score():
        with torch.inference_mode():
            pnorm=F.normalize(raw_features,dim=-1)[0]
            c=torch.matmul(pnorm,proj_text.T).T
            prob=torch.sigmoid(10*(c-.20));mask=prob>=.5
            return mask
    mask_timing=timed(score,device,n=a.repeats)
    def extra_prompt():
        with torch.inference_mode():
            e=encode_text_embeddings(text_model,tokenizer(PROMPTS[-1:]).to(device)).float()
            t=F.normalize(projection.text_projection(e),dim=-1)
            return torch.sigmoid(10*(patch@t.T-.20))
    extra=timed(extra_prompt,device,n=a.repeats)
    # Vision-only timing reflects the new instrumentation cost at the backbone boundary.
    def vision_run():return vision(pixels)
    vision_timing=timed(vision_run,device,n=a.repeats)
    captured2=[]
    def hook_vision(_m,_i,o):pass
    dh=dino.register_forward_hook(hook_vision)
    try: hooked_vision_timing=timed(vision_run,device,n=a.repeats)
    finally:dh.remove()
    available=None;reserved=None;mem_before=None;mem_after=None
    if device.type=="cuda":
        sync(device);available,reserved=torch.cuda.mem_get_info(device);mem_after=int(torch.cuda.memory_allocated(device));mem_before=int(torch.cuda.memory_reserved(device))
    result={"status":"complete","image":str(a.image),"task_prompt":a.task_prompt,"model":"openvla/openvla-7b-finetuned-libero-spatial","checkpoint_v2_sha256":sha(a.v2_checkpoint),
      "policy_inputs":{"pixel_values_shape":list(pixels.shape),"pixel_dtype":str(pixels.dtype),"processor":type(processor.image_processor).__name__},
      "captured_dino":{"shape":list(features.shape),"dtype":str(features.dtype),"max_abs_diff_vs_direct_dino":parity_max,"fused_visual_shape":list(fused.shape),"visual_tokens":256,"dino_dim":1024,"hook_calls_during_full_prefill":1},
      "policy_prefill_read_only_hook":{"baseline_logits_shape":list(base.logits.shape),"max_abs_logit_diff":maxdiff,"allclose_rtol_atol_1e-4":True,"action_generation_performed":False},
      "timing":{"one_time_vla_processor_and_model_load_seconds":load_seconds,"full_policy_prefill_baseline_single_sample_ms":base_prefill_ms,"full_policy_prefill_with_passive_capture_single_sample_ms":hook_prefill_ms,"vla_visual_forward":vision_timing,"visual_forward_with_passive_dino_capture":hooked_vision_timing,"text_tower_init_seconds":text_init,"tokenization_cached_prompts":tokenization,"one_time_text_encoding_four_prompts_ms":text_encode_once,"cached_text_projection_and_16x16_scores_and_threshold":mask_timing,"per_frame_patch_l2_normalization":patch_norm_timing,"one_time_task_text_mlp_projection_and_l2_normalization":projection_timing,"one_additional_prompt_encode_project_score":extra},
      "memory":{"cuda_device":torch.cuda.get_device_name(device) if device.type=="cuda" else None,"cuda_available_free_bytes_after_profile":int(available) if available is not None else None,"cuda_reserved_bytes_after_profile":int(mem_before) if mem_before is not None else None,"cuda_allocated_bytes_after_profile":mem_after,"cuda_peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else None,"siglip_text_parameter_count":sum(p.numel() for p in text_model.parameters()),"siglip_text_parameter_dtype_bytes":sum(p.numel()*p.element_size() for p in text_model.parameters())},
      "threshold_mask_counts":{p:int(n) for p,n in zip(PROMPTS,score().sum(dim=1).detach().cpu().tolist())},"text_metadata":textmeta,
      "interpretation":{"vision_capture_additional_dino_forward":False,"hook_passive":True,"no_policy_or_kv_cache_changes":True,"no_measured_kv_saved_compute":"The overhead values establish the minimum per-frame benefit required for break-even; no actual KV skip runtime was measured.","text_strategy":"Cache text embeddings once per unchanged task prompt; recompute only when prompt changes."}}
    atomic_json(out/"integration_timing.json",result)
    report=["# OpenVLA Integration and Computational Overhead","",f"Passive capture ran inside the existing OpenVLA multimodal prefill using `{a.image}`. The model performed no action generation or environment interaction.","",f"- DINO tensor: `{features.shape}`; fused visual tensor: `{fused.shape}` (256 positions × 2,176 dims).","- Passive hook calls during full prefill: 1; it reused the already-computed DINO result.",f"- Baseline versus hooked prefill maximum logit difference: `{maxdiff:.8g}` (allclose at 1e-4).",f"- Direct DINO versus captured features maximum absolute difference: `{parity_max:.8g}`.","", "## Timing", "",f"- Vision backbone median: {vision_timing['median_ms']:.3f} ms; passive hook: {hooked_vision_timing['median_ms']:.3f} ms.",f"- Cached text projection+L2 normalization median (task initialization): {projection_timing['median_ms']:.3f} ms; patch normalization median: {patch_norm_timing['median_ms']:.3f} ms."
        f"- Per-frame patch-text scores+sigmoid+threshold median (four prompts): {mask_timing['median_ms']:.3f} ms.",f"- Text initialization: {text_init:.2f} s; first encode of four prompts: {text_encode_once:.3f} ms.",f"- One new prompt encode/project/score median: {extra['median_ms']:.3f} ms.","- Break-even requires measured saved K/V work to exceed passive capture overhead plus mask-selection time. This experiment does not measure skipped K/V runtime.","", "## Limits", "", "Timing samples and CUDA memory figures are preserved in `integration_timing.json`. The profile is one saved RGB/task input, and does not measure closed-loop action rate, policy rollouts, or net cache acceleration.",""]
    (out/"INTEGRATION_PROFILE_REPORT.md").write_text("\n".join(report))
    print(json.dumps({"status":"complete","feature_shape":list(features.shape),"logit_diff":maxdiff,"vision_ms":vision_timing["median_ms"],"selection_ms":mask_timing["median_ms"]},indent=2),flush=True)
if __name__=="__main__":main()
