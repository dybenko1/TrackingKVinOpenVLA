#!/usr/bin/env python3
"""Apply the dev-selected frozen SigLIP/DINO postprocessing to saved LIBERO frames."""
import argparse,csv,json,re,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.kv_cache.experiments.trackb_talk2dino_siglip import phase3_evaluate as p3
from experiments.kv_cache.experiments.trackb_talk2dino_siglip import postprocess_optimization as post
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.extract_visual import load_vision,process_images
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',type=Path,default=Path('/opt/openvla/cache/datasets/talk2dino/coco2014'))
 ap.add_argument('--frames-root',type=Path,default=ROOT/'experiments/kv_cache/outputs/trackb_dinov2_segmentation_visual_trial/visualizations')
 ap.add_argument('--optimization-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--device',default='cuda:0')
 a=ap.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=True);dev=torch.device(a.device)
 selection=json.loads((a.optimization_dir/'development_selection.json').read_text())['selected_primary_native_grid16']
 thresholds=json.loads((a.optimization_dir/'threshold_calibration.json').read_text())['calibrated_thresholds']
 cp=selection['checkpoint'];method=selection['method'];threshold=float(thresholds[cp][method]['grid16']['threshold'])
 ckpath=post.CHECKPOINTS[cp];blob=torch.load(ckpath,map_location='cpu',weights_only=False)
 model=RegionTextProjection();model.load_state_dict(blob['model'],strict=True);model.to(dev).eval()
 frames=[]
 with (a.frames_root.parent/'frame_manifest.csv').open(newline='') as f: rows=list(csv.DictReader(f))
 for row in rows:
  image=a.frames_root/row['sample_id']/'camera_rgb.png'
  if not image.is_file():
   # Existing visual-trial images are nested under per-sample directories.
   matches=list(a.frames_root.rglob(row['sample_id']+'/camera_rgb.png'))
   if matches:image=matches[0]
  if image.is_file(): frames.append((row,image))
 if len(frames)<20: raise RuntimeError(f'Expected saved real LIBERO observations; found {len(frames)} RGB frames')
 model_v,processor,dino_device,torchmod=load_vision()
 if dino_device!=dev: raise RuntimeError(f'Vision device differs: {dino_device} vs {dev}')
 prompts=[]
 for row,_ in frames:
  inst=row['task_description'].lower()
  m=re.search(r'pick up the (.*?) (?:in|on|from|and place)',inst)
  target=m.group(1).strip() if m else 'manipulated object'
  prompts.append([target,'robot arm','robot gripper','table','background scene'])
 unique=sorted({p for ps in prompts for p in ps})
 p3.DATA=a.dataset_root;p3.SIGLIP=Path('/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384')
 tower,tokenizer,metadata=p3.load_siglip_text_tower(p3.SIGLIP,device=str(dev))
 txt=[]
 for phrase in unique:
  variants=[x.format(phrase) for x in post.TEMPLATES]
  tokens=tokenizer(variants).to(dev)
  with torch.inference_mode(): txt.append(p3.encode_text_embeddings(tower,tokens).mean(0))
 class_text=torch.stack(txt);pos={x:i for i,x in enumerate(unique)};del tower
 paths=[x[1] for x in frames];records,regions,patches,attns=process_images(model_v,processor,paths,dino_device,torchmod)
 if len(patches)!=len(frames): raise RuntimeError('DINO feature/frame count mismatch')
 pamr=p3.load_pamr(dev);indices=torch.arange(len(unique),device=dev)
 frame_outputs=[];outcsv=[];groups=defaultdict(list);start=time.time()
 for i,((row,path),patch,attn) in enumerate(zip(frames,patches,attns)):
  rgb=Image.open(path).convert('RGB');_,rgb_tensor=post.rgb224_tensor(rgb,dev)
  patch_t=torch.from_numpy(patch.astype(np.float32)).to(dev);attn_t=torch.from_numpy(attn.astype(np.float32)).to(dev)
  maps,_=post.make_maps(model,patch_t,attn_t,class_text,indices,rgb_tensor,pamr)
  phrase_list=prompts[i];indices_i=torch.tensor([pos[p] for p in phrase_list],device=dev)
  scores=maps[method]['grid16'][indices_i];masks=scores>=threshold
  target=(masks[0]|masks[1]|masks[2])
  item={'sample_id':row['sample_id'],'task_id':int(row['task_id']),'state_id':int(row['state_id']),'time_step':int(row['time_step']),
    'task_instruction':row['task_description'],'target_prompt':phrase_list[0],'selected_checkpoint':cp,'postprocessing':method,'threshold':threshold,
    'mask_counts':{phrase:int(mask.sum()) for phrase,mask in zip(phrase_list,masks)},'fresh_token_count_target_plus_robot_arm_gripper':int(target.sum()),
    'candidate_background_tokens':int(256-target.sum()),'ground_truth_available':False}
  frame_outputs.append(item);groups[(int(row['task_id']),int(row['state_id']))].append((int(row['time_step']),masks,phrase_list))
  outcsv.append(item)
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axs=plt.subplots(2,6,figsize=(20,7));axs[0,0].imshow(rgb);axs[0,0].set_title(f"{row['sample_id']}\nRGB")
  for j,(phrase,score,mask) in enumerate(zip(phrase_list,scores,masks),1):
   axs[0,j].imshow(rgb);heat=F.interpolate(torch.from_numpy(score)[None,None],(224,224),mode='nearest')[0,0].numpy()
   axs[0,j].imshow(heat,cmap='magma',alpha=.55,vmin=0,vmax=1);axs[0,j].set_title(f'{phrase} score')
   ax=axs[1,j];ax.imshow(rgb);visual=F.interpolate(mask.float()[None,None],(224,224),mode='nearest')[0,0].numpy()
   ax.imshow(np.ma.masked_where(visual<.5,visual),cmap='winter',alpha=.5,vmin=0,vmax=1);ax.set_title(f'{phrase} fresh mask: {int(mask.sum())}/256')
  union=F.interpolate((masks[0]|masks[1]|masks[2]).float()[None,None],(224,224),mode='nearest')[0,0].numpy()
  axs[1,0].imshow(rgb);axs[1,0].imshow(np.ma.masked_where(union<.5,union),cmap='cool',alpha=.48,vmin=0,vmax=1);axs[1,0].set_title('candidate fresh=target+robot')
  for ax in axs.ravel():ax.axis('off')
  fig.suptitle('Exploratory LIBERO mask feasibility — no policy or cache intervention');fig.tight_layout()
  fig.savefig(out/'visualizations'/f"{row['sample_id']}.png",dpi=130,bbox_inches='tight');plt.close(fig)
  print(f'LIBERO mask {i+1}/{len(frames)} {row["sample_id"]}',flush=True)
 temporal=[]
 for (task,state),items in groups.items():
  items.sort(key=lambda x:x[0])
  for a1,b1 in zip(items,items[1:]):
   step0,mask0,phrases=a1;step1,mask1,_=b1
   for j,phrase in enumerate(phrases):
    x=mask0[j].cpu().numpy().astype(bool);y=mask1[j].cpu().numpy().astype(bool);inter=np.logical_and(x,y).sum();union=np.logical_or(x,y).sum()
    temporal.append({'task_id':task,'state_id':state,'prompt':phrase,'source_step':step0,'current_step':step1,'frame_gap':step1-step0,
      'source_current_mask_iou':float(inter/union) if union else 1.,'flip_tokens':int(np.logical_xor(x,y).sum()),
      'source_fg_to_current_bg':int(np.logical_and(x,~y).sum()),'source_bg_to_current_fg':int(np.logical_and(~x,y).sum())})
 with (out/'per_frame_masks.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(outcsv[0]));w.writeheader();w.writerows(outcsv)
 with (out/'temporal_stability.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(temporal[0]));w.writeheader();w.writerows(temporal)
 summary={'status':'complete','frames':len(frames),'tasks':sorted({int(r['task_id']) for r,_ in frames}),'states':sorted({int(r['state_id']) for r,_ in frames}),
  'checkpoint':cp,'postprocessing':method,'threshold_from_coco_development_only':threshold,'prompt_set':unique,'per_frame':frame_outputs,
  'temporal_pairs':len(temporal),'mean_mask_iou_by_prompt':{p:float(np.mean([x['source_current_mask_iou'] for x in temporal if x['prompt']==p])) for p in unique},
  'mean_flip_tokens_by_prompt':{p:float(np.mean([x['flip_tokens'] for x in temporal if x['prompt']==p])) for p in unique},
  'ground_truth_masks_available':False,'limitation':'RGB observations come from prior LIBERO replay visualization; no paired oracle mask or simulator segmentation frame is available in these saved artifacts. Temporal snapshots are sparse early/middle/late frames, not adjacent control steps. No policy rollout or cache intervention was run.',
  'dino_model_load_reused_openvla_checkpoint':True,'text_checkpoint_metadata':metadata,'elapsed_seconds':time.time()-start}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (out/'LIBERO_FEASIBILITY_REPORT.md').write_text('# LIBERO mask feasibility (offline saved frames)\n\n'
  f"Applied frozen {cp} Talk2DINO-SigLIP projection and the development-selected `{method}` postprocessing (threshold {threshold:.2f}) to {len(frames)} saved RGB frames across tasks 4, 5, 7 and states 0–2. Prompts per frame include the instruction-derived manipulated object, robot arm, robot gripper, table, and background scene.\n\n"
  'This is a visualization and temporal-mask-stability probe only. It did not run policy inference, rollouts, or KV reuse. The saved frame set has no paired simulator oracle masks, so it cannot provide precision/recall/IoU against robot-object ground truth. Frame gaps are sparse early/middle/late snapshots; mask changes across them are not stepwise noise estimates. The score threshold is calibrated on COCO category prompts, not on these task-specific phrases. These masks are not yet validated as safe K/V reuse decisions.\n\n'
  'Inspect `visualizations/` for RGB, score maps and binary 16×16 fresh-token masks; `temporal_stability.csv` reports sparse mask IoU and token flips. `per_frame_masks.csv` records prompt token counts.\n')
 print(json.dumps({'frames':len(frames),'pairs':len(temporal),'output':str(out)},indent=2))
if __name__=='__main__':main()
