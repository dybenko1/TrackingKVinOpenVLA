#!/usr/bin/env python3
"""Resumable OpenVLA-native DINO extraction for full COCO val2014."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

MODULE=Path(__file__).resolve().parent
REPO=MODULE.parents[4]
sys.path.insert(0,str(REPO))
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.extract_visual import load_vision,process_images

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-root',type=Path,default=Path('/opt/openvla/cache/datasets/talk2dino/coco2014'))
    ap.add_argument('--feature-root',type=Path,required=True)
    ap.add_argument('--segmentation-manifest',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=1)
    ap.add_argument('--max-images',type=int,default=0)
    a=ap.parse_args(); a.feature_root.mkdir(parents=True,exist_ok=True)
    cap=json.loads((a.dataset_root/'annotations2014/captions_val2014.json').read_text())
    entries=[(int(x['id']),a.dataset_root/'val2014'/x['file_name']) for x in cap['images']]
    ids=np.asarray([x[0] for x in entries],np.int64)
    manifest=json.loads(a.segmentation_manifest.read_text())
    selected=[int(x['image_id']) for x in manifest['images']]
    if len(set(selected))!=len(selected): raise RuntimeError('Segmentation manifest IDs are duplicated')
    row_by_id={int(i):r for r,i in enumerate(ids)}
    if not set(selected)<=set(row_by_id): raise RuntimeError('Selected segmentation image missing from caption split')
    n=len(ids) if not a.max_images else min(len(ids),a.max_images)
    names={
      'all_ids':a.feature_root/'image_ids_full_val2014.npy','regions':a.feature_root/'region_features_full_val2014_fp16.npy',
      'done':a.feature_root/'region_done_full_val2014.npy','patches':a.feature_root/'patch_features_spatial_val2014.npy',
      'attention':a.feature_root/'attention_maps_spatial_val2014.npy','spatial_ids':a.feature_root/'spatial_image_ids_val2014.npy',
      'spatial_done':a.feature_root/'spatial_done_val2014.npy'}
    if names['all_ids'].exists():
      if not np.array_equal(np.load(names['all_ids'],mmap_mode='r'),ids[:n]): raise RuntimeError('Existing full val image ordering differs')
    else: np.save(names['all_ids'],ids[:n],allow_pickle=False)
    spatial_ids=np.asarray(selected,np.int64)
    if names['spatial_ids'].exists():
      if not np.array_equal(np.load(names['spatial_ids'],mmap_mode='r'),spatial_ids): raise RuntimeError('Spatial image order differs from frozen manifest')
    else: np.save(names['spatial_ids'],spatial_ids,allow_pickle=False)
    spatial_row={int(i):r for r,i in enumerate(spatial_ids)}
    shapes={'regions':((n,16,1024),np.float16),'done':((n,),np.uint8),
      'patches':((len(spatial_ids),256,1024),np.float16),'attention':((len(spatial_ids),16,256),np.float16),
      'spatial_done':((len(spatial_ids),),np.uint8)}
    mm={}
    for key,(shape,dtype) in shapes.items():
      p=names[key]; new=not p.exists()
      mm[key]=np.lib.format.open_memmap(p,mode='w+' if new else 'r+',dtype=dtype,shape=shape)
      if new: mm[key][:]=0
    model,processor,device,torch=load_vision()
    if device.type!='cuda': raise RuntimeError('Full val feature extraction requires the validated CUDA container')
    start=time.time(); done=mm['done']; sdone=mm['spatial_done']; failures=[]
    pending=[i for i in range(n) if not done[i] or (ids[i] in spatial_row and not sdone[spatial_row[int(ids[i])]])]
    total=len(pending)
    for pos,i in enumerate(pending,1):
      iid,path=entries[i]
      if not path.is_file(): raise FileNotFoundError(path)
      try:
        rows,regions,patches,attn=process_images(model,processor,[path],device,torch)
        if len(regions)!=1 or regions[0].shape!=(16,1024) or not np.isfinite(regions[0]).all(): raise RuntimeError('Invalid region output')
        mm['regions'][i]=regions[0]; done[i]=1
        if iid in spatial_row:
          sr=spatial_row[iid]
          if patches[0].shape!=(256,1024) or attn[0].shape!=(16,256): raise RuntimeError('Invalid spatial output')
          mm['patches'][sr]=patches[0]; mm['attention'][sr]=attn[0]; sdone[sr]=1
        if pos%10==0 or pos==total:
          for k in ('regions','done','patches','attention','spatial_done'): mm[k].flush()
          atomic_json(a.feature_root/'full_val2014_extraction_status.json',{'status':'running','total_images':n,
            'completed_region_images':int(np.asarray(done).sum()),'spatial_required':len(spatial_ids),
            'completed_spatial_images':int(np.asarray(sdone).sum()),'pending_this_run':total-pos,
            'last_image_id':iid,'elapsed_seconds':time.time()-start,'device':str(device),'errors':failures})
          print(f'val2014 features {int(np.asarray(done).sum())}/{n} spatial={int(np.asarray(sdone).sum())}/{len(spatial_ids)} last={iid}',flush=True)
      except Exception as e:
        failures.append({'image_id':iid,'error':repr(e)})
        atomic_json(a.feature_root/'full_val2014_extraction_status.json',{'status':'running_with_retries','total_images':n,
          'completed_region_images':int(np.asarray(done).sum()),'completed_spatial_images':int(np.asarray(sdone).sum()),
          'last_error':failures[-1],'errors':failures})
        raise
    for k in mm: mm[k].flush()
    complete=bool(np.asarray(done).all() and np.asarray(sdone).all())
    atomic_json(a.feature_root/'full_val2014_extraction_status.json',{'status':'complete' if complete else 'incomplete',
      'total_images':n,'completed_region_images':int(np.asarray(done).sum()),'spatial_required':len(spatial_ids),
      'completed_spatial_images':int(np.asarray(sdone).sum()),'source_caption_manifest_sha256':hashlib.sha256((a.dataset_root/'annotations2014/captions_val2014.json').read_bytes()).hexdigest(),
      'segmentation_manifest_sha256':hashlib.sha256(a.segmentation_manifest.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-start})
    if not complete: raise RuntimeError('Extraction ended with incomplete rows')
if __name__=='__main__': main()
