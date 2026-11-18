#!/usr/bin/env python3
"""Sequential, resumable supervisor for the expanded val2014 evaluation."""
import json, os, subprocess, sys, time, fcntl
from pathlib import Path

ROOT=Path('/opt/openvla/cache/worktrees/trackb-talk2dino-siglip')
SRC=ROOT/'experiments/kv_cache/experiments/trackb_talk2dino_siglip'
OUT=ROOT/'experiments/kv_cache/outputs/trackb_talk2dino_siglip/full_coco_training_20261117/postprocessing_optimization/full_val2014_holdout'
DATA=Path('/opt/openvla/cache/datasets/talk2dino/coco2014')
FEATURES=Path('/opt/openvla/cache/datasets/talk2dino/coco2014/features_full_validation_20261117')
PYTHON=sys.executable
LOG=OUT/'logs/supervisor.log'
STATUS=OUT/'run_status.json'

def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def run(stage,cmd,detail):
    atomic(STATUS,{'status':'running','stage':stage,'detail':detail,'pid':os.getpid(),'updated_unix':time.time(),'command':cmd})
    print(f'\n[{time.strftime("%F %T")}] START {stage}: {detail}\n$ {" ".join(cmd)}',flush=True)
    subprocess.run(cmd,cwd=ROOT,env=ENV,check=True)
    atomic(STATUS,{'status':'running','stage':stage,'detail':detail,'pid':os.getpid(),'updated_unix':time.time(),'last_stage_exit_code':0})
    print(f'[{time.strftime("%F %T")}] COMPLETE {stage}',flush=True)

def validate_features():
    import numpy as np
    expected={'image_ids_full_val2014.npy':(40504,),'region_features_full_val2014_fp16.npy':(40504,16,1024),'region_done_full_val2014.npy':(40504,),
      'caption_ids_val2014.npy':(202654,2),'text_features_val2014_fp16.npy':(202654,1152),'text_done_val2014.npy':(202654,),
      'patch_features_spatial_val2014.npy':(2500,256,1024),'attention_maps_spatial_val2014.npy':(2500,16,256),'spatial_image_ids_val2014.npy':(2500,),'spatial_done_val2014.npy':(2500,)}
    result={}
    for name,shape in expected.items():
        array=np.load(FEATURES/name,mmap_mode='r')
        if array.shape!=shape: raise RuntimeError(f'{name}: shape={array.shape}, expected={shape}')
        if 'done' in name:
            if not np.asarray(array).all(): raise RuntimeError(f'{name} incomplete')
        elif array.dtype.kind=='f' and not np.isfinite(array).all(): raise RuntimeError(f'{name} contains nonfinite values')
        result[name]={'shape':list(array.shape),'dtype':str(array.dtype),'complete':True}
    ids=np.load(FEATURES/'image_ids_full_val2014.npy',mmap_mode='r')
    caps=np.load(FEATURES/'caption_ids_val2014.npy',mmap_mode='r')
    if len(set(map(int,ids)))!=40504 or set(map(int,caps[:,1]))!=set(map(int,ids)): raise RuntimeError('Caption-image mapping mismatch')
    atomic(OUT/'FULL_FEATURE_VALIDATION.json',{'status':'complete','arrays':result,'unique_images':40504,'captions':202654,'all_caption_image_ids_resolve':True})

def main():
    OUT.joinpath('logs').mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'logs/supervisor.lock','w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Another expanded val2014 supervisor is active.')
    os.chdir(ROOT)
    log_stream=open(LOG,'a',buffering=1)
    os.dup2(log_stream.fileno(),1); os.dup2(1,2)
    os.environ['PYTHONPATH']='/tmp/talk2dino-benchmark-deps:'+str(ROOT)
    global ENV
    ENV=os.environ.copy()
    atomic(OUT/'logs/supervisor.pid',{'pid':os.getpid(),'container':'72cde43b85db','started_unix':time.time()})
    try:
        run('preflight',[PYTHON,'-c',"import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"],'CUDA runtime and fixed split manifests')
        run('visual_extraction',[PYTHON,'-u',str(SRC/'extract_val2014_full.py'),'--dataset-root',str(DATA),'--feature-root',str(FEATURES),'--segmentation-manifest',str(OUT/'segmentation_manifest.json'),'--batch-size','1'],'all val2014 images, plus dense spatial cache for 2500 segmentation images')
        run('text_extraction',[PYTHON,'-u',str(SRC/'extract_text.py'),'--dataset-root',str(DATA),'--feature-root',str(FEATURES),'--split','val','--batch-size','128'],'all 202654 val2014 captions')
        run('feature_integrity', [PYTHON,'-c','from experiments.kv_cache.experiments.trackb_talk2dino_siglip.run_expanded_val2014_validation import validate_features; validate_features()'],'shapes, finite features, completion flags and ID joins')
        run('retrieval',[PYTHON,'-u',str(SRC/'full_val2014_retrieval.py'),'--feature-root',str(FEATURES),'--split-manifest',str(OUT/'validation_split_manifest.json'),'--output-dir',str(OUT),'--scratch-dir',str(FEATURES/'retrieval_scratch'),'--device','cuda:0','--image-chunk','64','--caption-chunk','8192'],'exhaustive multi-positive retrieval at epochs 20, 50 and 100')
        run('segmentation',[PYTHON,'-u',str(SRC/'postprocess_optimization.py'),'--dataset-root',str(DATA),'--manifest',str(OUT/'segmentation_manifest.json'),'--feature-root',str(FEATURES),'--output-dir',str(OUT/'segmentation'),'--device','cuda:0','--resume'],'500 dev images for thresholds; 2000 held-out test images')
        run('libero_feasibility',[PYTHON,'-u',str(SRC/'libero_mask_feasibility.py'),'--dataset-root',str(DATA),'--optimization-dir',str(OUT/'segmentation'),'--output-dir',str(OUT/'libero_feasibility'),'--device','cuda:0'],'offline frozen-mask review on saved LIBERO images; no policy or KV experiment')
        run('report',[PYTHON,'-u',str(SRC/'finalize_expanded_validation.py'),'--output-dir',str(OUT)],'aggregate retrieval and segmentation artifacts')
        atomic(STATUS,{'status':'complete','stage':'complete','exit_code':0,'finished_unix':time.time(),'pid':os.getpid()})
        print(f'\n[{time.strftime("%F %T")}] ALL VALIDATION STAGES COMPLETE',flush=True)
    except BaseException as error:
        atomic(STATUS,{'status':'failed','stage':STATUS.exists() and json.loads(STATUS.read_text()).get('stage'),'error':repr(error),'pid':os.getpid(),'updated_unix':time.time()})
        print(f'FAILED: {error!r}',flush=True)
        raise
if __name__=='__main__':main()
