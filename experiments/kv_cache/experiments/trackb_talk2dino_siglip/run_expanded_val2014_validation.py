#!/usr/bin/env python3
"""Sequential, resumable supervisor for the expanded val2014 evaluation."""
import hashlib, json, os, subprocess, sys, time, fcntl
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

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def spatial_cache_matches_manifest():
    import numpy as np
    manifest=json.loads((OUT/'segmentation_manifest.json').read_text())
    ids=np.load(FEATURES/'spatial_image_ids_val2014.npy',mmap_mode='r')
    expected=np.asarray([int(row['image_id']) for row in manifest['images']],dtype=np.int64)
    return np.array_equal(ids,expected)

def retrieval_artifacts_complete():
    manifest_path=OUT/'validation_split_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest_hash=sha256_file(manifest_path)
    expected_checkpoint={
        20:'6364efd9f36a0341cbfb9e06c94b7245141d376c15817f7bc996136b9d305f38',
        50:'f1c57d5e3a52fde5ed6d17b5fad47aadaa6ce25dc58cab72a3aba9a180d015b1',
        100:'f07ec19c1660c155dbc8579692e96369c82e1cfe1fec0ac20d265249f25b2503',
    }
    for epoch in (20,50,100):
        for split in ('development','evaluation'):
            path=OUT/f'retrieval_{split}_epoch{epoch}.json'
            if not path.is_file(): return False
            result=json.loads(path.read_text())
            expected=manifest[split]
            if (result.get('status')!='complete' or result.get('checkpoint_epoch')!=epoch
                or result.get('split_manifest_sha256')!=manifest_hash
                or result.get('images')!=expected['image_count']
                or result.get('captions')!=expected['caption_count']): return False
            prefix=expected_checkpoint[epoch]
            if not result.get('checkpoint_sha256','').startswith(prefix): return False
    return True

def skip(stage,detail):
    atomic(STATUS,{'status':'running','stage':stage,'detail':detail,'pid':os.getpid(),'updated_unix':time.time(),'skipped_verified':True})
    print(f'\n[{time.strftime("%F %T")}] SKIP {stage}: verified existing artifacts ({detail})',flush=True)

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
        extraction_status=json.loads((FEATURES/'full_val2014_extraction_status.json').read_text())
        text_manifest=json.loads((FEATURES/'text_val2014_manifest.json').read_text())
        if extraction_status.get('status')=='complete' and spatial_cache_matches_manifest():
            skip('visual_extraction','40,504 region rows and all 2,500 spatial rows verified against frozen manifest order')
        else:
            run('visual_extraction',[PYTHON,'-u',str(SRC/'extract_val2014_full.py'),'--dataset-root',str(DATA),'--feature-root',str(FEATURES),'--segmentation-manifest',str(OUT/'segmentation_manifest.json'),'--batch-size','1'],'all val2014 images, plus dense spatial cache for 2500 segmentation images')
        if text_manifest.get('completed')==202654 and text_manifest.get('feature_shape')==[202654,1152]:
            skip('text_extraction','all 202,654 val2014 caption features verified')
        else:
            run('text_extraction',[PYTHON,'-u',str(SRC/'extract_text.py'),'--dataset-root',str(DATA),'--feature-root',str(FEATURES),'--split','val','--batch-size','128'],'all 202654 val2014 captions')
        run('feature_integrity', [PYTHON,'-c','from experiments.kv_cache.experiments.trackb_talk2dino_siglip.run_expanded_val2014_validation import validate_features; validate_features()'],'shapes, finite features, completion flags and ID joins')
        if retrieval_artifacts_complete():
            skip('retrieval','all six dev/final retrieval reports match the pinned checkpoints and split manifest')
        else:
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
