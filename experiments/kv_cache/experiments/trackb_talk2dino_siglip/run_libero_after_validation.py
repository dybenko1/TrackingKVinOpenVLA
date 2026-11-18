#!/usr/bin/env python3
"""Wait for COCO evaluation, then run the offline LIBERO mask feasibility pass."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path('/opt/openvla/cache/worktrees/trackb-talk2dino-siglip')
SRC=ROOT/'experiments/kv_cache/experiments/trackb_talk2dino_siglip'
OUT=ROOT/'experiments/kv_cache/outputs/trackb_talk2dino_siglip/full_coco_training_20261117/postprocessing_optimization/full_val2014_holdout'
DATA=Path('/opt/openvla/cache/datasets/talk2dino/coco2014')
LOG=OUT/'logs/supervisor.log'; STATUS=OUT/'run_status.json'
PYTHON=sys.executable

def atomic(obj):
 p=STATUS;t=p.with_suffix('.json.tmp');t.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def main():
 os.chdir(ROOT);OUT.joinpath('logs').mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env['PYTHONPATH']='/tmp/talk2dino-benchmark-deps:'+str(ROOT)
 while True:
  if STATUS.exists():
   state=json.loads(STATUS.read_text())
   if state.get('status')=='failed': raise RuntimeError('COCO validation supervisor failed; see run_status.json')
   if state.get('status')=='complete': break
  time.sleep(30)
 lock=open(OUT/'logs/supervisor.lock','w')
 while True:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(2)
 log=open(LOG,'a',buffering=1);os.dup2(log.fileno(),1);os.dup2(1,2)
 atomic({'status':'running','stage':'libero_feasibility','detail':'offline saved-frame mask review','pid':os.getpid(),'updated_unix':time.time()})
 cmd=[PYTHON,'-u',str(SRC/'libero_mask_feasibility.py'),'--dataset-root',str(DATA),
  '--optimization-dir',str(OUT/'segmentation'),'--output-dir',str(OUT/'libero_feasibility'),'--device','cuda:0']
 print(f'[{time.strftime("%F %T")}] START LIBERO feasibility: {" ".join(cmd)}',flush=True)
 try:
  subprocess.run(cmd,cwd=ROOT,env=env,check=True)
  subprocess.run([PYTHON,'-u',str(SRC/'finalize_expanded_validation.py'),'--output-dir',str(OUT)],cwd=ROOT,env=env,check=True)
  atomic({'status':'complete','stage':'complete','exit_code':0,'finished_unix':time.time(),'pid':os.getpid(),'libero_feasibility':'complete'})
  print(f'[{time.strftime("%F %T")}] LIBERO mask feasibility complete',flush=True)
 except BaseException as e:
  atomic({'status':'failed','stage':'libero_feasibility','error':repr(e),'updated_unix':time.time(),'pid':os.getpid()})
  raise
if __name__=='__main__':main()
