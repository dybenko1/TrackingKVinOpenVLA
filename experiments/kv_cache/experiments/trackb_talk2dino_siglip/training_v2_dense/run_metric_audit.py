"""Persistent corrected V1/V2 evaluation; no model training is performed."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback
from pathlib import Path

PKG="experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense"

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n");tmp.replace(path)

def run(cmd,repo,log,env):
    with log.open("a",buffering=1) as f:
        f.write("\nCOMMAND: "+" ".join(map(str,cmd))+"\n")
        f.flush()
        return subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=False).returncode

def main():
    p=argparse.ArgumentParser()
    for name in ("repo","dataset-root","cache-root","text-cache-root","audit-dir","train-output","checkpoint-dir","v1-checkpoint","split-manifest","v1-report","v1-postprocess-summary"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();out=a.audit_dir; logs=out/"logs";logs.mkdir(parents=True,exist_ok=True)
    status=out/"RUN_STATUS.json";log=logs/"audit_pipeline.log"
    env=os.environ.copy();env["PYTHONPATH"]=str(a.repo)+os.pathsep+env.get("PYTHONPATH","")
    eval_cmd=[sys.executable,"-m",PKG+".evaluate",
        "--dataset-root",str(a.dataset_root),"--cache-root",str(a.cache_root),
        "--text-cache-root",str(a.text_cache_root),"--output-dir",str(out),
        "--split-manifest",str(a.split_manifest),"--train-output",str(a.train_output),
        "--checkpoint-dir",str(a.checkpoint_dir),"--v1-checkpoint",str(a.v1_checkpoint),"--resume"]
    atomic_json(status,{"status":"running","stage":"corrected_evaluation","started_at":time.time(),"pid":os.getpid(),"resume_supported":True,"no_training":True})
    code=run(eval_cmd,a.repo,log,env)
    if code:
        atomic_json(status,{"status":"failed_recoverable","stage":"corrected_evaluation","exit_code":code,"updated_at":time.time(),"resume_command":"python -m "+PKG+".run_metric_audit ... --resume"})
        return code
    final_cmd=[sys.executable,"-m",PKG+".finalize_metric_audit","--audit-dir",str(out),"--train-output",str(a.train_output),"--v1-report",str(a.v1_report),"--v1-postprocess-summary",str(a.v1_postprocess_summary)]
    atomic_json(status,{"status":"running","stage":"validate_and_report","evaluation_exit_code":0,"updated_at":time.time(),"pid":os.getpid()})
    code=run(final_cmd,a.repo,log,env)
    if code:
        atomic_json(status,{"status":"failed_recoverable","stage":"validate_and_report","exit_code":code,"updated_at":time.time()})
        return code
    marker=json.loads((out/"evaluation_complete.json").read_text())
    atomic_json(status,{"status":"complete","stage":"complete","images":marker["images"],"exit_code":0,"completed_at":time.time(),"report":str(out/"AUDIT_REPORT.md"),"no_training":True})
    return 0

if __name__=="__main__":
    try:raise SystemExit(main())
    except Exception as e:
        print(traceback.format_exc(),file=sys.stderr)
        raise
