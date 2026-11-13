#!/usr/bin/env python3
"""Persistent, resumable Track C Experiment 1 supervisor.

Run inside openvla-cached:latest. The supervisor and all phase logs are written
to the bind-mounted output directory, so SSH is not part of the job lifetime.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/"experiments"/"kv_cache"/"outputs"/"trackc_kv_block_memory"

def atomic_json(path:Path, obj:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(obj,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def hardware():
    import torch
    p=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {"hostname":os.uname().nodename,"python":sys.version,"torch":torch.__version__,
            "cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),
            "device":None if p is None else {"name":p.name,"total_memory_bytes":int(p.total_memory),
                                               "compute_capability":[p.major,p.minor]},
            "started_at_utc":datetime.now(timezone.utc).isoformat()}

def run_phase(name:str, command:list[str], status:dict, log, progress:Path):
    status.update({"phase":name,"phase_started_at_utc":datetime.now(timezone.utc).isoformat(),"state":"running"})
    atomic_json(progress,status)
    log.write(f"\n===== {name}: {' '.join(command)} =====\n"); log.flush()
    p=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                       text=True,bufsize=1,env={**os.environ,"PYTHONUNBUFFERED":"1"})
    atomic_json(OUT/"logs"/"current_child.pid.json",{"pid":p.pid,"phase":name,"command":command})
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
    code=p.wait()
    status.setdefault("phases",[]).append({"name":name,"exit_code":code,
        "finished_at_utc":datetime.now(timezone.utc).isoformat(),"command":command})
    atomic_json(progress,status)
    if code: raise subprocess.CalledProcessError(code,command)

def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=OUT)
    ap.add_argument("--warmups",type=int,default=20)
    ap.add_argument("--repetitions",type=int,default=50)
    args=ap.parse_args(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    OUT=out
    (OUT/"logs").mkdir(exist_ok=True)
    lock=(OUT/"logs"/"supervisor.lock").open("w")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print("[LOCKED] another Track C supervisor owns this output directory",flush=True); return 73
    progress=OUT/"logs"/"progress.json"
    status={"experiment":"Track C Experiment 1 — KV Block Memory Management","state":"running",
            "started_at_utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),
            "output_directory":str(OUT),"hardware":hardware(),"phases":[],
            "warmups":args.warmups,"measured_repetitions":args.repetitions,
            "ssh_disconnect_persistent":True,"host_reboot_persistence_configured":False}
    atomic_json(OUT/"configuration.json",{"experiment":status["experiment"],"source_trace":"Stage 3C protect_10, 9 saved task/state results; no rollouts",
      "page_sizes_tokens":[1,4,8,16,32],"layers":32,"kv_heads":32,"head_dim":128,"dtype":"bfloat16",
      "warmups":args.warmups,"measured_repetitions":args.repetitions,
      "benchmark_conditions":["dense clone + index_copy","COW page update without dense materialization","COW + required dense materialization"],
      "hardware":status["hardware"],"no_policy_or_environment_changes":True})
    with (OUT/"logs"/"experiment.log").open("a",buffering=1) as log:
      try:
        print("TRACK C — Experiment 1: starting persistent supervisor",flush=True)
        run_phase("trace_analysis",[sys.executable,str(HERE/"analyze_traces.py"),"--out",str(OUT)],status,log,progress)
        run_phase("correctness_validation",[sys.executable,str(HERE/"gpu_benchmark.py"),"--out",str(OUT),"--warmups",str(args.warmups),"--repetitions",str(args.repetitions),"--validate-only"],status,log,progress)
        run_phase("gpu_benchmark",[sys.executable,str(HERE/"gpu_benchmark.py"),"--out",str(OUT),"--warmups",str(args.warmups),"--repetitions",str(args.repetitions)],status,log,progress)
        run_phase("aggregation",[sys.executable,str(HERE/"summarize_results.py"),"--out",str(OUT)],status,log,progress)
        status.update({"state":"complete","exit_code":0,"finished_at_utc":datetime.now(timezone.utc).isoformat()})
        atomic_json(progress,status); print("[QUEUE_COMPLETE] all phases validated and reports saved",flush=True)
      except BaseException as e:
        status.update({"state":"failed","error":repr(e),"exit_code":getattr(e,"returncode",1),
                       "failed_at_utc":datetime.now(timezone.utc).isoformat()})
        atomic_json(progress,status); print(f"[QUEUE_FAILED] {e!r}",flush=True)
        return int(status["exit_code"] or 1)
    return 0

if __name__=="__main__": raise SystemExit(main())
