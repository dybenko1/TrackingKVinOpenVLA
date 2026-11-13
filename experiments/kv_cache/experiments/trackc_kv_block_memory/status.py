#!/usr/bin/env python3
import json,sys
from pathlib import Path
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[4]/"experiments"/"kv_cache"/"outputs"/"trackc_kv_block_memory"/"logs"/"progress.json"
if not p.exists(): print(f"No progress file: {p}"); raise SystemExit(1)
d=json.loads(p.read_text())
print(f"Experiment: {d.get('experiment','Track C Experiment 1 — KV Block Memory Management')}\nState: {d.get('state',d.get('status','unknown'))}\nPhase: {d.get('phase','n/a')}\nPID: {d.get('pid','n/a')}")
if d.get("current_page_tokens") is not None: print(f"Current: page={d['current_page_tokens']} condition={d.get('current_condition')} iteration={d.get('iteration')}")
if d.get("completed_configurations") is not None: print(f"Completed GPU configurations: {d['completed_configurations']}/{d.get('total_configurations')}")
if d.get("error"): print(f"Error: {d['error']}")
if d.get("exit_code") is not None: print(f"Exit code: {d['exit_code']}")
