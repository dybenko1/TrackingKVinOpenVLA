#!/usr/bin/env python3
"""Restart only paired evaluation after both adaptation stages are complete."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path('/opt/openvla/cache/worktrees/trackb-dense-grounding-v2')
SRC=ROOT/'experiments/kv_cache/experiments/trackb_talk2dino_openvla_adaptation'
OUT=Path(os.environ.get('TRACKB_ADAPT_OUT',str(ROOT/'experiments/kv_cache/outputs/trackb_talk2dino_openvla_adaptation')))
CACHE=Path(os.environ.get('TRACKB_ADAPT_CACHE','/opt/openvla/cache/datasets/talk2dino_openvla_adaptation_v1'))
STATUS=OUT/'RUN_STATUS.json'; LOG=OUT/'logs/launcher.log'

def atomic_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,sort_keys=True,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def log(message):
    line=f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}] [evaluation_resume] {message}"
    print(line,flush=True)
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a') as f:f.write(line+'\n');f.flush();os.fsync(f.fileno())

def main():
    OUT.mkdir(parents=True,exist_ok=True);LOG.parent.mkdir(parents=True,exist_ok=True)
    state=json.loads(STATUS.read_text())
    if state.get('status') not in ('failed','running'):
        raise RuntimeError(f"Expected failed/in-progress pipeline for evaluation-only resume, got {state.get('status')}")
    for stage in ('e2','e3'):
        summary=json.loads((OUT/f'{stage}_training_summary.json').read_text())
        ckpt=CACHE/('checkpoints/e2/best_projection.pt' if stage=='e2' else 'checkpoints/e3/best_block24_projection.pt')
        if not ckpt.is_file():raise FileNotFoundError(ckpt)
        actual=hashlib.sha256(ckpt.read_bytes()).hexdigest()
        if summary.get('checkpoint_sha256')!=actual:raise RuntimeError(f'{stage} best checkpoint checksum mismatch')
    state.update({'status':'running','stage':'paired_final_evaluation','stage_status':'running','last_error':None,
        'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'completed_stages':list(dict.fromkeys(state.get('completed_stages',[])+['e2_training','e3_training']))})
    atomic_json(STATUS,state)
    env=os.environ.copy();env.update({'TRACKB_ADAPT_OUT':str(OUT),'TRACKB_ADAPT_CACHE':str(CACHE),
        'LIBERO_CONFIG_PATH':str(CACHE/'libero_user_config'),'NUMBA_CACHE_DIR':str(CACHE/'numba_cache'),
        'MPLCONFIGDIR':str(CACHE/'matplotlib'),'HOME':str(CACHE/'home'),
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
    cmd=[sys.executable,'-u',str(SRC/'evaluate_final.py')]
    log('Starting evaluation-only resume; E1 and E2/E3 training are not rerun.')
    with LOG.open('a') as f:
        p=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            text=line.rstrip('\n'); f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}] [paired_final_evaluation] {text}\n");f.flush()
        code=p.wait()
    if code:raise RuntimeError(f'evaluate_final.py exited {code}')
    required=['E1_calibration.json','E1_threshold_sweep.csv','dataset_manifest.json','feature_cache_validation.json',
        'e2_training_summary.json','e3_training_summary.json','per_frame_metrics.csv','role_metrics.csv','summary.json',
        'policy_parity_validation.json','latency_metrics.json','new_heldout_summary.json','new_heldout_metrics.csv',
        'new_heldout_role_metrics.csv','EXPERIMENT_REPORT.md']
    missing=[x for x in required if not (OUT/x).is_file()]
    if missing:raise RuntimeError(f'Final artifact gate missing {missing}')
    (OUT/'README.md').write_text('# Talk2DINO–OpenVLA adaptation\n\n'
        'See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for the complete E1/E2/E3 analysis.\n\n'
        '- [E1 calibration](E1_calibration.json) · [threshold sweep](E1_threshold_sweep.csv) · [curves](figures/E1_threshold_curves.png)\n'
        '- [E2 history](training_history_e2.csv) · [E3 history](training_history_e3.csv)\n'
        '- [Paired results](summary.json) · [per-frame](per_frame_metrics.csv) · [role metrics](role_metrics.csv)\n'
        '- [New demo-disjoint holdout](new_heldout_summary.json) · [latency](latency_metrics.json)\n'
        '- [Policy parity](policy_parity_validation.json)\n\n'
        'Large feature caches and trained checkpoints remain local under the ignored adaptation cache.\n')
    config_path=OUT/'configuration.json';config=json.loads(config_path.read_text())
    sources=sorted(SRC.glob('*.py'))
    file_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    config['implementation_files_sha256']=file_hashes
    config['implementation_source_sha256']=hashlib.sha256(''.join(f'{k}:{v}\n' for k,v in file_hashes.items()).encode()).hexdigest()
    config.pop('configuration_sha256',None)
    config['configuration_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    atomic_json(config_path,config)
    report=OUT/'EXPERIMENT_REPORT.md';text=report.read_text()
    note='## Execution and reproducibility notes\n\n'
    note+=('- The completed run required two implementation repairs before training: the training-config output path used an invalid `Path + str` operation; '
        'the batched attention capture retained only the first sample. Attention was reconstructed for every cached observation from its stored block-23 tokens through the frozen block 24. '
        'The adaptation-specific code now validates `[N,16,256]` attention alignment.\n')
    note+=('- E2 early-stopped after 13 epochs (best development checkpoint epoch 9); E3 early-stopped after 4 epochs (best development checkpoint epoch 1). '
        'The selected checkpoint epochs are development-selected, not final-evaluation-selected.\n')
    note+=('- The existing 60-frame evaluation split was examined in earlier experiments. It is a paired historical comparison, not an untouched test set. '
        'The additional holdout is demonstration-disjoint but not task-disjoint.\n')
    note+=('- Runtime retained Transformers 4.49.0 / tokenizers 0.21.0; OpenVLA remote code warns that it was authored for 4.40.1 / 0.19.1. '
        'No dependency or validated container changes were made.\n')
    if '## Execution and reproducibility notes' not in text:report.write_text(text.rstrip()+'\n\n'+note)
    state=json.loads(STATUS.read_text())
    state.update({'status':'complete_waiting_publication','stage':'publication','stage_status':'ready',
        'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'last_error':None,
        'completed_stages':list(dict.fromkeys(state.get('completed_stages',[])+['paired_final_evaluation'])),
        'report_path':str(report),'resume_command':'none; evaluation and artifact gates completed; publisher watcher handles commit/push'})
    atomic_json(STATUS,state)
    log(f'Final evaluation and artifact gates passed; waiting for publisher. Report={report}')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        try:
            state=json.loads(STATUS.read_text()) if STATUS.exists() else {}
            state.update({'status':'failed','stage':'paired_final_evaluation','stage_status':'failed',
                'last_error':f'{type(exc).__name__}: {exc}','error_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});atomic_json(STATUS,state)
        except Exception:pass
        log(f'FAILED {type(exc).__name__}: {exc}');raise
