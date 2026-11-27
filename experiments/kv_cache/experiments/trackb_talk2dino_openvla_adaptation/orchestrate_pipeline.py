#!/usr/bin/env python3
"""Persistent sequential runner for OpenVLA/Talk2DINO adaptation.

Runs inside the validated Docker container. All expensive stages are
restartable from complete demo files or epoch checkpoints. Git publication is
left to the host-side watcher because the container has no host Git auth.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/opt/openvla/cache/worktrees/trackb-dense-grounding-v2')
SRC = ROOT / 'experiments/kv_cache/experiments/trackb_talk2dino_openvla_adaptation'
OUT = ROOT / 'experiments/kv_cache/outputs/trackb_talk2dino_openvla_adaptation'
CACHE = Path('/opt/openvla/cache/datasets/talk2dino_openvla_adaptation_v1')
BASELINE = ROOT / 'experiments/kv_cache/outputs/trackb_talk2dino_openvla_multicategory'
CONTAINER = os.environ.get('TRACKB_ADAPT_CONTAINER', 'trackb-v1-coco-object-20261126')
STATUS = OUT / 'RUN_STATUS.json'
LOGS = OUT / 'logs'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_status(**updates) -> None:
    old = {}
    if STATUS.exists():
        try: old = json.loads(STATUS.read_text())
        except Exception: pass
    old.update(updates)
    old['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    atomic_json(STATUS, old)


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}"
    print(line, flush=True)
    with (LOGS / 'experiment.log').open('a') as f:
        f.write(line + '\n'); f.flush()


def config() -> dict:
    return {
        'experiment': 'Talk2DINO threshold calibration and OpenVLA DINOv2 adaptation',
        'starting_branch': 'thesis/trackb-dense-grounding-v2',
        'starting_commit': 'f815553ab0831c5b57b2c4895ea2b984cae1ee68',
        'baseline_report': str(BASELINE / 'EXPERIMENT_REPORT.md'),
        'openvla_checkpoint': 'openvla/openvla-7b-finetuned-libero-spatial',
        'openvla_revision': '962318cec55ac10993ff0f5f43eda9a270b4c873',
        'dino': {'identifier': 'vit_large_patch14_reg4_dinov2.lvd142m', 'blocks': 24,
                 'frozen_blocks_e2': '1-24', 'trainable_blocks_e3': '24 only',
                 'native_policy_features': 'block 23 output unchanged', 'spatial_grid': [16, 16]},
        'text': {'encoder': 'OpenAI CLIP ViT-B/16', 'sha256': '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f',
                 'templates': 7, 'frozen': True},
        'projection': {'initialization': 'original Talk2DINO ViT-L MLP',
                       'sha256': '4e28d24ecb905592c902b04d9c077484346ebafac8547b968bd5af16291d8d69'},
        'e1': {'thresholds': '0.450-0.600 step 0.001', 'dev_only_selection': True,
               'foreground_recall_constraints': [0.95, 0.98, 0.99]},
        'e2': {'trainable': 'projection MLP only', 'objective': '0.70 asymmetric BCE + 0.30 soft Dice',
               'max_epochs': 20, 'batch_size': 32, 'lr': 5e-5, 'early_stopping': True},
        'e3': {'initialization': 'OpenVLA block 24 + best E2 MLP',
               'trainable': ['DINO block 24', 'projection MLP'], 'max_epochs': 12,
               'batch_size': 8, 'block24_lr': 1e-6, 'projection_lr': 5e-5,
               'early_stopping': True},
        'dataset_split': {'unit': 'complete demonstration', 'task_scope': 'six predeclared LIBERO development tasks',
                          'demonstrations_in_existing_90_frame_benchmark_excluded': True,
                          'remaining_demo_split': {'train': 0.74, 'development': 0.13, 'new_heldout': 0.13},
                          'frames_per_demo': 8, 'recorded_actions_applied': False,
                          'six predeclared held-out tasks': 'excluded from training and selection'},
        'evaluation': 'same frozen 90-frame benchmark; 30 development/60 previously inspected evaluation frames',
        'no_new_policy_rollouts': True, 'no_kv_cache_changes': True,
        'large_features_and_checkpoints_local_ignored_cache': str(CACHE),
    }


def preflight() -> dict:
    import torch
    import h5py
    import importlib.util
    from pathlib import Path as P
    import json as json_module
    task_manifest = json_module.loads((ROOT / 'experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/libero_ground_truth_e2_completion/libero_task_manifest.json').read_text())
    tasks = [x for x in task_manifest['tasks'] if x.get('split') == 'development']
    if len(tasks) != 6: raise RuntimeError(f'expected 6 predeclared adaptation tasks, got {len(tasks)}')
    hdf5 = [P('/opt/openvla/cache/datasets/libero_ground_truth_v2_official') / x['hdf5_repo_path'] for x in tasks]
    absent = [str(x) for x in hdf5 if not x.is_file()]
    if absent: raise FileNotFoundError(f'missing demo files: {absent}')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    free, total = torch.cuda.mem_get_info()
    du = __import__('shutil').disk_usage('/opt/openvla/cache')
    if free < 12 * 1024**3: raise RuntimeError(f'insufficient free GPU memory: {free}')
    if du.free < 40 * 1024**3: raise RuntimeError(f'insufficient free disk: {du.free}')
    baseline_manifest = json_module.loads((BASELINE / 'frame_manifest.json').read_text())
    if not baseline_manifest.get('selection') and not baseline_manifest.get('frame_records'):
        raise RuntimeError('baseline frame manifest has no frame selection')
    if not (BASELINE / 'score_maps').is_dir(): raise FileNotFoundError(BASELINE / 'score_maps')
    return {'status': 'passed', 'cuda': True, 'gpu_free_bytes': int(free), 'gpu_total_bytes': int(total),
            'disk_free_bytes': int(du.free), 'torch': torch.__version__, 'hdf5_task_files': len(hdf5),
            'development_tasks': [x['task_id'] for x in tasks], 'task_manifest_sha256': task_manifest['manifest_sha256'],
            'baseline_manifest_sha256': sha256(BASELINE / 'frame_manifest.json')}


def run_stage(name: str, args: list[str], progress: dict | None = None) -> None:
    write_status(status='running', stage=name, stage_status='running', command=args,
                 latest_checkpoint=progress.get('latest_checkpoint') if progress else None,
                 last_error=None)
    log(f'STAGE START {name}: {" ".join(args)}')
    env = os.environ.copy()
    env['TRACKB_ADAPT_OUT'] = str(OUT)
    env['TRACKB_ADAPT_CACHE'] = str(CACHE)
    env['HF_HUB_OFFLINE'] = '1'; env['TRANSFORMERS_OFFLINE'] = '1'
    env['LIBERO_CONFIG_PATH'] = str(CACHE / 'libero_user_config')
    env['NUMBA_CACHE_DIR'] = str(CACHE / 'numba_cache')
    env['MPLCONFIGDIR'] = str(CACHE / 'matplotlib')
    env['HOME'] = str(CACHE / 'home')
    p = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip('\n')
        log(f'[{name}] {line}')
        tail.append(line)
        if len(tail) > 80: tail.pop(0)
        if name in ('e2_training', 'e3_training') and 'epoch=' in line:
            try:
                fragment = line.split('epoch=', 1)[1].split()[0]
                done, total = fragment.split('/')
                write_status(current_epoch=int(done), total_epochs=int(total), latest_log_line=line)
            except Exception: pass
    code = p.wait()
    if code:
        raise RuntimeError(f'{name} exited {code}; tail={tail[-30:]}')
    log(f'STAGE COMPLETE {name}')
    write_status(status='running', stage=name, stage_status='complete', last_error=None,
                 completed_stages=(json.loads(STATUS.read_text()).get('completed_stages', []) + [name]))


def commit_ready() -> None:
    files = list(OUT.glob('*.json')) + list(OUT.glob('*.csv')) + list(OUT.glob('*.md'))
    figures = list((OUT / 'figures').glob('*.png'))
    expected = ['E1_calibration.json', 'E1_threshold_sweep.csv', 'dataset_manifest.json',
                'feature_cache_validation.json', 'e2_training_summary.json', 'e3_training_summary.json',
                'per_frame_metrics.csv', 'role_metrics.csv', 'summary.json', 'policy_parity_validation.json',
                'latency_metrics.json', 'new_heldout_summary.json', 'new_heldout_metrics.csv', 'new_heldout_role_metrics.csv']
    missing = [name for name in expected if not (OUT / name).is_file()]
    if missing: raise RuntimeError(f'final compact artifact gate missing {missing}')
    report = OUT / 'EXPERIMENT_REPORT.md'
    if not report.is_file(): raise FileNotFoundError(report)
    log(f'Final artifacts ready: {len(files)} compact files, {len(figures)} figures')
    write_status(status='complete_waiting_publication', stage='publication', stage_status='ready',
                 report_path=str(report), compact_file_count=len(files), figure_count=len(figures),
                 checkpoint_paths={'e2': str(CACHE / 'checkpoints/e2/best_projection.pt'),
                                   'e3': str(CACHE / 'checkpoints/e3/best_block24_projection.pt')},
                 latest_checkpoint=None, last_error=None,
                 resume_command='docker exec -u 2011:2011 trackb-v1-coco-object-20261126 python /opt/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/experiments/trackb_talk2dino_openvla_adaptation/orchestrate_pipeline.py')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    # LIBERO normally prompts on first import and derives ~/.libero from the
    # container account's passwd entry (which resolves to / in this image).
    # Give it a private persistent config path and the package's default paths.
    libero_config = CACHE / 'libero_user_config'
    libero_config.mkdir(parents=True, exist_ok=True)
    for directory in ('numba_cache', 'matplotlib', 'home'):
        (CACHE / directory).mkdir(parents=True, exist_ok=True)
    config_file = libero_config / 'config.yaml'
    if not config_file.exists():
        config_file.write_text(
            'benchmark_root: /opt/LIBERO/libero/libero\n'
            'bddl_files: /opt/LIBERO/libero/libero/bddl_files\n'
            'init_states: /opt/LIBERO/libero/libero/init_files\n'
            'datasets: /opt/LIBERO/libero/datasets\n'
            'assets: /opt/LIBERO/libero/libero/assets\n')
    lock_path = CACHE / 'overnight_pipeline.lock'
    with lock_path.open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('another adaptation pipeline holds the lock; refusing duplicate worker')
        cfg = config(); cfg['configuration_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        atomic_json(OUT / 'configuration.json', cfg)
        write_status(status='running', stage='preflight', stage_status='running', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     pid=os.getpid(), container=CONTAINER, completed_stages=[], current_epoch=None, total_epochs=None,
                     latest_checkpoint=None, last_error=None)
        try:
            pf = preflight(); atomic_json(OUT / 'preflight_validation.json', pf)
            log(f'PREFLIGHT PASS: CUDA free {pf["gpu_free_bytes"] / 1024**3:.1f} GiB; disk free {pf["disk_free_bytes"] / 1024**3:.1f} GiB; HDF5 tasks {pf["hdf5_task_files"]}')
            commands = [
                ('e1_calibration', ['python', str(SRC / 'run_e1_calibration.py')]),
                ('simulator_label_preparation', ['python', str(SRC / 'prepare_libero_data.py')]),
                ('native_feature_cache', ['python', str(SRC / 'cache_features.py')]),
                ('e2_training', ['python', str(SRC / 'train_adaptation.py'), '--stage', 'e2']),
                ('e3_training', ['python', str(SRC / 'train_adaptation.py'), '--stage', 'e3']),
                ('paired_final_evaluation', ['python', str(SRC / 'evaluate_final.py')]),
            ]
            for name, args in commands:
                run_stage(name, args)
            (OUT / 'README.md').write_text(
                '# Talk2DINO–OpenVLA adaptation\n\n'
                'The complete experiment is summarized in [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).\n\n'
                '- E1 threshold calibration: [E1_calibration.json](E1_calibration.json), [sweep CSV](E1_threshold_sweep.csv), [curve](figures/E1_threshold_curves.png).\n'
                '- E2/E3 histories: [E2](training_history_e2.json), [E3](training_history_e3.json).\n'
                '- Paired metrics: [summary](summary.json), [per-frame CSV](per_frame_metrics.csv), [role CSV](role_metrics.csv), [new-demo holdout](new_heldout_summary.json).\n'
                '- Runtime: [latency and memory](latency_metrics.json).\n'
                '- Policy-path preservation: [policy_parity_validation.json](policy_parity_validation.json).\n'
                '- Large feature arrays and checkpoints remain in ignored local cache: `' + str(CACHE) + '`.\n')
            commit_ready()
        except Exception as e:
            log(f'PIPELINE FAILED: {type(e).__name__}: {e}')
            write_status(status='failed', stage_status='failed', last_error=f'{type(e).__name__}: {e}',
                         error_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         resume_command=f'python {SRC / "orchestrate_pipeline.py"}')
            raise


if __name__ == '__main__': main()
