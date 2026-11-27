#!/usr/bin/env python3
"""Host-side watcher: commit and push only after the detached run completes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'experiments/kv_cache/outputs/trackb_talk2dino_openvla_adaptation'
SRC = ROOT / 'experiments/kv_cache/experiments/trackb_talk2dino_openvla_adaptation'
BRANCH = 'thesis/trackb-dense-grounding-v2'
REMOTE = 'git@github.com:dybenko1/TrackingKVinOpenVLA.git'
STATUS = OUT / 'RUN_STATUS.json'


def run(args: list[str], *, check=True, capture=False):
    return subprocess.run(args, cwd=ROOT, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def status_update(**changes):
    data = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    data.update(changes)
    data['publication_updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    tmp = STATUS.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, STATUS)


def main():
    deadline = time.time() + 60 * 60 * 24 * 7
    while time.time() < deadline:
        if not STATUS.exists():
            time.sleep(10); continue
        try: data = json.loads(STATUS.read_text())
        except Exception:
            time.sleep(10); continue
        state = data.get('status')
        if state == 'failed':
            print('Worker failed; preserving outputs and not publishing incomplete results.', flush=True)
            return 2
        if state == 'complete_waiting_publication': break
        time.sleep(15)
    else:
        status_update(publication_status='watcher_timeout', last_error='No completed worker status within seven days')
        return 3

    branch = run(['git', 'branch', '--show-current'], capture=True).stdout.strip()
    if branch != BRANCH: raise RuntimeError(f'expected branch {BRANCH}, got {branch}')
    staged = run(['git', 'diff', '--cached', '--quiet'], check=False)
    if staged.returncode != 0: raise RuntimeError('index has staged user changes; refusing to mix commits')
    files = sorted([*SRC.glob('*.py'), *OUT.glob('*.json'), *OUT.glob('*.csv'), *OUT.glob('*.md'), *(OUT / 'figures').glob('*.png')])
    if not files: raise RuntimeError('no compact artifacts available to publish')
    total = sum(p.stat().st_size for p in files)
    if total > 50 * 1024 * 1024: raise RuntimeError(f'refusing to stage {total} bytes of artifacts')
    run(['git', 'add', '-f', '--', *map(str, files)])
    run(['git', 'diff', '--cached', '--check'])
    # Confirm the remote is an ancestor of this worktree before pushing; never
    # rewrite history or auto-rebase a user's concurrent work.
    env = os.environ.copy(); env['GIT_TERMINAL_PROMPT'] = '0'; env['GIT_SSH_COMMAND'] = 'ssh -o BatchMode=yes -o ConnectTimeout=20'
    remote = subprocess.run(['git', 'ls-remote', REMOTE, f'refs/heads/{BRANCH}'], cwd=ROOT,
                            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if remote.returncode: raise RuntimeError(f'noninteractive remote check failed: {remote.stdout}')
    remote_sha = remote.stdout.split()[0] if remote.stdout.split() else None
    if remote_sha:
        ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', remote_sha, 'HEAD'], cwd=ROOT)
        if ancestor.returncode: raise RuntimeError(f'remote {remote_sha} is not an ancestor of local HEAD')
    status_update(publication_status='committing', staged_file_count=len(files), staged_bytes=total,
                  remote_before_push=remote_sha)
    run(['git', 'commit', '-m', 'Complete Talk2DINO OpenVLA adaptation experiments'])
    local_sha = run(['git', 'rev-parse', 'HEAD'], capture=True).stdout.strip()
    pushed = subprocess.run(['git', 'push', REMOTE, f'{BRANCH}:{BRANCH}'], cwd=ROOT, env=env,
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if pushed.returncode: raise RuntimeError(f'git push failed: {pushed.stdout}')
    verify = subprocess.run(['git', 'ls-remote', REMOTE, f'refs/heads/{BRANCH}'], cwd=ROOT,
                            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if verify.returncode: raise RuntimeError(f'git ls-remote verification failed: {verify.stdout}')
    verified_sha = verify.stdout.split()[0] if verify.stdout.split() else None
    if verified_sha != local_sha: raise RuntimeError(f'remote SHA {verified_sha} != local commit {local_sha}')
    status_update(status='complete', publication_status='verified', commit_sha=local_sha,
                  remote_branch=BRANCH, remote_url=f'https://github.com/dybenko1/TrackingKVinOpenVLA/commit/{local_sha}',
                  last_error=None)
    print(f'PUBLISHED {local_sha}', flush=True)
    return 0


if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as e:
        try: status_update(publication_status='failed', last_error=f'{type(e).__name__}: {e}')
        except Exception: pass
        print(f'PUBLISH FAILED: {type(e).__name__}: {e}', flush=True)
        raise
