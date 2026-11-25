#!/usr/bin/env python3
"""Persistent CUDA smoke -> nine-scenario conservative V2/R5 experiment."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BRANCH = "thesis/trackb-dense-grounding-v2"
REMOTE = "git@github.com:dybenko1/TrackingKVinOpenVLA.git"
IMAGE = "openvla-talk2dino-siglip:20261116"
THRESHOLD = 0.15
V2_SHA = "8232ec8f657da47645d73741b16560a1140bfea89cce6826948b9457ce4854d4"
BASE = REPO / "experiments/kv_cache/outputs/conservative_v2_hybrid_threshold015"
E2 = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/libero_ground_truth_e2_completion"
LIVE = BASE / "live_20261125"
STATUS = BASE / "SUPERVISOR_STATUS.json"
LOG = BASE / "logs/supervisor.log"
CONTAINER = "trackb-conservative-v2-hybrid015-20261125"
SMOKE_CONTAINER = CONTAINER + "-smoke"
CONTAINER_REPO = "/opt/openvla/cache/worktrees/trackb-dense-grounding-v2"
CONTAINER_LIVE = CONTAINER_REPO + "/experiments/kv_cache/outputs/conservative_v2_hybrid_threshold015/live_20261125"


def run(*args, check=True, timeout=None, capture_output=True):
    return subprocess.run(args, check=check, text=True, timeout=timeout,
                          capture_output=capture_output)


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}\n")
        f.flush(); os.fsync(f.fileno())


def set_status(phase, message, **extra):
    BASE.mkdir(parents=True, exist_ok=True)
    data = {}
    if STATUS.exists():
        try: data = json.loads(STATUS.read_text())
        except Exception: pass
    data.update({"status": phase, "phase": phase, "message": message,
                 "updated_unix": time.time(), "branch": BRANCH,
                 "threshold": THRESHOLD, "container": CONTAINER, **extra})
    tmp = STATUS.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True); f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, STATUS)
    log(f"{phase}: {message}")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def docker_base(name, command, detached=False):
    args = ["docker", "run"]
    if detached: args += ["-d", "--name", name, "--restart=no"]
    else: args += ["--rm", "--name", name]
    args += ["--runtime", "nvidia", "--network", "host", "--group-add", "2011",
             "-e", "NVIDIA_DRIVER_CAPABILITIES=all", "-e", "MUJOCO_GL=egl",
             "-e", "PYOPENGL_PLATFORM=egl", "-e", "MPLCONFIGDIR=/tmp/matplotlib",
             "-e", "NUMBA_CACHE_DIR=/tmp/numba", "-e", "HF_HUB_OFFLINE=1",
             "-e", "TRANSFORMERS_OFFLINE=1", "-e", f"PYTHONPATH={CONTAINER_REPO}:/opt/LIBERO",
             "-e", f"LIBERO_CONFIG_PATH={CONTAINER_LIVE}/runtime/libero_config",
             "-v", "/home/oscar/projects/TrackingKVinOpenVLA/openvla:/opt/openvla",
             "-v", "/home/oscar/projects/TrackingKVinOpenVLA/LIBERO:/opt/LIBERO",
             "-v", "/home/oscar/projects/TrackingKVinOpenVLA/dusty-robosuite:/opt/robosuite",
             "-v", "/tmp:/tmp", "-w", CONTAINER_REPO, IMAGE, "bash", "-lc", command]
    return args


def check_no_duplicate(name):
    p = run("docker", "inspect", name, check=False)
    if p.returncode == 0:
        running = run("docker", "inspect", "-f", "{{.State.Running}}", name).stdout.strip()
        if running == "true": raise RuntimeError(f"Refusing duplicate: {name} already running")
        run("docker", "rm", name)


def create_libero_config():
    conf = LIVE / "runtime/libero_config"
    conf.mkdir(parents=True, exist_ok=True)
    (conf / "config.yaml").write_text(
        "benchmark_root: /opt/LIBERO/libero/libero\n"
        "bddl_files: /opt/LIBERO/libero/libero/bddl_files\n"
        "init_states: /opt/LIBERO/libero/libero/init_files\n"
        "datasets: /opt/LIBERO/libero/datasets\n"
        "assets: /opt/LIBERO/libero/libero/assets\n")
    # LIBERO_CONFIG_PATH points into the shared writable output mount.
    return CONTAINER_LIVE + "/runtime/libero_config/config.yaml"


def run_smoke(config_path):
    check_no_duplicate(SMOKE_CONTAINER)
    logpath = LIVE / "logs/cuda_smoke.log"
    logpath.parent.mkdir(parents=True, exist_ok=True)
    cmd = (f"python -u {CONTAINER_REPO}/experiments/kv_cache/experiments/hybrid_v2_task_relevance_runner.py "
           f"--mode smoke --v2-threshold 0.15 --output-dir {CONTAINER_LIVE}/validation_smoke "
           f"> {CONTAINER_LIVE}/logs/cuda_smoke.log 2>&1")
    set_status("smoke_running", "Running one 5-step CUDA R5 smoke test before any evaluation scenario")
    run(*docker_base(SMOKE_CONTAINER, cmd), capture_output=True, timeout=3600)
    smoke_status_path = LIVE / "validation_smoke/RUN_STATUS.json"
    scenario_path = LIVE / "validation_smoke/scenario_results/task4_state0.json"
    if not smoke_status_path.is_file() or not scenario_path.is_file():
        raise RuntimeError("CUDA smoke did not write status and episode evidence")
    smoke = json.loads(smoke_status_path.read_text())
    episode = json.loads(scenario_path.read_text())
    if smoke.get("status") != "complete" or episode.get("steps") != 5 or len(episode.get("records", [])) != 5:
        raise RuntimeError("CUDA smoke did not complete the expected 5-step cycle")
    if not any(r.get("type") == "reuse" and r.get("layers_validated") == 32 for r in episode["records"]):
        raise RuntimeError("CUDA smoke lacks 32-layer selective reuse validation")
    if any(r.get("type") == "reuse" and r.get("source_age") not in (1,2,3,4) for r in episode["records"]):
        raise RuntimeError("CUDA smoke has incorrect R5 source-age evidence")
    (BASE / "validation_smoke.json").write_text(json.dumps({
        "status":"passed", "steps":5, "scenario":"task4/state0 smoke only",
        "threshold":THRESHOLD, "full_refresh_action_and_logit_parity":True,
        "32_layer_kv_skip_and_r5_source_checks":True,
        "source_status":str(smoke_status_path.relative_to(BASE)),
        "episode_evidence":str(scenario_path.relative_to(BASE))}, indent=2)+"\n")
    set_status("smoke_passed", "Five-step CUDA smoke passed; launching the nine predeclared scenarios")


def run_scenarios(config_path):
    check_no_duplicate(CONTAINER)
    cmd = (f"python -u {CONTAINER_REPO}/experiments/kv_cache/experiments/hybrid_v2_task_relevance_runner.py "
           f"--mode run-all --v2-threshold 0.15 --output-dir {CONTAINER_LIVE} "
           f"> {CONTAINER_LIVE}/logs/experiment.log 2>&1")
    cid = run(*docker_base(CONTAINER, cmd, detached=True)).stdout.strip()
    set_status("running", "Detached Docker worker is executing all nine scenarios sequentially", docker_cid=cid)
    while run("docker", "inspect", "-f", "{{.State.Running}}", CONTAINER).stdout.strip() == "true":
        time.sleep(20)
        progress = {}
        try:
            d=json.loads((LIVE/"RUN_STATUS.json").read_text())
            progress={"completed_scenarios":len(d.get("completed",[])),
                      "current_scenario":d.get("current_scenario")}
        except Exception: pass
        set_status("running", "Nine-scenario Docker worker active", **progress)
    code=int(run("docker","inspect","-f","{{.State.ExitCode}}",CONTAINER).stdout.strip())
    if code != 0: raise RuntimeError(f"LIBERO worker exited {code}; inspect {LIVE/'logs/experiment.log'}")
    d=json.loads((LIVE/"RUN_STATUS.json").read_text())
    if d.get("status")!="complete" or len(d.get("completed",[]))!=9:
        raise RuntimeError("Worker exited but nine-scenario RUN_STATUS is incomplete")
    set_status("scenarios_complete", "All nine LIBERO scenarios finished", successes=sum(bool(x["success"]) for x in d["completed"]))


def finalize():
    summary=json.loads((LIVE/"summary.json").read_text())
    config=json.loads((LIVE/"configuration.json").read_text())
    offline=json.loads((BASE/"offline_preflight/offline_threshold_metrics.json").read_text())
    with (LIVE/"per_scenario_metrics.csv").open(newline="") as f:
        import csv
        scenario_metrics=list(csv.DictReader(f))
    refresh_count=sum(int(x["refresh_steps"]) for x in scenario_metrics)
    reuse_count=sum(int(x["reuse_steps"]) for x in scenario_metrics)
    total_decisions=summary["total_policy_decisions"]
    reuse_cov_selective=(summary["mean_reuse_candidates_per_reuse_decision"] or 0)/256
    timing=summary["latency"]; overhead=summary["overhead"]
    report=(LIVE/"HYBRID_V2_TASK_RELEVANCE_REPORT.md").read_text()
    config.update({"experiment":"Conservative V2 Hybrid KV Reuse",
                   "probability_threshold":THRESHOLD,
                   "internal_sigmoid_cosine_offset":0.20,
                   "internal_sigmoid_scale":10.0,
                   "starting_reference_commit":run("git","-C",str(REPO),"rev-parse","HEAD").stdout.strip(),
                   "E2_manifest_sha256":offline["e2_manifest_sha256"],
                   "previously_inspected_heldout":True,
                   "primary_threshold_predeclared":True})
    (BASE/"configuration.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
    rows=summary["scenario_summary"]
    (BASE/"summary.json").write_text(json.dumps({
        "status":"complete", "threshold":THRESHOLD,
        "successes":summary["successes"], "scenario_count":summary["scenario_count"],
        "scenario_summary":rows,
        "actual_reuse_coverage_including_full_refreshes":summary["actual_reuse_coverage_including_full_refreshes"],
        "mean_actual_reused_tokens_per_decision_including_full_refreshes":summary["mean_actual_reused_tokens_per_decision_including_full_refreshes"],
        "total_k_projection_rows_skipped":summary["total_k_projection_rows_skipped"],
        "total_v_projection_rows_skipped":summary["total_v_projection_rows_skipped"],
        "offline_threshold_manifest_sha256":offline["e2_manifest_sha256"],
        "offline_threshold_metrics_path":"offline_preflight/offline_threshold_metrics.json",
        "live_summary_path":"live_20261125/summary.json",
        "configuration_path":"configuration.json"},indent=2)+"\n")
    lines=["# Conservative V2 Hybrid KV Reuse — Probability Cutoff 0.15", "",
           f"Status: **complete**. Success: **{summary['successes']}/9**.", "",
           "## Frozen method", "",
           "This is the fixed Stage 3C-derived R5 hybrid (not the adaptive rolling policy). It changes exactly the external V2 foreground-probability classification cutoff from 0.20 to 0.15. The trained scoring function remains `sigmoid(10 × (cosine_similarity − 0.20))`; its cosine offset is unchanged. Frozen V2 epoch 40, current-frame native penultimate OpenVLA DINO features, existing E2 instruction prompts and SigLIP text embeddings, top-10% attention protection of current predicted background, layer-15 attention captured at full-refresh anchors, immutable independent R5 anchor, full Q and actual K/V skipping over 32 layers are retained. No confidence gate, rolling cache, continuous attention, or explicit FG-to-BG transition protection is used.", "",
           "## Offline E2 threshold diagnostic", "",
           "The pre-existing E2 cached score/mask/occupancy arrays were reused; no weights or features were regenerated. E2 held-out scores were previously inspected during the role-mapping audit, so these are regression diagnostics rather than a blind test. The predetermined live configuration remains 0.15; thresholds were not selected from these outputs.", "",
           "| Split | Cutoff | Task-relevant recall | Precision | False-background rate | Candidate BG coverage | Mean reusable tokens |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("development","held_out"):
        for t in ("0.10","0.15","0.20"):
            m=offline["thresholds"][t][split]; fg=m["role_metrics"]["task_relevant_foreground"]
            lines.append(f"| {split} | {t} | {fg['recall']:.4f} | {fg['precision']:.4f} | {fg['false_background_rate']:.4f} | {m['foreground']['mean_candidate_background_coverage']:.4f} | {m['foreground']['mean_reusable_tokens']:.1f}/256 |")
    lines += ["", "Offline reports: `offline_preflight/OFFLINE_THRESHOLD_REPORT.md`, `offline_preflight/offline_threshold_metrics.json`, `offline_preflight/offline_threshold_per_task.csv`; representative saved-score heatmaps are in `offline_preflight/visualizations/`.", "",
              "## Live LIBERO regression matrix", "",
              "| Task | State | Success | Steps |", "|---:|---:|:---:|---:|"]
    for x in rows: lines.append(f"| {x['task_id']} | {x['state_id']} | {'yes' if x['success'] else 'no'} | {x['steps']} |")
    lines += ["", "## Efficiency and timing", "",
              f"Realized reuse including full-refresh decisions: {summary['actual_reuse_coverage_including_full_refreshes']:.1%} ({summary['mean_actual_reused_tokens_per_decision_including_full_refreshes']:.1f}/256 tokens per policy decision). Selective-step reuse: {reuse_cov_selective:.1%}. Full-refresh frequency: {refresh_count/max(total_decisions,1):.1%} ({refresh_count}/{total_decisions}); selective decisions: {reuse_count}.", "",
              f"Skipped projection rows across all 32 layers: K {summary['total_k_projection_rows_skipped']:,}, V {summary['total_v_projection_rows_skipped']:,}. Mean skipped rows per decision: K {summary['mean_k_projection_rows_skipped_per_policy_decision']:.1f}, V {summary['mean_v_projection_rows_skipped_per_policy_decision']:.1f}. These counts are structural and are not a speedup claim.", "",
              f"Selection time: {overhead['complete_hybrid_selection_ms']['mean']:.3f} ms mean. Attention capture on R5 refreshes: {overhead['refresh_only_attention_capture_ms']['mean']:.3f} ms mean; amortized {overhead['attention_capture_amortized_ms_per_policy_decision']:.3f} ms per decision. Synchronized end-to-end latency: refresh {timing['end_to_end_refresh_decision_ms']['mean']:.2f} ms mean; reuse {timing['end_to_end_reuse_decision_ms']['mean']:.2f} ms mean. Historical baselines were not timed under fully matched instrumentation, so no latency improvement is claimed.", "",
              "## Historical regression references", "",
              "| Policy | Success on established 9-scenario matrix |", "|---|---:|",
              "| Fresh OpenVLA | 8/9 |", "| Stage 3C oracle FG + attention | 9/9 |", "| Stage 4A attention-only | 8/9 |", "| V2 hybrid R5, cutoff 0.20 | 7/9 |", "| Adaptive hybrid | 5/9 |", f"| Conservative V2 hybrid R5, cutoff 0.15 | {summary['successes']}/9 |", "",
              "The established nine scenarios are a regression benchmark, not an independent blind holdout. Different policies produce different observation trajectories. See `per_scenario_metrics.csv` and `scenario_results/` for per-scenario and per-step outcomes.", "",
              "## Mask and validation artifacts", "",
              "- `validation_smoke.json` — five-step CUDA smoke, full-refresh action/logit parity, R5 cache checks and 32-layer projection skip.",
              "- `mask_visualizations/` — live semantic, attention-protection, and reuse-candidate masks.",
              "- `offline_preflight/visualizations/` — 12 E2 probability-grid visual comparisons.",
              "- `configuration.json`, `summary.json`, `per_scenario_metrics.csv`, `RUN_STATUS.json` — frozen setup and metrics.", "",
              "## Interpretation and next protocol", "",
              "The offline caches show a recall/coverage tradeoff, but only the live nine-scenario outcome tests whether it affects policy success. Any association between more semantic protection and task success is not causal in this single threshold comparison. No broader threshold calibration was performed. For a later study, freeze a development scenario set disjoint by task from the held-out set before tuning 0.10/0.15/0.20, choose by task success then completion efficiency, measured reuse, and timing, and evaluate the frozen choice once on held-out tasks. Do not use these nine regression scenarios as independent validation.", ""]
    (BASE/"CONSERVATIVE_V2_HYBRID_REPORT.md").write_text("\n".join(lines))


def publish():
    required=[BASE/"CONSERVATIVE_V2_HYBRID_REPORT.md",BASE/"configuration.json",BASE/"summary.json",
              BASE/"validation_smoke.json",BASE/"offline_preflight/offline_threshold_metrics.json",
              LIVE/"summary.json",LIVE/"per_scenario_metrics.csv",LIVE/"RUN_STATUS.json"]
    if any(not p.is_file() for p in required): raise RuntimeError("Required publication artifact missing")
    data=json.loads((LIVE/"RUN_STATUS.json").read_text())
    if data.get("status")!="complete" or len(data.get("completed",[]))!=9: raise RuntimeError("Cannot publish incomplete rollout set")
    rel=[Path("experiments/kv_cache/experiments/hybrid_v2_task_relevance_runner.py"),
         Path("experiments/kv_cache/experiments/conservative_v2_hybrid_threshold_audit.py"),
         Path("experiments/kv_cache/experiments/supervise_conservative_v2_hybrid.py"),
         Path("experiments/kv_cache/experiments/test_hybrid_v2_task_relevance.py")]
    targets=[BASE/"CONSERVATIVE_V2_HYBRID_REPORT.md",BASE/"configuration.json",BASE/"summary.json",BASE/"validation_smoke.json",
             BASE/"offline_preflight/OFFLINE_THRESHOLD_REPORT.md",BASE/"offline_preflight/offline_threshold_metrics.json",
             BASE/"offline_preflight/offline_threshold_per_task.csv",BASE/"offline_preflight/visualizations",
             LIVE/"RUN_STATUS.json",LIVE/"summary.json",LIVE/"configuration.json",LIVE/"per_scenario_metrics.csv",
             LIVE/"comparison_references.json",LIVE/"HYBRID_V2_TASK_RELEVANCE_REPORT.md",LIVE/"mask_visualizations"]
    files=rel+[Path(p.relative_to(REPO)) for p in targets]
    run("git","-C",str(REPO),"add","-f",*[str(p) for p in files])
    run("git","-C",str(REPO),"diff","--cached","--check")
    if run("git","-C",str(REPO),"diff","--cached","--quiet",check=False).returncode == 1:
        run("git","-C",str(REPO),"commit","-m","Publish conservative V2 hybrid KV reuse at 0.15")
    sha=run("git","-C",str(REPO),"rev-parse","HEAD").stdout.strip()
    run("git","-C",str(REPO),"push",REMOTE,BRANCH,timeout=120)
    remote=run("git","ls-remote",REMOTE,f"refs/heads/{BRANCH}",timeout=60).stdout.split()[0]
    if remote!=sha: raise RuntimeError(f"GitHub SHA mismatch: expected {sha}, observed {remote}")
    return sha


def main():
    BASE.mkdir(parents=True,exist_ok=True); LIVE.mkdir(parents=True,exist_ok=True)
    (LIVE/"logs").mkdir(parents=True,exist_ok=True)
    lock=(BASE/"supervisor.lock").open("a+")
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: set_status("already_running","Another supervisor holds the lock"); return
    if run("git","-C",str(REPO),"branch","--show-current").stdout.strip()!=BRANCH:
        raise RuntimeError(f"Wrong branch: expected {BRANCH}")
    ckpt=REPO/"experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/best_projection.pt"
    if sha256(ckpt)!=V2_SHA: raise RuntimeError("V2 epoch-40 checkpoint SHA mismatch")
    if not (BASE/"offline_preflight/offline_threshold_metrics.json").is_file():
        raise RuntimeError("Offline E2 threshold audit must finish before CUDA work")
    set_status("preflight","Checking image, CUDA, checkpoint, branch and isolated output directories",
               image=IMAGE,v2_checkpoint_sha256=V2_SHA,e2_cache="reused")
    run("docker","run","--rm","--runtime","nvidia",IMAGE,"python","-c",
        "import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0),torch.cuda.mem_get_info())",timeout=120)
    config_path=create_libero_config()
    run_smoke(config_path)
    run_scenarios(config_path)
    finalize()
    set_status("publishing","All outputs validated; publishing compact metrics, report, code and figures")
    sha=publish()
    set_status("complete","Published and remote branch SHA verified",result_commit=sha,
               report_url=f"https://github.com/dybenko1/TrackingKVinOpenVLA/blob/{BRANCH}/experiments/kv_cache/outputs/conservative_v2_hybrid_threshold015/CONSERVATIVE_V2_HYBRID_REPORT.md",
               remote_sha_verified=True)


if __name__=="__main__":
    try: main()
    except Exception as e:
        set_status("failed",f"{type(e).__name__}: {e}",error=repr(e))
        log(f"ERROR {type(e).__name__}: {e}")
        raise
