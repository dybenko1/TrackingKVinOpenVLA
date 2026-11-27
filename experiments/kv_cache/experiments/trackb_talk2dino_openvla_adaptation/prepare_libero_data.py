#!/usr/bin/env python3
"""Build a demonstration-disjoint simulator-labeled LIBERO adaptation set.

This replays recorded MuJoCo states only. Recorded actions are never applied.
It reserves all benchmark-selected demonstrations, uses only the six predeclared
development tasks for adaptation, and splits remaining complete demonstrations
into train/dev/new held-out demo partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np

WORKTREE = Path("/opt/openvla/cache/worktrees/trackb-dense-grounding-v2")
REPO = Path("/opt/openvla")
OUT = Path(os.environ.get("TRACKB_ADAPT_OUT", str(WORKTREE / "experiments/kv_cache/outputs/trackb_talk2dino_openvla_adaptation")))
CACHE = Path(os.environ.get("TRACKB_ADAPT_CACHE", "/opt/openvla/cache/datasets/talk2dino_openvla_adaptation_v1"))
E2_SCRIPT = WORKTREE / "experiments/kv_cache/experiments/trackb_talk2dino_siglip/transfer_validation/libero_demo_ground_truth.py"
TASK_MANIFEST = WORKTREE / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/libero_ground_truth_e2_completion/libero_task_manifest.json"
BENCH_MANIFEST = WORKTREE / "experiments/kv_cache/outputs/trackb_talk2dino_openvla_multicategory/frame_manifest.json"
DEMO_DATA = Path("/opt/openvla/cache/datasets/libero_ground_truth_v2_official")
LIBERO = Path("/opt/LIBERO")
ROBOSUITE = Path("/opt/robosuite")


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_e2():
    sys.path.insert(0, str(WORKTREE)); sys.path.insert(0, str(REPO))
    import experiments.kv_cache
    experiments.kv_cache.__path__.append(str(WORKTREE / "experiments/kv_cache"))
    import experiments.kv_cache.experiments
    experiments.kv_cache.experiments.__path__.append(str(WORKTREE / "experiments/kv_cache/experiments"))
    spec = importlib.util.spec_from_file_location("adaptation_libero_e2", E2_SCRIPT)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def role_keys_for_prompt(role: str) -> list[str]:
    return {
        "manipulated_object": ["manipulated_objects"],
        "destination_or_fixture": ["destination_objects", "task_relevant_fixtures"],
        "task_fixture": ["task_relevant_fixtures"],
        "mentioned_context_object": ["task_relevant_objects"],
        "instruction_fallback": ["task_relevant_objects"],
        "robot_arm": ["robot_arm"],
        "robot_gripper": ["robot_gripper"],
    }.get(role, ["task_relevant_objects"])


def task_query_specs(e2, task):
    specs = e2.instruction_phrases(task["instruction"] or "")
    for q in specs:
        q["gt_keys"] = role_keys_for_prompt(q["role"])
    specs.extend([
        {"phrase": "robot arm", "role": "robot_arm", "source": "fixed_robot_role_prompt", "rule": "fixed", "gt_keys": ["robot_arm"]},
        {"phrase": "robot gripper", "role": "robot_gripper", "source": "fixed_robot_role_prompt", "rule": "fixed", "gt_keys": ["robot_gripper"]},
    ])
    dedup, seen = [], set()
    for q in specs:
        k = (q["phrase"], tuple(q["gt_keys"]))
        if k not in seen:
            seen.add(k); dedup.append(q)
    return dedup


def select_demo_splits(tasks, bench_manifest, seed=123):
    # Exclude each demonstration appearing in the existing 90-frame benchmark,
    # including development and evaluation examples, from all adaptation splits.
    excluded = {}
    records = bench_manifest.get("selection", bench_manifest.get("frame_records", []))
    if not records:
        raise RuntimeError("Benchmark manifest has no selection/frame_records list")
    for item in records:
        task_id = item.get("task_id") or f"{item['suite']}/{item.get('task', item.get('task_name'))}"
        excluded.setdefault(task_id, set()).add(item["demo"])
    rng = np.random.default_rng(seed)
    result = []
    for task in tasks:
        available = [d for d in task["episode_names"] if d not in excluded.get(task["task_id"], set())]
        # Stable task-local permutation; exactly complete demos remain isolated.
        permutation = rng.permutation(len(available))
        shuffled = [available[int(i)] for i in permutation]
        if len(shuffled) < 18:
            raise RuntimeError(f"Too few unreserved demos for three-way split: {task['task_id']} {len(shuffled)}")
        n_train = max(1, int(len(shuffled) * 0.74))
        n_dev = max(1, int(len(shuffled) * 0.13))
        train = shuffled[:n_train]
        dev = shuffled[n_train:n_train + n_dev]
        test = shuffled[n_train + n_dev:]
        if not test:
            raise RuntimeError(f"No new held-out demos for {task['task_id']}")
        result.append({"task": task, "excluded_benchmark_demos": sorted(excluded.get(task["task_id"], set())),
                       "train": train, "development": dev, "new_heldout": test})
    return result


def run(args):
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    e2 = load_e2()
    task_manifest = json.loads(TASK_MANIFEST.read_text())
    benchmark = json.loads(BENCH_MANIFEST.read_text())
    if task_manifest.get("manifest_sha256") != "d9cf911a1cb81b8bd635b8991d49e1977c2a50e164fc38cb38ebb1d5a371d156":
        raise RuntimeError("Unexpected E2 task manifest identity")
    tasks = [t for t in task_manifest["tasks"] if t["split"] == "development"]
    if len(tasks) != 6:
        raise RuntimeError(f"Expected six predeclared development tasks, found {len(tasks)}")
    plans = select_demo_splits(tasks, benchmark)
    if args.smoke:
        plans = plans[:1]
        for p in plans:
            p["train"] = p["train"][:1]
            p["development"] = []
            p["new_heldout"] = []
    # The RGB and label tensors are both native 224-square agentview captures.
    geometry = {"resize_size": [224, 224], "resize_interpolation": 3, "crop_size": (224, 224),
                "letterbox": False, "letterbox_fill": None, "resize_strategy": "resize-naive",
                "processor_type": "PrismaticImageProcessor"}
    # Keep stage-local progress separate from the host-side orchestration
    # status, which owns the end-to-end pipeline state.
    status_path = OUT / "stage_progress.json"
    status = {"status": "running", "stage": "render_oracle", "completed_demo_chunks": 0,
              "total_demo_chunks": sum(len(p[s]) for p in plans for s in ("train", "development", "new_heldout")),
              "completed_frames": 0, "error": None, "updated_unix": time.time()}
    atomic_json(status_path, status)
    manifest = {"schema_version": 1, "status": "building", "source_task_manifest_sha256": task_manifest["manifest_sha256"],
                "source_benchmark_manifest_sha256": digest(BENCH_MANIFEST.read_bytes()),
                "source_hdf5": [], "frame_selection": "8 unique evenly-spaced simulator states per complete demonstration",
                "split_unit": "complete HDF5 demonstration", "existing_90_demo_segments_excluded": True,
                "recorded_actions_applied": False, "image_size": [224, 224], "patch_grid": [16, 16],
                "tasks": [], "sample_count_by_split": {}}
    env_count = 0
    try:
        for plan in plans:
            task = plan["task"]
            h5_path = Path(task["hdf5_path"])
            if not h5_path.is_file():
                # Keep the schema paths compatible with both host and container.
                h5_path = DEMO_DATA / task["hdf5_repo_path"]
            if not h5_path.is_file(): raise FileNotFoundError(h5_path)
            task_record = {"task_id": task["task_id"], "suite": task["suite"], "instruction": task["instruction"],
                           "hdf5_path": str(h5_path), "hdf5_sha256": e2.sha256_file(h5_path),
                           "excluded_benchmark_demos": plan["excluded_benchmark_demos"], "splits": {}}
            specs = task_query_specs(e2, task)
            role_info = e2.parse_bddl_roles(Path(task["bddl_path"]))
            env, _ = e2.setup_segmentation_env(task, LIBERO, 224); env_count += 1
            try:
                for split in ("train", "development", "new_heldout"):
                    task_record["splits"][split] = []
                    for demo_name in plan[split]:
                        out_file = CACHE / split / re.sub(r"[^A-Za-z0-9_-]", "_", task["task_id"]) / f"{demo_name}.npz"
                        if out_file.exists():
                            try:
                                with np.load(out_file, allow_pickle=False) as z:
                                    if z["rgb"].shape[1:] == (224,224,3) and z["gt"].shape[1] == len(specs):
                                        task_record["splits"][split].append({"demo": demo_name, "cache": str(out_file), "cached": True,
                                            "frame_count": int(len(z["rgb"])), "source_hdf5_sha256": task_record["hdf5_sha256"]})
                                        status["completed_demo_chunks"] += 1; status["completed_frames"] += len(z["rgb"])
                                        status["updated_unix"] = time.time(); atomic_json(status_path,status)
                                        continue
                            except Exception:
                                pass
                        with h5py.File(h5_path, "r") as f:
                            demo = f[f"data/{demo_name}"]
                            states = np.asarray(demo["states"], dtype=np.float64)
                            xml = demo.attrs["model_file"]
                            if isinstance(xml, bytes): xml = xml.decode()
                        if len(states) < 8: raise RuntimeError(f"Short demo: {task['task_id']}:{demo_name}")
                        n_pick = 2 if args.smoke else 8
                        indices = np.unique(np.rint(np.linspace(0, len(states)-1, n_pick)).astype(np.int32))
                        if len(indices) != n_pick: raise RuntimeError(f"{n_pick} unique frame positions were not selected")
                        names = e2.prepare_demo_env(env, xml, libero_root=LIBERO, robosuite_root=ROBOSUITE)
                        expected = env.env.sim.get_state().flatten().shape[0]
                        if states.shape[1] != expected: raise RuntimeError(f"Simulator state shape mismatch {states.shape[1]} != {expected}")
                        rgb_rows=[]; query_rows=[]; rel_rows=[]; valid_rows=[]
                        for idx in indices:
                            env.env.sim.set_state_from_flattened(states[idx]); env.env.sim.forward()
                            env.check_success(); env.env._post_process(); env.env._update_observables(force=True)
                            obs = env.env._get_observations()
                            rgb = np.asarray(obs["agentview_image"]).copy()
                            seg = np.asarray(obs["agentview_segmentation_instance"]).squeeze(-1).astype(np.int32)
                            if rgb.shape != (224,224,3) or seg.shape != (224,224):
                                raise RuntimeError(f"Unexpected paired capture {rgb.shape}/{seg.shape}")
                            role_masks, _, joint = e2.grid_masks(seg, names, role_info, geometry)
                            qs=[]
                            for spec in specs:
                                merged = np.zeros(256, np.float32)
                                for role_key in spec["gt_keys"]:
                                    merged = np.maximum(merged, np.asarray(role_masks.get(role_key, np.zeros(256,np.float32))))
                                qs.append(merged)
                            rgb_rows.append(rgb.astype(np.uint8)); query_rows.append(qs)
                            rel_rows.append(np.asarray(joint[0],np.float32)); valid_rows.append(np.asarray(joint[1],np.float32))
                        states_sha = digest(np.ascontiguousarray(states[indices]).tobytes())
                        atomic_npz(out_file, rgb=np.asarray(rgb_rows,dtype=np.uint8), gt=np.asarray(query_rows,dtype=np.float16),
                                   relevant=np.asarray(rel_rows,dtype=np.float16), valid=np.asarray(valid_rows,dtype=np.float16),
                                   frame_indices=indices, queries_json=np.asarray(json.dumps(specs,sort_keys=True)),
                                   task_id=np.asarray(task["task_id"]), demo=np.asarray(demo_name), split=np.asarray(split),
                                   states_sha256=np.asarray(states_sha), hdf5_sha256=np.asarray(task_record["hdf5_sha256"]))
                        task_record["splits"][split].append({"demo": demo_name, "cache": str(out_file), "cached": False,
                            "frame_indices": indices.tolist(), "frame_count": int(len(indices)),
                            "states_sha256": states_sha, "source_hdf5_sha256": task_record["hdf5_sha256"]})
                        status["completed_demo_chunks"] += 1; status["completed_frames"] += len(indices)
                        status["current_task"] = task["task_id"]; status["current_demo"] = demo_name
                        status["updated_unix"] = time.time(); atomic_json(status_path,status)
                        print(f"[LABELS] {status['completed_demo_chunks']}/{status['total_demo_chunks']} {task['task_id']} {split}:{demo_name}",flush=True)
            finally:
                env.close()
            manifest["source_hdf5"].append({"task_id": task["task_id"], "path": str(h5_path), "sha256": task_record["hdf5_sha256"]})
            manifest["tasks"].append(task_record)
            atomic_json(CACHE / "dataset_manifest.partial.json", manifest)
        counts={s:sum(x["frame_count"] for t in manifest["tasks"] for x in t["splits"][s]) for s in ("train","development","new_heldout")}
        manifest["sample_count_by_split"]=counts; manifest["status"]="complete"
        manifest["manifest_sha256"]=digest(json.dumps(manifest,sort_keys=True,separators=(",",":" )).encode())
        atomic_json(CACHE / "dataset_manifest.json",manifest)
        atomic_json(OUT / "dataset_manifest.json",manifest)
        status.update({"status":"complete","stage":"render_oracle_complete","completed_frames":sum(counts.values()),
                       "dataset_manifest_sha256":manifest["manifest_sha256"],"updated_unix":time.time()})
        atomic_json(status_path,status)
    except Exception as e:
        status.update({"status":"failed","stage":"render_oracle","error":repr(e),"updated_unix":time.time()})
        atomic_json(status_path,status); raise


def main():
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=123); p.add_argument("--smoke",action="store_true"); run(p.parse_args())


if __name__ == "__main__": main()
