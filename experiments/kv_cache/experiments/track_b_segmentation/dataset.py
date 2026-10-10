"""Replay existing Stage 3A action traces and extract OpenVLA's visual features."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
KV_ROOT = REPO_ROOT / "experiments" / "kv_cache"
OUT = KV_ROOT / "outputs" / "track_b_segmentation"
DATA = OUT / "data"
STAGE3A = KV_ROOT / "outputs" / "stage3a_oracle_masked_matrix"
TASKS = (4, 5, 7)
STATES = (0, 1, 2)
R5_WINDOWS_PER_TRAJECTORY = 7
SEED = 7
MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
BACKGROUND_HINTS = ("table", "floor", "wall", "background", "arena", "skybox", "ground")
ROBOT_HINTS = ("panda", "rethinkmount", "gripper")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def update_status(status: str, **values: Any) -> None:
    path = OUT / "run_status.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update({"status": status, "updated_at_unix": time.time(), **values})
    atomic_json(path, current)


def _category(name: str) -> str:
    lowered = name.lower()
    if any(hint in lowered for hint in ROBOT_HINTS):
        return "foreground_robot"
    if any(hint in lowered for hint in BACKGROUND_HINTS):
        return "background_by_name_rule"
    return "foreground_default_unmapped"


def _oracle_masks(segmentation: np.ndarray, mapping: dict[int, str]) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Stage 3A semantic mapping and exact 256-token nearest mapping."""
    from experiments.kv_cache.experiments.oracle_foreground_mask import (
        foreground_mask,
        policy_mask,
        token_mask,
    )

    seg = np.asarray(segmentation)
    if seg.ndim == 3:
        if seg.shape[-1] != 1:
            raise ValueError(f"Expected one-channel instance segmentation, got {seg.shape}.")
        seg = seg[..., 0]
    semantic = foreground_mask(seg, mapping)
    policy_semantic = policy_mask(semantic[::-1, ::-1], center_crop=True)
    token_fg = token_mask(policy_semantic).reshape(16, 16)
    return policy_semantic.astype(np.uint8), token_fg.astype(np.uint8)


def _sample_windows(record_count: int) -> list[int]:
    starts = np.arange(0, max(0, record_count - 4), 5, dtype=np.int64)
    if not starts.size:
        raise ValueError(f"Trajectory only has {record_count} observations; an R5 window needs five.")
    count = min(R5_WINDOWS_PER_TRAJECTORY, len(starts))
    indices = np.rint(np.linspace(0, len(starts) - 1, count)).astype(np.int64)
    return sorted(set(int(starts[index]) for index in indices))


def _load_trace(task_id: int, state_id: int) -> tuple[dict[str, Any], Path, str]:
    path = STAGE3A / f"task{task_id}_state{state_id}" / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Required Stage 3A action trace is missing: {path}")
    data = json.loads(path.read_text())
    run = data["fresh"]
    if run.get("seed") != SEED or not run.get("records"):
        raise ValueError(f"Unexpected Stage 3A trace metadata in {path}.")
    if not all("policy_action" in record for record in run["records"]):
        raise ValueError(f"Stage 3A trace does not contain replayable actions: {path}")
    return run, path, sha256(path)


def _make_segmentation_env(task: Any) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import SegmentationRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = SegmentationRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
        camera_segmentations="instance",
    )
    env.seed(0)
    return env


def _replay_trajectories() -> dict[str, Any]:
    """Rebuild observations from saved Stage 3A actions; no policy is executed."""
    from libero.libero import benchmark
    from experiments.kv_analysis.experiments.extract_libero_image import (
        DEFAULT_STEP_ACTION,
        normalize_and_invert_openvla_gripper,
    )
    from experiments.kv_cache.experiments.oracle_foreground_mask import rendered_label_mapping
    from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import policy_image_from_observation

    random.seed(SEED)
    np.random.seed(SEED)
    (DATA / "samples").mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": "LIBERO Spatial observations replayed from Stage 3A Fresh action traces",
        "stage3a_trace_root": str(STAGE3A),
        "oracle_mapping_utility": "experiments.kv_cache.experiments.oracle_foreground_mask",
        "seed": SEED,
        "split_rule": "state_id 0 is training, state_id 1 is calibration, state_id 2 is evaluation; entire trajectories stay in one split",
        "r5_windows_per_trajectory": R5_WINDOWS_PER_TRAJECTORY,
        "trajectories": [],
        "samples": [],
        "oracle_class_audit": {},
        "unmapped_segmentation_ids": {},
        "feature_extraction": {"status": "pending"},
    }
    for task_id in TASKS:
        task = suite.get_task(task_id)
        for state_id in STATES:
            run, trace_path, trace_hash = _load_trace(task_id, state_id)
            records = run["records"]
            starts = _sample_windows(len(records))
            selected_steps = {step for start in starts for step in range(start, start + 5)}
            trajectory_id = f"task{task_id}_state{state_id}"
            split = {0: "training", 1: "calibration", 2: "evaluation"}[state_id]
            env = _make_segmentation_env(task)
            try:
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task_id)[state_id])
                mapping = rendered_label_mapping(env.env.model.instances_to_ids)
                for _ in range(10):
                    obs, _, _, _ = env.step(DEFAULT_STEP_ACTION)

                counts = {str(key): 0 for key in mapping}
                unknown_ids: dict[str, int] = {}
                for step, record in enumerate(records):
                    if step in selected_steps:
                        sample_id = f"{trajectory_id}_step{step:04d}"
                        sample_dir = DATA / "samples" / sample_id
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        raw_rgb = np.asarray(obs["agentview_image"], dtype=np.uint8)
                        if raw_rgb.ndim != 3 or raw_rgb.shape[-1] != 3:
                            raise ValueError(f"Unexpected LIBERO RGB shape {raw_rgb.shape}.")
                        policy_rgb = policy_image_from_observation(obs, center_crop=True)
                        segmentation = np.asarray(obs["agentview_segmentation_instance"])
                        seg_2d = segmentation[..., 0] if segmentation.ndim == 3 else segmentation
                        semantic_fg, token_fg = _oracle_masks(segmentation, mapping)
                        Image.fromarray(raw_rgb).save(sample_dir / "rgb_observation.png", optimize=True)
                        Image.fromarray(policy_rgb).save(sample_dir / "policy_rgb.png", optimize=True)
                        np.save(sample_dir / "instance_segmentation.npy", segmentation.astype(np.int32), allow_pickle=False)
                        np.save(sample_dir / "oracle_semantic_fg_mask.npy", semantic_fg, allow_pickle=False)
                        np.save(sample_dir / "oracle_token_mask.npy", token_fg.reshape(-1), allow_pickle=False)
                        file_hashes = {
                            "rgb_observation_sha256": sha256(sample_dir / "rgb_observation.png"),
                            "policy_rgb_sha256": sha256(sample_dir / "policy_rgb.png"),
                            "instance_segmentation_sha256": sha256(sample_dir / "instance_segmentation.npy"),
                            "oracle_semantic_mask_sha256": sha256(sample_dir / "oracle_semantic_fg_mask.npy"),
                            "oracle_token_mask_sha256": sha256(sample_dir / "oracle_token_mask.npy"),
                        }

                        present_ids, present_counts = np.unique(seg_2d, return_counts=True)
                        count_by_id = {int(label): int(count) for label, count in zip(present_ids, present_counts)}
                        for object_id in mapping:
                            counts[str(object_id)] = counts.get(str(object_id), 0) + count_by_id.get(int(object_id), 0)
                        for label, pixels in count_by_id.items():
                            if label != 0 and label not in mapping:
                                key = str(label)
                                unknown_ids[key] = unknown_ids.get(key, 0) + pixels

                        source_step = (step // 5) * 5
                        sample = {
                            "sample_id": sample_id,
                            "task_id": task_id,
                            "task_description": task.language,
                            "state_id": state_id,
                            "trajectory_id": trajectory_id,
                            "split": split,
                            "time_step": step,
                            "r5_source_step": source_step,
                            "r5_age": step - source_step,
                            "r5_source_frame_id": f"{trajectory_id}_step{source_step:04d}",
                            "stage3a_action_trace": str(trace_path.relative_to(REPO_ROOT)),
                            "stage3a_action_trace_sha256": trace_hash,
                            "sample_dir": str(sample_dir.relative_to(OUT)),
                            "rgb_observation_path": str((sample_dir / "rgb_observation.png").relative_to(OUT)),
                            "rgb_path": str((sample_dir / "policy_rgb.png").relative_to(OUT)),
                            "instance_segmentation_path": str((sample_dir / "instance_segmentation.npy").relative_to(OUT)),
                            "oracle_semantic_mask_path": str((sample_dir / "oracle_semantic_fg_mask.npy").relative_to(OUT)),
                            "oracle_token_mask_path": str((sample_dir / "oracle_token_mask.npy").relative_to(OUT)),
                            "feature_path": str((sample_dir / "features.npz").relative_to(OUT)),
                            "feature_status": "pending",
                            "artifact_sha256": file_hashes,
                            "oracle_class_mapping": [
                                {"render_id": int(object_id), "name": name, "class": _category(name)}
                                for object_id, name in mapping.items()
                            ],
                        }
                        manifest["samples"].append(sample)

                    action = normalize_and_invert_openvla_gripper(np.asarray(record["policy_action"], dtype=np.float64))
                    obs, _, done, _ = env.step(action.tolist())
                    if bool(done) and step + 1 < len(records):
                        # The saved trace may include terminal padding in the rare case; do not claim those states.
                        break
            finally:
                env.close()

            classes = [
                {"render_id": int(object_id), "name": name, "class": _category(name)}
                for object_id, name in mapping.items()
            ]
            class_key = f"task{task_id}_state{state_id}"
            manifest["oracle_class_audit"][class_key] = {
                "mapping_source": "rendered_label_mapping(env.env.model.instances_to_ids), the existing Stage 3A mapping",
                "robosuite_render_id_contract": "The installed robosuite instance renderer enumerates model.instances_to_ids.keys(), maps each instance to its zero-based index, and adds 1 to rendered IDs; unmatched/background pixels become 0. The existing Stage 3A helper maps those same ordered names to index+1.",
                "rendered_label_mapping": classes,
                "foreground_default_unmapped_labels": [x["name"] for x in classes if x["class"] == "foreground_default_unmapped"],
                "background_name_rule_labels": [x["name"] for x in classes if x["class"] == "background_by_name_rule"],
                "robot_foreground_labels": [x["name"] for x in classes if x["class"] == "foreground_robot"],
                "pixels_by_render_id_across_sampled_frames": counts,
                "unmapped_nonzero_render_ids_and_pixel_counts": unknown_ids,
            }
            manifest["trajectories"].append({
                "trajectory_id": trajectory_id,
                "task_id": task_id,
                "state_id": state_id,
                "split": split,
                "stage3a_action_trace": str(trace_path.relative_to(REPO_ROOT)),
                "stage3a_action_trace_sha256": trace_hash,
                "steps_available": len(records),
                "sampled_r5_window_starts": starts,
                "sample_count": len(selected_steps),
                "success_in_stage3a_trace": bool(run.get("success")),
                "termination_in_stage3a_trace": run.get("termination"),
            })
            manifest["oracle_class_audit"][class_key]["unmapped_nonzero_render_ids_and_pixel_counts"] = unknown_ids
            manifest["oracle_class_audit"][class_key]["mapping_ids_not_rendered_in_sampled_frames"] = [
                int(object_id) for object_id in mapping if counts.get(str(object_id), 0) == 0
            ]
            atomic_json(OUT / "dataset_manifest.json", manifest)

    if len(manifest["samples"]) < 200:
        raise RuntimeError(f"Only {len(manifest['samples'])} samples were reconstructed; expected at least 200.")
    ids_by_split: dict[str, set[str]] = {"training": set(), "calibration": set(), "evaluation": set()}
    for row in manifest["samples"]:
        ids_by_split[row["split"]].add(row["trajectory_id"])
    if ids_by_split["train_calibration"] & ids_by_split["evaluation"]:
        raise AssertionError("Trajectory leakage detected between calibration and evaluation.")
    manifest["sample_count"] = len(manifest["samples"])
    manifest["split_counts"] = {
        split: sum(row["split"] == split for row in manifest["samples"])
        for split in ("training", "calibration", "evaluation")
    }
    atomic_json(OUT / "dataset_manifest.json", manifest)
    return manifest


def _load_model_for_vision_only() -> tuple[Any, Any, Any, Any]:
    """Load exact checkpoint weights, keep the LLM on CPU, and move only vision to CUDA."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the OpenVLA container; refusing to run a CPU model pass.")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not hasattr(model, "vision_backbone") or not hasattr(model.vision_backbone, "fused_featurizer"):
        raise TypeError("The installed OpenVLA checkpoint is not the expected fused DINOv2/SigLIP model.")
    device = torch.device("cuda:0")
    model.vision_backbone.to(device).eval()
    for parameter in model.language_model.parameters():
        if parameter.device.type != "cpu":
            raise AssertionError("The language model was unexpectedly moved to the accelerator.")
    model.config._track_b_vision_only = True
    return model, processor, device, torch


def _feature_spec(model: Any, processor: Any, pixel_values: Any, dino: Any, siglip: Any) -> dict[str, Any]:
    import inspect

    vision = model.vision_backbone
    dino_vit, siglip_vit = vision.featurizer, vision.fused_featurizer
    dino_intermediate_defaults = inspect.signature(dino_vit.get_intermediate_layers).parameters
    siglip_intermediate_defaults = inspect.signature(siglip_vit.get_intermediate_layers).parameters
    if dino.ndim != 3 or siglip.ndim != 3 or dino.shape[:2] != (1, 256) or siglip.shape[:2] != (1, 256):
        raise AssertionError(f"Expected two [1,256,D] patch tensors; got {tuple(dino.shape)}, {tuple(siglip.shape)}.")
    if dino.shape[2] != dino_vit.embed_dim or siglip.shape[2] != siglip_vit.embed_dim:
        raise AssertionError("Runtime patch dimensions disagree with the installed TIMM vision encoders.")
    if tuple(pixel_values.shape[-2:]) != (224, 224) or pixel_values.shape[1] != 6:
        raise AssertionError(f"Unexpected processor output {tuple(pixel_values.shape)}; expected [1,6,224,224].")
    if dino_vit.patch_embed.grid_size != (16, 16) or siglip_vit.patch_embed.grid_size != (16, 16):
        raise AssertionError("The feature-to-token grid is not 16x16 for both visual branches.")

    image_processor = processor.image_processor
    return {
        "model_id": MODEL_ID,
        "model_type": model.config.model_type,
        "vision_backbone_id": model.config.vision_backbone_id,
        "timm_model_ids": list(model.config.timm_model_ids),
        "use_fused_vision_backbone": bool(model.config.use_fused_vision_backbone),
        "image_resize_strategy": model.config.image_resize_strategy,
        "processor_input_sizes": [list(x) for x in image_processor.input_sizes],
        "processor_interpolations": list(image_processor.interpolations),
        "processor_resize_params": image_processor.tvf_resize_params,
        "processor_crop_params": image_processor.tvf_crop_params,
        "processor_normalization_params": image_processor.tvf_normalize_params,
        "processor_does_letterbox": bool(image_processor.tvf_do_letterbox),
        "processor_letterbox_fill": image_processor.tvf_letterbox_fill,
        "processor_means": [list(x) for x in image_processor.means],
        "processor_stds": [list(x) for x in image_processor.stds],
        "processor_pixel_values_shape": list(pixel_values.shape),
        "dino": {
            "timm_id": model.config.timm_model_ids[0],
            "selected_block_index_zero_based": len(dino_vit.blocks) - 2,
            "num_blocks": len(dino_vit.blocks),
            "feature_dim": int(dino.shape[-1]),
            "patch_size": int(dino_vit.patch_embed.patch_size[0]),
            "grid": list(dino_vit.patch_embed.grid_size),
            "num_prefix_tokens_in_vit": int(dino_vit.num_prefix_tokens),
            "runtime_feature_shape": list(dino.shape),
            "prefix_tokens_returned": bool(dino_intermediate_defaults["return_prefix_tokens"].default),
            "layer_norm_applied": bool(dino_intermediate_defaults["norm"].default),
        },
        "siglip": {
            "timm_id": model.config.timm_model_ids[1],
            "selected_block_index_zero_based": len(siglip_vit.blocks) - 2,
            "num_blocks": len(siglip_vit.blocks),
            "feature_dim": int(siglip.shape[-1]),
            "patch_size": int(siglip_vit.patch_embed.patch_size[0]),
            "grid": list(siglip_vit.patch_embed.grid_size),
            "num_prefix_tokens_in_vit": int(siglip_vit.num_prefix_tokens),
            "runtime_feature_shape": list(siglip.shape),
            "prefix_tokens_returned": bool(siglip_intermediate_defaults["return_prefix_tokens"].default),
            "layer_norm_applied": bool(siglip_intermediate_defaults["norm"].default),
        },
        "fused_feature_dim": int(dino.shape[-1] + siglip.shape[-1]),
        "patch_sequence_order": "row-major raster order produced by timm ViT get_intermediate_layers",
        "openvla_visual_token_range_in_llm_prefill": [1, 256],
        "visual_projector_preserves_sequence_length": True,
        "branch_channel_order": ["DINOv2", "SigLIP"],
        "spatial_correspondence": "Both branches receive channel groups from the same OpenVLA processor output and use 224x224 inputs with 14x14-pixel patches.",
        "wrapper_forward_contract": "PrismaticVisionBackbone splits 6 channels into the configured branch inputs, calls the same two featurizers, and concatenates DINOv2 then SigLIP on the feature axis.",
    }


def _extract_features(manifest: dict[str, Any]) -> dict[str, Any]:
    import torch

    model, processor, device, torch = _load_model_for_vision_only()
    vision = model.vision_backbone
    feature_spec: dict[str, Any] | None = None
    timings: list[float] = []
    completed = 0
    for row in manifest["samples"]:
        sample_dir = OUT / row["sample_dir"]
        feature_path = OUT / row["feature_path"]
        if feature_path.exists():
            try:
                with np.load(feature_path) as cached:
                    if {"dino", "siglip", "fused"}.issubset(cached.files):
                        row["feature_status"] = "complete"
                        row["feature_sha256"] = sha256(feature_path)
                        if row.get("vision_backbone_ms") is not None:
                            timings.append(float(row["vision_backbone_ms"]))
                        completed += 1
                        continue
            except (OSError, ValueError):
                feature_path.unlink(missing_ok=True)

        image = Image.open(OUT / row["rgb_path"]).convert("RGB")
        pixel_values = processor.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        if pixel_values.shape[1] != 6:
            raise AssertionError(f"Expected OpenVLA's paired 6-channel input, got {tuple(pixel_values.shape)}.")
        dino_input, siglip_input = torch.split(pixel_values, [3, 3], dim=1)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            dino = vision.featurizer(dino_input)
            siglip = vision.fused_featurizer(siglip_input)
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
        if feature_spec is None:
            feature_spec = _feature_spec(model, processor, pixel_values, dino, siglip)
        fused = torch.cat([dino, siglip], dim=-1)
        dino_np = dino[0].float().cpu().numpy().astype(np.float16)
        siglip_np = siglip[0].float().cpu().numpy().astype(np.float16)
        fused_np = fused[0].float().cpu().numpy().astype(np.float16)
        temporary = feature_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, dino=dino_np, siglip=siglip_np, fused=fused_np)
        temporary.replace(feature_path)
        row["feature_status"] = "complete"
        row["feature_sha256"] = sha256(feature_path)
        row["vision_backbone_ms"] = timings[-1]
        completed += 1
        if completed % 10 == 0 or completed == len(manifest["samples"]):
            manifest["feature_extraction"] = {
                "status": "running",
                "completed_samples": completed,
                "total_samples": len(manifest["samples"]),
                "median_vision_backbone_ms": float(np.median(timings)) if timings else None,
            }
            atomic_json(OUT / "dataset_manifest.json", manifest)
            update_status("running", phase="feature_extraction", completed_samples=completed, total_samples=len(manifest["samples"]))
            print(f"[track_b] extracted visual features for {completed}/{len(manifest['samples'])} observations", flush=True)

    if feature_spec is None:
        # Fully resumed runs still need a fresh runtime geometry audit from cached files plus loaded configuration.
        first = manifest["samples"][0]
        image = Image.open(OUT / first["rgb_path"]).convert("RGB")
        pixel_values = processor.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        dino_input, siglip_input = torch.split(pixel_values, [3, 3], dim=1)
        with torch.inference_mode():
            dino, siglip = vision.featurizer(dino_input), vision.fused_featurizer(siglip_input)
        feature_spec = _feature_spec(model, processor, pixel_values, dino, siglip)
    if any(row.get("feature_status") != "complete" for row in manifest["samples"]):
        raise RuntimeError("Feature extraction ended with missing per-sample features.")
    existing_times = [float(row["vision_backbone_ms"]) for row in manifest["samples"] if row.get("vision_backbone_ms") is not None]
    if not timings:
        timings = existing_times
    manifest["feature_extraction"] = {
        "status": "complete",
        "completed_samples": completed,
        "total_samples": len(manifest["samples"]),
        "median_vision_backbone_ms": float(np.median(timings)) if timings else None,
        "p95_vision_backbone_ms": float(np.percentile(timings, 95)) if timings else None,
        "classifier_overhead_measured_separately": True,
    }
    manifest["feature_spec"] = feature_spec
    atomic_json(OUT / "dataset_manifest.json", manifest)
    (OUT / "feature_mapping_report.md").write_text(_feature_mapping_report(feature_spec))
    del model
    torch.cuda.empty_cache()
    return manifest


def _feature_mapping_report(spec: dict[str, Any]) -> str:
    return f"""# OpenVLA native feature extraction and spatial mapping

## Installed model and preprocessing

- Checkpoint: `{spec['model_id']}` (`{spec['vision_backbone_id']}`); both feature arrays come from this loaded checkpoint.
- Configured TIMM encoders: `{spec['timm_model_ids'][0]}` for DINOv2 and `{spec['timm_model_ids'][1]}` for SigLIP.
- The OpenVLA image processor emitted `{spec['processor_pixel_values_shape']}` using `{spec['image_resize_strategy']}` preprocessing. Per-branch input sizes are `{spec['processor_input_sizes']}`, resize interpolation is `{spec['processor_interpolations']}`, resize parameters are `{spec['processor_resize_params']}`, center-crop parameters are `{spec['processor_crop_params']}`, and normalization parameters are `{spec['processor_normalization_params']}`. Stored means and standard deviations are `{spec['processor_means']}` / `{spec['processor_stds']}`. Letterbox mode: `{spec['processor_does_letterbox']}`; fill: `{spec['processor_letterbox_fill']}`.
- The feature extractor loaded the full checkpoint to preserve its exact branch weights, left the language model on CPU, and moved only `model.vision_backbone` to CUDA. No language-model forward, action generation, or robot rollout was run for Track B.

## Feature layers and token mapping

| Branch | TIMM encoder | Selected block (zero based) | Blocks | Prefix tokens in ViT | Returned grid | Dimension |
|---|---|---:|---:|---:|---|---:|
| DINOv2 | `{spec['dino']['timm_id']}` | {spec['dino']['selected_block_index_zero_based']} | {spec['dino']['num_blocks']} | {spec['dino']['num_prefix_tokens_in_vit']} | {spec['dino']['grid']} | {spec['dino']['feature_dim']} |
| SigLIP | `{spec['siglip']['timm_id']}` | {spec['siglip']['selected_block_index_zero_based']} | {spec['siglip']['num_blocks']} | {spec['siglip']['num_prefix_tokens_in_vit']} | {spec['siglip']['grid']} | {spec['siglip']['feature_dim']} |

Both branches use the second-to-last transformer block through the same `get_intermediate_layers()` wrappers used by `PrismaticVisionBackbone`. The installed TIMM accessor strips all prefix tokens before returning its patch sequence; its defaults are `norm={spec['dino']['layer_norm_applied']}` for DINOv2 and `norm={spec['siglip']['layer_norm_applied']}` for SigLIP, with `return_prefix_tokens={spec['dino']['prefix_tokens_returned']}` and `{spec['siglip']['prefix_tokens_returned']}`. Thus the class token and DINOv2 register tokens are not returned. Runtime shape checks require `[1,256,{spec['dino']['feature_dim']}]` and `[1,256,{spec['siglip']['feature_dim']}]`. Each feature sequence is a row-major 16×16 patch raster from a 224×224 input with patch size 14.

OpenVLA's fused backbone splits the processor's six channels into DINOv2 and SigLIP inputs, then concatenates the branches on the feature axis in DINOv2-then-SigLIP order. It preserves the 256 patch positions. The language-model prefill places those projected tokens at positions 1–256, as independently recorded by the existing Stage 3A cache layout. The benchmark stores the native branch arrays and their concatenation `{spec['fused_feature_dim']}`-dimensionally; it does not run the projector or LLM.

## Spatial alignment check

The raw LIBERO RGB observation and the processor image are both saved. The processor image is produced by the same Stage 3A helper: rotate LIBERO's camera image by 180°, resize it with the existing evaluator path, and apply its centered 0.9-area crop. The semantic mask is flipped in the same orientation and passed through the existing `policy_mask()` nearest-neighbor resize and center crop; `token_mask()` then marks any foreground pixel in each 14×14 region. The saved comparison figures overlay these exact 16×16 labels on the processor input and draw token boundaries, providing a visual patch-to-token alignment check.

The extracted observations replay existing Stage 3A Fresh action traces to reconstruct pixels and instance maps; they do not invoke OpenVLA for actions. Oracle names classified by the explicit background-name rule, robot foreground hints, or foreground-default-unmapped are enumerated in `dataset_manifest.json`. Nonzero segmentation IDs absent from the rendered-label map are also recorded with pixel counts.
"""


def extract_dataset(resume: bool = True) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "dataset_manifest.json"
    if resume and manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text())
            all_files_exist = all((OUT / row["rgb_path"]).exists() for row in previous.get("samples", []))
            if all_files_exist and previous.get("sample_count", 0) >= 200:
                manifest = previous
            else:
                manifest = _replay_trajectories()
        except (OSError, ValueError, KeyError):
            manifest = _replay_trajectories()
    else:
        manifest = _replay_trajectories()
    update_status("running", phase="feature_extraction", completed_samples=0, total_samples=len(manifest["samples"]))
    manifest = _extract_features(manifest)
    return manifest
