#!/usr/bin/env python3
"""Live Stage 3C R5 KV reuse with a frozen V2 foreground mask.

The V2 DINO features are captured passively in OpenVLA's normal vision forward.
At reuse steps the hook updates the selector before control reaches the Llama
layers, so the policy gets one vision pass and no extra model forward.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# The container's editable OpenVLA install points ``experiments.kv_cache`` at
# the main checkout. Add this isolated branch's package paths first so the
# runner uses its matching Track B modules without copying files into main.
REPO = Path(__file__).resolve().parents[3]
import experiments.kv_cache as _kv_package
_kv_package.__path__.insert(0, str(REPO / "experiments/kv_cache"))
import experiments.kv_cache.experiments as _exp_package
_exp_package.__path__.insert(0, str(REPO / "experiments/kv_cache/experiments"))

from experiments.kv_cache.experiments.hybrid_v2_task_relevance import compose_hybrid_mask
from experiments.kv_cache.experiments.stage3b_oracle_full_kv_matrix import (
    HORIZON, LAYERS, R, SEED, STATES, TASKS, MaskedKVProjectionSkip,
)
from experiments.kv_cache.experiments.stage3c_oracle_task_relevance import RefreshAttentionSidecar
from experiments.kv_cache.experiments.stage4a_oracle_free_attention import (
    ProjectionRowMonitor, _action_for_env, _policy_call, cache_storage_versions,
    make_env, save_mask_visualization,
)
from experiments.kv_cache.experiments.closed_loop_language_kv_reuse import (
    assert_independent_cache_clone, policy_image_from_observation, seed_everything,
)
from experiments.kv_cache.experiments.static_reuse_experiment import (
    DEFAULT_MODEL_ID, clone_prefill_cache, load_model_and_processor, resolve_unnorm_key,
)

HERE = Path(__file__).resolve().parent
OUT = REPO / "experiments/kv_cache/outputs/hybrid_v2_task_relevance"
V2_CKPT = REPO / "experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/best_projection.pt"
SIGLIP = Path("/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384")
V2_SHA256 = "8232ec8f657da47645d73741b16560a1140bfea89cce6826948b9457ce4854d4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


class DINOFeatureTap:
    """Read DINO's existing penultimate patch output and optionally select KV."""
    def __init__(self, model, on_features=None):
        self.dino = getattr(model.vision_backbone, "dino_featurizer",
                            getattr(model.vision_backbone, "featurizer", None))
        if self.dino is None:
            raise RuntimeError("OpenVLA vision_backbone.dino_featurizer is unavailable")
        self.features = None
        self.calls = 0
        self.on_features = on_features
        self.handle = None

    def __enter__(self):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.calls += 1
            self.features = output.detach()
            if self.on_features is not None:
                self.on_features(self.features)
        self.handle = self.dino.register_forward_hook(hook)
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.remove()


def v2_probabilities(features: torch.Tensor, projected_text: torch.Tensor) -> np.ndarray:
    if tuple(features.shape) != (1, 256, 1024):
        raise RuntimeError(f"Expected native DINO [1,256,1024], got {tuple(features.shape)}")
    patch = F.normalize(features.float(), dim=-1)[0]
    text = F.normalize(projected_text.float(), dim=-1)
    probs = torch.sigmoid(10.0 * (patch @ text.T - 0.20)).T
    if tuple(probs.shape) != (projected_text.shape[0], 256) or not torch.isfinite(probs).all():
        raise RuntimeError("V2 probability maps have invalid shape or nonfinite values")
    return probs.detach().cpu().numpy()


def get_task_text(description: str, text_model, tokenizer, projection, device):
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.transfer_validation.libero_demo_ground_truth import instruction_phrases
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import encode_text_embeddings
    phrases = instruction_phrases(description.lower())
    prompts = [f"a photo of a {p['phrase']}" for p in phrases]
    if not prompts:
        raise RuntimeError(f"No task phrases extracted from {description!r}")
    ids = tokenizer(prompts).to(device)
    with torch.inference_mode():
        vectors = encode_text_embeddings(text_model, ids).float()
        projected = F.normalize(projection.text_projection(vectors), dim=-1)
    return phrases, projected


class LiveMaskedKVProjectionSkip(MaskedKVProjectionSkip):
    """Stage 3B projection skip whose mask is finalized by the earlier DINO hook."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_ready = False

    def set_selection(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (256,):
            raise ValueError(f"Live hybrid mask must have 256 entries, got {mask.shape}")
        self.fg = mask.copy()
        self.bg = ~self.fg
        self.selection_ready = True


def render_hybrid_masks(path: Path, image, selection, title):
    from PIL import Image, ImageDraw
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ((selection["v2_foreground"], "V2 foreground", (25, 170, 65)),
              (selection["attention_only_protected"], "Additional attention", (245, 150, 20)),
              (selection["reusable_visual"], "Reuse candidates", (45, 52, 62)))
    rgb = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB").resize((320, 320), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (1280, 366), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black")
    canvas.paste(rgb, (0, 36))
    draw.text((8, 20), "Policy RGB", fill="black")
    for i, (mask, label, color) in enumerate(labels, 1):
        grid = np.full((16, 16, 3), (235, 235, 235), dtype=np.uint8)
        grid.reshape(256, 3)[mask] = color
        panel = Image.fromarray(grid).resize((320, 320), Image.Resampling.NEAREST)
        canvas.paste(panel, (i * 320, 36))
        draw.text((i * 320 + 8, 20), f"{label}: {int(mask.sum())}/256", fill="black")
    canvas.save(path)


def one_episode(task_id, state_id, model, processor, device, unnorm_key,
                text_model, tokenizer, projection, *, smoke=False):
    seed_everything(SEED)
    suite, task, states, env, description = make_env(task_id)
    prompt = f"In: What action should the robot take to {description.lower()}?\nOut:"
    phrases, text_projected = get_task_text(description, text_model, tokenizer, projection, device)
    records, source, done = [], None, False
    try:
        env.reset()
        obs = env.set_init_state(states[state_id])
        from experiments.kv_analysis.experiments.extract_libero_image import DEFAULT_STEP_ACTION
        for _ in range(10):
            obs, _, done, _ = env.step(DEFAULT_STEP_ACTION)
            if done:
                raise RuntimeError("Environment terminated during settling")
        limit = 5 if smoke else HORIZON
        for step in range(limit):
            refresh = step % R == 0
            image = policy_image_from_observation(obs, True)
            context = None
            sidecar = RefreshAttentionSidecar(model) if refresh else None
            selected = {}
            current_probs = {}
            if not refresh:
                expected_anchor = step - step % R
                if source is None or source["step"] != expected_anchor:
                    raise AssertionError(f"R5 anchor mismatch at step {step}: expected {expected_anchor}")
                # The selector is updated inside DINOFeatureTap, before the
                # forward reaches Llama. Attention remains lagged from anchor.
                context = LiveMaskedKVProjectionSkip(model, source["cache"], source["layout"], np.zeros(256, dtype=bool))

                def select_from_live_dino(features):
                    p = v2_probabilities(features, text_projected)
                    selection = compose_hybrid_mask(p, source["attention"], .20, .10)
                    selected["value"] = selection
                    current_probs["value"] = p
                    context.set_selection(selection["fresh_visual"])

                tap = DINOFeatureTap(model, select_from_live_dino)
            else:
                tap = DINOFeatureTap(model)

            with tap:
                pred, wall_ms, monitor = _policy_call(model, processor, image, prompt, device, unnorm_key,
                                                       context if context is not None else sidecar)
            if tap.calls != 1 or tap.features is None:
                raise RuntimeError(f"Expected exactly one passive DINO call; observed {tap.calls}")
            if tuple(pred["layout"]["visual"]) != (1, 256):
                raise AssertionError(f"LLM visual positions unexpected: {pred['layout']['visual']}")
            if tuple(tap.features.shape) != (1, 256, 1024):
                raise AssertionError(f"DINO-to-LLM token alignment unexpected: {tuple(tap.features.shape)}")

            if refresh:
                if sidecar.scores is None or sidecar.scores.shape != (256,) or not np.isfinite(sidecar.scores).all():
                    raise RuntimeError("Refresh attention sidecar did not return 256 finite scores")
                probs = v2_probabilities(tap.features, text_projected)
                selection = compose_hybrid_mask(probs, sidecar.scores, .20, .10)
                source_cache = clone_prefill_cache(pred["cache"])
                assert_independent_cache_clone(source_cache, pred["cache"])
                source = {"cache": source_cache, "layout": dict(pred["layout"]), "step": step,
                          "fresh": True, "attention": sidecar.scores.copy(),
                          "versions": cache_storage_versions(source_cache)}
                selected["value"] = selection
                current_probs["value"] = probs
                age = 0
                if step == 0 and smoke:
                    # Inference instrumentation is passive: compare the exact
                    # prefill logits/actions with a normal call on same input.
                    baseline, _, _ = _policy_call(model, processor, image, prompt, device, unnorm_key,
                                                   monitor=False)
                    if baseline["action_token_ids"] != pred["action_token_ids"]:
                        raise AssertionError("Passive DINO+attention capture changed action tokens")
                    if not torch.equal(baseline["logits"], pred["logits"]):
                        if not torch.allclose(baseline["logits"], pred["logits"], rtol=0, atol=0):
                            delta = float((baseline["logits"].float() - pred["logits"].float()).abs().max())
                            raise AssertionError(f"Passive capture changed fresh logits (max abs {delta})")
                expected_rows = pred["prefill_length"]
                monitor.validate(prefill_length=pred["prefill_length"], projected_kv_rows=expected_rows)
                ctx_diag = None
            else:
                if not context.selection_ready:
                    raise AssertionError("DINO callback failed to set hybrid selection before Llama")
                selection = selected["value"]
                probs = current_probs["value"]
                context.validate(pred["cache"])
                mask = selection["fresh_visual"]
                lang_n = pred["layout"]["language"][1] - pred["layout"]["language"][0] + 1
                expected_rows = 1 + int(mask.sum()) + lang_n
                monitor.validate(prefill_length=pred["prefill_length"], projected_kv_rows=expected_rows)
                if len(context.diag["rows_by_layer"]) != 32:
                    raise AssertionError("K/V skip did not cover all 32 decoder layers")
                if any(r["k_projected_rows"] != expected_rows or r["v_projected_rows"] != expected_rows
                       or r["k_skipped_background_rows"] != int((~mask).sum())
                       or r["v_skipped_background_rows"] != int((~mask).sum())
                       for r in context.diag["rows_by_layer"]):
                    raise AssertionError("K/V projection rows do not match live hybrid mask")
                if cache_storage_versions(source["cache"]) != source["versions"]:
                    raise AssertionError("Reuse mutated the independent R5 anchor")
                age = step - source["step"]
                if age not in (1, 2, 3, 4):
                    raise AssertionError(f"Invalid R5 source age {age}")
                ctx_diag = context.diag

            selection = selected["value"]
            if not np.isfinite(pred["policy_action"]).all():
                raise AssertionError("Nonfinite policy action")
            if int(selection["fresh_visual"].sum() + selection["reusable_visual"].sum()) != 256:
                raise AssertionError("Hybrid partition does not cover 256 visual positions")
            if step in (0, 1):
                render_hybrid_masks(OUT / "mask_visualizations" / f"task{task_id}_state{state_id}_step{step}.png",
                                    image, selection, f"Task {task_id}, state {state_id}, step {step}; score age {age}")
            record = {"step": step, "type": "refresh" if refresh else "reuse",
                      "source_step": source["step"], "source_age": age,
                      "prompt_phrases": phrases, "visual_positions": [1, 256],
                      "dino_shape": list(tap.features.shape), "attention_scores": source["attention"].tolist() if refresh else None,
                      "v2_probability_min": float(probs.min()), "v2_probability_max": float(probs.max()),
                      "v2_foreground_count": selection["counts"]["v2_foreground"],
                      "attention_additional_count": selection["counts"]["attention_only_protected"],
                      "fresh_visual_count": selection["counts"]["fresh_visual"],
                      "reusable_visual_count": selection["counts"]["reusable_visual"],
                      "k_skipped_rows_per_layer": 0 if refresh else selection["counts"]["reusable_visual"],
                      "v_skipped_rows_per_layer": 0 if refresh else selection["counts"]["reusable_visual"],
                      "total_k_skipped_rows": 0 if refresh else 32 * selection["counts"]["reusable_visual"],
                      "total_v_skipped_rows": 0 if refresh else 32 * selection["counts"]["reusable_visual"],
                      "policy_ms": wall_ms, "attention_capture_ms": sidecar.capture_ms if refresh else None,
                      "layers_validated": 32 if refresh else len(ctx_diag["layers"]),
                      "action_token_ids": pred["action_token_ids"], "policy_action": pred["policy_action"].tolist(),
                      "anchor_immutable": True, "fresh_anchor_only": True}
            records.append(record)
            atomic_json(OUT / "partial" / f"task{task_id}_state{state_id}.json",
                        {"status": "running", "task_id": task_id, "state_id": state_id,
                         "steps_completed": len(records), "records": records})
            action = _action_for_env(pred)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
        return {"task_id": task_id, "state_id": state_id, "success": bool(done),
                "steps": len(records), "records": records, "smoke": smoke}
    finally:
        env.close()


def load_components(repo_root: Path, siglip_path: Path, checkpoint: Path):
    if sha256(checkpoint) != V2_SHA256:
        raise RuntimeError(f"Frozen V2 checkpoint SHA mismatch: {sha256(checkpoint)} != {V2_SHA256}")
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import load_siglip_text_tower
    model, processor, device = load_model_and_processor(DEFAULT_MODEL_ID)
    if device.type != "cuda":
        raise RuntimeError(f"CUDA is required for live OpenVLA validation, got {device}")
    key = resolve_unnorm_key(model, None)
    text_model, tokenizer, text_meta = load_siglip_text_tower(siglip_path, device=str(device))
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    projection = RegionTextProjection().to(device).eval()
    projection.load_state_dict(raw["model"], strict=True)
    for module in (model, text_model, projection):
        module.eval()
    for module in (text_model, projection):
        for p in module.parameters():
            p.requires_grad_(False)
    return model, processor, device, key, text_model, tokenizer, projection, text_meta


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "run-all"), required=True)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--v2-checkpoint", type=Path, default=V2_CKPT)
    ap.add_argument("--siglip-checkpoint", type=Path, default=SIGLIP)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    config = {"name": "V2@0.20 + top-10%-attention-protected V2 background; R5 full-layer K+V skip",
              "branch": "thesis/trackb-dense-grounding-v2", "seed": SEED, "tasks": list(TASKS),
              "states": list(STATES), "R": R, "layers": list(LAYERS), "v2_checkpoint": str(args.v2_checkpoint),
              "v2_sha256": V2_SHA256, "threshold": 0.20, "attention_background_fraction": 0.10,
              "dino_features": "passive hook on native OpenVLA DINO featurizer, penultimate [1,256,1024]",
              "attention": "Stage 3C layer-15 read-only Q/K sidecar, refreshed at R5 anchor",
              "policy_rollouts_in_smoke": True,
              "execution_mode": args.mode}
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["configuration_sha256"] = config_sha
    atomic_json(OUT / "configuration.json", config)
    status = {"status": "starting", "mode": args.mode, "configuration_sha256": config_sha,
              "started_unix": time.time(), "completed": []}
    atomic_json(OUT / "RUN_STATUS.json", status)
    try:
        model, processor, device, key, text_model, tokenizer, projection, text_meta = load_components(
            args.repo_root, args.siglip_checkpoint, args.v2_checkpoint)
        status.update({"device": str(device), "gpu": torch.cuda.get_device_name(device),
                       "text_encoder_metadata": text_meta, "status": "running"})
        atomic_json(OUT / "RUN_STATUS.json", status)
        scenarios = [(TASKS[0], STATES[0])] if args.mode == "smoke" else [(t, s) for t in TASKS for s in STATES]
        results = []
        for task_id, state_id in scenarios:
            result_path = OUT / "scenario_results" / f"task{task_id}_state{state_id}.json"
            prior = None
            if args.mode == "run-all" and result_path.is_file():
                try:
                    candidate = json.loads(result_path.read_text())
                    if (candidate.get("configuration_sha256") == config_sha
                            and not candidate.get("smoke")
                            and candidate.get("steps", 0) > 0
                            and (candidate.get("success") or candidate.get("steps") == HORIZON)):
                        prior = candidate
                except (ValueError, OSError, IndexError):
                    prior = None
            if prior is not None:
                result = prior
            else:
                status["current_scenario"] = {"task_id": task_id, "state_id": state_id}
                atomic_json(OUT / "RUN_STATUS.json", status)
                result = one_episode(task_id, state_id, model, processor, device, key,
                                     text_model, tokenizer, projection, smoke=(args.mode == "smoke"))
                result["configuration_sha256"] = config_sha
            results.append(result)
            atomic_json(result_path, result)
            status["completed"].append({"task_id": task_id, "state_id": state_id,
                                         "success": result["success"], "steps": result["steps"],
                                         "resumed": prior is not None})
            status.pop("current_scenario", None)
            status["status"] = "running"
            atomic_json(OUT / "RUN_STATUS.json", status)
        status["status"] = "complete"
        status["finished_unix"] = time.time()
        status["scenario_summary"] = [{"task_id": r["task_id"], "state_id": r["state_id"],
                                        "success": r["success"], "steps": r["steps"]} for r in results]
        atomic_json(OUT / "RUN_STATUS.json", status)
        write_report(config, results, status)
    except Exception as exc:
        status.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                       "failed_unix": time.time()})
        atomic_json(OUT / "RUN_STATUS.json", status)
        raise


def write_report(config, results, status):
    rows = [r for ep in results for r in ep["records"]]
    reuse = [r for r in rows if r["type"] == "reuse"]
    text = ["# Hybrid V2 + Task-Relevance KV Reuse", "",
            f"Execution status: **{status['status']}**.", "",
            "The frozen V2 epoch-40 mask at probability threshold 0.20 supplies foreground tokens. The top 10% of its predicted background, ranked by the most recent Stage 3C layer-15 text-to-visual relevance vector, is additionally protected. R5 retains an independent full-refresh source and allows four reuse decisions. DINO features are captured passively from the existing OpenVLA vision forward; no second DINO pass is run.", "",
            "## Results", "",
            f"Completed scenarios: {len(results)} / 9. Successes: {sum(bool(x['success']) for x in results)} / {len(results)}.", "",
            f"Across {len(reuse)} reuse decisions, mean reusable visual tokens: {np.mean([x['reusable_visual_count'] for x in reuse]) if reuse else float('nan'):.2f}/256. Mean skipped K rows per layer and decision: {np.mean([x['k_skipped_rows_per_layer'] for x in reuse]) if reuse else float('nan'):.2f}; same count for V. Mean synchronized policy-call time: {np.mean([x['policy_ms'] for x in rows]):.2f} ms. This is descriptive timing, not a speedup claim.", "",
            "| Task | State | Success | Steps | Reuse decisions | Mean reuse candidates | K+V rows skipped per reuse decision (32 layers) |", "|---:|---:|:---:|---:|---:|---:|---:|"]
    for ep in results:
        rr = [x for x in ep["records"] if x["type"] == "reuse"]
        text.append(f"| {ep['task_id']} | {ep['state_id']} | {ep['success']} | {ep['steps']} | {len(rr)} | {np.mean([x['reusable_visual_count'] for x in rr]) if rr else 0:.2f} | {sum(x['total_k_skipped_rows']+x['total_v_skipped_rows'] for x in rr)} |")
    references = {}
    stage3c_csv = REPO / "experiments/kv_cache/outputs/stage3c_oracle_task_relevance/per_scenario_metrics.csv"
    stage4a_summary = REPO / "experiments/kv_cache/outputs/stage4a_oracle_free_attention/summary.json"
    if stage3c_csv.is_file():
        with stage3c_csv.open(newline="") as f:
            baseline_rows = list(csv.DictReader(f))
        fresh = {(int(x["task_id"]), int(x["state_id"])): x["fresh_success"] == "True"
                 for x in baseline_rows if x["protection_percent"] == "10"}
        oracle = {(int(x["task_id"]), int(x["state_id"])): x["stage3c_success"] == "True"
                  for x in baseline_rows if x["protection_percent"] == "10"}
        references["fresh_stage3c_reference"] = {"successes": sum(fresh.values()), "scenarios": len(fresh),
            "source": str(stage3c_csv), "scenario_success": {f"{k[0]}:{k[1]}": v for k, v in fresh.items()}}
        references["stage3c_oracle_10pct_reference"] = {"successes": sum(oracle.values()), "scenarios": len(oracle),
            "source": str(stage3c_csv), "scenario_success": {f"{k[0]}:{k[1]}": v for k, v in oracle.items()}}
    if stage4a_summary.is_file():
        d = json.loads(stage4a_summary.read_text())
        references["stage4a_attention_only"] = {"budget_method_aggregates": d.get("budget_method_aggregates", []),
                                                 "source": str(stage4a_summary)}
    atomic_json(OUT / "comparison_references.json", references)
    text += ["", "## Historical matched-condition references", "",
             "These are saved historical outcomes under the published scenario matrix and seed; they are reference comparisons, not paired replays of the hybrid's changed trajectories.", "",
             "| Condition | Success | Source |", "|---|---:|---|"]
    for name, label in (("fresh_stage3c_reference", "Fresh OpenVLA"),
                        ("stage3c_oracle_10pct_reference", "Stage 3C oracle FG + 10% attention protection")):
        if name in references:
            ref = references[name]
            text.append(f"| {label} | {ref['successes']}/{ref['scenarios']} | `{Path(ref['source']).name}` |")
    if "stage4a_attention_only" in references:
        for agg in references["stage4a_attention_only"]["budget_method_aggregates"]:
            if agg.get("method") == "attention":
                text.append(f"| Stage 4A attention-only, {agg.get('fresh_budget_pct')}% fresh | {agg.get('successes')}/{agg.get('total')} | `summary.json` |")
    scenario_rows = []
    for ep in results:
        recs = ep["records"]
        rr = [x for x in recs if x["type"] == "reuse"]
        scenario_rows.append({"task_id": ep["task_id"], "state_id": ep["state_id"],
            "success": ep["success"], "steps": ep["steps"], "refresh_steps": sum(x["type"] == "refresh" for x in recs),
            "reuse_steps": len(rr), "mean_v2_fg_tokens_per_decision": float(np.mean([x["v2_foreground_count"] for x in recs])),
            "mean_attention_additional_tokens": float(np.mean([x["attention_additional_count"] for x in recs])),
            "mean_reuse_candidates_per_decision": float(np.mean([x["reusable_visual_count"] for x in rr])) if rr else 0.0,
            "total_k_rows_skipped": sum(x["total_k_skipped_rows"] for x in rr),
            "total_v_rows_skipped": sum(x["total_v_skipped_rows"] for x in rr),
            "mean_policy_ms": float(np.mean([x["policy_ms"] for x in recs])),
            "p50_policy_ms": float(np.median([x["policy_ms"] for x in recs])),
            "p95_policy_ms": float(np.percentile([x["policy_ms"] for x in recs], 95))})
    with (OUT / "per_scenario_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenario_rows[0]) if scenario_rows else [], lineterminator="\n")
        if scenario_rows:
            writer.writeheader(); writer.writerows(scenario_rows)
    summary = {"status": status["status"], "configuration": config,
        "scenario_count": len(results), "successes": sum(bool(x["success"]) for x in results),
        "mean_reused_tokens_per_reuse_decision": float(np.mean([x["reusable_visual_count"] for x in reuse])) if reuse else None,
        "mean_v2_foreground_tokens_per_decision": float(np.mean([x["v2_foreground_count"] for x in rows])) if rows else None,
        "mean_attention_additional_tokens_per_decision": float(np.mean([x["attention_additional_count"] for x in rows])) if rows else None,
        "mean_policy_ms": float(np.mean([x["policy_ms"] for x in rows])) if rows else None,
        "p50_policy_ms": float(np.median([x["policy_ms"] for x in rows])) if rows else None,
        "p95_policy_ms": float(np.percentile([x["policy_ms"] for x in rows],95)) if rows else None,
        "total_k_rows_skipped": int(sum(x["total_k_skipped_rows"] for x in reuse)),
        "total_v_rows_skipped": int(sum(x["total_v_skipped_rows"] for x in reuse)),
        "references": references,
        "scenario_summary": [{"task_id": e["task_id"], "state_id": e["state_id"],
                              "success": e["success"], "steps": e["steps"]} for e in results]}
    atomic_json(OUT / "summary.json", summary)
    text += ["", "## Validation notes", "",
             "The runtime asserts one native DINO hook call per policy input, 256 aligned DINO patch positions mapped to LLM positions 1–256, finite V2 probabilities, fresh action/logit parity during smoke validation, all 32 patched layers, full Q projection, exact fresh K/V row counts, source-cache immutability, and source age 1–4.", "",
             "The mask labels denote cache freshness priority. They are not ground-truth physical segmentation. Skipped projection counts are structural savings; synchronized wall-time measurements include policy execution and do not by themselves establish acceleration.", "",
             "## Artifacts", "", "- `RUN_STATUS.json` — persistent status and completed scenarios.", "- `scenario_results/` — per-scenario actions and per-step cache diagnostics.", "- `mask_visualizations/` — V2, attention-only, and reuse-candidate masks.", "- `configuration.json` — fixed experiment configuration and checkpoint identity.", ""]
    (OUT / "HYBRID_V2_TASK_RELEVANCE_REPORT.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
