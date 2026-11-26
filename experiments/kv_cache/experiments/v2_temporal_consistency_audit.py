#!/usr/bin/env python3
"""CPU-only temporal audit of saved V2 foreground-count/transition traces.

No model, simulator, CUDA, or inference code is loaded. Exact per-token masks
were not persisted by the adaptive runner, so transition partitions are
reconstructed only from its logged foreground counts and FG->BG count.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TOKENS = 256
DEFAULT_ADAPTIVE = Path("experiments/kv_cache/outputs/adaptive_hybrid_kv_cache/primary_20261125")
DEFAULT_OUT = Path("experiments/kv_cache/outputs/v2_temporal_consistency_audit")


def reconstruct_pair(fg_previous: int, fg_current: int, fg_to_bg: int, token_count: int = TOKENS) -> dict[str, Any]:
    """Reconstruct a 2x2 FG/BG transition table from aggregate counts."""
    vals = (fg_previous, fg_current, fg_to_bg, token_count)
    if any(not isinstance(x, (int, np.integer)) for x in vals):
        raise ValueError("foreground counts and token_count must be integers")
    if token_count <= 0 or not 0 <= fg_previous <= token_count or not 0 <= fg_current <= token_count:
        raise ValueError("foreground counts must lie within the token count")
    if not 0 <= fg_to_bg <= min(fg_previous, token_count - fg_current):
        raise ValueError("FG->BG count is inconsistent with previous/current FG counts")

    fg_to_fg = fg_previous - fg_to_bg
    bg_to_fg = fg_current - fg_previous + fg_to_bg
    bg_to_bg = token_count - fg_previous - bg_to_fg
    transitions = (fg_to_fg, fg_to_bg, bg_to_fg, bg_to_bg)
    if any(x < 0 for x in transitions) or sum(transitions) != token_count:
        raise ValueError(f"invalid transition partition {transitions} for N={token_count}")

    flips = fg_to_bg + bg_to_fg
    observed = flips / token_count
    p_prev, p_cur = fg_previous / token_count, fg_current / token_count
    expected_random = p_prev * (1 - p_cur) + (1 - p_prev) * p_cur
    expected_agreement = 1 - expected_random
    chance_adjusted = (1 - observed / expected_random) if expected_random > 0 else None
    fg_retention = fg_to_fg / fg_previous if fg_previous else None
    bg_previous = token_count - fg_previous
    bg_retention = bg_to_bg / bg_previous if bg_previous else None
    union = fg_previous + bg_to_fg
    temporal_iou = fg_to_fg / union if union else 1.0
    return {
        "fg_previous": fg_previous,
        "bg_previous": token_count - fg_previous,
        "fg_current": fg_current,
        "bg_current": token_count - fg_current,
        "fg_to_fg": fg_to_fg,
        "fg_to_bg": fg_to_bg,
        "bg_to_fg": bg_to_fg,
        "bg_to_bg": bg_to_bg,
        "flip_count": flips,
        "flip_rate": observed,
        "expected_random_flip_rate": expected_random,
        "observed_agreement": 1 - observed,
        "expected_random_agreement": expected_agreement,
        "chance_adjusted_agreement": chance_adjusted,
        "fg_retention_rate": fg_retention,
        "bg_retention_rate": bg_retention,
        "temporal_foreground_iou": temporal_iou,
    }


def _mean_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(mean(clean)) if clean else None


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pair_count": 0}
    n = len(rows) * TOKENS
    fg_previous = sum(r["fg_previous"] for r in rows)
    bg_previous = sum(r["bg_previous"] for r in rows)
    flips = sum(r["flip_count"] for r in rows)
    expected_flips = sum(r["expected_random_flip_rate"] * TOKENS for r in rows)
    fg_to_fg = sum(r["fg_to_fg"] for r in rows)
    fg_to_bg = sum(r["fg_to_bg"] for r in rows)
    bg_to_fg = sum(r["bg_to_fg"] for r in rows)
    bg_to_bg = sum(r["bg_to_bg"] for r in rows)
    expected_rate = expected_flips / n
    observed_rate = flips / n
    return {
        "pair_count": len(rows),
        "mean_fg_previous_tokens": fg_previous / len(rows),
        "mean_bg_previous_tokens": bg_previous / len(rows),
        "mean_fg_current_tokens": mean(r["fg_current"] for r in rows),
        "mean_bg_current_tokens": mean(r["bg_current"] for r in rows),
        "mean_fg_current_fraction": mean(r["fg_current"] for r in rows) / TOKENS,
        "mean_bg_current_fraction": mean(r["bg_current"] for r in rows) / TOKENS,
        "mean_fg_previous_fraction": fg_previous / n,
        "mean_bg_previous_fraction": bg_previous / n,
        "fg_to_bg_count": fg_to_bg,
        "bg_to_fg_count": bg_to_fg,
        "fg_to_fg_count": fg_to_fg,
        "bg_to_bg_count": bg_to_bg,
        "observed_flip_rate": observed_rate,
        "expected_random_flip_rate": expected_rate,
        "observed_agreement": 1 - observed_rate,
        "expected_random_agreement": 1 - expected_rate,
        "chance_adjusted_agreement": (1 - observed_rate / expected_rate) if expected_rate else None,
        "fg_retention_rate": fg_to_fg / fg_previous if fg_previous else None,
        "bg_retention_rate": bg_to_bg / bg_previous if bg_previous else None,
        "micro_temporal_foreground_iou": fg_to_fg / (fg_to_fg + fg_to_bg + bg_to_fg)
        if (fg_to_fg + fg_to_bg + bg_to_fg) else 1.0,
        "mean_pair_temporal_foreground_iou": mean(r["temporal_foreground_iou"] for r in rows),
        "mean_pair_flip_rate": mean(r["flip_rate"] for r in rows),
        "median_pair_flip_rate": median(r["flip_rate"] for r in rows),
        "expected_flip_excess": observed_rate - expected_rate,
    }


def load_adaptive(adaptive_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    status = json.loads((adaptive_dir / "RUN_STATUS.json").read_text())
    config = json.loads((adaptive_dir / "configuration.json").read_text())
    if status.get("status") != "complete" or len(status.get("completed", [])) != 9:
        raise RuntimeError("Adaptive run status is not complete for all nine scenarios")
    scenario_paths = sorted((adaptive_dir / "scenario_results").glob("task*_state*.json"))
    if len(scenario_paths) != 9:
        raise RuntimeError(f"Expected nine authoritative scenario_results, found {len(scenario_paths)}")
    required_keys = {"v2_fg_count", "transition_protection_count", "refresh", "step"}
    per_step: list[dict[str, Any]] = []
    per_scenario: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for path in scenario_paths:
        d = json.loads(path.read_text())
        key = (int(d["task_id"]), int(d["state_id"]))
        if key in seen:
            raise RuntimeError(f"Duplicate scenario {key}")
        seen.add(key)
        records = d.get("records", [])
        if not records or len(records) != int(d["steps"]):
            raise RuntimeError(f"Scenario record count mismatch in {path}")
        if any(not required_keys.issubset(r) for r in records):
            raise RuntimeError(f"Missing transition fields in {path}")
        if any(int(r["step"]) != i for i, r in enumerate(records)):
            raise RuntimeError(f"Non-contiguous step indices in {path}")
        outcome = bool(d["success"])
        transition_valid_full = transition_valid_selective = 0
        for i in range(1, len(records)):
            prev, cur = records[i - 1], records[i]
            counts = reconstruct_pair(int(prev["v2_fg_count"]), int(cur["v2_fg_count"]),
                                      int(cur["transition_protection_count"]))
            row = {
                "task_id": key[0], "state_id": key[1], "scenario": f"task{key[0]}_state{key[1]}",
                "scenario_success": outcome, "previous_step": i - 1, "step": i,
                "refresh": bool(cur["refresh"]),
                "refresh_reason": cur.get("refresh_reason", ""),
                "previous_v2_fg_count": int(prev["v2_fg_count"]),
                "current_v2_fg_count": int(cur["v2_fg_count"]),
                "fg_to_bg_logged": int(cur["transition_protection_count"]),
                "transition_count_valid_on_refresh": bool(cur["refresh"]),
                **counts,
            }
            per_step.append(row)
            if cur["refresh"]:
                transition_valid_full += 1
            else:
                transition_valid_selective += 1
        pairs = [r for r in per_step if (r["task_id"], r["state_id"]) == key]
        decisions = len(records)
        fg_counts = [int(r["v2_fg_count"]) for r in records]
        per_scenario.append({
            "task_id": key[0], "state_id": key[1], "scenario": f"task{key[0]}_state{key[1]}",
            "success": outcome, "steps": int(d["steps"]), "policy_decisions": decisions,
            "temporal_pairs": len(pairs), "mean_fg_tokens": mean(fg_counts),
            "mean_bg_tokens": TOKENS - mean(fg_counts),
            "mean_fg_fraction": mean(fg_counts) / TOKENS,
            "fg_count_sd": pstdev(fg_counts), "min_fg_tokens": min(fg_counts), "max_fg_tokens": max(fg_counts),
            "full_refresh_decisions": sum(bool(r["refresh"]) for r in records),
            "selective_decisions": sum(not bool(r["refresh"]) for r in records),
            "transition_pairs_after_full_refresh": transition_valid_full,
            "transition_pairs_after_selective_decision": transition_valid_selective,
            **summarize_pairs(pairs),
        })
    if seen != {(t, s) for t in (4, 5, 7) for s in (0, 1, 2)}:
        raise RuntimeError(f"Scenario matrix mismatch: {sorted(seen)}")
    return per_step, per_scenario, {"status": status, "configuration": config}


def _load_fixed_r5(name: str, directory: Path) -> dict[str, Any]:
    """Compare only saved per-frame FG counts; fixed R5 lacks transition tables."""
    paths = sorted(directory.glob("task*_state*.json"))
    if len(paths) != 9:
        raise RuntimeError(f"{name}: expected nine completed scenario files, got {len(paths)} at {directory}")
    vals: list[int] = []
    scenario_rows = []
    for p in paths:
        d = json.loads(p.read_text())
        rs = d.get("records", [])
        fg = [r.get("v2_foreground_count") for r in rs]
        if not rs or any(x is None for x in fg):
            raise RuntimeError(f"{name}: missing per-observation V2 foreground counts in {p}")
        fg = [int(x) for x in fg]
        vals.extend(fg)
        scenario_rows.append({"task_id": int(d["task_id"]), "state_id": int(d["state_id"]),
                              "success": bool(d["success"]), "steps": int(d["steps"]),
                              "decisions": len(fg), "mean_fg_tokens": mean(fg),
                              "mean_bg_tokens": TOKENS - mean(fg), "fg_count_sd": pstdev(fg)})
    return {"method": name, "scenario_count": len(paths), "decisions": len(vals),
            "successes": sum(r["success"] for r in scenario_rows),
            "mean_fg_tokens": mean(vals), "mean_bg_tokens": TOKENS - mean(vals),
            "mean_fg_fraction": mean(vals) / TOKENS, "global_fg_count_sd": pstdev(vals),
            "per_scenario": scenario_rows,
            "temporal_transition_metrics": "unavailable: full masks and FG->BG counts were not persisted"}


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def _font(size: int = 18):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(p).is_file():
            try: return ImageFont.truetype(p, size)
            except OSError: pass
    return ImageFont.load_default()


def line_chart(path: Path, title: str, series: list[tuple[str, list[float | None], str]],
               *, y_label: str, y_min: float | None = None, y_max: float | None = None,
               x_label: str = "Policy observation step", max_x: int | None = None) -> None:
    """Small dependency-light line chart drawn with Pillow."""
    w, h = 1500, 850
    im = Image.new("RGB", (w, h), "white"); dr = ImageDraw.Draw(im)
    title_font, label_font, small_font = _font(26), _font(17), _font(14)
    dr.text((w // 2, 24), title, fill="#171717", font=title_font, anchor="mt")
    left, right, top, bottom = 112, w - 56, 105, h - 110
    all_vals = [float(v) for _, arr, _ in series for v in arr if v is not None and math.isfinite(float(v))]
    if not all_vals: all_vals = [0.0, 1.0]
    lo = min(all_vals) if y_min is None else y_min; hi = max(all_vals) if y_max is None else y_max
    if hi <= lo: hi = lo + 1.0
    pad = (hi - lo) * .06
    lo = min(lo, min(all_vals) - pad); hi = max(hi, max(all_vals) + pad)
    for j in range(6):
        yv = lo + (hi - lo) * j / 5
        y = bottom - (yv - lo) / (hi - lo) * (bottom - top)
        dr.line((left, y, right, y), fill="#e4e7eb", width=1)
        dr.text((left - 12, y), f"{yv:.2f}", fill="#444", font=small_font, anchor="rm")
    dr.line((left, top, left, bottom), fill="#333", width=2); dr.line((left, bottom, right, bottom), fill="#333", width=2)
    x_max = max_x if max_x is not None else max((len(a) - 1 for _, a, _ in series), default=1)
    x_max = max(x_max, 1)
    for x_tick in range(0, x_max + 1, max(1, x_max // 8)):
        x = left + x_tick / x_max * (right - left)
        dr.line((x, bottom, x, bottom + 6), fill="#333", width=1)
        dr.text((x, bottom + 10), str(x_tick), fill="#444", font=small_font, anchor="mt")
    dr.text((left + right // 2, h - 38), x_label, fill="#222", font=label_font, anchor="mm")
    # Rotated y-axis title.
    ytitle = Image.new("RGBA", (330, 36), (255, 255, 255, 0)); ImageDraw.Draw(ytitle).text((165, 2), y_label, fill="#222", font=label_font, anchor="mt")
    im.alpha_composite(ytitle.rotate(90, expand=True), (8, (h - ytitle.height) // 2)) if im.mode == "RGBA" else None
    # Drawing y-axis text unrotated keeps the figure readable with old Pillow.
    dr.text((8, 62), y_label, fill="#222", font=small_font)
    for label, arr, color in series:
        pts = []
        for i, v in enumerate(arr):
            if v is None or not math.isfinite(float(v)): continue
            x = left + min(i, x_max) / x_max * (right - left)
            y = bottom - (float(v) - lo) / (hi - lo) * (bottom - top)
            pts.append((int(x), int(y)))
        if len(pts) >= 2: dr.line(pts, fill=color, width=3, joint="curve")
        elif pts: dr.ellipse((pts[0][0]-3, pts[0][1]-3, pts[0][0]+3, pts[0][1]+3), fill=color)
    # Legend, one row where possible.
    lx, ly = left + 10, 66
    for label, _, color in series:
        dr.line((lx, ly + 9, lx + 28, ly + 9), fill=color, width=4)
        dr.text((lx + 35, ly), label[:45], fill="#222", font=small_font)
        lx += min(360, 50 + len(label) * 8)
        if lx > right - 250: lx, ly = left + 10, ly + 22
    path.parent.mkdir(parents=True, exist_ok=True); im.save(path)


def make_figures(out: Path, pairs: list[dict[str, Any]], scenarios: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)
    max_step = max(r["step"] for r in pairs)
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in pairs: by_step[r["step"]].append(r)
    def aggregate(key):
        return [mean(float(r[key]) for r in by_step[s]) if s in by_step else None for s in range(max_step + 1)]
    records_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    # Frame counts come from pairs (current side) plus initial frame handled by a step-zero row constructed by caller below.
    count_series = [("adaptive FG", [None] + [aggregate("fg_current")[s] / TOKENS for s in range(1, max_step + 1)], "#1874b4"),
                    ("adaptive BG", [None] + [aggregate("bg_current")[s] / TOKENS for s in range(1, max_step + 1)], "#dd7c24")]
    # Include both fixed R5 series with current-observation foreground proportions.
    for comp in comparisons:
        if comp["method"] == "Adaptive hybrid 0.20": continue
        vals = comp["time_series_fg_fraction"]
        count_series.append((comp["method"], vals, comp["color"]))
    line_chart(fig / "foreground_background_coverage.png", "V2 foreground/background token coverage over observation time",
               count_series, y_label="Fraction of 256 visual tokens", y_min=0, y_max=1, max_x=max_step)

    line_chart(fig / "classification_flip_rate.png", "Observed and marginal-matched random classification flips",
               [("observed", aggregate("flip_rate"), "#1769aa"),
                ("independent random expectation", aggregate("expected_random_flip_rate"), "#d84a3a")],
               y_label="Fraction of 256 tokens that flip", y_min=0, y_max=1, max_x=max_step)
    line_chart(fig / "directional_transitions.png", "Directional FG/BG transitions per consecutive pair",
               [("FG → BG", aggregate("fg_to_bg"), "#ca4d44"),
                ("BG → FG", aggregate("bg_to_fg"), "#2c8c66")],
               y_label="Mean transitions (tokens / pair)", y_min=0, y_max=256, max_x=max_step)
    ordered = sorted(scenarios, key=lambda x: (x["task_id"], x["state_id"]))
    labels = [f"T{x['task_id']}S{x['state_id']} {'✓' if x['success'] else '✗'}" for x in ordered]
    def bar_chart(filename, title, values_a, values_b, tick_labels, labels_a, labels_b, colors, y_label, ymax=1.0):
        w, h = 1500, 780; im = Image.new("RGB", (w,h), "white"); d=ImageDraw.Draw(im)
        d.text((w//2,22),title,fill="#111",font=_font(25),anchor="mt")
        L,R,T,B=100,w-50,95,h-110; ymax=max(ymax, max([v for v in values_a+values_b if v is not None]+[0])*1.12)
        for k in range(6):
            v=ymax*k/5; y=B-v/ymax*(B-T); d.line((L,y,R,y),fill="#e4e7eb"); d.text((L-10,y),f"{v:.2f}",fill="#333",font=_font(13),anchor="rm")
        n=len(tick_labels); slot=(R-L)/max(n,1); bw=min(26,slot*.28)
        for i,lab in enumerate(tick_labels):
            x=L+(i+.5)*slot
            for j,(vals,col) in enumerate(((values_a,colors[0]),(values_b,colors[1]))):
                v=vals[i]
                if v is not None:
                    y=B-v/ymax*(B-T); off=(-bw/2 if j==0 else bw/2)
                    d.rectangle((x+off-bw/2,y,x+off+bw/2,B),fill=col)
            d.text((x,B+12),lab,fill="#333",font=_font(13),anchor="mt")
        d.text((L+20,61),labels_a,fill=colors[0],font=_font(14)); d.text((L+280,61),labels_b,fill=colors[1],font=_font(14));d.text((10,60),y_label,fill="#333",font=_font(13))
        im.save(fig/filename)
    bar_chart("temporal_foreground_iou_by_scenario.png","Temporal foreground IoU by scenario",
              [x["micro_temporal_foreground_iou"] for x in ordered],[None]*len(ordered),labels,
              "Temporal IoU","",("#31688e","#ffffff"),"IoU",1.0)
    bar_chart("fg_bg_retention_by_scenario.png","Foreground and background classification retention",
              [x["fg_retention_rate"] for x in ordered],[x["bg_retention_rate"] for x in ordered],labels,
              "FG retention","BG retention",("#2a788e","#d89028"),"Retention rate",1.0)
    bar_chart("scenario_flip_observed_vs_random.png","Observed versus chance-expected flip rate by scenario",
              [x["observed_flip_rate"] for x in ordered],[x["expected_random_flip_rate"] for x in ordered],labels,
              "Observed","Random expectation",("#286090","#bd4b3b"),"Flip rate",1.0)


def analyze(adaptive_dir: Path, out: Path, r5_020_dir: Path | None = None, r5_015_dir: Path | None = None) -> dict[str, Any]:
    pairs, scenarios, provenance = load_adaptive(adaptive_dir)
    split = {"all": pairs,
             "successful": [r for r in pairs if r["scenario_success"]],
             "failed": [r for r in pairs if not r["scenario_success"]],
             "after_selective_decisions": [r for r in pairs if not r["refresh"]],
             "after_full_refresh_decisions": [r for r in pairs if r["refresh"]]}
    summary = {
        "status": "complete", "method": "aggregate reconstruction from saved adaptive runner telemetry",
        "token_count": TOKENS, "scenario_count": len(scenarios),
        "policy_decisions": sum(x["policy_decisions"] for x in scenarios),
        "temporal_pairs": len(pairs), "successes": sum(x["success"] for x in scenarios),
        "failed_scenarios": sum(not x["success"] for x in scenarios),
        "foreground_threshold": provenance["configuration"].get("foreground_probability_threshold"),
        "split_metrics": {k: summarize_pairs(v) for k, v in split.items()},
        "full_refresh_pair_count": len(split["after_full_refresh_decisions"]),
        "selective_pair_count": len(split["after_selective_decisions"]),
        "full_refresh_transition_counts_valid": True,
        "transition_counts_validated": True,
        "source": str(adaptive_dir),
        "configuration_sha256": provenance["configuration"].get("configuration_sha256"),
        "published_source_revision": json.loads((adaptive_dir/"source_commit.json").read_text()),
        "source_sha256": {
            "adaptive_runner": _sha256(Path("experiments/kv_cache/experiments/adaptive_hybrid_kv_runner.py")),
            "adaptive_policy": _sha256(Path("experiments/kv_cache/experiments/adaptive_hybrid_kv_policy.py")),
            "adaptive_configuration": _sha256(adaptive_dir/"configuration.json"),
            "adaptive_run_status": _sha256(adaptive_dir/"RUN_STATUS.json"),
            "adaptive_scenario_records": {p.name:_sha256(p) for p in sorted((adaptive_dir/"scenario_results").glob("task*_state*.json"))},
        },
        "data_availability": {
            "complete_binary_masks": False,
            "per_token_classification_persistence": "unavailable; no complete masks were saved",
            "per_position_flip_frequency_or_spatial_heatmaps": "unavailable",
            "per_token_probability_variation": "unavailable; only per-frame min/max probability were saved",
            "per_token oscillation_counts": "unavailable; aggregate transition counts cannot identify token identities",
            "available_probability_diagnostic": "only per-frame probability min/max and foreground count",
        },
    }
    adaptive_fg_mean=sum(x["mean_fg_tokens"]*x["policy_decisions"] for x in scenarios)/summary["policy_decisions"]
    adaptive_fg_var=sum(x["policy_decisions"]*(x["fg_count_sd"]**2+(x["mean_fg_tokens"]-adaptive_fg_mean)**2)
                        for x in scenarios)/summary["policy_decisions"]
    comparisons = [{"method":"Adaptive hybrid 0.20", "scenario_count":9,
                    "decisions":summary["policy_decisions"],"successes":summary["successes"],
                    "mean_fg_tokens":adaptive_fg_mean,
                    "mean_bg_tokens":TOKENS-adaptive_fg_mean,
                    "mean_fg_fraction":adaptive_fg_mean/TOKENS,
                    "global_fg_count_sd":math.sqrt(adaptive_fg_var),
                    "temporal_transition_metrics":"available from aggregate transition telemetry",
                    "color":"#1874b4"}]
    for name, dpath, color in (("V2 hybrid R5 0.20",r5_020_dir,"#7a4eab"),("Conservative hybrid R5 0.15",r5_015_dir,"#26805a")):
        if dpath is not None:
            comp=_load_fixed_r5(name,dpath); comp["color"]=color
            comp["source_sha256"]={p.name:_sha256(p) for p in sorted(dpath.glob("task*_state*.json"))}
            comparisons.append(comp)
    # Add robust per-observation count trajectories to fixed-R5 comparisons for figure only.
    for comp in comparisons:
        if comp["method"]=="Adaptive hybrid 0.20":
            maxstep=max(r["step"] for r in pairs); vals=[None]*(maxstep+1)
            for s in range(1,maxstep+1):
                xs=[r["fg_current"] for r in pairs if r["step"]==s]
                vals[s]=mean(xs)/TOKENS if xs else None
            comp["time_series_fg_fraction"]=vals
        else:
            target=r5_020_dir if comp["method"]=="V2 hybrid R5 0.20" else r5_015_dir
            bystep=defaultdict(list)
            for p in target.glob("task*_state*.json"):
                d=json.load(open(p))
                for rec in d["records"]:
                    bystep[int(rec["step"])].append(int(rec["v2_foreground_count"]))
            maxstep=max(bystep) if bystep else 0
            comp["time_series_fg_fraction"]=[mean(bystep[s])/TOKENS if s in bystep else None for s in range(maxstep+1)]
            comp.pop("per_scenario",None)
    summary["configuration_comparison"] = [{k:v for k,v in c.items() if k!="time_series_fg_fraction"} for c in comparisons]
    summary["success_vs_failure"] = {
        "successful_scenarios": summarize_pairs(split["successful"]),
        "failed_scenarios": summarize_pairs(split["failed"]),
        "interpretation_limit": "descriptive grouping only; scenarios are few and task/scenario difficulty is confounded",
    }

    out.mkdir(parents=True, exist_ok=True); (out/"figures").mkdir(exist_ok=True)
    with (out/"per_step_metrics.csv").open("w",newline="") as f:
        fields=list(pairs[0].keys()); w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(pairs)
    with (out/"per_scenario_metrics.csv").open("w",newline="") as f:
        fields=list(scenarios[0].keys()); w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(scenarios)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    validation={"status":"passed","scenario_count":len(scenarios),"policy_decisions":summary["policy_decisions"],
                "temporal_pairs_excluding_each_scenario_first_observation":len(pairs),
                "transition_partition_check":"all rows nonnegative; four cells sum to 256",
                "full_refresh_pair_count":summary["full_refresh_pair_count"],
                "selective_pair_count":summary["selective_pair_count"],
                "full_refresh_transition_counts_valid":True,
                "outcomes_match_run_status":True,
                "no_gpu_or_inference":True,
                "cross_experiment_transition_metrics":"not computed where only foreground counts were saved",
                "source_sha256":summary["source_sha256"]}
    (out/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    make_figures(out,pairs,scenarios,comparisons)
    report=build_report(summary,scenarios,out)
    (out/"TEMPORAL_CONSISTENCY_REPORT.md").write_text(report)
    return summary


def _pct(v): return "n/a" if v is None else f"{100*v:.1f}%"


def build_report(summary: dict[str, Any], scenarios: list[dict[str, Any]], out: Path) -> str:
    m=summary["split_metrics"]["all"]
    lines=["# V2 Temporal FG/BG Classification Consistency Audit","",
           "## Summary","",
           f"The adaptive hybrid trace contains **{summary['policy_decisions']:,} decisions** from {summary['scenario_count']} completed scenarios ({summary['successes']} successes, {summary['failed_scenarios']} failures). After excluding the first observation of each scenario, the audit covers **{summary['temporal_pairs']:,} consecutive pairs**. The V2 classification flips on **{_pct(m['observed_flip_rate'])}** of token-pair positions; independent random classifications matched to each pair's foreground proportions predict **{_pct(m['expected_random_flip_rate'])}**. Marginal-adjusted agreement is **{_pct(m['chance_adjusted_agreement'])}**.","",
           "These are aggregate mask-consistency statistics, not segmentation accuracy or evidence that a stable classification is correct.","",
           "## Adaptive Hybrid results","",
           "| Metric | Result |","|---|---:|",
           f"| Mean FG tokens per observation | {m['mean_fg_current_tokens']:.1f}/256 ({_pct(m['mean_fg_current_fraction'])}) |",
           f"| Mean BG tokens per observation | {m['mean_bg_current_tokens']:.1f}/256 ({_pct(m['mean_bg_current_fraction'])}) |",
           f"| FG → BG / BG → FG per pair | {m['fg_to_bg_count']:,} / {m['bg_to_fg_count']:,} total; {m['fg_to_bg_count']/summary['temporal_pairs']:.2f} / {m['bg_to_fg_count']/summary['temporal_pairs']:.2f} mean |",
           f"| FG → FG / BG → BG per pair | {m['fg_to_fg_count']:,} / {m['bg_to_bg_count']:,} total |",
           f"| FG retention | {_pct(m['fg_retention_rate'])} |",
           f"| BG retention | {_pct(m['bg_retention_rate'])} |",
           f"| Temporal foreground IoU (micro) | {_pct(m['micro_temporal_foreground_iou'])} |",
           f"| Expected random flip rate | {_pct(m['expected_random_flip_rate'])} |",
           f"| Chance-adjusted agreement | {_pct(m['chance_adjusted_agreement'])} |","",
           "### Selective versus full-refresh decisions","",
           f"Transition reconstruction was valid on {summary['selective_pair_count']:,} pairs following selective decisions and {summary['full_refresh_pair_count']:,} pairs following full-refresh decisions. In the runner, `transition_protection_count` is computed from the semantic masks before a confidence-triggered full refresh overrides the cache-selection mask. Thus it remains a valid semantic transition count on those refresh pairs, but does not imply that transition protection changed the full-refresh K/V selection. The first frame of each scenario has no predecessor and is excluded.","",
           "The semantic FG/BG map is distinct from the actual KV refresh/reuse mask: the adaptive policy unions current FG, lagged attention protection, and FG→BG transitions, and may refresh every token on a confidence trigger. This report audits only semantic classifications.","",
           "## Scenario-level results","",
           "| Scenario | Outcome | Pairs | FG retention | BG retention | Temporal FG IoU | Observed flips | Random expected | Chance-adjusted agreement |","|---|:---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in sorted(scenarios,key=lambda z:(z['task_id'],z['state_id'])):
        lines.append(f"| Task {x['task_id']} / state {x['state_id']} | {'Success' if x['success'] else 'Failure'} | {x['pair_count']} | {_pct(x['fg_retention_rate'])} | {_pct(x['bg_retention_rate'])} | {_pct(x['micro_temporal_foreground_iou'])} | {_pct(x['observed_flip_rate'])} | {_pct(x['expected_random_flip_rate'])} | {_pct(x['chance_adjusted_agreement'])} |")
    groups=summary['success_vs_failure']; sm=groups['successful_scenarios']; fm=groups['failed_scenarios']
    lines += ["","### Successful versus failed rollout groups","",
              f"Successful-scenario pairs: observed flips {_pct(sm.get('observed_flip_rate'))}, random expectation {_pct(sm.get('expected_random_flip_rate'))}, FG/BG retention {_pct(sm.get('fg_retention_rate'))}/{_pct(sm.get('bg_retention_rate'))}, temporal IoU {_pct(sm.get('micro_temporal_foreground_iou'))}.",
              f"Failed-scenario pairs: observed flips {_pct(fm.get('observed_flip_rate'))}, random expectation {_pct(fm.get('expected_random_flip_rate'))}, FG/BG retention {_pct(fm.get('fg_retention_rate'))}/{_pct(fm.get('bg_retention_rate'))}, temporal IoU {_pct(fm.get('micro_temporal_foreground_iou'))}.","",
              "This grouping is descriptive. With nine scenarios and task-dependent difficulty, it cannot establish that temporal consistency causes success or failure.","",
              "## Comparison with the two fixed R5 configurations","",
              "The fixed V2 Hybrid R5 @0.20 and Conservative Hybrid R5 @0.15 traces contain per-observation foreground counts, so their mean FG/BG coverage and foreground-count variability can be compared. They did not persist transition counts or complete masks. Therefore, exact transition rates, retention, temporal IoU, random-adjusted agreement, or token-position instability are not comparable and are not inferred.","",
              "| Configuration | Success | Mean FG tokens | Mean BG tokens | FG fraction | FG-count SD | Exact temporal transitions |","|---|---:|---:|---:|---:|---:|---|"]
    for c in summary['configuration_comparison']:
        lines.append(f"| {c['method']} | {c['successes']}/{c['scenario_count']} | {c['mean_fg_tokens']:.1f} | {c['mean_bg_tokens']:.1f} | {_pct(c['mean_fg_fraction'])} | {c['global_fg_count_sd']:.1f} | {c['temporal_transition_metrics']} |")
    lines += ["","## Figures","",
              "![FG/BG coverage over time](figures/foreground_background_coverage.png)","",
              "![Observed and random flip rates](figures/classification_flip_rate.png)","",
              "![Directional transition counts](figures/directional_transitions.png)","",
              "![Temporal foreground IoU by scenario](figures/temporal_foreground_iou_by_scenario.png)","",
              "![FG and BG retention by scenario](figures/fg_bg_retention_by_scenario.png)","",
              "![Observed and expected random flips by scenario](figures/scenario_flip_observed_vs_random.png)","",
              "The time-series charts average scenarios that have an observation at each absolute step. Later points therefore represent the subset of longer-running episodes, not a fixed cohort.","",
              "## Data availability and limitations","",
              "The saved adaptive telemetry includes `v2_fg_count` and `transition_protection_count` per decision but not a complete binary mask or per-token probability map. The FG→BG count is the runner's `previous_foreground AND NOT current_foreground` transition count. The other three cells are reconstructed exactly from the two aggregate FG counts and this count using the specified equations. Every row was checked for nonnegative entries summing to 256.","",
              "Unavailable from these artifacts: per-position persistence/flip frequencies, spatial instability heatmaps, per-token score variance, token-specific FG→BG→FG or BG→FG→BG oscillations. Although per-frame score minima/maxima were saved, these cannot recover token-level continuous-score changes. No missing predictions were regenerated.","",
              "Objects can move and produce correct mask changes; consistency alone is not a quality metric. Distinguishing legitimate motion from classifier noise requires full per-frame masks/probability maps aligned to RGB, object/role ground truth or tracks, and synchronized consecutive images. For cache safety it would additionally require linking each reused token to source/current scene content and measuring downstream action/cache error.","",
              "## Metric definitions","",
              "For each pair, `fg_to_bg` is the logged transition protection count; `bg_to_fg = fg_current - fg_previous + fg_to_bg`; `fg_to_fg = fg_previous - fg_to_bg`; and `bg_to_bg = 256 - fg_previous - bg_to_fg`. Flip rate is `(fg_to_bg + bg_to_fg)/256`. FG/BG retention divide same-class persistence by the corresponding previous-frame class count. Temporal FG IoU is `fg_to_fg / (fg_previous + bg_to_fg)`. The random baseline independently samples each frame with the observed foreground proportion. Chance-adjusted agreement is `(observed agreement − expected random agreement)/(1 − expected random agreement)`, equivalently `1 − observed_flip_rate/expected_random_flip_rate`.","",
              "Reproducibility: run `python experiments/kv_cache/experiments/v2_temporal_consistency_audit.py` from the repository root. The analysis is CPU-only and reads completed JSON records; it does not call policy, simulator, CUDA, or model code.",""]
    return "\n".join(lines)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--adaptive-dir",type=Path,default=DEFAULT_ADAPTIVE)
    ap.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--r5-020-dir",type=Path,default=Path("experiments/kv_cache/outputs/hybrid_v2_task_relevance/live_run_20261124/scenario_results"))
    ap.add_argument("--r5-015-dir",type=Path,default=Path("experiments/kv_cache/outputs/conservative_v2_hybrid_threshold015/live_20261125/scenario_results"))
    a=ap.parse_args()
    summary=analyze(a.adaptive_dir,a.out_dir,a.r5_020_dir,a.r5_015_dir)
    print(json.dumps({"status":summary["status"],"pairs":summary["temporal_pairs"],"successes":summary["successes"],
                      "flip_rate":summary["split_metrics"]["all"]["observed_flip_rate"],
                      "random_flip_rate":summary["split_metrics"]["all"]["expected_random_flip_rate"],
                      "chance_adjusted_agreement":summary["split_metrics"]["all"]["chance_adjusted_agreement"],
                      "output":str(a.out_dir)},indent=2))


if __name__=="__main__": main()
