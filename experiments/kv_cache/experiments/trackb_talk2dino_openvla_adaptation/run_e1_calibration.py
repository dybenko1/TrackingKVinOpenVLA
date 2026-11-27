#!/usr/bin/env python3
"""Fine-grained E1 operating-point calibration from the frozen 90-frame cache.

Threshold selection is development-only. The frozen evaluation split is scored
only after thresholds have been written to the calibration artifact.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[4]
OUT = WORKTREE / "experiments/kv_cache/outputs/trackb_talk2dino_openvla_adaptation"
RUNNER = WORKTREE / "experiments/kv_cache/experiments/trackb_talk2dino_openvla_multicategory/run_experiment.py"
CONDITIONS = {
    "A_original_meta_final24": ("a_raw", "a_clean"),
    "B_openvla_final24": ("b_raw", "b_clean"),
    "C_openvla_native23": ("c_raw", "c_clean"),
}
TARGETS = (0.95, 0.98, 0.99)


def counts(score: np.ndarray, y: np.ndarray, valid: np.ndarray, tau: float) -> dict[str, float]:
    pred = np.asarray(score) >= tau
    y = np.asarray(y, dtype=np.float64)
    v = np.asarray(valid, dtype=np.float64)
    if pred.shape != y.shape or y.shape != v.shape:
        raise ValueError(f"score/target/valid shape mismatch: {pred.shape}/{y.shape}/{v.shape}")
    tp = float((pred * y * v).sum())
    fp = float((pred * (1 - y) * v).sum())
    fn = float(((~pred) * y * v).sum())
    tn = float(((~pred) * (1 - y) * v).sum())
    div = lambda a, b: a / b if b else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fg_recall": div(tp, tp + fn),
        "fg_precision": div(tp, tp + fp),
        "bg_precision": div(tn, tn + fn),
        "bg_recall": div(tn, tn + fp),
        "bg_iou": div(tn, tn + fp + fn),
        "fg_iou": div(tp, tp + fp + fn),
        "fg_coverage": div(tp + fp, tp + fp + fn + tn),
        "bg_coverage": div(tn + fn, tp + fp + fn + tn),
        "fg_as_bg": div(fn, tp + fn),
    }


def collect():
    sys.path.insert(0, str(RUNNER.parent))
    import run_experiment as baseline

    segments, manifest, manifest_sha = baseline.input_segments()
    data = []
    for seg in segments:
        arr = seg["arrays"]
        p = baseline.OUT / "score_maps" / seg["path"].name
        with np.load(p, allow_pickle=False) as z:
            m = json.loads(str(z["metadata"].item()))
            scores = {k: np.asarray(z[k], dtype=np.float32) for k in ("a_raw", "a_clean", "b_raw", "b_clean", "c_raw", "c_clean")}
        if len(arr["rgb"]) != 10:
            raise RuntimeError(f"Expected ten paired frames in {p}")
        for i in range(10):
            row = {"split": seg["split"], "suite": seg["suite"], "task_id": seg["task_id"],
                   "demo": seg["demo"], "frame": int(arr["selected_indices"][i]),
                   "source_archive_sha256": seg["source_sha256"],
                   "relevant_gt": np.asarray(arr["relevant_gt"][i], dtype=np.float32),
                   "valid": np.asarray(arr["valid"][i], dtype=np.float32),
                   "role_gt": np.asarray(arr["role_gt"][i], dtype=np.float32),
                   "role_keys": seg["role_keys"]}
            for key, value in scores.items():
                if value.shape != (10, len(seg["queries"]), 256):
                    raise RuntimeError(f"Unexpected {key} shape in {p}: {value.shape}")
                row[key] = value[i].max(axis=0)
            data.append(row)
    if len(data) != 90:
        raise RuntimeError(f"Expected 90 frames, found {len(data)}")
    return data, manifest_sha


def pooled(rows, score_key, tau):
    acc = {k: 0.0 for k in ("tp", "fp", "fn", "tn")}
    for r in rows:
        c = counts(r[score_key], r["relevant_gt"], r["valid"], tau)
        for k in acc:
            acc[k] += c[k]
    tp, fp, fn, tn = (acc[k] for k in ("tp", "fp", "fn", "tn"))
    div = lambda a, b: a / b if b else 0.0
    return {**acc, "fg_recall": div(tp, tp + fn), "fg_precision": div(tp, tp + fp),
            "bg_precision": div(tn, tn + fn), "bg_recall": div(tn, tn + fp),
            "bg_iou": div(tn, tn + fp + fn), "fg_iou": div(tp, tp + fp + fn),
            "fg_coverage": div(tp + fp, tp + fp + fn + tn),
            "bg_coverage": div(tn + fn, tp + fp + fn + tn),
            "fg_as_bg": div(fn, tp + fn)}


def exact_operating_point(rows, score_key, target):
    """Choose the observed score value that reaches target recall with max coverage.

    Descending score order is monotone in protected foreground recall and
    candidate background coverage, so the first score tie-group reaching the
    recall constraint is the exact finite-score optimum.
    """
    score = np.concatenate([np.asarray(r[score_key], np.float64).ravel() for r in rows])
    fg = np.concatenate([(np.asarray(r["relevant_gt"], np.float64) * np.asarray(r["valid"], np.float64)).ravel() for r in rows])
    valid = np.concatenate([np.asarray(r["valid"], np.float64).ravel() for r in rows])
    keep = valid > 0
    score, fg = score[keep], fg[keep]
    total = float(fg.sum())
    order = np.argsort(score, kind="stable")[::-1]
    cumulative_fg = np.cumsum(fg[order])
    i = int(np.searchsorted(cumulative_fg, target * total, side="left"))
    tau = float(score[order[min(i, len(order)-1)]])
    metrics = pooled(rows, score_key, tau)
    return {"threshold": tau, "dev": metrics, "constraint_met": metrics["fg_recall"] + 1e-12 >= target}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    rows, manifest_sha = collect()
    dev = [r for r in rows if r["split"] == "development"]
    ev = [r for r in rows if r["split"] == "evaluation"]
    if (len(dev), len(ev)) != (30, 60):
        raise RuntimeError(f"Frozen benchmark split mismatch: dev/eval={len(dev)}/{len(ev)}")
    taus = np.round(np.arange(0.45, 0.6001, 0.001), 3)
    calibration, sweep = {}, []
    for cond, (raw_key, clean_key) in CONDITIONS.items():
        for post, key in (("raw", raw_key), ("clean", clean_key)):
            # Every grid point is logged on development. Exact unique-score
            # thresholds are added as diagnostics, not used to tune evaluation.
            for tau in taus:
                m = pooled(dev, key, float(tau))
                sweep.append({"condition": cond, "postprocess": post, "split": "development",
                              "threshold": float(tau), **m})
            for target in TARGETS:
                feasible = []
                for tau in taus:
                    m = pooled(dev, key, float(tau))
                    if m["fg_recall"] + 1e-12 >= target:
                        feasible.append((m["bg_coverage"], float(tau), m))
                # If the primary interval has no feasible point, report that;
                # do not silently extend / tune outside the predeclared range.
                if feasible:
                    _, tau, dm = max(feasible, key=lambda z: (z[0], z[1]))
                    selection = {"threshold": tau, "dev": dm, "constraint_met": True}
                    em = pooled(ev, key, tau)
                else:
                    tau = float(taus[0]); dm = pooled(dev, key, tau)
                    selection = {"threshold": tau, "dev": dm, "constraint_met": False}
                    em = pooled(ev, key, tau)
                selection["evaluation"] = em
                exact = exact_operating_point(dev, key, target)
                exact["evaluation"] = pooled(ev, key, exact["threshold"])
                calibration.setdefault(cond, {}).setdefault(post, {})[f"exact_recall_{target:.2f}"] = exact
                calibration.setdefault(cond, {}).setdefault(post, {})[f"recall_{target:.2f}"] = selection
            # BG-IoU-maximizing dev point is reported as a secondary diagnostic.
            bg_best = max(((pooled(dev, key, float(t)), float(t)) for t in taus), key=lambda x: (x[0]["bg_iou"], x[1]))
            calibration.setdefault(cond, {}).setdefault(post, {})["dev_max_bg_iou"] = {
                "threshold": bg_best[1], "dev": bg_best[0], "evaluation": pooled(ev, key, bg_best[1])}
    with (OUT / "E1_threshold_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)
    (OUT / "E1_calibration.json").write_text(json.dumps({
        "status": "complete", "method": "grid thresholds 0.450 through 0.600 step 0.001; max BG coverage under dev FG recall constraints",
        "thresholds_selected_on": "30 development frames only", "evaluation_frames": 60,
        "selection_rule": "maximize pooled fractional background coverage subject to pooled fractional FG recall >= target",
        "exact_score_values": "supplemental operating points chosen from sorted unique observed development scores; primary reported calibration remains the fixed 0.001 grid",
        "source_manifest_sha256": manifest_sha, "calibration": calibration,
        "limitations": ["90-image benchmark was previously inspected", "results are diagnostics, not independent blind validation"]
    }, indent=2) + "\n")
    plot(rows)
    plot_distributions(rows)
    print(json.dumps({k: {p: {t: v["evaluation"]["fg_recall"] for t, v in vals.items() if t.startswith("recall_")} for p, vals in posts.items()} for k, posts in calibration.items()}, indent=2))


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = [r for r in rows if r["split"] == "development"]
    ev = [r for r in rows if r["split"] == "evaluation"]
    taus = np.round(np.arange(0.45, 0.6001, 0.001), 3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"A_original_meta_final24": "#2369bd", "B_openvla_final24": "#d33f49", "C_openvla_native23": "#23855b"}
    for cond, (rk, ck) in CONDITIONS.items():
        for post, key, ls in (("RAW", rk, "-"), ("CLEAN", ck, "--")):
            ms = [pooled(dev, key, float(t)) for t in taus]
            axes[0].plot([x["fg_recall"] for x in ms], [x["bg_coverage"] for x in ms], ls,
                         color=colors[cond], label=f"{cond[0]} {post} dev")
            me = [pooled(ev, key, float(t)) for t in taus]
            axes[1].plot([x["bg_coverage"] for x in me], [x["bg_precision"] for x in me], ls,
                         color=colors[cond], label=f"{cond[0]} {post} eval")
    axes[0].set(xlabel="FG recall", ylabel="BG coverage", title="Dev calibration curve")
    axes[1].set(xlabel="BG coverage", ylabel="BG precision", title="Frozen evaluation curve")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figures/E1_threshold_curves.png", dpi=160); plt.close(fig)


def plot_distributions(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = [r for r in rows if r["split"] == "development"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for row_i, (cond, keys) in enumerate(CONDITIONS.items()):
        for col, (post, key) in enumerate((("RAW", keys[0]), ("CLEAN", keys[1]))):
            ax = axes[row_i, col]
            fg = np.concatenate([r[key][(r["valid"] > 0) & (r["relevant_gt"] >= .98)] for r in dev])
            bg = np.concatenate([r[key][(r["valid"] > 0) & (r["relevant_gt"] <= .02)] for r in dev])
            ax.hist(bg, bins=40, density=True, alpha=.58, label="oracle background patches (occupancy ≤ .02)", color="#496ca8")
            ax.hist(fg, bins=40, density=True, alpha=.58, label="oracle foreground patches (occupancy ≥ .98)", color="#d95555")
            ax.set_title(f"{cond[0]} {post}"); ax.set_ylabel("Density"); ax.grid(alpha=.2)
            if row_i == 0 and col == 0: ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Foreground score"); axes[-1, 1].set_xlabel("Foreground score")
    fig.suptitle("E1 development score distributions (fractional FG/BG occupancy)")
    fig.tight_layout(); fig.savefig(OUT / "figures/E1_score_distributions.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    run()
