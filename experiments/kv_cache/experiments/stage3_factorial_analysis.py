"""Aggregate the Stage 3A-D oracle-masked KV factorial experiment."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage3_factorial_comparison"
PATHS = {
    "stage3a_k_l0_13": ROOT / "outputs/stage3a_oracle_masked_matrix",
    "stage3b_kv_l0_31": ROOT / "outputs/stage3b_oracle_full_kv_matrix",
    "stage3c_k_l0_31": ROOT / "outputs/stage3c_oracle_k_all_layers",
    "stage3d_kv_l0_13": ROOT / "outputs/stage3d_oracle_kv_first14",
}
LABELS = {
    "stage3a_k_l0_13": "3A K layers 0–13",
    "stage3b_kv_l0_31": "3B K+V layers 0–31",
    "stage3c_k_l0_31": "3C K layers 0–31",
    "stage3d_kv_l0_13": "3D K+V layers 0–13",
}
TASK_STATES = [(t, s) for t in (4, 5, 7) for s in (0, 1, 2)]


def _scenario_path(stage: str, task: int, state: int) -> Path:
    return PATHS[stage] / f"task{task}_state{state}" / "results.json"


def _load_scenario(stage: str, task: int, state: int):
    path = _scenario_path(stage, task, state)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if stage == "stage3a_k_l0_13":
        intervention = data["masked_true_skip"]
        fresh = data["fresh"]
    elif stage == "stage3b_kv_l0_31":
        intervention = data["stage3b_true_skip"]
        fresh = data["fresh"]
    else:
        intervention = data["true_skip"]
        ref = data["fresh_reference"]
        fresh = {k: ref[k] for k in ("success", "steps", "termination", "seed", "task_id", "state_id", "all_valid")}
    return data, fresh, intervention


def _skip_accounting(stage: str, intervention: dict[str, Any]) -> dict[str, Any]:
    records = intervention.get("records", [])
    if "total_skipped_k_projection_rows" in intervention:
        skip_k = int(intervention["total_skipped_k_projection_rows"])
        skip_v = int(intervention.get("total_skipped_v_projection_rows", 0))
        per_layer_k = intervention.get("skipped_k_rows_by_layer_aggregate")
        per_layer_v = intervention.get("skipped_v_rows_by_layer_aggregate")
        denom = int(intervention.get("normal_k_rows_all_prefills", 0))
    else:
        skip_k = skip_v = 0
        per_layer_k = [0] * 32
        per_layer_v = [0] * 32
        for row in records:
            if row.get("type") != "reuse":
                continue
            bg = int(row["background_tokens"])
            if stage == "stage3a_k_l0_13":
                count_k = int(row.get("reused_k_rows_per_selected_layer", bg)) * 14
                for layer in range(14): per_layer_k[layer] += count_k // 14
            else:
                for layer, value in enumerate(row.get("skipped_k_rows_by_layer", [])):
                    per_layer_k[layer] += int(value)
                for layer, value in enumerate(row.get("skipped_v_rows_by_layer", [])):
                    per_layer_v[layer] += int(value)
        skip_k, skip_v = sum(per_layer_k), sum(per_layer_v)
        denom = sum(int(row["prefill_length"]) * 32 for row in records)
    if denom == 0:
        # Stage 3A log format has full layouts in per-step records; this remains
        # the same denominator as all 32 layers' normal K projections.
        denom = sum(int(row["prefill_length"]) * 32 for row in records)
    ages = intervention.get("source_age_counts", {})
    reused = [r for r in records if r.get("type") == "reuse"]
    return {
        "normal_k_projection_rows_denominator": denom,
        "normal_v_projection_rows_denominator": denom,
        "skipped_k_rows": skip_k, "skipped_v_rows": skip_v,
        "k_rows_skipped_fraction": skip_k / denom if denom else None,
        "v_rows_skipped_fraction": skip_v / denom if denom else None,
        "combined_kv_rows_skipped_fraction": (skip_k + skip_v) / (2 * denom) if denom else None,
        "qkv_projection_rows_skipped_fraction": (skip_k + skip_v) / (3 * denom) if denom else None,
        "skipped_k_rows_by_layer": per_layer_k, "skipped_v_rows_by_layer": per_layer_v,
        "temporal_reuse_count": int(intervention.get("reuse_count", len(reused))),
        "refresh_count": int(intervention.get("refresh_count", len(records) - len(reused))),
        "temporal_reuse_fraction": float(intervention.get("reuse_fraction", len(reused) / len(records) if records else 0.0)),
        "source_age_counts": ages,
        "max_source_age": int(intervention.get("max_source_age", max((r.get("source_age", 0) for r in records), default=0))),
        "mean_background_tokens_on_reuse": float(np.mean([r["background_tokens"] for r in reused])) if reused else 0.0,
        "mean_foreground_tokens_on_reuse": float(np.mean([r["foreground_tokens"] for r in reused])) if reused else 0.0,
        "layers_directly_intervened": sorted({i for i, v in enumerate(per_layer_k) if v > 0 or per_layer_v[i] > 0}),
        "all_valid": bool(intervention.get("all_valid", True)),
        "latency_measured": False,
    }


def _write_csv(rows: list[dict[str, Any]]):
    path = OUT / "comparison.csv"
    columns = ["task_id", "state_id", "condition", "fresh_success", "fresh_steps", "reuse_success", "reuse_steps",
               "paired_success_regression", "paired_success_recovery", "refresh_count", "reuse_count", "reuse_fraction",
               "max_source_age", "background_tokens_mean", "foreground_tokens_mean", "skipped_k_rows",
               "skipped_v_rows", "normal_k_rows_denominator", "normal_v_rows_denominator",
               "k_skip_fraction", "v_skip_fraction", "combined_kv_skip_fraction", "qkv_skip_fraction", "validation_passed"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def _plots(rows: list[dict[str, Any]], summary: dict[str, Any]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    conditions = list(PATHS)
    labels = [LABELS[x] for x in conditions]
    scenarios = [f"T{r['task_id']} S{r['state_id']}" for r in rows if r["condition"] == conditions[0]]
    matrices = []
    for condition in conditions:
        indexed = {(r["task_id"], r["state_id"]): r for r in rows if r["condition"] == condition}
        matrices.append([1 if indexed.get(pair, {}).get("reuse_success") is True else 0 if pair in indexed else np.nan for pair in TASK_STATES])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(np.asarray(matrices).T, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(conditions)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(TASK_STATES)), [f"Task {t} / State {s}" for t, s in TASK_STATES])
    ax.set_title("Scenario-level success/failure (green=success, red=failure)")
    for y, pair in enumerate(TASK_STATES):
        for x, condition in enumerate(conditions):
            r = next((z for z in rows if z["condition"] == condition and (z["task_id"], z["state_id"]) == pair), None)
            if r: ax.text(x, y, f"{'✓' if r['reuse_success'] else '×'}\n{r['reuse_steps']}", ha="center", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "scenario_outcomes.png", dpi=180); plt.close(fig)

    rates = [(summary["conditions"][condition]["successes"] / summary["conditions"][condition]["total"]
              if summary["conditions"][condition]["total"] else 0.0) for condition in conditions]
    fig, ax = plt.subplots(figsize=(8, 4.7)); bars = ax.bar(labels, rates, color=["#4c78a8", "#e45756", "#72b7b2", "#f2cf5b"])
    ax.set_ylim(0, 1); ax.set_ylabel("Task success fraction"); ax.set_title("Paired task success by masked-reuse condition")
    ax.tick_params(axis="x", labelrotation=20)
    for bar, rate, condition in zip(bars, rates, conditions):
        count = summary["conditions"][condition]["total"]
        ax.text(bar.get_x()+bar.get_width()/2, rate+.025,
                f"{summary['conditions'][condition]['successes']}/{count}" if count else "N/A", ha="center")
    fig.tight_layout(); fig.savefig(OUT / "success_rates.png", dpi=180); plt.close(fig)

    skip_k = [summary["conditions"][c]["total_skipped_k_rows"] for c in conditions]
    skip_v = [summary["conditions"][c]["total_skipped_v_rows"] for c in conditions]
    x = np.arange(len(conditions)); width=.36
    fig, ax = plt.subplots(figsize=(9, 4.8)); ax.bar(x-width/2, skip_k, width, label="K rows omitted")
    ax.bar(x+width/2, skip_v, width, label="V rows omitted")
    ax.set_xticks(x, labels, rotation=20, ha="right"); ax.set_ylabel("Projection rows structurally skipped")
    ax.set_title("Structural projection-row omission (not latency)"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "projection_rows_omitted.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for condition in conditions:
        xs, ys = [], []
        for r in rows:
            if r["condition"] == condition and r["combined_kv_skip_fraction"] is not None:
                xs.append(100*r["combined_kv_skip_fraction"]); ys.append(1 if r["reuse_success"] else 0)
        ax.scatter(xs, ys, label=LABELS[condition], alpha=.8)
    ax.set_yticks([0,1],["Failure","Success"]); ax.set_xlabel("Combined K/V projection rows skipped (%)")
    ax.set_title("Structural skipping versus observed scenario outcome"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "compute_skip_vs_success.png", dpi=180); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stageb_summary = json.loads((PATHS["stage3b_kv_l0_31"] / "summary.json").read_text())
    rows = []
    fresh_lookup = {}
    for task, state in TASK_STATES:
        sb = json.loads(_scenario_path("stage3b_kv_l0_31", task, state).read_text())
        fresh_lookup[(task, state)] = sb["fresh"]
    for condition in PATHS:
        for task, state in TASK_STATES:
            loaded = _load_scenario(condition, task, state)
            if loaded is None:
                rows.append({"task_id": task, "state_id": state, "condition": condition,
                             "fresh_success": fresh_lookup[(task,state)]["success"],
                             "fresh_steps": fresh_lookup[(task,state)]["steps"],
                             "reuse_success": None,"reuse_steps":None,"validation_passed":False,
                             **{k:None for k in ("skipped_k_rows","skipped_v_rows","normal_k_rows_denominator","normal_v_rows_denominator","k_skip_fraction","v_skip_fraction","combined_kv_skip_fraction","qkv_skip_fraction")}})
                continue
            data, fresh, intervention = loaded
            if condition in ("stage3c_k_l0_31", "stage3d_kv_l0_13"):
                fresh = fresh_lookup[(task,state)]
            accounting = _skip_accounting(condition, intervention)
            row = {"task_id":task,"state_id":state,"condition":condition,
                   "fresh_success":bool(fresh["success"]),"fresh_steps":int(fresh["steps"]),
                   "reuse_success":bool(intervention["success"]),"reuse_steps":int(intervention["steps"]),
                   "paired_success_regression":bool(fresh["success"] and not intervention["success"]),
                   "paired_success_recovery":bool((not fresh["success"]) and intervention["success"]),
                   "refresh_count":int(intervention.get("refresh_count",0)),"reuse_count":accounting["temporal_reuse_count"],
                   "reuse_fraction":accounting["temporal_reuse_fraction"],"max_source_age":accounting["max_source_age"],
                   "background_tokens_mean":accounting["mean_background_tokens_on_reuse"],
                   "foreground_tokens_mean":accounting["mean_foreground_tokens_on_reuse"],
                   "skipped_k_rows":accounting["skipped_k_rows"],"skipped_v_rows":accounting["skipped_v_rows"],
                   "normal_k_rows_denominator":accounting["normal_k_projection_rows_denominator"],
                   "normal_v_rows_denominator":accounting["normal_v_projection_rows_denominator"],
                   "k_skip_fraction":accounting["k_rows_skipped_fraction"],"v_skip_fraction":accounting["v_rows_skipped_fraction"],
                   "combined_kv_skip_fraction":accounting["combined_kv_rows_skipped_fraction"],
                   "qkv_skip_fraction":accounting["qkv_projection_rows_skipped_fraction"],
                   "validation_passed":bool(intervention.get("all_valid",True)),"accounting":accounting}
            rows.append(row)

    conditions = {}
    for condition in PATHS:
        xs=[r for r in rows if r["condition"]==condition and r.get("reuse_success") is not None]
        fresh_ok=[r for r in xs if r["fresh_success"]]
        conditions[condition]={"successes":sum(r["reuse_success"] for r in xs),"total":len(xs),
           "fresh_successes":sum(r["fresh_success"] for r in xs),"successes_among_fresh_successes":sum(r["reuse_success"] for r in fresh_ok),
           "fresh_success_scenario_count":len(fresh_ok),"regressions":sum(r["paired_success_regression"] for r in xs),
           "recoveries":sum(r["paired_success_recovery"] for r in xs),
           "total_skipped_k_rows":sum(r["skipped_k_rows"] for r in xs),"total_skipped_v_rows":sum(r["skipped_v_rows"] for r in xs),
           "mean_temporal_reuse_fraction":float(np.mean([r["reuse_fraction"] for r in xs])) if xs else None,
           "mean_spatial_background_fraction":float(np.mean([r["accounting"]["mean_background_tokens_on_reuse"]/256 for r in xs])) if xs else None,
           "mean_k_skip_fraction":float(np.mean([r["k_skip_fraction"] for r in xs])) if xs else None,
           "mean_v_skip_fraction":float(np.mean([r["v_skip_fraction"] for r in xs])) if xs else None,
           "runtime_validations_passed":all(r["validation_passed"] for r in xs) if xs else False}

    rates={k:(v["successes"]/v["total"] if v["total"] else None) for k,v in conditions.items()}
    A,B,C,D=(rates.get(x) for x in ("stage3a_k_l0_13","stage3b_kv_l0_31","stage3c_k_l0_31","stage3d_kv_l0_13"))
    factorial={"v_reuse_effect_at_layers_0_13": (D-A) if D is not None and A is not None else None,
               "v_reuse_effect_at_layers_0_31": (B-C) if B is not None and C is not None else None,
               "average_descriptive_v_reuse_main_effect": ((D-A+B-C)/2) if None not in (A,B,C,D) else None,
               "layer_coverage_effect_k_only": (C-A) if C is not None and A is not None else None,
               "layer_coverage_effect_kv": (B-D) if B is not None and D is not None else None,
               "average_descriptive_layer_coverage_main_effect": ((C-A+B-D)/2) if None not in (A,B,C,D) else None,
               "difference_in_differences_interaction": ((B-C)-(D-A)) if None not in (A,B,C,D) else None,
               "interpretation":"Descriptive success-rate differences in a fixed nine-scenario, non-randomized matrix; no significance or causal generalization claimed."}
    matrix=[]
    for task,state in TASK_STATES:
        matrix.append({"task_id":task,"state_id":state,
                       "fresh_success":fresh_lookup[(task,state)]["success"],
                       "fresh_steps":fresh_lookup[(task,state)]["steps"],
                       **{c:next((r["reuse_success"] for r in rows if r["condition"]==c and r["task_id"]==task and r["state_id"]==state),None) for c in PATHS}})
    complete=all(conditions[c]["total"]==9 and conditions[c]["runtime_validations_passed"] for c in PATHS)
    summary={"status":"complete" if complete else "incomplete","configuration":{"tasks":[4,5,7],"states":[0,1,2],"seed":7,"R":5,"horizon":220,"visual_positions":[1,256],"fresh_reference_commit":"c9b05be688ab0ad5d474665300b8504f4c077ebb","latency_measured":False},
             "conditions":conditions,"scenario_matrix":matrix,"factorial_effects":factorial,"scenarios":rows}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    _write_csv(rows)
    try:
        _plots(rows,summary)
    except Exception as exc:
        summary["plot_error"]=f"{type(exc).__name__}: {exc}"
        (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# Stage 3A–3D oracle-masked KV factorial comparison", "",
           f"Status: **{summary['status'].upper()}**. No timing was performed; row savings are structural projection counts, not measured latency.","",
           "| Condition | Reused components | Layers | Success | Among Fresh-success | Regressions | Recoveries | K rows skipped | V rows skipped | Mean temporal reuse |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in PATHS:
        x=conditions[c]; layers=("0–13" if c in ("stage3a_k_l0_13","stage3d_kv_l0_13") else "0–31"); comp=("K+V" if "kv" in c else "K")
        mean_reuse = "N/A" if x["mean_temporal_reuse_fraction"] is None else f"{x['mean_temporal_reuse_fraction']:.1%}"
        lines.append(f"| {c} | {comp} | {layers} | {x['successes']}/{x['total']} | {x['successes_among_fresh_successes']}/{x['fresh_success_scenario_count']} | {x['regressions']} | {x['recoveries']} | {x['total_skipped_k_rows']:,} | {x['total_skipped_v_rows']:,} | {mean_reuse} |")
    lines += ["", "## Paired scenario outcomes", "", "| Task | State | Fresh | 3A K 0–13 | 3B K+V 0–31 | 3C K 0–31 | 3D K+V 0–13 |", "|---:|---:|---|---|---|---|---|"]
    for x in matrix:
        def fmt(v): return "—" if v is None else ("Success" if v else "Failure")
        lines.append(f"| {x['task_id']} | {x['state_id']} | {fmt(x['fresh_success'])} ({x['fresh_steps']}) | {fmt(x['stage3a_k_l0_13'])} | {fmt(x['stage3b_kv_l0_31'])} | {fmt(x['stage3c_k_l0_31'])} | {fmt(x['stage3d_kv_l0_13'])} |")
    lines += ["", "## Descriptive 2×2 effects", "",
              f"Observed average V-reuse main effect: `{factorial['average_descriptive_v_reuse_main_effect']}` success-rate points; layer-coverage main effect: `{factorial['average_descriptive_layer_coverage_main_effect']}`; interaction: `{factorial['difference_in_differences_interaction']}`.",
              "These are descriptive arithmetic contrasts on nine fixed scenarios, not statistical significance or causal generalization.","",
              "Projection-row fractions use the denominator stated per scenario in `comparison.csv`: total ordinary K rows (32 × sum of observed prefill lengths) and total ordinary V rows (same). Combined KV fraction divides skipped K+V rows by both full-projection denominators; QKV fraction divides by three times the common denominator. They are not model FLOPs or measured latency.","",
              "## Figures", "", "- `scenario_outcomes.png` — scenario-level success matrix.", "- `success_rates.png` — overall success fractions.", "- `projection_rows_omitted.png` — structural K/V row counts.", "- `compute_skip_vs_success.png` — scenario outcome against combined structural row fraction.","",
              "Completion-step counts are descriptive and are not interpreted as quality improvements or degradation.","",
              "## Source artifacts", "", "Inputs: Stage 3A/B/C/D output directories listed in `summary.json`; Stage 3A/B outputs were not modified by this analysis."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":summary["status"],"conditions":conditions,"factorial_effects":factorial},indent=2))


if __name__ == "__main__":
    main()
