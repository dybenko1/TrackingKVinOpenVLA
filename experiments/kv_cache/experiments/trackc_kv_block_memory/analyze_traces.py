#!/usr/bin/env python3
"""Analyze real Stage 3C R5 masks under token-page/COW storage models.

This is an offline accounting model. It does not alter or rerun Stage 3C.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[4]
TRACE_ROOT = REPO / "experiments" / "kv_cache" / "outputs" / "stage3c_oracle_task_relevance" / "protect_10"
OUT = REPO / "experiments" / "kv_cache" / "outputs" / "trackc_kv_block_memory"
PAGE_SIZES = (1, 4, 8, 16, 32)
LAYERS, COMPONENTS, HEADS, HEAD_DIM, DTYPE_BYTES = 32, 2, 32, 128, 2
SYNTHETIC_SEED = 41027


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name): os.unlink(temp_name)


def masks_for_record(record: dict, scenario: str, source_step: int):
    actual = sorted(set(int(i) for i in record["reusable_background_indices"]))
    n = len(actual)
    seed = SYNTHETIC_SEED + sum(ord(c) for c in scenario) + source_step * 101 + int(record["step"])
    rng = random.Random(seed)
    random_mask = sorted(rng.sample(range(256), n))
    # Cluster the reused set into one contiguous interval; disperse the same
    # count evenly over the 16x16 grid. Both preserve the observed reuse budget.
    start = max(0, (256 - n) // 2)
    clustered = list(range(start, start + n))
    if n == 256:
        dispersed = list(range(256))
    else:
        dispersed = sorted({min(255, int(round(k * 256 / n))) for k in range(n)})
        # Rounding can collide. Fill deterministically to preserve exactly n.
        if len(dispersed) < n:
            pool = [i for i in range(256) if i not in set(dispersed)]
            dispersed.extend(pool[:n-len(dispersed)])
            dispersed.sort()
    return {"real": actual, "synthetic_clustered": clustered,
            "synthetic_dispersed": dispersed, "synthetic_random": random_mask}


def page_metrics(seq_len: int, reused_visual: list[int], page_tokens: int) -> dict:
    reused_seq = {i + 1 for i in reused_visual}  # patch index 0 maps to sequence position 1.
    pages = []
    for p in range(math.ceil(seq_len / page_tokens)):
        lo, hi = p * page_tokens, min((p + 1) * page_tokens, seq_len)
        n_reused = sum(pos in reused_seq for pos in range(lo, hi))
        valid = hi - lo
        if n_reused == valid:
            kind = "share"
        elif n_reused == 0:
            kind = "fresh_only"
        else:
            kind = "mixed_cow"
        pages.append({"page": p, "lo": lo, "hi": hi, "valid_slots": valid,
                      "reused_slots": n_reused, "kind": kind})
    shared = [p for p in pages if p["kind"] == "share"]
    cow = [p for p in pages if p["kind"] == "mixed_cow"]
    fresh_only = [p for p in pages if p["kind"] == "fresh_only"]
    n_reused = len(reused_seq)
    fresh_rows = seq_len - n_reused
    slot_bytes = HEADS * HEAD_DIM * DTYPE_BYTES
    total_cache_bytes = seq_len * slot_bytes * LAYERS * COMPONENTS
    source_reads_cow = sum(p["valid_slots"] for p in cow) * slot_bytes * LAYERS * COMPONENTS
    physical_cow_writes = sum(page_tokens for _ in cow) * slot_bytes * LAYERS * COMPONENTS
    fresh_rw = 2 * fresh_rows * slot_bytes * LAYERS * COMPONENTS
    # Dense clone reads and writes the complete source; index_copy reads the
    # projected updates and writes them back into the clone.
    baseline_clone_rw = 2 * total_cache_bytes
    baseline_total_rw = baseline_clone_rw + fresh_rw
    cow_update_rw = source_reads_cow + physical_cow_writes + fresh_rw
    materialize_rw = 2 * total_cache_bytes
    num_pages = len(pages)
    allocated = len(cow) + len(fresh_only)
    # A fixed-capacity page pool retains B slots for every allocated page,
    # including the final partial page.
    source_live = num_pages * page_tokens * slot_bytes * LAYERS * COMPONENTS
    cow_target_live = allocated * page_tokens * slot_bytes * LAYERS * COMPONENTS
    mapping_bytes = num_pages * 8 * LAYERS * COMPONENTS
    refcount_bytes = (num_pages + allocated) * 4 * LAYERS * COMPONENTS
    return {
        "sequence_length": seq_len, "page_tokens": page_tokens,
        "logical_pages": num_pages, "updated_pages": allocated,
        "updated_page_indices": [p["page"] for p in pages if p["kind"] != "share"],
        "shared_unchanged_pages": len(shared), "mixed_copy_on_write_pages": len(cow),
        "fresh_only_pages": len(fresh_only),
        "update_page_touch_ratio": allocated / num_pages if num_pages else 0.0,
        "shared_page_ratio": len(shared) / num_pages if num_pages else 0.0,
        "visual_reused_tokens": n_reused, "fresh_sequence_rows": fresh_rows,
        "reused_source_slots_in_cow_pages": sum(p["reused_slots"] for p in cow),
        "valid_source_slots_read_by_cow": sum(p["valid_slots"] for p in cow),
        "physical_page_slots_copied_by_cow": len(cow) * page_tokens,
        "fixed_page_capacity_slots_allocated": allocated * page_tokens,
        "baseline_full_clone_payload_bytes": total_cache_bytes,
        "baseline_clone_read_write_bytes": baseline_clone_rw,
        "baseline_fresh_index_copy_read_write_bytes": fresh_rw,
        "baseline_total_update_traffic_bytes": baseline_total_rw,
        "cow_page_source_read_bytes": source_reads_cow,
        "cow_page_write_capacity_bytes": physical_cow_writes,
        "cow_fresh_update_read_write_bytes": fresh_rw,
        "cow_update_traffic_bytes": cow_update_rw,
        "cow_plus_dense_materialization_traffic_bytes": cow_update_rw + materialize_rw,
        "materialization_read_write_bytes": materialize_rw,
        "cow_write_amplification_vs_fresh_rows": (physical_cow_writes + fresh_rw / 2) / (fresh_rows * slot_bytes * LAYERS * COMPONENTS) if fresh_rows else 0.0,
        "page_table_metadata_bytes_per_cache": mapping_bytes,
        "refcount_metadata_bytes_source_plus_output": refcount_bytes,
        "refcount_increments_for_shares_per_update": len(shared) * LAYERS * COMPONENTS,
        "refcount_decrements_on_target_release": len(shared) * LAYERS * COMPONENTS,
        "estimated_live_unique_bytes_source_plus_cow_output": source_live + cow_target_live,
        "estimated_live_unique_bytes_with_dense_attention_cache": source_live + cow_target_live + total_cache_bytes,
    }


def collect_trace() -> list[dict]:
    records = []
    last_updated_pages = {}
    scenario_dirs = sorted(p for p in TRACE_ROOT.iterdir() if p.is_dir())
    if len(scenario_dirs) != 9:
        raise RuntimeError(f"Expected 9 Stage 3C protect-10 scenarios; found {len(scenario_dirs)} at {TRACE_ROOT}.")
    for sdir in tqdm(scenario_dirs, desc="[1/4] Stage 3C trace files", unit="scenario"):
        path = sdir / "results.json"
        if not path.exists(): raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        if data.get("status") != "complete" or not data.get("all_valid"):
            raise RuntimeError(f"Incomplete or invalid Stage 3C trace: {path}")
        source_by_step = {}
        for r in data["records"]:
            step = int(r["step"])
            age = int(r["source_age"])
            refresh = r["type"] == "refresh"
            expected_source = step - step % 5
            if refresh:
                if age != 0 or r.get("source_step") != step:
                    raise AssertionError(f"Invalid refresh provenance at {sdir.name} step {step}")
                source_by_step[step] = step
            else:
                if age != step % 5 or r.get("source_step") != expected_source:
                    raise AssertionError(f"Invalid R5 source age/source at {sdir.name} step {step}")
                if expected_source not in source_by_step:
                    raise AssertionError(f"Missing source refresh at {sdir.name} step {step}")
                n = len(r.get("reusable_background_indices", []))
                if n != int(r["background_tokens_reused"]):
                    raise AssertionError(f"Reuse-index count mismatch at {sdir.name} step {step}")
                masks = masks_for_record(r, sdir.name, expected_source)
                seq_len = int(r["prefill_length"])
                for pattern, reused in masks.items():
                    for block in PAGE_SIZES:
                        metric = page_metrics(seq_len, reused, block)
                        temporal_key = (sdir.name, expected_source, pattern, block)
                        current_pages = set(metric.pop("updated_page_indices"))
                        previous_pages = last_updated_pages.get(temporal_key)
                        metric["newly_affected_pages_vs_prior_age"] = (
                            len(current_pages) if previous_pages is None else len(current_pages - previous_pages)
                        )
                        metric["repeatedly_affected_pages_vs_prior_age"] = (
                            0 if previous_pages is None else len(current_pages & previous_pages)
                        )
                        last_updated_pages[temporal_key] = current_pages
                        records.append({
                            "scenario": sdir.name, "task_id": int(data["task_id"]),
                            "state_id": int(data["state_id"]), "step": step,
                            "source_step": expected_source, "source_age": age,
                            "pattern": pattern, "reuse_count_match": len(reused) == n,
                            "foreground_tokens_from_stage3c": int(r["foreground_tokens"]),
                            "protected_background_tokens": int(r["task_relevant_background_protected"]),
                            "background_reused_tokens": n if pattern == "real" else len(reused),
                            **metric,
                        })
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise RuntimeError("No reuse trace rows were produced.")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows); f.flush(); os.fsync(f.fileno())


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows: groups[(r["pattern"], r["page_tokens"])].append(r)
    metrics = ["update_page_touch_ratio", "shared_page_ratio", "mixed_copy_on_write_pages",
               "shared_unchanged_pages", "baseline_total_update_traffic_bytes",
               "cow_update_traffic_bytes", "cow_plus_dense_materialization_traffic_bytes",
               "cow_write_amplification_vs_fresh_rows", "estimated_live_unique_bytes_source_plus_cow_output"]
    out=[]
    for (pattern, block), rr in sorted(groups.items()):
        row={"pattern":pattern,"page_tokens":block,"reuse_observations":len(rr)}
        for metric in metrics:
            vals=[float(r[metric]) for r in rr]
            row[metric+"_mean"]=statistics.fmean(vals)
            row[metric+"_median"]=statistics.median(vals)
            row[metric+"_p95"]=sorted(vals)[min(len(vals)-1,math.ceil(.95*len(vals))-1)]
        out.append(row)
    return out


def plot_summary(path: Path, summary: list[dict]) -> None:
    patterns=["real","synthetic_clustered","synthetic_dispersed","synthetic_random"]
    labels={"real":"Stage 3C actual","synthetic_clustered":"Synthetic clustered",
            "synthetic_dispersed":"Synthetic dispersed","synthetic_random":"Synthetic random"}
    colors={"real":"#1565c0","synthetic_clustered":"#2e7d32",
            "synthetic_dispersed":"#c62828","synthetic_random":"#6a1b9a"}
    fig, axes=plt.subplots(2,3,figsize=(15,8))
    specs=[("shared_page_ratio_mean","Unchanged pages shared / all pages",axes[0,0]),
           ("cow_update_traffic_bytes_mean","COW update traffic (MiB)",axes[0,1]),
           ("cow_plus_dense_materialization_traffic_bytes_mean","COW + dense materialization traffic (MiB)",axes[1,0]),
           ("update_page_touch_ratio_mean","Pages needing update / all pages",axes[1,1]),
           ("cow_write_amplification_vs_fresh_rows_mean","COW write amplification",axes[0,2]),
           ("estimated_live_unique_bytes_source_plus_cow_output_mean","Estimated live source + COW pages (MiB)",axes[1,2])]
    for metric,title,ax in specs:
        for pat in patterns:
            subset=sorted([r for r in summary if r["pattern"]==pat],key=lambda r:r["page_tokens"])
            if subset:
                vals=[r[metric] / (1024*1024) if ("traffic" in metric or "bytes" in metric) else r[metric] for r in subset]
                ax.plot([r["page_tokens"] for r in subset],vals,marker="o",label=labels[pat],color=colors[pat])
        ax.set_title(title); ax.set_xlabel("KV page capacity (tokens)"); ax.set_xticks(PAGE_SIZES); ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    axes[1,0].set_ylabel("MiB per reuse decision; theoretical traffic")
    fig.suptitle("Track C Experiment 1 — Stage 3C R5 token-page trace model")
    fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=170); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args(); out=args.out
    rows=collect_trace()
    write_csv(out/"trace_analysis.csv",rows)
    aggs=aggregate(rows)
    write_csv(out/"trace_analysis_aggregate.csv",aggs)
    plot_summary(out/"figures"/"page_copy_spatial_analysis.png",aggs)
    real=[r for r in rows if r["pattern"]=="real"]
    age_stats=[]
    for age in range(1,5):
        subset=[r for r in real if r["source_age"]==age]
        age_stats.append({"age":age,"reuse_steps":len({(r['scenario'],r['step']) for r in subset}),
                          "mean_reused_visual_tokens":statistics.fmean(r["background_reused_tokens"] for r in subset),
                          "mean_shared_pages_by_block":{str(b):statistics.fmean(r["shared_unchanged_pages"] for r in subset if r["page_tokens"]==b) for b in PAGE_SIZES},
                          "mean_cow_pages_by_block":{str(b):statistics.fmean(r["mixed_copy_on_write_pages"] for r in subset if r["page_tokens"]==b) for b in PAGE_SIZES},
                          "mean_newly_affected_pages_by_block":{str(b):statistics.fmean(r["newly_affected_pages_vs_prior_age"] for r in subset if r["page_tokens"]==b) for b in PAGE_SIZES},
                          "mean_repeatedly_affected_pages_by_block":{str(b):statistics.fmean(r["repeatedly_affected_pages_vs_prior_age"] for r in subset if r["page_tokens"]==b) for b in PAGE_SIZES}})
    cycles=defaultdict(list)
    for r in real: cycles[(r["scenario"],r["source_step"],r["page_tokens"])].append(r)
    cycle_rows=[]
    for (scenario,source_step,block), rr in sorted(cycles.items()):
        rr.sort(key=lambda x:x["source_age"])
        cache_bytes=int(rr[0]["baseline_full_clone_payload_bytes"])
        cycle_rows.append({"scenario":scenario,"source_step":source_step,"page_tokens":block,
                           "reuse_ages_observed":len(rr),"baseline_reuse_traffic_bytes":sum(int(x["baseline_total_update_traffic_bytes"]) for x in rr),
                           "cow_reuse_update_traffic_bytes":sum(int(x["cow_update_traffic_bytes"]) for x in rr),
                           "cow_plus_dense_materialization_traffic_bytes":sum(int(x["cow_plus_dense_materialization_traffic_bytes"]) for x in rr),
                           "dense_source_snapshot_clone_read_write_bytes_once":2*cache_bytes,
                           "baseline_cycle_including_source_snapshot_bytes":2*cache_bytes+sum(int(x["baseline_total_update_traffic_bytes"]) for x in rr),
                           "cow_cycle_including_dense_materialization_and_source_conversion_excluded_bytes":2*cache_bytes+sum(int(x["cow_plus_dense_materialization_traffic_bytes"]) for x in rr),
                           "newly_affected_pages_total":sum(int(x["newly_affected_pages_vs_prior_age"]) for x in rr),
                           "repeatedly_affected_pages_total":sum(int(x["repeatedly_affected_pages_vs_prior_age"]) for x in rr)})
    write_csv(out/"r5_cycle_copy_summary.csv",cycle_rows)
    # Temporal page activity plot from the actual, not synthetic, masks.
    fig,ax=plt.subplots(figsize=(8,4.5))
    for block in PAGE_SIZES:
        vals=[]
        for age in range(1,5):
            subset=[r for r in real if r["page_tokens"]==block and r["source_age"]==age]
            vals.append(statistics.fmean(r["repeatedly_affected_pages_vs_prior_age"] for r in subset) if subset else 0)
        ax.plot(range(1,5),vals,marker="o",label=f"{block} token pages")
    ax.set_xlabel("R5 source age (reuse steps)"); ax.set_ylabel("Mean pages also affected at previous age")
    ax.set_title("Temporal page activity from actual Stage 3C reuse masks"); ax.set_xticks([1,2,3,4]); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out/"figures"/"page_temporal_activity_by_age.png",dpi=170); plt.close(fig)
    summary={"status":"trace_analysis_complete","source":"Stage 3C protect_10 per-scenario results.json; exact reuse-index traces",
             "scenarios":9,"reuse_observations":len({(r['scenario'],r['step']) for r in real}),
             "page_sizes_tokens":list(PAGE_SIZES),"layers":LAYERS,"components":["K","V"],
             "actual_model_dims":{"layers":32,"kv_heads":32,"head_dim":128,"dtype":"bfloat16","batch":1,
                                  "source_config":"OpenVLAConfig.text_config for llama2-7b-pure; local-only load verified in openvla-cached:latest"},
             "r5_semantics":"reuse always references most recent full fresh refresh; reuse outputs are not promoted",
             "bytes_model":"traffic counts source reads + destination writes; fresh projected row read/write included. COW assumes full physical page copy for mixed pages; materialization reads all page values and writes dense K/V for current SDPA interface.",
             "synthetic_masks":"same reusable visual-token count as each actual reuse record; clustered contiguous run, evenly dispersed, seeded random; not policy masks",
             "age_summary":age_stats,"r5_cycle_summary_rows":len(cycle_rows),"output_rows":len(rows)}
    atomic_json(out/"trace_analysis_summary.json",summary)
    print(f"[TRACE_DONE] {summary['reuse_observations']} reuse records; {len(rows)} rows across patterns/page sizes",flush=True)


if __name__=="__main__": main()
