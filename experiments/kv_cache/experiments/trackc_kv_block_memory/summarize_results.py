#!/usr/bin/env python3
"""Validate benchmark artifacts and generate Track C report/figures."""
from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def readcsv(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def mib(x):return float(x)/(1024**2)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();o=a.out
    trace=json.loads((o/"trace_analysis_summary.json").read_text())
    val=json.loads((o/"correctness_validation.json").read_text())
    bench=json.loads((o/"gpu_benchmark_summary.json").read_text())
    rows=readcsv(o/"gpu_benchmark.csv")
    if trace.get("status")!="trace_analysis_complete" or trace.get("scenarios")!=9:raise RuntimeError("Trace outputs incomplete")
    if not val.get("all_cases_passed"):raise RuntimeError("Correctness suite did not pass")
    if bench.get("status")!="complete":raise RuntimeError("GPU benchmark incomplete")
    if not rows:raise RuntimeError("GPU benchmark raw CSV is empty")
    raw={}
    for r in rows:
        if r["status"]!="sample":continue
        k=(int(r["page_tokens"]),r["condition"]);raw.setdefault(k,[]).append(r)
    page_sizes=sorted({k[0] for k in raw})
    required=["A_dense_clone_index_copy","B_COW_page_update_no_materialization","C_COW_plus_dense_SDPA_materialization"]
    result=[]
    reps=int(bench["configuration"]["repetitions"])
    selected={}
    for (b,c),rs in raw.items():
        by_attempt={}
        for x in rs:by_attempt.setdefault(x["attempt_id"],[]).append(x)
        complete=[v for v in by_attempt.values() if len(v)==reps]
        if complete:selected[(b,c)]=complete[-1]
    for b in page_sizes:
        for cond in required:
            rs=selected.get((b,cond),[])
            if len(rs)!=reps:raise RuntimeError(f"Expected {reps} raw samples for page={b}, {cond}; got {len(rs)}")
            walls=[float(x["wall_ms"]) for x in rs];cuda=[float(x["cuda_event_ms"]) for x in rs]
            result.append({"page_tokens":b,"condition":cond,"n":len(rs),"wall_mean_ms":statistics.fmean(walls),
              "wall_median_ms":statistics.median(walls),"wall_p95_ms":sorted(walls)[math.ceil(.95*len(walls))-1],
              "cuda_event_mean_ms":statistics.fmean(cuda),"cuda_event_median_ms":statistics.median(cuda),
              "peak_allocated_mib_max":max(mib(int(x["peak_allocated_bytes"])) for x in rs),
              "peak_reserved_mib_max":max(mib(int(x["peak_reserved_bytes"])) for x in rs)})
    # Include isolated copy and attention measurements, separately from the update path.
    ancillary={}
    for (b,c),rs in raw.items():
        if c in required:continue
        ancillary.setdefault(c,[]).extend(rs)
    anc_summary={}
    for name,rs in ancillary.items():
        anc_summary[name]={"samples":len(rs),"wall_median_ms":statistics.median(float(x["wall_ms"]) for x in rs),
          "cuda_event_median_ms":statistics.median(float(x["cuda_event_ms"]) for x in rs),
          "peak_allocated_mib_max":max(mib(int(x["peak_allocated_bytes"])) for x in rs)}
    for b in page_sizes:
        by={r["condition"]:r for r in result if r["page_tokens"]==b}
        by[required[0]]["cow_no_materialize_delta_wall_ms_vs_dense"]=by[required[1]]["wall_median_ms"]-by[required[0]]["wall_median_ms"]
        by[required[0]]["cow_plus_materialize_delta_wall_ms_vs_dense"]=by[required[2]]["wall_median_ms"]-by[required[0]]["wall_median_ms"]
    out=o/"figures";out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,5))
    names={required[0]:"Dense clone + index_copy",required[1]:"COW update (no attention materialization)",required[2]:"COW + dense materialization"}
    for c in required:
        rr=[r for r in result if r["condition"]==c]
        ax.plot([r["page_tokens"] for r in rr],[r["wall_median_ms"] for r in rr],marker="o",label=names[c])
    ax.set_xlabel("Page capacity (tokens)");ax.set_ylabel("GPU-synchronized operation wall time (ms)")
    ax.set_xticks(page_sizes);ax.set_title("Track C: storage update and materialization latency");ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"gpu_update_latency.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for c in required:
        rr=[r for r in result if r["condition"]==c]
        ax.plot([r["page_tokens"] for r in rr],[r["peak_allocated_mib_max"] for r in rr],marker="o",label=names[c])
    ax.set_xlabel("Page capacity (tokens)");ax.set_ylabel("Peak extra CUDA allocated memory (MiB)")
    ax.set_xticks(page_sizes);ax.set_title("Peak incremental allocation during measured operation");ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"gpu_peak_allocation.png",dpi=170);plt.close(fig)
    trace_csv=readcsv(o/"trace_analysis_aggregate.csv")
    real=[r for r in trace_csv if r["pattern"]=="real"]
    best=min(real,key=lambda r:float(r["cow_update_traffic_bytes_mean"]))
    overall={"status":"complete","trace_analysis":trace,"correctness_validation":val,
      "benchmark":{"configuration":bench["configuration"],"per_page_condition":result,"ancillary_measurements":anc_summary,
        "raw_samples_csv":"gpu_benchmark.csv","n_samples_per_condition_page":reps},
      "trace_best_page_by_COW_update_traffic":{"page_tokens":int(best["page_tokens"]),"mean_bytes_per_reuse":float(best["cow_update_traffic_bytes_mean"])},
      "interpretation_limits":["Trace bytes are a modeled traffic estimate, not hardware-counter traffic.",
        "COW-without-materialization is not usable by the current dense DynamicCache/SDPA path by itself.",
        "COW+materialization includes the full dense tensor reconstruction needed before SDPA.",
        "This benchmark does not measure complete OpenVLA inference or robot action equivalence."]}
    (o/"summary.json").write_text(json.dumps(overall,indent=2,sort_keys=True)+"\n")
    report=f'''# Track C Experiment 1 — KV Block Memory Management\n\n## Status\n\nCompleted offline analysis of nine saved Stage 3C R5 traces and a CUDA microbenchmark on the OpenVLA Jetson container. No LIBERO rollouts were run and no policy code was changed.\n\n## Trace-based update pattern\n\nThe real Stage 3C masks contain **{trace['reuse_observations']}** reuse decisions over nine task/state scenarios. Analysis covers 32 layers, K and V, BF16, and token page capacities 1, 4, 8, 16 and 32. The trace model preserves the immutable full-refresh source and actual reusable visual indices. Same-budget clustered, dispersed and seeded-random masks are synthetic controls.\n\nThe lowest modeled COW update traffic among page sizes occurred at **{int(best['page_tokens'])} tokens/page** ({mib(float(best['cow_update_traffic_bytes_mean'])):.2f} MiB mean per reuse decision, excluding dense attention materialization). This is a trace-model result, not a measured bandwidth result.\n\nSee `trace_analysis.csv`, `trace_analysis_aggregate.csv`, `r5_cycle_copy_summary.csv`, and `figures/page_copy_spatial_analysis.png`.\n\n## CUDA microbenchmark\n\nActual tensor dimensions came from the local OpenVLA config: {bench['configuration']['model']['layers']} layers, {bench['configuration']['model']['kv_heads']} KV heads, head dimension {bench['configuration']['model']['head_dim']}, BF16. Inputs use one real Stage 3C task4/state0 age-1 reuse mask. Each measured update condition uses 20 warmups and 50 repetitions. Wall times include Python/page management and synchronized GPU work; CUDA-event times are retained separately. This is an isolated cache-update microbenchmark, not end-to-end OpenVLA latency.\n\n| Page tokens | Dense clone/index_copy median ms | COW update only median ms | COW + dense materialization median ms |\n|---:|---:|---:|---:|\n'''
    for b in page_sizes:
        d={r["condition"]:r for r in result if r["page_tokens"]==b}
        report+=f"| {b} | {d[required[0]]['wall_median_ms']:.4f} | {d[required[1]]['wall_median_ms']:.4f} | {d[required[2]]['wall_median_ms']:.4f} |\n"
    report+=f'''\nCorrectness: **{len(val['passed_cases'])}/{len(val['passed_cases'])}** synthetic cases passed exact logical K/V equality, immutable-source check, and reference-count release checks. The current Transformers DynamicCache + SDPA implementation expects dense `[batch, heads, sequence, head_dim]` K/V; the COW-only case is not directly consumable. Its required dense reconstruction is measured separately and in the combined C condition. SDPA dense-reference and source snapshot clone/pageification measurements are ancillary in `summary.json` and raw CSV.\n\n## Assessment\n\nPhysical page sharing may reduce update copying in the storage-only model, especially at small page sizes, but the existing attention interface requires full dense K/V materialization. Compare the COW-only row to the combined C row before inferring value: savings that disappear after materialization do not support integrating this manager. Page metadata/refcount work and fixed pool allocation are included or separately measured as recorded in the raw results. No actual GPU copy-traffic counters or end-to-end policy latency were collected.\n\n**Conclusion:** this experiment evaluates update mechanics, not PagedAttention and not end-to-end inference acceleration. Integration into live OpenVLA is not justified unless a future compatible attention read-view can avoid the dense reconstruction without a custom kernel.\n\n## Reproduction\n\nSee `experiments/kv_cache/experiments/trackc_kv_block_memory/README.md` for the Docker command, validation, launch, monitoring and resume instructions.\n'''
    (o/"REPORT.md").write_text(report)
    print("[AGGREGATION_VALIDATED] report, summary, CSV checks and plots saved",flush=True)
if __name__=="__main__":main()
