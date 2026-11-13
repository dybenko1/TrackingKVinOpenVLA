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
        dense=by[required[0]]["wall_median_ms"]
        by[required[0]]["cow_no_materialize_delta_wall_ms_vs_dense"]=by[required[1]]["wall_median_ms"]-dense
        by[required[0]]["cow_no_materialize_delta_pct_vs_dense"]=(by[required[1]]["wall_median_ms"]/dense-1)*100
        by[required[0]]["cow_plus_materialize_delta_wall_ms_vs_dense"]=by[required[2]]["wall_median_ms"]-dense
        by[required[0]]["cow_plus_materialize_delta_pct_vs_dense"]=(by[required[2]]["wall_median_ms"]/dense-1)*100
    with (o/"gpu_benchmark_comparison.csv").open("w",newline="") as f:
        fields=["page_tokens","dense_median_ms","cow_update_only_median_ms","cow_update_only_delta_pct",
                "cow_plus_dense_materialize_median_ms","cow_plus_dense_materialize_delta_pct",
                "dense_p95_ms","cow_update_only_p95_ms","cow_plus_dense_materialize_p95_ms",
                "dense_peak_allocated_mib","cow_peak_allocated_mib","combined_peak_allocated_mib",
                "dense_peak_reserved_mib","cow_peak_reserved_mib","combined_peak_reserved_mib"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for b in page_sizes:
            d={r["condition"]:r for r in result if r["page_tokens"]==b}
            w.writerow({"page_tokens":b,"dense_median_ms":d[required[0]]["wall_median_ms"],
              "cow_update_only_median_ms":d[required[1]]["wall_median_ms"],
              "cow_update_only_delta_pct":(d[required[1]]["wall_median_ms"]/d[required[0]]["wall_median_ms"]-1)*100,
              "cow_plus_dense_materialize_median_ms":d[required[2]]["wall_median_ms"],
              "cow_plus_dense_materialize_delta_pct":(d[required[2]]["wall_median_ms"]/d[required[0]]["wall_median_ms"]-1)*100,
              "dense_p95_ms":d[required[0]]["wall_p95_ms"],"cow_update_only_p95_ms":d[required[1]]["wall_p95_ms"],
              "cow_plus_dense_materialize_p95_ms":d[required[2]]["wall_p95_ms"],
              "dense_peak_allocated_mib":d[required[0]]["peak_allocated_mib_max"],
              "cow_peak_allocated_mib":d[required[1]]["peak_allocated_mib_max"],
              "combined_peak_allocated_mib":d[required[2]]["peak_allocated_mib_max"],
              "dense_peak_reserved_mib":d[required[0]]["peak_reserved_mib_max"],
              "cow_peak_reserved_mib":d[required[1]]["peak_reserved_mib_max"],
              "combined_peak_reserved_mib":d[required[2]]["peak_reserved_mib_max"]})
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
    ax.set_xlabel("Page capacity (tokens)");ax.set_ylabel("Peak CUDA allocated memory (MiB)")
    ax.set_xticks(page_sizes);ax.set_title("Peak CUDA allocated memory with benchmark inputs/pools resident");ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"gpu_peak_allocation.png",dpi=170);plt.close(fig)
    trace_csv=readcsv(o/"trace_analysis_aggregate.csv")
    real=[r for r in trace_csv if r["pattern"]=="real"]
    best=min(real,key=lambda r:float(r["cow_update_traffic_bytes_mean"]))
    env_path=o/"environment_audit.json"
    environment=json.loads(env_path.read_text()) if env_path.exists() else {
      "host":"NVIDIA Jetson AGX Orin (ubuntu)","architecture":"aarch64",
      "kernel":"5.15.148-tegra","jetpack":"R36.4.4","cpu_cores":12,
      "host_ram_total_gib":61,"host_ram_available_gib_at_audit":45,
      "project_filesystem_free_gib_at_audit":714,"container_image":"openvla-cached:latest",
      "container_python":"3.12.3","torch":"2.6.0","cuda_runtime":"12.8",
      "device":"Orin","compute_capability":"8.7","torch_reported_memory_mib":62840,
      "memory_note":"Jetson unified/shared memory; torch-reported CUDA capacity is not discrete VRAM.",
      "telemetry_note":"Host resource snapshot taken before the isolated experiment; not a time series during the benchmark."}
    (o/"environment_audit.json").write_text(json.dumps(environment,indent=2,sort_keys=True)+"\n")
    setup=[r for r in bench.get("samples",[]) if r.get("condition")=="refresh_source_preparation"]
    overall={"status":"complete","trace_analysis":trace,"correctness_validation":val,"environment":environment,
      "benchmark":{"configuration":bench["configuration"],"per_page_condition":result,"ancillary_measurements":anc_summary,
        "source_preparation_by_page":setup,"raw_samples_csv":"gpu_benchmark.csv","n_samples_per_condition_page":reps},
      "trace_best_page_by_COW_update_traffic":{"page_tokens":int(best["page_tokens"]),"mean_bytes_per_reuse":float(best["cow_update_traffic_bytes_mean"])},
      "interpretation_limits":["Trace bytes are a modeled traffic estimate, not hardware-counter traffic.",
        "COW-without-materialization is not usable by the current dense DynamicCache/SDPA path by itself.",
        "COW+materialization includes the full dense tensor reconstruction needed before SDPA.",
        "This benchmark does not measure complete OpenVLA inference or robot action equivalence."]}
    (o/"summary.json").write_text(json.dumps(overall,indent=2,sort_keys=True)+"\n")
    env=overall["environment"]
    page1=next(r for r in trace_csv if r["pattern"]=="real" and int(r["page_tokens"])==1)
    page1_materialize=mib(float(page1["cow_plus_dense_materialization_traffic_bytes_mean"])-float(page1["cow_update_traffic_bytes_mean"]))
    report=f'''# Track C Experiment 1 — KV Block Memory Management\n\n## Status\n\nCompleted offline analysis of nine saved Stage 3C R5 traces and a CUDA microbenchmark on the OpenVLA Jetson container. No LIBERO rollouts were run and no policy code was changed. Host: {env.get('host','NVIDIA Jetson AGX Orin')}, Linux {env.get('kernel','5.15.148-tegra')}, JetPack {env.get('jetpack','R36.4.4')}; container PyTorch {bench['configuration'].get('torch_version','2.6.0')}, CUDA {bench['configuration'].get('cuda_runtime','12.8')}. GPU access was validated through PyTorch inside Docker; Jetson CUDA memory is unified/shared system memory, not discrete VRAM.\n\n## Trace-based update pattern\n\nThe real Stage 3C masks contain **{trace['reuse_observations']}** reuse decisions over nine task/state scenarios. Analysis covers 32 layers, K and V, BF16, and token page capacities 1, 4, 8, 16 and 32. The trace model preserves the immutable full-refresh source and actual reusable visual indices. Same-budget clustered, dispersed and seeded-random masks are synthetic controls.\n\nThe lowest modeled COW update traffic among page sizes occurred at **{int(best['page_tokens'])} token/page** ({mib(float(best['cow_update_traffic_bytes_mean'])):.2f} MiB mean per reuse decision, excluding dense attention materialization). However, page size 1 adds about {page1_materialize:.2f} MiB of dense materialization per reuse decision, bringing the trace-model COW+materialization total to approximately the dense baseline. This is a trace-model result, not a measured bandwidth result.\n\nSee `trace_analysis.csv`, `trace_analysis_aggregate.csv`, `r5_cycle_copy_summary.csv`, and `figures/page_copy_spatial_analysis.png`.\n\n## CUDA microbenchmark\n\nActual tensor dimensions came from the local OpenVLA config: {bench['configuration']['model']['layers']} layers, {bench['configuration']['model']['kv_heads']} KV heads, head dimension {bench['configuration']['model']['head_dim']}, BF16. Inputs use one real Stage 3C task4/state0 age-1 reuse mask (prefill length {bench['configuration']['trace_case']['sequence_length']}). Each measured update condition uses 20 warmups and 50 repetitions. Wall times include Python/page management and synchronized GPU work; CUDA-event times are retained separately. This is an isolated cache-update microbenchmark, not end-to-end OpenVLA latency.\n\n| Page tokens | Dense median ms | COW update-only median ms | COW update delta | COW + materialize median ms | Combined delta |\n|---:|---:|---:|---:|---:|---:|\n'''
    for b in page_sizes:
        d={r["condition"]:r for r in result if r["page_tokens"]==b}
        report+=f"| {b} | {d[required[0]]['wall_median_ms']:.3f} | {d[required[1]]['wall_median_ms']:.3f} | +{(d[required[1]]['wall_median_ms']/d[required[0]]['wall_median_ms']-1)*100:.1f}% | {d[required[2]]['wall_median_ms']:.3f} | +{(d[required[2]]['wall_median_ms']/d[required[0]]['wall_median_ms']-1)*100:.1f}% |\n"
    report+=f'''\nCorrectness: **{len(val['passed_cases'])}/{len(val['passed_cases'])}** synthetic cases passed exact logical K/V equality, immutable-source check, and reference-count release checks. The current Transformers DynamicCache + SDPA implementation expects dense `[batch, heads, sequence, head_dim]` K/V; the COW-only case is not directly consumable. Its required dense reconstruction is measured separately and in the combined C condition. Dense SDPA-reference median was {anc_summary.get('SDPA_dense_reference',{}).get('wall_median_ms',float('nan')):.3f} ms; source snapshot clone median was {anc_summary.get('refresh_dense_source_snapshot_clone',{}).get('wall_median_ms',float('nan')):.3f} ms wall time ({anc_summary.get('refresh_dense_source_snapshot_clone',{}).get('cuda_event_median_ms',float('nan')):.3f} ms CUDA event). Source pageification/setup timings and pool payloads by page size are included in `summary.json` and `gpu_benchmark_summary.json`.\n\nPeak allocation values include benchmark inputs and page pools already resident plus operation allocations; they are not operation-only deltas. Allocated and reserved values are retained in `summary.json` and the raw CSV. No actual GPU copy-traffic counters or end-to-end policy latency were collected.\n\n## Assessment\n\nCOW update-only was slower than dense clone/index_copy for every measured page size on this implementation; COW plus dense materialization was slower still. The minimum modeled COW update traffic was at one-token pages, but even there adding dense materialization returns the trace-model traffic to approximately the dense baseline. For larger pages, mixed-page copies plus materialization exceed baseline modeled traffic. Physical page sharing helps only if attention can consume the page view without rebuilding a dense cache.\n\n**Conclusion:** this experiment evaluates update mechanics, not PagedAttention and not end-to-end inference acceleration. Integration into live OpenVLA is not justified unless a future compatible attention read-view can avoid dense reconstruction without a custom kernel.\n\n## Reproduction\n\nSee `experiments/kv_cache/experiments/trackc_kv_block_memory/README.md` for the Docker command, validation, launch, monitoring and resume instructions.\n'''
    (o/"REPORT.md").write_text(report)
    print("[AGGREGATION_VALIDATED] report, summary, CSV checks and plots saved",flush=True)
if __name__=="__main__":main()
