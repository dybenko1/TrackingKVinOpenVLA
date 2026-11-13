# Track C Experiment 1 — KV Block Memory Management

## Status

Completed offline analysis of nine saved Stage 3C R5 traces and a CUDA microbenchmark on the OpenVLA Jetson container. No LIBERO rollouts were run and no policy code was changed. Host: NVIDIA Jetson AGX Orin (ubuntu), Linux 5.15.148-tegra, JetPack R36.4.4; container PyTorch 2.6.0, CUDA 12.8. GPU access was validated through PyTorch inside Docker; Jetson CUDA memory is unified/shared system memory, not discrete VRAM.

## Trace-based update pattern

The real Stage 3C masks contain **879** reuse decisions over nine task/state scenarios. Analysis covers 32 layers, K and V, BF16, and token page capacities 1, 4, 8, 16 and 32. The trace model preserves the immutable full-refresh source and actual reusable visual indices. Same-budget clustered, dispersed and seeded-random masks are synthetic controls.

The lowest modeled COW update traffic among page sizes occurred at **1 token/page** (173.97 MiB mean per reuse decision, excluding dense attention materialization). However, page size 1 adds about 289.10 MiB of dense materialization per reuse decision, bringing the trace-model COW+materialization total to approximately the dense baseline. This is a trace-model result, not a measured bandwidth result.

See `trace_analysis.csv`, `trace_analysis_aggregate.csv`, `r5_cycle_copy_summary.csv`, and `figures/page_copy_spatial_analysis.png`.

## CUDA microbenchmark

Actual tensor dimensions came from the local OpenVLA config: 32 layers, 32 KV heads, head dimension 128, BF16. Inputs use one real Stage 3C task4/state0 age-1 reuse mask (prefill length 292). Each measured update condition uses 20 warmups and 50 repetitions. Wall times include Python/page management and synchronized GPU work; CUDA-event times are retained separately. This is an isolated cache-update microbenchmark, not end-to-end OpenVLA latency.

| Page tokens | Dense median ms | COW update-only median ms | COW update delta | COW + materialize median ms | Combined delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.390 | 19.331 | +44.4% | 163.926 | +1124.2% |
| 4 | 13.330 | 38.280 | +187.2% | 92.389 | +593.1% |
| 8 | 13.378 | 51.424 | +284.4% | 74.109 | +454.0% |
| 16 | 10.242 | 54.065 | +427.9% | 68.623 | +570.0% |
| 32 | 10.235 | 42.538 | +315.6% | 57.750 | +464.3% |

### Copy model and temporal pattern

The modeled dense update moves 463.07 MiB per reuse decision on average (clone read/write plus fresh-row read/write). For actual masks, fully shareable page fraction was 39.8% at one-token pages, 18.0% at four-token pages, and 4.2% at eight-token pages; it falls below 1% at 16 and is 0% at 32. At 8-token pages, the same-budget synthetic clustered mask shared 37.5% of pages, while dispersed shared 0% and random shared 0.2%. The actual mask is therefore spatially dispersed at these block sizes, limiting page sharing.

Reuse decisions by age were age 1: 222, age 2: 221, age 3: 219, age 4: 217. The source mask averaged about 115.3 reusable visual tokens. At one-token pages, most updated pages reappeared at the following age: mean repeated pages were 173.3, 173.3, and 173.4 for ages 2–4, after about 173.8 newly affected pages at age 1.

The trace-model R5 cycle including one dense source-snapshot clone averaged 2122.6 MiB with the dense baseline. COW plus required materialization averaged 2122.6 MiB at one-token pages; this estimate excludes source-to-page conversion.

## Correctness and limits

**30/30** synthetic cases passed exact logical K/V equality, immutable-source checks, page shape/order checks, and reference-count release checks. The current Transformers DynamicCache + SDPA path consumes dense `[batch, heads, sequence, head_dim]` K/V, so the COW-only view cannot be handed directly to attention. Its required dense reconstruction is measured in the combined C condition. Dense one-layer SDPA reference median was 0.546 ms; dense source snapshot clone median was 4.774 ms wall time (4.223 ms CUDA event). Source pageification/setup timings and pool payloads by page size are included in `summary.json` and `gpu_benchmark_summary.json`.

Peak allocation values include benchmark inputs and page pools already resident plus operation allocations; they are not operation-only deltas. Allocated and reserved values are retained in `summary.json` and the raw CSV. We did not have an actual GPU copy-traffic counter available, and did not measure full OpenVLA inference or robot action equivalence.

## Assessment

COW update-only was slower than dense clone/index_copy for every measured page size on this implementation; COW plus dense materialization was slower still. Page size 1 minimizes the trace-model update traffic but only works as a storage saving before materialization. The combined measured implementation remained slower than dense clone/index_copy at every page size. This differs from PagedAttention's primary serving setting, where managing dynamic caches and sharing across concurrent requests is designed to improve memory use and serving throughput ([PagedAttention paper](https://arxiv.org/abs/2309.06180)). The present single-policy path has no paged attention read-view.

**Conclusion:** this experiment evaluates update mechanics, not end-to-end acceleration. Do not integrate this manager into live OpenVLA unless a compatible attention read-view can consume shared pages without dense reconstruction.

## Reproduction and provenance

Benchmark implementation commit: `c0c489c6e1602d998e9e48bd73c07f4a650a5021`. Final report generator commit: `eb2840bfeb40a3c67c128d56de2c01de517f188a`. Stage 3C trace results source: `4b0c1f7`. Container ID: `6ee4a4f91218784fca52848b8bc8d132b489acc96c8509c0f91faff9799b8556`. Raw timing rows are in `gpu_benchmark.csv`. See `experiments/kv_cache/experiments/trackc_kv_block_memory/README.md` for the Docker command, monitoring and resume instructions.
