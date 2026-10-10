# Stage 3B: full-layer oracle-masked K/V reuse

## Run status

Complete. Nine task/state scenarios ran with paired Fresh and Stage 3B policies. No latency benchmark was performed. The result commit contains per-step raw records and this aggregate report.

## Configuration and implementation revision

- Suite: LIBERO Spatial; tasks 4, 5, 7; states 0, 1, 2.
- Seed 7, official-style 10 settling steps, horizon 220, greedy OpenVLA checkpoint `openvla/openvla-7b-finetuned-libero-spatial`.
- R=5; refresh at `step % 5 == 0`; age 1–4 reuses the most recent independent fresh source.
- All 32 Llama layers; visual cache positions 1–256; background K and V reused; foreground K and V projected fresh; Q full sequence; BOS/text fresh; action decoding delegated normally.
- Implementation commit: `699fa6cbe7f0a8ece11cf6a96a0eef52634a7c36`.
- Exact commands: `python experiments/kv_cache/experiments/stage3b_oracle_full_kv_matrix.py --validate-only`; after a passing validation, `python experiments/kv_cache/experiments/stage3b_oracle_full_kv_matrix.py --matrix` (both run inside container `f5240fe761b5`, working directory `/opt/openvla`).

## Results

| Task | State | Stage 3A Fresh (steps) | Stage 3A K-only (steps) | Stage 3B Fresh (steps) | Stage 3B K+V (steps) | Refresh | Reuse | Temporal reuse | K rows omitted | V rows omitted |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 | 0 | Success (140) | Success (142) | Success (140) | Fail (220) | 44 | 176 | 80.0% | 666,592 | 666,592 |
| 4 | 1 | Success (145) | Fail (220) | Success (145) | Fail (220) | 44 | 176 | 80.0% | 643,648 | 643,648 |
| 4 | 2 | Success (130) | Success (132) | Success (130) | Success (124) | 25 | 99 | 79.8% | 397,760 | 397,760 |
| 5 | 0 | Success (105) | Success (113) | Success (105) | Success (110) | 22 | 88 | 80.0% | 342,880 | 342,880 |
| 5 | 1 | Success (89) | Success (90) | Success (89) | Success (103) | 21 | 82 | 79.6% | 325,760 | 325,760 |
| 5 | 2 | Success (90) | Success (103) | Success (90) | Fail (220) | 44 | 176 | 80.0% | 680,704 | 680,704 |
| 7 | 0 | Success (111) | Success (183) | Success (111) | Success (194) | 39 | 155 | 79.9% | 681,856 | 681,856 |
| 7 | 1 | Success (120) | Success (125) | Success (120) | Success (123) | 25 | 98 | 79.7% | 423,616 | 423,616 |
| 7 | 2 | Fail (220) | Success (132) | Fail (220) | Fail (220) | 44 | 176 | 80.0% | 709,728 | 709,728 |

Stage 3A success counts: Fresh **8/9**, K-only masked reuse **8/9**. Stage 3B: Fresh **8/9**, full-layer K+V masked reuse **5/9**. For each Stage 3B scenario, `summary.json` reports skipped projection row totals by layer, aggregate K/V rows, effective K/V and combined row fractions, spatial background-token fraction, and temporal reuse fraction. The approximately 80% temporal reuse rate is distinct from the foreground-mask spatial fraction and the actual projection-row fraction. No measured latency is included.

## Reproduction and validation

The supervisor first ran the six-step task-4/state-0 same-observation/source comparison between compute-then-overwrite and true skip. It required all 32 layers to be instrumented, exact stale-background K/V cache provenance and cache shape agreement, full Q rows, original-position RoPE for fresh K, unrotated V, delegated one-token decoding, no unexpected fallbacks, finite actions, and valid R5 source ages. Validation JSON is `validation.json`.

After validation passed, the supervisor ran the full matrix sequentially. Each scenario JSON was saved immediately, and `summary.json` was updated after every scenario. Reproduce the full sequence with `python experiments/kv_cache/experiments/supervise_stage3b.py` on the lab host. The supervisor performs post-run integrity checks, writes a compact status log excerpt and file-hash manifest, then commits and pushes the artifacts.

Result files and hashes are listed in `manifest.json`; the compact execution excerpt is `execution_excerpt.log`. Runtime log is `supervisor.log` locally.
