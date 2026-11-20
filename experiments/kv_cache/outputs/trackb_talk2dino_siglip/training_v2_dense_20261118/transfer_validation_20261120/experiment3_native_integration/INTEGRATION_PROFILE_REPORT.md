# OpenVLA Integration and Computational Overhead

Read-only profiling used saved frame `/opt/openvla/experiments/kv_cache/outputs/trackb_dinov2_segmentation_visual_trial/visualizations/task4_state0_step0000/camera_rgb.png` with its task instruction. No action generation, rollout, or cache mutation occurred.

- Native DINO output: `[1, 256, 1024]`; fused visual output: `[1, 256, 2176]`.
- Hook executions in each full prefill: 1; no second DINO pass was used.
- Direct versus captured feature max absolute error: 0.
- Baseline versus hooked prefill logits max absolute error: 0; allclose: True.

## Timing

Times are synchronized CPU wall-clock around GPU operations, with CUDA synchronization before and after timed calls; CUDA events were not used.
- Vision forward median over 30 repeats: 60.245 ms (p95 60.833).
- Vision forward with passive DINO hook median: 60.158 ms (p95 60.425); the small negative/positive difference is within timing noise, so no capture speedup or overhead reduction is claimed.
- Cached four-prompt patch normalization + similarities + sigmoid + threshold median: 0.415 ms/frame (p95 0.455). Patch L2 normalization alone was 0.216 ms and is already included in that total.
- Task setup for four representative prompts: tokenization 0.492 ms, text encoding 185.275 ms, MLP projection + text normalization 0.445 ms, total 186.212 ms.
- Changed/new single prompt encode + project + score median: 25.426 ms; this can be cached until the instruction changes.
- Text tower initialization: 10.06 s; VLA processor/model loading: 17.48 s.

| Unchanged-prompt control horizon | Task setup amortized + cached scoring (ms/control step) |
|---:|---:|
| 5 | 37.657 |
| 10 | 19.036 |
| 20 | 9.726 |
| 50 | 4.139 |

## Memory and limitations

- Device: Orin; peak CUDA allocation 17.93 GB; reserved 18.10 GB; allocated 17.45 GB after profiling.
- SigLIP text parameters occupy 1.80 GB at the loaded dtype. This is a parameter-footprint estimate; incremental allocator usage for text-only residency was not isolated.
- The full-policy-prefill single samples were 887.6 ms baseline and 321.6 ms with the hook. Their order/warmup sensitivity makes them non-comparable; they are retained in JSON but excluded from timing conclusions.
- No isolated K/V saved-computation timing was measured. Therefore no net acceleration is claimed; measured selection overhead is the minimum work that a future cache implementation must save to break even.
- The profile used four representative prompt strings and one saved observation. It did not profile all 80 COCO class prompts or task rollouts.
