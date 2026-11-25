# Hybrid integration smoke validation

A live CUDA smoke cycle completed on LIBERO Spatial task 4, initial state 0, in the validated `openvla-talk2dino-siglip:20261116` container. It covered one full refresh and the next four reuse decisions. This is an implementation validation, not a task-success rollout.

- One ordinary OpenVLA forward captured the native DINO `[1,256,1024]` output; no second DINO pass was run.
- The layer-15 Stage 3C sidecar captured 256 finite text-to-visual scores on the refresh.
- Fresh action-token and prefill-logit parity assertions passed with passive instrumentation.
- The same DINO hook computed current-frame V2 masks during reuse prefill and updated the Stage 3B skip context before Llama layers.
- All 32 layers validated full Q and true selected K/V projection; source ages were 1, 2, 3, and 4.
- The independent fresh anchor remained unchanged; reused K/V came from that anchor only.

| Step | Mode | Age | V2 fresh | Additional attention | Total fresh | Reuse candidates | K/V skipped per layer |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | refresh | 0 | 57 | 20 | 77 | 179 | 0 |
| 1 | reuse | 1 | 54 | 20 | 74 | 182 | 182 |
| 2 | reuse | 2 | 55 | 20 | 75 | 181 | 181 |
| 3 | reuse | 3 | 54 | 20 | 74 | 182 | 182 |
| 4 | reuse | 4 | 52 | 20 | 72 | 184 | 184 |

The initial un-isolated container lacked `open_clip`; the smoke was rerun in the already available isolated Talk2DINO-SigLIP image, which had the required dependency. No packages were installed and the existing COCO preparation container was not modified.
