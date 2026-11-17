# Real-Weight Model Smoke Test

Date: 2026-11-16. This is a bounded validation using three saved Track B images and three short captions. It is not a feature-extraction pilot, training run or policy evaluation.

## Results

- **SigLIP text tower:** official pinned 384 checkpoint, strict text-only load of 330 tensors, no missing/unexpected keys, CPU evaluation. Configuration assertions passed: context 64, width/output 1152, 27 layers, 16 heads, positional embedding `[64,1152]`.
- **Text outputs:** finite `[3,1152]`; repeated evaluation exactly equal. Text-only extracted tower and full-checkpoint text tower were exactly equal on the test input (max/mean absolute difference 0; relative L2 0).
- **DINOv2 region route:** existing OpenVLA DINOv2 TIMM backbone on CUDA produced `[1,256,1024]` final patch features, `[1,16,256]` CLS-to-patch attention, and `[1,16,1024]` region descriptors for each of three images.
- **Cross-modal path:** similarity matrix `[3,3]` finite; loss finite (`1.1335974`); projection MLP gradients finite and present; frozen vision/text backbones had no gradients; optimizer contained only MLP parameters.
- **OpenVLA preservation:** standard fused feature outputs before and after sidecar evaluation were exactly equal for all three images (max and mean absolute difference 0; relative L2 0).
- **No hidden extra model:** text-only loading did not load the SigLIP visual tower. DINO used the already initialized OpenVLA model; no redundant DINO backbone was instantiated for the smoke test.

## Resource observations

Environment: Jetson AGX Orin, CUDA 12.8, PyTorch 2.6.0, OpenCLIP 2.32.0. The text-only checkpoint and full-checkpoint reference were loaded on CPU; the existing OpenVLA DINO ran on `cuda:0`.

- Process peak RSS: 7,600,218,112 bytes (~7.08 GiB).
- CUDA allocated after OpenVLA load: 1,475,333,120 bytes (~1.37 GiB).
- Peak CUDA allocated: 1,501,627,392 bytes (~1.40 GiB).
- Peak CUDA reserved: 1,539,309,568 bytes (~1.43 GiB).

The per-image DINO sidecar path times recorded by the smoke harness were 117.09 ms, 24.74 ms and 27.39 ms. These are three descriptive observations with first-call overhead in the first value; they are not a controlled latency benchmark and do not represent the complete future extraction or segmentation cost.

## Scope and caveat

The test generated no robot actions and changed no default OpenVLA forward behavior. No COCO features, training or LIBERO rollouts were run. The existing OpenVLA runtime emitted its known Transformers/tokenizers compatibility warning (it expected Transformers 4.40.1/tokenizers 0.19.1, while this fixed environment has 4.49.0/0.21.0). No dependency was changed; the validation completed. This warning should be retained as an environment caveat for the next controlled pilot.

Machine-readable evidence: `real_model_smoke_test.json`.
