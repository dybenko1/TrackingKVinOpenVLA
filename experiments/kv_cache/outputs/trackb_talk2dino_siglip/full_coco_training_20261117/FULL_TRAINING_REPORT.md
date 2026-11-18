# Full COCO Talk2DINO-SigLIP Training Report

## Status

The full COCO feature extraction, 100-epoch projection training, and fixed Phase 3 spatial comparison completed. The first post-training invocation reached spatial evaluation and stopped on a missing wrapper argument; training and its checkpoint were already complete. Recovery resumed from epoch 100, corrected the invocation, and did not repeat optimization. Final status is complete.

## Data and configuration

| Item | Result |
|---|---:|
| COCO train images | 82,783 |
| COCO train captions | 414,113 |
| Captions per epoch | 414,113, each once |
| Same-image batch collisions | 0 across every epoch |
| Batches per epoch | 3,238 (partial batches retained) |
| Epochs | 100 |
| Optimizer / learning rate | Adam / 0.0001 |
| Objective | Symmetric diagonal InfoNCE, temperature 0.07 |
| Trainable module | 1152 → 1024 → 1024 MLP with Tanh |
| Frozen image backbone | OpenVLA DINOv2 ViT-L/14-register |
| Frozen text encoder | SigLIP-384, context 64, output 1152 |

The caption-round sampler was checked against the entire annotation index before launch and confirmed to use every caption exactly once with no repeated image IDs in a batch. Training logs confirm these invariants for all 100 epochs.

## Feature integrity and training

The feature gate passed: all 82,783 DINO region rows and all 414,113 SigLIP text rows have the expected shapes, finite values, complete flags, unique IDs and aligned joins. Stored feature payloads were approximately 2.71 GB for DINO regions and 0.95 GB for SigLIP text vectors. The source SigLIP-384 safetensors file matched the pinned SHA-256 recorded in `configuration.json`.

Training peak process RSS was approximately 4.83 GiB, peak CUDA allocated memory approximately 98 MiB, and peak CUDA reserved memory 128 MiB. The vision/text feature extraction stages load their own backbones and are not represented by these MLP-only training memory figures.

Mean training InfoNCE decreased from **1.2326** in epoch 1 to **0.7005** in epoch 100. Every epoch processed 414,113 annotations in 3,238 batches and recorded zero same-image collisions. Aggregate optimizer time was approximately 2.56 hours; this excludes feature extraction and final spatial evaluation.

## Held-out retrieval monitoring

Multi-positive retrieval was evaluated every five epochs on the existing fixed 200-image / 1,000-caption validation cache. Validation did not control training or checkpoint selection.

| Epoch | Train InfoNCE | Image→text R@1 | Text→image R@1 | Symmetric multi-positive loss |
|---:|---:|---:|---:|---:|
| 5 | 0.9297 | 0.815 | 0.685 | 1.2302 |
| 10 | 0.8572 | 0.825 | 0.689 | 1.1951 |
| 20 | 0.7995 | 0.800 | 0.691 | 1.1901 |
| 40 | 0.7538 | 0.815 | 0.691 | 1.2061 |
| 60 | 0.7290 | 0.805 | 0.681 | 1.2240 |
| 80 | 0.7129 | 0.785 | 0.691 | 1.2375 |
| 100 | 0.7005 | 0.800 | 0.681 | 1.2519 |

The final retrieval result is above the 1,000-image pilot on this small cached subset: image-to-text R@1 was 0.800 versus 0.570 for the original pilot and 0.550 for the Phase 2 corrected pilot; text-to-image R@1 was 0.681 versus 0.452 and 0.460. The best monitored multi-positive loss occurred at epoch 20 and increased afterward, so retrieval does not show uniform improvement through the final epochs. This is a single small validation subset and does not establish broad generalization.

## Frozen Phase 3 spatial comparison

The Phase 3 manifest was reused unchanged: 20 development images and 80 evaluation images. All four MLP states were scored using the same cached DINO features, category prompts, cleaning procedure, and fixed 0.54 threshold; PAMR was disabled. Across 240 category-image pairs on the evaluation set:

| MLP checkpoint | Cleaned macro category mIoU | Raw macro category mIoU | Cleaned AUROC | Cleaned pointing accuracy |
|---|---:|---:|---:|---:|
| Untrained initialization | 0.0000 | 0.0000 | 0.5462 | 0.2042 |
| Original 1,000-image pilot | 0.2210 | 0.2520 | 0.7893 | 0.4667 |
| Phase 2 corrected pilot | 0.2277 | 0.2648 | 0.7929 | 0.4750 |
| Full COCO, 100 epochs | **0.3059** | **0.2880** | **0.8231** | **0.5375** |

Full COCO training improved these measured spatial metrics over both pilot checkpoints on the frozen evaluation subset. The cleaned IoU remains modest and object-size dependent: for the full model, cleaned IoU was 0.3264 on large objects, 0.1663 on medium objects, and 0.0710 on small objects. This supports improved language-conditioned localization on this COCO benchmark, while leaving small-object quality and out-of-domain robot-scene performance unresolved.

## Checkpoints and reproduction

The final projection checkpoint is `final_projection.pt`; SHA-256: `eef3c4b9ab278d5cb6334518c69c411d035e4b1f7281973edc8cffd23cad4f7e`. `checkpoint_latest.pt` stores the epoch-100 model, Adam state, RNG state and configuration. Milestone checkpoints are saved every ten epochs. Large model/feature artifacts are retained locally and are not intended for Git publication.

The execution used the persistent `trackb_talk2dino_full_20261117` container from `openvla-talk2dino-siglip:20261116`. Full commands and environment metadata are in `logs/worker.log`, `logs/environment.json`, and `configuration.json`.

## Interpretation and limitations

The full-data training run completed successfully with the required caption coverage and deterministic collision-free batch construction. The 200-image retrieval results improved over both small pilot checkpoints, but the late-epoch validation loss trend and small validation size limit the conclusion. Spatial object localization is still being evaluated against the frozen Phase 3 manifest. No LIBERO policy rollouts or KV reuse experiments were run, and no end-to-end inference speed claim follows from these representation results.
