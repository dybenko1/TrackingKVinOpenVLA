# Phase 3 — Talk2DINO-SigLIP Spatial Segmentation Validation

**Status:** complete. **Worktree:** `/home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/worktrees/trackb-talk2dino-siglip`. **Branch:** `thesis/trackb-talk2dino-siglip`. **Reference commit:** `ca9fb6cc6264bef3c8447bec39424b2547523e9f`.

## Executive summary

We evaluated the three frozen MLP states on 80 held-out images and 240 image/category pairs. The manifest was fixed before viewing predictions: 128 cached images met the criteria, 100 were selected deterministically, 20 reserved as development, and 80 used for final evaluation (covering 65 distinct categories and 240 image/category pairs). No training or visual feature extraction was run.

The trained projections produce useful ranking signal for some large/common categories, but the result is not yet strong general open-vocabulary segmentation: at the inherited 0.54 sigmoid-map threshold, raw grid IoU is modest, small objects are poorly localized, and the upstream attention cleaner reduces the principal raw-IoU metric. The corrected Phase 2 checkpoint is only slightly ahead of the original checkpoint; paired image bootstrap intervals mostly include zero.

## Method and upstream fidelity

Reference source was the local Talk2DINO checkout at `6dac47dbe3500df63f3b3df20dd28334290919cc`. We reproduced category cosine scores from normalized final DINO patch descriptors and normalized projected text, the upstream `sigmoid(similarity)` soft-mask path, `similarity_assignment_weighted()` with `lambda_bg=0.2`, and optional upstream PAMR (10 iterations, dilations `[1,2,4,8,12,24]`). The weighted-cleaning implementation was compared against the original method body on identical synthetic tensors: maximum absolute difference was 0.0.

This uses the upstream seven `sub_imagenet_template` prompts per category, encoded by the paired SigLIP-384 text tower; template embeddings are averaged before projection. The upstream COCO-object config uses `bg_thresh=0.54`; we preserved it as a reference setting but it is not calibrated for this SigLIP substitution. We did not tune it on this evaluation set. Raw-map AUROC is included as a threshold-independent diagnostic.

Deviations/limits: original Talk2DINO uses CLIP text and its COCO segmentation config uses 448-pixel inputs; this adaptation uses SigLIP-384 text and cached 224-pixel Talk2DINO features, yielding a 16x16 grid. Ground-truth category names are supplied per image to evaluate category-conditioned localization; this is not category discovery or individual-instance segmentation. All same-class instances are unioned, including `iscrowd` RLE regions.

## Aggregate metrics

| MLP state | Map | mIoU (macro by category) | Pair IoU | Precision | Recall | Pointing | AUROC | 224 IoU | Union IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| untrained | raw | 0.000 | 0.000 | 0.000 | 0.000 | 0.196 | 0.545 | 0.000 | 0.000 |
| untrained | cleaned | 0.000 | 0.000 | 0.000 | 0.000 | 0.204 | 0.546 | 0.000 | 0.000 |
| untrained | pamr | 0.000 | 0.000 | 0.000 | 0.000 | 0.229 | 0.563 | 0.000 | — |
| original_pilot | raw | 0.252 | 0.193 | 0.289 | 0.398 | 0.475 | 0.789 | 0.192 | 0.366 |
| original_pilot | cleaned | 0.221 | 0.159 | 0.255 | 0.274 | 0.467 | 0.789 | 0.156 | 0.288 |
| original_pilot | pamr | 0.222 | 0.158 | 0.232 | 0.235 | 0.567 | 0.825 | 0.157 | — |
| phase2_corrected | raw | 0.265 | 0.200 | 0.303 | 0.396 | 0.475 | 0.792 | 0.199 | 0.383 |
| phase2_corrected | cleaned | 0.228 | 0.161 | 0.252 | 0.261 | 0.475 | 0.793 | 0.160 | 0.297 |
| phase2_corrected | pamr | 0.241 | 0.165 | 0.245 | 0.230 | 0.537 | 0.826 | 0.164 | — |

mIoU is an unweighted mean over category-level IoUs; Pair IoU is the mean across image/category pairs. PAMR acts on the cleaned maps. Pointing accuracy is the fraction of category/image pairs whose highest-scoring location is inside the target. Binary metrics use the same fixed `sigmoid(cosine) >= 0.54` threshold.

### Checkpoint comparison

| Phase 2 corrected minus original pilot | Mean paired image IoU difference | Paired bootstrap 95% interval |
|---|---:|---:|
| raw | +0.007 | [-0.002, +0.015] |
| cleaned | +0.005 | [-0.002, +0.012] |
| pamr | +0.009 | [+0.001, +0.018] |

The corrected checkpoint has modest positive differences. Only the PAMR paired interval excludes zero in this descriptive bootstrap; this small selected sample and threshold-dependent evaluation do not establish a robust improvement. Both trained states substantially exceed the untrained initialization on raw AUROC and thresholded IoU. The untrained model produces effectively empty maps at the fixed threshold (zero IoU), so its pointing/AUROC values should not be read as meaningful segmentation.

### Object size

| Checkpoint | Method | Small mIoU (2–3 grid cells) | Medium (4–15) | Large (≥16) |
|---|---|---:|---:|---:|
| original_pilot | raw | 0.015 | 0.112 | 0.293 |
| original_pilot | cleaned | 0.016 | 0.090 | 0.241 |
| original_pilot | pamr | 0.032 | 0.084 | 0.237 |
| phase2_corrected | raw | 0.019 | 0.117 | 0.302 |
| phase2_corrected | cleaned | 0.018 | 0.090 | 0.245 |
| phase2_corrected | pamr | 0.039 | 0.090 | 0.245 |

Small objects remain difficult even when occupying at least two 16x16 cells. Phase 2 raw mIoU is approximately 0.019 for this group versus 0.302 for large targets. This is consistent with coarse spatial sampling and score-map upsampling smoothing boundaries.

## Effect of background cleaning and PAMR

At the fixed reference threshold, weighted background cleaning decreases raw mIoU from 0.252 to 0.221 for the original pilot, and from 0.265 to 0.228 for Phase 2. It also lowers union IoU (raw 0.383, cleaned 0.297). Thus the faithful cleaner is implemented, but under this uncalibrated SigLIP threshold it does not improve hard mask overlap.

PAMR applied after cleaning raises Phase 2 pointing accuracy from 0.475 to 0.537 and AUROC from 0.793 to 0.826; Phase 2 pair IoU changes only from 0.161 to 0.165. This is mixed evidence, not a general boundary-quality win.

## Qualitative examples

Twelve panels compare RGB, category-union ground truth, raw cosine maps, cleaned threshold masks, and PAMR for all three MLP states. They include both strong and weak examples; filenames and category lists are in `qualitative_results/qualitative_manifest.json`.

- `comparison_08_image_279521.png` includes a strong bus example: the trained scores broadly localize the bus category, although thresholded/cleaned maps do not reproduce its precise boundary.
- `comparison_01_image_35891.png` includes weaker person/traffic-light/handbag regions with partial or misplaced activation.

These are representative panels from a deterministic category-diverse selection, not cherry-picked solely by prediction quality.

## Annotation and engineering validation

The official `annotations_trainval2014.zip` was already present; no download occurred. `instances_val2014.json` was extracted to the ignored COCO cache. The selected 100 images contain 915 polygon annotations and 6 uncompressed crowd RLE annotations. The six selected RLE masks were checked against encoded run lengths and image dimensions. Their COCO `area` metadata differed from decoded run area in all six crowd cases and was not used as pixel ground truth. Compressed RLE decoding is covered by synthetic tests. Crowd regions are included in semantic unions, so these are not COCO instance-AP metrics.

Seven CPU/synthetic unit tests pass. Mask transforms exactly match installed torchvision PIL `Resize(short edge=224, nearest)` plus `CenterCrop(224)` on five aspect ratios; center-crop uses rounded offsets, including odd margins. Weighted cleaning matched upstream exactly on synthetic inputs. All 16 attention maps are normalized over the 256 patch positions.

## Compute and scope

The frozen SigLIP text tower was strictly loaded without its visual tower. The evaluation used the existing DINO patch and attention cache; no new DINO forward or feature extraction ran. Recorded device `cuda:0`, CUDA peak allocated 2.08 GB, reserved 2.35 GB, process RSS peak 5.71 GB. The 206.6s figure includes text-tower loading, encoding, scoring, PAMR and artifact generation; it is not a policy latency or isolated segmentation latency benchmark.

No model training, OpenVLA policy modification, or LIBERO rollout occurred. Existing pilot checkpoints and prior phase outputs were read-only.

## Answers and recommendation

1. **Trained vs initialization:** Both trained MLP checkpoints improve raw patch ranking and IoU over initialization. The fixed threshold makes the initialization masks empty.
2. **Different categories:** The category-conditioned maps vary by text class and some large objects are localized, but localization is inconsistent; this evaluation is not an instance separation test.
3. **Background cleaning:** Exact upstream weighting is reproduced, but hard IoU falls at the inherited threshold.
4. **PAMR:** It improves some pointing/AUROC results and slightly changes IoU; gains are mixed and not consistent across target sizes.
5. **Corrected vs original:** Phase 2 is modestly better descriptively; only the PAMR paired bootstrap is positive at the 95% interval in this sample.
6. **Resolution:** 16x16 features substantially limit small-object localization and boundaries.
7. **Multiple objects:** Evidence supports some category-specific localization on selected images, but not reliable multi-object segmentation across classes.
8. **Next step:** Do not scale directly to full COCO training yet. First investigate why the fixed CLIP-derived threshold does not calibrate to SigLIP and evaluate threshold-independent/localization behavior on a larger, predeclared held-out set; any calibration should use development data only. This is an evaluation-method issue, not justification to tune on the final set.

## Reproduction

Run in the existing `openvla-talk2dino-siglip:20261116` container with cached annotations, features and checkpoints mounted; no downloads are required. From the worktree root:

```bash
python3 -m unittest experiments.kv_cache.experiments.trackb_talk2dino_siglip.tests.test_phase3_spatial -v
python3 experiments/kv_cache/experiments/trackb_talk2dino_siglip/phase3_evaluate.py --prepare-manifest
python3 experiments/kv_cache/experiments/trackb_talk2dino_siglip/phase3_evaluate.py --evaluate --device cuda:0 --pamr
python3 experiments/kv_cache/experiments/trackb_talk2dino_siglip/render_phase3_gallery.py
```

Outputs: `summary.json`, `segmentation_metrics.json`, `per_category_metrics.csv`, `checkpoint_comparison.json`, `configuration.json`, `evaluation_manifest.json`, `implementation_equivalence_tests.json`, `resource_metrics.json`, `instance_annotations_audit.json`, `qualitative_results/`, and `logs/`.
