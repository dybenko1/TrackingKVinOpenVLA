# Talk2DINO-SigLIP Controlled Pilot — 1,000 COCO Images

## Outcome

The controlled pilot completed its extraction, integrity gate, 10-epoch projection-MLP training and held-out visual diagnostics. It processed **800 train images / 4002 captions** and **200 validation images / 1000 captions**. Feature validation: **passed**. MLP training: **complete**, 10 epochs. Qualitative panels: **8 held-out validation images**, two captions per image.

This is a pipeline smoke pilot, not a final COCO training run or evidence of segmentation quality.

## Sampling and reproducibility

Images were sampled uniformly without replacement from sorted unique COCO image IDs using Python `random.Random(123)`: 800 from train2014 and 200 from val2014. All captions for each sampled image were retained in annotation-file order. Train/validation image IDs are disjoint. The published `sample_ids.json` records the selected image and annotation IDs; the full local manifest, including caption text and local paths, is not published.

## Feature extraction

- DINO visual branch: existing OpenVLA fine-tuned `vit_large_patch14_reg4_dinov2.lvd142m`; Talk2DINO bicubic 224 resize/center crop and ImageNet normalization. Existing shared extractor produced final normalized patches `[N,256,1024]`, CLS-to-patch maps `[N,16,256]`, and Talk2DINO attention-weighted mean regions `[N,16,1024]`.
- Text: frozen `timm/ViT-SO400M-14-SigLIP-384`, revision `ac16108d567c4389e6cd2b11c9b8585f7474435b`; OpenCLIP tokenizer configured with `canonicalize`, context 64, output 1152. One vector per caption annotation.
- All pilot arrays were stored as float32. Visual features are image-level and shared across captions; text vectors are annotation-level. Files and SHA-256 checksums are listed in `feature_validation.json`.
- Selected captions exceeding context 64: 0 / 5002.
- Encoder-state hashes matched before and after extraction. Re-extraction was exact for the sampled validation subset. Standard OpenVLA fused features were exactly unchanged across the sidecar extraction check.

## Validation and training

Feature validation passed: all selected images/captions were processed; image-caption joins and unique IDs were checked; train/validation image IDs were disjoint; dimensions and float32 dtype matched; all values were finite; attention normalized over patches (max absolute sum error: train 2.4e-07, val 2.4e-07); region descriptors were nonzero; saved arrays reloaded and checksummed.

The existing `RegionTextProjection` was trained for ten epochs, with Adam (`lr=1e-4`), batch 128, seed 123, fixed temperature 0.07 and canonical bidirectional InfoNCE. Only the `1152→1024→1024` MLP was optimized. Repeated image IDs in batches were counted as potential diagonal-label false negatives. Per-epoch train/validation losses and duplicate counts are in `training_history.json`; initial and final MLP states are under `pilot_mlp_checkpoint/`.

Training mean loss (epoch 1 → 10): 3.6674 → 0.7800; final mean validation loss: 2.6273. Potential repeated-image rows logged across epochs: train 2519, validation 7010. These are pilot optimization diagnostics, not final model-quality estimates.

## Qualitative findings and limits

`qualitative_results/` contains eight held-out validation examples, each with two caption queries, the 16 raw DINO CLS-attention maps, trained and initial-MLP selected-head maps, patch cosine scores, and attention-guided overlays. These maps are exploratory attention/similarity diagnostics, **not validated segmentation masks**. This pilot does not implement or validate the full background-cleaning/mask-generation stages as a segmentation result.

No oracle labels were used for sampling, feature selection, training or visualization. No policy actions, LIBERO rollouts, KV caching, full COCO extraction or 100-epoch training were run. Standard OpenVLA inference behavior and pretrained weights were left unchanged.

## Performance and resources

- DINO model load + initial state-hash time: 12.50 s.
- DINO model-forward time per image: mean 29.82 ms, median 27.67 ms, P95 33.36 ms.
- DINO per-image extraction pipeline (including image open/preprocess/call): mean 43.64 ms; total measured image-call time 43.64 s (22.92 images/s).
- DINO stage total, including load and validation: 64.42 s.
- SigLIP text: 5,002 captions encoded in 89.47 s of encoder time (55.91 captions/s; 17.89 ms/caption averaged over batches); tokenization took 2.89 s. **This timing came from the full-set saved-feature verification pass**; the initial feature-writer timing was not persisted, so it does not represent the original write-pass timing. SigLIP stage total, including load and verification, was 111.72 s.
- Training wall time: 13.53 s.
- Peak CUDA allocated/reserved: DINO 1.40/1.44 GiB; SigLIP text 1.91/2.07 GiB; training 77.2/106.0 MiB.
- Peak process RSS: DINO 4.05 GiB; SigLIP text 3.90 GiB; training 1.03 GiB. These stages ran sequentially.
- Feature arrays occupy 1.07 GiB; free disk after feature creation was 832.3 GiB.
- GPU utilization was not measured because `tegrastats` was unavailable in the isolated container.

Extraction, verification and training timing scopes are separate. The first image call includes warm-up effects; no end-to-end inference speedup is claimed.

## Execution notes

The first detached attempt stopped before feature extraction because the pilot feature directory was not initialized; this was fixed before retry. The next attempt completed and persisted all features, then correctly blocked training when a 4-caption repeat check differed slightly from the original 64-row text batch. The deterministic check now re-encodes with the same batch shape and compares against the saved rows; the numerical tolerance was not loosened. The feature validator also caught and corrected a structured caption-ID indexing error before training, after which the feature gate passed. The retry reused validated visual features and verified completed text arrays before training. Training completed; the first qualitative finalization attempt caught NumPy arrays being included in JSON metadata. The figures were produced, the metadata serializer was corrected to retain only scalar diagnostics, and the worker resumed without repeating extraction or training. Full attempt logs are preserved in `logs/pilot.log`.

## Recommendation

The pipeline is ready to proceed to a separately authorized larger COCO extraction run, subject to reviewing the qualitative maps and training curves first. This 1,000-image pilot validates data plumbing and optimization behavior only; it does not establish foreground segmentation quality or generalization.
