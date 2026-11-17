# Phase 2 Validation Report — Talk2DINO-SigLIP Corrected Pilot

## 1. Executive summary

The 1,000-image pilot was retrained for 10 epochs with an epoch-based caption-round batch sampler. Every one of the 4002 training captions is used once per epoch, and no minibatch contains two captions from one image. This fixes the specific same-image diagonal-negative collision in training. The original pilot outputs/checkpoints remain unchanged.

Validation now uses the full global image × caption similarity matrix and treats all captions belonging to an image as positives for image-to-text retrieval. Baseline and corrected checkpoints were evaluated on the identical 200 validation images and 1000 captions using the existing max-over-16-region-head scoring function.

This is a small pilot and does not establish robust spatial segmentation or generalization.

## 2. Original Talk2DINO methodology and deviations

Upstream `COCOCaptions` creates one dataset row per caption annotation and does not group multiple captions from an image. `do_train` shuffles annotation rows; `ContrastiveLoss` uses diagonal image-caption labels. Upstream InfoNCE divides the already mean-reduced symmetric cross entropy by `batch_size**2`. Our prior implementation report records our adaptation's canonical scaling (without that extra division). The original pilot also used `drop_last=True`, omitting 34/4,002 training captions and 104/1,000 validation captions per epoch.

The corrected pilot preserves annotation-level examples, the diagonal symmetric InfoNCE loss, model, features, optimizer, LR, batch cap, seed and epoch count. It changes training batch construction to caption rounds and keeps partial batches. Validation changes to multi-positive retrieval because diagonal validation on repeated-image captions is not a valid retrieval metric. This is a deliberate correction, not attributed to upstream Talk2DINO.

## 3. Dataset diversity and mapping audit

See `DATA_AUDIT.md` and `data_audit.json`. Train: 800 unique images / 4002 captions; validation: 200 / 1000. Caption counts per image are train min/median/mean/max 5/5.0/5.00/6 and validation 5/5.0/5.00/5. Exact caption duplicates: train excess 6; validation excess 2. Feature ID/order joins passed; train/validation image IDs are disjoint. Caption-count distribution: train 798 images have 5 captions and 2 have 6; validation all 200 have 5. Using 1.5×IQR fences, train has 0 unusually few and 2 unusually many images; validation has 0/0. These results and representative varied scenes/captions are listed in `DATA_AUDIT.md`.

This random sample includes diverse captioned scenes, but only 800 training images constrain the MLP; the full COCO corpus is not represented. Local COCO instances annotation files are absent, so a mask-based spatial evaluation was not run. Obtaining the official instance annotation archives and mapping prompts to categories is the minimum added data step; no download was started.

## 4. Original pilot duplicate-counter diagnosis

`repeated_image_id_count` counts surplus rows after the first occurrence of each image within a minibatch (`sum(count - 1)`). It is not a count of all pairwise false-negative relations. The pilot logged 241–271 repeated-image rows per training epoch (mean 251.9; 2,519 over 10 epochs), therefore about 252 extra rows in its 31 full random batches share an image ID with another row in that same batch. It does not prove 251 distinct false negatives; a batch with three captions from one image contributes two rows but has multiple incorrect off-diagonal pairings. Validation's 701 counter was for only the first 896 sequential validation rows because `drop_last=True`; 104 captions were excluded.

## 5. Corrected sampler and tests

Per epoch, the sampler groups all annotation-row indices by image ID, independently shuffles each image's captions, selects at most one remaining caption from each image per round, shuffles each round, then chunks that round into batches of at most 128. Partial batches are retained. Tests verify once-only coverage, no same-image batch collisions, unequal group counts, partial batches, epoch rotation and deterministic replay. Results are in `sampler_test_status.json` and `sampler_tests.log`.

Across all corrected training epochs: every caption row was consumed exactly once; same-image repeated rows per batch were zero. Each epoch had 36 batches (6 partial batches). Train diagonal InfoNCE changed 3.5304 → 0.6939. These losses are optimization diagnostics, not the primary evaluation.

## 6. Baseline versus corrected multi-positive validation

Both models use the same global scorer: normalize visual regions and projected text, compute cosine similarity for each of 16 regions, and take the maximum region score. The baseline checkpoint was the original pilot's final MLP; the corrected MLP started from the exact saved original initial MLP weights (hashes are in each metrics JSON).

| Checkpoint | I→T R@1 | I→T R@5 | I→T R@10 | T→I R@1 | T→I R@5 | T→I R@10 | Multi-positive InfoNCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original pilot | 0.570 | 0.860 | 0.920 | 0.452 | 0.896 | 0.971 | 2.1996 |
| Corrected pilot | 0.550 | 0.880 | 0.935 | 0.460 | 0.887 | 0.965 | 2.2087 |

Mean/median retrieval ranks and complete score-matrix stats are in `baseline_multi_positive_validation.json` and `corrected_multi_positive_validation.json`. The old diagonal validation loss (2.6273) is historical and not directly comparable to the new multi-positive metric. **The corrected model did not consistently outperform the original:** its symmetric multi-positive InfoNCE is slightly higher (2.2087 vs 2.1996); image-to-text R@5/R@10 improve modestly while R@1 falls, and text-to-image metrics are mixed. This is descriptive on one 200-image validation split, with no significance claim.

## 7. Multi-positive metric definition

For image-to-text, the positive numerator is `logsumexp` over every caption whose image ID matches the query image. For text-to-image, its corresponding unique image is the positive. In each direction the diagnostic loss is `logsumexp(all candidates) - logsumexp(positives)`; the reported symmetric value averages directional means at temperature 0.07. Retrieval ranks count any corresponding caption as positive for I→T; T→I ranks the matching image.

## 8. Spatial evaluation and limitations

The current COCO dataset folder has `captions_train2014.json` and `captions_val2014.json`, but not `instances_train2014.json` or `instances_val2014.json`. Existing extracted features do contain 256 patch vectors and attention maps, but without object masks they cannot support defensible IoU, precision/recall or pointing-accuracy evaluation. The pilot's qualitative attention panels are not segmentation evidence. `qualitative_results/` contains baseline-versus-corrected patch-cosine and selected-head attention-guided maps for the same eight held-out images and two captions per image. The maps are exploratory diagnostics and are not validated segmentation masks. Existing saved patches are sufficient for a later activation-map trial; the missing requirement is ground-truth instance masks, not another DINO extraction.

## 9. Multi-positive InfoNCE follow-up (not run)

For each image query, a multi-positive objective replaces the single positive log-probability with `-log(sum_{j∈P(i)} exp(logit_ij) / sum_j exp(logit_ij))`; the symmetric text-to-image direction uses the matching image (or multiple image rows if an image table is duplicated). This can stop captions of the same image being negatives and may better fit the data. It also changes the original objective, can overweight images with many captions unless normalized carefully, and requires a fair separate run from the same initialization, sampler, seed, epochs and data. This training was not run; it remains an explicit next objective comparison, not an observed result.

## 10. Conclusions and next step

1. Same-image false negatives within training batches: **eliminated by construction**, tested, and zero in every corrected epoch.
2. Caption usage: **all 4,002 training annotations were consumed once each epoch**, including partial batches.
3. Validation: the multi-positive retrieval objective is appropriate for multiple captions per image and avoids treating sibling captions as negatives.
4. Corrected versus original: there is no consistent retrieval improvement. The corrected checkpoint is slightly worse on symmetric multi-positive InfoNCE and mixed across Recall@K; the sample is too small for significance claims.
5. Spatial alignment: this pilot does not establish it. Retrieval metrics and attention heatmaps do not prove segmentation quality.
6. Scale-up: a 1,000-image pilot is not sufficient evidence for the full experiment. First obtain COCO instance masks and run a small, fixed spatial benchmark on a held-out multi-object subset, then decide whether to expand.
7. Smallest uncertainty-reducing next experiment: evaluate the frozen corrected MLP on ~100 held-out images with available COCO instance masks, predeclared object phrases and IoU/pointing metrics, after acquiring the missing official annotation archive. Keep policy inference unchanged.

## 11. Reproducibility

Configuration: `phase2_config.json`; sample seed 123; model/features/checkpoint hashes in `run_metadata.json`; the console log is `logs/phase2.log`. Command is recorded in `run_metadata.json`. No COCO feature re-extraction, additional checkpoint download, full dataset training or LIBERO rollout was done.
