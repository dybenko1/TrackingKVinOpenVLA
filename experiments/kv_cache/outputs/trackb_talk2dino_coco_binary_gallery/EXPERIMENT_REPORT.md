# Original Talk2DINO COCO binary segmentation gallery

This report evaluates the frozen original pretrained Talk2DINO model on 30 COCO Object val2017 images. Selection used ground-truth category area only; no model scores were consulted. Each image has one independent category query and the category mask is the union of every pixel annotated with that semantic class.

## Protocol

- Original Meta DINOv2 ViT-L/14-register, original CLIP ViT-B/16, original Talk2DINO projection MLP, original `sub_imagenet_template` prompt ensemble, and original attention-based cleaning (`lambda_bg=0.2`). All checkpoint hashes and source revisions are recorded in `configuration.json`.
- One square 448×448 DINOText input per image; the score map is aligned back to the source image size for pixel scoring. RAW is `sigmoid(cosine)`; CLEAN is the upstream cleaned soft mask. Both use the fixed 0.54 probability threshold. PAMR is disabled.
- Size bins use the standard COCO area cutoffs (<1,024, 1,024–9,215, ≥9,216 pixels), applied to the semantic category-union mask. The downloaded val2017 material contains semantic maps, not per-instance polygons; therefore these are category-union area strata, not verified individual-instance sizes.
- Pixel-level metrics use the original resolution semantic labels. Patch metrics use the model's native 32×32 DINO grid at 448 input: GT is first nearest-neighbor warped with the same 448 square geometry, then area pooled. Binary patch GT means any target occupancy in the 14×14 cell; fractional occupancy-weighted patch metrics are also included.
- This is a 30-image visual diagnostic, not the complete COCO Object benchmark and not directly comparable to the prior 5,000-image 81-class mIoU. The earlier full benchmark uses sliding-window inference and multiclass semantic assignment; this gallery uses fixed single-scale category-wise binary maps.

## Aggregated metrics

### All selected images

Pooled pixel metrics (counts pooled before ratios):

| Group | N | Method | Precision | Recall | IoU | FG coverage | BG coverage |
|---|---:|---|---:|---:|---:|---:|---:|
| all | 30 | RAW | 0.244 | 0.795 | 0.229 | 14.2% | 85.8% |
| all | 30 | CLEAN | 0.314 | 0.726 | 0.281 | 10.1% | 89.9% |

Binary 32×32 patch metrics are available per image in `per_image_metrics.csv`; these use any-positive-occupancy token labels. Fractional occupancy-weighted token-grid scores are in the same CSV.

### By target size

Pooled pixel metrics (counts pooled before ratios):

| Group | N | Method | Precision | Recall | IoU | FG coverage | BG coverage |
|---|---:|---|---:|---:|---:|---:|---:|
| large | 10 | RAW | 0.323 | 0.805 | 0.299 | 29.7% | 70.3% |
| large | 10 | CLEAN | 0.379 | 0.739 | 0.334 | 23.3% | 76.7% |
| medium | 10 | RAW | 0.078 | 0.720 | 0.075 | 10.0% | 90.0% |
| medium | 10 | CLEAN | 0.119 | 0.618 | 0.111 | 5.5% | 94.5% |
| small | 10 | RAW | 0.036 | 0.617 | 0.036 | 3.0% | 97.0% |
| small | 10 | CLEAN | 0.063 | 0.537 | 0.059 | 1.5% | 98.5% |

Binary 32×32 patch metrics are available per image in `per_image_metrics.csv`; these use any-positive-occupancy token labels. Fractional occupancy-weighted token-grid scores are in the same CSV.

### By category

Pooled pixel metrics (counts pooled before ratios):

| Group | N | Method | Precision | Recall | IoU | FG coverage | BG coverage |
|---|---:|---|---:|---:|---:|---:|---:|
| bicycle | 3 | RAW | 0.393 | 0.794 | 0.356 | 13.3% | 86.7% |
| bicycle | 3 | CLEAN | 0.516 | 0.731 | 0.434 | 9.5% | 90.5% |
| bottle | 3 | RAW | 0.329 | 0.926 | 0.320 | 4.1% | 95.9% |
| bottle | 3 | CLEAN | 0.547 | 0.888 | 0.512 | 2.4% | 97.6% |
| bowl | 3 | RAW | 0.565 | 0.571 | 0.397 | 15.9% | 84.1% |
| bowl | 3 | CLEAN | 0.781 | 0.434 | 0.386 | 8.9% | 91.1% |
| bus | 3 | RAW | 0.766 | 0.932 | 0.726 | 10.2% | 89.8% |
| bus | 3 | CLEAN | 0.850 | 0.910 | 0.784 | 9.0% | 91.0% |
| car | 3 | RAW | 0.257 | 0.968 | 0.255 | 13.3% | 86.7% |
| car | 3 | CLEAN | 0.304 | 0.937 | 0.298 | 10.8% | 89.2% |
| cup | 3 | RAW | 0.266 | 0.917 | 0.260 | 7.5% | 92.5% |
| cup | 3 | CLEAN | 0.487 | 0.846 | 0.448 | 3.8% | 96.2% |
| fork | 3 | RAW | 0.049 | 1.000 | 0.049 | 30.2% | 69.8% |
| fork | 3 | CLEAN | 0.068 | 1.000 | 0.068 | 21.4% | 78.6% |
| knife | 3 | RAW | 0.186 | 0.939 | 0.184 | 11.9% | 88.1% |
| knife | 3 | CLEAN | 0.242 | 0.902 | 0.236 | 8.7% | 91.3% |
| spoon | 3 | RAW | 0.064 | 0.861 | 0.063 | 24.1% | 75.9% |
| spoon | 3 | CLEAN | 0.087 | 0.855 | 0.086 | 17.5% | 82.5% |
| wine glass | 3 | RAW | 0.168 | 0.982 | 0.167 | 11.8% | 88.2% |
| wine glass | 3 | CLEAN | 0.215 | 0.965 | 0.213 | 9.0% | 91.0% |

Binary 32×32 patch metrics are available per image in `per_image_metrics.csv`; these use any-positive-occupancy token labels. Fractional occupancy-weighted token-grid scores are in the same CSV.

## Failure analysis

- **Small-object localization:** pooled pixel IoU is 0.036 RAW and 0.059 CLEAN, compared with 0.299/0.334 on the large-target stratum. The small-bin results are weak despite the fixed threshold.
- **Complete misses:** 3 of 10 small-target examples have zero pixel recall for both RAW and CLEAN at 0.54 (000000118405 bowl, 000000266981 bus, 000000572620 bottle).
- **Foreground spill:** small-bin precision is 0.036 RAW and 0.063 CLEAN. CLEAN improves precision, but does not eliminate false-positive foreground.
- **Cleaning trade-off:** across all selected images, CLEAN changes pooled IoU from 0.229 to 0.281 and precision from 0.244 to 0.314, while recall falls from 0.795 to 0.726. It reduces predicted foreground coverage from 14.2% to 10.1%; the change is a precision/recall trade-off, not a uniform improvement.
- **Disconnected regions:** predicted connected-component counts have median/max 10/34 RAW and 6/24 CLEAN. Multiple annotated instances may form separate valid components; this count alone does not prove fragmentation.
- **Semantic confusions:** this binary evaluation counts every non-target pixel as false-positive foreground; it does not identify the specific competing category. Individual panels show spill visually, but no confusion matrix between non-target semantic classes is reported.

## What the gallery can establish

Visual inspection should focus on whether category-specific RAW/CLEAN predictions cover the full annotated category union, whether activation spills onto semantically related objects, and how often small regions vanish on the 32×32 grid. Pixel metrics are based on the upsampled patch score field, so they do not imply pixel-level boundary modeling.

Potential failure modes are quantified through false-positive/false-negative counts, prediction connected-component counts, and recall in each size stratum. A disconnected mask can be a legitimate result for a category with multiple instances; it is not automatically an error.

The prior LIBERO individual-object study used cached 224×224 RGB images and 16×16 patch-aligned role masks, while this COCO gallery uses 448×448 inputs, 32×32 native patch masks, full-resolution semantic category unions, and a different 30-image sample. Any observed difference is descriptive and cannot be attributed to domain shift alone.

## Gallery

### Small target regions

#### Image 000000118405 — bowl (321 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000118405_bowl.png)

#### Image 000000572620 — bottle (192 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000572620_bottle.png)

#### Image 000000091500 — cup (896 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000091500_cup.png)

#### Image 000000145597 — wine glass (243 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000145597_wine_glass.png)

#### Image 000000162092 — fork (852 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000162092_fork.png)

#### Image 000000153527 — knife (460 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000153527_knife.png)

#### Image 000000376322 — spoon (265 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000376322_spoon.png)

#### Image 000000099114 — bicycle (581 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000099114_bicycle.png)

#### Image 000000266981 — bus (683 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000266981_bus.png)

#### Image 000000424975 — car (391 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000424975_car.png)

### Medium target regions

#### Image 000000488673 — bowl (1,615 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000488673_bowl.png)

#### Image 000000324258 — bottle (1,303 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000324258_bottle.png)

#### Image 000000080932 — cup (8,719 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000080932_cup.png)

#### Image 000000404678 — wine glass (2,459 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000404678_wine_glass.png)

#### Image 000000054605 — fork (4,840 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000054605_fork.png)

#### Image 000000337498 — knife (4,890 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000337498_knife.png)

#### Image 000000009590 — spoon (2,151 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000009590_spoon.png)

#### Image 000000139099 — bicycle (1,375 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000139099_bicycle.png)

#### Image 000000204871 — bus (5,465 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000204871_bus.png)

#### Image 000000176037 — car (1,380 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000176037_car.png)

### Large target regions

#### Image 000000527695 — bowl (132,340 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000527695_bowl.png)

#### Image 000000520077 — bottle (10,692 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000520077_bottle.png)

#### Image 000000163611 — cup (10,318 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000163611_cup.png)

#### Image 000000312192 — wine glass (15,637 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000312192_wine_glass.png)

#### Image 000000414638 — fork (10,849 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000414638_fork.png)

#### Image 000000250766 — knife (9,645 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000250766_knife.png)

#### Image 000000562059 — spoon (12,431 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000562059_spoon.png)

#### Image 000000508370 — bicycle (52,907 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000508370_bicycle.png)

#### Image 000000355817 — bus (71,546 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000355817_bus.png)

#### Image 000000423617 — car (29,795 semantic-union pixels)
![RAW and CLEAN binary segmentation comparison](figures/coco_000000423617_car.png)

## Provenance

- Selected 30 unique image IDs from 5000 val2017 images, exactly 10 per size group.
- Model identity: host cached Talk2DINO source revision `6dac47dbe3500df63f3b3df20dd28334290919cc` and DINO source `cached-local-repository`. The mounted inference container lacked usable Git metadata, so exact byte-for-byte identity of its upstream checkout could not be independently confirmed.
- Exact image, annotation, selected category, target size, and source SHA-256 for every panel: `gallery_manifest.json`.
- Full per-image pixel, binary patch, fractional patch, and size/category metrics: `per_image_metrics.csv`.
