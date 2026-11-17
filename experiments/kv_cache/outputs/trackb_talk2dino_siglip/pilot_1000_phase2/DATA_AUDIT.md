# Phase 2 data audit

## Split summary

| Split | Unique images | Caption annotations | Captions/image (min/median/mean/max) | Exact duplicate excess rows | Tukey outlier images (few/many) | Feature mapping |
|---|---:|---:|---:|---:|---:|---|
| Train | 800 | 4002 | 5 / 5.0 / 5.003 / 6 | 6 | 0 / 2 | passed |
| Validation | 200 | 1000 | 5 / 5.0 / 5.000 / 5 | 2 | 0 / 0 | passed |

Image IDs are disjoint: **True** (overlap 0). Exact feature row joins and ordering were checked against the pilot manifest for every annotation and image.

## Repeated caption counter interpretation

The original trainer's `repeated_image_id_count` computes `sum(n_image_in_batch - 1)` for image IDs occurring more than once within each batch. It counts surplus rows after the first row per image in that batch. It is not the number of all positive/negative pair collisions, and it does not enumerate individual false-negative pairs. The old validation counter used the first seven sequential batches only because `drop_last=True`; it is not a count over all 1,000 validation captions.

## Original pilot sampling and dropped rows

Original Talk2DINO `COCOCaptions` creates one sample per caption annotation and filters by image filename/split; it does not group captions by image. `do_train` uses a shuffled DataLoader, the loss uses diagonal positives, and validation is sequential with the same diagonal loss. DataLoader's default keeps partial batches upstream, but our first pilot explicitly used `drop_last=True` in train and validation. Thus the first pilot trained on 31×128 = 3,968 of 4,002 train captions per epoch (34 omitted) and validated on 7×128 = 896 of 1,000 captions (104 omitted). The upstream InfoNCE loss also divides the already mean-reduced symmetric CE by B², shrinking magnitudes without removing false negatives; our baseline used canonical loss scaling as documented in its config.

## Caption diversity examples

The sampled panels and captions span varied entities and scenes (e.g. transport and garages, household items, animals, and people), but this 800/200 subset remains small relative to COCO. Image IDs below are selected representative caption examples; see the existing eight-image pilot overview for the actual RGB scenes.

### Train

- Image `101891` (COCO_train2014_000000101891.jpg, 5 captions): “Bicycles, cars and a trash can in a garage.” / “Clutter blocks antique cars from leaving the garage” / “A garage filled with old cards and old bicycles.” / “Some antique cars sit in a garage with some bicycles” / “A garage with bicycles, cars and a garbage can.”
- Image `397832` (COCO_train2014_000000397832.jpg, 6 captions): “A giraffe standing in an exhibit by a rock wall.” / “A giraffe standing in a zoo exhibit by a rock wall.” / “A giraffe is standing behind a stone wall.  ” / “A small giraffe standing next to a stone wall.” / “a giraffe is standing near a large rock” / “A single giraffe standing inside a zoo enclosure.”
- Image `179904` (COCO_train2014_000000179904.jpg, 5 captions): “A man with bunches of bananas stands next to them.” / “Man with bananas still on stem in outdoors area” / “A man with a large cart filled with bunches of bananas.” / “A man holds up a branch of bananas he is trimming with a machete” / “A man is next to a large bunch of bananas.”
- Image `381390` (COCO_train2014_000000381390.jpg, 5 captions): “a street sign next to a traffic light with a sky background” / “Traffic light and street light for Belmont Avenue” / “A traffic light and sign show travelers where to go.” / “A photo of a street lamp with a street light and a street sign.” / “A stoplight and street sign attached to a street lamp pole.”

### Validation

- Image `410627` (COCO_val2014_000000410627.jpg, 5 captions): “The young man is stirring his pot of food with a wooden spoon.” / “A man enjoys cooking food in a pan” / “A smiling man is cooking in his kitchen.” / “A man smiles as he stirs his food in the pot.” / “A boy smiles while holding a pot and spoon.”
- Image `353435` (COCO_val2014_000000353435.jpg, 5 captions): “four people sitting at a table eating and laughing” / “The people are at a table eating breakfast. ” / “a couple of people are sitting around a table” / “A group of people are smiling at a table.” / “Four smiling people are sitting around a dining table.”
- Image `185865` (COCO_val2014_000000185865.jpg, 5 captions): “A sequential shot of a tennis player slamming a ball.” / “Three shots of a man swinging at a tennis ball” / “A woman standing on a tennis court holding a racquet.” / “A time lapse image of a tennis player in action.” / “a split photo of a person hitting a tennis ball ”
- Image `391300` (COCO_val2014_000000391300.jpg, 5 captions): “The tailgate of a car open and two suitcases inside.” / “A couple of pieces of brown luggage sitting in a backseat.” / “Two suitcases in the back of a vehicle.” / “A car trunk with two pieces of luggage in it.” / “A green car with vintage luggage in the back of it.”

## Spatial labels

Local instances annotations found: train=False, val=False. Only caption annotation JSONs are present. A quantitative COCO spatial evaluation would require obtaining the official instance annotation archives; no such download or feature extraction was started.
