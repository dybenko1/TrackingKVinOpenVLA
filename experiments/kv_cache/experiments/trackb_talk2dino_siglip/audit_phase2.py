#!/usr/bin/env python3
"""Audit sampled COCO captions and annotation-to-feature joins (CPU only)."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def quantile(values: list[int], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def audit_split(name: str, manifest_split: dict, feature_root: Path, dataset_root: Path) -> dict:
    captions = manifest_split["captions"]
    images = manifest_split["images"]
    counts = Counter(int(row["image_id"]) for row in captions)
    values = list(counts.values())
    exact_text_counts = Counter(str(row["caption"]) for row in captions)
    duplicate_text_excess = sum(count - 1 for count in exact_text_counts.values() if count > 1)
    duplicate_text_records = sum(count > 1 for count in exact_text_counts.values())
    same_image_counts = Counter()
    text_to_ids: dict[str, set[int]] = defaultdict(set)
    for row in captions:
        text = str(row["caption"])
        image_id = int(row["image_id"])
        text_to_ids[text].add(image_id)
        same_image_counts[(text, image_id)] += 1
    same_image_exact_duplicates = sum(count - 1 for count in same_image_counts.values() if count > 1)

    tag = "train2014" if name == "train" else "val2014"
    image_ids = np.load(feature_root / f"image_ids_{tag}.npy", mmap_mode="r")
    caption_ids = np.load(feature_root / f"caption_ids_{tag}.npy", mmap_mode="r")
    region_features = np.load(feature_root / f"region_features_{tag}.npy", mmap_mode="r")
    text_features = np.load(feature_root / f"text_features_{tag}.npy", mmap_mode="r")
    manifest_pairs = [(int(row["image_id"]), str(row["file_name"])) for row in images]
    manifest_id_order = [image_id for image_id, _ in manifest_pairs]
    annotation_pairs = [(int(row["annotation_id"]), int(row["image_id"])) for row in captions]
    feature_caption_pairs = [(int(row[0]), int(row[1])) for row in caption_ids]
    mapping_ok = (
        image_ids.tolist() == manifest_id_order
        and feature_caption_pairs == annotation_pairs
        and len(region_features) == len(image_ids)
        and len(text_features) == len(caption_ids)
    )
    if not mapping_ok:
        raise AssertionError(f"Cached image/caption feature mapping failed for {name}.")

    ordered_counts = sorted(values)
    min_count, max_count = ordered_counts[0], ordered_counts[-1]
    q1, q3 = quantile(ordered_counts, 0.25), quantile(ordered_counts, 0.75)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    unusual_low = sorted((image_id, count) for image_id, count in counts.items() if count < lower_fence)
    unusual_high = sorted((image_id, count) for image_id, count in counts.items() if count > upper_fence)
    lexical = Counter()
    for row in captions:
        lexical.update(re.findall(r"[a-z]+", str(row["caption"]).lower()))
    examples = []
    sorted_ids = sorted(counts)
    chosen_ids = [sorted(counts, key=lambda key: counts[key])[0],
                  sorted(counts, key=lambda key: counts[key])[-1],
                  sorted_ids[len(sorted_ids) // 3], sorted_ids[2 * len(sorted_ids) // 3]]
    chosen_ids = list(dict.fromkeys(chosen_ids))
    for image_id in chosen_ids:
        examples.append({
            "image_id": int(image_id),
            "file_name": next(file_name for item_id, file_name in manifest_pairs if item_id == image_id),
            "captions_for_image": [row["caption"] for row in captions if int(row["image_id"]) == image_id],
            "annotation_count": int(counts[image_id]),
        })

    return {
        "unique_images": len(images),
        "caption_annotations": len(captions),
        "captions_per_image": {
            "min": min_count, "max": max_count,
            "mean": statistics.mean(values), "median": statistics.median(values),
            "p25": q1, "p75": q3, "iqr": iqr,
            "count_distribution": {str(key): int(val) for key, val in sorted(Counter(values).items())},
            "outlier_rule": "Tukey 1.5×IQR fences over per-image annotation counts",
            "unusually_few_rule": f"caption count < {lower_fence}",
            "unusually_many_rule": f"caption count > {upper_fence}",
            "unusually_few_image_count": len(unusual_low),
            "unusually_many_image_count": len(unusual_high),
            "unusually_few_images": unusual_low[:20],
            "unusually_many_images": unusual_high[:20],
        },
        "exact_duplicate_caption_strings": {
            "unique_strings_repeated": duplicate_text_records,
            "excess_annotation_rows": duplicate_text_excess,
            "same_image_excess_rows": same_image_exact_duplicates,
            "cross_image_repeated_strings": sum(len(ids) > 1 for ids in text_to_ids.values()),
        },
        "caption_vocabulary": {
            "unique_lowercase_alpha_tokens": len(lexical),
            "top_30_tokens": lexical.most_common(30),
        },
        "feature_mapping": {
            "status": "passed", "image_id_order_matches_manifest": image_ids.tolist() == manifest_id_order,
            "annotation_image_pairs_match_manifest_in_order": feature_caption_pairs == annotation_pairs,
            "image_feature_rows": len(region_features), "caption_feature_rows": len(text_features),
        },
        "representative_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.pilot_output / "pilot_manifest.json").read_text())
    feature_root = args.pilot_output / "features"
    train = audit_split("train", manifest["splits"]["train"], feature_root, args.dataset_root)
    val = audit_split("val", manifest["splits"]["val"], feature_root, args.dataset_root)
    train_ids = {int(item["image_id"]) for item in manifest["splits"]["train"]["images"]}
    val_ids = {int(item["image_id"]) for item in manifest["splits"]["val"]["images"]}
    disjoint = train_ids.isdisjoint(val_ids)
    if not disjoint:
        raise AssertionError("Pilot train/validation image IDs overlap.")
    annotation_root = args.dataset_root / "annotations2014"
    instance_files = [annotation_root / "instances_train2014.json", annotation_root / "instances_val2014.json"]
    result = {
        "status": "passed", "seed": manifest["seed"],
        "sampled_train_validation_image_ids_disjoint": disjoint,
        "train_validation_overlap_count": len(train_ids & val_ids),
        "splits": {"train": train, "validation": val},
        "spatial_annotation_availability": {
            "instances_train2014_json": instance_files[0].is_file(),
            "instances_val2014_json": instance_files[1].is_file(),
            "available_annotation_files": sorted(path.name for path in annotation_root.glob("*.json")),
        },
        "interpretation": "This sampled subset has caption and image diversity but is not a replacement for broad COCO training. No COCO instance masks are available in the local dataset, so quantitative spatial evaluation requires additional annotations and is not run here.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# Phase 2 data audit", "", "## Split summary", "",
             "| Split | Unique images | Caption annotations | Captions/image (min/median/mean/max) | Exact duplicate excess rows | Tukey outlier images (few/many) | Feature mapping |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, data in (("Train", train), ("Validation", val)):
        cp = data["captions_per_image"]
        lines.append(f"| {name} | {data['unique_images']} | {data['caption_annotations']} | {cp['min']} / {cp['median']} / {cp['mean']:.3f} / {cp['max']} | {data['exact_duplicate_caption_strings']['excess_annotation_rows']} | {cp['unusually_few_image_count']} / {cp['unusually_many_image_count']} | passed |")
    lines += ["", f"Image IDs are disjoint: **{disjoint}** (overlap {len(train_ids & val_ids)}). Exact feature row joins and ordering were checked against the pilot manifest for every annotation and image.",
              "", "## Repeated caption counter interpretation", "",
              "The original trainer's `repeated_image_id_count` computes `sum(n_image_in_batch - 1)` for image IDs occurring more than once within each batch. It counts surplus rows after the first row per image in that batch. It is not the number of all positive/negative pair collisions, and it does not enumerate individual false-negative pairs. The old validation counter used the first seven sequential batches only because `drop_last=True`; it is not a count over all 1,000 validation captions.",
              "", "## Original pilot sampling and dropped rows", "",
              "Original Talk2DINO `COCOCaptions` creates one sample per caption annotation and filters by image filename/split; it does not group captions by image. `do_train` uses a shuffled DataLoader, the loss uses diagonal positives, and validation is sequential with the same diagonal loss. DataLoader's default keeps partial batches upstream, but our first pilot explicitly used `drop_last=True` in train and validation. Thus the first pilot trained on 31×128 = 3,968 of 4,002 train captions per epoch (34 omitted) and validated on 7×128 = 896 of 1,000 captions (104 omitted). The upstream InfoNCE loss also divides the already mean-reduced symmetric CE by B², shrinking magnitudes without removing false negatives; our baseline used canonical loss scaling as documented in its config.",
              "", "## Caption diversity examples", "",
              "The sampled panels and captions span varied entities and scenes (e.g. transport and garages, household items, animals, and people), but this 800/200 subset remains small relative to COCO. Image IDs below are selected representative caption examples; see the existing eight-image pilot overview for the actual RGB scenes.", ""]
    for split_name, data in (("Train", train), ("Validation", val)):
        lines += [f"### {split_name}", ""]
        for ex in data["representative_examples"]:
            lines.append(f"- Image `{ex['image_id']}` ({ex['file_name']}, {ex['annotation_count']} captions): " + " / ".join(f"“{text}”" for text in ex["captions_for_image"]))
        lines.append("")
    lines += ["## Spatial labels", "", f"Local instances annotations found: train={instance_files[0].is_file()}, val={instance_files[1].is_file()}. Only caption annotation JSONs are present. A quantitative COCO spatial evaluation would require obtaining the official instance annotation archives; no such download or feature extraction was started.", ""]
    (args.output_dir / "DATA_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps({"status": "passed", "train": [train["unique_images"], train["caption_annotations"]], "val": [val["unique_images"], val["caption_annotations"]], "disjoint": disjoint}, indent=2))


if __name__ == "__main__":
    main()
