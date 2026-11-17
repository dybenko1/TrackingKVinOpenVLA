#!/usr/bin/env python3
"""Compare original and corrected pilot MLP maps on the same held-out images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase2_pilot import load_projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--phase2-output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from experiments.kv_cache.experiments.trackb_talk2dino_siglip.preprocessing import (
        IMAGENET_MEAN, IMAGENET_STD, preprocess_talk2dino_image,
    )

    pilot, phase2 = args.pilot_output, args.phase2_output
    gallery = phase2 / "qualitative_results"
    gallery.mkdir(parents=True, exist_ok=True)
    feature_root = pilot / "features"
    image_records = json.loads((pilot / "pilot_manifest.json").read_text())["splits"]["val"]["images"]
    captions = json.loads((pilot / "pilot_manifest.json").read_text())["splits"]["val"]["captions"]
    selected_manifest = json.loads((pilot / "qualitative_results/qualitative_manifest.json").read_text())
    selected_ids = [int(row["image_id"]) for row in selected_manifest["examples"]]
    records_by_id = {int(row["image_id"]): row for row in image_records}
    captions_by_id: dict[int, list[dict]] = {}
    for row in captions:
        captions_by_id.setdefault(int(row["image_id"]), []).append(row)
    image_ids = np.load(feature_root / "image_ids_val2014.npy", mmap_mode="r")
    caption_ids = np.load(feature_root / "caption_ids_val2014.npy", mmap_mode="r")
    regions = np.load(feature_root / "region_features_val2014.npy", mmap_mode="r")
    patches = np.load(feature_root / "patch_features_val2014.npy", mmap_mode="r")
    attention = np.load(feature_root / "attention_maps_val2014.npy", mmap_mode="r")
    text = np.load(feature_root / "text_features_val2014.npy", mmap_mode="r")
    image_row = {int(value): index for index, value in enumerate(image_ids)}
    caption_row = {int(value): index for index, value in enumerate(caption_ids[:, 0])}

    device = torch.device("cpu")
    baseline, _ = load_projection(pilot / "pilot_mlp_checkpoint/final_projection.pt", device)
    corrected, _ = load_projection(phase2 / "corrected_projection.pt", device)
    manifest = []
    panels = []
    for image_id in selected_ids:
        ir = image_row[image_id]
        image_record = records_by_id[image_id]
        image_path = args.dataset_root / image_record["relative_image_path"]
        image = Image.open(image_path).convert("RGB")
        pixels = preprocess_talk2dino_image(image).float()
        pixels = (pixels * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None]).clamp(0, 1)
        rgb = pixels.permute(1, 2, 0).numpy()
        region = torch.from_numpy(np.array(regions[ir], copy=True)).float().unsqueeze(0)
        patch = F.normalize(torch.from_numpy(np.array(patches[ir], copy=True)).float(), dim=-1)
        attn = np.array(attention[ir], copy=True)
        queries = captions_by_id[image_id][:2]
        calculated = []
        for query in queries:
            tr = torch.from_numpy(np.array(text[caption_row[int(query["annotation_id"])]], copy=True)).float().unsqueeze(0)
            with torch.inference_mode():
                base_score, base_head = baseline(region, tr, return_head_indices=True)
                corr_score, corr_head = corrected(region, tr, return_head_indices=True)
                base_text = F.normalize(baseline.text_projection(tr), dim=-1)[0]
                corr_text = F.normalize(corrected.text_projection(tr), dim=-1)[0]
                base_map = (patch @ base_text).reshape(16, 16).cpu().numpy()
                corr_map = (patch @ corr_text).reshape(16, 16).cpu().numpy()
            bh, ch = int(base_head[0, 0]), int(corr_head[0, 0])
            base_guided = base_map * attn[bh].reshape(16, 16)
            corr_guided = corr_map * attn[ch].reshape(16, 16)
            calculated.append({
                "caption": str(query["caption"]), "annotation_id": int(query["annotation_id"]),
                "baseline_head": bh, "corrected_head": ch,
                "baseline_max_region_score": float(base_score[0, 0]),
                "corrected_max_region_score": float(corr_score[0, 0]),
                "baseline_patch_cosine": base_map, "corrected_patch_cosine": corr_map,
                "baseline_attention_guided": base_guided, "corrected_attention_guided": corr_guided,
            })

        fig = plt.figure(figsize=(19, 8), constrained_layout=True)
        grid = fig.add_gridspec(3, 5, height_ratios=[0.6, 1, 1])
        ax = fig.add_subplot(grid[0, 0]); ax.imshow(rgb); ax.set_title(f"COCO val image {image_id}"); ax.axis("off")
        ax = fig.add_subplot(grid[0, 1:5]); ax.axis("off")
        ax.text(0, 0.95, "Same frozen DINO/SigLIP features; baseline vs corrected MLP (not masks)", va="top", fontsize=12, weight="bold")
        for qidx, item in enumerate(calculated):
            row = qidx + 1
            ax = fig.add_subplot(grid[row, 0]); ax.axis("off")
            ax.text(0, 0.9, f"Caption {qidx + 1}: {item['caption']}\nselected region head: {item['baseline_head']} → {item['corrected_head']}", va="top", wrap=True, fontsize=9)
            maps = [
                (item["baseline_patch_cosine"], "Baseline patch cosine"),
                (item["corrected_patch_cosine"], "Corrected patch cosine"),
                (item["baseline_attention_guided"], "Baseline attention × cosine"),
                (item["corrected_attention_guided"], "Corrected attention × cosine"),
            ]
            for col, (data, title) in enumerate(maps, start=1):
                ax = fig.add_subplot(grid[row, col])
                if col < 3:
                    lo, hi = -1.0, 1.0
                else:
                    pair = np.concatenate([item["baseline_attention_guided"].ravel(), item["corrected_attention_guided"].ravel()])
                    bound = max(float(np.max(np.abs(pair))), 1e-6)
                    lo, hi = -bound, bound
                im = ax.imshow(data, cmap="coolwarm", vmin=lo, vmax=hi, interpolation="nearest")
                ax.set_title(title, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Talk2DINO-SigLIP Phase 2 — matched held-out maps; diagnostic, not segmentation", fontsize=14)
        output_path = gallery / f"val_image_{image_id}_baseline_vs_corrected.png"
        fig.savefig(output_path, dpi=130)
        plt.close(fig)
        panels.append(output_path)
        manifest.append({
            "image_id": image_id, "relative_image_path": image_record["relative_image_path"],
            "queries": [{key: value for key, value in item.items() if not isinstance(value, np.ndarray)} for item in calculated],
            "figure": str(output_path),
            "warning": "Patch cosine and attention-guided maps are exploratory; they are not validated segmentation masks.",
        })

    thumbs = []
    for path in panels:
        im = Image.open(path).convert("RGB")
        im.thumbnail((960, 430))
        canvas = Image.new("RGB", (980, 450), "white")
        canvas.paste(im, ((980 - im.width) // 2, 10))
        thumbs.append(canvas)
    overview = Image.new("RGB", (1960, 1800), "white")
    for idx, im in enumerate(thumbs):
        overview.paste(im, ((idx % 2) * 980, (idx // 2) * 450))
    overview_path = gallery / "phase2_baseline_vs_corrected_overview.png"
    overview.save(overview_path)
    (gallery / "manifest.json").write_text(json.dumps({"status": "complete", "images": manifest,
        "overview": str(overview_path), "scope": "Same eight validation images and two caption queries each; no oracle masks or segmentation thresholds."}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "images": len(manifest), "overview": str(overview_path)}))


if __name__ == "__main__":
    main()
