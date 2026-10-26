#!/usr/bin/env python3
"""Render a review gallery from saved Stage 4A scores, without model inference."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
PHASE = ROOT / "experiments/kv_cache/outputs/stage4a_siglip_foreground/phase1"
TRACK_B = ROOT / "experiments/kv_cache/outputs/track_b_segmentation"
JSON = PHASE / "matching_diagnostics.json"
OUT_PNG = PHASE / "UNMATCH_VISUAL_REVIEW.png"
OUT_PDF = PHASE / "UNMATCH_VISUAL_REVIEW.pdf"
OUT_MATCHED_PNG = PHASE / "UNMATCH_MATCHED_MISMATCH.png"
SAMPLE_IDS = ["task4_state0_step0000", "task5_state0_step0000", "task7_state0_step0000"]
SCORE_RANGE = (-0.08, 0.15)
DIFF_RANGE = (-0.08, 0.08)

try:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT = ImageFont.truetype(FONT_PATH, 17)
    SMALL = ImageFont.truetype(FONT_PATH, 13)
    TITLE = ImageFont.truetype(FONT_BOLD_PATH, 23)
except OSError:
    FONT = SMALL = TITLE = ImageFont.load_default()


def prompt_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def color_map(values: np.ndarray, low: float, high: float, diverging: bool = False) -> Image.Image:
    """Small fixed RGB palette: magma-like for affinity, blue-white-red for delta."""
    arr = np.asarray(values, dtype=np.float32)
    t = np.clip((arr - low) / (high - low), 0, 1)
    if diverging:
        anchors = np.asarray([[42, 92, 170], [220, 230, 240], [247, 247, 247],
                             [250, 213, 195], [178, 24, 43]], dtype=np.float32)
    else:
        anchors = np.asarray([[0, 0, 4], [39, 14, 72], [116, 29, 108],
                             [200, 70, 75], [249, 142, 8], [252, 253, 191]], dtype=np.float32)
    x = t * (len(anchors) - 1)
    idx = np.minimum(x.astype(np.int32), len(anchors) - 2)
    frac = (x - idx)[..., None]
    rgb = anchors[idx] * (1 - frac) + anchors[idx + 1] * frac
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 255)), "RGB")


def grid_image(values: np.ndarray, *, binary: bool = False, difference: bool = False) -> Image.Image:
    grid = np.asarray(values).reshape(16, 16)
    if binary:
        pix = np.zeros((16, 16, 3), dtype=np.uint8)
        pix[:] = (38, 43, 51)
        pix[grid.astype(bool)] = (255, 207, 51)
        image = Image.fromarray(pix, "RGB")
    else:
        bounds = DIFF_RANGE if difference else SCORE_RANGE
        image = color_map(grid, *bounds, diverging=difference)
    return image.resize((224, 224), Image.NEAREST)


def overlay(rgb: np.ndarray, values: np.ndarray) -> Image.Image:
    base = Image.fromarray(rgb.astype(np.uint8), "RGB").resize((224, 224), Image.BICUBIC)
    heat = grid_image(values)
    return Image.blend(base, heat, 0.52)


def load_samples():
    data = json.loads(JSON.read_text())
    records = {r["sample_id"]: r for r in data["samples"]}
    result = {}
    for sid in SAMPLE_IDS:
        row = records[sid]
        with np.load(PHASE / row["affinity_npz"]) as z:
            maps = {}
            for prompt, label in ((row["task_description"], "instruction"),
                                  ("black bowl", "matched"), ("red cup", "mismatched")):
                for variant in ("penultimate", "final"):
                    key = f"{prompt_key(prompt)}__{variant}"
                    if key in z.files:
                        maps[f"{label}_{variant}"] = z[key].astype(np.float32)
        for variant in ("penultimate", "final"):
            a = maps[f"instruction_{variant}"]
            if a.ndim != 2 or a.shape[1] != 256 or not np.isfinite(a).all():
                raise ValueError(f"Invalid saved patch-text affinity for {sid}/{variant}: {a.shape}")
        rgb_path = TRACK_B / row["image_path"]
        oracle = np.load(TRACK_B / row["oracle_token_mask_path"]).astype(bool).reshape(256)
        result[sid] = (row, np.asarray(Image.open(rgb_path).convert("RGB")), oracle, maps)
    return result


def draw_panel(canvas: Image.Image, x: int, y: int, width: int, title: str,
               image: Image.Image, subtitle: str = "") -> None:
    draw = ImageDraw.Draw(canvas)
    title_box = draw.multiline_textbbox((0, 0), title, font=SMALL, align="center", spacing=2)
    th = title_box[3] - title_box[1]
    draw.multiline_text((x + (width - (title_box[2] - title_box[0])) / 2, y), title,
                        fill=(20, 24, 30), font=SMALL, align="center", spacing=2)
    image = image.resize((224, 224), Image.NEAREST)
    ix = x + (width - 224) // 2
    iy = y + th + 7
    canvas.paste(image, (ix, iy))
    if subtitle:
        sb = draw.multiline_textbbox((0, 0), subtitle, font=SMALL, align="center", spacing=2)
        draw.multiline_text((x + (width - (sb[2] - sb[0])) / 2, iy + 228), subtitle,
                            fill=(55, 61, 70), font=SMALL, align="center", spacing=2)


def make_contact_sheet(samples) -> Image.Image:
    columns = ["Original RGB", "Penultimate · full instruction", "Final · full instruction",
               "Final relevance overlay", "Top 25% scores", "Top 50% scores", "Top 75% scores",
               "Final − penultimate", "Oracle FG (post-hoc)"]
    cell_w, cell_h, left, top = 286, 322, 185, 106
    width, height = left + 9 * cell_w + 22, top + 3 * cell_h + 134
    canvas = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(canvas)
    d.text((24, 16), "Stage 4A · UNMATCH-inspired SigLIP patch–text review", fill=(15, 20, 27), font=TITLE)
    d.text((24, 49), "Visual→text score = max cosine over retained instruction tokens; values are affinities, not probabilities.", fill=(45, 50, 58), font=FONT)
    for c, name in enumerate(columns):
        box = d.multiline_textbbox((0, 0), name, font=SMALL, align="center")
        d.multiline_text((left + c * cell_w + (cell_w - (box[2]-box[0]))/2, 82), name,
                         fill=(20, 24, 30), font=SMALL, align="center")
    for r, sid in enumerate(SAMPLE_IDS):
        row, rgb, oracle, maps = samples[sid]
        prompt = row["task_description"]
        p = maps["instruction_penultimate"].max(axis=0)
        f = maps["instruction_final"].max(axis=0)
        d.text((16, top + r * cell_h + 8), f"Task {row['task_id']}\nstate {row['state_id']}\n{sid.rsplit('_',1)[-1]}",
               fill=(20, 24, 30), font=SMALL, spacing=3)
        x0, y0 = left, top + r * cell_h
        panels = [Image.fromarray(rgb), grid_image(p), grid_image(f), overlay(rgb, f)]
        for q in (0.25, 0.50, 0.75):
            n = int(np.ceil(q * 256))
            selected = np.argsort(f, kind="stable")[-n:]
            mask = np.zeros(256, dtype=np.uint8); mask[selected] = 1
            panels.append(grid_image(mask, binary=True))
        panels.extend([grid_image(f-p, difference=True), grid_image(oracle, binary=True)])
        subtitles = [prompt[:67] + ("…" if len(prompt) > 67 else ""),
                     "same cosine scale", "same cosine scale", "same cosine scale",
                     "heuristic only · 64 patches", "heuristic only · 128 patches",
                     "heuristic only · 192 patches", "score-map difference only", f"{oracle.sum()} oracle FG patches"]
        for c, (title, im, sub) in enumerate(zip(columns, panels, subtitles)):
            draw_panel(canvas, x0 + c * cell_w, y0, cell_w, title, im, sub)
    # Consistent, explicit color legends.
    bar_y = top + 3 * cell_h + 25
    d.text((left, bar_y), "Common cosine-affinity scale", fill=(30, 35, 42), font=SMALL)
    bar = color_map(np.linspace(*SCORE_RANGE, 256)[None, :], *SCORE_RANGE).resize((420, 18))
    canvas.paste(bar, (left + 225, bar_y))
    d.text((left + 655, bar_y), f"{SCORE_RANGE[0]:.2f} to {SCORE_RANGE[1]:.2f}", fill=(45, 50, 58), font=SMALL)
    d.text((left + 850, bar_y), "Score-difference scale", fill=(30, 35, 42), font=SMALL)
    diffbar = color_map(np.linspace(*DIFF_RANGE, 256)[None, :], *DIFF_RANGE, diverging=True).resize((300, 18))
    canvas.paste(diffbar, (left + 1040, bar_y))
    d.text((24, height - 54), "Top-score masks are illustrative heuristic visualizations, NOT validated foreground segmentation. Oracle is shown post hoc only.", fill=(150, 36, 27), font=FONT)
    d.text((24, height - 27), "Task-4/task-5 instruction text is context-limited in the saved diagnostic; see the report for truncation. Final−penultimate compares relevance maps, not raw feature tensors.", fill=(55, 61, 70), font=SMALL)
    return canvas


def make_phrase_page(samples) -> Image.Image:
    sid = SAMPLE_IDS[0]
    row, rgb, _, maps = samples[sid]
    cell_w, cell_h, margin = 310, 290, 30
    canvas = Image.new("RGB", (margin * 2 + 4 * cell_w, 94 + 2 * cell_h + 95), "white")
    d = ImageDraw.Draw(canvas)
    d.text((margin, 15), "Task 4/state 0 · same image, matched vs mismatched phrase", fill=(15, 20, 27), font=TITLE)
    d.text((margin, 50), "All score maps use the same cosine scale; map values are visual→text max affinities.", fill=(45,50,58), font=FONT)
    headers = ["RGB", "Matched phrase: black bowl", "Mismatched phrase: red cup", "Matched − mismatched"]
    for r, variant in enumerate(("penultimate", "final")):
        y = 90 + r * cell_h
        p = maps[f"matched_{variant}"].max(axis=0).reshape(16,16)
        n = maps[f"mismatched_{variant}"].max(axis=0).reshape(16,16)
        ims = [Image.fromarray(rgb), grid_image(p), grid_image(n), grid_image(p-n, difference=True)]
        for c, (title, im) in enumerate(zip(headers, ims)):
            x = margin + c * cell_w
            draw_panel(canvas, x, y, cell_w, title if r == 0 else f"{variant.title()} · {title}", im)
    d.text((margin, canvas.height - 55), "Binary masks are not inferred here; this page inspects saved local affinities only.", fill=(150,36,27), font=FONT)
    return canvas


def main() -> None:
    samples = load_samples()
    gallery = make_contact_sheet(samples)
    phrase = make_phrase_page(samples)
    gallery.save(OUT_PNG, optimize=True)
    phrase.save(OUT_MATCHED_PNG, optimize=True)
    # PDF is two pages, enlarged for closer inspection.
    gallery.save(OUT_PDF, "PDF", resolution=150.0, save_all=True, append_images=[phrase])
    print(f"saved {OUT_PNG} ({OUT_PNG.stat().st_size:,} bytes)")
    print(f"saved {OUT_MATCHED_PNG} ({OUT_MATCHED_PNG.stat().st_size:,} bytes)")
    print(f"saved {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
