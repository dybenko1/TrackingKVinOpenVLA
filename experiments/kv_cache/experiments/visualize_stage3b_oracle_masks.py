"""Render Stage 3B oracle masks over saved LIBERO scene images.

Stage 3B stores the exact per-step foreground token indices but not RGB frames.
This gallery uses saved Track B RGB frames only when their oracle token-mask
hash exactly matches the corresponding Stage 3B mask. Thus every colored 16x16
grid is the actual Stage 3B mask; the figure metadata explicitly notes that
the RGB replay may differ from the Stage 3B trajectory after policy divergence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
STAGE3B = OUTPUTS / "stage3b_oracle_full_kv_matrix"
TRACKB = OUTPUTS / "track_b_segmentation"
OUT = STAGE3B / "oracle_token_mask_gallery"
SCENARIOS = {
    (4, 2): (0, 20, 40, 60, 63),
    (5, 0): (0, 1, 2, 15, 35),
    (7, 0): (0, 4, 21, 35, 50),
}
TILE = 288
PANEL_W = 310
PANEL_H = 356
LEFT, TOP, PAD = 235, 145, 20


def _font(size: int, bold: bool = False):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=bool).reshape(-1).tobytes()).hexdigest()


def _overlay(rgb: Image.Image, mask: np.ndarray) -> Image.Image:
    nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    base = rgb.convert("RGB").resize((TILE, TILE), nearest).convert("RGBA")
    color_layer = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(color_layer)
    grid = np.asarray(mask, dtype=bool).reshape(16, 16)
    cell = TILE // 16
    for row in range(16):
        for col in range(16):
            color = (40, 220, 84, 124) if grid[row, col] else (42, 122, 235, 124)
            x0, y0 = col * cell, row * cell
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=color)
    result = Image.alpha_composite(base, color_layer)
    grid_draw = ImageDraw.Draw(result)
    for n in range(17):
        pos = n * cell
        grid_draw.line((pos, 0, pos, TILE), fill=(255, 255, 255, 230), width=1)
        grid_draw.line((0, pos, TILE, pos), fill=(255, 255, 255, 230), width=1)
    return result.convert("RGB")


def main():
    manifest_path = TRACKB / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Track B saved-frame manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_index = {(int(x["task_id"]), int(x["state_id"]), int(x["time_step"])): x
                    for x in manifest["samples"]}
    gallery_frames = {}
    records_meta = []
    task_descriptions = {}

    for (task_id, state_id), steps in SCENARIOS.items():
        stage_path = STAGE3B / f"task{task_id}_state{state_id}" / "results.json"
        scenario = json.loads(stage_path.read_text(encoding="utf-8"))
        rollout = scenario["stage3b_true_skip"]
        by_step = {int(x["step"]): x for x in rollout["records"]}
        task_descriptions[(task_id, state_id)] = rollout["task_description"]
        for step in steps:
            if step not in by_step:
                raise RuntimeError(f"Stage 3B result has no task {task_id}/state {state_id}/step {step}.")
            sample = sample_index.get((task_id, state_id, step))
            if sample is None:
                raise RuntimeError(f"Track B has no saved RGB/mask at task {task_id}/state {state_id}/step {step}.")
            saved_mask = np.load(TRACKB / sample["oracle_token_mask_path"]).astype(bool).reshape(-1)
            ref = by_step[step]
            logged_mask = np.zeros(256, dtype=bool)
            logged_mask[np.asarray(ref["mask_foreground_indices"], dtype=np.int64)] = True
            logged_hash = _mask_digest(logged_mask)
            saved_hash = _mask_digest(saved_mask)
            if logged_hash != ref["mask_sha256"]:
                raise AssertionError("Stage 3B logged foreground indices do not reproduce its mask hash.")
            if saved_hash != ref["mask_sha256"]:
                raise RuntimeError(
                    f"Track B and Stage 3B mask mismatch for task {task_id}/state {state_id}/step {step}; "
                    "refusing to render a potentially misaligned example."
                )
            rgb = Image.open(TRACKB / sample["rgb_path"])
            gallery_frames[(task_id, state_id, step)] = ( _overlay(rgb, logged_mask), logged_mask, ref)
            records_meta.append({
                "task_id": task_id, "state_id": state_id, "step": step,
                "mode": ref["type"], "source_age": int(ref["source_age"]),
                "foreground_tokens": int(logged_mask.sum()),
                "background_tokens": int((~logged_mask).sum()),
                "mask_sha256": ref["mask_sha256"],
                "mask_matches_stage3b_log": True,
                "rgb_source": sample["rgb_path"],
                "rgb_is_guaranteed_stage3b_observation": False,
            })

    width = LEFT + len(next(iter(SCENARIOS.values()))) * PANEL_W + PAD
    height = TOP + len(SCENARIOS) * PANEL_H + 118
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    title, head, body, small = _font(25, True), _font(15, True), _font(13), _font(11)
    draw.text((PAD, 18), "Stage 3B — Oracle foreground/background visual-token masks", font=title, fill=(25, 35, 48))
    draw.text((PAD, 53), "Overlay colors show the exact logged Stage 3B 16×16 oracle mask; each cell is one visual token.", font=body, fill=(45, 55, 68))
    draw.text((PAD, 75), "RGB panels are saved Track B replays. Only frames whose token-mask SHA-256 matches Stage 3B are included.", font=body, fill=(45, 55, 68))
    draw.text((PAD, 97), "Because Track B replays Stage 3A actions, its RGB is not guaranteed identical to the Stage 3B trajectory.", font=small, fill=(100, 55, 45))
    draw.text((PAD, 115), "Green = foreground (fresh K/V). Blue = background (reused K/V on R5 reuse steps; all fresh on refresh steps).", font=small, fill=(45, 55, 68))

    for row, ((task_id, state_id), steps) in enumerate(SCENARIOS.items()):
        y = TOP + row * PANEL_H
        draw.text((PAD, y + 4), f"Task {task_id}\nState {state_id}", font=head, fill=(25, 35, 48), spacing=5)
        description = task_descriptions[(task_id, state_id)]
        for j in range(3):
            draw.text((PAD, y + 52 + j * 17), description[j * 25:(j + 1) * 25], font=small, fill=(70, 80, 90))
        for col, step in enumerate(steps):
            image, mask, ref = gallery_frames[(task_id, state_id, step)]
            x = LEFT + col * PANEL_W
            schedule = "REFRESH" if ref["type"] == "refresh" else f"REUSE age {ref['source_age']}"
            draw.text((x + 2, y - 20), f"step {step} · {schedule}", font=small, fill=(45, 55, 68))
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + TILE - 1, y + TILE - 1), outline=(25, 35, 48), width=2)
            draw.text((x + 2, y + TILE + 7), f"FG {int(mask.sum()):3d}   BG {int((~mask).sum()):3d} / 256", font=body, fill=(30, 40, 55))
            draw.text((x + 2, y + TILE + 28), f"mask {ref['mask_sha256'][:12]}", font=small, fill=(80, 88, 98))

    legend_y = TOP + len(SCENARIOS) * PANEL_H + 22
    draw.rectangle((PAD, legend_y, PAD + 22, legend_y + 22), fill=(40, 220, 84), outline=(25, 35, 48))
    draw.text((PAD + 32, legend_y + 3), "Oracle foreground", font=body, fill=(30, 40, 55))
    draw.rectangle((PAD + 240, legend_y, PAD + 262, legend_y + 22), fill=(42, 122, 235), outline=(25, 35, 48))
    draw.text((PAD + 272, legend_y + 3), "Oracle background", font=body, fill=(30, 40, 55))

    OUT.mkdir(parents=True, exist_ok=True)
    image_path = OUT / "stage3b_oracle_foreground_background_grid.png"
    canvas.save(image_path, optimize=True)
    metadata = {
        "status": "complete",
        "experiment": "Stage 3B oracle-masked full-layer K+V reuse, R=5",
        "visual_grid": [16, 16],
        "source_note": "Stage 3B logged foreground indices are overlaid on saved Track B RGB frames only when the saved oracle token-mask hash exactly matches the Stage 3B per-step hash. Track B RGB comes from Stage 3A action replays and is not guaranteed to be the identical Stage 3B observation after trajectories diverge.",
        "legend": {"green": "oracle foreground; K/V fresh", "blue": "oracle background; stale K/V on reuse steps, fresh on refresh steps"},
        "frames": records_meta,
        "gallery": str(image_path.relative_to(ROOT)),
    }
    _atomic_json(OUT / "stage3b_oracle_foreground_background_grid.json", metadata)
    print(json.dumps({"status": "complete", "gallery": str(image_path), "verified_frames": len(records_meta)}, indent=2))


if __name__ == "__main__":
    main()
