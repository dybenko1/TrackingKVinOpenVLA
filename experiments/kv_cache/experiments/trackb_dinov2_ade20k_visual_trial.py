"""Visual ADE20K trial on saved LIBERO observations using official DINOv2.

This is an offline visualization experiment. It does not load OpenVLA's policy,
generate actions, use oracle masks for inference, or change any KV-cache code.
The official path reproduces the released ViT-L/14 ADE20K linear head. A second
diagnostic applies that head to already-saved OpenVLA DINO features; it is an
out-of-distribution compatibility probe, not a supported checkpoint pairing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
TRACK_B = ROOT / "experiments/kv_cache/outputs/track_b_segmentation"
OUT = ROOT / "experiments/kv_cache/outputs/trackb_dinov2_segmentation_visual_trial"
HEAD = ROOT / "cache/models/dinov2/ade20k/dinov2_vitl14_ade20k_linear_head.pth"
HEAD_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_ade20k_linear_head.pth"
HEAD_SHA256 = "59137f319d7e83c79f3253ab7bf63041f1f884e54ba865a2c7ef0046655306a0"
DINOV2_HF_REVISION = "0ff9d1340c9524c60f3f03e8573c57a1f8197f24"
DINOV2_HF_DIR = ROOT / "cache/models/dinov2/facebook_dinov2_large" / DINOV2_HF_REVISION
DINOV2_HF_SHA256 = "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"
TASKS = (4, 5, 7)
STATES = (0, 1, 2)
SIDE = 512
PALETTE: np.ndarray | None = None
CLASS_NAMES: list[str] | None = None


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def select_samples() -> list[dict[str, Any]]:
    manifest = json.loads((TRACK_B / "dataset_manifest.json").read_text())
    rows = manifest["samples"]
    chosen: list[dict[str, Any]] = []
    for task in TASKS:
        for state in STATES:
            group = sorted((r for r in rows if r["task_id"] == task and r["state_id"] == state), key=lambda r: r["time_step"])
            if len(group) < 3:
                raise RuntimeError(f"Need 3 saved observations for task{task}/state{state}; found {len(group)}")
            # Earliest, median, and latest saved observation approximate early/mid/late.
            ids = sorted({0, len(group) // 2, len(group) - 1})
            for phase, idx in zip(("early", "middle", "late"), ids):
                row = dict(group[idx])
                row["phase"] = phase
                row["policy_image_abs"] = str(TRACK_B / row["rgb_path"])
                row["camera_image_abs"] = str(TRACK_B / row["rgb_observation_path"])
                row["feature_abs"] = str(TRACK_B / row["feature_path"])
                if not all(Path(row[k]).is_file() for k in ("policy_image_abs", "camera_image_abs", "feature_abs")):
                    raise FileNotFoundError(f"Missing saved sample artifact for {row['sample_id']}")
                chosen.append(row)
    return chosen


def _load_official_meta(head_path: Path, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    obj = torch.load(head_path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj)
    meta = obj.get("meta", {})
    if tuple(state["decode_head.conv_seg.weight"].shape) != (150, 1024, 1, 1):
        raise ValueError("Unexpected released ADE20K ViT-L/14 linear head shape.")
    if len(meta.get("CLASSES", ())) != 150 or len(meta.get("PALETTE", ())) != 150:
        raise ValueError("Official checkpoint metadata does not contain 150 ADE classes and colors.")
    return state, meta


def _make_head(torch: Any, state: dict[str, Any], device: Any):
    import torch.nn as nn

    class OfficialBNHead(nn.Module):
        # Equivalent to the released MMSeg BNHead when input_transform selects
        # the final feature map and dropout_ratio is zero.
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(1024, eps=1e-5, affine=True, track_running_stats=True)
            self.conv_seg = nn.Conv2d(1024, 150, kernel_size=1, bias=True)

        def forward(self, x):
            return self.conv_seg(self.bn(x))

    module = OfficialBNHead().eval()
    module.load_state_dict({k.removeprefix("decode_head."): v for k, v in state.items()}, strict=True)
    return module.to(device=device, dtype=torch.float32)


def _center_pad_to_patch_multiple(x: Any, patch: int, torch: Any):
    h, w = x.shape[-2:]
    ph, pw = (-h) % patch, (-w) % patch
    return torch.nn.functional.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def official_preprocess(image: Image.Image, torch: Any, device: Any, dtype: Any):
    # Official ADE20K test config: keep aspect ratio with short side 512,
    # RGB normalization using ImageNet mean/std in 0..255 units.
    w, h = image.size
    scale = SIDE / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = image.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    arr = (arr - mean) / std
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return x, resized.size


def infer_official(model: Any, head: Any, image: Image.Image, torch: Any, device: Any, dtype: Any):
    x, _ = official_preprocess(image, torch, device, dtype)
    inference_hw = tuple(x.shape[-2:])
    x = _center_pad_to_patch_multiple(x, 14, torch)
    with torch.inference_mode():
        outputs = model(pixel_values=x, return_dict=True)
        # HF's final hidden state is after the model's final LayerNorm, matching
        # the official DINOv2 get_intermediate_layers(..., n=[20,21,22,23])
        # output selected at index 3 by the ADE20K config.
        tokens = outputs.last_hidden_state[:, 1:, :]
        gh, gw = x.shape[-2] // 14, x.shape[-1] // 14
        if tokens.shape[1] != gh * gw:
            raise RuntimeError(f"Unexpected final DINO patch tokens {tokens.shape}; expected grid {gh}x{gw}")
        feature = tokens.transpose(1, 2).reshape(1, 1024, gh, gw).float()
        logits = head(feature)
        # The official notebook pads the backbone input internally but the
        # segmentation framework resizes logits to the unpadded crop extent.
        logits = torch.nn.functional.interpolate(logits, size=inference_hw, mode="bilinear", align_corners=False)
    # Match MMSeg's resize-back to original LIBERO policy image dimensions.
    logits = torch.nn.functional.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
    return logits[0].argmax(0).to("cpu").numpy().astype(np.uint8), feature


def infer_saved_openvla_head(head: Any, feature_np: np.ndarray, torch: Any, device: Any):
    if feature_np.shape != (256, 1024):
        raise ValueError(f"Expected saved OpenVLA DINO feature [256,1024], got {feature_np.shape}")
    x = torch.from_numpy(feature_np.astype(np.float32)).to(device=device).transpose(0, 1).reshape(1, 1024, 16, 16)
    with torch.inference_mode():
        logits = head(x)
        logits = torch.nn.functional.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False)
    return logits[0].argmax(0).to("cpu").numpy().astype(np.uint8), x


def colorize(ids: np.ndarray) -> np.ndarray:
    assert PALETTE is not None
    return PALETTE[np.asarray(ids, dtype=np.int64)]


def overlay(rgb: np.ndarray, seg_rgb: np.ndarray, alpha: float = 0.52) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * (1-alpha) + seg_rgb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def top_classes(ids: np.ndarray, classes: list[str], n: int = 8) -> list[dict[str, Any]]:
    vals, counts = np.unique(ids, return_counts=True)
    order = np.argsort(counts)[::-1][:n]
    total = int(ids.size)
    return [{"class_id": int(vals[i]), "class": str(classes[int(vals[i])]), "pixels": int(counts[i]), "fraction": float(counts[i] / total)} for i in order]


def _font(size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit_rgb(arr: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)


def make_frame_panel(row: dict[str, Any], dest: Path, classes: list[str]) -> None:
    policy = np.asarray(Image.open(row["policy_image_abs"]).convert("RGB"))
    official_ids = np.asarray(Image.open(dest / "official_class_ids.png"))
    official = colorize(official_ids)
    native_ids = np.asarray(Image.open(dest / "openvla_native_class_ids.png"))
    native = colorize(native_ids)
    w, h = policy.shape[1], policy.shape[0]
    cols, gap, title_h, legend_h = 5, 12, 52, 110
    cell_w, cell_h = 280, 280
    canvas = Image.new("RGB", (cols*cell_w+(cols+1)*gap, title_h+cell_h+legend_h+2*gap), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    f = _font(19); small = _font(14)
    title = f"Task {row['task_id']} / state {row['state_id']} · {row['phase']} · step {row['time_step']}"
    draw.text((gap, 8), title, font=f, fill=(10, 10, 10))
    camera = np.asarray(Image.open(dest / "camera_rgb.png").convert("RGB"))
    panels = [
        (camera, "Raw camera RGB"),
        (policy, "Policy RGB input"),
        (official, "Official DINOv2 ViT-L/14 + ADE20K"),
        (overlay(policy, official), "Official prediction overlay"),
        (overlay(policy, native), "OpenVLA saved DINO feature probe*"),
    ]
    y0 = title_h
    for c, (arr, label) in enumerate(panels):
        x0 = gap + c*(cell_w+gap)
        im = _fit_rgb(arr, (cell_w, cell_h-28))
        canvas.paste(im, (x0, y0+28))
        draw.text((x0, y0+4), label, font=small, fill=(15, 15, 15))
    top = top_classes(official_ids, classes, 7)
    summary = "Official ADE20K top categories: " + ", ".join(f"{x['class']} {100*x['fraction']:.0f}%" for x in top)
    draw.text((gap, y0+cell_h+8), summary, font=small, fill=(10, 10, 10))
    draw.text((gap, y0+cell_h+34), "*Exploratory head-on-cached-feature probe; feature layer/register/preprocessing do not match official ADE training.", font=small, fill=(130, 35, 25))
    canvas.save(dest / "comparison_panel.png")


def make_group_sheet(rows: list[dict[str, Any]], out: Path, classes: list[str]) -> None:
    width_each, height_each = 260, 255
    label_h, header_h, gap = 48, 34, 10
    canvas = Image.new("RGB", (3*width_each+4*gap, header_h+3*(height_each+label_h)+4*gap), "white")
    d = ImageDraw.Draw(canvas); small = _font(13); bold = _font(16)
    d.text((gap, 8), "Left: policy RGB  |  Center: official ADE20K  |  Right: overlay", font=bold, fill="black")
    for i, row in enumerate(sorted(rows, key=lambda r: r["time_step"])):
        dest = OUT / "visualizations" / row["sample_id"]
        ids = np.asarray(Image.open(dest / "official_class_ids.png"))
        rgb = np.asarray(Image.open(row["policy_image_abs"]).convert("RGB"))
        views = [rgb, colorize(ids), overlay(rgb, colorize(ids))]
        y = header_h+gap+i*(height_each+label_h+gap)
        d.text((gap, y), f"{row['phase']} · timestep {row['time_step']}", font=bold, fill="black")
        y += 22
        for c, arr in enumerate(views):
            x = gap+c*(width_each+gap)
            canvas.paste(_fit_rgb(arr, (width_each, height_each)), (x, y))
        cats = ", ".join(x["class"] for x in top_classes(ids, classes, 4))
        d.text((gap, y+height_each+4), cats[:140], font=small, fill=(20,20,20))
    canvas.save(out)


def make_global_overview(rows: list[dict[str, Any]], path: Path, classes: list[str]) -> None:
    # One representative middle frame from each task/state. Each scenario cell
    # shows policy RGB and the official ADE20K prediction side-by-side.
    cell_w, img_h, cap_h, gap = 500, 250, 66, 18
    canvas = Image.new("RGB", (3*cell_w+4*gap, 3*(img_h+cap_h)+4*gap), "white")
    d = ImageDraw.Draw(canvas); f = _font(18); small = _font(13)
    for row in rows:
        if row["phase"] != "middle":
            continue
        r, c = TASKS.index(row["task_id"]), STATES.index(row["state_id"])
        x, y = gap+c*(cell_w+gap), gap+r*(img_h+cap_h+gap)
        dest = OUT / "visualizations" / row["sample_id"]
        policy = np.asarray(Image.open(row["policy_image_abs"]).convert("RGB"))
        seg = np.asarray(Image.open(dest/"official_class_ids.png"))
        panel = np.concatenate([policy, overlay(policy, colorize(seg))], axis=1)
        canvas.paste(_fit_rgb(panel, (cell_w, img_h)), (x, y))
        cats = ", ".join(o["class"] for o in top_classes(seg, classes, 4))
        d.text((x, y+img_h+4), f"Task {row['task_id']} / state {row['state_id']} · step {row['time_step']}", font=f, fill="black")
        d.text((x, y+img_h+28), f"RGB | Official ADE20K overlay · {cats[:55]}", font=small, fill=(30,30,30))
    canvas.save(path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["sample_id", "task_id", "task_description", "state_id", "phase", "time_step", "source", "official_top_classes", "official_unique_classes", "native_top_classes", "official_inference_ms", "official_head_ms", "native_head_ms", "official_feature_spec", "native_feature_spec"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def run(out: Path = OUT, limit: int | None = None) -> dict[str, Any]:
    global PALETTE, CLASS_NAMES
    import torch

    out.mkdir(parents=True, exist_ok=True)
    (out / "visualizations").mkdir(exist_ok=True)
    (out / "contact_sheets").mkdir(exist_ok=True)
    if not HEAD.exists() or sha256(HEAD) != HEAD_SHA256:
        raise RuntimeError(f"Official ADE20K head absent or hash mismatch: {HEAD}")
    state, meta = _load_official_meta(HEAD, torch)
    CLASS_NAMES = [str(x).strip() for x in meta["CLASSES"]]
    PALETTE = np.asarray(meta["PALETTE"], dtype=np.uint8)
    rows = select_samples()
    if limit:
        rows = rows[:limit]
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if use_cuda else torch.float32)
    if not (DINOV2_HF_DIR / "config.json").is_file() or not (DINOV2_HF_DIR / "model.safetensors").is_file():
        raise RuntimeError("Official Meta DINOv2 HF weights/config missing; follow download instructions in REPORT.md.")
    if sha256(DINOV2_HF_DIR / "model.safetensors") != DINOV2_HF_SHA256:
        raise RuntimeError("Meta DINOv2 HF checkpoint SHA-256 mismatch.")
    from transformers import Dinov2Model
    model = Dinov2Model.from_pretrained(str(DINOV2_HF_DIR), local_files_only=True, torch_dtype=dtype)
    model.eval().to(device=device, dtype=dtype)
    config_model = model.config
    if config_model.num_hidden_layers != 24 or getattr(config_model, "num_register_tokens", 0) != 0 or config_model.hidden_size != 1024 or config_model.patch_size != 14:
        raise RuntimeError("Expected a register-free DINOv2 ViT-L/14 model (24 blocks, width 1024, patch 14).")
    head = _make_head(torch, state, device)
    torch.cuda.empty_cache() if use_cuda else None
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    # One unmeasured forward removes first-call CUDA initialization from the
    # reported per-frame segmentation latency.
    warm_image = Image.open(select_samples()[0]["policy_image_abs"]).convert("RGB")
    _ = infer_official(model, head, warm_image, torch, device, dtype)
    if use_cuda:
        torch.cuda.synchronize()

    config = {
        "experiment": "Track B DINOv2 ADE20K semantic segmentation visual trial",
        "tasks": list(TASKS), "states": list(STATES), "frames_per_scenario": 3,
        "sampling": "earliest, median, latest among saved Track B sample frames; no LIBERO replay",
        "input_image": "saved policy_rgb.png: OpenVLA-oriented/resized/cropped observation; DINOv2 ADE pipeline then rescales short side to 512",
        "official_backbone": "Meta facebook/dinov2-large DINOv2 ViT-L/14; 24 blocks, 1024 dim, zero registers; final normalized patch tokens used by ADE config",
        "official_backbone_hf_source": "Meta's official facebook/dinov2-large Transformers checkpoint, loaded with Dinov2Model; revision pinned",
        "official_backbone_hf_revision": DINOV2_HF_REVISION,
        "official_backbone_hf_sha256": DINOV2_HF_SHA256,
        "official_head": str(HEAD.relative_to(ROOT)), "official_head_url": HEAD_URL, "official_head_sha256": sha256(HEAD),
        "official_head_checkpoint_meta": {k: meta.get(k) for k in ("mmseg_version", "exp_name", "iter", "epoch")},
        "official_head_architecture": "MMSeg BNHead: BatchNorm2d(1024) then Conv2d(1024,150,1), applied to output index 3 of blocks [20,21,22,23]",
        "native_probe": "same official head on saved OpenVLA DINO features (256x1024), no extra vision forward; known mismatch: OpenVLA reg4, block 22, no feature norm per Track B manifest, 224px policy preprocessing vs official register-free final block 23 and 512-short-side ADE preprocessing",
        "inference_device": str(device), "vision_dtype": str(dtype), "class_count": 150,
        "oracle_used_for_inference": False, "policy_or_KV_modified": False,
        "official_DINOv2_source": "https://github.com/facebookresearch/dinov2",
        "official_notebook": "https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb",
        "head_checkpoint_source": HEAD_URL,
        "official_test_preprocess": "MMSeg ADE20K config: RGB, keep aspect ratio with 512 short side, normalize 0..255 by mean [123.675,116.28,103.53] and std [58.395,57.12,57.375], center-pad to patch multiple 14 as notebook",
    }
    (out / "configuration.json").write_text(json.dumps(config, indent=2)+"\n")

    records = []
    full_ms: list[float] = []
    head_ms: list[float] = []
    native_ms: list[float] = []
    peak_mem = 0
    for i, row in enumerate(rows):
        sample_out = out / "visualizations" / row["sample_id"]
        sample_out.mkdir(parents=True, exist_ok=True)
        policy = Image.open(row["policy_image_abs"]).convert("RGB")
        camera = Image.open(row["camera_image_abs"]).convert("RGB")
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        ids, feature = infer_official(model, head, policy, torch, device, dtype)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start)*1000
        full_ms.append(elapsed)
        ids_img = Image.fromarray(ids, mode="L")
        ids_img.save(sample_out / "official_class_ids.png")
        colorize(ids).astype(np.uint8)
        official_rgb = colorize(ids)
        official_rgb_img = Image.fromarray(official_rgb)
        official_rgb_img.save(sample_out / "official_ade20k_colors.png")
        Image.fromarray(overlay(np.asarray(policy), official_rgb)).save(sample_out / "official_overlay.png")
        Image.fromarray(np.asarray(camera)).save(sample_out / "camera_rgb.png")

        z = np.load(row["feature_abs"])
        native_feat = z["dino"]
        # Head-only timing on cached OpenVLA features (no backbone computation).
        xnative = torch.from_numpy(native_feat.astype(np.float32)).to(device=device).transpose(0,1).reshape(1,1024,16,16)
        if use_cuda:
            torch.cuda.synchronize(); n0,n1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            for _ in range(2): head(xnative)
            if use_cuda:
                torch.cuda.synchronize()
            h0,h1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            h0.record();
            for _ in range(10): head(xnative)
            h1.record()
            if use_cuda:
                torch.cuda.synchronize(); native_head=float(h0.elapsed_time(h1))/10
            else:
                native_head=0.0
        native_ms.append(native_head)
        native_ids,_ = infer_saved_openvla_head(head,native_feat,torch,device)
        Image.fromarray(native_ids,mode="L").save(sample_out/"openvla_native_class_ids.png")
        Image.fromarray(colorize(native_ids)).save(sample_out/"openvla_native_ade20k_colors.png")
        native_policy = np.asarray(policy.resize((224,224),Image.Resampling.BILINEAR))
        Image.fromarray(overlay(native_policy,colorize(native_ids))).save(sample_out/"openvla_native_overlay.png")

        # Head-only timing using the official backbone's already-extracted map.
        if use_cuda:
            torch.cuda.synchronize(); eh0,eh1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            eh0.record()
            with torch.inference_mode():
                for _ in range(10): head(feature.float())
            eh1.record(); torch.cuda.synchronize(); official_head=float(eh0.elapsed_time(eh1))/10
        else:
            t=time.perf_counter()
            with torch.inference_mode():
                for _ in range(10): head(feature.float())
            official_head=(time.perf_counter()-t)*100
        head_ms.append(official_head)
        peaks = torch.cuda.max_memory_allocated(device) / (1024**2) if use_cuda else 0
        peak_mem=max(peak_mem,int(peaks))
        top = top_classes(ids, CLASS_NAMES)
        native_top=top_classes(native_ids,CLASS_NAMES)
        record = {k:row.get(k) for k in ("sample_id","task_id","task_description","state_id","trajectory_id","phase","time_step","split","r5_age")}
        record.update({
            "source":"existing Track B replayed Stage 3A Fresh observation",
            "official_top_classes":top,"official_unique_classes":int(np.unique(ids).size),
            "native_top_classes":native_top,"native_unique_classes":int(np.unique(native_ids).size),
            "official_inference_ms":elapsed,"official_head_ms":official_head,"native_head_ms":native_head,
            "official_feature_spec":"DINOv2 ViT-L/14 block 23 normalized patch map [1,1024,37,37] after official center pad",
            "native_feature_spec":"OpenVLA track-B cached dino [256,1024], block 22, reg4, no norm per feature manifest; head output is compatibility probe",
            "artifacts": {"official_ids":str((sample_out/"official_class_ids.png").relative_to(out)),"official_color":str((sample_out/"official_ade20k_colors.png").relative_to(out)),"official_overlay":str((sample_out/"official_overlay.png").relative_to(out)),"native_ids":str((sample_out/"openvla_native_class_ids.png").relative_to(out)),"native_color":str((sample_out/"openvla_native_ade20k_colors.png").relative_to(out)),"native_overlay":str((sample_out/"openvla_native_overlay.png").relative_to(out))}
        })
        records.append(record)
        make_frame_panel(row,sample_out,CLASS_NAMES)
        print(f"[ADE_FRAME] {row['sample_id']} official_ms={elapsed:.2f} unique={len(np.unique(ids))} top={top[0]['class']}",flush=True)

    for task in TASKS:
        for state_id in STATES:
            group=[r for r in rows if r["task_id"]==task and r["state_id"]==state_id]
            make_group_sheet(group,out/"contact_sheets"/f"task{task}_state{state_id}.png",CLASS_NAMES)
    make_global_overview(rows,out/"global_overview.png",CLASS_NAMES)
    # Global legend for the categories that appeared most often in predictions.
    counts={}
    for r in records:
        for v in r["official_top_classes"]:
            counts[v["class"]]=counts.get(v["class"],0)+v["pixels"]
    frequent=sorted(counts,key=counts.get,reverse=True)[:20]
    legend=Image.new("RGB",(700,40+28*len(frequent)),"white"); d=ImageDraw.Draw(legend); f=_font(16)
    d.text((10,6),"ADE20K categories predicted in the selected sample panels",font=f,fill="black")
    for j,name in enumerate(frequent):
        class_id=CLASS_NAMES.index(name); x=12;y=36+j*28
        d.rectangle((x,y+3,x+22,y+25),fill=tuple(int(q) for q in PALETTE[class_id]))
        d.text((x+32,y+4),f"{class_id}: {name}",font=f,fill="black")
    legend.save(out/"contact_sheets"/"category_legend.png")
    write_csv(records,out/"frame_manifest.csv")
    if use_cuda:
        torch.cuda.synchronize()
    summary={
        "status":"complete","frame_count":len(records),"scenario_count":len(TASKS)*len(STATES),
        "official_success":True,"native_head_application":"completed as compatibility probe; not a matched pretrained feature path",
        "official_inference_ms":{"median":statistics.median(full_ms),"mean":statistics.mean(full_ms),"p95":float(np.percentile(full_ms,95)),"n":len(full_ms),"scope":"synchronized wall-clock per-image path: PIL RGB resize/normalization prep, host-to-device tensor, DINOv2 backbone, ADE head, output resize and label decoding; after one warm-up and model load, excludes download/load"},
        "head_only_ms":{"official_final_feature":statistics.median(head_ms),"openvla_cached_feature":statistics.median(native_ms),"scope":"ADE head only with features already on GPU; CUDA event timing"},
        "peak_cuda_memory_mib":peak_mem,
        "head_parameter_bytes":sum(int(v.numel()*v.element_size()) for v in state.values()),
        "all_frames_top_classes":[{"sample_id":r["sample_id"],"task":r["task_id"],"state":r["state_id"],"phase":r["phase"],"top_classes":r["official_top_classes"][:5]} for r in records],
        "limitations":["ADE20K labels are generic scene categories and have no dedicated robot/gripper/tool classes.","OpenVLA-native head probe uses cached penultimate block-22 reg4 features with different preprocessing, while the released head expects final block-23 normalized register-free DINOv2 ViT-L/14 features; probe is not a valid native segmentation result.","No quantitative oracle evaluation or foreground/background threshold selection was performed."],
        "artifacts":{"global_overview":"global_overview.png","scenario_contact_sheets":9,"per_frame_panels":len(records)}
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def render_existing(out: Path = OUT) -> None:
    """Regenerate galleries from stored prediction maps without model inference."""
    global PALETTE, CLASS_NAMES
    import torch
    _, meta = _load_official_meta(HEAD, torch)
    CLASS_NAMES = [str(x).strip() for x in meta["CLASSES"]]
    PALETTE = np.asarray(meta["PALETTE"], dtype=np.uint8)
    rows = select_samples()
    for row in rows:
        make_frame_panel(row, out / "visualizations" / row["sample_id"], CLASS_NAMES)
    for task in TASKS:
        for state_id in STATES:
            group = [r for r in rows if r["task_id"] == task and r["state_id"] == state_id]
            make_group_sheet(group, out / "contact_sheets" / f"task{task}_state{state_id}.png", CLASS_NAMES)
    make_global_overview(rows, out / "global_overview.png", CLASS_NAMES)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUT)
    p.add_argument("--limit",type=int,default=None,help="Debug only; select first N frames")
    p.add_argument("--render-only",action="store_true",help="Regenerate images from saved predictions without loading DINOv2")
    a=p.parse_args()
    if a.render_only:
        render_existing(a.output)
        print("[RENDER_ONLY_COMPLETE] reused saved segmentation predictions",flush=True)
        return
    result=run(a.output,a.limit)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
