"""Calibrate and compare lightweight foreground segmentation from native VLA features."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dataset import OUT, atomic_json, sha256, update_status

METHODS = (
    "dino_pca_per_frame",
    "dino_pca_calibration_basis",
    "dino_lda",
    "siglip_lda",
    "fused_lda",
)
METHOD_TITLES = {
    "dino_pca_per_frame": "DINOv2 PCA · per frame",
    "dino_pca_calibration_basis": "DINOv2 PCA · fixed basis",
    "dino_lda": "DINOv2 LDA",
    "siglip_lda": "SigLIP LDA",
    "fused_lda": "Fused LDA",
}
SELECTED_VARIANTS = ("current", "dilated", "source_union")
TARGET_UNSAFE_FG_REUSE = 0.05
PCA_DIM = 128
SEED = 7


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    fg = tp + fn
    bg = fp + tn
    pred_bg = fn + tn
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "foreground_recall": float(tp / fg) if fg else float("nan"),
        "foreground_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "foreground_iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "false_negative_foreground_rate": float(fn / fg) if fg else float("nan"),
        "false_positive_background_rate": float(fp / bg) if bg else float("nan"),
        "predicted_background_fraction": float(pred_bg / (fg + bg)) if fg + bg else float("nan"),
        "unsafe_fg_reuse": float(fn / fg) if fg else float("nan"),
        "reuse_opportunity": float(pred_bg / (fg + bg)) if fg + bg else float("nan"),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=bool).reshape(-1)
    prediction = np.asarray(y_pred, dtype=bool).reshape(-1)
    tp = int(np.count_nonzero(truth & prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    fn = int(np.count_nonzero(truth & ~prediction))
    tn = int(np.count_nonzero(~truth & ~prediction))
    return _metrics_from_counts(tp, fp, fn, tn)


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Expand foreground to every patch in its 8-connected one-cell neighborhood."""
    x = np.asarray(mask, dtype=bool).reshape(16, 16)
    padded = np.pad(x, 1, mode="constant", constant_values=False)
    out = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + 16, dx : dx + 16]
    return out.reshape(-1)


def _unit_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


class RandomizedPCA:
    """Small randomized PCA implementation to avoid an undeclared sklearn runtime dependency."""

    def __init__(self, n_components: int, random_state: int = SEED, oversample: int = 8) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.oversample = int(oversample)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "RandomizedPCA":
        x = np.asarray(values, dtype=np.float32)
        if x.ndim != 2 or self.n_components < 1 or self.n_components > min(x.shape) or self.n_components >= x.shape[0]:
            raise ValueError(f"Invalid PCA fit shape/components: {x.shape}, {self.n_components}")
        self.mean_ = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        centered = x - self.mean_
        sketch_dim = min(self.n_components + self.oversample, x.shape[1], x.shape[0] - 1)
        omega = np.random.default_rng(self.random_state).standard_normal((x.shape[1], sketch_dim)).astype(np.float32)
        y = centered @ omega
        # One subspace iteration improves separation of the leading directions at modest cost.
        y = centered @ (centered.T @ y)
        q, _ = np.linalg.qr(y, mode="reduced")
        compressed = q.T @ centered
        _u, singular_values, vh = np.linalg.svd(compressed, full_matrices=False)
        self.components_ = vh[: self.n_components].astype(np.float32)
        variances = np.square(singular_values.astype(np.float64))
        total = float(np.square(centered).sum(dtype=np.float64))
        self.explained_variance_ratio_ = (variances[: self.n_components] / max(total, 1e-20)).astype(np.float32)
        return self

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fit before transform.")
        x = np.asarray(values, dtype=np.float32)
        return (x - self.mean_) @ self.components_.T


class ShrinkageLDA:
    """Binary pooled-covariance LDA with a documented diagonal shrinkage target."""

    def __init__(self, shrinkage: float = 0.1) -> None:
        self.shrinkage = float(shrinkage)
        self.classes_ = np.asarray([0, 1], dtype=np.uint8)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.priors_: np.ndarray | None = None
        self.regularization_: float | None = None

    def fit(self, values: np.ndarray, labels: np.ndarray) -> "ShrinkageLDA":
        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(labels, dtype=np.uint8).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.size or set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("ShrinkageLDA needs a two-class label for each feature row.")
        x0, x1 = x[y == 0], x[y == 1]
        mean0, mean1 = x0.mean(axis=0), x1.mean(axis=0)
        centered0, centered1 = x0 - mean0, x1 - mean1
        covariance = (centered0.T @ centered0 + centered1.T @ centered1) / max(len(y) - 2, 1)
        diagonal = np.diag(covariance).copy()
        diagonal_scale = float(max(diagonal.mean(), 1e-12))
        covariance = (1.0 - self.shrinkage) * covariance + self.shrinkage * np.diag(diagonal)
        ridge = diagonal_scale * 1e-6
        difference = mean1 - mean0
        while True:
            try:
                weight = np.linalg.solve(covariance + ridge * np.eye(covariance.shape[0]), difference)
                break
            except np.linalg.LinAlgError:
                ridge *= 10
                if ridge > diagonal_scale:
                    raise
        self.priors_ = np.asarray([len(x0) / len(y), len(x1) / len(y)], dtype=np.float64)
        self.coef_ = weight.astype(np.float32)
        self.intercept_ = float(-0.5 * np.dot(mean1 + mean0, weight) + np.log(self.priors_[1] / self.priors_[0]))
        self.regularization_ = float(ridge)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LDA must be fit before prediction.")
        logits = np.asarray(values, dtype=np.float32) @ self.coef_ + self.intercept_
        probabilities = np.empty_like(logits, dtype=np.float32)
        positive = logits >= 0
        probabilities[positive] = 1 / (1 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1 + exp_logits)
        return np.column_stack([1 - probabilities, probabilities])


def _feature_for_method(sample: dict[str, np.ndarray], method: str) -> np.ndarray:
    if method in ("dino_pca_per_frame", "dino_pca_calibration_basis"):
        return sample["dino"].astype(np.float32, copy=False)
    if method == "dino_lda":
        return _unit_rows(sample["dino"])
    if method == "siglip_lda":
        return _unit_rows(sample["siglip"])
    if method == "fused_lda":
        dino = _unit_rows(sample["dino"])
        siglip = _unit_rows(sample["siglip"])
        return np.concatenate([dino, siglip], axis=-1) * np.float32(1 / math.sqrt(2))
    raise KeyError(method)


def _per_frame_pca_score(values: np.ndarray, reference: np.ndarray, polarity: float) -> tuple[np.ndarray, float]:
    x = np.asarray(values, dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    # SVD is performed per image, without labels; calibration supplies only the fixed sign convention.
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    alignment = float(np.dot(direction, reference))
    if alignment < 0:
        direction = -direction
        alignment = -alignment
    return (centered @ direction * polarity).astype(np.float32), alignment


def _threshold_candidates(calibration_scores: list[np.ndarray], method: str) -> np.ndarray:
    scores = np.concatenate([np.asarray(x).reshape(-1) for x in calibration_scores])
    if method.endswith("_lda"):
        base = np.linspace(0.02, 0.98, 49, dtype=np.float32)
    else:
        base = np.quantile(scores, np.linspace(0.02, 0.98, 49)).astype(np.float32)
    return np.unique(np.concatenate([base, np.asarray([np.nextafter(scores.min(), -np.inf), scores.max()])]))


def _select_threshold(
    scores: list[np.ndarray], labels: list[np.ndarray], method: str
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    candidates = _threshold_candidates(scores, method)
    truth = np.concatenate([x.reshape(-1).astype(bool) for x in labels])
    values = np.concatenate([x.reshape(-1) for x in scores])
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        metric = _metrics(truth, values >= threshold)
        rows.append({"method": method, "threshold": float(threshold), **metric})
    safe = [row for row in rows if row["unsafe_fg_reuse"] <= TARGET_UNSAFE_FG_REUSE]
    if safe:
        chosen = max(safe, key=lambda row: (row["reuse_opportunity"], row["foreground_recall"], -row["foreground_precision"]))
        rule = "max calibration reuse opportunity subject to unsafe foreground reuse <= 5%; ties use recall"
    else:
        chosen = min(rows, key=lambda row: (row["unsafe_fg_reuse"], -row["reuse_opportunity"]))
        rule = "no calibration threshold met the 5% safety target; chose lowest unsafe foreground reuse, then highest reuse"
    return float(chosen["threshold"]), {"rule": rule, "target_unsafe_fg_reuse": TARGET_UNSAFE_FG_REUSE, "calibration_metrics": chosen}, rows


def _fit_lda(
    method: str,
    features: dict[str, dict[str, np.ndarray]],
    train_rows: list[dict[str, Any]],
    train_labels: list[np.ndarray],
) -> tuple[Any, Any, dict[str, Any]]:
    train_x = np.concatenate([_feature_for_method(features[row["sample_id"]], method) for row in train_rows], axis=0)
    train_y = np.concatenate([x.reshape(-1).astype(np.uint8) for x in train_labels])
    if np.unique(train_y).size != 2:
        raise ValueError(f"Training split for {method} must contain both foreground and background tokens.")
    n_components = min(PCA_DIM, train_x.shape[0] - 1, train_x.shape[1])
    pca = RandomizedPCA(n_components=n_components, random_state=SEED)
    reduced = pca.fit_transform(train_x).astype(np.float32)
    lda = ShrinkageLDA(shrinkage=0.1)
    lda.fit(reduced, train_y)
    report = {
        "method": method,
        "normalization": "L2-normalize each branch per token; fused features normalize branches independently and scale by 1/sqrt(2)",
        "training_only_pca": True,
        "pca_input_dim": int(train_x.shape[1]),
        "pca_output_dim": int(n_components),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "lda_solver": "binary pooled-covariance linear discriminant",
        "lda_shrinkage": "0.10 toward the diagonal covariance target plus 1e-6 relative ridge (increased only if solve is singular)",
        "lda_actual_ridge": lda.regularization_,
        "training_tokens": int(train_y.size),
        "training_foreground_fraction": float(train_y.mean()),
    }
    return pca, lda, report


def _lda_scores(
    method: str,
    sample: dict[str, np.ndarray],
    pca: Any,
    lda: Any,
    timings: list[float] | None = None,
) -> np.ndarray:
    start = time.perf_counter()
    normalized = _feature_for_method(sample, method)
    reduced = pca.transform(normalized).astype(np.float32)
    probabilities = lda.predict_proba(reduced)
    fg_col = int(np.flatnonzero(lda.classes_ == 1)[0])
    score = probabilities[:, fg_col].astype(np.float32)
    if timings is not None:
        timings.append((time.perf_counter() - start) * 1000.0)
    return score


def _load_data(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    features: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, np.ndarray] = {}
    for row in manifest["samples"]:
        with np.load(OUT / row["feature_path"]) as archive:
            features[row["sample_id"]] = {key: archive[key].astype(np.float32) for key in ("dino", "siglip", "fused")}
        labels[row["sample_id"]] = np.load(OUT / row["oracle_token_mask_path"], allow_pickle=False).astype(bool).reshape(-1)
        if features[row["sample_id"]]["dino"].shape != (256, manifest["feature_spec"]["dino"]["feature_dim"]):
            raise ValueError(f"Bad DINO feature shape for {row['sample_id']}.")
        if features[row["sample_id"]]["siglip"].shape != (256, manifest["feature_spec"]["siglip"]["feature_dim"]):
            raise ValueError(f"Bad SigLIP feature shape for {row['sample_id']}.")
        if labels[row["sample_id"]].shape != (256,):
            raise ValueError(f"Bad oracle token label shape for {row['sample_id']}.")
    return features, labels


def _sample_metrics_row(
    row: dict[str, Any], method: str, variant: str, truth: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    metrics = _metrics(truth, prediction)
    return {
        "sample_id": row["sample_id"],
        "task_id": row["task_id"],
        "state_id": row["state_id"],
        "trajectory_id": row["trajectory_id"],
        "split": row["split"],
        "time_step": row["time_step"],
        "r5_source_frame_id": row["r5_source_frame_id"],
        "r5_age": row["r5_age"],
        "method": method,
        "variant": variant,
        **metrics,
    }


def _aggregate_metrics(
    rows: list[dict[str, Any]],
    labels: dict[str, np.ndarray],
    predictions: dict[str, dict[str, np.ndarray]],
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [row for row in rows if row["split"] == split]
    micro_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for variant in SELECTED_VARIANTS:
            truth = np.concatenate([labels[row["sample_id"]] for row in selected])
            pred = np.concatenate([predictions[method][variant][row["sample_id"]] for row in selected])
            micro = _metrics(truth, pred)
            per_traj = []
            for trajectory_id in sorted({row["trajectory_id"] for row in selected}):
                trajectory = [row for row in selected if row["trajectory_id"] == trajectory_id]
                y_true = np.concatenate([labels[row["sample_id"]] for row in trajectory])
                y_pred = np.concatenate([predictions[method][variant][row["sample_id"]] for row in trajectory])
                stats = _metrics(y_true, y_pred)
                entry = {"method": method, "variant": variant, "trajectory_id": trajectory_id,
                         "task_id": trajectory[0]["task_id"], "state_id": trajectory[0]["state_id"], **stats}
                trajectory_rows.append(entry)
                per_traj.append(stats)
            macro = {
                key: float(np.nanmean([metric[key] for metric in per_traj]))
                for key in ("foreground_recall", "foreground_precision", "foreground_iou",
                            "false_negative_foreground_rate", "false_positive_background_rate",
                            "predicted_background_fraction", "unsafe_fg_reuse", "reuse_opportunity")
            }
            micro_rows.append({"split": split, "method": method, "variant": variant, **micro,
                               **{f"mean_trajectory_{key}": value for key, value in macro.items()}})
    return micro_rows, trajectory_rows


def _threshold_sweep(
    rows: list[dict[str, Any]],
    labels: dict[str, np.ndarray],
    scores: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    eval_rows = [row for row in rows if row["split"] == "evaluation"]
    sweep: list[dict[str, Any]] = []
    for method in METHODS:
        y_true = np.concatenate([labels[row["sample_id"]] for row in eval_rows])
        y_score = np.concatenate([scores[method][row["sample_id"]] for row in eval_rows])
        for threshold in thresholds[method]:
            raw = (y_score >= threshold).reshape(-1, 256)
            for variant in ("current", "dilated"):
                pred = raw if variant == "current" else np.stack([_dilate(x) for x in raw])
                stats = _metrics(y_true, pred.reshape(-1))
                sweep.append({"split": "evaluation", "method": method, "variant": variant,
                              "threshold": float(threshold), **stats})
    return sweep


def _draw_grid(ax: Any) -> None:
    ax.set_xticks(np.arange(-0.5, 224, 14), minor=True)
    ax.set_yticks(np.arange(-0.5, 224, 14), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.65)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _overlay(rgb: np.ndarray, fg: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    mask = np.asarray(fg, dtype=bool).reshape(16, 16)
    mask_pixels = np.repeat(np.repeat(mask, 14, axis=0), 14, axis=1)
    color = np.zeros_like(image)
    color[..., 0] = 1.0
    return np.where(mask_pixels[..., None], image * (1 - alpha) + color * alpha, image)


def _error_grid(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true_fg = np.asarray(truth, dtype=bool).reshape(16, 16)
    predicted_fg = np.asarray(pred, dtype=bool).reshape(16, 16)
    # 0=TN neutral, 1=TP green, 2=FP yellow, 3=FN red.
    categories = np.zeros((16, 16), dtype=np.uint8)
    categories[true_fg & predicted_fg] = 1
    categories[~true_fg & predicted_fg] = 2
    categories[true_fg & ~predicted_fg] = 3
    colors = np.asarray([[0.18, 0.20, 0.23], [0.11, 0.66, 0.35], [1.0, 0.82, 0.16], [0.91, 0.20, 0.22]])
    return colors[categories]


def _expand_token_rgb(token_rgb: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor expand a 16×16 RGB token map to its matching policy image."""
    height, width = map(int, image_shape)
    token_rgb = np.asarray(token_rgb, dtype=np.float32)
    if token_rgb.shape != (16, 16, 3):
        raise ValueError(f"Expected a 16×16 RGB token map, got {token_rgb.shape}.")
    if height % 16 == 0 and width % 16 == 0:
        return np.repeat(
            np.repeat(token_rgb, height // 16, axis=0), width // 16, axis=1
        )
    # Preserve the discrete token colors when a policy image is not divisible by 16.
    pixels = np.clip(np.rint(token_rgb * 255), 0, 255).astype(np.uint8)
    nearest = getattr(Image, "Resampling", Image).NEAREST
    return np.asarray(
        Image.fromarray(pixels).resize((width, height), resample=nearest),
        dtype=np.float32,
    ) / 255.0


def _pillow_font(size: int) -> Any:
    from PIL import ImageFont

    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _pillow_grid(image: Any) -> Any:
    from PIL import ImageDraw

    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for x in range(0, 224, 14):
        draw.line((x, 0, x, 223), fill=(255, 255, 255), width=1)
    for y in range(0, 224, 14):
        draw.line((0, y, 223, y), fill=(255, 255, 255), width=1)
    return image


def _save_visuals_pillow(
    manifest: dict[str, Any], rows: list[dict[str, Any]], labels: dict[str, np.ndarray],
    scores: dict[str, dict[str, np.ndarray]], thresholds: dict[str, float],
    predictions: dict[str, dict[str, dict[str, np.ndarray]]], reason: str,
) -> dict[str, Any]:
    """Portable figure fallback for containers whose Matplotlib binary is ABI-incompatible."""
    import csv
    from PIL import Image, ImageDraw

    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    lanczos = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    row_by_id = {row["sample_id"]: row for row in rows}
    method_order = list(METHODS)
    avg_iou = {
        row["sample_id"]: float(np.mean([_metrics(labels[row["sample_id"]], predictions[method]["current"][row["sample_id"]])["foreground_iou"] for method in METHODS]))
        for row in rows
    }
    selected_ids: list[str] = []
    for task_id, state_id in ((4, 0), (5, 1), (7, 2)):
        candidates = [row for row in rows if row["task_id"] == task_id and row["state_id"] == state_id]
        if candidates:
            candidates.sort(key=lambda row: avg_iou[row["sample_id"]])
            selected_ids.append(candidates[len(candidates) // 2]["sample_id"])
    eval_rows = [row for row in rows if row["split"] == "evaluation"]
    if eval_rows:
        for candidate in (max(eval_rows, key=lambda row: avg_iou[row["sample_id"]]),
                          min(eval_rows, key=lambda row: avg_iou[row["sample_id"]])):
            if candidate["sample_id"] not in selected_ids:
                selected_ids.append(candidate["sample_id"])
    selected_ids = selected_ids[:6]
    font = _pillow_font(14)
    small = _pillow_font(11)
    tiny = _pillow_font(9)
    tile_w, tile_h, image_size = 250, 274, 224
    pdf_pages = []

    def panel(rgb: np.ndarray, kind: str, sid: str, method: str | None = None, variant: str = "current") -> Any:
        if kind == "rgb":
            return Image.fromarray(rgb).convert("RGB")
        if kind == "oracle":
            mask = labels[sid]
            return _pillow_grid(Image.fromarray(np.asarray(np.clip(_overlay(rgb, mask) * 255, 0, 255), dtype=np.uint8)))
        if kind == "oracle_map":
            mask = labels[sid].reshape(16, 16)
            colors = np.zeros((16, 16, 3), dtype=np.uint8)
            colors[mask] = (225, 48, 52)
            return Image.fromarray(colors).resize((224, 224), nearest)
        assert method is not None
        prediction = predictions[method][variant][sid]
        if kind == "error":
            image = Image.fromarray(np.asarray(np.clip(_error_grid(labels[sid], prediction) * 255, 0, 255), dtype=np.uint8))
            return image.resize((224, 224), nearest)
        return _pillow_grid(Image.fromarray(np.asarray(np.clip(_overlay(rgb, prediction) * 255, 0, 255), dtype=np.uint8)))

    def sheet(title: str, columns: list[str], ids: list[str], builder: Any, filename: str) -> Any:
        margin = 18
        header_h = 54
        canvas = Image.new("RGB", (margin * 2 + tile_w * len(columns), margin * 2 + header_h + tile_h * len(ids)), (20, 24, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, margin), title, font=font, fill=(240, 243, 247))
        top = margin + header_h
        for ci, column in enumerate(columns):
            x = margin + ci * tile_w
            draw.multiline_text((x + 2, margin + 24), column, font=small, fill=(210, 218, 226), spacing=1)
        for ri, sid in enumerate(ids):
            row = row_by_id[sid]
            rgb = np.asarray(Image.open(OUT / row["rgb_path"]).convert("RGB"))
            y = top + ri * tile_h
            caption = f"{row['trajectory_id']} · t={row['time_step']} · {row['split']}"
            draw.text((margin, y), caption[:54], font=tiny, fill=(190, 200, 210))
            for ci, column in enumerate(columns):
                x = margin + ci * tile_w
                image = builder(sid, row, rgb, ci, column)
                if image.size != (image_size, image_size):
                    image = image.resize((image_size, image_size), nearest if "error" in column.lower() else lanczos)
                canvas.paste(image, (x, y + 20))
                draw.rectangle((x, y + 20, x + 223, y + 243), outline=(70, 77, 86), width=1)
        path = figures / filename
        canvas.save(path, format="PNG", optimize=True)
        pdf_pages.append(canvas.copy())
        return canvas

    columns = ["RGB observation", "Oracle FG/BG"] + [METHOD_TITLES[method] for method in method_order]
    def overlay_builder(sid: str, row: dict[str, Any], rgb: np.ndarray, ci: int, _column: str) -> Any:
        if ci == 0: return panel(rgb, "rgb", sid)
        if ci == 1: return panel(rgb, "oracle", sid)
        method = method_order[ci - 2]
        image = panel(rgb, "prediction", sid, method)
        stats = _metrics(labels[sid], predictions[method]["current"][sid])
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 187, 224, 224), fill=(0, 0, 0))
        draw.text((4, 190), f"R {stats['foreground_recall']:.2f}  IoU {stats['foreground_iou']:.2f}", font=tiny, fill=(255, 255, 255))
        draw.text((4, 205), f"BG reusable {stats['reuse_opportunity']:.0%}", font=tiny, fill=(255, 255, 255))
        return image
    sheet("Track B · matched segmentation overlays", columns, selected_ids, overlay_builder, "segmentation_overlays.png")

    error_columns = ["RGB observation", "Oracle FG"] + [METHOD_TITLES[m] for m in method_order]
    def error_builder(sid: str, row: dict[str, Any], rgb: np.ndarray, ci: int, _column: str) -> Any:
        if ci == 0: return panel(rgb, "rgb", sid)
        if ci == 1: return panel(rgb, "oracle_map", sid)
        method = method_order[ci - 2]
        image = panel(rgb, "error", sid, method)
        stats = _metrics(labels[sid], predictions[method]["current"][sid])
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 198, 224, 224), fill=(0, 0, 0))
        draw.text((3, 201), f"R {stats['foreground_recall']:.2f} IoU {stats['foreground_iou']:.2f} BG {stats['reuse_opportunity']:.0%}", font=tiny, fill="white")
        return image
    sheet("Token errors: TP green · FP yellow · FN red · TN dark", error_columns, selected_ids, error_builder, "error_maps.png")

    dilation_columns = ["RGB", "Oracle"] + [title for method in method_order for title in (METHOD_TITLES[method] + "\nbase", "one-patch dilation")]
    def dilation_builder(sid: str, row: dict[str, Any], rgb: np.ndarray, ci: int, _column: str) -> Any:
        if ci == 0: return panel(rgb, "rgb", sid)
        if ci == 1: return panel(rgb, "oracle", sid)
        method = method_order[(ci - 2) // 2]
        variant = "current" if (ci - 2) % 2 == 0 else "dilated"
        return panel(rgb, "prediction", sid, method, variant)
    sheet("Predicted foreground before/after one 8-neighbor patch dilation", dilation_columns, selected_ids[:3], dilation_builder, "dilation_comparison.png")

    temporal_meta = []
    for task_id in (4, 5, 7):
        candidates = [row for row in eval_rows if row["task_id"] == task_id and row["r5_age"] == 0]
        if not candidates: continue
        source = candidates[len(candidates) // 2]
        sequence = [row for row in rows if row["trajectory_id"] == source["trajectory_id"] and row["r5_source_frame_id"] == source["sample_id"]]
        sequence.sort(key=lambda row: row["r5_age"])
        if len(sequence) != 5: continue
        columns_t = ["RGB", "Oracle"] + [METHOD_TITLES[method] for method in method_order]
        sequence_ids = [row["sample_id"] for row in sequence]
        def temporal_builder(sid: str, row: dict[str, Any], rgb: np.ndarray, ci: int, _column: str) -> Any:
            if ci == 0: return panel(rgb, "rgb", sid)
            if ci == 1: return panel(rgb, "oracle", sid)
            return panel(rgb, "prediction", sid, method_order[ci - 2])
        filename = f"temporal_task{task_id}_{source['sample_id']}.png"
        sheet(f"R5 window · task {task_id} · source step {source['time_step']}", columns_t, sequence_ids, temporal_builder, filename)
        temporal_meta.append({"task_id": task_id, "trajectory_id": source["trajectory_id"], "source_sample_id": source["sample_id"],
                              "steps": [row["time_step"] for row in sequence], "figure": filename})

    audit = next(row for row in eval_rows)
    audit_rgb = np.asarray(Image.open(OUT / audit["rgb_path"]).convert("RGB"))
    mapping_canvas = Image.new("RGB", (720, 290), (20, 24, 30))
    mapping_draw = ImageDraw.Draw(mapping_canvas)
    mapping_draw.text((12, 8), "224×224 OpenVLA processor input → 16×16 token grid", font=font, fill=(245, 245, 245))
    mapping_canvas.paste(Image.fromarray(audit_rgb), (10, 35))
    mapping_canvas.paste(panel(audit_rgb, "oracle", audit["sample_id"]), (250, 35))
    mapping_canvas.paste(Image.fromarray(labels[audit["sample_id"]].reshape(16, 16).astype(np.uint8) * 255).resize((224, 224), nearest), (490, 35))
    mapping_draw.text((10, 264), "RGB · orientation/crop shared with oracle", font=tiny, fill=(230, 230, 230))
    mapping_draw.text((250, 264), "Oracle over image · 14-pixel token grid", font=tiny, fill=(230, 230, 230))
    mapping_draw.text((490, 264), "256 visual token labels", font=tiny, fill=(230, 230, 230))
    mapping_canvas.save(figures / "feature_mapping_overlay.png", optimize=True)
    pdf_pages.append(mapping_canvas.copy())

    sweep_rows = list(csv.DictReader((OUT / "threshold_sweep.csv").open(encoding="utf-8")))
    chart = Image.new("RGB", (1200, 800), (250, 250, 250))
    draw = ImageDraw.Draw(chart)
    draw.text((50, 20), "Reuse opportunity vs unsafe foreground reuse · evaluation threshold sweep", font=font, fill=(15, 20, 27))
    x0, y0, x1, y1 = 90, 80, 850, 690
    for step in range(6):
        x = x0 + (x1 - x0) * step / 5
        y = y0 + (y1 - y0) * step / 5
        draw.line((x, y0, x, y1), fill=(215, 218, 222), width=1)
        draw.line((x0, y, x1, y), fill=(215, 218, 222), width=1)
        draw.text((x - 10, y1 + 8), f"{step/5:.1f}", font=tiny, fill=(40, 40, 40))
        draw.text((x0 - 45, y - 5), f"{1-step/5:.1f}", font=tiny, fill=(40, 40, 40))
    draw.text((330, 740), "Reuse opportunity · predicted background fraction", font=small, fill=(20, 20, 20))
    draw.text((12, 340), "Unsafe foreground reuse", font=small, fill=(20, 20, 20))
    palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
    for method_index, method in enumerate(METHODS):
        for variant, dashed in (("current", False), ("dilated", True)):
            subset = sorted((item for item in sweep_rows if item["method"] == method and item["variant"] == variant),
                            key=lambda item: float(item["reuse_opportunity"]))
            points = []
            for item in subset:
                x = x0 + float(item["reuse_opportunity"]) * (x1 - x0)
                y = y1 - float(item["unsafe_fg_reuse"]) * (y1 - y0)
                points.append((x, y))
            for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
                if not dashed or index % 2 == 0:
                    draw.line((a, b), fill=palette[method_index], width=3 if not dashed else 2)
            legend_y = 85 + method_index * 56 + (20 if dashed else 0)
            draw.line((890, legend_y + 5, 930, legend_y + 5), fill=palette[method_index], width=3 if not dashed else 2)
            draw.text((940, legend_y - 3), METHOD_TITLES[method] + (" · dilated" if dashed else " · current"), font=tiny, fill=(30, 35, 42))
        union_truth = np.concatenate([labels[row["sample_id"]] for row in eval_rows])
        union_predictions = np.concatenate([predictions[method]["source_union"][row["sample_id"]] for row in eval_rows])
        union_stats = _metrics(union_truth, union_predictions)
        union_x = x0 + union_stats["reuse_opportunity"] * (x1 - x0)
        union_y = y1 - union_stats["unsafe_fg_reuse"] * (y1 - y0)
        draw.line((union_x - 6, union_y - 6, union_x + 6, union_y + 6), fill=palette[method_index], width=3)
        draw.line((union_x - 6, union_y + 6, union_x + 6, union_y - 6), fill=palette[method_index], width=3)
    safety_y = y1 - TARGET_UNSAFE_FG_REUSE * (y1 - y0)
    for x in range(x0, x1, 16): draw.line((x, safety_y, min(x + 8, x1), safety_y), fill=(20, 20, 20), width=2)
    draw.text((890, 410), "Dashed black: 5% calibration safety target", font=tiny, fill=(30, 30, 30))
    chart.save(figures / "reuse_opportunity_vs_unsafe_fg_reuse.png", optimize=True)
    pdf_pages.append(chart.copy())

    pdf_path = figures / "comparison_report.pdf"
    pdf_pages[0].save(pdf_path, "PDF", save_all=True, append_images=pdf_pages[1:], resolution=160)
    return {
        "renderer": "Pillow fallback",
        "matplotlib_import_error": reason,
        "representative_sample_ids": selected_ids,
        "representative_sample_avg_iou": {sid: avg_iou[sid] for sid in selected_ids},
        "temporal_sequences": temporal_meta,
        "files": sorted(path.name for path in figures.iterdir() if path.is_file()),
    }


def _save_visuals(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    labels: dict[str, np.ndarray],
    scores: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:
        return _save_visuals_pillow(manifest, rows, labels, scores, thresholds, predictions,
                                    f"{type(exc).__name__}: {exc}")

    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    row_by_id = {row["sample_id"]: row for row in rows}
    avg_iou: dict[str, float] = {}
    for row in rows:
        sid = row["sample_id"]
        values = []
        for method in METHODS:
            stat = _metrics(labels[sid], predictions[method]["current"][sid])
            values.append(float(stat["foreground_iou"]))
        avg_iou[sid] = float(np.mean(values))

    selected_ids: list[str] = []
    for task_id, state_id in ((4, 0), (5, 1), (7, 2)):
        candidates = [r for r in rows if r["task_id"] == task_id and r["state_id"] == state_id]
        if candidates:
            candidates.sort(key=lambda r: avg_iou[r["sample_id"]])
            selected_ids.append(candidates[len(candidates) // 2]["sample_id"])
    eval_rows = [r for r in rows if r["split"] == "evaluation"]
    if eval_rows:
        best = max(eval_rows, key=lambda r: avg_iou[r["sample_id"]])["sample_id"]
        worst = min(eval_rows, key=lambda r: avg_iou[r["sample_id"]])["sample_id"]
        for candidate in (best, worst):
            if candidate not in selected_ids:
                selected_ids.append(candidate)
    selected_ids = selected_ids[:6]
    method_order = list(METHODS)
    common_pdf_path = figures / "comparison_report.pdf"

    with PdfPages(common_pdf_path) as pdf:
        # Matched image panels: original, oracle and the same five predictions for each observation.
        fig, axes = plt.subplots(len(selected_ids), 2 + len(METHODS), figsize=(3.05 * (2 + len(METHODS)), 3.05 * len(selected_ids)), squeeze=False)
        for column, title in enumerate(["RGB observation", "Oracle FG/BG"] + [METHOD_TITLES[m] for m in method_order]):
            axes[0, column].set_title(title, fontsize=10, pad=7)
        for r_index, sid in enumerate(selected_ids):
            row = row_by_id[sid]
            rgb = np.asarray(Image.open(OUT / row["rgb_path"]).convert("RGB"))
            truth = labels[sid]
            axes[r_index, 0].imshow(rgb)
            axes[r_index, 1].imshow(_overlay(rgb, truth))
            _draw_grid(axes[r_index, 1])
            axes[r_index, 0].set_ylabel(f"{row['trajectory_id']} · t={row['time_step']} · {row['split']}", fontsize=8)
            for index, method in enumerate(method_order, start=2):
                pred = predictions[method]["current"][sid]
                axes[r_index, index].imshow(_overlay(rgb, pred))
                _draw_grid(axes[r_index, index])
                stats = _metrics(truth, pred)
                axes[r_index, index].text(
                    0.02, 0.02,
                    f"R {stats['foreground_recall']:.2f}  IoU {stats['foreground_iou']:.2f}\nBG {stats['reuse_opportunity']:.0%}",
                    transform=axes[r_index, index].transAxes, fontsize=7, color="white",
                    bbox={"facecolor": "black", "alpha": 0.68, "pad": 2},
                )
        for ax in axes.flat:
            ax.set_axis_off()
        fig.suptitle("Track B · same observations across native-feature segmentation methods", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(figures / "segmentation_overlays.png", dpi=200, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Matched error maps with explicit TP/FP/FN/TN colors.
        fig, axes = plt.subplots(len(selected_ids), 2 + len(METHODS), figsize=(3.0 * (2 + len(METHODS)), 2.9 * len(selected_ids)), squeeze=False)
        axes[0, 0].set_title("RGB observation", fontsize=10)
        axes[0, 1].set_title("Oracle FG", fontsize=10)
        for i, method in enumerate(method_order, start=2):
            axes[0, i].set_title(METHOD_TITLES[method], fontsize=9)
        for r_index, sid in enumerate(selected_ids):
            row = row_by_id[sid]
            rgb = np.asarray(Image.open(OUT / row["rgb_path"]).convert("RGB"))
            truth = labels[sid]
            axes[r_index, 0].imshow(rgb)
            oracle_grid = np.zeros((16, 16, 3), dtype=np.float32)
            oracle_grid[truth.reshape(16, 16)] = (0.9, 0.3, 0.2)
            axes[r_index, 1].imshow(_expand_token_rgb(oracle_grid, rgb.shape[:2]))
            _draw_grid(axes[r_index, 1])
            axes[r_index, 0].set_ylabel(f"{row['trajectory_id']} · t={row['time_step']}", fontsize=8)
            for i, method in enumerate(method_order, start=2):
                pred = predictions[method]["current"][sid]
                # The grid overlay uses 224-pixel coordinates. Keep the error image in
                # the same pixel extent/aspect as RGB; plotting the raw 16×16 map makes
                # those fixed grid ticks expand the axes and leaves the mask tiny.
                error_image = _expand_token_rgb(_error_grid(truth, pred), rgb.shape[:2])
                axes[r_index, i].imshow(error_image, interpolation="nearest")
                _draw_grid(axes[r_index, i])
                stats = _metrics(truth, pred)
                axes[r_index, i].text(
                    0.02, 0.02,
                    f"R {stats['foreground_recall']:.2f} · IoU {stats['foreground_iou']:.2f} · BG {stats['reuse_opportunity']:.0%}",
                    transform=axes[r_index, i].transAxes, fontsize=6.7, color="white",
                    bbox={"facecolor": "black", "alpha": 0.7, "pad": 2},
                )
        for ax in axes.flat:
            ax.set_axis_off()
        legend = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#1ca859", markersize=9, label="TP foreground"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#ffd129", markersize=9, label="FP foreground"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e83338", markersize=9, label="FN foreground"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2e333a", markersize=9, label="TN background"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
        fig.suptitle("Token-level error maps · panels span splits; reported metrics use evaluation only", fontsize=13)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        fig.savefig(figures / "error_maps.png", dpi=220, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # A before/after one-token dilation comparison on three varied examples.
        dilation_ids = selected_ids[:3]
        fig, axes = plt.subplots(len(dilation_ids), 2 + 2 * len(METHODS), figsize=(2.35 * (2 + 2 * len(METHODS)), 2.55 * len(dilation_ids)), squeeze=False)
        axes[0, 0].set_title("RGB", fontsize=8)
        axes[0, 1].set_title("Oracle", fontsize=8)
        for i, method in enumerate(method_order):
            axes[0, 2 + 2 * i].set_title(METHOD_TITLES[method] + "\nbase", fontsize=8)
            axes[0, 3 + 2 * i].set_title("one-patch dilation", fontsize=8)
        for r_index, sid in enumerate(dilation_ids):
            row = row_by_id[sid]
            rgb = np.asarray(Image.open(OUT / row["rgb_path"]).convert("RGB"))
            axes[r_index, 0].imshow(rgb)
            axes[r_index, 1].imshow(_overlay(rgb, labels[sid]))
            _draw_grid(axes[r_index, 1])
            axes[r_index, 0].set_ylabel(f"{row['trajectory_id']} · t={row['time_step']}", fontsize=8)
            for i, method in enumerate(method_order):
                base = predictions[method]["current"][sid]
                axes[r_index, 2 + 2 * i].imshow(_overlay(rgb, base))
                axes[r_index, 3 + 2 * i].imshow(_overlay(rgb, _dilate(base)))
                _draw_grid(axes[r_index, 2 + 2 * i])
                _draw_grid(axes[r_index, 3 + 2 * i])
        for ax in axes.flat:
            ax.set_axis_off()
        fig.suptitle("Predicted foreground before and after 8-neighbor one-patch dilation", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(figures / "dilation_comparison.png", dpi=220, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Three 5-frame sequences show same R5 source/current windows, one per task.
        temporal_meta: list[dict[str, Any]] = []
        sequence_specs = []
        for task_id in (4, 5, 7):
            candidates = [r for r in rows if r["task_id"] == task_id and r["split"] == "evaluation" and r["r5_age"] == 0]
            if not candidates:
                continue
            source = candidates[len(candidates) // 2]
            group = [r for r in rows if r["trajectory_id"] == source["trajectory_id"] and r["r5_source_frame_id"] == source["sample_id"]]
            group.sort(key=lambda item: item["r5_age"])
            if len(group) == 5:
                sequence_specs.append(group)
        for sequence in sequence_specs:
            source_id = sequence[0]["sample_id"]
            task_id = sequence[0]["task_id"]
            fig, axes = plt.subplots(5, 2 + len(METHODS), figsize=(2.75 * (2 + len(METHODS)), 2.75 * 5), squeeze=False)
            for i, title in enumerate(["RGB", "Oracle"] + [METHOD_TITLES[m] for m in method_order]):
                axes[0, i].set_title(title, fontsize=9)
            for r_index, row in enumerate(sequence):
                sid = row["sample_id"]
                rgb = np.asarray(Image.open(OUT / row["rgb_path"]).convert("RGB"))
                axes[r_index, 0].imshow(rgb)
                axes[r_index, 0].set_ylabel(f"age {row['r5_age']}\nt={row['time_step']}", fontsize=8)
                axes[r_index, 1].imshow(_overlay(rgb, labels[sid]))
                _draw_grid(axes[r_index, 1])
                for i, method in enumerate(method_order, start=2):
                    axes[r_index, i].imshow(_overlay(rgb, predictions[method]["current"][sid]))
                    _draw_grid(axes[r_index, i])
            for ax in axes.flat:
                ax.set_axis_off()
            fig.suptitle(f"R5 temporal window · task {task_id} · {sequence[0]['trajectory_id']} · source step {sequence[0]['time_step']}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            filename = f"temporal_task{task_id}_{source_id}.png"
            fig.savefig(figures / filename, dpi=210, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            temporal_meta.append({"task_id": task_id, "trajectory_id": sequence[0]["trajectory_id"],
                                  "source_sample_id": source_id, "steps": [r["time_step"] for r in sequence], "figure": filename})

        # Explicit patch-to-token visual audit.
        audit_row = next(r for r in rows if r["split"] == "evaluation")
        sid = audit_row["sample_id"]
        rgb = np.asarray(Image.open(OUT / audit_row["rgb_path"]).convert("RGB"))
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        axes[0].imshow(rgb); axes[0].set_title("OpenVLA processor input")
        axes[1].imshow(_overlay(rgb, labels[sid])); _draw_grid(axes[1]); axes[1].set_title("Oracle token mask on 16×16 patch grid")
        axes[2].imshow(labels[sid].reshape(16, 16), cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        axes[2].set_xticks(np.arange(-0.5, 16, 1), minor=True); axes[2].set_yticks(np.arange(-0.5, 16, 1), minor=True)
        axes[2].grid(which="minor", color="black", linewidth=0.35); axes[2].set_title("256 token foreground mask")
        for ax in axes: ax.set_axis_off()
        fig.suptitle("Spatial correspondence: 224×224 processor image → 16×16 OpenVLA visual tokens", fontsize=12)
        fig.tight_layout()
        fig.savefig(figures / "feature_mapping_overlay.png", dpi=240, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Reuse opportunity versus unsafe foreground reuse, with dilation as a second curve.
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
        sweep_rows = list(csv.DictReader((OUT / "threshold_sweep.csv").open(encoding="utf-8")))
        for color, method in zip(colors, METHODS):
            for variant, style in (("current", "-"), ("dilated", "--")):
                subset = [r for r in sweep_rows if r["method"] == method and r["variant"] == variant]
                subset.sort(key=lambda r: float(r["reuse_opportunity"]))
                ax.plot([float(r["reuse_opportunity"]) for r in subset],
                        [float(r["unsafe_fg_reuse"]) for r in subset],
                        linestyle=style, marker="o", markersize=2.8, linewidth=1.3,
                        color=color, label=f"{METHOD_TITLES[method]} · {variant}")
            selected = predictions[method]["source_union"]
            y_true = np.concatenate([labels[r["sample_id"]] for r in eval_rows])
            y_union = np.concatenate([selected[r["sample_id"]] for r in eval_rows])
            stat = _metrics(y_true, y_union)
            ax.scatter([stat["reuse_opportunity"]], [stat["unsafe_fg_reuse"]], color=color,
                       marker="x", s=50, zorder=5)
        ax.axhline(TARGET_UNSAFE_FG_REUSE, color="black", linestyle=":", linewidth=1, label="5% calibration safety target")
        ax.set_xlabel("Reuse opportunity · fraction of tokens predicted background")
        ax.set_ylabel("Unsafe foreground reuse · oracle foreground predicted background")
        ax.set_title("Threshold sweep on held-out evaluation trajectories\nSolid=current; dashed=one-patch dilation; ×=source/current union at calibrated threshold")
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(figures / "reuse_opportunity_vs_unsafe_fg_reuse.png", dpi=220, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return {
        "representative_sample_ids": selected_ids,
        "representative_sample_avg_iou": {sid: avg_iou[sid] for sid in selected_ids},
        "temporal_sequences": temporal_meta,
        "files": sorted(p.name for p in figures.iterdir() if p.is_file()),
    }


def run_evaluation(manifest: dict[str, Any]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    update_status("running", phase="evaluation", completed_samples=len(manifest["samples"]))
    rows = manifest["samples"]
    features, labels = _load_data(manifest)
    training_rows = [row for row in rows if row["split"] == "training"]
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    evaluation_rows = [row for row in rows if row["split"] == "evaluation"]
    if not training_rows or not calibration_rows or not evaluation_rows:
        raise ValueError("Training, calibration, and evaluation splits must all contain complete trajectories.")
    if set(r["trajectory_id"] for r in training_rows) & set(r["trajectory_id"] for r in calibration_rows + evaluation_rows):
        raise AssertionError("Training trajectory leakage detected.")
    if set(r["trajectory_id"] for r in calibration_rows) & set(r["trajectory_id"] for r in evaluation_rows):
        raise AssertionError("Calibration trajectory leakage detected.")

    # PCA baselines: basis fit only on calibration observations; polarity and thresholds use calibration labels.
    cal_dino = np.concatenate([features[row["sample_id"]]["dino"] for row in calibration_rows], axis=0).astype(np.float32)
    fixed_pca = RandomizedPCA(n_components=1, random_state=SEED, oversample=4)
    fixed_pca.fit(cal_dino)
    calibration_fixed_raw = [fixed_pca.transform(features[row["sample_id"]]["dino"]).reshape(-1) for row in calibration_rows]
    calibration_labels = [labels[row["sample_id"]] for row in calibration_rows]
    calibration_y = np.concatenate(calibration_labels)
    calibration_fixed = np.concatenate(calibration_fixed_raw)
    fixed_polarity = 1.0 if calibration_fixed[calibration_y].mean() >= calibration_fixed[~calibration_y].mean() else -1.0
    calibration_fixed = [np.asarray(x * fixed_polarity, dtype=np.float32) for x in calibration_fixed_raw]
    calibration_frame: list[np.ndarray] = []
    frame_alignments: list[float] = []
    reference = fixed_pca.components_[0].astype(np.float32)
    for row in calibration_rows:
        frame_score, alignment = _per_frame_pca_score(features[row["sample_id"]]["dino"], reference, fixed_polarity)
        calibration_frame.append(frame_score)
        frame_alignments.append(alignment)
    frame_y = np.concatenate(calibration_labels)
    frame_flat = np.concatenate(calibration_frame)
    # The per-frame sign is fixed by alignment to the calibration PCA direction; only one global label-calibrated polarity is used.
    frame_orientation = 1.0 if frame_flat[frame_y].mean() >= frame_flat[~frame_y].mean() else -1.0
    calibration_frame = [x * frame_orientation for x in calibration_frame]
    frame_polarity = fixed_polarity * frame_orientation

    scores: dict[str, dict[str, np.ndarray]] = {method: {} for method in METHODS}
    calibration_scores: dict[str, list[np.ndarray]] = {
        "dino_pca_per_frame": calibration_frame,
        "dino_pca_calibration_basis": calibration_fixed,
    }
    calibration_frame_by_id = {row["sample_id"]: score for row, score in zip(calibration_rows, calibration_frame)}
    calibration_fixed_by_id = {row["sample_id"]: score for row, score in zip(calibration_rows, calibration_fixed)}
    for row in rows:
        sid = row["sample_id"]
        if row["split"] == "calibration":
            scores["dino_pca_per_frame"][sid] = calibration_frame_by_id[sid]
            scores["dino_pca_calibration_basis"][sid] = calibration_fixed_by_id[sid]
        else:
            frame_score, _alignment = _per_frame_pca_score(features[sid]["dino"], reference, frame_polarity)
            scores["dino_pca_per_frame"][sid] = frame_score
            scores["dino_pca_calibration_basis"][sid] = fixed_pca.transform(features[sid]["dino"]).reshape(-1).astype(np.float32) * fixed_polarity

    model_reports: list[dict[str, Any]] = [
        {"method": "dino_pca_per_frame", "basis": "unsupervised per-frame first principal component", "polarity": "calibration label-selected global polarity after alignment to calibration PCA direction", "reference_direction_cosine_abs_mean": float(np.mean(frame_alignments)), "labels_used_for_calibration": True},
        {"method": "dino_pca_calibration_basis", "basis": "first component fit on calibration trajectories and held fixed", "polarity": "selected by foreground/background score difference on calibration labels", "labels_used_for_calibration": True},
    ]
    threshold_values: dict[str, float] = {}
    threshold_info: dict[str, dict[str, Any]] = {}
    calibration_threshold_rows: list[dict[str, Any]] = []
    for method in METHODS[:2]:
        value, info, details = _select_threshold(calibration_scores[method], calibration_labels, method)
        threshold_values[method], threshold_info[method] = value, info

    classifier_timings: dict[str, list[float]] = {method: [] for method in METHODS}
    lda_models: dict[str, tuple[Any, Any]] = {}
    for method in METHODS[2:]:
        lda_method = "fused_lda" if method == "fused_lda" else method
        pca, lda, report = _fit_lda(lda_method, features, training_rows, [labels[r["sample_id"]] for r in training_rows])
        lda_models[method] = (pca, lda)
        model_reports.append(report)
        cal_scores = []
        for row in calibration_rows:
            sid = row["sample_id"]
            score = _lda_scores(method, features[sid], pca, lda)
            scores[method][sid] = score
            cal_scores.append(score)
        calibration_scores[method] = cal_scores
        value, info, details = _select_threshold(cal_scores, calibration_labels, method)
        threshold_values[method], threshold_info[method] = value, info
        calibration_threshold_rows.extend(details)
        for row in rows:
            if row["split"] == "calibration":
                continue
            sid = row["sample_id"]
            score = _lda_scores(method, features[sid], pca, lda, classifier_timings[method] if row["split"] == "evaluation" else None)
            scores[method][sid] = score

    # Time the two PCA projection paths on evaluation frames separately from backbone execution.
    pca_timings: dict[str, list[float]] = {method: [] for method in METHODS[:2]}
    for row in evaluation_rows:
        sid = row["sample_id"]
        values = features[sid]["dino"]
        started = time.perf_counter()
        _score, _alignment = _per_frame_pca_score(values, reference, frame_polarity)
        pca_timings["dino_pca_per_frame"].append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        fixed_pca.transform(values)
        pca_timings["dino_pca_calibration_basis"].append((time.perf_counter() - started) * 1000.0)

    for method in METHODS:
        if method.endswith("_lda"):
            continue
        # Scores for calibration rows were generated above; eval PCA score times are measured separately.
    for method in METHODS:
        value, info, details = _select_threshold(calibration_scores[method], calibration_labels, method)
        threshold_values[method], threshold_info[method] = value, info
        if method in METHODS[2:]:
            calibration_threshold_rows.extend(details)

    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for method in METHODS:
        predictions[method] = {variant: {} for variant in SELECTED_VARIANTS}
        threshold = threshold_values[method]
        raw_by_id = {row["sample_id"]: (scores[method][row["sample_id"]] >= threshold) for row in rows}
        for row in rows:
            sid = row["sample_id"]
            current = raw_by_id[sid].astype(bool)
            predictions[method]["current"][sid] = current
            predictions[method]["dilated"][sid] = _dilate(current)
            source_id = row["r5_source_frame_id"]
            predictions[method]["source_union"][sid] = current | raw_by_id[source_id].astype(bool)

    eval_micro, trajectory_metrics = _aggregate_metrics(rows, labels, predictions, "evaluation")
    sample_metric_rows: list[dict[str, Any]] = []
    for row in evaluation_rows:
        sid = row["sample_id"]
        for method in METHODS:
            for variant in SELECTED_VARIANTS:
                sample_metric_rows.append(_sample_metrics_row(row, method, variant, labels[sid], predictions[method][variant][sid]))

    # R5 temporal agreement is reported independently from segmentation accuracy.
    temporal_rows = []
    for method in METHODS:
        pair_agreement: list[float] = []
        for row in evaluation_rows:
            if row["r5_age"] == 4:
                continue
            next_row = next((candidate for candidate in evaluation_rows
                             if candidate["trajectory_id"] == row["trajectory_id"]
                             and candidate["r5_source_frame_id"] == row["r5_source_frame_id"]
                             and candidate["r5_age"] == row["r5_age"] + 1), None)
            if next_row is None:
                continue
            a = predictions[method]["current"][row["sample_id"]]
            b = predictions[method]["current"][next_row["sample_id"]]
            agreement = float(np.mean(a == b))
            pair_agreement.append(agreement)
            temporal_rows.append({"method": method, "trajectory_id": row["trajectory_id"],
                                  "source_frame_id": row["r5_source_frame_id"], "from_age": row["r5_age"],
                                  "to_age": next_row["r5_age"], "token_agreement": agreement,
                                  "token_disagreement": 1 - agreement})
        temporal_rows.append({"method": method, "trajectory_id": "ALL_EVALUATION", "source_frame_id": "ALL_R5_WINDOWS",
                              "from_age": "aggregate", "to_age": "aggregate",
                              "token_agreement": float(np.mean(pair_agreement)) if pair_agreement else float("nan"),
                              "token_disagreement": float(1 - np.mean(pair_agreement)) if pair_agreement else float("nan")})

    thresholds_for_sweep = {method: _threshold_candidates(calibration_scores[method], method) for method in METHODS}
    threshold_sweep = _threshold_sweep(rows, labels, scores, thresholds_for_sweep)
    _write_csv(OUT / "threshold_sweep.csv", threshold_sweep)
    _write_csv(OUT / "metrics.csv", eval_micro)
    _write_csv(OUT / "per_sample_metrics.csv", sample_metric_rows)
    _write_csv(OUT / "per_trajectory_metrics.csv", trajectory_metrics)
    _write_csv(OUT / "temporal_consistency.csv", temporal_rows)
    _write_csv(OUT / "calibration_thresholds.csv", calibration_threshold_rows)

    classifier_overhead: dict[str, dict[str, float | None]] = {}
    for method in METHODS:
        values = classifier_timings[method] if method in classifier_timings else pca_timings.get(method, [])
        classifier_overhead[method] = {
            "median_ms_per_observation": float(np.median(values)) if values else None,
            "p95_ms_per_observation": float(np.percentile(values, 95)) if values else None,
            "timing_split": "evaluation only",
            "includes": "feature normalization/PCA projection/classifier score calculation; excludes vision-backbone forward",
        }

    for row in eval_micro:
        method = row["method"]
        row["selected_calibration_threshold"] = threshold_values[method]
        row["calibration_threshold_rule"] = threshold_info[method]["rule"]
        row["classifier_median_ms"] = classifier_overhead[method]["median_ms_per_observation"]
        row["classifier_p95_ms"] = classifier_overhead[method]["p95_ms_per_observation"]
        row["vision_backbone_median_ms"] = manifest["feature_extraction"].get("median_vision_backbone_ms")

    pred_npz_path = OUT / "data" / "evaluation_scores.npz"
    pred_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_npz_path,
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        **{f"score_{method}": np.stack([scores[method][row["sample_id"]] for row in rows]).astype(np.float16) for method in METHODS},
    )
    visual_meta = _save_visuals(manifest, rows, labels, scores, threshold_values, predictions)
    mapping_spec = manifest["feature_spec"]
    with (OUT / "feature_mapping_report.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## Runtime visual alignment figure\n\nSee [`figures/feature_mapping_overlay.png`](figures/feature_mapping_overlay.png): it displays the exact processor input, oracle mask and 16×16 patch grid used for the feature/token alignment audit.\n")

    summary = {
        "status": "complete",
        "experiment": "Track B · OpenVLA native-feature offline foreground/background segmentation",
        "code_commit": manifest.get("code_commit"),
        "dataset_sample_count": len(rows),
        "split_counts": manifest.get("split_counts"),
        "trajectory_counts": {split: len({r["trajectory_id"] for r in rows if r["split"] == split}) for split in ("training", "calibration", "evaluation")},
        "evaluation_trajectory_ids": sorted({r["trajectory_id"] for r in evaluation_rows}),
        "methods": model_reports,
        "thresholds_selected_on_calibration_only": {
            method: {"threshold": threshold_values[method], **threshold_info[method]} for method in METHODS
        },
        "evaluation_metrics": eval_micro,
        "semantic_segmentation_metrics": [
            {key: row[key] for key in ("method", "variant", "foreground_recall", "foreground_precision",
                                       "foreground_iou", "false_negative_foreground_rate",
                                       "false_positive_background_rate", "tp", "fp", "fn", "tn")}
            for row in eval_micro
        ],
        "kv_reuse_suitability_metrics": [
            {key: row[key] for key in ("method", "variant", "unsafe_fg_reuse", "reuse_opportunity",
                                       "predicted_background_fraction", "foreground_recall")}
            for row in eval_micro
        ],
        "temporal_consistency": [r for r in temporal_rows if r["trajectory_id"] == "ALL_EVALUATION"],
        "classifier_overhead_ms": classifier_overhead,
        "vision_backbone_ms": manifest["feature_extraction"].get("median_vision_backbone_ms"),
        "vision_backbone_p95_ms": manifest["feature_extraction"].get("p95_vision_backbone_ms"),
        "feature_spec": mapping_spec,
        "visual_gallery": visual_meta,
        "reuse_metrics_definition": {
            "unsafe_fg_reuse": "oracle foreground tokens predicted as background / total oracle foreground tokens",
            "reuse_opportunity": "predicted background tokens / 256",
            "dilation": "binary 8-neighbor 3x3 expansion on the 16x16 grid; applied to predicted foreground",
            "source_union": "current predicted foreground unioned with the prediction for its R5 source frame",
        },
        "actual_kv_reuse_performance_measured": False,
        "closed_loop_action_generation_run": False,
        "finalized_at_unix": time.time(),
    }
    atomic_json(OUT / "summary.json", summary)
    (OUT / "REPORT.md").write_text(_make_report(summary), encoding="utf-8")
    manifest["evaluation"] = {
        "status": "complete",
        "methods": list(METHODS),
        "classifier_overhead_ms": classifier_overhead,
        "thresholds_selected_on_calibration_only": {method: threshold_values[method] for method in METHODS},
        "evaluation_scores_path": str(pred_npz_path.relative_to(OUT)),
        "evaluation_scores_sha256": sha256(pred_npz_path),
        "figures": visual_meta["files"],
    }
    atomic_json(OUT / "dataset_manifest.json", manifest)
    return summary


def _make_report(summary: dict[str, Any]) -> str:
    metric_rows = summary["evaluation_metrics"]
    lines = [
        "# Track B — OpenVLA native-feature foreground/background segmentation",
        "",
        "## Run status",
        "",
        "Complete. This is an offline segmentation benchmark over replayed LIBERO observations. It does not implement selective KV skipping and does not measure closed-loop task success, KV-reuse latency, inference speedup, or action preservation.",
        "",
        "## Data and split",
        "",
        f"- {summary['dataset_sample_count']} observations across tasks 4, 5, and 7; sampled as seven five-frame R5 windows from each of nine Stage 3A Fresh trajectories.",
        f"- Complete trajectories are separated by initial state: training={summary['split_counts']['training']}, calibration={summary['split_counts']['calibration']}, evaluation={summary['split_counts']['evaluation']} samples. Each split has three trajectories, one per task.",
        "- Evaluation oracle labels were not used for classifier fitting, PCA polarity, or threshold selection. Classifier PCA dimensionality reduction is fitted on training features only; native DINOv2 PCA bases, polarity, and thresholds are fitted/calibrated on calibration trajectories.",
        "- Stage 3A action records reconstruct observations and segmentation maps. No OpenVLA action generation was run during Track B extraction.",
        "",
        "## 1. Semantic segmentation accuracy",
        "",
        "The table reports evaluation micro averages across 256 visual tokens per observation. Recall, precision, IoU, false-negative rate, and false-positive background rate describe semantic foreground/background agreement. The source union and dilation rows also show how candidate masks change at token resolution.",
        "",
        "| Method | Variant | FG recall | FG precision | FG IoU | Unsafe FG reuse | FP background rate | Reuse opportunity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {METHOD_TITLES[row['method']]} | {row['variant']} | {row['foreground_recall']:.3f} | {row['foreground_precision']:.3f} | {row['foreground_iou']:.3f} | {row['unsafe_fg_reuse']:.3f} | {row['false_positive_background_rate']:.3f} | {row['reuse_opportunity']:.3f} |"
        )
    lines += [
        "",
        "Thresholds were selected using calibration labels by maximizing reuse opportunity under a 5% calibration unsafe-foreground-reuse target. If no threshold met the target, the minimum-unsafe threshold was selected and that deviation is recorded in `summary.json`. The full held-out threshold sweep is in `threshold_sweep.csv`; evaluation curves are descriptive and were not used to choose thresholds.",
        "",
        "## 2. Potential background-token KV-reuse suitability",
        "",
        "Unsafe foreground reuse is oracle foreground predicted background divided by total oracle foreground tokens. Reuse opportunity is the predicted-background fraction of all 256 visual tokens. The dilation rows measure a conservative one-patch foreground expansion; the source-union rows protect locations predicted foreground in either the R5 source or current frame. These are mask-level opportunity and risk proxies, not measured compute savings.",
        "",
        "## Temporal reuse safety",
        "",
        "`temporal_consistency.csv` reports adjacent-frame token-mask agreement within the sampled five-frame R5 windows. `per_sample_metrics.csv` reports both the current-frame mask and the source/current foreground union. The union protects a token marked foreground in either frame; its changes in unsafe reuse and reuse opportunity can be read from `metrics.csv`.",
        "",
        "## Computational overhead",
        "",
        f"- Median OpenVLA DINOv2+SigLIP feature forward: {summary['vision_backbone_ms'] if summary['vision_backbone_ms'] is not None else 'unavailable'} ms per observation; p95 {summary['vision_backbone_p95_ms'] if summary['vision_backbone_p95_ms'] is not None else 'unavailable'} ms.",
        "- `summary.json` reports each segmentation method's median and p95 classifier overhead separately, measured on evaluation samples and excluding the vision-backbone forward.",
        "- These are component measurements, not end-to-end latency or speedup claims.",
        "",
        "## 3. Actual KV-reuse performance",
        "",
        "Not measured in Track B. There was no selective KV-cache skipping, decoder execution, action generation, latency benchmark, or task-success evaluation. Segmentation quality alone does not establish speedup or downstream action preservation.",
        "",
        "## Visual results",
        "",
        "See `figures/segmentation_overlays.png`, `figures/error_maps.png`, `figures/dilation_comparison.png`, the three `figures/temporal_*.png` sequences, `figures/reuse_opportunity_vs_unsafe_fg_reuse.png`, and the combined `figures/comparison_report.pdf`. The gallery includes task/state medians plus the strongest and weakest held-out samples by average method IoU; examples are labeled with their split and should not be read as a substitute for the aggregate metrics.",
        (f"- Renderer: {summary['visual_gallery'].get('renderer')}; fallback reason: `{summary['visual_gallery'].get('matplotlib_import_error')}`." if summary["visual_gallery"].get("renderer") == "Pillow fallback" else "- Figures were rendered with Matplotlib."),
        "",
        "## Method notes and limitations",
        "",
        "1. DINOv2 PCA per frame is unsupervised in its per-frame component fit, but label-based calibration supplies a global polarity and threshold. Per-frame components are sign-aligned to the calibration PCA direction.",
        "2. Calibration-basis DINOv2 PCA fits its basis to calibration observations and holds it fixed for all other frames. Its foreground polarity and threshold use calibration labels.",
        "3. LDA methods train only on the training trajectories. To condition a high-dimensional covariance, each branch uses per-token L2 normalization and a training-only randomized PCA to 128 dimensions, followed by binary pooled-covariance LDA with 10% shrinkage toward a diagonal target and a 1e-6 relative ridge. This is a documented regularized linear discriminant implementation; the same reduction and regularization are used for all three comparisons, and explained variance/ridge values are recorded in `summary.json`.",
        "4. Oracle semantics use the Stage 3A rendered-label utility: explicit background name hints are background; robot hints are foreground; all other names are foreground by default and are listed as `foreground_default_unmapped`. Nonzero segmentation IDs not represented by the name map are also recorded in `dataset_manifest.json`.",
        "5. Oracle pixels are mapped to the 16×16 token grid by the existing Stage 3A nearest-neighbor resize/crop and any-foreground-pixel reduction. The feature mapping report and overlay document the spatial assumptions.",
        "6. The benchmark samples 315 observations from nine pre-existing trajectories; the held-out evaluation split contains three complete state-2 trajectories, one per task. Generalization to other tasks, camera views, object taxonomies, or policies is not established.",
        "7. Segmentation results indicate mask quality and theoretical background-token opportunity only. Actual KV-reuse performance, projection savings, latency, downstream action relevance, and task-success preservation were not measured.",
        "",
        "## Reproducibility artifacts",
        "",
        "- `configuration.json`: experiment parameters and code revision.",
        "- `dataset_manifest.json`: action-trace source hashes, trajectory split manifests, oracle mapping audit, sample paths/checksums, encoder mapping, and feature extraction status.",
        "- `data/`: local-only RGB, instance masks, semantic masks, token masks, native feature tensors, and evaluation scores. Large arrays are not committed.",
        "- `logs/`: supervisor and execution logs; raw feature checksums are in `dataset_manifest.json`.",
        "",
    ]
    return "\n".join(lines)
