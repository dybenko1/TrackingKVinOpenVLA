#!/usr/bin/env python3
"""Corrected 1,000-image Phase 2 pilot: caption-round batches + multi-positive eval."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.caption_round_sampler import CaptionRoundBatchSampler
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.dataset import CocoAnnotationDataset, repeated_image_id_count
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.losses import symmetric_infonce
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_projection(checkpoint: Path, device: torch.device) -> tuple[RegionTextProjection, str]:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    model = RegionTextProjection(text_dim=1152, hidden_dim=1024, region_dim=1024)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise AssertionError(f"Projection checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval(), state_hash(state)


def multi_positive_metrics(model: RegionTextProjection, image_regions: np.ndarray,
                           image_ids: np.ndarray, text_features: np.ndarray,
                           caption_ids: np.ndarray, device: torch.device,
                           temperature: float = 0.07) -> dict:
    with torch.inference_mode():
        regions = torch.from_numpy(np.array(image_regions, dtype=np.float32, copy=True)).to(device)
        texts = torch.from_numpy(np.array(text_features, dtype=np.float32, copy=True)).to(device)
        scores = model(regions, texts).float()
        if scores.shape != (len(image_ids), len(caption_ids)) or not torch.isfinite(scores).all():
            raise AssertionError(f"Invalid retrieval score matrix: {tuple(scores.shape)}")
        image_id_tensor = torch.as_tensor(np.array(image_ids, dtype=np.int64, copy=True), device=device)
        caption_image_ids = torch.as_tensor(np.array(caption_ids[:, 1], dtype=np.int64, copy=True), device=device)
        positive = image_id_tensor[:, None] == caption_image_ids[None, :]
        if not positive.any(dim=1).all() or not positive.any(dim=0).all():
            raise AssertionError("Every validation image and caption must have at least one positive pair")

        # Stable descending rank; every caption for the image is a valid positive.
        i2t_order = torch.argsort(scores, dim=1, descending=True, stable=True)
        i2t_rank_rows = []
        for row in range(len(image_ids)):
            ranked_positive = positive[row].index_select(0, i2t_order[row])
            i2t_rank_rows.append(int(torch.nonzero(ranked_positive, as_tuple=False)[0, 0].item()) + 1)
        i2t_ranks = torch.tensor(i2t_rank_rows, dtype=torch.float64)

        t2i_scores = scores.transpose(0, 1)
        t2i_order = torch.argsort(t2i_scores, dim=1, descending=True, stable=True)
        image_row_by_id = {int(image_id): row for row, image_id in enumerate(image_ids.tolist())}
        t2i_rank_rows = []
        for caption_row, image_id in enumerate(caption_ids[:, 1].tolist()):
            target_row = image_row_by_id[int(image_id)]
            where = torch.nonzero(t2i_order[caption_row] == target_row, as_tuple=False)
            t2i_rank_rows.append(int(where[0, 0].item()) + 1)
        t2i_ranks = torch.tensor(t2i_rank_rows, dtype=torch.float64)

        logits = scores / float(temperature)
        i2t_denominator = torch.logsumexp(logits, dim=1)
        i2t_numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
        t2i_denominator = torch.logsumexp(logits.transpose(0, 1), dim=1)
        t2i_numerator = logits.transpose(0, 1)[
            torch.arange(len(caption_ids), device=device),
            torch.as_tensor([image_row_by_id[int(i)] for i in caption_ids[:, 1]], device=device),
        ]
        mp_i2t = i2t_denominator - i2t_numerator
        mp_t2i = t2i_denominator - t2i_numerator
        result = {
            "image_to_text": {
                f"recall_at_{k}": float((i2t_ranks <= k).double().mean()) for k in (1, 5, 10)
            },
            "text_to_image": {
                f"recall_at_{k}": float((t2i_ranks <= k).double().mean()) for k in (1, 5, 10)
            },
            "image_to_text_rank": {"mean": float(i2t_ranks.mean()), "median": float(np.median(i2t_rank_rows)), "max": int(i2t_ranks.max())},
            "text_to_image_rank": {"mean": float(t2i_ranks.mean()), "median": float(np.median(t2i_rank_rows)), "max": int(t2i_ranks.max())},
            "multi_positive_infonce": {
                "image_to_text": float(mp_i2t.mean()),
                "text_to_image": float(mp_t2i.mean()),
                "symmetric_mean": float((mp_i2t.mean() + mp_t2i.mean()) / 2),
                "temperature": float(temperature),
                "formula": "mean(logsumexp(all) - logsumexp(all positives)) in each direction, averaged",
            },
            "score_matrix": {
                "shape": list(scores.shape), "min": float(scores.min()), "max": float(scores.max()),
                "mean": float(scores.mean()), "std": float(scores.std(unbiased=False)),
                "positive_pairs": int(positive.sum()), "negative_pairs": int((~positive).sum()),
            },
        }
        return result


def train_corrected(train_dataset: CocoAnnotationDataset, val_dataset: CocoAnnotationDataset,
                    initial_state: dict[str, torch.Tensor], output: Path,
                    seed: int, epochs: int, batch_size: int, lr: float,
                    device: torch.device) -> dict:
    seed_all(seed)
    model = RegionTextProjection(text_dim=1152, hidden_dim=1024, region_dim=1024).to(device)
    model.load_state_dict(initial_state, strict=True)
    if state_hash(model.state_dict()) != state_hash(initial_state):
        raise AssertionError("Corrected run did not start from the exact original pilot initial MLP state")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in model.parameters()}:
        raise AssertionError("Only the MLP parameters may be optimized")

    sampler = CaptionRoundBatchSampler([row[1] for row in train_dataset.caption_rows], batch_size=batch_size, seed=seed)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    history = []
    train_start = time.perf_counter()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        flattened = [row for batch in batches for row in batch]
        if sorted(flattened) != list(range(len(train_dataset))):
            raise AssertionError(f"Epoch {epoch} did not consume every caption row exactly once")
        duplicate_rows = sum(repeated_image_id_count([train_dataset.caption_rows[row][1] for row in batch]) for batch in batches)
        if duplicate_rows:
            raise AssertionError(f"Caption-round sampler produced {duplicate_rows} same-image repeated rows in epoch {epoch}")
        model.train()
        weighted_loss = 0.0
        examples = 0
        for batch_rows in batches:
            batch = [train_dataset[row] for row in batch_rows]
            regions = torch.stack([item[0] for item in batch]).to(device)
            texts = torch.stack([item[1] for item in batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = symmetric_infonce(model(regions, texts), temperature=0.07, legacy_extra_batch_square=False)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(batch_rows)
            examples += len(batch_rows)

        model.eval()
        val_weighted = 0.0
        val_examples = 0
        val_diagonal_duplicate_rows = 0
        with torch.inference_mode():
            for regions, texts, image_ids, _ in val_loader:
                val_diagonal_duplicate_rows += repeated_image_id_count(image_ids)
                regions, texts = regions.to(device), texts.to(device)
                val_loss = symmetric_infonce(model(regions, texts), temperature=0.07, legacy_extra_batch_square=False)
                val_weighted += float(val_loss) * len(image_ids)
                val_examples += len(image_ids)
        record = {
            "epoch": epoch + 1, "train_mean_diagonal_infonce": weighted_loss / examples,
            "validation_historical_diagonal_infonce": val_weighted / val_examples,
            "train_caption_rows_used": examples, "train_caption_rows_total": len(train_dataset),
            "train_batches": len(batches), "train_partial_batches": sum(len(batch) < batch_size for batch in batches),
            "same_image_repeated_rows_in_train_batches": duplicate_rows,
            "validation_rows_seen_by_historical_diagonal_loss": val_examples,
            "validation_duplicate_rows_counter_if_diagonal_batched": val_diagonal_duplicate_rows,
            "elapsed_seconds": time.perf_counter() - train_start,
        }
        history.append(record)
        atomic_json(output / "training_history.json", {"status": "training", "completed_epochs": len(history), "history": history})
        print(json.dumps(record), flush=True)

    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint_path = output / "corrected_projection.pt"
    tmp = checkpoint_path.with_suffix(".pt.tmp")
    torch.save({"state_dict": state, "initial_state_dict": initial_state,
                "history": history, "seed": seed, "epochs": epochs,
                "sampler": "caption_round_no_same_image_per_batch_v1",
                "temperature": 0.07, "optimizer": "Adam", "lr": lr}, tmp)
    os.replace(tmp, checkpoint_path)
    atomic_json(output / "training_history.json", {
        "status": "complete", "completed_epochs": len(history), "history": history,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "initial_state_sha256": state_hash(initial_state), "final_state_sha256": state_hash(state),
    })
    return {"status": "complete", "epochs": epochs, "seconds": time.perf_counter() - train_start,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
            "initial_state_sha256": state_hash(initial_state), "final_state_sha256": state_hash(state),
            "history": {"history": history}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to train without --execute")
    existing_allowed = {"data_audit.json", "DATA_AUDIT.md", "logs"}
    if args.output.exists() and any(path.name not in existing_allowed for path in args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite phase-2 outputs in {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; corrected training was not started")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"experiment": "Talk2DINO-SigLIP Phase 2 corrected 1000-image pilot",
              "seed": args.seed, "batch_size": args.batch_size, "epochs": args.epochs,
              "optimizer": "Adam", "learning_rate": args.lr, "training_objective": "canonical symmetric diagonal InfoNCE",
              "temperature": 0.07, "sampler": "caption-round, at most one caption per image per batch",
              "validation": "global multi-positive retrieval and multi-positive InfoNCE diagnostic",
              "train_feature_rows": 4002, "validation_feature_rows": 1000,
              "reuse_cached_features": True, "reextract_features": False}
    atomic_json(args.output / "phase2_config.json", config)
    atomic_json(args.output / "status.json", {"status": "running", "stage": "audit_and_tests", "exit_code": None})
    pilot = args.pilot_output
    features = pilot / "features"
    train_dataset = CocoAnnotationDataset(
        features / "image_ids_train2014.npy", features / "region_features_train2014.npy",
        features / "caption_ids_train2014.npy", features / "text_features_train2014.npy")
    val_dataset = CocoAnnotationDataset(
        features / "image_ids_val2014.npy", features / "region_features_val2014.npy",
        features / "caption_ids_val2014.npy", features / "text_features_val2014.npy")
    if train_dataset.dropped_missing_images or val_dataset.dropped_missing_images:
        raise AssertionError("Some caption annotations lack cached image features")
    if len(train_dataset) != 4002 or len(val_dataset) != 1000:
        raise AssertionError(f"Expected 4002/1000 caption rows, got {len(train_dataset)}/{len(val_dataset)}")

    audit_command = [sys.executable, str(Path(__file__).with_name("audit_phase2.py")),
                     "--pilot-output", str(pilot), "--dataset-root", str(args.dataset_root),
                     "--output-dir", str(args.output)]
    subprocess.run(audit_command, check=True)
    sampler_test_command = [sys.executable, "-m", "unittest",
                            "experiments.kv_cache.experiments.trackb_talk2dino_siglip.tests.test_caption_round_sampler", "-v"]
    test = subprocess.run(sampler_test_command, cwd=REPO, capture_output=True, text=True)
    (args.output / "sampler_tests.log").write_text(test.stdout + test.stderr)
    if test.returncode:
        raise RuntimeError(f"Sampler tests failed; see {args.output / 'sampler_tests.log'}")
    atomic_json(args.output / "sampler_test_status.json", {"status": "passed", "command": sampler_test_command,
                "test_count": 5, "returncode": test.returncode})

    base_output = pilot / "pilot_mlp_checkpoint"
    initial_raw = torch.load(base_output / "initial_projection.pt", map_location="cpu", weights_only=True)
    initial_state = initial_raw.get("state_dict", initial_raw)
    final_path = base_output / "final_projection.pt"
    baseline_model, baseline_sha = load_projection(final_path, torch.device("cuda"))
    image_ids = np.load(features / "image_ids_val2014.npy", mmap_mode="r")
    regions = np.load(features / "region_features_val2014.npy", mmap_mode="r")
    caption_ids = np.load(features / "caption_ids_val2014.npy", mmap_mode="r")
    texts = np.load(features / "text_features_val2014.npy", mmap_mode="r")
    baseline_metrics = multi_positive_metrics(baseline_model, regions, image_ids, texts, caption_ids, torch.device("cuda"))
    del baseline_model
    torch.cuda.empty_cache()
    atomic_json(args.output / "baseline_multi_positive_validation.json", {
        "checkpoint": str(final_path), "checkpoint_sha256": sha256_file(final_path),
        "state_sha256": baseline_sha, "historical_diagonal_validation_loss_epoch10": 2.6272903851100375,
        "metrics": baseline_metrics,
    })

    training = train_corrected(train_dataset, val_dataset, initial_state, args.output,
                               args.seed, args.epochs, args.batch_size, args.lr, torch.device("cuda"))
    corrected_model, corrected_sha = load_projection(Path(training["checkpoint"]), torch.device("cuda"))
    corrected_metrics = multi_positive_metrics(corrected_model, regions, image_ids, texts, caption_ids, torch.device("cuda"))
    atomic_json(args.output / "status.json", {"status": "running", "stage": "corrected_training", "exit_code": None})
    atomic_json(args.output / "corrected_multi_positive_validation.json", {
        "checkpoint": training["checkpoint"], "checkpoint_sha256": training["checkpoint_sha256"],
        "state_sha256": corrected_sha, "metrics": corrected_metrics,
    })

    # Counterfactual visual segmentation feasibility check: sampled COCO subset has captions only.
    audit = json.loads((args.output / "data_audit.json").read_text())
    hist = training["history"]["history"]
    first, last = hist[0], hist[-1]
    report = f"""# Phase 2 Validation Report — Talk2DINO-SigLIP Corrected Pilot

## 1. Executive summary

The 1,000-image pilot was retrained for {args.epochs} epochs with an epoch-based caption-round batch sampler. Every one of the {len(train_dataset)} training captions is used once per epoch, and no minibatch contains two captions from one image. This fixes the specific same-image diagonal-negative collision in training. The original pilot outputs/checkpoints remain unchanged.

Validation now uses the full global image × caption similarity matrix and treats all captions belonging to an image as positives for image-to-text retrieval. Baseline and corrected checkpoints were evaluated on the identical {len(image_ids)} validation images and {len(caption_ids)} captions using the existing max-over-16-region-head scoring function.

This is a small pilot and does not establish robust spatial segmentation or generalization.

## 2. Original Talk2DINO methodology and deviations

Upstream `COCOCaptions` creates one dataset row per caption annotation and does not group multiple captions from an image. `do_train` shuffles annotation rows; `ContrastiveLoss` uses diagonal image-caption labels. Upstream InfoNCE divides the already mean-reduced symmetric cross entropy by `batch_size**2`. Our prior implementation report records our adaptation's canonical scaling (without that extra division). The original pilot also used `drop_last=True`, omitting 34/4,002 training captions and 104/1,000 validation captions per epoch.

The corrected pilot preserves annotation-level examples, the diagonal symmetric InfoNCE loss, model, features, optimizer, LR, batch cap, seed and epoch count. It changes training batch construction to caption rounds and keeps partial batches. Validation changes to multi-positive retrieval because diagonal validation on repeated-image captions is not a valid retrieval metric. This is a deliberate correction, not attributed to upstream Talk2DINO.

## 3. Dataset diversity and mapping audit

See `DATA_AUDIT.md` and `data_audit.json`. Train: {audit['splits']['train']['unique_images']} unique images / {audit['splits']['train']['caption_annotations']} captions; validation: {audit['splits']['validation']['unique_images']} / {audit['splits']['validation']['caption_annotations']}. Caption counts per image are train min/median/mean/max {audit['splits']['train']['captions_per_image']['min']}/{audit['splits']['train']['captions_per_image']['median']}/{audit['splits']['train']['captions_per_image']['mean']:.2f}/{audit['splits']['train']['captions_per_image']['max']} and validation {audit['splits']['validation']['captions_per_image']['min']}/{audit['splits']['validation']['captions_per_image']['median']}/{audit['splits']['validation']['captions_per_image']['mean']:.2f}/{audit['splits']['validation']['captions_per_image']['max']}. Exact caption duplicates: train excess {audit['splits']['train']['exact_duplicate_caption_strings']['excess_annotation_rows']}; validation excess {audit['splits']['validation']['exact_duplicate_caption_strings']['excess_annotation_rows']}. Feature ID/order joins passed; train/validation image IDs are disjoint.

This random sample includes diverse captioned scenes, but only 800 training images constrain the MLP; the full COCO corpus is not represented. Local COCO instances annotation files are absent, so a mask-based spatial evaluation was not run. Obtaining the official instance annotation archives and mapping prompts to categories is the minimum added data step; no download was started.

## 4. Original pilot duplicate-counter diagnosis

`repeated_image_id_count` counts surplus rows after the first occurrence of each image within a minibatch (`sum(count - 1)`). It is not a count of all pairwise false-negative relations. The pilot's reported 251 per training epoch therefore means about 251 extra rows in its 31 full random batches share an image ID with another row in that same batch. It does not prove 251 distinct false negatives; a batch with three captions from one image contributes two rows but has multiple incorrect off-diagonal pairings. Validation's 701 counter was for only the first 896 sequential validation rows because `drop_last=True`; 104 captions were excluded.

## 5. Corrected sampler and tests

Per epoch, the sampler groups all annotation-row indices by image ID, independently shuffles each image's captions, selects at most one remaining caption from each image per round, shuffles each round, then chunks that round into batches of at most 128. Partial batches are retained. Tests verify once-only coverage, no same-image batch collisions, unequal group counts, partial batches, epoch rotation and deterministic replay. Results are in `sampler_test_status.json` and `sampler_tests.log`.

Across all corrected training epochs: every caption row was consumed exactly once; same-image repeated rows per batch were zero. Each epoch had {first['train_batches']} batches ({first['train_partial_batches']} partial batches). Train diagonal InfoNCE changed {first['train_mean_diagonal_infonce']:.4f} → {last['train_mean_diagonal_infonce']:.4f}. These losses are optimization diagnostics, not the primary evaluation.

## 6. Baseline versus corrected multi-positive validation

Both models use the same global scorer: normalize visual regions and projected text, compute cosine similarity for each of 16 regions, and take the maximum region score. The baseline checkpoint was the original pilot's final MLP; the corrected MLP started from the exact saved original initial MLP weights (hashes are in each metrics JSON).

| Checkpoint | I→T R@1 | I→T R@5 | I→T R@10 | T→I R@1 | T→I R@5 | T→I R@10 | Multi-positive InfoNCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original pilot | {baseline_metrics['image_to_text']['recall_at_1']:.3f} | {baseline_metrics['image_to_text']['recall_at_5']:.3f} | {baseline_metrics['image_to_text']['recall_at_10']:.3f} | {baseline_metrics['text_to_image']['recall_at_1']:.3f} | {baseline_metrics['text_to_image']['recall_at_5']:.3f} | {baseline_metrics['text_to_image']['recall_at_10']:.3f} | {baseline_metrics['multi_positive_infonce']['symmetric_mean']:.4f} |
| Corrected pilot | {corrected_metrics['image_to_text']['recall_at_1']:.3f} | {corrected_metrics['image_to_text']['recall_at_5']:.3f} | {corrected_metrics['image_to_text']['recall_at_10']:.3f} | {corrected_metrics['text_to_image']['recall_at_1']:.3f} | {corrected_metrics['text_to_image']['recall_at_5']:.3f} | {corrected_metrics['text_to_image']['recall_at_10']:.3f} | {corrected_metrics['multi_positive_infonce']['symmetric_mean']:.4f} |

Mean/median retrieval ranks and complete score-matrix stats are in `baseline_multi_positive_validation.json` and `corrected_multi_positive_validation.json`. The old diagonal validation loss (2.6273) is historical and not directly comparable to the new multi-positive metric.

## 7. Multi-positive metric definition

For image-to-text, the positive numerator is `logsumexp` over every caption whose image ID matches the query image. For text-to-image, its corresponding unique image is the positive. In each direction the diagnostic loss is `logsumexp(all candidates) - logsumexp(positives)`; the reported symmetric value averages directional means at temperature 0.07. Retrieval ranks count any corresponding caption as positive for I→T; T→I ranks the matching image.

## 8. Spatial evaluation and limitations

The current COCO dataset folder has `captions_train2014.json` and `captions_val2014.json`, but not `instances_train2014.json` or `instances_val2014.json`. Existing extracted features do contain 256 patch vectors and attention maps, but without object masks they cannot support defensible IoU, precision/recall or pointing-accuracy evaluation. The pilot's qualitative attention panels are not segmentation evidence.

## 9. Multi-positive InfoNCE follow-up (not run)

For each image query, a multi-positive objective replaces the single positive log-probability with `-log(sum_{{j∈P(i)}} exp(logit_ij) / sum_j exp(logit_ij))`; the symmetric text-to-image direction uses the matching image (or multiple image rows if an image table is duplicated). This can stop captions of the same image being negatives and may better fit the data. It also changes the original objective, can overweight images with many captions unless normalized carefully, and requires a fair separate run from the same initialization, sampler, seed, epochs and data. This training was not run; it remains an explicit next objective comparison, not an observed result.

## 10. Conclusions and next step

1. Same-image false negatives within training batches: **eliminated by construction**, tested, and zero in every corrected epoch.
2. Caption usage: **all 4,002 training annotations were consumed once each epoch**, including partial batches.
3. Validation: the multi-positive retrieval objective is appropriate for multiple captions per image and avoids treating sibling captions as negatives.
4. Corrected versus original: compare the paired metrics table; the historical diagonal validation loss is not a fair comparison.
5. Spatial alignment: this pilot does not establish it. Retrieval metrics and attention heatmaps do not prove segmentation quality.
6. Scale-up: a 1,000-image pilot is not sufficient evidence for the full experiment. First obtain COCO instance masks and run a small, fixed spatial benchmark on a held-out multi-object subset, then decide whether to expand.
7. Smallest uncertainty-reducing next experiment: evaluate the frozen corrected MLP on ~100 held-out images with available COCO instance masks, predeclared object phrases and IoU/pointing metrics, after acquiring the missing official annotation archive. Keep policy inference unchanged.

## 11. Reproducibility

Configuration: `phase2_config.json`; sample seed {args.seed}; model/features/checkpoint hashes in `run_metadata.json`; the console log is `logs/phase2.log`. Command is recorded in `run_metadata.json`. No COCO feature re-extraction, additional checkpoint download, full dataset training or LIBERO rollout was done.
"""
    (args.output / "PHASE2_VALIDATION_REPORT.md").write_text(report)
    atomic_json(args.output / "summary.json", {
        "status": "complete", "baseline_metrics": baseline_metrics,
        "corrected_metrics": corrected_metrics, "training_epochs": args.epochs,
        "train_captions_per_epoch": len(train_dataset), "same_image_batch_collisions_per_epoch": 0,
        "spatial_instance_annotations_available": audit["spatial_annotation_availability"],
    })
    atomic_json(args.output / "run_metadata.json", {
        "experiment": "Talk2DINO-SigLIP Phase 2 corrected 1000-image pilot",
        "command": sys.argv, "seed": args.seed, "batch_size": args.batch_size,
        "epochs": args.epochs, "learning_rate": args.lr, "temperature": 0.07,
        "train_caption_rows": len(train_dataset), "validation_caption_rows": len(val_dataset),
        "original_initial_checkpoint_sha256": sha256_file(base_output / "initial_projection.pt"),
        "original_final_checkpoint_sha256": sha256_file(final_path),
        "corrected_checkpoint_sha256": training["checkpoint_sha256"],
        "training_initial_state_sha256": training["initial_state_sha256"],
        "corrected_final_state_sha256": training["final_state_sha256"],
        "torch_version": torch.__version__, "device": torch.cuda.get_device_name(0),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "feature_arrays_reused": True, "feature_reextraction": False,
    })
    atomic_json(args.output / "status.json", {"status": "complete", "exit_code": 0, "epochs": args.epochs,
                "features_reused": True, "all_train_captions_each_epoch": True,
                "same_image_collisions": 0, "completed_at_unix": time.time()})
    print(json.dumps({"status": "complete", "baseline": baseline_metrics,
                      "corrected": corrected_metrics, "outputs": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
