"""Prompt robustness audit for frozen Talk2DINO-SigLIP V2.

Uses the fixed COCO val2014 development/final manifests and cached V2
penultimate / V1 final DINO features. It never updates either projection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.text_encoder import (
    encode_text_embeddings,
    load_siglip_text_tower,
)
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.projection import RegionTextProjection
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.phase3_evaluate import (
    TEMPLATES as V1_TEMPLATES,
)
from experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.evaluate import (
    auc_binary,
    category_targets,
    confusion,
    pointing_accuracy,
)

METHODS = ("A_training_prompt", "B_natural_description", "C_picture_phrase",
           "D_noun_phrase", "E_embedding_ensemble", "F_score_ensemble",
           "V1_seven_template")
THRESHOLDS = np.round(np.arange(.05, .951, .05), 2).tolist()
PROMPT_FUNCS = (
    lambda n: f"a photo of a {n}",
    lambda n: f"the {n}",
    lambda n: f"a picture of the {n}",
    lambda n: n,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def load_frozen_projection(path: Path, device):
    """Read V1's full epoch checkpoint or a projection-only state safely."""
    # V1 epoch checkpoints contain NumPy RNG and PyTorch version metadata in
    # addition to tensors. Keep the restricted loader and allow only those
    # inert metadata types; never enable arbitrary pickle execution.
    safe = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, torch.torch_version.TorchVersion]
    if hasattr(np, "dtypes") and hasattr(np.dtypes, "UInt32DType"):
        safe.append(np.dtypes.UInt32DType)
    with torch.serialization.safe_globals(safe):
        raw = torch.load(path, map_location="cpu", weights_only=True)
    state = raw.get("model", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    model = RegionTextProjection()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), sha256(path)


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def encode_category_text(text_model, tokenizer, names, device):
    four = [[fn(name) for name in names] for fn in PROMPT_FUNCS]
    flat = [p for row in four for p in row]
    tokens = tokenizer(flat).to(device)
    v1_phrases = [template.format(name) for name in names for template in V1_TEMPLATES]
    v1_tokens = tokenizer(v1_phrases).to(device)
    with torch.inference_mode():
        feats = torch.cat([encode_text_embeddings(text_model, tokens[i:i+64]) for i in range(0, len(tokens), 64)])
        v1_encoded = torch.cat([encode_text_embeddings(text_model, v1_tokens[i:i+64]) for i in range(0, len(v1_tokens), 64)])
    feats = feats.reshape(4, len(names), -1).float()
    v1 = v1_encoded.reshape(len(names), len(V1_TEMPLATES), -1).mean(dim=1)
    return four, feats, v1.float()


@torch.inference_mode()
def make_probabilities(v1_projection, v2_projection, patch_v1, patch_v2, text_four, text_v1):
    p1 = F.normalize(patch_v1.float(), dim=-1)
    p2 = F.normalize(patch_v2.float(), dim=-1)
    v1_text = F.normalize(v1_projection.text_projection(text_v1), dim=-1)
    v1_cos = p1 @ v1_text.T
    v1_prob = torch.sigmoid(v1_cos.T)

    # A-D use the same training MLP and frozen text tower; only phrase differs.
    projected = v2_projection.text_projection(text_four.reshape(-1, text_four.shape[-1]))
    projected = F.normalize(projected, dim=-1).reshape(4, text_four.shape[1], -1)
    cosine = torch.einsum("pd,qcd->qcp", p2, projected)
    probs = torch.sigmoid(10.0 * (cosine - 0.20))
    # E averages the four raw SigLIP text vectors before the MLP, as specified.
    avg_text = text_four.mean(dim=0)
    e_text = F.normalize(v2_projection.text_projection(avg_text), dim=-1)
    e_cosine = p2 @ e_text.T
    e_prob = torch.sigmoid(10.0 * (e_cosine.T - 0.20))
    f_prob = probs.mean(dim=0)
    return {METHODS[i]: probs[i] for i in range(4)} | {
        "E_embedding_ensemble": e_prob,
        "F_score_ensemble": f_prob,
        "V1_seven_template": v1_prob,
    }, probs


def load_existing_records(path: Path, split_hash: str):
    done = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("split_manifest_sha256") == split_hash:
                        done[int(row["image_id"])] = row
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
    return done


def score_split(split_name, ids, *, output_dir, split_hash, dataset_root, feature_root,
                categories, instance_data, geometry, text_four, text_v1,
                v1_projection, v2_projection, device, gallery_limit=12):
    image_meta = {int(x["id"]): x for x in instance_data["images"]}
    anns = defaultdict(list)
    for a in instance_data["annotations"]:
        anns[int(a["image_id"])].append(a)
    cat_index = {int(c["id"]): i for i, c in enumerate(categories)}
    ids_path = feature_root / "image_ids_val2014.npy"
    feature_ids = np.load(ids_path, mmap_mode="r")
    feature_rows = {int(i): r for r, i in enumerate(feature_ids)}
    penult = np.load(feature_root / "patch_features_val2014_fp16.npy", mmap_mode="r")
    final = np.load(feature_root / "patch_features_val2014_final_norm_fp16.npy", mmap_mode="r")
    rows_path = output_dir / f"{split_name}_per_image.jsonl"
    done = load_existing_records(rows_path, split_hash)
    ids = [int(i) for i in ids]
    if not set(done).issubset(set(ids)):
        raise RuntimeError(f"{split_name}: existing resume rows do not match the immutable split")
    mode = "a" if done else "w"
    import matplotlib.pyplot as plt

    start = time.time()
    for idx, image_id in enumerate(ids, 1):
        if image_id in done:
            continue
        if image_id not in feature_rows or image_id not in image_meta:
            raise RuntimeError(f"{split_name}: missing image/feature row for {image_id}")
        targets = category_targets(image_meta[image_id], anns[image_id], cat_index, geometry)
        present = [cid for cid in targets if cid in cat_index]
        rec = {"image_id": image_id, "feature_row": int(feature_rows[image_id]),
               "split_manifest_sha256": split_hash, "categories": {}}
        if present:
            idxs = [cat_index[c] for c in present]
            tf = text_four[:, idxs].to(device)
            tv1 = text_v1[idxs].to(device)
            p2 = torch.from_numpy(np.array(penult[feature_rows[image_id]], dtype=np.float32, copy=True)).to(device)
            p1 = torch.from_numpy(np.array(final[feature_rows[image_id]], dtype=np.float32, copy=True)).to(device)
            probs, prompt_probs = make_probabilities(v1_projection, v2_projection, p1, p2, tf, tv1)
            for j, cid in enumerate(present):
                gt224, gt16, valid224, valid16 = targets[cid]
                gt = np.asarray(gt16, dtype=np.float32).reshape(-1)
                valid = np.asarray(valid16, dtype=np.float32).reshape(-1)
                area = float((gt * valid).sum())
                pmap = {name: val[j].detach().float().cpu().numpy().reshape(-1) for name, val in probs.items()}
                category_row = {"category_name": categories[cat_index[cid]]["name"],
                                "occupied_patch_equivalents": area, "methods": {}}
                for method, values in pmap.items():
                    sweep = {}
                    small_sweep = {}
                    # Keep the published V1 threshold as a fixed reference in
                    # the summary, while retaining a dev-only sweep for its
                    # separately named recalibrated diagnostic.
                    method_thresholds = THRESHOLDS
                    for threshold in method_thresholds:
                        tp, fp, fn, tn = confusion(values, gt, valid, threshold)
                        sweep[f"{threshold:.2f}"] = [tp, fp, fn]
                        small_sweep[f"{threshold:.2f}"] = [tp, fp, fn] if 0.0 < area < 4.0 else None
                    binary_threshold = .55 if method == "V1_seven_template" else .50
                    tp, fp, fn, _ = confusion(values, gt, valid, binary_threshold)
                    category_row["methods"][method] = {
                        "threshold_sweep_tp_fp_fn": sweep,
                        "small_object_threshold_sweep_tp_fp_fn": small_sweep,
                        "fixed_threshold": binary_threshold,
                        "fixed_iou": tp / max(1e-12, tp + fp + fn),
                        "fixed_precision": tp / max(1e-12, tp + fp),
                        "fixed_recall": tp / max(1e-12, tp + fn),
                        "fixed_false_negative_rate": fn / max(1e-12, tp + fn),
                        "fixed_pointing": pointing_accuracy(values, gt, valid),
                        "fixed_auroc": auc_binary(values, gt, valid),
                        "small_object": bool(0.0 < area < 4.0),
                        "small_fixed_tp_fp_fn": [tp, fp, fn] if 0.0 < area < 4.0 else None,
                        "fixed_empty_mask": int(np.count_nonzero(values >= binary_threshold) == 0),
                        "probability_mean": float(values.mean()),
                        "probability_var": float(values.var()),
                    }
                prompt_stack = np.stack([pmap[METHODS[i]] for i in range(4)], axis=0)
                category_row["prompt_rewording_sensitivity"] = {
                    "mean_pixel_probability_variance": float(prompt_stack.var(axis=0).mean()),
                    "mean_pixel_probability_std": float(prompt_stack.std(axis=0).mean()),
                    "foreground_patch_counts_at_0_20": [int((x >= .20).sum()) for x in prompt_stack],
                }
                rec["categories"][str(cid)] = category_row

            if idx <= gallery_limit:
                from PIL import Image
                img = Image.open(dataset_root / "val2014" / image_meta[image_id]["file_name"]).convert("RGB")
                rgb = np.asarray(img.resize((224, 224), Image.Resampling.BICUBIC))
                fig, axes = plt.subplots(len(present), 9, figsize=(25, 3.0 * len(present)), squeeze=False)
                for j, cid in enumerate(present):
                    gt224 = targets[cid][0]
                    axes[j, 0].imshow(rgb); axes[j, 0].set_title(f"{categories[cat_index[cid]]['name']} · GT")
                    axes[j, 1].imshow(gt224, cmap="gray"); axes[j, 1].set_title("GT mask")
                    for k, method in enumerate(METHODS):
                        score = probs[method][j].detach().float().cpu().numpy().reshape(16, 16)
                        axes[j, k + 2].imshow(score, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
                        axes[j, k + 2].set_title(method.replace("_", "\n"), fontsize=7)
                for ax in axes.flat: ax.axis("off")
                fig.suptitle(f"Prompt robustness · {split_name} · image {image_id} · fixed scores")
                fig.tight_layout()
                fig.savefig(output_dir / "qualitative" / f"{split_name}_image_{image_id}.png", dpi=110)
                plt.close(fig)
        with rows_path.open(mode) as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush(); os.fsync(f.fileno())
        mode = "a"
        if idx % 100 == 0 or idx == len(ids):
            print(f"{split_name}: {idx}/{len(ids)} images", flush=True)

    records = load_existing_records(rows_path, split_hash)
    if set(records) != set(ids):
        raise RuntimeError(f"{split_name}: expected {len(ids)} complete image rows, have {len(records)}")
    atomic_json(output_dir / f"{split_name}_completed.json", {
        "status": "complete", "image_count": len(ids), "unique_image_count": len(records),
        "split_manifest_sha256": split_hash, "elapsed_seconds": round(time.time() - start, 3),
    })
    return list(records.values())


def aggregate(records, threshold_by_method):
    counts = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
    pair_ious = defaultdict(lambda: defaultdict(list))
    points = defaultdict(lambda: defaultdict(list))
    empty = defaultdict(lambda: defaultdict(list))
    small_counts = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
    aurocs = defaultdict(lambda: defaultdict(list))
    prompt_variance = []
    prompt_std = []
    prompt_fg_counts = [[] for _ in range(4)]
    for image in records:
        for cid, item in image["categories"].items():
            area = item["occupied_patch_equivalents"]
            for method, m in item["methods"].items():
                threshold = float(threshold_by_method[method])
                if f"{threshold:.2f}" not in m["threshold_sweep_tp_fp_fn"]:
                    # Final split stores only the selected threshold's count row.
                    tpf = m.get("selected_tp_fp_fn")
                else:
                    tpf = m["threshold_sweep_tp_fp_fn"][f"{threshold:.2f}"]
                if tpf is None: continue
                counts[cid][method] = [x + y for x, y in zip(counts[cid][method], tpf)]
                tp, fp, fn = tpf
                pair_ious[cid][method].append(tp / max(1e-12, tp + fp + fn))
                points[cid][method].append(m.get("fixed_pointing"))
                if f"{threshold:.2f}" in m.get("threshold_sweep_tp_fp_fn", {}):
                    empty[cid][method].append(int(tpf[0] + tpf[1] == 0))
                elif m.get("fixed_empty_mask") is not None:
                    empty[cid][method].append(m["fixed_empty_mask"])
                small_sweep = m.get("small_object_threshold_sweep_tp_fp_fn", {})
                small_tpf = small_sweep.get(f"{threshold:.2f}")
                if small_tpf is None and m.get("small_object"):
                    # Older per-image records already contain the complete
                    # threshold sweep plus a per-pair small-object flag. Use
                    # those exact counts so small-object metrics match the
                    # selected threshold without re-running the model.
                    small_tpf = m.get("threshold_sweep_tp_fp_fn", {}).get(f"{threshold:.2f}")
                if small_tpf is not None:
                    small_counts[cid][method] = [x + y for x, y in zip(small_counts[cid][method], small_tpf)]
                elif m.get("small_fixed_tp_fp_fn") is not None:
                    # Backward-compatible fallback for old cached records; this
                    # value is explicitly tied to the legacy fixed threshold.
                    small_counts[cid][method] = [x + y for x, y in zip(small_counts[cid][method], m["small_fixed_tp_fp_fn"])]
                if m.get("fixed_auroc") is not None:
                    aurocs[cid][method].append(m["fixed_auroc"])
            sens = item.get("prompt_rewording_sensitivity")
            if sens:
                prompt_variance.append(sens["mean_pixel_probability_variance"])
                prompt_std.append(sens["mean_pixel_probability_std"])
                for i, n in enumerate(sens["foreground_patch_counts_at_0_20"]): prompt_fg_counts[i].append(n)
    result = {}
    for method, threshold in threshold_by_method.items():
        class_iou=[]; class_pair=[]; total=[0.,0.,0.]; macro_prec=[]; macro_rec=[]; macro_fn=[]; macro_small=[]; point_values=[]; empty_values=[]
        per_category={}
        for cid in counts:
            v=counts[cid][method]
            if not any(v): continue
            tp,fp,fn=v; iou=tp/max(1e-12,tp+fp+fn)
            pair=float(np.mean(pair_ious[cid][method])) if pair_ious[cid][method] else 0.
            class_iou.append(iou); class_pair.append(pair); total=[a+b for a,b in zip(total,v)]
            macro_prec.append(tp/max(1e-12,tp+fp)); macro_rec.append(tp/max(1e-12,tp+fn)); macro_fn.append(fn/max(1e-12,tp+fn))
            point_values.extend(x for x in points[cid][method] if x is not None)
            empty_values.extend(x for x in empty[cid][method] if x is not None)
            stp,sfp,sfn=small_counts[cid][method]
            per_category[cid]={"pooled_iou":iou,"mean_positive_pair_iou":pair,"precision":tp/max(1e-12,tp+fp),"recall":tp/max(1e-12,tp+fn),"pairs":len(pair_ious[cid][method]),
                "small_object_pooled_iou":stp/max(1e-12,stp+sfp+sfn) if stp+sfp+sfn else None,
                "mean_auroc":float(np.mean(aurocs[cid][method])) if aurocs[cid][method] else None}
        result[method]={"threshold":float(threshold),"pooled_macro_iou":float(np.mean(class_iou)) if class_iou else None,
            "mean_positive_pair_macro_iou":float(np.mean(class_pair)) if class_pair else None,
            "precision_macro_category":float(np.mean(macro_prec)) if macro_prec else None,
            "recall_macro_category":float(np.mean(macro_rec)) if macro_rec else None,
            "false_negative_foreground_rate_macro_category":float(np.mean(macro_fn)) if macro_fn else None,
            "pointing_accuracy_mean":float(np.mean(point_values)) if point_values else None,
            "empty_mask_frequency":float(np.mean(empty_values)) if empty_values else None,
            "small_object_pooled_macro_iou":float(np.mean([v["small_object_pooled_iou"] for v in per_category.values() if v["small_object_pooled_iou"] is not None])) if any(v["small_object_pooled_iou"] is not None for v in per_category.values()) else None,
            "mean_pair_auroc":float(np.mean([x for cid in aurocs for x in aurocs[cid][method]])) if any(aurocs[cid][method] for cid in aurocs) else None,
            "positive_category_count":len(per_category),"positive_image_category_pairs":sum(v["pairs"] for v in per_category.values()),
            "per_category":per_category}
    result["prompt_sensitivity"]={"mean_pixel_probability_variance":float(np.mean(prompt_variance)) if prompt_variance else None,
        "mean_pixel_probability_std":float(np.mean(prompt_std)) if prompt_std else None,
        "mean_foreground_patch_count_by_prompt_at_0_20":[float(np.mean(x)) if x else None for x in prompt_fg_counts]}
    return result


def choose_thresholds(records):
    scores = defaultdict(lambda: defaultdict(lambda: [0.0,0.0,0.0]))
    for row in records:
        for cid, item in row["categories"].items():
            for method, m in item["methods"].items():
                for threshold, values in m["threshold_sweep_tp_fp_fn"].items():
                    key=(method,threshold)
                    scores[cid][key]=[a+b for a,b in zip(scores[cid][key],values)]
    candidates={m:[] for m in METHODS}
    for method in METHODS:
        thresholds=THRESHOLDS
        for threshold in thresholds:
            ious=[]
            for cid in scores:
                v=scores[cid][(method,f"{threshold:.2f}")]
                if v[0]+v[1]+v[2]>0: ious.append(v[0]/(v[0]+v[1]+v[2]))
            candidates[method].append((float(np.mean(ious)) if ious else -1.0,float(threshold)))
    selected={}
    details={}
    for method, values in candidates.items():
        best=max(values,key=lambda q:(q[0],-abs(q[1]-.5),-q[1]))
        selected[method]=best[1]
        details[method]={"threshold":best[1],"development_pooled_macro_iou":best[0],"tie_break":"nearest 0.50, then lower"}
    # V1's published reference remains fixed; report its separate dev-calibrated threshold.
    selected["V1_development_recalibrated"]=selected["V1_seven_template"]
    details["V1_development_recalibrated"]={**details["V1_seven_template"],"derived_from":"V1 seven-template development sweep"}
    return selected,details


def main():
    p=argparse.ArgumentParser()
    for k in ("dataset-root","feature-root","training-output","split-manifest","v1-checkpoint","siglip-checkpoint","output-dir"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args(); out=a.output_dir; out.mkdir(parents=True,exist_ok=True); (out/"qualitative").mkdir(exist_ok=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split=json.loads(a.split_manifest.read_text()); split_hash=sha256(a.split_manifest)
    categories=json.loads((a.training_output/"categories.json").read_text())
    instances=json.loads((a.dataset_root/"annotations2014/instances_val2014.json").read_text())
    geometry=json.loads((a.feature_root/"patch_features_val2014_manifest.json").read_text())["preprocessing_geometry"]
    text_model,tokenizer,text_meta=load_siglip_text_tower(a.siglip_checkpoint,device=str(device))
    names=[c["name"] for c in categories]
    prompts,text_four,text_v1=encode_category_text(text_model,tokenizer,names,device)
    v1,v1_hash=load_frozen_projection(a.v1_checkpoint,device)
    raw=torch.load(a.training_output/"best_projection.pt",map_location="cpu",weights_only=True)
    v2=RegionTextProjection().to(device);v2.load_state_dict(raw["model"],strict=True);v2.eval()
    v2_hash=sha256(a.training_output/"best_projection.pt")
    atomic_json(out/"configuration.json",{"seed":0,"manifests":{"development_sha256":split_hash,"development_images":len(split["development"]["image_ids"]),"evaluation_sha256":split_hash,"evaluation_images":len(split["evaluation"]["image_ids"])},
        "checkpoint_v2_sha256":v2_hash,"checkpoint_v1_sha256":v1_hash,"v2_checkpoint_epoch":40,"v2_scoring":"sigmoid(10*(cosine-0.20)); no PAMR",
        "prompt_conditions":{"A":prompts[0],"B":prompts[1],"C":prompts[2],"D":prompts[3],"E":"mean four raw text embeddings, then MLP projection", "F":"mean four probability maps"},
        "v1_prompt_templates":list(V1_TEMPLATES),"threshold_grid":list(THRESHOLDS),"v1_fixed_threshold":0.55,
        "feature_source":"cached OpenVLA penultimate DINO for V2, cached V1 final-normalized DINO for V1","text_encoder":text_meta,
        "unseen_concept_caveat":"COCO category wording tests are known-category rewording; attributes and genuinely unseen concepts require independent masks and are not claimed from COCO."})
    files={"train":json.loads((a.feature_root/"patch_features_train2014_manifest.json").read_text()),"val":json.loads((a.feature_root/"patch_features_val2014_manifest.json").read_text())}
    if files["val"].get("status")!="complete": raise RuntimeError("Cached validation patch features are not marked complete")
    fixed={"V1_seven_template":.55, **{m:.50 for m in METHODS if m!="V1_seven_template"}}
    dev=score_split("development",split["development"]["image_ids"],output_dir=out,split_hash=split_hash,
        dataset_root=a.dataset_root,feature_root=a.feature_root,categories=categories,instance_data=instances,geometry=geometry,
        text_four=text_four,text_v1=text_v1,v1_projection=v1,v2_projection=v2,device=device)
    selected,cal=choose_thresholds(dev)
    atomic_json(out/"development_thresholds.json",{"manifest_sha256":split_hash,"selected_thresholds":selected,"calibration":cal,"method":"maximize native-grid pooled category IoU on development only; thresholds fixed before final scoring"})
    dev_metrics=aggregate(dev,fixed)
    dev_calibrated=aggregate(dev,{m:selected[m] for m in METHODS})
    dev_calibrated["V1_development_recalibrated"] = aggregate(
        dev, {**{m: selected[m] for m in METHODS}, "V1_seven_template": selected["V1_development_recalibrated"]}
    )["V1_seven_template"]
    atomic_json(out/"development_metrics.json",{"fixed_original_thresholds":dev_metrics,"development_calibrated_thresholds":dev_calibrated})

    # Use the preselected strategy and its development threshold on the already-inspected holdout.
    selected_method=max(METHODS[:-1],key=lambda m:(cal[m]["development_pooled_macro_iou"],m=="A_training_prompt"))
    final_threshold=float(selected[selected_method])
    # Existing final records contain all prompt scores; this remains a secondary, previously inspected result.
    final=score_split("evaluation",split["evaluation"]["image_ids"],output_dir=out,split_hash=split_hash,
        dataset_root=a.dataset_root,feature_root=a.feature_root,categories=categories,instance_data=instances,geometry=geometry,
        text_four=text_four,text_v1=text_v1,v1_projection=v1,v2_projection=v2,device=device)
    final_thresholds={m:(.55 if m=="V1_seven_template" else selected[m]) for m in METHODS}
    final_metrics=aggregate(final,final_thresholds)
    v1_recal=aggregate(final,{**final_thresholds,"V1_seven_template":selected["V1_development_recalibrated"]})["V1_seven_template"]
    atomic_json(out/"evaluation_metrics_secondary_previously_inspected.json",{"status":"complete","method":selected_method,"threshold_from_development":final_threshold,
        "previously_inspected_holdout":True,"metrics":final_metrics,"V1_development_recalibrated_secondary":v1_recal,"image_count":len(final)})
    summary={"status":"complete","development_image_count":len(dev),"final_image_count":len(final),"final_is_previously_inspected":True,
        "selected_prompt_strategy":selected_method,"selected_threshold":final_threshold,"v1_published_threshold":.55,
        "v1_development_recalibrated_threshold":selected["V1_development_recalibrated"],"development_fixed":dev_metrics,
        "development_calibrated":dev_calibrated,"secondary_holdout_selected_condition":final_metrics,
        "prompt_robustness_interpretation":"COCO object names were seen during training; phrase changes test paraphrase robustness only, not zero-shot category transfer.",
        "unseen_attributes_and_concepts":"Not scored without independent attribute/concept masks; deferred to the LIBERO ground-truth workstream.",
        "thresholds_selected_only_on_development":True,"v1_prompts":list(V1_TEMPLATES),"v2_checkpoint_sha256":v2_hash,"v1_checkpoint_sha256":v1_hash}
    atomic_json(out/"prompt_ablation_metrics.json",summary)
    report=["# Prompt Robustness and Language Ablation","",
        "## Protocol","",
        f"Frozen V2 epoch 40 (SHA-256 `{v2_hash}`) is scored on cached penultimate OpenVLA DINO patches. No weights are updated. The dense score is `sigmoid(10 × (cosine − 0.20))`. The split manifest SHA-256 is `{split_hash}`.","",
        "Conditions A–D change wording only. E averages the four raw SigLIP text embeddings before MLP projection. F averages the four probability maps. V1 retains its seven-template averaged embedding and the fixed 0.55 reference threshold; a separate development-calibrated V1 threshold is also recorded.","",
        "## Results","",
        f"Development selected `{selected_method}` at binary threshold `{final_threshold:.2f}`. The final-set result is secondary and previously inspected; it did not select the strategy or threshold.","",
        "See `development_metrics.json`, `development_thresholds.json`, and `evaluation_metrics_secondary_previously_inspected.json` for per-category and aggregate measurements.","",
        "## Interpretation and limits","",
        "COCO object categories appear in V2's training data. These results measure prompt rewording for known categories, not zero-shot generalization to unseen classes. `black bowl` attributes and unseen concepts such as `plate`, `ramekin`, `drawer`, or `robot gripper` are not assigned accuracy without independent ground-truth masks.","",
        "Prompt sensitivity is summarized from the four equivalent descriptions' per-pixel probability variance and foreground-count differences. The 3,000-image evaluation split was inspected in an earlier run, so it is not a new blind test.",""]
    (out/"PROMPT_ABLATION_REPORT.md").write_text("\n".join(report))
    print(json.dumps({"status":"complete","selected":selected_method,"threshold":final_threshold,"dev_images":len(dev),"eval_images":len(final)},indent=2),flush=True)


if __name__=="__main__": main()
