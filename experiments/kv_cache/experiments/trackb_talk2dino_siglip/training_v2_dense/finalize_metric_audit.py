"""Validate and summarize the corrected V2 metric/prompt audit outputs."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n")
    tmp.replace(path)

def f(v):
    return "—" if v is None else f"{100*v:.2f}%"

def metric_row(label,split,method,prompt,threshold,summary):
    if method is None:
        return f"| {label} | {split} | {prompt} | — | {f(0.3059)} (reported) | — | {threshold} |"
    q=summary[method]
    a=q["grid16"];b=q["224"]
    return f"| {label} | {split} | {prompt} | {f(a.get('macro_category_miou_pooled'))} | {f(a.get('macro_mean_positive_image_category_iou'))} | {f(b.get('macro_category_miou_pooled'))} | {threshold} |"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audit-dir",type=Path,required=True)
    p.add_argument("--train-output",type=Path,required=True)
    p.add_argument("--v1-report",type=Path,required=True)
    p.add_argument("--v1-postprocess-summary",type=Path,required=True)
    a=p.parse_args(); out=a.audit_dir
    complete=json.loads((out/"evaluation_complete.json").read_text())
    if complete.get("status")!="complete" or complete.get("images")!={"development":1000,"evaluation":3000}:
        raise RuntimeError(f"evaluation coverage/status invalid: {complete}")
    for split,n in (("development",1000),("evaluation",3000)):
        path=out/f"{split}_per_image.jsonl"
        with path.open() as h:rows=[json.loads(line) for line in h if line.strip()]
        ids=[int(r["image_id"]) for r in rows]
        if len(rows)!=n or len(set(ids))!=n:raise RuntimeError(f"{split} per-image integrity failed: {len(rows)} rows, {len(set(ids))} unique")
    thresholds=json.loads((out/"development_thresholds.json").read_text())
    dev=json.loads((out/"development_metrics.json").read_text())
    final=json.loads((out/"evaluation_metrics.json").read_text())
    corrected=json.loads((a.train_output/"evaluation_audit_20261120/corrected_training_dev_metrics.json").read_text())
    completion=json.loads((a.train_output/"training_completion.json").read_text())
    cats=json.loads((a.train_output/"categories.json").read_text())
    catnames={str(c["id"]):c["name"] for c in cats}
    with (out/"per_category_metrics.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=["category_id","category_name","method","scale","pooled_iou","mean_positive_image_iou","positive_image_category_pairs","precision","recall","tp","fp","fn"],lineterminator="\n")
        w.writeheader()
        for row in final["per_category"]:
            w.writerow({"category_id":row["category_id"],"category_name":catnames.get(str(row["category_id"]),""),
                "method":row["method"],"scale":row["scale"],"pooled_iou":row["iou"],
                "mean_positive_image_iou":row["mean_positive_image_iou"],
                "positive_image_category_pairs":row["positive_image_category_pairs"],
                "precision":row["precision"],"recall":row["recall"],
                "tp":row["tp"],"fp":row["fp"],"fn":row["fn"]})
    # Historical V1 values are carried forward exactly as published in the
    # earlier Phase 3 report; they are not recomputed on this new split.
    v1pub={"checkpoint":"full COCO V1 epoch 100","images":80,"category_image_pairs":240,
        "cleaned_macro_category_miou":0.3059,"raw_macro_category_miou":0.2880,
        "cleaned_auroc":0.8231,"cleaned_pointing_accuracy":0.5375,
        "threshold":0.54,"pamr":False,"source_report":str(a.v1_report)}
    post=json.loads(a.v1_postprocess_summary.read_text())
    v1_earlier=post["evaluation_aggregate"]["epoch_050"]["pamr_raw"]["grid16"]["development_calibrated"]
    local_only=[]
    for path in [out/"development_per_image.jsonl",out/"evaluation_per_image.jsonl",out/"logs/audit_pipeline.log",*sorted((out/"qualitative_results").glob("*"))]:
        if path.is_file():local_only.append({"path":str(path),"bytes":path.stat().st_size,"sha256":sha256(path)})
    publication={"github_candidate_files":["AUDIT_REPORT.md","summary.json","per_category_metrics.csv","development_metrics.json","evaluation_metrics.json","development_thresholds.json","evaluation_metadata.json","evaluation_complete.json","corrected_training_dev_metrics.json"],"kept_local_with_integrity_hashes":local_only}
    summary={"status":"complete","evaluation_counts":complete["images"],
        "split_manifest_sha256":sha256(out/"validation_manifest.json"),
        "v1_prompt_templates":json.loads((out/"evaluation_metadata.json").read_text())["v1_prompt_templates"],
        "thresholds":thresholds["selected_thresholds"],"corrected_training_dev_metric":corrected,
        "v1_published_baseline":v1pub,
        "v1_epoch50_prior_2000_image_holdout":{"grid16_macro_mean_positive_image_iou":v1_earlier["macro_mean_category_iou"],"grid16_mean_positive_pair_iou":v1_earlier["mean_pair_iou"],"category_image_pairs":v1_earlier["category_image_pairs"],"method":"PAMR raw","source_summary":str(a.v1_postprocess_summary)},
        "v1_recomputed_new_holdout":final["summary"].get("v1_raw_pamr"),
        "v2_recomputed_new_holdout":{"raw":final["summary"].get("v2_raw"),"pamr":final["summary"].get("v2_pamr")},
        "development_metrics":dev["summary"],"final_metrics":final["summary"],
        "v1_checkpoint_sha256":complete["v1_checkpoint_sha256"],"v2_checkpoint_sha256":complete["v2_checkpoint_sha256"],
        "v2_best_epoch_original_selection":completion["best"],
        "selection_limitation":"Epoch 40 was originally selected using the old development tie-break metric. Only best_dev.pt, latest.pt and the initial weights are available; earlier epoch model snapshots are unavailable, so corrected metrics cannot retroactively select another epoch.",
        "final_set_disclosure":"The earlier evaluator processed the final split and its one-row aggregate was inspected. Its per-image JSONL was repeatedly truncated, leaving one row per split. This corrected recomputation uses development-only threshold calibration but the final split is no longer an uninspected blind set.",
        "legacy_invalid_artifact":{"path":"../evaluation/evaluation_complete.json","recorded_images":{"development":1,"evaluation":1},"reason":"The per-image JSONL writer reopened in write mode on every iteration; this artifact is preserved but invalid for aggregate claims."},"artifact_manifest":publication}
    atomic_json(out/"artifact_manifest.json",publication)
    atomic_json(out/"summary.json",summary)
    table=[
        metric_row("V1 published baseline","prior 80-image evaluation; V1 epoch 100",None,"seven templates; no PAMR","0.54",{}),
        f"| V1 prior larger holdout reference | prior 2,000-image holdout; epoch 50 + raw PAMR | seven templates | — | {f(v1_earlier['macro_mean_category_iou'])} | — | prior dev calibration |",
        metric_row("V1 recomputed","new 3,000-image holdout; V1 epoch 50 + raw PAMR","v1_raw_pamr","faithful seven-template mean","fixed 0.55",final["summary"]),
        metric_row("V2 raw","same new 3,000-image holdout; best V2 epoch 40","v2_raw","single training prompt","development-calibrated",final["summary"]),
        metric_row("V2 + PAMR","same new 3,000-image holdout; best V2 epoch 40","v2_pamr","single training prompt","development-calibrated",final["summary"]),
    ]
    report=["# Track B V2 Metric and Prompt Audit","",
        "## Status and scope","",
        "The 40-epoch V2 training is preserved; this audit performs no training, policy rollouts, or KV-cache changes. All corrected metrics below are based on the existing frozen checkpoints and saved COCO val2014 features/masks.","",
        "## Corrected development checkpoint metric","",
        f"The saved epoch-{corrected['checkpoint_epoch']} checkpoint has native-grid pooled macro-category mIoU **{corrected['best_macro_category_miou']:.4f}** at threshold **{corrected['best_threshold']:.2f}**. Corrected small-object IoU is **{corrected['best_small_object_iou']:.4f}** (effective valid occupancy >0 and <4 patch equivalents; absent categories excluded). The old reported value was {completion['best'].get('small'):.4f}; it included absent categories and is invalid. The macro mIoU and threshold are unchanged.","",
        "The saved best epoch was selected under the original metric. Only `best_dev.pt`, `latest.pt`, and `initial_mlp.pt` are available; earlier epoch weights are not. Thus this is a corrected evaluation of the saved checkpoint, not corrected epoch selection.","",
        "## Comparison results","",
        "All IoU values in the table are percentages. `Pooled macro IoU` sums intersections/unions across all positive image/category pairs per category, then averages categories; absent-category images are excluded from both measures. `Mean pair IoU` averages IoU over positive image/category pairs within each category, then macro-averages categories; this matches the prior V1 Phase 3 aggregation.","",
        "| Result | Split / checkpoint | Prompt configuration | 16x16 pooled macro IoU | 16x16 mean positive-pair macro IoU | 224x224 pooled macro IoU | Threshold |","|---|---|---|---:|---:|---:|---|",
        *table,"",
        "## Prompt and representation differences","",
        "V1 uses the seven template strings from the existing Phase 3 `encode_class_templates()` path: `itap of a {}`, `a bad photo of a {}`, `a origami {}`, `a photo of the large {}`, `a {} in a video game`, `art of the {}`, and `a photo of the small {}`. Their raw text embeddings are averaged before V1 projection and normalization. V2 uses the single `a photo of a <category>` prompt used during training. V1 uses final-layer normalized DINO patches; V2 uses the penultimate DINO patches from OpenVLA's standard path. Therefore the comparison is a system-level comparison, not an isolated loss ablation.","",
        "## Ground truth and aggregation audit","",
        "COCO val2014 non-crowd instance masks are unioned by semantic category. Crowd pixels are ignored. Grid targets are fractional 16x16 occupancy with fractional valid-cell weights; 224x224 targets are binary and use valid non-crowd pixels. Categories with no valid positive ground-truth occupancy are excluded. Pointing accuracy selects the highest-scoring valid location. Both pooled per-category IoU and mean positive image/category IoU are emitted.","",
        "## Previous evaluator defect and held-out disclosure","",
        "The earlier evaluator repeatedly opened each per-image JSONL in `w` mode. Although it processed the configured 1,000/3,000 images, only one record survived in each file; the old `evaluation_complete.json` therefore records 1/1 and its metrics are invalid. Those files are preserved unchanged. The corrected run writes to this separate audit directory and refuses completion unless exact split counts are present.","",
        "The final split was already processed by the earlier run and its one-record aggregate was inspected. Corrected thresholds are selected only on the fixed development split, never the final split, but the final evaluation is not an untouched blind test. See `development_thresholds.json`, `evaluation_metadata.json`, and `summary.json` for provenance.","",
        "## Files","",
        "- `summary.json`: corrected metrics, provenance, and disclosure.",
        "- `development_metrics.json` and `evaluation_metrics.json`: summary and per-category metrics.",
        "- `per_category_metrics.csv`: corrected held-out per-category metrics.",
        "- `development_thresholds.json`: development-only threshold calibration.",
        "- `development_per_image.jsonl` and `evaluation_per_image.jsonl`: complete image records.",
        "- `evaluation_metadata.json`: prompt, feature, and ground-truth definitions.",
        "- `corrected_training_dev_metrics.json`: corrected small-object metric and checkpoint-selection caveat.",
        "- `artifact_manifest.json`: hashes and sizes for local-only per-image rows, log, and figures.","",
        "## Limitations","",
        "The V1 and V2 feature layers and prompt aggregation differ. V1 historical numbers use another subset and metric/reporting context. The selected V2 checkpoint cannot be retroactively reselected without earlier epoch weights. The final holdout has been previously inspected. These data do not establish LIBERO task success, safe KV reuse, or latency gains.",""]
    (out/"AUDIT_REPORT.md").write_text("\n".join(report))
    print("Validated audit outputs; wrote",out/"AUDIT_REPORT.md")

if __name__=="__main__":main()
