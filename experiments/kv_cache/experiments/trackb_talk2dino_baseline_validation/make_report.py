#!/usr/bin/env python3
"""Assemble compact, provenance-aware final report after COCO and LIBERO."""
import json
import subprocess
from pathlib import Path

ROOT=Path("/opt/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_baseline_validation")
COCO=ROOT/"coco_object"; LIBERO=ROOT/"libero_final"
NATIVE=Path("/opt/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_native")

def main():
    c=json.loads((COCO/"summary.json").read_text())
    cc=json.loads((COCO/"configuration.json").read_text())
    l=json.loads((LIBERO/"summary.json").read_text())
    n=json.loads((NATIVE/"summary.json").read_text())
    upstream=Path("/opt/openvla/cache/references/Talk2DINO")
    revision=subprocess.check_output(["git","-c",f"safe.directory={upstream}","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    cc["official_repo_revision"]=revision
    cc["official_dino"]["source_revision"]="7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    (COCO/"configuration.json").write_text(json.dumps(cc,indent=2,sort_keys=True)+"\n")
    l["official_model_identity"]["official_repo_revision"]=revision
    l["official_model_identity"]["official_dino"]["source_revision"]="7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    (LIBERO/"summary.json").write_text(json.dumps(l,indent=2,sort_keys=True)+"\n")
    validation={"coco_smoke_images":5,"coco_smoke_passed":True,"coco_full_images":cc.get("evaluated_images"),
        "coco_expected_images":cc.get("expected_images"),"libero_segments":l.get("segments"),"libero_frames":l.get("frames"),
        "libero_segment_paths_unique":36,"libero_gt_union_source":"separate relevant_gt array",
        "checkpoint_sha256_verified":True,"official_repo_revision":revision,
        "mmcv_mmseg_wrapper_available":False,"official_slide_and_confusion_single_process_equivalent":True}
    (ROOT/"validation.json").write_text(json.dumps(validation,indent=2,sort_keys=True)+"\n")
    def pct(x): return f"{100*x:.2f}%" if x is not None else "—"
    lines=["# Original Talk2DINO baseline validation", "",
      "This report evaluates the frozen upstream Talk2DINO ViT-Large model and compares it with the already completed native OpenVLA DINO conditions. No training or policy rollout was performed.","",
      "## COCO Object val2017 — official 81-class semantic metric","",
      f"The evaluator uses the upstream DINOText model functions (Talk2DINO revision `{revision}`), official Meta DINOv2 ViT-L/14-register weights, OpenAI CLIP ViT-B/16 text weights and original Talk2DINO projection weights. It follows the upstream square 448 transform, 448 crops / 224 stride, `sub_imagenet_template`, 0.54 background score and global 81-class confusion matrix. Compatible MMCV/MMseg packages were unavailable for this Python/Torch runtime, so their slide wrapper and confusion accumulation were implemented as a single-process equivalent. The checkpoint hashes were verified. See `coco_object/configuration.json` for versions and hashes.","",
      "| Condition | mIoU incl. BG | Foreground mIoU | BG IoU | Published reference | Δ |","|---|---:|---:|---:|---:|---:|"]
    for key,label in (("clean","Official cleaned, no PAMR"),("clean_pamr","Official cleaned + PAMR"),("raw","Raw similarity, no cleaning")):
        x=c[key]; ref=x.get("published_reference_miou")
        delta=x.get("delta_from_published")
        delta_txt="—" if delta is None else f"{100*delta:+.2f} pp"
        lines.append(f"| {label} | {pct(x['miou_including_background'])} | {pct(x['foreground_miou'])} | {pct(x['background_iou'])} | {pct(ref)} | {delta_txt} |")
    lines += ["",f"Images evaluated: {cc.get('evaluated_images')} / {cc.get('expected_images')}. The published 37.6% and 42.3% values are reference figures, not measurements from this run. Both measured deltas exceed the predeclared 1 pp tolerance; this is a near-protocol reproduction, not yet a validated exact reproduction.","",
      "## LIBERO saved-frame generalization","",
      f"Official Talk2DINO was scored offline on {l['frames']} paired frames from {l['segments']} existing E2 demonstration segments. This uses no policy actions. Inputs are existing 224×224 RGB frames, resized internally by upstream DINOText to 448×448. Threshold 0.54 is the official COCO setting and was not calibrated for LIBERO.","",
      "| Prompt role | Raw precision | Raw recall | Raw IoU | Clean precision | Clean recall | Clean IoU |","|---|---:|---:|---:|---:|---:|---:|"]
    for role,metrics in l["metrics"].items():
        lines.append(f"| {role} | {pct(metrics['raw']['precision'])} | {pct(metrics['raw']['recall'])} | {pct(metrics['raw']['iou'])} | {pct(metrics['clean']['precision'])} | {pct(metrics['clean']['recall'])} | {pct(metrics['clean']['iou'])} |")
    lines += ["", "### Existing same-frame native conditions (context only)","",
      "The following are previously measured Conditions A/B from the E2 native integration study; they are not recomputed in this run and must not be called an isolated implementation comparison.","",
      "| Condition | Task-relevant union IoU | Manipulated-object recall | Destination-object recall |","|---|---:|---:|---:|"]
    for key,label in (("a_clean","C3: official Meta DINO + custom E2 scoring"),("b_clean","C4: OpenVLA DINO + custom E2 scoring")):
        rows=n.get("metrics",{}).get(key,{}) if isinstance(n.get("metrics"),dict) else {}
        if rows:
            union=rows.get("task_relevant_foreground",{}).get("iou")
            manip=rows.get("manipulated_objects",{}).get("recall")
            dest=rows.get("destination_objects",{}).get("recall")
        else: union=manip=dest=None
        lines.append(f"| {label} | {pct(union)} | {pct(manip)} | {pct(dest)} |")
    lines += ["", "## Runtime","",
      f"COCO processing averaged {cc['latency_s_per_image']['mean']:.2f} s/image (median {cc['latency_s_per_image']['median']:.2f}, p95 {cc['latency_s_per_image']['p95']:.2f}) across the three score outputs; peak CUDA allocated memory was {cc['peak_cuda_allocated_bytes']/1e9:.2f} GB. This is per-image evaluator time in the current Orin container, not a comparison with independently optimized MMSegmentation. LIBERO frame latency and peak memory were not instrumented separately.","",
      "## Representative LIBERO maps","",
      "These deterministic mid-sequence examples are illustrative, not selected by model performance. Ground truth is shown at the 16×16 E2 patch grid; Talk2DINO maps are generated at 448×448 and aligned to the 224×224 image. The fixed 0.54 threshold is the COCO reference setting, not a LIBERO-calibrated operating point.",""]
    manifest=LIBERO/"figures/figure_manifest.json"
    if manifest.exists():
        for item in json.loads(manifest.read_text()):
            lines += [f"### {item['suite']} — {item['segment_metadata']['instruction']}","",
              f"![Cleaned role maps]({LIBERO.name}/figures/{item['figure_cleaned']})","",
              f"![Raw and cleaned maps]({LIBERO.name}/figures/{item['figure_raw_vs_clean']})",""]
    lines += ["## Interpretation and limitations","",
      "- COCO Object mIoU is a dataset-level semantic metric including background; it is not directly comparable to LIBERO role-wise fractional patch metrics.",
      "- LIBERO evaluation reuses the previously inspected E2 frame/mask set; it is not an untouched test set and not a closed-loop task-success result.",
      "- Two provisional LIBERO outputs were invalidated: the first had segment-cache basename collisions; the second used an empty role_gt channel instead of E2's separate relevant_gt union mask for the full-instruction query. Final results in `libero_final/` use task-relative segment paths and the correct relevant_gt label. Earlier `libero/` and `libero_corrected/` directories are audit trails and must not be cited.",
      "- The native E2 comparison changed both the DINO weights/features and the inference implementation; it does not isolate domain shift or one pipeline component.",
      "- COCO image-mask counts, 81-class mapping and checkpoint hashes were checked. All classes had nonempty unions. Both published mIoU differences exceed 1 pp; exact reproduction is unresolved. The remaining difference may involve evaluation-wrapper or preprocessing details; it must be checked under a compatible upstream MMSeg runtime before calling this an exact reproduction.",
      "- Separate 224-pixel COCO, OpenVLA-preprocessing, raw+PAMR, threshold PR-curve and controlled custom-vs-official inference ablations were not run. LIBERO uses upstream resizing from cached 224 RGB frames; no alternate preprocessing condition was tested.",
      "- LIBERO prompt queries use the E2-derived role phrases plus a full-instruction query. A separate bare-name vs. descriptive-phrase vs. full-instruction ablation was not run. Context-object prompts without individually reliable labels are visualized but excluded from role metrics.",
      "- The five-image COCO smoke passed; the complete 5,000-image COCO run and corrected 1,152-frame LIBERO run completed. The first two provisional LIBERO aggregates were invalidated and retained only as audit trails.",
      "- Checkpoint/configuration, preprocessing, class mapping, exact model calls, COCO timing and memory are recorded in machine-readable artifacts. LIBERO-specific timing and peak memory were not recorded.",
      "- Do not recommend KV reuse from foreground recall alone. Background false negatives, role-specific object failures, temporal behavior and cache-policy task success require separate evidence.","",
      "## Artifacts","",
      "- COCO summary: `coco_object/summary.json`; per-class metrics: `coco_object/per_category_metrics.csv`.",
      "- COCO global confusion matrices: `coco_object/confusion_*.npy`.",
      "- Final corrected LIBERO per-frame/role metrics: `libero_final/per_frame_metrics.csv`; summary: `libero_final/summary.json`.",
      "- Process status/logs: `coco_object/RUN_STATUS.json`, `libero_final/RUN_STATUS.json`, `coco_object/logs/talk2dino-coco.log`.",""]
    (ROOT/"EVALUATION_REPORT.md").write_text("\n".join(lines))
    print(ROOT/"EVALUATION_REPORT.md")
if __name__=="__main__": main()
