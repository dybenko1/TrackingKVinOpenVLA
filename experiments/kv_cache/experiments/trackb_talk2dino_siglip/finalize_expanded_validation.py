#!/usr/bin/env python3
"""Assemble compact final report from completed full-val retrieval/segmentation."""
import argparse,json,hashlib,subprocess
from pathlib import Path

def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();o=a.output_dir
 retrieval=json.loads((o/'full_val2014_retrieval_summary.json').read_text())
 segdir=o/'segmentation'; seg=json.loads((segdir/'summary.json').read_text())
 libero_path=o/'libero_feasibility'/'summary.json'
 libero=json.loads(libero_path.read_text()) if libero_path.is_file() else None
 split=json.loads((o/'validation_split_manifest.json').read_text())
 thresholds=json.loads((segdir/'threshold_calibration.json').read_text())
 best=json.loads((segdir/'development_selection.json').read_text())['selected_primary_native_grid16']
 selected_eval=seg['evaluation_aggregate'][best['checkpoint']]
 feature_validation=json.loads((o/'FULL_FEATURE_VALIDATION.json').read_text())
 split_manifest=json.loads((o/'validation_split_manifest.json').read_text())
 seg_manifest=json.loads((o/'segmentation_manifest.json').read_text())
 dino_status=json.loads(Path('/opt/openvla/cache/datasets/talk2dino/coco2014/features_full_validation_20261117/full_val2014_extraction_status.json').read_text())
 git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[4],text=True).strip()
 config={'code_commit':git_sha,'dataset':'COCO val2014','dataset_counts':{'images':40504,'captions':202654},
  'dataset_root':'/opt/openvla/cache/datasets/talk2dino/coco2014','caption_annotations_sha256':split_manifest['annotation_sha256'],
  'instance_annotations_sha256':file_sha(Path('/opt/openvla/cache/datasets/talk2dino/coco2014/annotations2014/instances_val2014.json')),
  'retrieval_split_manifest_sha256':hashlib.sha256((o/'validation_split_manifest.json').read_bytes()).hexdigest(),
  'segmentation_manifest_sha256':hashlib.sha256((o/'segmentation_manifest.json').read_bytes()).hexdigest(),
  'segmentation_split':seg_manifest['selection'],'retrieval_split':{'development':split_manifest['development'],'evaluation':split_manifest['evaluation']},
  'model':{'openvla_checkpoint':'openvla/openvla-7b-finetuned-libero-spatial','vision':'fine-tuned DINOv2 ViT-L/14 reg4 TIMM backbone','vision_feature_definition':'final normalized DINO patches and final-block CLS-to-patch attention for segmentation; Talk2DINO 16-head weighted region descriptors for retrieval','input_resolution':224,'grid':'16x16','text':'timm/ViT-SO400M-14-SigLIP-384','text_checkpoint_metadata':seg['siglip_text_metadata'],'text_context':64,'no_448':True},
  'checkpoints':retrieval['epochs'],'segmentation_configuration':seg['runtime'],'feature_validation':feature_validation,
  'visual_feature_extraction':dino_status,'threshold_candidates':thresholds['calibrated_thresholds'],
  'methods':['raw','attention_cleaned_lambda_bg_0.2','pamr_raw','attention_cleaned_then_pamr'],'pamr':{'iterations':10,'dilations':[1,2,4,8,12,24]},
  'selection_policy':'Choose checkpoint and postprocess by development native 16x16 macro class mIoU; final evaluation is reported separately.',
  'no_training':True,'no_policy_rollouts':True,'no_kv_changes':True}
 (o/'configuration.json').write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
 lines=['# Full COCO val2014 retrieval and segmentation validation','',
  '## Design and split integrity','',
  f"- Retrieval covered **{split['image_count']:,} images and {split['caption_count']:,} captions**. Development: {split['development']['image_count']:,} images / {split['development']['caption_count']:,} captions; final evaluation: {split['evaluation']['image_count']:,} images / {split['evaluation']['caption_count']:,} captions.",
  f"- Segmentation used {seg['development_images']} development and {seg['evaluation_images']} final images with COCO instance masks. The manifests are disjoint and exclude all previously examined Phase 3 manifest images.",
  '- Checkpoints were frozen epochs 20, 50, 100; OpenVLA-native DINO features and preprocessing remain 224×224. No training, CLIP baseline, or 448px input was used.',
  '- Threshold calibration used development images only. Final evaluation metrics were not used to select the checkpoint or thresholds.', '',
  '## Full val2014 multi-positive retrieval','',
  '| Epoch | Split | I→T R@1 | I→T R@5 | I→T R@10 | T→I R@1 | T→I R@5 | T→I R@10 | Symmetric multi-positive InfoNCE |',
  '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
 for epoch,cp in retrieval['epochs'].items():
  for sn in ('development','evaluation'):
   x=cp[sn]; i=x['image_to_text'];t=x['text_to_image']; n=x['multi_positive_infonce']['symmetric_mean']
   lines.append(f"| {epoch} | {sn} | {i['recall_at_1']:.4f} | {i['recall_at_5']:.4f} | {i['recall_at_10']:.4f} | {t['recall_at_1']:.4f} | {t['recall_at_5']:.4f} | {t['recall_at_10']:.4f} | {n:.4f} |")
 lines += ['', 'Image-to-text retrieval counts any caption for that image as positive. Text-to-image uses its paired image. Similarities are streamed in chunks; no full similarity matrix is materialized. Ranking ties are treated as not outranking a positive.', '',
  '## Segmentation postprocessing ablation','',
  f"Development-selected primary native-grid condition: **{best['checkpoint']} + {best['method']}**, threshold {best['threshold']:.2f}, development macro class mIoU {best['development_macro_class_miou']:.4f}.", '',
  '| Checkpoint | Method | Scale | Dev-calibrated macro class mIoU | AUROC | Precision | Recall | Pointing |',
  '|---|---|---|---:|---:|---:|---:|---:|---:|']
 for cp,methods in seg['evaluation_aggregate'].items():
  for method,scales in methods.items():
   for scale,policies in scales.items():
    x=policies['development_calibrated']
    lines.append(f"| {cp} | {method} | {scale} | {x['macro_mean_category_iou']:.4f} | {x['mean_auroc']:.4f} | {x['mean_precision']:.4f} | {x['mean_recall']:.4f} | {x['mean_pointing_accuracy']:.4f} |")
 lines += ['', f"### Small-object and postprocessing diagnostics ({best['checkpoint']}, native grid)", '',
  'Size groups are based on the number of occupied 16×16 cells in a per-image category-union mask: small <4, medium 4–15, large ≥16. This is a grid-resolution proxy, not physical object-size annotation.', '',
  '| Method | Small mIoU (pairs) | Medium mIoU (pairs) | Large mIoU (pairs) |',
  '|---|---:|---:|---:|']
 for method in ('raw','cleaned','pamr_raw','cleaned_pamr'):
  x=selected_eval[method]['grid16']['development_calibrated']['by_size_group']
  cells=[]
  for group in ('small','medium','large'):
   v=x.get(group);cells.append(f"{v['iou']:.4f} ({v['pairs']})" if v else '—')
  lines.append(f"| {method} | {cells[0]} | {cells[1]} | {cells[2]} |")
 raw=selected_eval['raw']['grid16']['development_calibrated']['macro_mean_category_iou']
 deltas={m:selected_eval[m]['grid16']['development_calibrated']['macro_mean_category_iou']-raw for m in ('cleaned','pamr_raw','cleaned_pamr')}
 lines += ['', 'Held-out macro category mIoU change relative to raw at this same checkpoint/mask scale: '
  + ', '.join(f"{m} {d:+.4f}" for m,d in deltas.items()) + '. These paired aggregate differences indicate whether cleaning/refinement helped; they do not identify causal failure mechanisms by themselves.', '']
 prior=0.30588934948829305
 current=seg['evaluation_aggregate']['epoch_100']['cleaned']['grid16']['historical_phase3_0.54']['macro_mean_category_iou']
 lines += [f"For historical context only, epoch-100 cleaned at the fixed 0.54 threshold scored {current:.4f} on this new final holdout versus the prior 0.3059 on the previously examined 80-image subset (descriptive difference {current-prior:+.4f}). The subsets differ, so this is not a paired estimate of improvement.", '']
 lines += ['', 'Per-category and per-image metrics are in `segmentation/per_category_metrics.csv` and `segmentation/per_image_metrics.jsonl`. The gallery contains example RGB/GT/raw/cleaned/PAMR maps for the final evaluation subset.', '',
  '## Limits','',
  '- This is semantic category-union evaluation against COCO instance masks projected to the 16×16 patch grid and aligned 224×224 view; it is not instance segmentation AP.',
  '- The evaluation subset was selected from annotation availability and geometric visibility, without model predictions. It excludes the older 100-image Phase 3 manifest but remains one COCO dataset/domain.',
  '- The 0.3059 prior mIoU is a historical result on the previously examined 80-image subset and is not a matched test-set baseline for this holdout.',
  '- PAMR and attention-based cleaning are postprocessing; there is no policy/KV reuse or LIBERO action rollout.', '',
  '## LIBERO feasibility', '']
 if libero:
  lines += [f"- Offline mask review covers {libero['frames']} saved RGB frames across tasks {', '.join(map(str,libero['tasks']))}, states {', '.join(map(str,libero['states']))}.",
    f"- Selected configuration: {libero['checkpoint']} + `{libero['postprocessing']}`, development-calibrated threshold {libero['threshold_from_coco_development_only']:.2f}.",
    f"- Sparse-frame temporal mask IoU by prompt: `{json.dumps(libero['mean_mask_iou_by_prompt'],sort_keys=True)}`; source/current patch flips: `{json.dumps(libero['mean_flip_tokens_by_prompt'],sort_keys=True)}`.",
    '- These RGB-only artifacts have no paired oracle segmentation mask; the sparse frames are not adjacent control steps. This is qualitative feasibility evidence, not accuracy or cache-policy validation.',
    '- See `libero_feasibility/LIBERO_FEASIBILITY_REPORT.md` and its per-frame visualizations.', '']
 else:
  lines += ['The saved-frame LIBERO mask review is pending and will be appended when available.', '']
 lines += [
  '## Artifacts','',
  '- `validation_split_manifest.json`, `segmentation_manifest.json` — frozen image-level splits and selection criteria.',
  '- `full_val2014_retrieval_summary.json` — full retrieval metrics.',
  '- `segmentation/summary.json`, `segmentation/threshold_calibration.json`, `segmentation/per_category_metrics.csv`, `segmentation/per_image_metrics.jsonl` — segmentation metrics.',
  '- `configuration.json` and `run_status.json` — exact data paths, hashes, software/checkpoint identities, and completion status.', '']
 (o/'POSTPROCESS_ABLATION_REPORT.md').write_text('\n'.join(lines))
 summary={'status':'complete','retrieval':retrieval,'segmentation':seg,'libero_feasibility':libero,'configuration':config,
  'split':{'retrieval_images':split['image_count'],'retrieval_captions':split['caption_count'],
    'retrieval_dev_images':split['development']['image_count'],'retrieval_eval_images':split['evaluation']['image_count'],
    'retrieval_dev_captions':split['development']['caption_count'],'retrieval_eval_captions':split['evaluation']['caption_count']},
  'best_dev_selected_native_grid16':best,'threshold_calibration':thresholds['calibrated_thresholds'],
  'historical_phase3_0_3059_note':'Previously examined 80-image subset; descriptive context only, not a matched holdout baseline.'}
 tmp=o/'summary.json.tmp';tmp.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');tmp.replace(o/'summary.json')
 print(o/'POSTPROCESS_ABLATION_REPORT.md')
if __name__=='__main__':main()
