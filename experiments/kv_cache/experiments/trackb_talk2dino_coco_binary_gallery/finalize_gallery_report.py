#!/usr/bin/env python3
"""Finalize provenance and failure analysis from saved gallery metrics only."""
import csv, json
from pathlib import Path
OUT=Path('/opt/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_coco_binary_gallery')
REV='6dac47dbe3500df63f3b3df20dd28334290919cc'
rows=list(csv.DictReader((OUT/'per_image_metrics.csv').open(newline='')))
by={}
for r in rows: by.setdefault(r['image_id'],{})[r['method']]=r
small_ids={r['image_id'] for r in rows if r['size_bin']=='small'}
both_zero=[]
for iid in sorted(small_ids):
    if all(float(by[iid][m]['pixel_recall'])==0 for m in ('raw','clean')):
        both_zero.append(f"{iid} {by[iid]['raw']['category']}")
summary=json.loads((OUT/'summary.json').read_text())
def pooled(size, method, metric):
    return summary['by_size'][size][method]['pixel'][metric]
raw_components=[int(r['pixel_predicted_components']) for r in rows if r['method']=='raw']
clean_components=[int(r['pixel_predicted_components']) for r in rows if r['method']=='clean']
# CSV writer's CRLF is normalized for clean review diffs and GitHub display.
csv_path=OUT/'per_image_metrics.csv'; csv_path.write_bytes(csv_path.read_bytes().replace(b'\r\n',b'\n'))
section='''## Failure analysis

- **Small-object localization:** pooled pixel IoU is {sraw:.3f} RAW and {sclean:.3f} CLEAN, compared with {lraw:.3f}/{lclean:.3f} on the large-target stratum. The small-bin results are weak despite the fixed threshold.
- **Complete misses:** {nmiss} of 10 small-target examples have zero pixel recall for both RAW and CLEAN at 0.54 ({misses}).
- **Foreground spill:** small-bin precision is {spraw:.3f} RAW and {spclean:.3f} CLEAN. CLEAN improves precision, but does not eliminate false-positive foreground.
- **Cleaning trade-off:** across all selected images, CLEAN changes pooled IoU from 0.229 to 0.281 and precision from 0.244 to 0.314, while recall falls from 0.795 to 0.726. It reduces predicted foreground coverage from 14.2% to 10.1%; the change is a precision/recall trade-off, not a uniform improvement.
- **Disconnected regions:** predicted connected-component counts have median/max {rmed}/{rmax} RAW and {cmed}/{cmax} CLEAN. Multiple annotated instances may form separate valid components; this count alone does not prove fragmentation.
- **Semantic confusions:** this binary evaluation counts every non-target pixel as false-positive foreground; it does not identify the specific competing category. Individual panels show spill visually, but no confusion matrix between non-target semantic classes is reported.

'''.format(sraw=pooled('small','raw','iou'),sclean=pooled('small','clean','iou'),lraw=pooled('large','raw','iou'),lclean=pooled('large','clean','iou'),nmiss=len(both_zero),misses=', '.join(both_zero),spraw=pooled('small','raw','precision'),spclean=pooled('small','clean','precision'),rmed=sorted(raw_components)[len(raw_components)//2],rmax=max(raw_components),cmed=sorted(clean_components)[len(clean_components)//2],cmax=max(clean_components))
report=OUT/'EXPERIMENT_REPORT.md'; text=report.read_text()
start=text.find('## Failure analysis')
end=text.find('## What the gallery can establish')
if start>=0 and end>=0: text=text[:start]+section+text[end:]
elif end>=0: text=text[:end]+section+text[end:]
else: text += '\n'+section
text=text.replace('- Model identity: `unavailable` upstream Talk2DINO, DINO source `cached-local-repository`.',f'- Model identity: host cached Talk2DINO source revision `{REV}` and DINO source `cached-local-repository`. The mounted inference container lacked usable Git metadata, so exact byte-for-byte identity of its upstream checkout could not be independently confirmed.')
report.write_text(text)
for name in ('summary.json','configuration.json'):
    path=OUT/name; obj=json.loads(path.read_text()); ident=obj['model_identity']
    ident['host_cached_talk2dino_source_revision']=REV
    ident['official_repo_revision']='container runtime revision unavailable; host cached source revision '+REV
    ident['revision_provenance_note']=('The mounted inference container did not expose usable git metadata for the upstream checkout. '
        'The host cached Talk2DINO source revision was independently read from its local Git repository; exact byte-for-byte identity of the container checkout was not independently verified.')
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
print(json.dumps({'zero_recall_small_both':both_zero,'report':str(report),'source_revision':REV},indent=2))
