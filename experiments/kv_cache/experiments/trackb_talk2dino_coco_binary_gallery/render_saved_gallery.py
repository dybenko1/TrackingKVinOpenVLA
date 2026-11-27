#!/usr/bin/env python3
"""Regenerate gallery panels and provenance text from saved scores only.

No DINO, CLIP, MLP or COCO inference is performed. Intended for the validated
container where the previously generated local prediction cache is mounted.
"""
from __future__ import annotations
import ast, csv, importlib.util, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image

WORKTREE = Path('/opt/openvla/cache/worktrees/trackb-dense-grounding-v2')
UPSTREAM = Path('/opt/openvla/cache/references/Talk2DINO')
OUT = WORKTREE / 'experiments/kv_cache/outputs/trackb_talk2dino_coco_binary_gallery'
SRC = WORKTREE / 'experiments/kv_cache/experiments/trackb_talk2dino_coco_binary_gallery/run_gallery.py'

def load_module():
    spec = importlib.util.spec_from_file_location('gallery_source', SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def upstream_class_names():
    path = UPSTREAM / 'src/open_vocabulary_segmentation/segmentation/datasets/coco_object.py'
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'COCOObjectDataset':
            for item in node.body:
                if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CLASSES' for t in item.targets):
                    return ast.literal_eval(item.value)
    raise RuntimeError('Upstream COCO Object class list not found')

def label_lut():
    path = UPSTREAM / 'src/open_vocabulary_segmentation/convert_dataset/convert_coco_object.py'
    tree = ast.parse(path.read_text())
    mapping = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'clsID_to_trID' for t in n.targets))
    lut = np.zeros(256, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[int(src)] = 255 if int(src) == 255 else (0 if int(src) > 90 else int(dst) + 1)
    return lut

def numeric(row, prefix):
    return {k: float(row[prefix + k]) for k in ('precision','recall','iou')} | {
        'foreground_coverage': float(row[prefix+'foreground_coverage']),
        'background_coverage': float(row[prefix+'background_coverage'])}

def main():
    mod = load_module()
    manifest = json.loads((OUT/'gallery_manifest.json').read_text())
    assert len(manifest['selected']) == 30
    assert len({x['image_id'] for x in manifest['selected']}) == 30
    class_names = upstream_class_names(); lut = label_lut()
    assert len(class_names) == 81
    rows = {}
    with (OUT/'per_image_metrics.csv').open() as f:
        for row in csv.DictReader(f): rows.setdefault(row['image_id'], {})[row['method']] = row
    rendered = []
    for item in manifest['selected']:
        iid = item['image_id']; rgb = np.asarray(Image.open(item['image_path']).convert('RGB'))
        raw_labels = np.asarray(Image.open(item['gt_path']), dtype=np.uint8)
        gt = lut[raw_labels] == int(item['class_id'])
        if int(gt.sum()) != int(item['semantic_union_area_px']):
            raise RuntimeError(f'Ground-truth mismatch for {iid}')
        with np.load(OUT/'predictions_local'/f'{iid}.npz') as z:
            if str(z['query'].item()) != item['category']:
                raise RuntimeError(f'Saved score cache prompt mismatch for {iid}')
            maps = {'raw': {'pixel': z['raw448'].astype(np.float32)},
                    'clean': {'pixel': z['clean448'].astype(np.float32)}}
        metrics = {}
        for method in ('raw','clean'):
            row = rows[iid][method]
            metrics[method] = {'pixel': numeric(row,'pixel_'), 'patch_binary': numeric(row,'patch_binary_')}
        target = OUT/'figures'/item['figure']
        mod.render_panel(item, rgb, gt, maps, metrics, target)
        rendered.append(target.name)
    # Recover the exact cached host/upstream commit without mutating global git config.
    cmd = ['git','-c',f'safe.directory={UPSTREAM}','-C',str(UPSTREAM),'rev-parse','HEAD']
    revision = subprocess.check_output(cmd, text=True).strip()
    for name in ('summary.json','configuration.json'):
        path = OUT/name; data = json.loads(path.read_text()); ident = data['model_identity']
        ident['host_cached_talk2dino_source_revision'] = revision
        ident['official_repo_revision'] = 'runtime checkout git metadata unavailable; host cached source is ' + revision
        ident['revision_provenance_note'] = ('The mounted inference container did not expose usable git metadata for the upstream checkout. '
            'The host cached Talk2DINO source revision was independently read from its local Git repository; exact byte-for-byte identity of the container checkout was not independently verified.')
        path.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    report = OUT/'EXPERIMENT_REPORT.md'; text = report.read_text()
    text = text.replace('`not embedded in runtime checkout; host cached source revision 6dac47dbe3500df63f3b3df20dd28334290919cc` upstream Talk2DINO (the container checkout lacked accessible git metadata, so runtime revision could not be independently confirmed)',
                        f'host cached source revision `{revision}` upstream Talk2DINO (the container checkout lacked usable Git metadata, so exact runtime source identity could not be independently confirmed)')
    report.write_text(text)
    print(json.dumps({'rendered':len(rendered),'revision':revision,'output':str(OUT)},indent=2))

if __name__ == '__main__': main()
