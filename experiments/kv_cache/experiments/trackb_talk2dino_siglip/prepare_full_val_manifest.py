#!/usr/bin/env python3
"""Create deterministic disjoint full-val retrieval and segmentation manifests."""
import argparse, hashlib, json, random
from collections import defaultdict
from pathlib import Path

from experiments.kv_cache.experiments.trackb_talk2dino_siglip import phase3_evaluate as p3

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-root',type=Path,default=Path('/opt/openvla/cache/datasets/talk2dino/coco2014'))
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=20261117)
    ap.add_argument('--development-fraction',type=float,default=.2)
    ap.add_argument('--segmentation-development',type=int,default=500)
    ap.add_argument('--segmentation-evaluation',type=int,default=2000)
    ap.add_argument('--exclude-manifest',type=Path,default=Path('/opt/openvla/cache/worktrees/trackb-talk2dino-siglip/experiments/kv_cache/outputs/trackb_talk2dino_siglip/full_coco_training_20261117/spatial_evaluation/evaluation_manifest.json'))
    a=ap.parse_args()
    p3.DATA=a.dataset_root
    ann_path=a.dataset_root/'annotations2014/captions_val2014.json'
    coco=json.loads(ann_path.read_text())
    ids=sorted(int(x['id']) for x in coco['images'])
    if len(ids)!=40504 or len(coco['annotations'])!=202654:
        raise RuntimeError(f'COCO val2014 counts differ from expected: images={len(ids)}, captions={len(coco["annotations"])}')
    rng=random.Random(a.seed); shuffled=ids.copy(); rng.shuffle(shuffled)
    ndev=round(len(ids)*a.development_fraction)
    dev=set(shuffled[:ndev]); evaluation=set(shuffled[ndev:])
    if dev & evaluation or len(dev)+len(evaluation)!=len(ids): raise AssertionError('Split is not a disjoint partition')
    by_id={int(x['id']):x for x in coco['images']}
    captions_by_image=defaultdict(int)
    for x in coco['annotations']: captions_by_image[int(x['image_id'])]+=1
    retrieval={
      'dataset':'COCO Captions 2014 val2014','seed':a.seed,'split_unit':'image_id; all captions for an image stay in its split',
      'image_count':len(ids),'caption_count':len(coco['annotations']),
      'development':{'image_ids':sorted(dev),'image_count':len(dev),'caption_count':sum(captions_by_image[i] for i in dev)},
      'evaluation':{'image_ids':sorted(evaluation),'image_count':len(evaluation),'caption_count':sum(captions_by_image[i] for i in evaluation)},
      'annotation_sha256':hashlib.sha256(ann_path.read_bytes()).hexdigest(),
    }
    images,_,cats=p3.read_coco()
    image_by_id={int(x['id']):x for x in images['images']}
    anns=defaultdict(list)
    for x in images['annotations']: anns[int(x['image_id'])].append(x)
    valid={'development':[],'evaluation':[]}
    excluded=set()
    if a.exclude_manifest.is_file():
      old=json.loads(a.exclude_manifest.read_text())
      excluded={int(x['image_id']) for x in old.get('images',[])}
    for subset, pool, count in [('development',dev,a.segmentation_development),('evaluation',evaluation,a.segmentation_evaluation)]:
      candidates=[]
      order=sorted(pool-excluded); random.Random(a.seed+(0 if subset=='development' else 1)).shuffle(order)
      for iid in order:
        meta=image_by_id.get(iid)
        if not meta or not anns[iid] or not (a.dataset_root/'val2014'/meta['file_name']).is_file(): continue
        gids,g224,g16,sizes=p3.make_gt(meta,anns,cats)
        visible=[cid for cid in gids if sizes[cid]>0]
        if visible:
          candidates.append({'image_id':iid,'file_name':meta['file_name'],'width':meta['width'],'height':meta['height'],
            'category_ids':visible,'category_names':[cats[c]['name'] for c in visible]})
        if len(candidates)>=count: break
      chosen=candidates
      if len(chosen)<count: raise RuntimeError(f'Need {count} {subset} segmentation images; found {len(chosen)}')
      valid[subset]=chosen
    dev_seg_ids={x['image_id'] for x in valid['development']}; eval_seg_ids={x['image_id'] for x in valid['evaluation']}
    if dev_seg_ids & eval_seg_ids: raise AssertionError('Segmentation dev/evaluation overlap')
    manifest={'selection':{'dataset':'COCO val2014 instance annotations','seed':a.seed,
      'development_count':len(valid['development']),'evaluation_count':len(valid['evaluation']),
      'criterion':'image has at least one category whose union mask occupies >=1 nearest-neighbor 16x16 patch cell; no model prediction used',
      'selection_used_model_predictions':False,'overlap_count':0,
      'prior_80_image_evaluation_had_been_examined':True,'prior_phase3_manifest_excluded_ids':len(excluded),
      'excluded_phase3_manifest_sha256':hashlib.sha256(a.exclude_manifest.read_bytes()).hexdigest() if a.exclude_manifest.is_file() else None},
      'images':[dict(x,subset=subset,feature_row=i) for subset in ('development','evaluation') for i,x in enumerate(valid[subset])],
      'annotation_sha256':retrieval['annotation_sha256']}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'validation_split_manifest.json').write_text(json.dumps(retrieval,indent=2)+'\n')
    (a.output_dir/'segmentation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'images':len(ids),'captions':len(coco['annotations']),'retrieval_dev':len(dev),'retrieval_eval':len(evaluation),
      'retrieval_dev_captions':retrieval['development']['caption_count'],'retrieval_eval_captions':retrieval['evaluation']['caption_count'],
      'segmentation_dev':len(valid['development']),'segmentation_eval':len(valid['evaluation']),'overlap':0},indent=2))
if __name__=='__main__': main()
