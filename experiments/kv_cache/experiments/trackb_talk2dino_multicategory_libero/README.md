# Original Talk2DINO multi-category LIBERO experiment

This standalone experiment evaluates the original Talk2DINO ViT-L model on the
90 selected cached E2 LIBERO observations. It uses the official Meta DINOv2
ViT-L/14-register backbone, OpenAI CLIP ViT-B/16 text tower, upstream prompt
ensemble, original projection MLP, original category-specific attention
cleaning (`lambda_bg=0.2`), and native 16×16 patch maps. It does not run OpenVLA,
train weights, use PAMR, or launch policy rollouts.

The source is `run_experiment.py`. The run requires the existing
`trackb-v1-coco-object-20261126` container and local E2 archives/checkpoints.

```bash
# One-image CUDA smoke validation
docker exec trackb-v1-coco-object-20261126 sh -lc \
  'cd /opt/openvla/cache/worktrees/trackb-dense-grounding-v2 && \
   PYTHONPATH=/opt/openvla/cache/worktrees/trackb-dense-grounding-v2:/opt/openvla \
   python experiments/kv_cache/experiments/trackb_talk2dino_multicategory_libero/run_experiment.py --smoke'

# Full 90-image run; saves score maps locally under ignored output storage
docker exec trackb-v1-coco-object-20261126 sh -lc \
  'cd /opt/openvla/cache/worktrees/trackb-dense-grounding-v2 && \
   PYTHONPATH=/opt/openvla/cache/worktrees/trackb-dense-grounding-v2:/opt/openvla \
   python experiments/kv_cache/experiments/trackb_talk2dino_multicategory_libero/run_experiment.py'

# Rebuild summary/report/figures from saved score maps; performs no model inference
docker exec trackb-v1-coco-object-20261126 sh -lc \
  'cd /opt/openvla/cache/worktrees/trackb-dense-grounding-v2 && \
   PYTHONPATH=/opt/openvla/cache/worktrees/trackb-dense-grounding-v2:/opt/openvla \
   python experiments/kv_cache/experiments/trackb_talk2dino_multicategory_libero/run_experiment.py --report-only'
```

Compact, reproducible results are published in
`outputs/trackb_talk2dino_multicategory_libero/`. The larger per-frame score
arrays are intentionally kept in ignored local storage and are not committed.
The published report discloses that the original per-image latency distribution
was not retained; it reports the preserved mean and leaves unavailable
percentiles unset.
