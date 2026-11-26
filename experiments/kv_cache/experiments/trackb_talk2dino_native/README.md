# Native Talk2DINO Integration into OpenVLA

This is an optional, frozen segmentation sidecar. It does not modify the
standard OpenVLA action path. `model.py` contains the text-only OpenAI CLIP
ViT-B/16 tower, the official Talk2DINO ViT-L projection MLP, and adapters for
the official Meta DINOv2 model and OpenVLA's TIMM DINO model.

The integrated adapter passively captures the block-23 policy patch output,
block-24 token output, and final-block QKV from OpenVLA's existing DINO
forward. In installed TIMM 0.9.16, `get_intermediate_layers()` traverses all
24 blocks while returning the requested penultimate output, so block 24 is
already computed. The sidecar adds no second DINO forward or block execution.
OpenVLA's penultimate feature is pre-final-LayerNorm; the Talk2DINO final
patch feature applies the final LayerNorm.

## Reproduction

Run in the validated `trackb-v1-coco-object-20261126` container from the
isolated `thesis/trackb-dense-grounding-v2` worktree:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"
python -u experiments/kv_cache/experiments/trackb_talk2dino_native/run_libero_e2.py --smoke
python -u experiments/kv_cache/experiments/trackb_talk2dino_native/run_libero_e2.py --resume
```

The full evaluator replays the frozen E2 simulator states and saves completed
segments atomically under ignored cache storage. `--smoke` writes to a separate
cache subtree and cannot be mistaken for full sequences. The evaluator does
not apply actions to the environment. The one-time parity check invokes
`predict_action` on the same image without stepping the simulator.

Weights and demonstration data remain in local cache directories. Compact
summaries, reports, metrics and figures are written under
`experiments/kv_cache/outputs/trackb_talk2dino_native/`.
