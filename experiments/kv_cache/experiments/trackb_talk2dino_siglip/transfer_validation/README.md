# Track B V2 Transfer Validation

This package runs three sequential frozen-model studies: known-class prompt robustness on the existing COCO val2014 split, an audit of LIBERO paired segmentation data availability, and passive native OpenVLA DINO feature capture / overhead profiling. It does not train, generate actions, run policy rollouts, or edit KV-cache code.

## Reproducible environment

Use the existing `openvla-talk2dino-siglip:20261116` image and the `launch_transfer_validation.sh` remote launcher. The launcher refuses duplicate container names and validates checkpoints, feature caches, and the frozen validation manifest before starting. Its Docker container uses `on-failure:3`; output/status/logs are mounted under the new `transfer_validation_20261120` folder.

Monitor after connecting to the host:

```bash
docker ps --filter name=trackb-v2-transfer-validation-20261120
docker logs -f trackb-v2-transfer-validation-20261120
cat /home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/transfer_validation_20261120/RUN_STATUS.json
tail -f /home/oscar/projects/TrackingKVinOpenVLA/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/transfer_validation_20261120/logs/pipeline.log
```

The pipeline keeps per-image prompt metrics as JSONL and flushes each completed record, so prompt evaluation can resume after interruption. The LIBERO audit is CPU-only and records exact source/data availability. Integration timings are a short repeatable profile; if interrupted, rerunning the module recomputes the profile. The orchestrator uses a lock and skips stages with verified stage output artifacts.

## Local CPU checks

```bash
python -m py_compile experiments/kv_cache/experiments/trackb_talk2dino_siglip/transfer_validation/*.py
python -m pytest -q experiments/kv_cache/experiments/trackb_talk2dino_siglip/transfer_validation/test_prompt_ablation.py
```

## Inputs

- V2: frozen `best_projection.pt` from dense V2, expected epoch 40.
- V1: frozen full-training epoch-50 checkpoint, loaded from its `model` state dictionary with PyTorch's weights-only mode.
- DINO patch feature cache: standard OpenVLA penultimate-layer output; V1's separate final-normalized patch cache is used only for V1.
- Existing `validation_manifest.json`: 1,000 development images and 3,000 previously inspected secondary images; never rewritten.
- Local SigLIP-384 text checkpoint and COCO val2014 annotations/images.
