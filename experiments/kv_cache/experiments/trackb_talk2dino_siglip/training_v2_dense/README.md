# Track B Dense Grounding V2

This module trains a dense SigLIP-text to native OpenVLA DINOv2 patch projection. It does not alter OpenVLA policy inference and does not run robot rollouts.

The full pipeline is orchestrated by `orchestrate.py`. It audits the full caption sampler, prepares resumable feature and fractional-mask caches, trains only the projection MLP with atomic epoch checkpoints, evaluates on frozen disjoint COCO val2014 development/final manifests, and runs an offline visualization diagnostic on existing LIBERO RGB frames.

The exact runtime is the existing `openvla-talk2dino-siglip:20261116` container. Dataset and feature paths are arguments to the orchestrator; defaults target the NVIDIA Origin mount at `/opt/openvla`.

```bash
python -m experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.orchestrate
python -m experiments.kv_cache.experiments.trackb_talk2dino_siglip.training_v2_dense.orchestrate --resume
python -m pytest -q experiments/kv_cache/experiments/trackb_talk2dino_siglip/training_v2_dense/tests
```

For detached operation, run the orchestrator as the container's main process with Docker restart policy `unless-stopped`. The durable status is `RUN_STATUS.json`; logs are written both to container stdout and `logs/pipeline.log`. Large caches/checkpoints are under the ignored COCO cache tree.

Development and held-out IDs are selected from annotation metadata before predictions and saved in `validation_manifest.json`. The held-out set is evaluated once after development checkpoint selection. The LIBERO diagnostic uses previously saved RGB frames only; it has no simulator labels and makes no task-accuracy claim.
