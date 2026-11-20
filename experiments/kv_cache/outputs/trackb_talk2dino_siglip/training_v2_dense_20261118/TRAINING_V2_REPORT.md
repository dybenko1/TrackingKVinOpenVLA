# Track B Training V2 — Dense Language-Grounded Segmentation

> **Evaluation audit correction (2026-11-20):** The legacy held-out metrics below are invalid and must not be cited. The evaluator reopened its per-image JSONL in write mode for every image, leaving only one image record per split despite processing the full split. The old V1 comparison also used the single V2 category prompt instead of V1's seven-template average. See the corrected, separate [V2 metric and prompt audit](evaluation_audit_20261120_corrected/AUDIT_REPORT.md) for recomputed results. Training/checkpoints were not changed.

Training status: **40/40 epochs complete**. The earlier legacy evaluation status and metrics are superseded by the audit.

## Method

Frozen SigLIP text features supervise a trainable projection against the unchanged OpenVLA penultimate DINOv2 patch features. Training uses COCO 2014 instance-mask fractional 16x16 occupancy: weighted BCE, soft Dice, top-16 hard-negative penalty, and patch log-mean-exp symmetric InfoNCE. The only trainable module is Linear(1152,1024) -> Tanh -> Linear(1024,1024). Development/final val2014 manifests contain 1,000/3,000 images and exclude previous Phase 3 and 2,500-image segmentation IDs.

## Training

- Completed epochs: 40 (maximum 40; early stopping after at least 12).
- Best development checkpoint epoch: 40.
- Best native-grid macro category mIoU: 0.2794143148723514.
- Development-selected threshold: 0.44999998807907104.
- Latest total objective: 0.8288049848084099.

## Development threshold calibration

{
  "v1_raw_pamr": 0.55,
  "v2_pamr": 0.15000000596046448,
  "v2_raw": 0.20000001788139343
}

## Legacy held-out evaluation — INVALID, retained for audit only

| Condition | 16x16 macro IoU | 224 macro IoU | Precision | Recall | FN rate | Background coverage proxy |
|---|---:|---:|---:|---:|---:|---:|
| v1_raw_pamr | 0.6008623499634042 | 0.6189502385821404 | 0.6066279735313643 | 0.9844284086703616 | 0.015571591329638406 | 0.57421875 |
| v2_pamr | None | None | 0.0 | 0.0 | 0.0 | 1.0 |
| v2_raw | None | None | 0.0 | 0.0 | 0.0 | 1.0 |

## Provenance and limitations

- V1 baseline reference commit: 1dcb8ade36980b0d4170583aea28840f09986ed1.
- V1 epoch-50 comparison uses final-layer DINO features; V2 uses OpenVLA standard penultimate DINO features. This compares the systems as a whole, not the loss in isolation.
- Native 16x16 labels are fractional occupancies; aligned 224 metrics compare lifted maps with exact transformed instance masks.
- COCO category localization does not establish LIBERO task relevance, cache safety, or policy success.
- Candidate background coverage is a segmentation proxy, not a KV-reuse safety metric or latency result.
Full status: RUN_STATUS.json. Durable worker log: logs/pipeline.log.

## Offline LIBERO feasibility diagnostic

{
  "status": "complete",
  "frame_count": 27,
  "tasks": [
    4,
    5,
    7
  ],
  "prompts": [
    "robot arm",
    "robot gripper",
    "table"
  ],
  "object_prompts_by_task": {
    "4": [
      "black bowl",
      "plate"
    ],
    "5": [
      "black bowl",
      "plate"
    ],
    "7": [
      "black bowl",
      "plate"
    ]
  },
  "mask_threshold": 0.20000001788139343,
  "no_oracle_labels": true,
  "no_policy_rollouts": true,
  "temporal_comparison_caveat": "These are sparse saved snapshots with large step gaps; mask overlap is descriptive, not frame-to-frame stability or ground-truth accuracy.",
  "mean_foreground_patches_by_prompt": {
    "black bowl": 9.296296296296296,
    "plate": 0.0,
    "robot arm": 10.25925925925926,
    "robot gripper": 39.629629629629626,
    "table": 6.2592592592592595
  },
  "sampled_pair_count": 90,
  "text_encoder": {
    "checkpoint_dir": "/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384",
    "checkpoint_repository_expected": "timm/ViT-SO400M-14-SigLIP-384",
    "checkpoint_revision": "ac16108d567c4389e6cd2b11c9b8585f7474435b",
    "weights_path": "/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384/open_clip_model.safetensors",
    "text_context_length": 64,
    "text_width": 1152,
    "text_output_dim": 1152,
    "text_layers": 27,
    "text_heads": 16,
    "text_positional_embedding_shape": [
      64,
      1152
    ],
    "text_state_tensor_count": 330,
    "strict_text_state_load": true,
    "missing_text_keys": [],
    "unexpected_text_keys": [],
    "open_clip_version": "2.32.0",
    "torch_version": "2.6.0",
    "tokenizer": {
      "checkpoint_dir": "/opt/openvla/cache/models/timm/ViT-SO400M-14-SigLIP-384",
      "tokenizer_class": "T5TokenizerFast",
      "clean": "canonicalize",
      "canonicalization": "OpenCLIP HFTokenizer.clean_fn from checkpoint configuration",
      "context_length": 64,
      "pad_token_id": 1,
      "eos_token_id": 1
    },
    "visual_tower_loaded": false,
    "frozen": true
  },
  "checkpoint": "/opt/openvla/cache/worktrees/trackb-dense-grounding-v2/experiments/kv_cache/outputs/trackb_talk2dino_siglip/training_v2_dense_20261118/best_projection.pt"
}
