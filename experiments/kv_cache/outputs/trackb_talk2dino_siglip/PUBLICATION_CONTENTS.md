# Talk2DINO-SigLIP published artifact scope

This directory publishes the implementation and human-readable reports, validation summaries, training metrics, sanitized sample-ID manifest, and qualitative PNG panels. Phase 2 publishes its concise execution log and the sampler test log for reproducibility.

The local pilot outputs also contain raw/large artifacts intentionally not tracked: COCO-derived feature arrays (`*.npy`), the full caption-bearing sample manifest, and intermediate or trained projection checkpoints (`*.pt`). The Phase 2 corrected checkpoint remains local at `pilot_1000_phase2/corrected_projection.pt`; its size and SHA-256 are recorded in the run metadata. Full extraction/worker logs also remain local. The COCO dataset and pretrained model weights are not included.
