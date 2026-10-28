# Track B — SigLIP vision/text alignment audit

## Evidence sources

This is a separate, offline diagnostic. It reuses the original Stage 4A saved affinity matrices and Track B native OpenVLA patch features. No policy inference, rollout, training, or mask selection occurred.

## Paired pretrained image/text path

The saved original SigLIP dual-encoder run used the checkpoint's paired visual tower, MAP image pooling, final text-token pooling, learned text projection, and cosine scoring. Across nine same-image task-description vs `red cup` comparisons, mean task-description advantage was 0.0633; `black bowl` mean advantage was -0.0045. This validates that the paired global path responds plausibly on average, not that local patches ground language.

## Fine-tuned OpenVLA visual tower

Only 27/342 visual tensors matched the paired pretrained weights after BF16 casting; aggregate relative L2 was 0.1413. The saved checkpoint’s OpenVLA native penultimate SigLIP features are therefore materially drifted from the original image/text pairing. A global image/text score for those fine-tuned patches is not directly comparable: the native path exposes penultimate patch states and does not apply the original trained MAP pooler/paired visual weights.

## Local patch affinity and token dominance

Mean broad-oracle instruction AUROC by saved patch variant: {"OpenVLA fine-tuned penultimate": 0.48466617318719213, "OpenVLA fine-tuned final": 0.3336362907141278, "paired pretrained final": 0.4481822665226519}. Same-image local maps for “black bowl” versus mismatched “red cup” remain correlated (means: {"OpenVLA fine-tuned penultimate": 0.7001095557566223, "OpenVLA fine-tuned final": 0.7628777028906325, "paired pretrained final": 0.7228061316491894}). These broad oracle metrics are descriptive; the simulator labels robot, objects, and fixtures, not phrase-specific object identity. Per-token winners and cosine extrema are in `text_token_dominance.csv`.

The valid-token matrix was rederived from saved features and the paired text tower. Including all 16 text positions, including EOS/padding, changed maps; mean valid-only/all-position correlation was 1.000. The corresponding mean broad-FG AUROC was 0.483 valid-only and 0.483 with special/padding positions included. The method correctly excludes special/padding states in its primary affinity.

## Penultimate/final features and temporal evidence

`local_feature_comparison.csv` compares the saved penultimate, final, and paired-pretrained final feature maps on the same images. Existing Stage 4A evidence found final-block OpenVLA features reduced the full-instruction broad-FG AUROC relative to penultimate features. Per-token maxima are available for two task-4/state-0 frames only; they are plotted in `task4_token_evidence_over_time.csv` and its figure as exploratory evidence, not evidence of task phase.

## Conclusion

The original paired global encoder works plausibly on average, while the local max-cosine method remains weak for patch grounding. The fine-tuned OpenVLA visual tower has substantial weight drift relative to the external text tower, supporting alignment drift as a contributor. Generic/action-token dominance and high matched/mismatched local-map correlation support a separate local-aggregation failure. Preprocessing and dimensions were validated in the original saved report; these data do not isolate all possible alignment causes. No new segmentation method is recommended from this audit alone.
