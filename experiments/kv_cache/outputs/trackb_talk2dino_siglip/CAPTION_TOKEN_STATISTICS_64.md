# COCO Caption Token Statistics — SigLIP Context 64

Tokenizer: checkpoint-associated OpenCLIP `HFTokenizer` using `T5TokenizerFast` and the configured `canonicalize` cleaner. Counts include special tokens and are measured before truncation. The context is 64; no context extension was applied. Percentiles use nearest-rank selection.

| Split | Captions | Within 64 | Over 64 | Over-context % | Mean | Median | P90 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train 2014 | 414,113 | 414,112 | 1 | 0.00024148% | 13.7789 | 13 | 18 | 24 | 65 |
| Validation 2014 | 202,654 | 202,654 | 0 | 0% | 13.7779 | 13 | 18 | 24 | 62 |

The one over-context training caption (65 tokens including special tokens) is:

> A doorway frames a telescopic view from a far off space with a multi-paned door, a chest of drawers and an oval rug on a wooden floor, up to a yellow room that has an oriental rug, chair, standing lamp, a widescreen T.V.,leather bench and ottoman.

The complete token sequence and machine-readable split statistics are in `caption_token_statistics_64.json`.

## Historical context-16 comparison

The earlier Stage 4A audit used a manually written text-cleaning path rather than this checkpoint tokenizer's `canonicalize`; it is therefore historical and not an apples-to-apples tokenizer comparison. That audit reported 62,586/414,113 training captions (15.1133%) and 30,789/202,654 validation captions (15.1929%) over 16 tokens. The current 64-token audit uses the actual SigLIP-384 checkpoint tokenizer/configuration and reports 1 training caption and 0 validation captions over context.

No caption text or tokenizer settings were changed to obtain these results. No feature extraction or training was started.
