# Stage 4A Phase 1B — UNMATCH-inspired SigLIP patch–text matching

## Executive result

The local affinity computation is technically runnable with the verified paired SigLIP checkpoint, and the checkpoint's ordinary pooled image–text path behaves plausibly for complete task descriptions. However, the proposed token-to-patch max-cosine maps did **not** provide dependable task-object localization. They were diffuse, changed little between matched and deliberately mismatched phrases, and were often dominated by generic/action words. Final-block features were not an improvement over OpenVLA's native penultimate features on the broad post-hoc foreground diagnostic.

The key compatibility finding is that OpenVLA's fine-tuned SigLIP visual tower is materially different from the paired pretrained visual tower. Continuing OpenVLA's visual states through its own final block is exact and inexpensive, but those states no longer share the original paired visual weights used to train the downloaded text tower. No mask threshold, segmentation rule, KV reuse, policy inference, or LIBERO rollout was tested.

**Recommendation: Revise matching mechanism.** Do not advance this raw max-cosine map to Stage 4A foreground mask selection.

## Method and relation to UNMATCH

UNMATCH Sections 3.2–3.4 describe frozen SigLIP feature extraction, a local text-token × image-patch cosine-affinity matrix, both directional max reductions, multiscale coverage/statistic features, and a trained global/local prediction classifier. This diagnostic implements only the frozen encoders and local affinity/reductions; it does not reproduce the trained classifier, objective, or multiscale aggregation. See the [UNMATCH paper](https://arxiv.org/abs/2609.31766).

For each prompt and feature variant, final text-token states and 256 visual-patch states were L2-normalized in FP32 and multiplied to give `S: [valid_text_tokens, 256]`. `text→visual` is the maximum over patch positions for each text token; `visual→text` is the maximum over retained text tokens for each patch. Both argmax correspondences and the un-reduced matrix are retained. Scores are cosine affinities, **not probabilities**.

The valid text positions exclude all tokenizer special IDs, including EOS and repeated EOS padding. The visual sequence is row-major on the validated 16×16 grid. Oracle labels are used only for post-hoc broad foreground/background summaries and visualization, never to fit or choose a foreground mask.

## Checkpoint and feature compatibility

The paired checkpoint was loaded strictly from:

- Repository/revision: `timm/ViT-SO400M-14-SigLIP`, `c85ba2402be4b49b5d178f2e2384fdf50f8c7961`.
- File: `cache/models/timm/ViT-SO400M-14-SigLIP/c85ba2402be4b49b5d178f2e2384fdf50f8c7961/open_clip_model.safetensors`.
- SHA-256: `0a04a48b797e187a568335ab67e57a8958c32cee5a707a3feb6d4c97149dcd9c`.
- Paired config: SO400M/14, 224×224, 1,152-wide image/text features, 27 visual and text blocks, 256 patches, learned MAP image pooling, no extra visual projection, and text context length 16.
- Tokenizer: checkpoint-local `T5TokenizerFast`; canonicalized lowercase/no punctuation; pad ID set to EOS ID 1 to match OpenCLIP's HF-tokenizer convention.

The paired model's visual trunk state has 342 entries, all mapping by name and shape to the Timm SO400M architecture. **Matching names/shapes did not imply matching weights.** Direct comparison against the visual weights in OpenVLA fine-tuned snapshot `962318cec55ac10993ff0f5f43eda9a270b4c873` found:

- only 27/342 tensors equal after BF16 casting;
- aggregate relative L2 difference `0.1413`, max absolute difference `0.9542`;
- relative L2 was `0.1975` in patch embedding and `0.1594` in transformer blocks.

This indicates that fine-tuning changed the visual tower. The exact OpenVLA fine-tuned block-25 features recomputed from the checkpoint matched all nine saved Track B native feature tensors exactly after their saved FP16 representation (`max abs = 0`, relative L2 = 0). Thus the tests below use actual OpenVLA visual features rather than accidentally substituting the paired pretrained visual tower.

Feature variants:

| Variant | Visual representation | Purpose |
|---|---|---|
| A | Actual fine-tuned OpenVLA block 25/27, no final norm; native 256×1,152 feature | Existing OpenVLA path |
| B | Continue actual OpenVLA block-25 hidden states through block 26 and `trunk.norm` | Test final-layer patch features without recomputing the vision encoder |
| Paired-final reference | Final block + norm from the paired pretrained visual tower | Matched original SigLIP text/vision checkpoint reference; not the OpenVLA visual path |

For all nine images, continuation through block 26 plus norm exactly matched Timm's direct final-layer extraction (`max abs = 0`). This proves continuation is structurally feasible using the *same OpenVLA weights*. It does not restore the original pretrained image/text pairing.

The ordinary paired pooled image/text route was also exercised: the checkpoint's MAP image pooler, last-token text pooling and text projection produced the same pooled text result as the direct token-state path (max difference 0). Complete task instructions scored above a mismatched `red cup` on average (`+0.0633` cosine), and `plate` exceeded `red cup` (`+0.0354`). But `black bowl` did not (`−0.0045`). This is a useful global-pairing sanity check, not evidence that patch maxima ground the words.

## Diagnostic set and results

Nine existing Track B observations were used: three time points each from task 4, task 5 and task 7, all state 0. Each image was evaluated with object phrases, individual words, generic/action-word controls, and its task instruction. There was no training or threshold selection.

Mean broad oracle-FG ranking AUC (nine frames; AUC is only a coarse post-hoc diagnostic, not target-object IoU):

| Text prompt | OpenVLA penultimate | OpenVLA final | Paired pretrained final |
|---|---:|---:|---:|
| Full task instruction | 0.485 | 0.334 | 0.448 |
| “black bowl” | 0.687 | 0.472 | 0.370 |
| “black” | 0.773 | 0.695 | 0.454 |
| “bowl” | 0.804 | 0.724 | 0.595 |
| “plate” | 0.547 | 0.541 | 0.528 |
| Mismatched “red cup” | 0.704 | 0.611 | 0.478 |

The oracle label marks all task-relevant foreground regions, not the specific named object in a phrase. These AUCs must not be read as object grounding quality. In particular, a mismatched phrase's broad-FG AUC can be high because the mask contains the robot, multiple objects, and fixtures.

Prompt changes altered maps somewhat but did not produce strong object-specific shifts: the mean correlation between the “black bowl” and mismatched “red cup” patch-score maps was `0.70` for penultimate, `0.76` for OpenVLA final, and `0.72` for paired-pretrained final features. On task 4/state 0/step 0, both `black bowl` and `red cup` had their highest final-feature score at grid cell `(row 15, column 4)`, near the bottom image edge rather than the visible bowl. In task 5 and task 7 full-instruction maps, the pronoun token “it” was the winning text token for 238/256 and 216/256 patches, respectively. For task 4, “place” won on 146/256 final-feature patches. These are concrete generic/action-word failure modes of max-over-text-token aggregation.

Final-block features did not improve the broad instruction diagnostic: mean task-instruction AUC decreased from `0.485` to `0.334`; the matched “black bowl” phrase decreased from `0.687` to `0.472`. The paired-pretrained final reference also remained weak on full instructions (`0.448`). Individual words such as `bowl` can rank broad foreground above background in some frames, but the same is not true of phrases, instructions, or mismatched controls, and the oracle labels are too broad to validate exact object localization.

### Context-length limitation

The actual encoder context is 16 positions. Task 4 tokenizes to 20 positions including EOS and is truncated before “on the plate”; task 5 tokenizes to 18 and truncates before “plate” (and “ramekin” splits into subword pieces); task 7 fits in 15 positions including EOS. The truncated task-4/task-5 results are not equivalent to the complete instructions and are treated as a limitation, not silently interpreted as full-prompt matching.

## Cost and feasibility

Measurements used the NVIDIA-enabled `openvla-cached:latest` image, Orin, CUDA 12.8, PyTorch 2.6.0, Timm 0.9.16, CUDA events with synchronization, and fixed inputs. GPU timing used warm-up and repeated samples. Values below are diagnostic and exclude Llama prefill/action inference.

| Operation | Median |
|---|---:|
| OpenVLA penultimate SigLIP feature path | 24.56 ms |
| Continue block 26 + final norm | 1.03 ms |
| Local affinity/reductions, 2-token phrase | 0.676 ms |
| Local affinity/reductions, longest task instruction | 0.684 ms |
| Paired model original pooled image encoder | 25.73 ms |
| Text token-state encoding (once per instruction) | 24.55 ms |
| CPU tokenization (once per instruction) | 7.01 ms |

The local patch-score path can reuse OpenVLA's already-computed block-25 output; its incremental image-side work is about 1.03 ms for block 26/norm plus about 0.68 ms for scoring on this diagnostic. Text encoding is roughly 24.6 ms but is once per instruction and can be cached. Calling the paired model's complete original image encoder would add about 25.7 ms because its pretrained visual weights differ from the fine-tuned OpenVLA tower; that is not an attractive substitute for continuing existing features.

The diagnostic process allocated about 2.65 GB total while loading the full paired model plus an additional OpenVLA visual trunk. The BF16 text-tower parameters alone are estimated at about 899 MB; actual incremental production memory would depend on loading only the text tower and its activations. Peak temporary allocation for the local affinity operation was about 2.37 MB. These are not end-to-end latency or memory measurements for a deployed OpenVLA policy, and no speedup is claimed.

## Conclusion

1. The exact paired text encoder and original pooled image/text scoring are available and load cleanly without training.
2. Existing OpenVLA penultimate patch features are reproducible, but their fine-tuned visual weights differ materially from the paired pretrained vision tower.
3. Continuing through the actual OpenVLA final block and norm is possible without recomputing earlier blocks, but final features reduced the measured coarse alignment rather than improving it.
4. Local cosine maxima were finite, nonuniform, and prompt-sensitive in some locations, but were not sufficiently selective: matched and mismatched maps were highly correlated, bottom-edge maxima appeared for both, and generic/action words often won most patches.
5. Therefore, these results do not justify constructing a foreground mask from this UNMATCH-inspired max-cosine mechanism. A revised mechanism would need to address fine-tuned image/text mismatch and generic-token dominance before any mask-selection or KV-reuse test.

## Reproducibility and files

Run inside the existing CUDA-enabled container from the repository root:

```bash
docker run --rm --runtime nvidia \
  -e PYTHONPATH=/tmp/stage4a_openclip_pkg \
  -v /home/oscar/projects/TrackingKVinOpenVLA/openvla:/opt/openvla \
  -v /tmp:/tmp -w /opt/openvla openvla-cached:latest \
  python -u experiments/kv_cache/experiments/stage4a_unmatch_siglip_diagnostics.py
```

`open_clip_torch==2.31.0` was installed only into `/tmp/stage4a_openclip_pkg` with `--no-deps`; the repository/runtime dependencies were not changed. The installed image lacked `ftfy`, so the diagnostic stubs the unused cleaner only for importing OpenCLIP; the actual local checkpoint T5 tokenizer is used directly. The scoring output does not call that stub.

- `matching_diagnostics.json`: full sample/prompt/variant measurements, tokenization, score vectors, correspondences, pooled scores, validation and timing/memory metadata.
- `configuration.json`: model revisions, prompt set, sample IDs, feature variants and run scope.
- `timing_memory.json`: timing and memory summary.
- `affinity_matrices/*.npz`: all 405 raw matrices, including text-token × 256-patch values for all three visual variants; per-token instruction matrices are separate `*_per_token.npz` files.
- `unmatch_figures/`: matched/mismatched heatmaps, per-token instruction maps, and variant summary plot. Oracle masks in these diagnostic figures are post-hoc only.
- `experiments/kv_cache/experiments/stage4a_unmatch_siglip_diagnostics.py`: executable diagnostic harness.

No Stage 3 files or results were modified. No model weights were added to Git. No actions, rollouts, KV reuse, or segmentation masks were produced.
