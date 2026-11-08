# Long-memory screen re-audit R1

Status: read-only review of completed historical runs and conditional recipes.
No model inference, fit, profile, source adapter, or L4 admission occurred in
this review. The official validation curves below were inspected and are
exposed development evidence.

## Evidence inspected

The six completed 12-epoch runs are in
`C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard/native/`:

- GRU controls: `l4-gru_s101001_monitor_opt_r3-fit` and
  `l4-gru_s101002_monitor_opt_r3-fit`.
- Existing GRU-local: `l4-gru_local_s101001-fit` and
  `l4-gru_local_s101002-fit`.
- Standalone Mamba2: `l4-mamba2_s101001-fit` and
  `l4-mamba2_s101002-fit`.

Each has twelve `monitor_evaluations.jsonl` rows for metric
`global_wp_clip2_v1`, full official valid membership (1,873 sequences and
3,528,273 scored rows), split SHA256
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`, and the
same validation `is_scored` mask after warmup row 99. This is an exposed
chronological development set. The 32-member `train_eval` metric is diagnostic
only and has a different population and mask; it is useful here only to show
the direction of fitting-cohort versus valid curves, not to calibrate their
gap.

The frozen per-run configs are in the read-only sibling worktree
`C:/Users/admin/Documents/Codex/reviews/WunderFunder-architecture-execution-20261107/research/experiment_drafts/architecture-execution-20261107/l4/configs-runtime-r3/`.
Both seed-matched GRU controls use source-tree SHA256
`5bbbf68eccdc8c8773ff4b4eda2c4026d9c3f31b145dfc0d683b17d20a1987d0`; the
GRU-local and Mamba2 runs use
`3ea26eb2cc9beefd7d07e9d2216c937cc6f78f6ee99719272261ce47e15f2f2a`.
All six used FP32, raw112 features, 10,325 all-fit normalization sequences,
12 epochs, batch 16, chunk 2,000, accumulation 1, AdamW LR 3e-4 / WD .01,
gradient clip 1, 5% warmup and warmup-cosine schedule. The logged eval loss
definition is weighted MSE against `clip(target, -2, 2)`, weighted by the
absolute clipped target. They used no KD or hint loss. Model sizes and
family-specific settings differ:

| Family | Parameters | Architecture details | Old config file SHA256 (seed 101001 / 101002) |
| --- | ---: | --- | --- |
| GRU | 950,274 | 256 hidden, 2 layers | `f486001a8c7963d2986d55334c6ddb02e498441bdfbdda05cb256e40033a667a` / `d1819ba510c7dd4fe57df590819f801ec9d4a5f82b45f0bdbe4dd647d488ad25` |
| GRU-local | 1,362,532 | GRU 256×2; RoPE local attention 128×2, 4 heads, context 256; `ff_multiplier=4` | `1316ad83ff26a30e6e8aa2c1236f2a6c425022be8ed698a3155a35d9b1e226dc` / `903f5747569515adbefbbe6b068b050db8a0301974f99d60d0b87bea17465138` |
| Mamba2 | 647,378 | hidden 128, 6 layers, state size 16, conv width 4, expansion 2, chunk 256 | `2b1f387c40cb415d9b2aeb93de0cd0a785b6f92d4251dd267c496216669c3aba` / `725b85d594dc7e6412200c709cb15d20115304221715004de42d0986fcb2b8f3` |

## Full official-valid WP curves

Values are the recorded epoch-end `valid/wp` values, E1 through E12. No best
epoch was chosen as the fixed endpoint; the peak column below is descriptive
post-hoc inspection of this already-exposed set.

| Run | E1 → E12 `global_wp_clip2_v1` | Observed peak |
| --- | --- | ---: |
| GRU s101001 | .668766973, .672372791, .673331437, .681742634, .681151062, .681927917, .681033001, .676334040, .672140679, .666474702, .663898731, .661724660 | E6 .681927917 |
| GRU s101002 | .665198491, .669865674, .680840731, .680254749, .682716348, .682530034, .681464775, .677598001, .672949765, .669899853, .665976368, .664740030 | E5 .682716348 |
| GRU-local s101001 | .662056495, .671065719, .670842262, .680922567, .680395455, .682724438, .679843523, .675960741, .668073611, .659878243, .658879421, .655230562 | E6 .682724438 |
| GRU-local s101002 | .659658596, .662347040, .677674054, .677813607, .681294481, .680936079, .679676202, .676417254, .667245489, .665564123, .657667048, .656087633 | E5 .681294481 |
| Mamba2 s101001 | .666331649, .665679699, .672900276, .679929490, .678732041, .680357721, .678791075, .675074350, .670523866, .660723374, .654887576, .651481886 | E6 .680357721 |
| Mamba2 s101002 | .655202963, .665713636, .677032523, .677381555, .680766669, .682010790, .681403615, .678717178, .672800308, .667586225, .662449871, .660485050 | E6 .682010790 |

The E6 versus E12 fitting-cohort/valid pairs show a consistent divergence in
all six runs:

| Run | E6 train-eval / valid WP | E12 train-eval / valid WP |
| --- | ---: | ---: |
| GRU s101001 | .492231022 / .681927917 | .551221263 / .661724660 |
| GRU s101002 | .490317866 / .682530034 | .541164104 / .664740030 |
| GRU-local s101001 | .495306442 / .682724438 | .561764253 / .655230562 |
| GRU-local s101002 | .490125891 / .680936079 | .562371652 / .656087633 |
| Mamba2 s101001 | .495234244 / .680357721 | .582117997 / .651481886 |
| Mamba2 s101002 | .482416538 / .682010790 | .566729579 / .660485050 |

Across the two matched seeds, mean valid WP changes from E6 to E12 by
−.018997 for GRU, −.026171 for GRU-local, and −.025201 for Mamba2. At E6,
GRU-local minus GRU is +.000797 for seed 101001 and −.001594 for seed 101002,
mean −.000399. At E12 the paired deltas are −.006494 and −.008652, mean
−.007573. Mamba2 minus GRU at E6 is −.001570 and −.000519, mean −.001045;
at E12 it is −.010243 and −.004255, mean −.007249. With two seeds, these are
descriptive contrasts, not reliable estimates of training variance or tests of
significance.

## What the historical runs do and do not establish

The existing `gru_local` implementation is
`src/wnn/training/sequence_gru_local.py`: a 256-wide two-layer GRU state path,
a 128-wide two-layer RoPE attention path with its own causal key/value cache,
a fusion gate, and a direct linear head over the current standardized raw112
features. `context_rows=256` describes the attention cache (current row plus up
to 255 previous rows); it does not cap the recurrent branch. The standalone
`mamba2` implementation is `src/wnn/training/sequence_mamba2.py`, with a
128-wide, six-layer selective state-space/convolution cache; it has no local
attention branch. The proposed interaction is the combination of these memory
mechanisms and the existing direct-head idea, not a relabeling of standalone
Mamba2 or GRU-local.

`gru_local` is already an evaluated architecture with two completed 12-epoch
fits. Its E6 checkpoint scores do not make a new 6-epoch run redundant in the
strict recipe sense: E6 is halfway through the old 12-epoch cosine schedule,
whereas a newly frozen 6-epoch schedule reaches its minimum LR at its own
endpoint. However, neither seed shows a consistent GRU-local advantage at E6,
and the old-to-new recipe delta is more than schedule length. A shorter
schedule is justified as a general overfit-control question, but the existing
evidence does not justify presenting GRU-local as a novel representation or
promising quality candidate.

The four local configs currently in this packet are therefore **conditional
short-schedule re-test recipes only**:

| Arm | Configs | Fixed seed pair | Model parameters |
| --- | --- | --- | ---: |
| GRU control | `configs/l4_long_memory_gru_control_s101103.json`, `...s101104.json` | 101103, 101104 | 950,274 |
| Existing GRU-local package | `configs/l4_long_memory_gru_local_s101103.json`, `...s101104.json` | 101103, 101104 | 1,165,156 |

They clone the current `raw112_d0` six-epoch baseline: B32, chunk 1,000,
accumulation 1, AdamW 3e-4 / WD .01, clip 1, 5% warmup/cosine, FP32, all-fit
normalization, label-only WMSE v2 and fixed E6 endpoint. The local branch uses
`local_hidden=128`, two layers, four heads, context 256, and
`ff_multiplier=1`; it is smaller than the old 12E local config's
`ff_multiplier=4`. Thus these configs test a changed short-schedule package,
not only a shorter schedule. The seeds are provisional and must be reconciled
against active work before any admission. No fits are authorized by these
files.

## Conditional new mechanism recommendation

If the next 4-fit screen is approved, the more distinct hypothesis is a new
**Mamba2 memory + local-attention + direct-head hybrid**, compared with a
freshly trained Mamba2 control. Standalone Mamba2 and GRU-local are already
tested; the combined architecture has not been tested and must not be
misreported as either existing family. The package hypothesis is that
Mamba2's carried state and a short causal local-attention path provide
complementary long and recent context, while the direct raw-feature head
preserves an easy path for current-row signal. A learned fusion makes this a
package test; it cannot isolate each component.

Suggested conditional panel: two new paired seeds × (Mamba2 control, hybrid),
four new fits total. Use identical raw112 data, official fitting membership,
all-fit normalization, label-only WMSE, batch 32, chunk/TBPTT 1,000, accumulation
1, the same optimizer/schedule, six epochs, and fixed E6 endpoint. Fit only on
the 10,325 official training sequences; evaluate the complete exposed 1,873
official-valid set for development diagnostics. Keep 112 raw features, no
teachers, no initializer, no validation-derived inputs. Empty all model state
at each sequence boundary; carry state through chunks and detach every 1,000
rows. A proposed attention context 256 includes the current row and at most
255 prior rows; it is not the gradient horizon or a claim of full-sequence
attention. Keep the Mamba2 state and local key/value cache semantics distinct.

This is an idea, not an implementation or admission: the hybrid adapter,
causality/reset/chunk-state tests, static parameter count and actual-shape L4
profile are missing. The old 12E Mamba2 fits are informative historical
curves, but not matched controls for the proposed six-epoch training schedule;
the four-fit panel therefore needs fresh Mamba2 controls. Any decision
threshold must be frozen before admission. Two paired seeds support only a
descriptive screen, not significance or public/serving claims. ROOT owns all
future admission and GPU execution.

## Preparation checks

The four conditional GRU/GRU-local configs were loaded through the current
config/model/runtime-contract path on CPU. Their model construction produced
950,274 parameters for GRU and 1,165,156 for the proposed compact GRU-local
variant; this is a static count, not a memory or speed profile. Existing causal
and state tests were run, not rewritten:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONPATH = '<worktree>\src;<worktree>'
python -m pytest tests/test_compact_sequence_models.py -k 'compact_causal_full_chunk_step_reset_and_gradients or compact_chunks_preserve_attached_state_gradients' -q
python -m pytest tests/test_sequence_fit_early_stop.py -k report_only -q
```

Results were 6 passed / 6 deselected and 3 passed / 2 deselected,
respectively. These tests cover small synthetic causal prefix behavior,
full-versus-chunked outputs, reset and detached-state behavior, finite
gradients, and the report-only policy retaining the fixed endpoint. They do not
test a combined Mamba2/local-attention architecture or its L4 resource cost.
