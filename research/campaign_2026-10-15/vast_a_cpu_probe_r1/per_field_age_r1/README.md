# Per-field causal age representation R1

The unlaunched [matched screen R2](MATCHED_SCREEN_R2.md) supersedes this page's
proposed fit/export recipe: common original warm parent, two matched epochs,
fixed CP1 and full1,873-valid-group export followed by468-group evaluation.
It adds the missing production Torch/oracle check before actual GPU profiling.
The R1 engineering results below remain historical evidence.

**Lineage:** H-087/H-138; E-PER-FIELD-CLOCK-RESOLUTION-F0-20261015-R1.

**Status:** source and matched recipe prepared; no fit, GPU profile, export, or grade has been admitted. The historical 211-input path remains the default and is unchanged.

This branch adds two append-only feature modes to the existing trainer. For each raw float32 column, `field_age` resets its age to zero when the value differs exactly from the preceding row, increments otherwise, caps at 10,000, then encodes `log1p(age)`. `group_age` uses the same clock but resets when any member of one of the seven fixed raw-field ranges changes, broadcasting that clock over the range. Row zero starts at age zero. These are deterministic encodings of observed history, not additional information.

The implementation is in [`research_features.py`](../../../../src/wnn/research_features.py); trainer integration is limited to the explicit `--research-features` modes and append-only initialization check in [`train_rnn.py`](../../../../scripts/train_rnn.py). The original `none` mode, checkpoint feature width, and normalization behavior remain the existing 211-input path. [`test_field_age_features.py`](../../../../tests/test_field_age_features.py) contains four already-passing CPU-only tests against a separate tiny row-loop oracle, including group broadcasting, field/group distinction, chunk and reset parity, and future causality. Tests were not rerun while preparing this note.

## One-pass fixture timing

[`CPU_FEATURE_AGE_PROFILE_R1.json`](CPU_FEATURE_AGE_PROFILE_R1.json) records one full-sequence NumPy feature-construction pass over the eight hash-bound sequences in the parent [runtime fixture](../CPU_TIMING_FIXTURE_R1.json). It reads only `features.bin`; request flags, labels, official masks, teacher/cache and propensity payloads, and the parquet body were not read. For 20,000 rows, group-clock transform times ranged from 0.144 to 0.183 seconds per sequence; per-field times ranged from 0.275 to 0.614 seconds, with the first field-mode call slowest. This is a single-pass feature-transform timing, not a serving callback measurement, stable comparative CPU result, or target-host capacity estimate. The fixture is exposed F0 fitting data and supports no quality claim.

The completed pass used the existing `scripts/limited` wrapper with one CPU, a 500 MB memory cap, no swap, and a 120-second runtime cap:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=500000000 WNN_RUNTIME_MAX=120s scripts/limited \
  .venv/bin/python scripts/profile_field_age_fixture.py \
  --manifest research/campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_TIMING_FIXTURE_R1.json \
  --payload-root artifacts/vast_a_cpu_probe_r1 \
  --output research/campaign_2026-10-15/vast_a_cpu_probe_r1/per_field_age_r1/CPU_FEATURE_AGE_PROFILE_R1.json
```

## Matched fit recipe for a later root admission

Use three matched children from the same source-qualified parent checkpoint `student_f0_warm_s3_alpha0_r1_p1@ep1` (SHA-256 `5ac7b148388fbe42952377ab8582a7dc43332028b3546bd282dbafbd7b6ac8d0`), whose provenance is `fac44ce5cc752cd77a599cf087616711f9ab586f609152a537b2d37e84afafdc`. Preserve its `c20261003_combo_s3_f0@ep3` ancestry (checkpoint `d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e`; provenance `9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f`). The parent is a warm, label-only alpha-zero continuation, not a cold model; alpha zero means the bound clean-T3 cache contributes no teacher coefficient during that continuation.

| Arm | Input | Feature mode | Purpose |
|---|---:|---|---|
| C | 211 | existing | Parent representation control |
| G | 323 | `group_age` | Seven clocks broadcast across 112 suffix coordinates |
| F | 323 | `field_age` | 112 per-field clocks |

Primary comparison is F−G; C is secondary context. G and F have equal input/projection widths, but G repeats seven unique clocks across 112 columns while F supplies 112 clocks. Their input rank and optimizer geometry therefore differ. F−G tests finer temporal resolution **together with** the resulting optimization geometry; it is not a pure causal-information comparison or a parameter-count control.

For G/F, append exactly 112 normalization means of zero and standard deviations of one to the parent's frozen 211-element buffers (mean SHA-256 `bb85c72b87656979f97c1786400aec6bc9e43acd337078000eb482c09c5a2da8`; std SHA-256 `b730600851e52e246fc638230844d6e4f2828e4972b45db1f0128cce4e4454c5`). Preserve the parent's original input projection and zero-initialize only the appended columns. Do not refit normalization independently by arm. C keeps the original 211-element buffers and projection.

All arms should use the same source-qualified fold-0 split (SHA-256 `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`), seed 3, fixed sequence order, batch 16, chunk 2,000, six complete continuation epochs, fixed epoch-5 checkpoint, FP16 and the existing OneCycle/AdamW, weighted-loss and auxiliary-target recipe. Start a fresh optimizer from the parent weights. The parent-level cache is `cache/clean_t3_f0_h390_t3r5_a9_r2` (metadata SHA-256 `1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8`); keep its homogeneous family, teacher ancestry, targets, masks, propensity weighting, label clipping, and alpha-zero setting identical across arms. The frozen split has 11,940 fit groups; at batch 16 with `drop_last`, 11,936 are used per epoch and four omitted, yielding 746 batches / 7,460 chunk updates per epoch. Record attempted/completed updates, AMP skips and finite-gradient counts per arm; do not extend training or select checkpoints from evaluation quality.

After a later root admission, predict all 1,405 fold-0 valid groups from epoch 5 using the source-qualified manifest and existing `scripts/predict_run.py` causal exporter. Grade the same 468 official fold-0 evaluation groups with `global_wp_clip2_v1`, paired across C/G/F and preserving the fixed row-position buckets 99–199, 200–999, 1,000–4,999, 5,000–9,999, and 10,000–19,999. This fold is exposed development evidence, not a fresh holdout. No current score exists.

The current Python batch/stream feature paths and trainer integration have not yet been exercised on a fitted checkpoint or real GPU batch. The native C++ serving path does not implement these new feature modes. Before considering a submission, implement and test exact native causal age/reset/chunk behavior, compare native predictions to Python, then measure package size and runtime. The fit/export/grade sequence, native integration, and all quality claims remain unimplemented and require separate source review and resource admission.

## Shared-control successor R3

[Warm mechanism wave R3](../../gru_parent_growth_r1/WARM_MECHANISM_WAVE_R3.md) supersedes the unlaunched alpha0 R2 allocation with a newly matched alpha.9 C/G/F/W comparison and explicit seeded sampler. [Nine combined tests](../../gru_parent_growth_r1/CPU_TEST_RESULT_R1.json) now include the previously missing production Torch/oracle age check; GPU profiles and quality fits remain pending.
