# TL3 fresh train-only restoration

The user requests restoring TL3 training and measuring the complete resulting
predictor on our official primary validation. This is a new adaptation: historical
TL3 fitted train+valid and its published score0.66832 is not our independent-valid
baseline. No historical weights or fitted-data scores are promoted.

## Contract

- **Question:** does a freshly rebuilt TL3-style teacher/distillation chain beat
  the clean GRU256x2 benchmark, or improve its fixed blend, on full official valid?
- **Endpoint:** the three original final roles, with predeclared .34/.33/.33
  coefficients and original fixed epoch references. The recovered graph contains
  17 neural predictor fits /54 configured epochs, not just three final fits.
  Source-qualified node definitions are being checked before admission.
- **Data:**10,325 official-train sequences after282 feature-connected exclusions;
  all1,873 official-valid sequences /3,528,273 scored rows. Split SHA256
  `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
  See [ADR0044](../../../specs/adr/0044-chronological-primary-validation.md).
- **Mandatory adaptation:** no learned propensity and no valid-mask weighting.
  Official train has no is_scored labels; the old propensity fit used valid.
  Retain target-magnitude WMSE and the recovered causal since-event weighting.
  All learned ancestors, preprocessing and cached targets use fitting members only.
- **Numerics:** prospective strict FP32, autocast/TF32 off, no quantization.
  Historical FP16 is recorded as another declared reproduction difference.
- **Implementation:** reuse the canonical modular legacy RNN trainer and its
  mixed distillation target, chunk-local updates, EMA and auxiliary losses.
  Do not substitute the simpler sequence trainer's additive KD objective.
- **Selection:** ancestor endpoint epochs are fixed before inspecting valid;
  monitoring valid does not choose teacher/initializer weights. The final fixed
  endpoint is primary; any best-epoch diagnostic is exposed exploratory evidence.
- **Benchmark:** clean plain GRU256x2 .6778561232160607 on full valid. A separately
  frozen80/20 GRU+GALAM-LSTM128 blend scored .681219678579555. These are exposed,
  single-seed local results; neither establishes public/private performance.
- **Resources:** reuse the existing eight-L4 node and SSH session. Seven GALAM
  fits continue unchanged; start with free GPU0. No new instance, rental or billing.
  Initially at most8 CPU cores,16GiB loader/support RAM and one23GB L4 per admitted
  lane. Profile before fitting; preserve original batch/chunk/update counts.
  Admission closes after17 predictor fits,168 wall hours or256 GPU-hours,
  whichever comes first; a deadline leaves an explicitly partial chain, not a
  silently shortened scientific recipe. Leave at least12GiB filesystem headroom.
- **Abort:** nonfinite updates, source/membership mismatch, insufficient disk,
  foreign resource ownership or inconsistent cache ancestry. No blind retry.
- **Evidence:** immutable resolved recipes/source hashes, profile/update evidence,
  train objective, full-valid WP by target, LR/preclip gradients, checkpoints,
  resource telemetry, ancestry/cache hashes and an evaluated final prediction
  bundle. The first teacher is complete; the full TL3 endpoint is not yet available.

This is empirical software research (tierB): exact code checks can falsify bugs;
finite exposed validation cannot establish market transfer. No fresh hidden
confirmation data is available. Fixed ensemble coefficients have historical
selection exposure; new metrics remain exposed development evidence.

## Source recovery

- [Historical TL3 source closure](../../campaign_2026-10-18/quality_gap_r1/tl3_source_closure_r1/README.md):
  archived trainers, tiedlead shell recipe, logs and dependency names; source
  snapshot is not authenticated as the exact source present at historical fit time.
- [Fin256 metadata execution](../../campaign_2026-10-18/quality_gap_r1/tl3_fin256_metadata_execution_r1/README.md):
  saved foundation arguments for seeds3/4, including EMA, since-event weighting,
  auxiliary heads and fixed four-epoch horizon.
- [Canonical training map](../../../src/wnn/training/README.md): reusable trainer
  components, unlike the historical monolithic source snapshots.
- Branch basis:8482de5c71ce8bdd9974780438d518ea50bfeca8; this work stacks on PR46.
  Read-only inspected origin/research-team7e4074314a62d0907c9608f9220cf72958708a2d;
  origin/main3eeb2a9c2a0f456112f76175122deab376c764b3. No unrelated merge performed.

## Navigation

Receiving profile, exact node recipes, independent source criticism and launch
receipts are appended here as each artifact exists. Prepared is not running;
partial is not completed; training on official valid is forbidden.

- [Historical source DAG receipt](SOURCE_DAG_R1.json) preserves the first scout's
  recovery as received, including historical FP16 settings and a superseded cost
  calculation. It is evidence, not an executable new recipe.
- [Adapted DAG](EXECUTION_DAG_R2.json) records FP32, neutral propensity and explicit
  fresh AdamW for fin256's unknown historical optimizer. The fixed recipe module
  below additionally preserves recovered since-event32 weighting for fbc1k,
  tiedlead and orgbase. Other valid-only fine-tunes now fit official train.
- [Independent review](INDEPENDENT_TL3_TRAINONLY_REVIEW_R1.json) identified the
  contradictory checkpoint gate, inference schema and cache ancestry/order gaps.
  These are implementation issues requiring actual receiving checks; no quality
  approval is inferred from an agent review.
- [Frozen17-node recipe assembly](../../../scripts/tl3_chronological_recipes.py)
  names every fresh run, parent, endpoint and original teacher coefficient. The
  canonical loader drops incomplete batches: B32 has322 batches and B16 has645;
  the corrected graph total is257,740 chunk updates /54 configured epochs.
- [Fixed endpoint scorer](../../../scripts/score_tl3_chronological.py) streams
  full-valid targets, verifies prediction/checkpoint identities and applies the
  frozen .34/.33/.33 blend before metric clipping. It is not yet an evaluated TL3
  result. [Recipe/arithmetic tests](../../../tests/test_tl3_chronological_recipes.py)
  check all17 parsed chronological recipes and clipping order (3CPU passes).
- [Receiving evidence](receiving/README.md) contains the first actual40-update
  profile and detached fin256_s3 launch. Source-r3 is immutable while it runs;
  later cache/optimizer/dashboard changes use new source exports.

## Current execution

One new scientific fit, **tl3c_fin256_s3**, completed on free L4 GPU0 using the
already-admitted source-r3. It is a scratch GRU256x2, batch32/chunk2000, four
epochs, WMSE plus .3 structural/.3 next-change heads and EMA consistency1,
since-event32 weighting, fresh AdamW .001 + OneCycle, strict FP32. The selected
teacher endpoint is fixed ep3. It ran detached from SSH with a six-hour per-fit
limit; no old GALAM fit or source was modified. Actual profile:40/40 completed
updates, finite losses and preclip gradients,17.21 sequences/s;322 batches/epoch
implies about10minutes of fitting plus validation/checkpoint time per epoch.

A brief co-resident engineering profile achieved9.11seq/s while another fit
occupied GPU0, versus17.21seq/s alone; this does not establish a reliable
throughput improvement, so another scientific fit is not admitted to the same
card. The finite chain will use verified released GPUs. Full TL3 restoration and
its validation score remain pending, not completed.

The fixed first-teacher endpoint scores **WP0.6882596673755748**, versus the exposed
plain GRU benchmark0.6778561232160607 (delta+0.0104035441595141). Full-valid epoch
scores are0.6744168482,0.6852992218,0.6878730780 and0.6882596674. This is one teacher
fit with different inputs/objective, not a causal ablation or the final TL3 score.
See the [actual first-teacher receipt](receiving/FIRST_TEACHER_PROGRESS_R1.json).

The same [validation dashboard](http://127.0.0.1:18766/) now contains all four
checkpoint-derived train/valid WMSE and WP points under **tl3c_fin256_s3**.
The train diagnostic uses32 frozen sorted fitting groups and all rows after
warmup; full valid uses the official is_scored mask. Their absolute loss gap is
not a directly matched-mask overfit estimate. Composite online training objective
is a separate series. The bridge did not restart training or modify its source.

The remaining16 fits and automatic native-log dashboard views are being admitted
through the fixed finite dependency graph. Only actual controller state establishes
running status. The first completed teacher must be adopted, never fitted again.
