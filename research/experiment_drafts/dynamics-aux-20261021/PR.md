# [Experiment] dynamics-aux-20261021: future-input dynamics auxiliary head

1. **Status, lineage and owner.** `H-FUTURE-INPUT-DYNAMICS-20261021-R1` /
   `E-FUTURE-INPUT-DYNAMICS-20261021-R1`. Code is implemented on local branch
   `experiment/dynamics-aux-20261021`, implementation commit
   `b83147cf3264df68d6fe77b827fcedbca0be6a50` based on
   `3eeb2a9c2a0f456112f76175122deab376c764b3`; four focused CPU tests and
   static checks pass. No profile, scientific fit, export grade, or result
   review has run. The screen is cold-initialized and cold-screen-only.
   Owner: root for profile, fits, predictions and evidence integration. A
   strong-parent continuation requires a separately reviewed extension because
   the current auxiliary CLI validation rejects `--init`.
2. **Why it should work.** A causal recurrent state may be more useful when it
   predicts local future input dynamics as well as the existing targets. The
   auxiliary predicts mean standardized change and log change energy for 13
   raw-input groups over 16- and 256-row horizons. The strongest alternative is
   that this task consumes capacity or regularizes toward predictable but
   target-irrelevant motion. The matched weight-zero head and full endpoint
   comparison distinguish that from a useful target representation.
3. **Compute advantage.** This adds a small linear head and no teacher fits.
   Each state reads its future input targets from the already loaded full
   sequence; target-statistics fitting adds one feature-only pass over the
   selected fitting membership. No scaling law, quality gain, or public score is
   assumed. Follow-up is conditional on target-quality evidence and fixed
   complete-endpoint contribution; any strong-parent use needs a new recipe.
4. **Frozen comparison.** Both arms use cold H512×1 plain GRU, one-layer
   readout, batch8, chunk2,000, AdamW/OneCycle (`lr=.001`, `wd=.01`, `beta2=.999`,
   clip1), three epochs, fixed CP2, strict FP32, and unchanged chunk-normalized
   WMSE. The shared 367-slot `i1_history` control is allocated but all-zero
   (`--research-feature-control all`, `--drop-i1` false). Both arms attach the
   same 52-output recurrent-state head and use the same cell seed. The control
   weight is0; treatment weight is0.05. Per-output target mean/population std
   are fit only on unique selected fitting groups, with std floor1e-3. The
   main two-output readout is unchanged. Evaluation returns its two outputs;
   serving exporters strip the training-only head and buffers.
5. **Bounded fitting panel.** Two fits per arm, four new scientific fits total:
   F0/seed1001 control and treatment; F2/seed1002 control and treatment. The
   frozen F0 subset is reused byte-for-byte (1,118 train/118 valid); F2 uses the
   frozen sibling-generated subset (1,148 train/141 valid), from commit
   `b1b680d4a700d353f3e6db4f761f6670bd02fc7f`. These are membership
   artifacts, not reused model fits. There are0 reused exact model/control
   fits,0 teachers, and0 students. Root admits the entire panel only after
   sustained actual-shape profiling under the accepted three-GPU-hour screen
   allocation.
6. **Validation and uncertainty.** Use `global_wp_clip2_v1` and full source-
   qualified evaluation folds: 468 groups on F0 and443 on F2. Preserve all four
   scores and paired deltas, plus the fixed 90/10 control/candidate mixture in
   each cell. Two cells cannot separate fold and seed effects or
   establish significance; component resampling is conditional on each fitted
   pair and exposed fold. All folds are exposed. The +0.0003 mean official
   improvement from the accepted cycle plan is a practical follow-up target,
   not a kill gate. Auxiliary prediction quality by itself is insufficient.
7. **Leakage and coverage.** The group registry is
   `wnn.kalman_group_features.FEATURE_GROUPS`, spanning raw channels0:112. For
   each `h ∈ {16,256}`, `z=(raw-model.mean[:112])/model.std[:112]`,
   `d[k]=z[t+k]-z[t+k-1]`; predict each group's `mean(d)` and
   `log(mean(d²)+1e-8)`. Origins require `t>=warmup` and `t+h<T`. Future rows may
   cross a TBPTT boundary but never enter input assembly. Target stats use
   `SequenceFile.read(targets=False,mask=False)` on exact selected unique train
   plus valid-fit groups. Raw future values remain unthinned targets; an
   existing retained-mass mask gates target origins. Strong-parent ancestry is
   out of scope and currently rejected. The two-fold/two-seed panel is small
   and exposed.
8. **Decision-changing result.** Compare target quality, both WMSE targets,
   source/fold and sequence-position bins, each paired delta, and fixed
   complete-endpoint contribution. Consistent modest improvement or useful
   fixed-mixture contribution may earn a conditional follow-up. Mixed results
   require diagnosis; an interval crossing zero alone does not reject the
   direction. Auxiliary loss reduction alone does not justify a larger run.
9. **How to run.** From repository root, use the `fits` list and exact template
   in [recipe.json](recipe.json), substituting the frozen split and subset
   manifests for each fold. The implementation requires the canonical
   `uv` project environment and supplied local parquet data. Before any fit,
   root must complete and admit the sustained B-control profile within the
   accepted 3-hour dynamics allocation and shared resource limits. Export each
   fixed CP2 full evaluation fold with `predict_run.py`; score using the
   unchanged official command in the recipe. No fitting is authorized by this
   code-only handoff.
10. **Evidence to return.** For each cell preserve source/config/args hashes,
    exact split and subset hashes, per-fit target-normalization membership/hash
    and settings, head initialization identity, progress and preclip gradient
    norm, optimizer attempts/steps, LR, fixed checkpoint, full-fold prediction
    sidecar/header hashes, official scores, paired component uncertainty,
    wall time and peak RAM/VRAM. Keep evidence and weights archives separate.
    The training checkpoint retains auxiliary state for restoration and
    provenance; standard and native serving exporters strip it.
11. **Results and trace.** There are no fit, profile, parity, score, export, or
    leaderboard results yet. Focused test command:
    `uv run pytest tests/test_dynamics_auxiliary.py -q` passed 4 tests in2.35s
    under root's serialized 1-core/1.8-GB slot; Ruff and the structure checker
    pass. The current limitation is no strong-parent continuation support. The
    F2 fit subset is frozen in sibling branch `experiment/i1-screen-20261021`
    and must be integrated by exact hash before running. Link any
    execution receipts and resulting grade from the packet and unified ideas
    README; do not infer quality from this implementation.
