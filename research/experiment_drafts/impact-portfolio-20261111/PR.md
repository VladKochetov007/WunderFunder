# [Experiment] IMPACT81: three implemented Transformer representation screens

[Portfolio and status](README.md) · [shared contract](CONTRACT.md) ·
[implementation details](IMPLEMENTATION.md) · [B300 commands](B300_COMMANDS.md) ·
[PR60–80 review](PR_REVIEW.md)

1. **Status, lineage and owner.** Cards01–03 are implemented in this PR and
   pending receiver acceptance; cards04–10 remain proposals. The three
   implemented hypothesis/experiment pairs are `H-IMPACT-20261111-01` /
   `E-IMPACT-20261111-01` (continuous numerical PLE),
   `H-IMPACT-20261111-02` / `E-IMPACT-20261111-02` (same-row feature-group
   tokens), and `H-IMPACT-20261111-03` / `E-IMPACT-20261111-03`
   (target-private adapters). Preparation owner: Ivan / Codex. Receiving owner
   and allocated B300s remain unassigned. No receiver checks, profiles, fits,
   grades or quality measurements have been recorded. The training runtime is
   pinned to the plain Transformer source in PR75 at
   `8626000a4b965763a70011a0291a891aa6535d7e`, loaded from a separate clean,
   read-only worktree. This supersedes the earlier suggestion to put the first
   three mechanisms in separate implementation PRs; card02's PR8 continuation
   is included here as a dated recipe amendment, not a duplicate PR.
2. **Why it should work.** Card01 gives the shared temporal model a continuous
   nonlinear basis over each scalar while keeping the raw projection bypass.
   Card02 mixes the canonical 16 feature groups at the current row before the
   unchanged temporal model. Card03 allows two nonlinear readouts to adapt to
   target-specific features while both losses continue training one shared
   trunk. The predicted observations, falsifiers and capacity alternatives are
   specified in [cards01–03](cards/). These mechanisms do not establish that
   representation limits explain current errors; receiver metrics decide that.
3. **Compute advantage.** The planned B300 fits compare each treatment with a
   fresh same-seed control on the same full-data recipe. They test three distinct
   representation mechanisms; they do not claim faster training, better scaling,
   or a leaderboard gain. New input/readout operations have real runtime cost.
   Hardware and throughput optimization are outside this change. The remaining
   seven ideas are proposal-only and are not part of the 12-fit plan.
4. **Frozen comparison.** Each card uses two fresh seeds, `204101` and `204102`,
   with one plain control and one treatment per seed. The common candidate is the
   PR75 plain Transformer: H768, 24 layers, 12 heads, FF multiplier4, context1024,
   gradient horizon256, batch128 under the pinned runtime semantics, accumulation4,
   and fixed final epoch4. Muon/AdamW roles, LR3e-4, weight decay0.01, clip1,
   5% warmup/cosine floor0.1, compiled temporal blocks, and BF16/BF16x9 behavior
   are inherited unchanged. Use all 10,325 train sequences and all 1,873 valid
   sequences; the split receipt SHA256 is
   `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
   Canonical normalization remains over all fit rows. Card01 estimates its 16
   linear quantile knots from 64 evenly spaced row positions in each train
   sequence (660,800 label-independent sample rows); these are not exact quantiles
   over every row. Valid/test data do not contribute to preprocessing. Per-card
   deltas are paired by seed, with no old control reuse or teacher fit.
5. **Bounded fitting panel.** The current plan is 12 new scientific fits: each
   of cards01, 02 and03 has two fresh controls and two fresh treatments. No fit
   has run, and no old PR75 control is reused. There are no teacher or
   pretraining fits. Profile runs are separate engineering work. Cards04–10 are
   not selected for execution; the 40-fit total in `recipe.json` is the inventory
   only if all ten cards are eventually selected. No automatic multi-cell launch
   is supplied. Every full-size cell needs its own accepted 50-update profile,
   reload evidence and operator-approved resource receipt before its fit.
6. **Validation and uncertainty.** The primary endpoint is the canonical pooled
   `global_wp_clip2_v1` on all 3,528,273 scored valid rows. Return both seed
   deltas, per-target t0/t1 values, coverage and fixed-E4 identities. The official
   valid split is repeatedly exposed development data. Two seeds on this one
   split do not estimate broad training variance, establish significance, or
   demonstrate fold/source robustness. No fit scores, intervals or conclusions
   exist for these cards yet. The working allocation screen in
   [CONTRACT.md](CONTRACT.md) is not an automatic promotion or submission rule.
7. **Leakage and coverage.** Inputs at row t use features observed through t and
   the canonical carried state only. All transformations and learned parameters
   use train data. PLE knots use the fixed train-only sample described above;
   canonical moments still use every fit row. Group attention sees the 16 groups
   at the same row, never a future window. Target adapters receive hidden states,
   not target values or validity flags; both target losses update the shared
   backbone. Reset, prefix-causality, chunk carry and reconstruction are checked
   by the receiver before profiling. No valid/test updates are introduced.
8. **Decision-changing result.** Apply the existing screen as a descriptive
   allocation heuristic: mean paired WP gain at least 0.0005, both seed deltas
   positive, and review any per-target loss worse than 0.001. Preserve mixed,
   negative or incomplete endpoints. This is not a significance test, automatic
   winner rule, or forecast. Adapter contribution/group diagnostics may help
   distinguish active mechanisms from unused capacity, but are not all emitted
   by the current runtime; any additional analysis must be identified as
   receiver-derived or separately instrumented. No valid-driven tuning of
   coefficients or post hoc endpoint selection is allowed.
9. **How to run.** Commands are prepared in
   [B300_COMMANDS.md](B300_COMMANDS.md) and have not been executed. On the
   provisioned Linux receiver, stage PR81 in its own worktree and the exact PR75
   commit in a separate detached worktree; bind the actual data and split; create
   one shared preprocessing artifact; run receiver checks on an available B300;
   then profile one selected cell for 50 updates. Review the real profile and
   reload evidence before writing approval; fit only with that receipt and the
   same GPU pair/runtime. Export and score the fixed E4 checkpoint with full row
   coverage, then compare the four endpoints for one card. The launcher is
   one-stage only, requires an owner/devices/time limit, has no retry or device
   search, and leaves other jobs untouched. Do not run packet model stages on
   macOS. See [MIDAS handoff instructions](midas/README.md) for evidence import;
   the [route preview](midas/route-preview.json) is metadata-only and does not
   represent an allocated worker or execution.
10. **Evidence to return.** Preserve per-cell source/config/runtime/data/split
    identity; preprocessing artifact; receiver acceptance and profile/reload
    receipts; owner and device allocation; producer-stamped updates, losses, LR,
    preclip gradient norm, skipped/nonfinite counts, elapsed time and peak
    memory; all epoch checkpoints and resume state; fixed-E4 prediction manifest,
    exact row coverage, canonical score and both paired deltas. Keep evidence and
    weights archives separate with hashes and storage locations. The
    [MIDAS evidence exporter](midas_evidence.py) packages completed run evidence
    without copying weights; producer-provided weight hashes are not byte
    verification when the archive omits weight bytes.
11. **Results and trace.** There are no measured results from this PR: 12 fits
    are planned, zero executed; receiver acceptance and profiles are pending.
    [Implementation](IMPLEMENTATION.md), [commands](B300_COMMANDS.md),
    [cards](cards/), [recipe](recipe.json), [append-only hypothesis ledger](../../hypotheses.jsonl),
    and [experiment ledger](../../experiments.jsonl) carry the current design and
    status. [PR_REVIEW.md](PR_REVIEW.md), [SOURCES.md](SOURCES.md), and
    [REVIEW.md](REVIEW.md) retain prior evidence, literature and independent
    criticism. PR76's separate CPU ZIP is unchanged and is not evaluated here.

No local model imports, construction, tests, inference, training, profiles or
existing-job changes were performed. The receiver must provide actual acceptance
and fit receipts before any scientific result is claimed.

Static validation: Ruff lint/format, AST structure, frozen config/hash checks,
shell-command syntax and independent integration review. See
[the static receipt](STATIC_IMPLEMENTATION_REVIEW.json); no numerical or GPU
acceptance is claimed.
