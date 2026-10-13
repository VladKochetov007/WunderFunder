---
kind: review
status: complete
created_utc: 2026-10-13T23:49:44Z
updated_utc: 2026-10-13T23:49:44Z
hypothesis_ids: [H-CPU-POLICY-20261014, H-STUDENT-LADDER-20261014, H-STATE-INIT-20261014]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/PLAN.md
  - research/campaign_2026-10-14/cpu_student_frontier_r1/CONTRACT.json
  - research/campaign_2026-10-13/qat_renewable_r1/final_release_r1/BASELINE_BENCHMARK_BINDING_R3.json
  - cache/clean_t3_f0_h390_t3r5_a9_r2/metadata.json
  - src/wnn/distillation_cache.py
  - scripts/train_rnn.py
  - src/wnn/nn.py
  - src/wnn/specialist.py
---

# CPU student frontier: first-wave specification audit

## Question or claim

Does the first-wave plan identify enough immutable inputs and invariants to support exact A/B/C/D CPU comparisons, a matched eight-shape student screen, and a causal state-initializer experiment before any result is interpreted?

## Mechanism and method

I read the campaign plan and contract, relevant team-sharing and cross-fold decisions, the empirical-validation protocol, the existing xteam_h4 package and binding receipt, the clean-T3 cache metadata and ancestry, and the trainer/model/cache-reader paths. I did not edit implementation code, launch profiles or fits, evaluate predictions, upload a package, or re-read the large teacher-target arrays. This is the bounded pre-fit specification review; exact actor source and command review remains a separate pass.

## Evidence and interpretation

### Fixed xteam_h4 CPU comparison

The original package is available at `artifacts/team_models_20261013/xteam_h4.zip`, SHA-256 `f1635d14c0e1b47848b5caeb2ec4ad9362a85800f1b2e2df80702127d3867b27`. I independently confirmed that the extracted `members.json`, `solution.py`, precision file, four member NPZs, and three native libraries match the hashes in the existing binding. The fixed panel is ordered as `weights_0` through `weights_3`, each at 0.25; `precision.txt` selects `q8vnni`. The wrapper fixes `budget_s=None`, selects a native level from the CPU flags (or an explicit override), and logs the actual level and precision. The ZIP is therefore a suitable frozen serving reference.

The binding records source labels for all four members but no original checkpoint paths or hashes. That limits claims about their training ancestry; it does not prevent an exact engine comparison against these archive bytes.

For the A/B/C/D comparison, bind the archive and every member hash, callback input/preprocessing, coefficients, precision, runtime library, CPU flags, and reset order in each run receipt. Log an engine-specific assertion as well as the selected native level: `v3` alone does not prove a B run used 256-bit EVEX VNNI. Review generated assembly for the intended YMM VNNI instructions and absence of a silent fallback. For C, verify packed layouts invert to the same logical weights and per-row scales. For D, verify the GRU equations use the old hidden state and retain separate input/recurrent quantization scales and biases. If any treatment changes arithmetic, preserve it as a numerical variant rather than calling it exact.

The decisive parity check is A versus each treatment over complete 20,000-row sequences, with the actual callback and sequence resets. Include repeated A-B-A replay and the fixed full F0 evaluation sequences; require finite outputs and bitwise equality for an exact claim. Run timing separately from parity, then report actual package bytes, resident memory, ABI/import status, callback wall time, and the eventual competition-host duration. No such parity or runtime result exists in this review.

### Clean F0 teacher targets and the eight-shape ladder

The available cache is `cache/clean_t3_f0_h390_t3r5_a9_r2`, metadata SHA-256 `1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8`. Its source-qualified split is pinned by SHA-256 `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`. Metadata binds three distinct scratch-initialized H384×2 teacher checkpoints at epoch 2 (seeds 3, 4, and 5), their provenance hashes, 10,535 fit-train and 1,405 fit-valid groups, and zero of the 468 evaluation groups. The homogeneous target is `float32(0.8125 * float32_mean(three_epoch2_predictions))`; the cache construction itself does not clip it. The target arrays are joined by row-group ID and the current validator checks ordered membership, shape, dtype, finite values, and payload hashes.

**Blocking finding — the existing cache reader rejects the proposed ladder as a consumer.** `src/wnn/distillation_cache.py::_validate_cleanfold_t3_profile` also hard-binds the producer metadata to its historical student recipe: H256×2, batch 16, chunk 2,000, six epochs, fixed checkpoint 5, alpha 0.9. A direct call from the new screen will reject other widths/depths and a one-epoch screen before fitting, even though the teacher targets and membership are valid. Keep the producer metadata and target arrays immutable. Add an explicit, hash-bound consumer recipe for this campaign that validates the frozen producer contract separately and admits only the declared new shapes and common training settings. The legacy reader must retain its exact old behavior; do not add a broad bypass or relabel the old six-epoch contract. This is an implementation dependency, not evidence against the model family.

The teacher cache and student initialization are separate ancestry. The existing H256×2 CP5 student, checkpoint SHA `e30577ff…b776eda`, was not scratch initialized: its provenance points to `c20261003_combo_s3_f0@ep3`, whose own two teacher parents were F0 seed-3/4 runs. It is not a neutral initializer for the larger and narrower candidates. The new recipe must state one initialization policy for all eight candidates (for example, scratch initialization for each architecture) and record it. Do not compare a candidate initialized from this H256 parent with scratch candidates as if architecture were the only treatment.

**Required before fit — common normalization and sample order.** `RNNModel` uses 211 normalized inputs, a linear projection, stacked GRU, then a readout over concatenated recurrent output and projection. The current `fit_normalization` estimates mean/std from the first two loader batches. Training provenance identifies the sampler and loader RNG as the global CPU RNG; model construction occurs before that normalization pass. A shared seed therefore does not by itself bind the same sampled examples or identical normalization across architectures. Existing seed-3 CP5 preprocessing hashes are `bb85c72b…c5a2da8` / `b7306008…e4454c5`; the three teacher seeds also have different mean/std hashes. Freeze one fitting-only 211-feature mean/std pair for the student comparison and assert its hashes in all eight receipts. Bind the batch/group order independently of model initialization, then compare exact attempted/completed optimizer updates and the terminal checkpoint rule. Keep target mix, official label weighting/clipping, auxiliary losses, propensity, loss denominator, optimizer, schedule, and update budget common. The root’s planned one-epoch full-F0 FP32 screen is a new bounded discovery recipe; it is not an equal-budget comparison with the historical six-epoch CP5 student.

F0 target and evaluation groups have prior project exposure. Report this screen as exposed-F0 discovery; paired component uncertainty and later seed/fold variation answer different questions. Cross-fold robustness does not create a pristine holdout, and a component interval crossing zero is not an automatic stop rule.

### Causal state initialization

The repository’s `PrefixAdapterGRUModel` is not the proposed state-initializer treatment. It computes an inclusive running mean from up to the first 99 normalized rows and uses that context to alter the input encoding at every row. The plan instead requires normal zero-state processing of rows 0–98, followed by exactly one bounded residual update to the carried GRU state after row 98 and before row 99. Keep this as a new arm; the existing prefix adapter can be a related comparator only if labelled separately.

For the new prefix-state arm, freeze the summary definition, normalization, clipping/bound on the residual, and shape `[layers, batch, hidden]`. The summary must use only rows 0–98 (including last observation row 98); row 99 and later inputs, targets, score masks, and sequence identifiers are forbidden. Split the first training chunk at that boundary so the adjustment is applied once and gradients follow the declared joint/frozen-backbone condition. A useful invariant is that, with identical base weights, rows 0–98 match the zero-state control exactly and only later outputs can differ. Test whole-sequence versus chunks split at 99 and at ordinary TBPTT boundaries, A-B-A sequence replay, and complete reset of hidden state plus prefix accumulators. First-row initialization remains a separate arm and must initialize before consuming row 0.

## Limits and validity

The baseline ZIP, member files, native libraries, clean-T3 metadata and small teacher checkpoint/provenance files were checked by hash. The 1.9 GB teacher target arrays were not independently rehashed here; their identity is recorded in metadata and is checked by the production cache validator. No candidate source, actual command vector, full-sequence predictions, training run, fitted score, timing result, or public transfer was reviewed. Original xteam_h4 checkpoint provenance is unavailable in the supplied archive record. All current folds are exposed.

## Decision and next step

The CPU reference is concretely pinned and the cache producer ancestry is suitable for an F0 screen, subject to the separate-consumer-contract repair. Before any architecture fit, freeze the consumer recipe, initialization policy, shared normalization and sample order, exact update/checkpoint rule, and common loss semantics. Implement the prefix state update as its own causal arm. Before labeling any engine variant exact, require the policy assertion, generated-assembly evidence, and full-sequence bitwise replay. A second independent review should check the exact consumer patch and commands, then later inspect the produced model/export artifacts; no quality or runtime conclusion follows from this memo.

## Source records

[Campaign plan](../../campaign_2026-10-14/cpu_student_frontier_r1/PLAN.md), [contract](../../campaign_2026-10-14/cpu_student_frontier_r1/CONTRACT.json), [live frontier](../../campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md), [xteam_h4 benchmark binding](../../campaign_2026-10-13/qat_renewable_r1/final_release_r1/BASELINE_BENCHMARK_BINDING_R3.json), [clean-T3 ancestry and recipe](../../campaign_2026-10-13/qat_renewable_r1/delivery_r1/README.md), [student cache metadata](../../../cache/clean_t3_f0_h390_t3r5_a9_r2/metadata.json), [cache consumer](../../../src/wnn/distillation_cache.py), [trainer](../../../scripts/train_rnn.py), [GRU model](../../../src/wnn/nn.py), [prefix adapter](../../../src/wnn/specialist.py), [empirical-validation protocol](../../../.agents/skills/conduct-creative-research/references/empirical-validation.md).
