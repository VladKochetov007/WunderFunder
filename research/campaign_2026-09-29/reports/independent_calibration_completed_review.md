# Completed calibration replay and handoff — 2026-09-30

Verdict: declared package/family replay arithmetic PASSES; an expected-public
score >=0.6660 remains UNSUPPORTED. The audit is stopped at the user's instruction
to start no further runs. No dataset or prediction-file scan and no GPU work was
performed.

## Independent replay

One CPU-only invocation of `independent_calibration_replay.py` ran under
`WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G scripts/limited`. Its independent
OLS uses centered long-double sums, not the author's `polyfit`/`inner_error`.

Both scenarios completed assertions covering all 21x11 matrix entries' shape and
finiteness, variant order, nested selection, all 11 outer predictions, selected
inner errors, full slope/intercept, RMSE/MAE/bias/range, top-pair accuracy, and all
five declared-family holdouts. All compared numbers agreed within 2e-11. Both
scenarios use identical first-nine matrix columns, public observations, panel IDs
and recorded input identities; their single-member EMA6 columns are identical.

The subsequent ZIP identity phase completed: all three recorded local archive
hashes matched, and both failed packages contain identical member-0 weights and
v2 native library bytes with a v2 single-member branch.

The invocation then failed in an extra reviewer metric fixture: I asserted that
a float32 0.35/nextafter fixture would give raw score zero, but it gave official
and raw scores both equal to one. This is an incorrect reviewer fixture
expectation, not a detected calibration regression. The trace proves execution
reached this phase after replay/identity assertions. The overall command exited
1 and its aggregate JSON was not written. Its tool-reported elapsed wall time was
0.3964 s; no per-phase timing or measured peak RSS was emitted.

Raw evidence is preserved in `independent_calibration_replay_v1.log`; the exact
reviewer code is `independent_calibration_replay.py`. No retry was launched after
the user's stop instruction.

| Scenario | Package LOSO RMSE | Declared-family holdout RMSE | Best reported exploratory forecast |
| --- | ---: | ---: | ---: |
| Pair | 0.00179541 | 0.00257672 | 0.66210225 |
| Single | 0.00077825 | 0.00167112 | 0.66266487 |

The six candidate local scores are not persisted as a matrix. Their reported
forecast maxima above were read from the artifacts; independent replay verified
their fitted calibration coefficients, but did not regenerate the candidate
scores from dataset/prediction files.

## Concrete limitations and source issue

1. **Identity ambiguity remains, and the lower single-scenario error must not
   select the scenario.** The single scenario assigns one identical deterministic
   predictor to public scores 0.6616 and 0.6612. Those scores cannot both result
   from that predictor on an identical evaluation panel under identical execution.
   Archive identity is verified; portal CPU path, panel identity and user-reported
   associations are not directly observed. The pair scenario is consistent with
   distinct predictors. This consistency evidence favors that explanation only
   under the stated same-panel/exact-association assumptions.

2. **Family labels are a sensitivity partition, not ancestor-disjoint groups.**
   `scripts/campaign_calibration_audit.py:17` groups early/s4/ema4/since/ema6.
   Baseline is present in early's held-out package and in training s4/ema4 blends.
   The since parent's raw checkpoint remains in the mixed EMA6 package while its
   fine-tuned descendant is held out. The outer labeled family is excluded
   correctly, but proxy selection inside it remains package LOSO (`:50`). The
   artifacts properly call the family definitions post hoc. Do not interpret five
   families as five independent ancestor groups or the resulting RMSE as a mean
   confidence interval. A future stronger sensitivity would group shared members
   and ancestors, acknowledging the much smaller number of usable groups.

3. **Static finiteness bug in the historical score builder.**
   `scripts/holdout_compare.py:32` clips raw predictions before any nonfinite
   check. Selected +/-infinity therefore becomes +/-2 and can enter a finite
   calibration score. `scripts/decide.py:55` has the same pattern. The saved
   matrix's finiteness cannot prove its raw predictions were finite. Validate raw
   required predictions before clipping and use the audited centered reference
   for calibration. Raw-moment precision is a documented additional limitation;
   this invocation's attempted nearconstant counterexample did not reproduce it.
   Actual impact on these artifacts is unestablished because dataset scans were
   prohibited. No nested-selection arithmetic bug was found.

4. **Selection and public transport are still unaccounted for.** Both LOSO methods
   condition on the historically chosen proxy family and submission inventory.
   Packages share ancestors and the public period; upstream model/blend search is
   not replayed. The fitted slope is not a validated map from novel clean OOF
   mechanism gains to public gains. Keep forecasts exploratory, retain failed
   submissions, and do not grant expected-public credit from these artifacts.

## Handoff

The corrected paired-head local-screen review remains in
`independent_pair_review.md`. The broader expectation rationale/formula remains
in `independent_expectation_review.md`; this completed replay strengthens its
arithmetic audit but does not change its target conclusion.

If the user later authorizes more work: preserve old artifacts; validate raw
prediction finiteness and centered score parity; persist candidate local-score
matrix plus a hash of the family source artifact; review truly ancestor-grouped
sensitivity and novel-mechanism transport. No further computation is authorized
by this handoff.
