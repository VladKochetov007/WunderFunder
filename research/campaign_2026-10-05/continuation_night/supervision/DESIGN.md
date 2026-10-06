# Surviving-span label reanchor: registered before implementation

Status: registered, no scientific fits admitted. Parent contract: ../CONTRACT.md.
Hypothesis H-124 / experiment E-494 (proposal CN-SUP-01): retaining a small original
outcome contribution on stopped-bootstrap surviving rows improves supervision.
Source: reflect_05_day/SECOND_PASS_PRIORITY_TRIAGE_2026-10-05.md:93,122,158.

Three fresh arms share the exact strong recipe described in
../activation_transfer/DESIGN.md: bootstrap8/reanchor0, bootstrap8/reanchor.1,
and bootstrap0/reanchor0. All use the historical GELU readout. No alpha search.
The fresh unchanged control is explicitly shared with E-493 at the same
seed/fold and immutable recipe, rather than selected from historical fits.
For an origin t, survival requires next_target_pair_change(t)>t+8 and t+8
inside the sequence. On surviving rows only, target becomes
.9*(.5*clip(teacher_t)+.5*clip(teacher_t8))+.1*clip(Y_t), with clip at +/-2.
Other rows retain the historical target exactly. Loss weights always remain
abs(clip(Y_t))*historical_row_weight. Warmup/mask/propensity/auxiliary/EMA
treatments, boundaries and final-head fitting remain unchanged.

The .1 coefficient is a fraction of the final blended target, not merely the
substituted half. Require constant-alpha cached distillation and a positive
bootstrap horizon when it is nonzero; reject invalid/NaN coefficients and
incompatible options before opening any data. Zero is the historical default.
Both ordinary and pooled macro training paths must implement the same target.

Screen seed3/fold0/fixed ep3. Confirmation seed4/fold0 and seed3/fold1, final
head seed3/fixed ep0 throughout; complete equal parent/head predictor primary.
Keep identical fold-specific teacher caches and fitting order across arms.
Use pooled official metric, target/class/event-age diagnostics and 2,000 paired
component bootstrap draws. Record fraction and origin-weight mass of surviving
rows as diagnostics without tuning coefficient from evaluation labels.

Before fit: independent review and bounded CPU boundary/algebra/loss-gradient
tests, real full-shape resource probe and root admission. Compare unchanged and
treated allocation using production B32/T1000 objective/EMA/Muon, separately
audit loader parent-tree RSS. Prefer one-worker Kaggle while local memory is
tight. Contract quotas/headroom/deadline apply; no fit is launched by this card.
Resource failures are censored, not hypothesis rejection. A target construction
or lineage failure stops the affected branch. A null complete effect or failed
seed/block replication is evidence against this fixed treatment only.

Keep bootstrap-off even if the reanchor parent screen is weak, and fit every
declared final head. No combining treatment with activation candidate here.
The original-outcome privileged teacher, selection weighting, cadence and
normalizer families remain separate queued hypotheses. Plots/tables go in
eda/ and root merges ledger_packet.json; this actor does not alter root ledgers.
