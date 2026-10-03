PASS the frozen fold2+3 adoption gate as written, with supported metadata and
score-implementation alignment. No material correction is indicated. This is
confirmation of a COMBINED local recipe, not attribution to an individual
mechanism or a claim about public score. Real prediction/checkpoint tensors were
not loaded and the reported real-data score/interval was not independently
recomputed from payloads.

Inputs were byte-frozen into this disjoint directory before audit:
combo_fold23_prereg_report.json, FOLD23_CONFIRMATION.txt and the score design
combo_fold23_prereg.json. INPUTS.json records original paths/hashes. The report's
embedded design and screen SHA match the frozen design exactly: folds2/3,
single fixed member per arm, COMBO minus CTRL,2000 paired component resamples,
seed271828, primary clipped pooled GlobalWP. The gate text explicitly admits
prior exploration of these folds; this is additional fixed-procedure confirmation,
not an untouched or multiplicity-free test set.

| Declared gate | Verified measurement | Result |
|---|---:|---|
|Positive pooled mean|+.001787771655|PASS|
|Positive pooled95% lower bound|+.000924909294; upper+.002747123854|PASS|
|Positive pub143 diagnostic mean|+.001873155099; CI[+.000774940442,+.003085195660]|PASS|
|Positive pooled class3 delta|+.003252448128|PASS|
|Fold2 class3 no worse than-.002|-.000334103070|PASS|
|Fold3 class3 no worse than-.002|+.006778813572|PASS|

Fold2 overall delta is+.000866225444, CI[-.000286447068,+.001907584525].
Fold3 is+.002602113890, CI[+.001334885861,+.004274566984]. The specified gate
does not require each fold's interval to exclude0. Class3 is locally worse in
fold2; it was not hidden or replaced. The class guard is a point-estimate guard,
not a per-class significance test. Pooled class0/1/2 deltas are approximately
+.00183530/+.00202087/+.00127120. Reported pooled scores are .705543514400
for CTRL and .707331286055 for COMBO; these are local fold scores only.

Actual scoring route: scripts/screen_score.py imports the frozen
research/campaign_2026-09-30/audit/campaign_score.py. score_sequence applies
step>=99 AND the stored score mask, validates required outputs, clips both
target/prediction to[-2,2], weights by abs(clipped target), and calculates centered
moments. pooled merges those moments over all selected rows and averages the
two target correlations. It does not average sequence scores or omit constant/
one-sided sequences from the primary metric. The named softened-mask and legacy
sequence-mean sections remain diagnostics. All reported t0/t1 means and
bootstrap/exact deltas agree internally within1e-12.

Two small executed checks exercise this route without real arrays: an adversarial
panel with constant one-sided sequences, clipped extremes and warmup/mask
exclusions matches the actual standalone root utils.weighted_pearson helper
within1e-12; an identical-arm paired bootstrap preserves zero mean/CI while
merging shared component IDs. The actual pub143 reweight implementation also
matches an integer-expanded standalone-helper calculation. Reweighting scales
sequence weight/M2/covariance to the requested CLASS SHARES OF SEQUENCES, leaves
the means intact, then pools centered moments. It is not a weighted average of
class correlation coefficients and not an official public-domain score.

Member/metadata checks cover all four CTRL/COMBO fold2/3 members:

- Same GRU256/two layers, seed3, four epochs, alpha.5 and evaluation EMA; the
  snapshot is fixed epoch INDEX3, with no evaluation during fitting. Provenance
  selected_epoch, sidecar epoch, report member epoch and model_ep3 hash entries
  agree. Training uses zero-based epoch indices, so index3 is the fourth epoch.
- COMBO records recurrent Muon throughout the schedule, chunk1000 and strict
  distill_boot_h8; CTRL records chunk2000/bootstrap0 with AdamW defaults.
  The dcurve job helper fixes the epoch and predictor invocation; it does not
  choose an interim high-scoring checkpoint. The optimizer label is recurrent
  Muon, not a claim that every head/projection parameter uses Muon.
- Sidecar checkpoint/prediction identities agree with report and provenance
  metadata; fold-manifest hashes match. Tiny NPY HEADER reads verify complete
  float32 shape(1873,20000,2). All sidecar row orders agree with the SMALL cached
  row_group index. No prediction payload, target/mask or model tensor was loaded.
- Fit/evaluation rows equal the frozen manifests, and fit components exclude
  evaluation components. There are443/505 evaluation sequences and144/145
  disjoint components in folds2/3, matching pooled948 sequences/289 components.
- Both students use identical recorded cached-teacher identities within each
  fold. Eight train/valid teacher metadata entries bind to their sidecars,
  epoch/checkpoint/prediction IDs and fold manifest; teacher fit metadata also
  excludes the evaluated components. This audits the metadata chain, not the
  teacher/prediction bytes or a new full source-lineage reconstruction.

screen_score.bound_member checks prediction hashes/split/row order at scoring
time, but its quick-screen route has weaker fixed-epoch/fit-provenance enforcement
than the full frozen validate_member route. The independent metadata checks
above fill that gap for THESE actual members; it is not a claimed current bug.
This audit does not independently rehash the large prediction/checkpoint payloads
or recover original bootstrap vectors. No specific inconsistency justified those
bulk reads under the requested scope. Pairing and source arithmetic are verified;
raw-payload score reproduction remains a separate verification tier.

Contrast with the previously exposed folds0/1:

| Existing contrast | Pooled delta |95% interval | Scope |
|---|---:|---:|---|
|COMBO seed3 vs matched CTRL seed3|+.00215711|[+.00052926,+.00390595]|folds0/1|
|COMBO seed4 vs matched CTRL seed4|+.00178819|[+.00024309,+.00343348]|folds0/1|
|Two-seed COMBO bag vs two-seed CTRL bag|+.00185360|[+.00042018,+.00340938]|folds0/1|
|Muon-only seed3 vs CTRL|+.00159503|[+.00025158,+.00301330]|folds0/1|
|COMBO seed3 vs Muon-only|+.00056208|[-.00031198,+.00135397]|folds0/1|
|Width320 vs256 seed3|+.00007249|[-.00121126,+.00131341]|folds0/1|
|Width384 vs256 seed3|+.00200859|[+.00072262,+.00331725]|fold0 only, earlier screen|

The new seed3 fold2/3 result has a similar mean scale to both earlier matched
seed contrasts. Fold1 effects were much smaller than fold0, and fold2 smaller
than fold3; heterogeneity remains. Seed bags are paired recipe mixtures, not
independent new components. Width320's weak result does not disprove every
capacity mechanism; width384's positive single-fold screen is less confirmed.
These reports overlap data and selection history, so they are not independent
p-values or interchangeable population samples. Prior report identities and
numbers are preserved in results/summary.

The strongest supported mechanism statement is limited: Muon alone improved
the earlier paired screen, while the additional chunk1000/bootstrap8 bundle
versus Muon alone remains unresolved there. Positive fold2/3 COMBO vs AdamW
does not settle that incremental contrast, identify a stopping law or prove
bootstrap8 is conditionally unbiased. Likewise, it cannot distinguish optimizer
geometry, gradient truncation, additional update opportunities, clipping,
normalization or their interaction.

Actual update metadata makes the confounding concrete. Fold2 CTRL has14920
optimizer attempts/14913 completed updates/14920 EMA updates over4 epochs;
COMBO has29840 attempts/29828 completed/29840 EMA updates. The batch count is
identical1492. OneCycle's total is batches*ceil(20000/chunk)*epochs, scheduler
advances per attempt, and EMA advances after attempts including skipped AMP
steps. Thus the shorter chunk doubles these clocks per row as well as changing
gradient truncation and per-chunk loss denominators. For unchanged decay.999,
two EMA advances retain coefficient.999^2=.998001 on the older state over the
same2000-row exposure, rather than.999; that algebra does not equate the teacher
trajectories because student updates intervene. Matching seed/fit rows alone
also does not establish identical batch RNG/order or optimizer trajectories.

Recommended ONE long-term mechanistic experiment is the cadence/objective-
matched2x2 in MECHANISTIC_PREREG.json: gradient detach1000/2000 crossed with
bootstrap0/8, recurrent Muon fixed. Every arm uses ONE2000-row macrostep, one
clip/optimizer/scheduler/EMA advance, genuine raw/EMA state and identical pooled
loss numerators/denominators. Only the gradient path is cut at1000 in the shorter
arm; forward state is carried. This controlled macrostep normalization is a NEW
objective, explicitly not exact reproduction of the current combo.

Its primary observable is the paired truncation main effect, averaging horizon
contrasts and seed-specific effects with shared component resamples. A point
effect>=.0005 with95% lower>0 and both seed means positive supports an independent
credit-path contribution under those controls. An entire interval within
[-.0005,+.0005] supports only scoped equivalence; an interval crossing0 is
otherwise inconclusive. Bootstrap and interaction contrasts are predeclared
secondary/descriptive, not post-hoc mechanism selection. The rival is that most
gain comes from Muon and residual extras depend on cadence/mass normalization
or target semantics rather than shorter gradient credit. A null does not prove
which rival caused the original gain. Exact forward/loss/step and skipped-update
instrument parity precedes any future training. The complete long-term design
would be4 arms x2 seeds x4 folds; duration/resources need a separate controller
decision. Nothing is scheduled or self-approved by this recommendation.

Reproduce this bounded audit from repo root with
`bash research/campaign_2026-10-03/continuation/round4/combo_confirmation_review/run.sh`.
The CPU wrapper is round2-critic, onecore/2GiB/900s. Body cost about.09s,
CPU.17s including imports, peak RSS85812KiB; exact cost is in STATE/COST.
No GPU, fitting, queue or ledger writes occurred. Every new artifact is confined
to combo_confirmation_review; previous actor/review files are preserved.
