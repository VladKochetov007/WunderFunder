# Measured EDA, with no fitted models

Date: 2026-10-17. There are two different scopes; do not confuse them.

* The **census** reads all 12,480 sequences / 249,600,000 rows, but only four
  selected price-code columns, two targets and masks/sequence IDs. It took
  165.77 seconds at one CPU thread, with 1,037,440 KiB peak RSS. Its script,
  exact counts, compressed per-sequence statistics and hashes are retained.
* The **stratified sample** reads all 112 features in 256 full sequences /
  5,120,000 rows: 32 sequences from each of four retrospective classes in
  train and valid, seed 20261017. It took 24.70 seconds and 1,247,116 KiB.
  Sampling is uniform within class, not the natural population mixture.

All existing folds remain exposed. Labels define diagnostic classes, never an
input or serving gate. Train has no true scoring mask: its descriptive mass uses
`need_prediction`; valid uses `need_prediction & is_scored`. Counts are not
independent trials. No forecast quality, model fit or hidden-test distribution
was inferred from these descriptions.

## The old exact target clock is falsified

| Census | Eligible target changes | Changes outside p0 AND p11 | Outside p0 OR p11 | Outside AND, officially scored |
|---|---:|---:|---:|---:|
| Train | 26,810,254 | 941 | 595 | No train scoring mask |
| Valid | 6,468,490 | 312 | 207 | 92 |

Thus H-001's universal implication has 1,253 counterexamples. The first
stratified investigation retained 52 explicit exceptions, all need-eligible and
with tick-sized changes; that size characterization was not measured for every
census exception. Example: valid row group 1384, step 14166 changes targets by
approximately (-1.6893, +2.1720) while both selected i0 price codes stay fixed.
The census keeps first-in-order examples, not a representative exception sample.

This corrects a scientific assumption; it does not prove the existing event
features harmed a fitted model. These features may remain useful. The code
must not treat their clock as an exact target trigger, freeze output between
events, or assume a true lagged target has matured. No model behavior was changed
on that basis. Nor do the four selected price columns exhaust possible input events.

## Where valid target weight lies

The following sums are from every valid sequence. "Age" means rows since the
last joint change of the selected i0 components. Before the first such change,
the sequence-start sentinel supplies the origin; this is left-censored history.
This age is not elapsed time since a target change or a physical market timestamp.

| Retrospective class | Sequences | Scored rows | Selected magnitude-weight sum, both targets | Fraction of its weight at age >=64 | Fraction at clipped target boundary |
|---|---:|---:|---:|---:|---:|
| 0 | 398 | 990,483 | 1,382,738.6 | 7.409% | 18.538% |
| 1 | 347 | 536,650 | 772,584.4 | 10.643% | 13.861% |
| 2 | 431 | 284,961 | 405,526.3 | 25.564% | 6.784% |
| 3 | 697 | 1,716,179 | 2,334,458.7 | 43.690% | 3.449% |

Class 3 contains about 47.69% of the combined magnitude weight. This is **not**
an additive 47.69% contribution to the final correlation, whose centering and
normalization are global and target-specific. Its long quiet intervals deserve
error decomposition on existing predictions; the observation alone supports
neither a hard class gate nor a new weighting objective. The sample estimate
36.27% of class-3 weight at age >=64 is superseded by the census value 43.69%.

The census includes train row group 528 and finds its read columns finite. The
historical target cache excluded it and has no class for it; the report retains
class -1. No historical fitting membership or cached row order was rewritten.

## Full-vector sample findings

* No adjacent complete 112-feature vectors were identical in these 256
  sequences. At least one extra feature changes on every adjacent transition.
  Literal repeated-row compression has no opportunity in this sample; sparse
  component updates, approximate states and non-adjacent recurrence are different
  questions. The census did not read all features and cannot extend this claim.
* Clipped target-pair correlations were approximately -0.65, -0.75, -0.89,
  -0.90 by valid class. A simple invertible output rotation adds no expressive
  power. A useful joint-target proposal must change sharing, distributional
  structure or regularization without imposing the disproven exact label law.
* Class-conditional marginal train/valid histogram JS reaches 0.2349 nats
  (`i1_v15`, class 1) and 0.1773 (`i1_v13`, class 2). Bins are anchored in train;
  each side contributes 128 sampled rows from each of 32 sequences. Sequence
  composition and heterogeneity can explain this. It is not a significance test,
  proof of temporal drift, or reason to discard i1.
* The sampled lag tables retain event co-counts. They are not causal-effect
  estimates or target prediction results. Fixed lag/shift controls are a separate
  diagnostic, recorded separately when completed.

## Reproduction and independent review

Use [`stream_audit.py`](stream_audit.py) and
[`full_event_audit.py`](full_event_audit.py), one thread for Arrow/BLAS, with
`--data ~/WunderFunder/datasets --cache ~/WunderFunder-creative/cache --output NEW.json`.
The scripts refuse existing output paths. The first adds `--per-class 32`.
The data files remain on beleriand. Source hashes in the receipts bind the exact
scripts as executed; no formatting rewrite was applied after execution.

[`EDA_AND_NOVELTY_R3.md`](../reviews/EDA_AND_NOVELTY_R3.md) independently reviews
mask semantics, pooling, sequence heterogeneity, left censoring, limited feature
coverage and exception sampling. The official storage contract gives one sequence
per row group; these scripts assert length but do not independently re-verify
every row's sequence ID. This is a stated verification limit.
