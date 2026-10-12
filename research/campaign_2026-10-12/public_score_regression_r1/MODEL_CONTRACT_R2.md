# Public-score diagnostic model contract, R2

This contract preserves R1's feature set, `alpha=10`, splits, and metrics. It
corrects two deterministic source-family mapping errors: source identifiers
take precedence over aliases, and an audited alias containing `h230` maps to
the existing H230 HFT student category when source identifiers are absent. R2
results are therefore a post-hoc implementation correction/sensitivity, not
independent confirmation of the already-inspected R1 results. It does not
change ADR0030 submission selection. The target is the public score for rows
whose portal status is `SCORE`; three timeouts, one check error and three
unlisted portal events have no numeric score target.

## Broad metadata model

Use the 38 scored rows and only metadata already recorded in the artifact map:

1. Displayed upload timestamp as a continuous age feature. Parse the supplied
   local display string without assigning a timezone; standardize on training
   rows only.
2. Number of declared logical member specifications, and a routed-package
   indicator when the source has conditional routes. For a routed artifact,
   use the maximum declared branch count. This is not a count of native tensor
   members or measured serving cost. Missing member specs are represented as
   missing plus a training-fold-imputed value.
3. Audited precision category: Q8VNNI, i16, other declared, or unknown. Names
   alone do not establish precision.
4. Audited fit scope: all-data, fold0, fold1, or unknown. This is exposure
   metadata, not evidence of clean validation.
5. Source family category from exact member/source identifiers, falling back
   to a cited package alias only when source identifiers are absent. Fixed
   categories are: fold1 student; H388/H384 teacher; H230 HFT student;
   H230 hetero/C1 student; 2026-10-04 combo; D256/D07; legacy since/EMA;
   chunked ensemble; other source-qualified; unknown.

When source/member identifiers exist, map only their joined text. Consult the
package alias only when no source/member identifier is present. Family mapping
uses this fixed case-insensitive precedence: `chunk500`;
`vt_f1`/`vt_f0`; `vt_all_h388`; `c1v2`/`hetero`; `c20261004_combo`;
`d256`/`d07`/`alpha07`; `bd26_h230` or `h230` or `z1c`/`z2c`/`z4`/`z5`;
`ema`/`since`/`ab6`/`ab7`/`ab8`; any remaining nonempty source; no source.
The order is intentional, and the derived category is a coarse metadata label,
not a claim of shared weights or recipe equivalence.

No runtime, memory, portal status, local validation metric, or target score is
an input. The small ridge model has an unpenalized intercept and standardized
non-intercept columns, using the training-fold mean and scale only. The fixed
penalty is `alpha=10` in `sum_i (y_i - a - x_i b)^2 + 10 * ||b||^2`.
There is no penalty sweep, feature search, target-derived category, or row
random split.

The mean baseline is the arithmetic mean of the training artifacts. The
recent-history baseline is the arithmetic mean of up to the five most recent
training artifacts, ordered by displayed timestamp; exact timestamp ties stay
together in a test block. This is retrospective because score availability
times are not observed.

For the ancestry-grouped evaluation, connected components are formed from
overlapping source/member identifiers and explicit audited cluster IDs. Rows
whose member identity is unresolved share one conservative unresolved group.
One component is held out at a time; components are never split between train
and test. For the chronological evaluation, three contiguous test blocks are
made from distinct displayed timestamps after an initial half-history warmup.
Train rows sharing any source component with that block are purged. A block
with fewer than ten remaining training rows is reported as infeasible rather
than repaired with future/evaluation outcomes. Every result is retrospective
reconstruction, not a prospective forecast.

## Local-score diagnostics

Local-score models are separate from the broad metadata model and never use
missing-value imputation to give a row a validation score. Use the same
standardized one-feature ridge penalty `alpha=10`, unpenalized intercept, and
the same fold-only preprocessing. The receipt-bound fold1 float subset is
reported separately from the package-manifest-only fold1 value; both are
exposed/selected, and the receipt-bound 7/9/pair rows share one component.
The historical 368-group source-reported values are a separate protocol, with
unverified checkpoint ancestry; the two EMA values are conditional member0
proxies and form a third scenario subset. Any protocol with fewer than four
independent ancestry groups is declared insufficient for grouped predictive
evaluation. Descriptive local-minus-public offsets may be tabulated, but they
are not a fitted transfer function and do not correct scores for current
candidate ZIPs.

All reported performance summaries include row/group counts and MAE/RMSE.
Selecting a validation scheme or interpreting a retrospective fit adds
exposure; no numerical current-candidate forecast is promoted unless a
compatible out-of-group evaluation supports it.
