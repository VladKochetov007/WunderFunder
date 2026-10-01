# Independent event-clock reproduction

Verdict: **SHIP one matched GPU screen of the 8-event clock correction**, with
the horizon-24 control held fixed. This is an experimental gate only. No
submission or public-score improvement follows from the CPU study.

## Fixed panel and information boundary

`preregistration.json` (SHA-256
`4de64a3f281b09697e5c817593aac6a08469057a9e6956b4756e55dc0e26725b`)
selected 160 TRAIN groups and 32 distinct fold-2 VALID components by a
metadata hash before reading labels or masks. It excluded components in prior
v1/v2 reconstruction panels and any TRAIN group linked to this evaluation
panel. The decoder uses only TRAIN target labels: class-3 code spreads inferred
from target identities and ask-move votes from unclamped target jumps, with
support/confidence gates 3/0.90 and 3/0.80. Neither validation targets nor
scoring masks fit the decoder. This is a supervised diagnostic coordinate, not
a label-free reconstruction of physical quotes. Fold 2 had been inspected in
historical research, so this is an independent *implementation and panel*, not
an untouched public-transfer estimate.

The feature-only gate passed before validation targets or masks were opened:
237,533 quote4-eligible required rows, 37.30% of the 32-group panel. Retrospective
target lattices put 12 sequences in class 3. The class-3 TRAIN decoder had 73
accepted transition codes and 47 nonzero move codes; all price-code transitions
in the 12 evaluation sequences mapped, although this does not validate a
physical interpretation. Target-implied and TRAIN-mapped spreads agreed on
all 29,219 eligible scored rows.

## Four-event law on identical scored rows

The clock definitions are explicit: **code event** means any change in the 22
i0 price representation columns; **quote event** means a confidently decoded
nonzero ask or bid change. On the same 29,219 eligible class-3 scored rows, the
next-four-event rounded common midpoint produced:

| Clock | Exact target pairs | Exact fraction | Failed eligible clipped-target mass |
|---|---:|---:|---:|
| Representation code | 22,294 | 76.30% | 18,982.88 / 38,068.17 (49.87%) |
| Decoded quote | 24,487 | 83.81% | 13,908.46 / 38,068.17 (36.54%) |

The quote clock gains 7.51 percentage points in exact pairs and reduces
failed mass by 13.33 percentage points. All 12 components improve in exact
pair fraction. Component bootstrap resampling with a fixed RNG and 10,000
replicates gave an illustrative 95% interval of +6.04 to +9.04 percentage
points for exact-pair gain. This interval describes the already inspected
diagnostic panel; it is **not** a Global WP or public-score interval. Among
eligible scored class-3 rows, 27,444/29,219 are quiet rather than quote-event
rows, and their failure mass is 13,437.44/36,295.15. The failures cannot be
hidden by reporting event rows alone.

The post-replay E-079 check is explicitly exploratory. Of 7,851 eligible rows
whose true future reference lies outside the current spread, the quote-clock
path gets the **exact future tick tier** on 3,825 (48.72%). It gets 19,931 of
20,609 spread-one touch rows exactly (96.71%). Those are not the same
classification statistics as the historical E-079 ledger's 97.5%/99.5% and
cannot be compared numerically as a reproduction of those claims. The
historical scratchpad implementation is unavailable. The four-event rule is
strong near the touch but is not an exact generator on this panel.

## E-251 auxiliary-label semantics

The historical E-251 cache advanced 8/24 horizons on representation-code
changes, despite calling them book events. On the independent class-3 scored
panel, paired labels from the same TRAIN decoder and the same eligible rows
differ by >0.001 on:

| Horizon | Different eligible scored rows | Changed target mass / full scored target mass | Mean absolute label difference, target-mass weighted |
|---|---:|---:|---:|
| 8 | 23,169 / 29,156 | 31,672.50 / 38,675.11 (81.89%) | 0.17177 |
| 24 | 24,921 / 28,639 | 33,746.61 / 38,675.11 (87.26%) | 0.16747 |

The count of code changes is 91,461 versus 60,516 known decoded quote
changes: 30,945 code changes (33.83%) do not move the decoded quote. The
differences diagnose the labels' meaning; they do not explain the prior
E-251 model's negative fold score by themselves.

An exploratory equal-row alignment check compared each auxiliary label with
`0.5*(clip(t0)-clip(t1))`, weighted by the sum of absolute clipped targets.
At horizon 8, correlation improves from 0.93739 (code) to 0.95582 (quote).
At horizon 24, it *falls* from 0.94041 to 0.90893. This motivates changing
only the 8-event head in the preregistered GPU screen. It is a choice made
after viewing these fold-2 target labels; fold 2 cannot serve as fresh
confirmation of that choice.

## Resource and provenance ledger

- Frozen panel replay: durable CPU job `clock_reproduction_fold2`, 5.01 wall
  seconds, 4.46 CPU seconds, peak RSS 1,171,124 KiB, exit 0. Exact command,
  source hashes and device accounting are in
  `../jobs/clock_reproduction_fold2.json`; raw measurements are in
  `results.json`.
- Exploratory E-079 and label-alignment checks: 7.3 and 6.3 wall seconds under
  the one-core/3GB wrapper; outputs in `e079_diagnostic.json` and
  `label_alignment.json`.
- First full clean fold-0 auxiliary cache build failed after 31.06 seconds
  (exit 1). `lattice_audit_f0.json` identified exactly one target-lattice
  failure in 10,535 fit TRAIN groups: class-2 group 2315, max residual
  0.00151332 against the 0.001 ceiling, with very large raw targets.
  `lattice_audit_valid_f0.json` found zero failures in 1,405 fit VALID groups.
  The failed job and empty output folder remain preserved.
- Independent synthetic scalar checks covered every eligible origin at code
  and decoded-quote horizons 4/8/24; vectorized future midpoint means matched
  direct loops to below 1e-9. A 20,000-row half-zero-move path confirmed equal
  cache masks and an unchanged horizon-24 control.
- The second cache build was stopped cleanly after 90.67 seconds (exit -15)
  while a GPU parent was active, because shared file cache approached the
  campaign RAM cap. Its durable record is `clock_aux_cache_f0_v2`; no cache
  arrays were written. The optimized builder now explicitly excludes group
  2315 from the **auxiliary decoder/cache only** and records that choice.
  Main GRU training retains its original fold manifest and the group's main
  targets. `aux_class_index.py` first indexes allowed class-3 groups from
  fit-only targets, allowing the decoder/cache passes to skip other classes.
  Its output and the third cache build should be checked before GPU work.

The frozen GPU screen, exact commands, equal-data gates and measured parent
runtime/VRAM basis are in `GPU_SCREEN.md`. No GPU job has been launched by this
reproduction role.
