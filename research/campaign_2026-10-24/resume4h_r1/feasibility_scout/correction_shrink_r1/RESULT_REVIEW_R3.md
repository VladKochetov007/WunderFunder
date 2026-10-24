# Independent result review: correction shrink R3

## Findings

### One exposed-cell scale comparison does not identify an optimum or establish equivalence

R3 evaluates a single fixed additional multiplier, `alpha=0.1`, on five saved
F0 endpoints. The shrink-versus-parent point deltas range from +0.0001290 to
+0.0001362 WP, but every paired component-bootstrap 95% interval crosses zero
(141 source-qualified components; 468 evaluation groups; 2,000 draws). The
shrink-versus-raw point deltas are +0.0006806 to +0.0007549, but those
conditional intervals are much wider and also cross zero. These are
fixed-prediction intervals, not training-seed or fold uncertainty. The
positive point changes are measured descriptive evidence for this fixed-alpha
follow-up; the intervals limit confidence about component-level repeatability,
not the existence of those observed point estimates.

Raw mean-only A and relationship arms B/C are descriptively close on this
exposed F0 sample: their five overall WP scores span only 0.0000812
(0.6805286–0.6806098). After the additional 0.1 shrink, the five scores span
0.0000072 (0.6812831–0.6812904). The shrinking operation deliberately
contracts every endpoint toward the common parent, so the tighter post-shrink
spread does not show that the mean-only and relationship mechanisms are
equivalent. No equivalence margin or equivalence test was specified. The
source recipe defines A as mean-only, B as slope/intercept updates through
row 98 then frozen, and C as continual slope/intercept updates.

The evidence supports only this exposed-sample description: raw endpoints score
below their parent, and one fixed extra 0.1 scale moves their point scores
above the parent on this exposed cell. It does not identify an optimal scale,
establish a repeatable gain, or generalize to other folds. Because the original head already applies
its residual coefficient of 0.1, this operation scales the exported
parent-relative endpoint difference by another factor of 0.1; it is not a
fresh fit or a test of whether parent preservation exists.

### Artifact and raw-reproduction checks pass

The grade binds the admitted R3 recipe SHA
`b2bf2a065687e3d970db852fa55cacd0d202da43eb587527b99683a3881e6802`, source
manifest SHA `62d96dbe1881ef2e42939fecdb8179c3dfdd6caa79122b961e31b4acf28711bb`,
the original F0 grade SHA
`bfc06c9820f48b5214bd32d13783a268a3e35da8ffc8c61f65b91912352e2899`, and
preflight receipt SHA
`f8fd01dab2226fdc46660e72a1cc25d1da3457429f404d0d0f036e18f01d7692`. The
execution receipt reports the exact R3 command, return code 0, 48.83 seconds,
zero new fits, zero swap and zero OOM/OOM-kill events. The recipe and result
contain the expected five primary and thirteen secondary contrasts.

I compared all five unique raw candidates across five time bands for T0, T1,
WP, selected-row count and metric identifier: **125 of 125 fields match the
original F0 grade exactly**. A-lag16's omitted endpoint is an exact raw alias
of A-lag1 in every band, as reported by the preflight identity and the original
grade. The target-cache receipt records the expected digest before labels, the
same digest after scoring, and `stable_during_score=true`.

### Resource caveat

The admitted cap was 400,000,000 bytes and the execution receipt records a
399,998,976-byte memory peak, only 1,024 bytes below the cap. It completed with
no OOM event and no swap, but the margin is negligible. This successful run is
valid; a repeat under this exact cap has little observed headroom.

## Exposure and limits

This is an exploratory, post-original-grade F0 result (fold 0, seed 1001),
using exposed predictions over 468 groups and 141 components. The bootstrap is
conditional on these fixed exports. The original A/B/C head panel remains
incomplete for F1 and the historical 12-fit panel remains incomplete. This
review used only the scalar recipe, result, admission, execution receipt,
original grade and existing `REVIEW_R3.md`; it did not open target or
prediction arrays or execute the grader.

**Disposition:** result integrity and raw baseline reproduction pass. Treat
all contrasts as exposed single-cell diagnostics. Do not call the model
families equivalent or the fixed 0.1 coefficient optimal based on this grade.
