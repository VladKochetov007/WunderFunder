# Second independent E-517 audit

Reviewer: `gpt-6-luna` reasoning max, separate review session. No project files
were changed and no fit was launched.

**Artifact/calculation integrity: PASS. Preregistered screen: FAIL.** Under
one CPU, 1.5 GB maximum memory and 300 seconds, the reviewer hashed all 56
E-514 manifest sources plus E-517 preregistration/code/loader/notebook,
result/attempt, OOF files, plot and chain summary; registered digests and sizes
matched. The reconstructed pure fold-0 population is 432 groups, 123 complete
chains, 41 chains per fold and group counts 150/130/152. The row-group digest
and candidate sidecar row order matched.

Independent fitting reproduced zero gated and global coefficients in all
folds, thresholds `0.46923828125`, `0.412109375` and `0.41375732421875`, and
the reported routed-row and routed-weight fractions. Pooled anchor, gated and
global WP are all `0.6832264634983685`; per-fold endpoints also match exactly.
Because predictions equal the anchor on every scored row, all paired
complete-chain bootstrap deltas are necessarily `[0, 0]`; the reviewer
verified the seed/count metadata and fold-stratified sampling procedure and
derived the degenerate interval algebraically rather than replaying the draws.

All four pass conditions fail. Caveat: the recorded sensor-output SHA is
consistent across preregistration and result, but raw `sensors` output was not
retained, so its seed derivation cannot be regenerated. This does not affect
the zero intervals. The review checked the registered source bytes and
ancestry record; it did not rerun the original candidate training.
