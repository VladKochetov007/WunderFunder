# Validation robustness audit (R1, zero-fit)

User question: most robust validation / cross-validation for public LB transfer and long-term
out-of-sample stability. Method note and protocol: [ROBUST_VALIDATION_20261029_R1](../../unified/methods/ROBUST_VALIDATION_20261029_R1.md).

- Chain-link direction (PR49 `sequence_chains.py --source-qualified`, scratch root): 9,193 links;
  train→valid 48, valid→train 42, within-split 9,103. Valid is block-structured; boundaries both ways.
- [Adversarial validation](adversarial_validation.py) ([result](adversarial_validation.json)):
  component-grouped 5-fold OOF AUC **0.984** (shuffle 0.499) from mean/std/last of 28 raw columns.
  Driver rerun with saved features: `adversarial_validation_r2.json` (pending).

**Prior evidence:** this replicates E-094 (AUC 0.989–0.993 within class), E-161 (90 links vs ≈2,300 expected) and E-110 (stress panel ordering 3/3 LB comparisons) from 2026-09-26; see the method note §0.

## Driver analysis (r2, `adversarial_validation_r2.json`)

Same OOF AUC 0.984. **No single feature separates train from valid**: the best univariate AUC is
0.70 (`mean:i1_v18`), then `mean:i0_p12` 0.65, `last:i1_v18` 0.64, `std:i1_v10` 0.64, and the
rest are 0.59–0.61 (volume-code means and stds of i0/i1, `a0`/`a4` level and spread). The
shift is many small level/scale moves that are jointly separable, not one broken column.
`i1_v18` mean and volume-code statistics replicate the E-094 drivers (2026-09-26). Implication:
dropping or fixing one feature cannot remove the shift; per-sequence scale robustness and
regime-held-out selection are the relevant defences. Features saved in
`adversarial_validation_r2_features.npz` (local, not committed if large) for regime clustering.
