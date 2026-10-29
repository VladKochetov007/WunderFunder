# Validation robustness audit (R1, zero-fit)

User question: most robust validation / cross-validation for public LB transfer and long-term
out-of-sample stability. Method note and protocol: [ROBUST_VALIDATION_20261029_R1](../../unified/methods/ROBUST_VALIDATION_20261029_R1.md).

- Chain-link direction (PR49 `sequence_chains.py --source-qualified`, scratch root): 9,193 links;
  train→valid 48, valid→train 42, within-split 9,103. Valid is block-structured; boundaries both ways.
- [Adversarial validation](adversarial_validation.py) ([result](adversarial_validation.json)):
  component-grouped 5-fold OOF AUC **0.984** (shuffle 0.499) from mean/std/last of 28 raw columns.
  Driver rerun with saved features: `adversarial_validation_r2.json` (pending).

**Prior evidence:** this replicates E-094 (AUC 0.989–0.993 within class), E-161 (90 links vs ≈2,300 expected) and E-110 (stress panel ordering 3/3 LB comparisons) from 2026-09-26; see the method note §0.
