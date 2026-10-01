# Why positive pooled WP can coexist with negative deltas on both folds

The saved reports are internally consistent. Pooled weighted Pearson centers and scales all selected observations together; the per-fold scores center and scale each fold separately. Their deltas therefore need not have the same sign. This is a valid property of the official metric, not evidence of leakage or a scoring defect.

This is a **posthoc, descriptive analysis of exposed folds**. It uses exactly three existing reports and their six-statistic summaries. It changes no predictions, fits no calibration, constructs no blends, performs no new model scoring or bootstrap, and does not change any gate or rescue a closed recipe. Component allocations below are algebraic, non-causal, and depend on the stated symmetric decomposition.

## Existing results reproduced

All numbers are candidate minus reference. One micro-WP unit below means `1e-6` WP.

| Existing comparison | Pooled delta | Fold 0 delta | Fold 1 delta |
| --- | ---: | ---: | ---: |
| TRANSHEAD-01-primary | +0.000036168402 | −0.000003463307 | −0.000000418443 |
| CTX-joint-memory-vs-current | +0.000055787529 | −0.000048406496 | −0.000003062491 |
| Q3-quant-vs-baseline3 | +0.000258384849 | +0.000375393233 | +0.000044130462 |

These are the same fixed comparisons already reported. Original target arrays and prediction caches were never opened. The frozen metadata contains 425 pure sequences in fold 0 and 435 in fold 1. The evaluator concatenates the two folds in that order, which identifies the saved-statistic boundary; independently reproducing both fold deltas verifies it. Maximum discrepancy from any existing pooled/per-fold readout is `1.11e-16`.

## Exact identities

For one target and fold `f`, the saved sums are

`(W_f, Σw y, Σw p, Σw y², Σw p², Σw y p)`.

The evaluator has already clipped target and prediction to `[-2,2]`, applied its mask and warmup, and used `w=|clipped target|`. No row operation is repeated here. Let `a_f=W_f/ΣW_f`; let `μ_yf, μ_pf`, `V_yf, V_pf`, and `C_f` be the weighted fold means, population variances, and covariance. Define pooled means `μ_y=Σa_f μ_yf`, `μ_p=Σa_f μ_pf`. Then

```
C_W = Σ_f a_f C_f
C_B = Σ_f a_f (μ_yf−μ_y)(μ_pf−μ_p)
C   = C_W + C_B

V_y = Σ_f a_f V_yf + Σ_f a_f (μ_yf−μ_y)²
V_p = Σ_f a_f V_pf + Σ_f a_f (μ_pf−μ_p)²
D   = sqrt(V_y V_p)
r   = C / D
```

For two folds, the between term simplifies to `C_B=a_0 a_1(μ_y1−μ_y0)(μ_p1−μ_p0)`. The corresponding between variances are `a_0 a_1` times each squared mean gap.

Writing `r_f=C_f/sqrt(V_yf V_pf)` and `κ_f=a_f sqrt(V_yf V_pf)/D` gives

`r = Σ_f κ_f r_f + C_B/D`.

The `κ_f` are **not fixed averaging weights**: they depend on the model's fold prediction scales and pooled denominator, and need not sum to one. Weighted target fractions are also target-specific: fold 0 has 0.478045 of t0's weight and 0.469074 of t1's, rather than simply its fraction of sequence count. Candidate/reference target sums are identical.

For candidate `c` and reference `r`, a symmetric exact product identity allocates the difference as

```
Δcorrelation = Σ_f mean(κ_cf,κ_rf) × (r_cf−r_rf)
            + Σ_f mean(r_cf,r_rf) × (κ_cf−κ_rf)
            + (C_Bc/D_c − C_Br/D_r).
```

The three rows are, respectively, the **within-fold correlation**, **fold coefficient/normalization**, and **normalized between-fold covariance** terms. The coefficient term includes changes in relative fold scales and the complete pooled variance denominator, including variance from fold mean separation. It is not a pure or causal estimate of a scale effect. Arithmetic means of these target-specific identities give WP.

## What accounts for each result

Terms below are micro-WP units; each row sums to its existing pooled delta.

| Comparison / target | Within-fold correlation | Coefficient / normalization | Between-fold covariance | Sum |
| --- | ---: | ---: | ---: | ---: |
| TRANS t0 | +0.075551 | +102.710609 | −56.816608 | +45.969553 |
| TRANS t1 | −3.837920 | −109.619606 | +139.824778 | +26.367253 |
| **TRANS mean WP** | **−1.881184** | **−3.454498** | **+41.504085** | **+36.168402** |
| CTX t0 | −4.373086 | +303.907591 | −131.255756 | +168.278749 |
| CTX t1 | −45.784489 | +18.535926 | −29.455129 | −56.703692 |
| **CTX mean WP** | **−25.078787** | **+161.221759** | **−80.355442** | **+55.787529** |
| Q3 t0 | −142.009353 | +59.211348 | −25.956723 | −108.754727 |
| Q3 t1 | +549.147027 | −151.223419 | +227.600817 | +625.524425 |
| **Q3 mean WP** | **+203.568837** | **−46.006035** | **+100.822047** | **+258.384849** |

**TRANS:** The favorable mean-separation term exceeds the two small negative terms. The target mean gaps (fold 1 minus fold 0) are −0.034417 for t0 and −0.091689 for t1. Prediction mean gaps change from −0.076553 to −0.066943 for t0, and from −0.078192 to −0.087159 for t1. Thus the normalized between covariance falls for t0 but rises more for t1: the net contribution is +41.504 micro-WP. Fold prediction standard deviations move only slightly. The positive pooled delta does not imply better average within-fold correlation.

**CTX:** Here the between-fold term actually becomes less favorable. The coefficient/normalization term accounts for the positive pooled delta. For t0, the prediction mean gap narrows from −0.101554 to −0.079169, reducing the between-fold prediction variance from 0.00257331 to 0.00156391. Within-fold prediction variance also falls. The t0 pooled prediction variance falls from 1.17664750 to 1.17059101, reducing `D` from 1.45118305 to 1.44744344. Fold 1's prediction standard deviation falls more (1.067145→1.063222) than fold 0's (1.101177→1.100526). The t0 coefficient for the higher-correlation fold 0 rises 0.486212→0.487179; fold 1's falls 0.512517→0.511952. This is exactly the information discarded by inspecting fold correlations alone. It yields a positive coefficient term large enough to exceed the negative within-correlation and between-covariance terms.

Negative WP deltas in both folds mean the **average over the two targets** is negative in each fold. They do not say every target/fold cell worsened: TRANS and CTX each have a small positive t0 delta in fold 1. The exact identity retains those cells instead of assuming that both target correlations declined everywhere.

**Q3 contrast:** Its within-fold correlation term is positive at +203.569 micro-WP, consistent with its already-reported positive deltas in both folds. This is led by t1; t0 worsens. Mean separation also contributes, so even this pooled delta is not simply an average of its fold deltas. This descriptive contrast establishes no new quality gate or confirmation.

## Verification and limits

The companion Python script uses local NumPy with numerical-library thread counts set to one. It reads only the three named saved-statistic NPZs, their reports, frozen evaluator source files, and the metadata arrays `split,fold,pure,row_group`. Twenty-four generated direct-centering cases verify the covariance/variance identity; a separate generated example verifies that two negative fold deltas can coexist with a positive pooled delta. Maximum generated direct-centering error is `2.22e-15`. Saved-data covariance/variance reconstruction errors are at most `4.45e-16`; symmetric target-delta reconstruction error is at most `3.34e-16`, against a declared absolute tolerance of `2e-12`. Runtime was under one second.

The JSON records every input/source SHA256 and complete fold/pooled moments. Key bindings:

- Frozen evaluator `scripts/fold_eval.py`: `d05791f5f8c6ea64d9863d4f4b45430eb46b9d0b29ccbdac1a4ff5b36a2573c0`.
- Metadata `cache/chain_folds.npz`: `315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35`.
- Diagnostic JSON: `72bae89912252d9bf96a75d9b43af261c095bc91cfe9fb83c7894245931e2286`.

The existing intervals are conditional on fixed fitted models; they do not quantify training-seed variation, adaptive selection, or future domain shift. All three existing 95% intervals include zero. This analysis adds no uncertainty estimate and supplies no justification for tuning fold-dependent means/scales using these exposed outcomes. The original decisions remain unchanged.
