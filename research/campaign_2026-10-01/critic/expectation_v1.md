# Public expectation audit v1: SP2, SP4, SP2+D (independent critic, 2026-10-01 ~05:10)

I did not author any of these candidates. Everything below runs on CPU and is reproducible:

- `expectation_bridge.py` -> `expectation_bridge.out`: historical holdout-to-public paired bridge.
- `sp2_pub143_f01.json` -> `sp2_pub143_f01_report.json`: SP2 analog on folds 0+1 under the public-like class mix.
- `expectation_mc.py` -> `expectation_mc.out`: predictive Monte Carlo and the f2/f3 update table.
- `expectation_sens.py` -> `expectation_sens.out`: sensitivity grid.

The intervals are **model-based scientific forecasts under the assumptions stated below. They are not calibrated
intervals** (per PRE_SCORE_PROTOCOL). Audit decision: **conditional expectation only. No candidate has a credible mean
>= 0.6641, and none has a non-negligible chance of reaching 0.6660.**

## 1. Bottom line

| candidate | expected public | 80% predictive interval | P(> 0.6622) | P(> 0.6641) | P(>= 0.6660) |
|---|---:|---|---:|---:|---:|
| **SP2 now** (pre f2/f3) | **0.6626** | [0.6617, 0.6635] | 0.71 | 0.02 | < 0.001 |
| SP2 after f2/f3 | 0.6623 + 0.375 m (section 5) | about mean ± 0.0009 | 0.57-0.74 | 0.005-0.04 | < 0.001 |
| SP4 (+ seeds 5, 6 trunks) | 0.6632 | [0.6621, 0.6642] | 0.88 | 0.12 | 0.001 |
| SP2+D (+ distilled GRU256) | 0.6628 | [0.6618, 0.6639] | 0.76 | 0.06 | < 0.001 |
| SP4+D (illustrative; likely over the runtime budget) | 0.6634 | [0.6623, 0.6645] | 0.91 | 0.19 | 0.002 |

Readings:

- SP2 is a small, probably positive step: about +0.0004 expected, and a 29% chance it shows ≤ 0.6622.
- Only SP4 (12%) and SP4+D (19%) give P(> 0.6641) a non-trivial value.
- P(≥ 0.6660) is not meaningfully nonzero for any candidate. It needs about +0.0038 public, roughly 5 times the largest
  selection-corrected public-like local gain on disk.
- Across every assumption set that keeps a selection correction (section 4), the SP2 mean stays in 0.6624-0.6628 and
  P(> 0.6641) stays ≤ 0.04.

## 2. Evidence used (exact rows, sources)

**SP2 identity.** Trunks `ab67_final_since_s{3,4}`. Each trunk's head is an exact output average of the parent head
(ep3) and the head-only gcorr FT head (`_gcorrh`); this is a shared trunk and equals the 4-member output mix.

- I verified `ab67_final_since_s{3,4}/model.pt` == `model_ep3.pt` tensor-for-tensor. The legacy best-on-holdout selection
  picked the last epoch, so the parents in the holdout screens are the SP2 parents.
- The head-only FT was fitted on dev valid only (holdout reserved).

| quantity | value ± component-bootstrap SD | source |
|---|---|---|
| SP2 analog minus FT pair, folds 0+1, all rows | +0.000983 ± 0.000623 (f0 +0.00073, f1 +0.00110) | pool_shared_f01_report |
| same, public-like mix pub143 (c3 = 0.459) | **+0.001326 ± 0.000714** | critic/sp2_pub143_f01_report |
| decomposition (pub143) | head-only vs two-stage +0.00086 ± 0.00051; + parent heads +0.00047 ± 0.00029 | same |
| SP2 minus incumbent, 368 holdout, final artifacts, all rows | **+0.000379 ± 0.000550** | holdout_trunks_report |
| same, pub143 weighting | +0.000711 ± 0.000638 | holdout_trunks_report |
| FT pair minus single member (1 -> 2 trunks), folds 0+1, pub143 | +0.00139 ± 0.00047 | critic/sp2_pub143_f01_report |
| pair + d256 head (thirds) minus pair, fold 0 | +0.00043 ± 0.00054 | d256_f0_report |
| 4-member mix + d256 (merged) minus 4-member mix, fold 0 | +0.00020..+0.00026 ± 0.00032..0.00036 | combo_f0_report |
| diversity trunks added to SP2 (c500, onorm), holdout | -0.00004, +0.00001 | holdout_trunks_report |

## 3. Model

`public = 0.6622 + U(±0.00005) + Delta_public`, with `Delta_public` drawn 50/50 from two routes. Neither bridge has been
validated for this mechanism, so I don't pick one.

**Route H: holdout bridge on actual final artifacts.**
- `Delta_public = b * 0.000379 + e`.
- The fit uses the 6 adjacent GRU package steps actually submitted: s4 -> s4ft -> s4lpft -> ema4 -> since2, plus
  since2 -> mixed-EMA6 and since2 -> EMA6.
- Through-origin slope b = 0.657 (jackknife SD 0.24), residual SD 0.00064.
- The residual already contains each step's holdout sampling noise and the public paired noise, so no separate LB term
  is added.
- I excluded the baseline-blend step. The organizer baseline is known to do better publicly than any local proxy
  (public +0.0021 on holdout -0.0007); keeping it drops b to 0.48 and inflates the residual to 0.0012.
- The single-path EMA6 scenario (protocol: keep both) gives b = 0.51, residual 0.00054; the mean changes by < 0.0001.
- Step list and residuals: `expectation_bridge.out`.
- Route H mean: 0.66245.

**Route F: clean folds 0+1, public-like weighting.**
- `Delta_public = tau * theta + N(0, 0.0006)`, where 0.0006 is the stated public paired noise.
- Selection correction: SP2's mechanism was found post hoc on folds 0/1 (and holdout), so theta is the empirical-Bayes
  posterior of the pub143 local delta under prior N(0.0003, 0.0006²).
  - The prior is the spread of tonight's cheap ensemble/FT tweaks E-287..E-295. Method-of-moments true-effect SD is
    about 0.0005; I widened it slightly, and set a slightly positive mean because mixing two near-equal predictors is
    rarely harmful.
  - Shrinkage factor 0.41: theta = 0.00072 ± 0.00046 instead of the raw 0.00133. This is the selection correction. It
    plays the role of the protocol's plug-in optimism without needing the full candidate-replay bootstrap, which tonight's
    ad-hoc screens do not support.
- Transfer slope tau ~ N(0.75, 0.35):
  - S2's point estimate is 0.8 (range 0.63-0.92), but that was fitted on level differences.
  - Historical fold-to-public outcomes for small deltas range from about 2.8 (s4 -> s4ft: folds +0.0005, public +0.0014)
    to below 0 (mixed EMA6: pure-fold +0.00114, public -0.0006).
  - The SD therefore covers a negative-transfer tail (P(tau < 0) ≈ 1.6%).
- Route F mean: 0.66274.

**What makes SP2 different from mixed EMA6.** EMA6 also passed folds and failed publicly, but its holdout delta was
already about 0 (-0.00002). SP2's holdout delta is positive on the final artifacts. It is also consistent in sign on
fold 0, fold 1 and holdout, and its class-3 gain (+0.0041 on folds 0/1, +0.0028 on holdout) points toward the
public-heavier class.

## 4. Sensitivity (SP2 now; `expectation_sens.out`)

| assumption changed | mean | 80% PI | P(>0.6641) |
|---|---:|---|---:|
| central (above) | 0.6626 | [0.6617, 0.6635] | 0.023 |
| EMA6 single-path scenario | 0.6626 | [0.6617, 0.6634] | 0.021 |
| tau ~ N(0.8, 0.12) (S2 narrow) | 0.6626 | [0.6617, 0.6635] | 0.019 |
| tau ~ N(0.5, 0.3) | 0.6625 | [0.6617, 0.6634] | 0.011 |
| prior N(0, 0.0005) (strict) | 0.6625 | [0.6616, 0.6634] | 0.011 |
| route H only | 0.66245 | [0.6616, 0.6633] | 0.005 |
| route F only | 0.6627 | [0.6618, 0.6637] | 0.041 |
| **no selection correction** (raw folds, protocol-prohibited) | 0.6628 | [0.6618, 0.6640] | 0.086 |
| no selection correction, folds only, tau 0.8 | 0.6633 | [0.6622, 0.6644] | 0.16 |

The only way to make P(> 0.6641) look meaningful for SP2 is to drop the selection correction and the holdout evidence
together. That is exactly the shortcut the protocol prohibits.

## 5. Update after the preregistered f2/f3 confirmation

The confirmation report gives the pooled f2+f3 delta of the `shared_parent_head` arm versus `ft_pair`. Re-score it with
`"class_mix": {"pub143": [0.1832, 0.1597, 0.1984, 0.4588]}` to get o23 and its SD s23 (expected about 0.0007). Then:

- `1/v = 1/0.0006² + kappa/0.000714² + 1/s23²`
- `m = v * (0.0003/0.0006² + kappa * 0.001326/0.000714² + o23/s23²)`
- `E[public SP2] = 0.6622 + 0.5 * 0.657 * 0.000379 + 0.5 * 0.75 * m = 0.66232 + 0.375 m`
- 80% PI ≈ E ± 0.0009.

Use kappa = 0 as primary: confirmation only, per protocol step 3, because folds 0/1 were the discovery data. Report
kappa = 1 as the sensitivity case. Precomputed rows (`expectation_mc.out`, s23 = 0.0007):

| o23 (pub143) | m (kappa 0) | E public (kappa 0 / 1) | P(>0.6641) (kappa 0 / 1) |
|---:|---:|---|---|
| -0.0005 | -0.00004 | 0.6623 / 0.6625 | 0.005 / 0.008 |
| 0.0000 | 0.00017 | 0.6624 / 0.6625 | 0.007 / 0.012 |
| +0.0005 | 0.00039 | 0.6625 / 0.6626 | 0.010 / 0.017 |
| +0.0010 | 0.00060 | 0.66255 / 0.6626 | 0.017 / 0.025 |
| +0.0015 | 0.00081 | 0.6626 / 0.6627 | 0.028 / 0.035 |
| +0.0020 | 0.00102 | 0.6627 / 0.6627 | 0.043 / 0.047 |

**Gate reading.** If the prereg gate fails (pooled < +0.0005 or either fold ≤ 0), SP2's expected gain falls to about
+0.0002. That is a third of the public paired noise and essentially a coin flip against the incumbent. Even a strong
confirmation moves the mean only to about 0.6627, because the holdout route (+0.00025) is half the forecast and does
not change. If the team wants route H to update too, the only new holdout information would come from new final
artifacts (SP4, SP2+D).

## 6. SP4 and SP2+D: method and how to update

**SP4.**
- `SP4 = SP2 + tau_s * g4 + N(0, 0.0003)`.
- The local increment comes from the ensemble-size law: gain(2 -> 4 trunks) ≈ ½ gain(1 -> 2) = ½ × 0.00139 ≈ +0.0007
  (SD 0.00025). Teammates' seed curve (+0.00074 then +0.00037 for 2 -> 4) shows the same ½ ratio.
- tau_s ~ N(0.8, 0.25). Same-recipe seed averaging is the most transfer-robust lever: pure variance reduction with no
  new inductive bias. Even so, no public point measures it for this family.
- Not selected post hoc, so no shrinkage.
- The +0.0003 extra noise covers the larger prediction difference to the incumbent.
- **Update when seeds 5/6 land:**
  - Score SP4 - SP2 on the 368 holdout and map it with route H (×0.657, residual 0.00064).
  - Average 50/50 with the size-law prior above.
  - If the holdout increment is ≤ 0 (as for the c500/onorm diversity trunks), treat the size law as falsified for these
    seeds and revert to SP2's numbers.
  - Also check that both new trunks individually match s3/s4 on holdout (±0.001). A weak seed breaks the size law.

**SP2+D.**
- `SP2+D = SP2 + tau * theta_D + N(0, 0.0003)`.
- theta_D is the EB posterior of the fold-0 increment over the 4-member mix (+0.00025 ± 0.00035 -> 0.00026 ± 0.00030).
- **Update:**
  - When d256 fold 1 lands, add it as another likelihood term in the same normal formula: prior N(0.0003, 0.0006²),
    each fold weighted by 1/SD².
  - When the final d256 trunk lands, score SP2+D - SP2 on holdout and average route H 50/50 with route F, as for SP2.
  - Use public-like (pub143) deltas. The S2 tool shows d256's class-3 gain and class-0 loss cancel under pub143, so
    all-rows ≈ pub143 here.
- **Runtime:** 2×h128 + h256 is projected at 51.5 min against a 52-min gate. A timeout or failure is a public score of 0
  (or a void), and this forecast does not price that risk. Do not submit SP2+D without a measured margin. SP4 (about
  39 min projected) is safe; SP4+D (≈ 70 min: 4×h128 ≈ 38.6 + one h256 ≈ 31.8) is not.

**Teammate anchors (0.6639-0.6641).** Their GRU256 + 2×GRU128 head-FT/distilled family sits about +0.0018 over our
incumbent. Their own baseline public score is unknown, so it cannot calibrate our tau. It is an existence proof that
GRU256-containing bags can reach 0.664. Our local evidence for d256 (+0.0002..0.0004 over the 4-member mix, fold 0)
does not reproduce that size, so I keep it as an unpriced upside scenario. If fold 1 and the final holdout both show
≥ +0.0008 for D, revisit.

## 7. Caveats that are not in the numbers

- Holdout panel: 368 sequences inspected many times, with dev-valid neighbour exposure. The historical bridge is fitted
  on 6 dependent steps from at most 3 lineages, so the effective n is about 3-5.
- Route-F prior and tau are declared scientific assumptions, not identified quantities. The 50/50 route weight is a
  choice; the sensitivity rows show it moves the SP2 mean by ±0.00015.
- Public scores are displayed to 4 decimals. "> 0.6641" means a displayed 0.6642 or higher; ties at display precision
  are not counted as wins.
- Package/engine parity (int16, |ΔWP| ≤ 4e-7) and the merged-head identity (max diff 1e-6) are taken from the campaign
  report and were not re-verified by me. They are negligible at this scale if true.
