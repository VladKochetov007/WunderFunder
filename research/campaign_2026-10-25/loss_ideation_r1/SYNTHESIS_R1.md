---
kind: review
status: complete
created_utc: 2026-10-25T00:40:00Z
updated_utc: 2026-10-25T00:40:00Z
hypothesis_ids: [H-LOSS25-UV-AUX-R1, H-LOSS25-EVENT-HAZARD-R1, H-LOSS25-FAITHFUL-NLL-R1, H-LOSS25-RUNAGE-MEASURE-R1, H-LOSS25-PR32-BLEND-PROBE-R1]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-25/loss_ideation_r1/zero_fit_target_mass_valid_R1.json
  - research/campaign_2026-10-25/loss_ideation_r1/external/
  - research/campaign_2026-10-25/loss_ideation_r1/INVOCATIONS_R1.json
---

# Loss/objective review, PR updates and external red team — 2026-10-25

## Question

User request: read current research state and PR updates (gh), read our custom-loss
research, explain why it succeeded or failed, and propose further ideas/approaches to
improve public score; external OpenCode/Grok/Freebuff ideators and red teamers allowed.
Tier C (literature/document review) plus one zero-fit tier-B data measurement.
Zero model fits, zero uploads.

## What changed in the PRs since the October 24 close

- **PR32/PR40 merged 2026-10-25 00:10 UTC.** Teammate evidence branch
  `experiment/pr32-b300-evidence-20261024` (Semyon Epanov, head `170fd485b`) posts
  hourly held-out diagnostics for the four B300 causal Transformer teachers:

  | Cell | Epochs seen | Train WMSE (online) | Val WMSE | Val official WP |
  |---|---|---:|---:|---:|
  | t1280_f0 (630M) | 28→30 (final) | 0.0135→0.0105 | 1.550→1.568 | 0.173→**0.157** |
  | t1280_f2 (630M) | 28→30 (final) | 0.0132→0.0103 | 1.598→1.597 | 0.168→**0.169** |
  | t1792_f0 (1.54B) | 15→20 | 0.146→0.068 | ~1.51–1.52 | 0.372→0.385→0.306 |
  | t1792_f2 (1.54B) | 15→19 | — | 1.107→1.154 | 0.414→0.309 |

  Earlier epoch checkpoints were overwritten. These outer folds are exposed; scores
  are not used for selection. Our GRU endpoints are ~0.68 on the same kind of fold.
- **PR37 (cold-label NLL vs WMSE, 8×B300)** never produced a graded pair: paused
  2026-10-22 when the node was shared; four H256 epoch-0 states are saved.
- No new comments on PR39/PR41.

**Interpretation (zero-fit check, this note).** Scored-valid weighted target variance
is **1.79 (t0) / 1.82 (t1)**; a zero predictor scores WMSE 1.94
([receipt](zero_fit_target_mass_valid_R1.json)). So B300 val WMSE ≈1.55 is a weighted
R² of only ~0.14, versus ~0.97 expected for a WP-0.68 model, while train WMSE 0.01 is
R²≈0.995. This is memorization, not a pipeline anomaly, and the 1.54B net is simply
earlier on the same curve (val WP falling every epoch). The PR32 "+0.005 lever" from
the October24 value review is **not available from these checkpoints as teachers**;
the recipe (LR3e-4, 30 epochs, no early stop/regularization, overwritten epochs) is the
failure, not necessarily the Transformer family.

## Why our loss research succeeded or failed

Bayes alignment: with w=|clip(y)|, pooled-WP's population optimum is an affine map of
m(x)=E[|y|y|x]/E[|y||x], which is exactly the |y|-weighted-MSE minimizer. Hence:

1. **Distance-function changes failed because they cannot move the target.** Gaussian
   NLL (−0.00004 matched; earlier warm +0.0007 crossing zero), per-chunk Pearson
   (H-018 refuted), Huber, code-softmax readout, DIST/Pearson KD and channel-weighted
   KD (REV-7 null; sum0 −0.0016) all target the same m(x) or worse. At distillation
   a=0.9 the target is 90% teacher, so student-side distance tweaks touch only ~10% of
   the signal. All four external reviewers agree on this reading.
2. **But the NLL test was confounded, not negative.** 100% of NLL updates were clipped
   (mean factor 0.04 vs 0.99 for WMSE), so the mean path received ~4% steps. Primary
   literature predicts exactly this pathology and its fix: Seitzer et al. β-NLL
   ([2203.09168](https://arxiv.org/abs/2203.09168)) and Stirn et al. *Faithful
   Heteroscedastic Regression* ([2212.09184](https://arxiv.org/abs/2212.09184)), which
   keeps the mean on the MSE gradient and stops variance gradients from distorting the
   shared trunk. Grok's claim that our NLL targets unweighted E[y|x] is rejected: the
   [gradient note](../../unified/sources/NLL_GRADIENT_WEIGHTING_20261021_R1.md) records
   the mean derivative as `2*w*exp(-s)*r/W` with the same |y| row weights; the learned
   variance then multiplies them by exp(−s), which *does* tilt the finite-sample fit.
3. **Measure changes succeeded.** The reproducible teacher gains are row-weighting
   (propensity, label-weight floor, class-3 weight; H-360, H-007/H-023/H-029).
   Importance weighting keeps its effect only when the model does not interpolate the
   training set (Byrd & Lipton [1812.03372](https://arxiv.org/abs/1812.03372); Xu et al.
   [2103.15209](https://arxiv.org/abs/2103.15209)); our GRUs are far from interpolation,
   so weighting acts as capacity allocation. That is the live family.
4. **Distillation succeeded as variance reduction** of step-function labels from
   heterogeneous teachers; it fails when the teacher is weaker than the student (PR32).
5. **The residual is structural, and no loss we tested addresses it** (from the
   October11 residual EDA): 94.4% of residual SS is between label runs; correlation
   decays 0.816→0.599 with rows since the label changed; labels step only on book
   events; the label leads features by ~1 row (+0.009 k=+1 ceiling); a non-causal
   run-mean oracle gives +0.031; difference mode u=(y0−y1)/2 has prediction variance
   only 35% of label variance (corr 0.634) while v is matched; class 3 (component
   attribute, purity 1.0) holds ~46–50% of weight at corr 0.44.

## Verified external red-team claims (accepted / rejected)

Outputs: [Grok 4.7](external/grok_4.7_R1.md), [DeepSeek v4.1 flash](external/oc_deepseek_R1.md),
[GLM 5.3 flash](external/oc_glm_R1.md), [Nemotron 3 ultra](external/oc_nemotron_R1.md).
Freebuff has no verified headless interface (CLAUDE.md) and was not invoked.

| Claim | Source | Verdict |
|---|---|---|
| Per-target loss denominators fix a gradient imbalance | DeepSeek, GLM, Grok, Nemotron | **Rejected**: measured t0 mass share 0.496 — already balanced |
| B300 val WMSE 1.55 is worse than predicting zero / implies a leak | Grok, DeepSeek | **Rejected**: zero predictor = 1.94; R²≈0.14. Memorization suffices |
| B300 is protocol failure; keep every epoch, select on val WP, 30–200M params, dropout/WD, LR ≤1e-4 | all four | **Accepted** |
| NLL result is an optimization incident, not evidence against heteroscedastic heads | all four | **Accepted** (with literature above) |
| Never distill (a=0.9) from a teacher weaker than the student | all four | **Accepted** |
| Oracle 0.968 branch readout is label-derived, not a causal bound | Grok | **Accepted**; causal class-3 branch AUC was ~0.74 |
| Local 0.682 vs public 0.668 is a 0.013 transfer failure | Grok, DeepSeek, GLM | **Rejected as stated**: different data/class mix (valid overstated test by ~0.022 in Sept); it is a level shift, ranking transfer is the open question |
| Loss uses unclipped y; mismatch explains tail gap | Nemotron | **Rejected**: losses clamp to ±2 |
| tanh output saturation at ±2 | DeepSeek | **Low value**: only 3% of rows hit the clip; predictions are under-dispersed (slope 1.10–1.15) |
| Stop sub-0.001 screens without ≥2 folds × ≥2 seeds | all four | **Accepted** as policy reminder (consistent with value review) |

## Ranked ideas (new or never tested)

Ordering = expected decision value per GPU-hour, structural first. All initial screens
follow the 4–6-fit policy (matched F0/F2 × control/candidate unless noted).

1. **Event-hazard auxiliary head (H-362 → H-LOSS25-EVENT-HAZARD-R1).** Target the
   dominant residual (label-run level and step timing). Aux heads: P(label changes at
   this book event) and post-change level/sign, train-only labels, causal inputs.
   *Zero-fit gate first:* logistic/GBM probe of change-at-book-event from causal
   features/hidden state on held-out sequences; AUC <0.60 kills it. Then 4 fits.
2. **Difference-mode auxiliary loss (H-365 → H-LOSS25-UV-AUX-R1).** Keep per-target
   |y| WMSE as main term (preserves the Bayes target); add λ·weighted (û−u)²/var_u with
   u=(y0−y1)/2, λ∈{0.1,0.3}. Addresses u under-dispersion (35% of label variance) that
   post-hoc rescaling could not fix — i.e. a representation, not calibration, deficit.
   Not the failed sum0 KD arm (that removed v). 4 fits.
   **Self-correction (loop tick 1):** a Bayes-shrunk predictor with u-corr 0.634 should
   carry var ratio ρ²≈0.40; observed 0.35 is close, so "u under-dispersion" is mostly
   expected shrinkage. The anomaly is v (pred variance 0.171 ≈ label 0.164 at corr
   0.599, i.e. ~3× the shrunk expectation), yet shrinking v hurt post hoc — |y|-weighted
   per-target Pearson does not separate cleanly into modes. **Downgraded below idea 3.**
   The existing `--aux-sign` head already supervises the u sign.
3. **Rows-since-book-event / run-age measure weighting (H-LOSS25-RUNAGE-MEASURE-R1).**
   Long runs hold 56% of weight and 72% of residual SS. Weight by a *causal* x-function
   (rows since last book event), so the population target is unchanged and only
   capacity moves — the one family with reproducible gains. 4 fits, mean-only 2 arms.
4. **PR32 checkpoint blend probe (zero fits, H-LOSS25-PR32-BLEND-PROBE-R1).** Ask the
   teammate for F0/F2 predictions of t1792 epoch15/16 and t1280 final; score fixed
   small weights (0.05/0.10) against our endpoint on the same folds. A 0.41 model with
   low residual correlation can still add blend value; costs only predictions.
5. **PR32 protocol redesign (teammate handoff).** Per-epoch checkpoints kept;
   val-WP selection on an inner fold; 30–200M params; dropout 0.1–0.3, WD 0.1; LR
   1e-4 cosine; epochs 1–5 evaluated first; teacher admitted only above student WP;
   sequence-offset augmentation. Gate: no 30-epoch spend before an epoch-1..5 curve.
6. **Faithful/β-NLL rerun (H-LOSS25-FAITHFUL-NLL-R1).** Detached variance (Stirn) or β=0.5, per-group clipping, admission gate
   clip fraction <5%. Expected mean gain ≈0; the value is a per-row reliability signal
   for blend gating. Two fits on PR37's saved H256 states would be cheapest.
7. **Batch-mass EMA denominator (C4, coded, ungraded).** Cheap (4 fits); under AdamW a
   0.1–9.7× per-step mass swing acts as per-step LR noise. Small expected effect.
8. **Non-recurrent diversity member** (from the value review; GBM/ridge on causal
   window summaries): pool is 100% GRU-family; blend gain tracks 1−residual corr
   (r=−0.92). Independent of loss work.

Explicitly deprioritized: per-target normalization (measured no-op), more KD distance
functions at a=0.9, per-chunk/batch Pearson, Muon, output saturation, focal weighting,
same-sequence mixup (breaks the step-function label structure), gradient surgery.

## Limits

Agreement among four external models is same-brief agreement, not independent
measurement; the brief itself framed the evidence. The weighted-variance receipt is on
valid rows; train class mix differs. Class-3 causal inferability remains unverified. No idea here has a measured gain.

## Next step

Run the zero-fit gates (idea1 probe, idea4 teammate predictions),
then admit one 4-fit screen for idea 2 or 3 on the local GPU within the user's
5 GB VRAM / 10-core / 30 GB envelope, after user confirmation of a fit allocation.

## Loop tick 1 update (2026-10-25 ~03:40 Kyiv)

- **Idea 1 gate ran and is uninformative as specified.**
  [Probe](event_change_probe.py) → [receipt](event_change_probe_R1.json): HistGBM on 400
  valid sequences (1.52M train / 0.74M test i0 book-event rows, 85 held-out chains).
  Label change at a book event is predicted with **AUC 1.000** in every class, and the
  difference-mode direction given a change with AUC 0.994. This is quote arithmetic,
  not hidden information: t0=F−ask and t1=bid−F move mechanically when the quotes move.
  So the "label changes only on book events" observation does not identify a learnable
  hazard. The informative quantity is the move of F *relative to* the quotes, i.e. the
  ordinary near-future price-direction problem, which the GRU already targets.
  H-LOSS25-EVENT-HAZARD-R1 (and H-362 as phrased) → `screen_failed_trivial_target`.
  A meaningful successor needs a target that excludes the mechanical quote update
  (e.g. ΔF in ticks for class 3 via the exact target law). Not admitted.
- **Idea 3 is a duplicate.** `main_row_weights` already has `--since-weight`, a causal
  rows-since-i0-price-event weight (1/(1+since/s), floored) used in the public
  since/since lineage — and it *down*-weights long runs. The proposed up-weighting is
  the opposite direction of an existing arm, so it is not a new mechanism. Treat it as an
  optional sign-flip ablation of `--since-weight`, not a new card.
- Revised ranking: PR32 blend probe (teammate predictions requested in
  [PR32 comment](https://github.com/VladKochetov007/WunderFunder/pull/32#issuecomment-6391381202)),
  then the coded C4 batch-mass EMA screen, then faithful/β-NLL; idea 2 stays downgraded.
  The loss family offers no remaining candidate above the 0.001 noise floor on current
  evidence. The structural bottleneck is the F-direction information itself.
