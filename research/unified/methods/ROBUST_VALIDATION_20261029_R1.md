---
kind: method
status: proposed
created_utc: 2026-10-29T12:30:00Z
updated_utc: 2026-10-29T12:30:00Z
hypothesis_ids: [H-VALSHIFT29-R1]
experiment_ids: [E-VALSHIFT29-CHAINS-R1, E-VALSHIFT29-ADV-R1]
evidence_paths:
  - research/campaign_2026-10-29/validation_r1/README.md
  - research/campaign_2026-10-29/validation_r1/adversarial_validation.json
  - specs/adr/0044-chronological-primary-validation.md (PR49 branch)
---

# Robust validation for public transfer and out-of-sample stability (October 29)

User: "I think validation is issue. think about most robust way for models validation / cross
validation, so it may help for public lb and for long term stability in out of sample."

## 0. Prior evidence (found after the re-run; credit and correction)

Today's results **replicate a month-old finding**; they are not new:
- **E-094 (2026-09-26)**: adversarial validation train vs valid within each class, AUC
  0.989–0.993; drivers i0/i1 volume-code means and a-column stds (scale/regime). Claim then:
  "test likely differs again, so robustness to per-sequence scale/regime shift matters more than
  fit on validation."
- **E-161**: 90 train↔valid chain links vs ≈2,300 expected under random split; "train vs valid =
  different time periods", likely four instruments (one per class).
- **E-110**: a **stress panel** — the two thirds of valid sequences least train-like by OOF
  adversarial score (`cache/adv_valid_score.npy`) — ordered all three then-known LB comparisons
  correctly where plain valid misordered one; adopted as primary selection proxy on 2026-09-26.
  **Correction:** it was demoted the same day for a recorded reason — forensic review W3
  (`research/panel/forensic.md`): 3 LB constraints, 2 knobs, terciles picked post hoc, one
  constraint a −1.4σ noise point, worse out of sample (forecasts off by 0.0013–0.0018). The
  2026-09-26 plan kept full valid as primary and stress λ0.05 only as a guard cell. The same plan
  refuted domain importance weights and train filtering (AUC 0.99 leaves n_eff ≈ 1e-5 of train).
New today: the train-only (ADR0044) context, and that the 90 boundary links run in both
directions (48/42), which argues against one later period.

## 1. What we measured today (zero-fit)

| Evidence | Result | Meaning |
|---|---|---|
| Recovered time links (tail→head, feature-matched; PR49 builder) | 9,193 links: train→train 7,808, valid→valid 1,295, **train→valid 48, valid→train 42** | Cross-split links are 1% of links; a random per-sequence split would give ≈25% (2·0.15·0.85). Valid is made of **whole blocks** (493 pure-valid chains, 85 mixed). Boundaries run both ways, so valid is not one later period (or some links are false). |
| Adversarial validation (sequence summaries of 28 raw columns, component-grouped 5-fold) | **OOF AUC 0.984** (label shuffle 0.499) | Valid is an almost perfectly separable population: a different regime/period/series set, not a random sample of train. |
| Historical random component folds vs public | local→public offsets −0.006 to −0.010; local blend gains (+0.0023) not transferring (H-382/383/385) | Folds that mix train and valid regimes measure in-regime interpolation and overstate transfer. |
| Fresh reproduction of PR49 fin256_s3 under the official split | per-epoch full-valid WP within 0.0002 of PR49 | The official-split protocol is reproducible across machines. |

Interpretation: the competition hold-outs (valid, public, private) are **regime-shifted blocks**.
Validation must hold out whole regimes and measure how much a model's advantage survives the
shift. Random or component folds over the pooled data cannot see this.

## 2. Recommended protocol

1. **Unit of splitting = regime block, never a sequence.** Keep the recovered chain components
   whole and the 282 valid-connected train components out of fitting (ADR0044 split already does).
2. **Outer holdout = official valid** (shifted, 1,873 sequences). ADR0044's conclusion stands, but
   its stated reason ("train precedes valid") is contradicted by the two-way links; the robust
   reason is the measured shift. Use it for **confirmation only**: pre-declared rule, one look per
   frozen candidate, every look logged.
3. **Inner selection = leave-regime-out CV inside train.** Cluster train components into K≈5
   regime groups (adversarial/feature-summary clusters, chain components kept whole), fit on K−1,
   evaluate on the held-out group. All hyperparameters, blend weights, epochs and distillation
   settings are chosen here, not on valid. This gives several shifted folds instead of one.
4. **Stress panel as a guard cell, not co-primary.** Report the least-train-like two thirds
   (OOF adversarial score) beside full valid; investigate a sign flip, but do not select on it
   (forensic W3: it was fitted to 3 LB points and forecast worse out of sample). Re-test it only
   with the public calibration submissions below. Do not use adversarial importance weights or
   train filtering for fitting (refuted 2026-09-26: AUC ≈0.99 leaves n_eff ≈ 1e-5).
5. **Choose for stability, not only the mean.** Prefer the candidate whose gain is positive on
   most regime folds (sign consistency, worst fold), with fewer tuned degrees of freedom (equal
   or fixed blend weights, fixed epochs, seed averages). Under shift these transfer better.
6. **Seeds.** At any adoption decision, ≥3 matched seeds per arm; report mean, paired seed SD and
   the component bootstrap. The +0.001 threshold must exceed ≈2 combined SE.
7. **Public LB = one more shifted fold, used as an instrument.** Submissions are designed
   experiments with pre-declared expected differences, not hill-climbing: (a) a valid→public
   transfer slope from 2–3 train-only models spanning a known valid range; (b) the value of the
   final refit (same recipe, train-only vs train+valid). Never pick blend weights from LB.
8. **Final model = refit on train+valid** with the selected fixed recipe and several seeds:
   more regimes in training is the most direct defence against the next shift.

## 3. Immediate next steps

- Driver analysis of the adversarial shift (which features separate valid: level/scale drift vs
  structure) — rerun with saved features is running.
- Build the regime-cluster inner folds (zero-fit) and check that a leave-regime-out score of an
  existing train-only model sits closer to its official-valid score than a random-fold score does.
- Public calibration (user offered quota): train-only parent P (valid 0.6867), labels-only child
  C0 (0.6756) and the reproduced fbc1k teacher (expected ≈0.691) as single-GRU ZIPs give a
  three-point valid→public slope with an 0.015 valid range, larger than LB noise.

## Questions to keep asking

- Does a candidate's advantage survive every held-out regime, or come from one?
- How many times has valid been looked at for this decision?
- Is the gain larger than seed noise *and* shift noise?
- Would the decision change if the worst regime fold were the test?
