---
kind: review
status: complete
created_utc: 2026-10-20T22:31:00Z
updated_utc: 2026-10-20T22:31:00Z
hypothesis_ids: [H-NLL-GRADIENT-WEIGHTING-20261021-R1]
experiment_ids: [E-OVERNIGHT-NLL-ADAPTATION-R1]
evidence_paths:
  - research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/nll_epoch0_diagnostics_r1.json
---

# Variance weighting as a conditional NLL diagnostic

## Question or claim

Does learned residual variance change which observations receive useful mean
updates, and interact with global clipping in the running GRU comparison?
This is a possible explanation, not an observed quality failure.

## Mechanism and method

Seitzer, Tavakoli, Antić and Martius (ICLR2022) demonstrate poor mean fits under
Gaussian likelihood optimization and analyze inverse-variance gradient weighting.
Their beta-NLL attenuates this dependence. This is external evidence from other
tasks, not evidence of a GRU or competition gain.
[Primary paper, sections3–4](https://arxiv.org/html/2203.09168).

The authors' implementation multiplies each NLL contribution by detached
predicted variance raised to beta. Detachment is necessary to preserve their
gradient prescription. [Author implementation, pinned revision](https://github.com/martius-lab/beta-nll/blob/669c9f251eb41464c1ec8b43751c7c742dd75cf9/README.md#quick-implementation-of-beta-nll).
No external code was imported or executed.

For our loss, let `s=clamp(log_variance,-4,4)`, residual `r=mean-target`, and
`W=sum(weights)`. The current mean derivative is `2*w*exp(-s)*r/W`.
A detached `exp(beta*s)` multiplier changes this to
`2*w*exp((beta-1)*s)*r/W`. At beta0.5, the variance-dependent factor ranges
from0.1353to7.3891, versus0.0183to54.5982 at beta0. These are analytical
bounds from our clamp, not measured distributions. Keep the original weight
denominator and scalar convention when defining a future comparison.

## Evidence and interpretation

The [matched epoch0 receipt](../../campaign_2026-10-20/submission_delivery_r1/nll_adaptation/nll_epoch0_diagnostics_r1.json)
records100%NLL clipping with mean factor0.0418, versus0.9866 for WMSE. Its
aggregate head gradient includes mean, auxiliary and variance rows; it cannot
identify the responsible rows or establish a paper-specific failure mode.
Global clipping rescales all gradients, while AdamW's moments also respond to
that scaling. A small clip factor alone does not quantify the resulting
parameter updates or prove that recurrent learning is suppressed.

## Limits and validity

Our variance estimates describe residual scale for a teacher/label mixture;
they are not established estimates of irreducible market noise. At beta1,
the direct mean derivative matches WMSE, but shared-trunk variance gradients,
auxiliary losses and clipping can still change training.

## Decision and next step

Finish and grade the frozen beta0/WMSE pair first. If its quality and gradient
diagnostics justify a new branch, a single fixed beta0.5 treatment with exact
compatible controls could discriminate gradient weighting from the broader
heteroscedastic hypothesis. Its allocation and recipe would need explicit
freezing; no fit, implementation, sweep or PR is admitted by this source note.

## Source records

- [Current comparison](../../campaign_2026-10-20/submission_delivery_r1/nll_adaptation/README.md): optimization diagnostics, fixed endpoints and pending scores.
- Producer `68ff682977b822f38351aac0f621146f5f13b440`, `src/wnn/training/mean_variance.py`: exact current loss and clamp.
- Primary sources inspected20October2026UTC; paper version2203.09168v2, author-code revision669c9f251eb41464c1ec8b43751c7c742dd75cf9.
