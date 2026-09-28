# Research iteration after the mixed-pair public result

## Decision

Keep `final_since2_gclp.zip` as the measured champion at **0.6622**. The user reported
`final_mixed_ema6_shared.zip` at **0.6616**. No new ZIP passed the predeclared
exact-mask, two-fold, and runtime gates in this iteration. No GPU training or
submission was run.

The current [organizer metric](https://wundernn.io/connectome/docs/data_overview) pools
all selected rows for each target, clips targets and predictions to ±2, weights by
clipped `|y|`, and averages the two correlations. The local `utils.py` block
metric is legacy; `src/wnn/gmetric.py` and the new audits use the official formula.
The [organizer FAQ](https://wundernn.io/connectome/docs/faq) says the test mask is
hidden and may reward different moments from the visible validation mask.

## What the cheap audits found

| Check | Exact-mask result | Decision |
| --- | --- | --- |
| Mixed public package versus submitted corrected pair, 368 reserved sequences | **−0.000018 ± 0.000776** chain bootstrap; at λ=.02 it is +0.000653 | The public −0.0006 is compatible with the direct local tie. Do not use λ=.02 as a public-score estimate. |
| Six-epoch seed 3 plus fine-tuned seed 4 | **+0.000489 ± 0.000493** on reserved sequences; pure-chain OOF proxy **−0.000384** fold 0, **+0.001373** fold 1 | Fails the positive-both-folds gate. |
| Both six-epoch seeds | **−0.000300 ± 0.001164** on reserved sequences | No promotion. |
| Class-mix and source-shift sensitivity | The mixed pair is −0.000049 at the full-validation class mix and +0.000411 at the train mix, both at λ=0. Its train-unlikeness tercile deltas have signs −,+,−. | A single holdout lift does not identify transfer across source compositions. |

The final-pair values come from [`scripts/audit_final_pairs.py`](../scripts/audit_final_pairs.py)
and [`final_pair_audit.json`](artifacts/final_pair_audit.json). That script reads
cached final predictions, computes official six-moment sufficient statistics,
reweights class proportions **before** pooled correlation, and keeps softened
masks as sensitivity checks. The exact observed-holdout mixed result reproduces
E-201. The fold proxy uses `ab62_base_s4` as its anchor and is not an exact
replica of the final fine-tuned pair.

The new shared native v3 package agrees with its batch blend on the cached
20-sequence identical-row check. A temporary v2 package that kept both GRUs
also scored **0.717577** on those rows, identical to the mixed v3 result, but
took **98.3 µs/row** in a one-vCPU scorer image: **64.5 minutes** projected for
39.4 million rows. Its maximum prediction difference from v3 was below
**4.8 × 10⁻⁷** on identical rows. See
[`cpu_fallback_audit.json`](artifacts/cpu_fallback_audit.json).
This exceeds the planned 50-minute safety gate and the
organizer's [60-minute limit](https://wundernn.io/connectome/docs/submission_guide).
The submitted mixed wrapper selects one GRU on v2. Public CPU flags remain
unknown; runtime alone does not establish that v2 ran publicly. The temporary
dual-v2 ZIP was left outside the repository and is not a submission candidate.

## Representation screens

The four-event class-3 target reconstruction disagrees with true labels on
**16.3%** of known scored development rows and **14.3%** of known scored
reserved rows in fixed 64-sequence samples. Those disagreements carry
**37.8% / 33.3%** of clipped target weight and **55.8% / 49.0%** of the old
pair's weighted absolute error, respectively. On the reserved sample, the
pair predicts the correct target side on 78.7% of target weight, yet those
rows still carry 62.1% of weighted absolute error. The largest disagreements
have the correct side but understate the winning tier: for example, true
tick targets `(2, −3)` versus reconstructed `(1, −3)`. The earlier
`research/panel/brainteaser_timing.md` already proposed a side × tier
factorization; this audit adds scorer-weight and error concentration evidence.

Replacing predictions on disagreement rows with true labels is an **oracle
diagnostic** and yields a large class-3 sample gain. It uses future labels.
Simple causal indicators poorly identify those rows on the reserved sample:
seed-disagreement AUC **0.514**, book-event-age AUC **0.542**, and prediction
magnitude AUC **0.563** after choosing its favorable direction. The previously
trained next-change auxiliary logit gives **−0.000020** pooled exact-mask score
as a cross-fitted residual readout. A cheap routing or logit correction has no
support. The class-3 reserved sample has now been used for exploratory
diagnostics and is not a pristine selection set for this mechanism. Raw results:
[`class3_law_audit_dev.json`](artifacts/class3_law_audit_dev.json) and
[`class3_law_audit.json`](artifacts/class3_law_audit.json).

For denoising or self-supervised representations, 112 sequence-level input
means show substantial source movement. Within class, the top-eight train/valid
covariance-subspace overlaps are **0.47–0.57**, below train-half overlaps
**0.55–0.91** and valid-half overlaps **0.75–0.87**. This is a descriptive
screen, not a proof an autoencoder will fail. The prior train/valid adversarial
classifier reached AUC about 0.99 within each class (E-094). See
[`domain_geometry.json`](artifacts/domain_geometry.json). The
[DRW Crypto winner](https://www.kaggle.com/competitions/drw-crypto-market-prediction/writeups/drw-solution-1st)
reported useful autoencoder features; the
[Jane Street winner](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)
warned about fitting an encoder across a validation split. These results keep
an encoder behind a fold-specific, train-only screen.

## Next bounded candidate

The best new modeling question is whether **tier supervision** can reduce the
large correct-side class-3 error without changing the two-GRU inference path.
Use a training-only auxiliary head for the winning target's clipped tick tier
and the already available side signal; discard the auxiliary head at inference.
This differs from the failed next-2/4-target-change auxiliaries (E-107).
Predeclare its weight and fixed epoch before training. Run one matched fold-0
fit, compare the *pair* to the closest corrected-pair OOF analogue at λ=0,
and stop unless the paired gain reaches **+0.0005**. Advance only if fold 1 is
also positive and pooled exact-mask gain is at least +0.0005; check λ=.02/.05,
both class-mix endpoints, and untouched folds 2/3 before final fitting.
Training remains deferred under the current no-heavy-experiment instruction.

Predictive-state pretraining on future *observable* book events remains a
separate idea from [contrastive predictive coding](https://arxiv.org/abs/1807.03748).
The existing sign head's negative residual result is only a partial falsifier;
it does not test future observation pretraining. Group DRO and risk
extrapolation need stable, meaningful source groups before fitting. The current
train-unlikeness terciles are diagnostic proxies, not known chronological periods.

All new audits were CPU-only and used cached predictions or small samples.
They preserve `PredictionModel.predict(DataPoint)` and the submitted artifacts.
