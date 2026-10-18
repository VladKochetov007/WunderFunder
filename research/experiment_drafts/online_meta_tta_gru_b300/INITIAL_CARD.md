# Initial hypothesis card — meta-trained input-only online adaptation

**Hypothesis ID:** `H-ONLINE-META-TTA-GRU-B300-20261018-R1`  
**Experiment ID:** `E-ONLINE-META-TTA-GRU-B300-20261018-R1`  
**Status:** source-grounded proposal; no data screen, fit, or profile has run.  
**Lineage:** teacher/student connection; the adaptive model is a training-time
teacher, with a conventional GRU student as the required competition endpoint.

**Post-review clarification (18 October 2026):** the initial sentence below
calling the GRU encoder “frozen” was inaccurate for the proposed meta-training
loop. Preserve this first card as written; the revised packet treats the GRU,
task head, adapter initializer, and decoder as outer-trainable, with only the
per-sequence adapter weights changed by the inner update. See the
[implementation review](../../unified/reviews/ONLINE_META_TTA_IMPLEMENTATION_REVIEW_20261018_R1.md)
and current [experiment packet](README.md).

## Mechanism

The callback receives a long, ordered input stream but never receives target
labels or the scoring mask. Ordinary target-error online learning is therefore
not available. Instead, test a causal self-supervised update policy: a frozen
GRU encoder supplies recurrent states; a small low-rank fast adapter changes
the readout representation; and an auxiliary decoder predicts the next
observed raw 112-feature state from the preceding recurrent state. Once a
support block has arrived, one gradient update of only the fast adapter uses
that block's already-observed input-prediction loss. The updated adapter
predicts the following query block. It resets at each sequence and carries
between later blocks within that sequence.

Training is episodic. A supervised outer loss on the query block trains the
initial fast adapter, the self-supervised decoder, and optionally the recurrent
encoder through the declared inner update. The adaptation objective sees only
input features. Task targets appear only in the outer training loss on fitting
sequences; they are absent from inference adaptation, evaluation-time fitting,
and any test path. The final endpoint must distill the adapted teacher into the
ordinary causal GRU unless direct CPU-serving adaptation earns a separate,
measured runtime path.

## Why test it

The competition exposes 20,000 ordered inputs per independent sequence and
allows only current/past rows, with 99 warm-up rows before the first required
prediction. This is a meaningful online context budget. Existing studies show
that self-supervised test-time updates can help when the inner and outer task
gradients align, and long-context test-time training explicitly meta-learns an
initialization for continued self-supervised updates. Time-series TTA studies
show that causal adaptation is feasible but can be highly regime-dependent.
Their effects do not transfer automatically to this task.

The key observable is not improved input reconstruction. It is improved
official WP on the next query block after adaptation, followed by an improved
fold-clean student endpoint. Measure input/task gradient cosine on fitting
episodes as a mechanism diagnostic, not as a replacement for the scored
comparison.

## Assumptions and falsifiers

- The observed input process has local shifts whose useful structure is
  recoverable from next-input residuals.
- The update has positive conditional alignment with target prediction often
  enough to outweigh harmful windows; gradient alignment must be measured.
- A low-rank update with shrinkage prevents fast-weight drift and forgetting.
- The episodic support/query schedule used in training matches streaming
  inference, including the 99-row warm-up and sequence resets.
- The adapted teacher provides targets or representations that improve a
  matched GRU student; teacher-only improvement is not submission value.

The exact mechanism is weakened if input reconstruction improves while query
WP does not, if task/auxiliary gradient alignment is nonpositive or strongly
regime-dependent, if resets/prefix replay alter results, or if the adapted
teacher's advantage fails to transfer to the same-budget student. A negative
result narrows this low-rank next-input update and schedule; it does not reject
online adaptation with a different auxiliary signal.

## Cheapest discriminating experiment

After a bounded real-shape B300 profile, train a supervised GRU control and a
matched online-adaptation teacher on the same clean fitting groups. Compare
official scores on the next held-out contiguous query blocks under source-
qualified component folds. Add a no-update auxiliary-training control so an
improvement from ordinary joint prediction is not misattributed to online
updates. Then compare GRU students distilled from the adapted teacher and the
matched non-adaptive teacher. No train/valid Parquet metadata scan is admitted
until the receiving worker verifies an independently bounded producer
manifest and actual memory profile.

## Literature basis and transfer limits

- Sun et al., *Test-Time Training with Self-Supervision for Generalization
  under Distribution Shifts*, arXiv:1909.13231. The paper motivates
  self-supervised updates and gives a positive gradient-alignment condition in
  a convex analysis; experiments are image corruption benchmarks, not noisy
  market features. https://arxiv.org/abs/1909.13231
- Kim et al., *Battling the Non-stationarity in Time Series Forecasting via
  Test-time Adaptation*, arXiv:2501.04970. TAFAS exploits partially observed
  future ground truth, which this callback does not provide. Only its
  scheduling and preservation lessons can transfer; its label-feedback
  mechanism is out of scope. https://arxiv.org/abs/2501.04970
- Wu et al., *Test-Time Adaptation for Non-stationary Time Series: From
  Synthetic Regime Shifts to Financial Markets*, arXiv:2602.00073. The paper
  reports that adaptation can help gradual synthetic shifts while financial
  series prefer different, sometimes harmful, update policies. It supports a
  cautious test, not an expected gain. https://arxiv.org/abs/2602.00073
- Tandon et al., *End-to-End Test-Time Training for Long Context*,
  arXiv:2512.23675. It meta-learns an initialization for continued
  self-supervision in long-context language modeling. Its token objective and
  scale differ substantially from this GRU task. https://arxiv.org/abs/2512.23675

**Initial card committed before independent criticism.** No data payload,
target values, masks, predictions, or checkpoint were read to produce this
proposal.
