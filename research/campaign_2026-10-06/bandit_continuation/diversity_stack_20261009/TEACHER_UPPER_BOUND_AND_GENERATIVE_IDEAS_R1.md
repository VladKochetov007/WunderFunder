# Raising the teacher upper bound: mixtures, target weighting, scale, augmentation, generative models (2026-10-09)

Root (Claude Code) note, from the user's ideas of 2026-10-09: mix different teachers, optimizers and losses; weight the target
distribution (in distillation and when training teachers); move the upper bound of teacher quality with scale or data
augmentation; recurrent VAE / GAN / Transformer for synthetic data and "scale to the max". Labels: **[measured]** our data,
**[lit]** published, **[hyp]** untested reasoning, **[queued]** already in a queue, **[build]** needs code.
Companions: [research plan](RESEARCH_PLAN_NIGHT_20261009_R1.md), [blend theory](MECHANISTIC_INTERPRETABILITY_AND_BLEND_THEORY_R1.md),
[optimizers and schedules](OPTIMIZERS_SCHEDULES_STABILITY_R1.md).

## 0. Where the upper bound stands

- Scratch teachers on fold1 sit at 0.659-0.669 with seed noise about 0.001; widening past 384 and recipe variants did not move
  the best single model; the distilled-lineage students reach 0.669 (fold1) and 0.682 (fold0, where scratch teachers reach 0.670). [measured]
- Capacity alone does not help at 4 epochs: GRU512 is 0.0082 below the baseline, GRU384 ties it, and none of our models use dropout.
  That pattern (bigger is worse or equal, no regularization) is what a data-limited or optimization-limited regime looks like. [measured, interpretation hyp]
- 12.5k sequences (10.6k train plus 1.9k valid) of 20,000 rows each is a small number of independent sequences even though the row count is large.
  So the first question for any "scale" idea is: is the gap from train to evaluation large (overfitting) or is training simply too short?
  The telemetry arms (`tr_base`, `tr_ep6`) and the regularization arms below answer it. [queued]

## 1. Teacher mixtures, optimizers, losses

| Idea | Status | Why it might work | Test |
|---|---|---|---|
| Mixture of teachers by decorrelation (not by score) | rule validated on 66 pairs [measured] | blend gain needs corr(i,j) < r_j/r_i | pick teacher sets with the closed form before building a cache |
| Mixture by regime (class-gated weights) in prediction space | no gain [measured] | would help only if different models win in different classes | do not repeat in prediction space |
| Teacher mixture inside distillation, per-sample random convex weights ("teacher dropout / Dirichlet mixing") | **[build]** | the cache already stores per-family predictions (`families_*.npy`, 4 families x 2 targets); randomizing the family weights per batch is target-space augmentation and stops the student overfitting one fixed blend | add `--teacher-mix-dirichlet A` in the distillation branch; matched student vs fixed 0.8125 mix |
| Disagreement-weighted distillation (weight rows by teacher spread or confidence) | **[build]** (earlier folder `disagreement_guided_H204`) | rows where teachers agree carry reliable signal; disagreement marks hard rows | per-row weight from the family spread; compare |
| Different optimizers per stage (Muon teachers, AdamW students) | students use AdamW 1e-4 warm start today | optimizer for the distillation stage never ablated | student with Muon (needs LR scale), matched seed |
| Distillation loss families | trainer allows only wMSE for cached distillation | correlation-style or per-target losses might match the metric better | relax the guard behind an explicit flag, matched student (build) |
| Teacher loss families | wmse (base), `wmse_gcorr`, `corr`, `huber` | different losses give different error structure | wave 8 `corr`, `huber`; wave 3 `wmse_gcorr` 0.66625 [measured] |

## 2. Target weighting

Distillation side [measured]: tail power 1.5 hurt (-0.00055, significant); zero-label floor 0.1 was neutral (0.66885 vs 0.66882).
Not tested: per-class target weights, teacher-confidence weights, per-family alpha, target temperature.

Teacher side (the user's question: weighting when learning teachers):
- Existing, now queued as scratch teachers [queued, wave 8]: `--label-weight-power 1.5` (tail emphasis; newly added to the teacher
  objective, defaults unchanged), `--label-weight-floor 0.1` (zero-label rows keep signal), `--class-weight 1,1,1,2` (class 3 carries 50% of the metric weight at wp 0.44).
- Existing knobs not yet swept: `--since-weight` (32 in the recipe), `--prop-power` (2), `--prop-eps`, `--episode-weight` (redundancy down-weighting of repeated labels). [build: small sweep wave]
- Quant view: the metric weights each row by |clip(y)|, so the loss already matches it; a power above 1 trades the metric match for tail
  emphasis, and the distillation result says it hurts there. For teachers the sign of the effect is open because the teacher also sets
  the *shape* of the target the student learns. Decision: keep a variant if it improves the *blend* (decorrelation) even when its own score is lower.

## 3. Scale and data augmentation

Gate before spending on scale: show that the regime is data- or regularization-limited.
- Regularization first (cheap, no new code) [queued, wave 8]: dropout 0.2 on GRU384 and GRU512, weight decay 0.05. If dropout rescues GRU512
  (0.658 to at least 0.667) the capacity ladder reopens and "scale to the max" becomes a plan (section 5).
- **Symmetry and invariance discovery** (a quant brainteaser; if one holds it doubles data for free) [build, EDA first]: features are
  anonymous, so test candidate transforms empirically. For each candidate T (sign flip of a block of columns with a sign flip of
  both targets, swapping paired columns, global rescaling of a block, time-reversal excluded because the model is causal) measure whether a trained
  model's prediction on T(x) equals T-transformed prediction on x. A transform that passes is an augmentation we may train with.
  Also check target symmetry: distribution of (t0, t1) under sign change, and the class structure under T.
- **Noisy-student distillation** [build]: cached teacher targets come from the clean input, so the student can train on noised or dropped
  features for free (the targets are fixed). Add train-only Gaussian noise and column dropout to the normalized inputs inside the model;
  vary strength; the student must stay at least as good on clean evaluation.
- Row-level supervision already extended by distillation: cached teacher targets cover every row, including rows without labels
  (the propensity weight handles the mismatch). [measured design]
- Using unlabeled test-time features for self-training is NOT assumed allowed. Needs a rules check in `docs/` before any use. [open question]
- Longer training and multi-seed ensembles as scale proxies: `tr_ep6` (6 epochs) and the seed ladder [queued].

## 4. Generative models (recurrent VAE, GAN, diffusion, Transformer)

Assessment [hyp]:
- A generator trained on the same 12.5k sequences can only reproduce what it learned; for a (features, future-label) joint it must learn the same
  predictive relation we are trying to learn, so synthetic samples mostly encode the current model's bias and add little new information.
  GANs add instability; a diffusion model over 20,000-step multichannel sequences is expensive. Low prior for synthetic-label data.
- Where the same machinery is plausible:
  1. **Self-supervised pretraining** of the recurrent trunk on features only (next-step or masked prediction, contrastive, or a recurrent VAE as a
     representation learner), then fine-tune on labels. The trainer already has the `aux-next` and `aux-struct` auxiliary heads; a stronger pretraining stage is the cheap version. [build; test on fold1]
  2. **Conditional resampling of nuisance factors** (volatility rescaling, time warping) only if the invariance EDA shows the labels respect them.
  3. **Transformer as a teacher, not a generator**: serving limits do not bind teachers (ADR0008). A causal windowed Transformer or a deep causal TCN
     offers a non-recurrent, differently-biased teacher; decorrelation is what the blend rule rewards. The `tcn` architecture is queued as the cheap probe
     (receptive field 2^10 rows); a real attention teacher needs windowed memory across the 1000-row chunks (earlier attention folders `attention_training_H218`, `interaction_attention_tcn_H264` hold prior attempts to reuse as ideas).
  4. **Synthetic stress tests** of the models (shuffled-causality, regime shifts) as evaluation, not training.
- Gate: build a generator only if (a) the regularization/ep6/dropout arms show overfitting, and (b) the symmetry EDA finds no free augmentation.
  Smallest first test: recurrent-VAE pretraining of the trunk (item 1), compared with the aux-head baseline on the same fold.

## 5. "Scale to the max" plan (conditional)

If dropout and augmentation unlock capacity: width ladder 384-768 with dropout; 6-10 epoch training (WSD or one-cycle per the schedule note);
multi-seed teachers on four folds; distill into the serving-size student; blend by the decorrelation rule. Compute: the existing 2x RTX 4090
box handles about 4 GRU384 fits per hour pair; a second machine is justified only when a scale arm shows a positive transfer on two folds.
Kaggle GPU quota is exhausted for the campaign, so scale work runs on Vast.

## 6. Queue now (existing Vast box; interleaved so concurrent jobs differ)

Wave 8 (families: weighting, regularization, architecture, input view, loss): `lw15`, `lwfloor`, `cls3w2`, `gru384_drop2`, `gru512_drop2`, `wd05`,
`tcn`, `structured`, `eventcontent`, `rawonly` (raw-only 112-column view, the teammate's main diversity source), `corr`, `huber`.
Earlier waves still queued: 4-fold width evidence, class-3 specialist, gated variants, telemetry, SGDR snapshots, optimizer grid, fold0 diagnostics.
The queue is reordered by `controller.py reorder` (round-robin over mechanism families: fold, schedule, optimizer, weighting, architecture, width, loss, regularization),
so the four GPU slots run unlike experiments at the same time.

## 7. Build list (priority order)

1. Invariance and symmetry EDA on the existing models and data (CPU-light, uses saved checkpoints and the data registry).
2. Noisy-student input augmentation in the distillation path, plus Dirichlet teacher mixing.
3. Hyper-parameter sweep wave for `since-weight`, `prop-power`, `episode-weight`.
4. Recurrent-VAE (or next-step) self-supervised pretraining of the trunk; matched fold1 comparison.
5. Distillation loss alternatives behind an explicit flag; disagreement-weighted distillation.
6. Vast-hosted student pipeline so that wide and diverse teachers can be tested as *student teachers* (research plan section 4).
