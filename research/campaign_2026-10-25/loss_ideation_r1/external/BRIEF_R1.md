# Public task brief — loss/objective ideation and red team (2026-10-25)

You are an independent ML researcher. Read only this brief. No code access needed.
Deliver plain Markdown. Be concrete and quantitative; no filler.

## Task

Causal sequence regression competition (Wunder Challenge 2, ends 2026-11-15).
Anonymized limit-order-book data for two instruments: 112 float features per row.
Sequences of 20,000 rows; rows 0-98 warm-up; predict two targets (t0, t1) on every
later row, but only a sparse subset of rows is scored (`is_scored`, ~ propensity
predictable from causal features). Targets are anonymized future-price indicators of
instrument i0 (structurally like t0 = F - ask, t1 = bid - F for a future price F).
Serving: strictly causal, row-by-row streaming callback, state reset per sequence,
CPU-only Python 3.11, ZIP <= 20 MB, total runtime <= 60 min on the hidden test set
(our current packages already use ~54 min). 10,607 train + 1,873 valid sequences.

Metric per target: pooled weighted Pearson over all scored rows of all sequences,
y and p each clipped to [-2, 2], weight w = |clip(y)|; final = mean over the 2 targets.
Note: WP is invariant to a global affine map of p per target, NOT to per-sequence
or per-state rescaling.

## Current state

- Public best 0.66832 (3-member blend: one distilled GRU student 0.34 + two
  feature-tokenizer-Transformer members 0.33 each). Leaderboard leader 0.67243;
  the whole rank 22 -> rank 1 spread is 0.0041.
- Local cross-validation F0 endpoints ~0.681-0.684; measured seed noise of a full
  endpoint is 0.0007-0.0014 WP. Most recent "gains" are +0.0001-0.0004 (below noise).
- Main model family: GRU 256x2 (also 512) with canonical inputs, trained with
  weighted MSE: loss = sum w*(p - y_clip)^2 / sum w, w = |y_clip| * scored-propensity,
  plus distillation from heterogeneous recurrent teachers: target =
  (1-a)*y + a*teacher, a = 0.9 for strongest students.
- Note on Bayes alignment: maximizing pooled WP with w=|y| has population optimum
  p(x) = affine(E[|y| y | x] / E[|y| | x]); weighted MSE with weight |y| has
  exactly that minimizer. So the objective is already "consistent"; gains must come
  from estimation variance / optimization / representation.
- One target "class" (high-|y| regime) carries ~43-48% of metric weight and our WP
  there is only ~0.43 vs 0.74-0.81 elsewhere. An oracle branch readout reaches 0.968.

## Loss/objective history (measured)

- Direct per-chunk weighted Pearson loss: worse than corr+WMSE mix (refuted, Sept).
- WMSE alone beats corr + 0.5 WMSE mix when propensity^2*|y| weights are used.
- Propensity-weighted training (scored-row propensity) helps; tempered p^alpha helps.
- Reproducible teacher-level gains cluster in loss *weighting*: propensity-0, label-
  weight floor (zero-label rows keep teacher signal), class-3 upweighting.
- Gaussian NLL (mean + log-variance head) vs matched WMSE: delta -0.00004 (warm,
  fold F0), earlier warm teacher-blend NLL +0.0007 [-0.0003,+0.0019]. NLL clipped
  100% of updates (mean clip factor 0.04 vs 0.99 for WMSE) under global grad clip.
  beta-NLL (detached variance^beta) proposed, untested.
- Distillation-loss variants (difference-channel weighting, Pearson/DIST matching,
  run-level) are null (+0.0000..+0.0002); sum-channel-zero variant -0.0016.
- Batch-mass normalization (running EMA denominator instead of per-batch sum w) coded,
  not graded; per-batch weight mass varies 0.1x-9.7x.
- Training pools both targets in one denominator; metric averages two correlations.
- Muon optimizer vs AdamW: -0.005 on a synthetic screen despite 6x higher recurrent
  stable rank.
- NEW (teammate, 8xB300): 630M and 1.54B causal Transformer teachers trained 30
  epochs at LR 3e-4 massively overfit: train WMSE 0.01 vs validation WMSE ~1.55;
  validation WP 0.16-0.17 (630M at epoch 30) and 0.31-0.41 (1.54B, epochs 15-20,
  declining every epoch). GRU students score ~0.68. Early-epoch checkpoints were
  overwritten.

## Resources

Local: one 6 GB laptop GPU (5 GB usable), 10 CPU cores, 30 GB RAM; Kaggle GPU quota.
Teammates have 8x B300 occasionally. Strict FP32 training policy.

## What we need from you

Part A — IDEAS (6-10). Diverse mechanisms that could plausibly add >= +0.001 local WP
or improve public transfer, focused on loss/objective/target design but you may cross
into representation, regularization, data augmentation, ensembling or serving when
it interacts with the loss. For each: change; why it could help (given the Bayes
alignment note — what variance/optimization/representation problem it fixes);
cheapest discriminating test (fits <= 6, GRU 256x2 ~1-2 GPU-hours each); falsifier;
risk. Include at least one unconventional idea and one idea about the high-|y| class.

Part B — RED TEAM. Attack our reasoning: (1) why did NLL, Pearson losses and
distillation-loss variants fail — is our explanation (objective already consistent,
gains are variance-limited) right or is something else going on? (2) what does the
B300 transformer overfit tell us and how should the teacher program be re-designed?
(3) which assumption in this brief is most likely wrong? (4) what would you stop doing?

Keep total under 1,800 words.
