# L4 GRU impact batch 2: ten screens built on the since32 result

**Status: proposal, no fits executed.** Preparation owner: Vlad / Claude Code. Receiving and
launch owner: Semyon. Same contract as [PR #82](https://github.com/VladKochetov007/WunderFunder/pull/82)
(PR80 K16 recipe, L4, fixed E6, train 10,325 / full valid 1,873 / 3,528,273 rows, split
`0c4de3a2…`, seeds 203301/203302, mean paired gain ≥ +0.0005 with both seeds positive).

## What batch 1 told us (Semyon's readout, 2026-11-12)

| Cell | Seed 203301 | Seed 203302 | Mean Δ vs K16 |
|---|---|---|---|
| K16 control | 0.683019 | 0.682629 | — |
| **since32** (card 03) | **0.686306** | **0.685904** | **+0.00328** |
| GRU + Muon at LR 3e-4 (card 01) | 0.683235 | 0.680832 | −0.00079 |

Since32 is the strongest single change on this contract so far; it needs a same-code control
rerun before adoption. Muon at the K16 LR is mixed/negative; our positive Muon evidence used
LR 1e-3.

**Why since32 works (our ledger).** Class-3 labels are the rounded mean mid over the **next four
book-price events** (E-079, 97.5% exact on event rows): the target lives on an *event clock*,
not a row clock. On rows deep inside quiet stretches the label is fixed by events that have not
happened yet, so those rows are mostly label noise for any causal model. Down-weighting them by
`max(0.25, 1/(1 + since/32))` removes that noise from the gradient. E-351 also found that
since-weighting and distillation **substitute** for each other (each ≈ +0.0015 alone, ≈ +0.0017
together, interaction −0.0012): both reduce the same deep-row label variance. Several cards
below test this event-clock explanation directly.

**Correction to PR #82 card 07.** E-118 (Sep 26) already tested per-sequence online z-score
inputs: +0.0003/+0.0006, "adds nothing". Card 07 (causal regime channels) differs (raw
statistics appended, not replacing inputs) but its prior should be lowered accordingly.

## Cards

All cards add two treatment fits at seeds 203301/203302 unless stated; **controls** are the
PR80 K16 fits, or the same-code since32 fits where the card says "on since32". Code changes are
named per card; none are implemented in this PR.

### B01. Since-weight curve: τ 8 and τ 128 (exploit the winner, map its shape)
Changed: since τ 32 → 8 and → 128 (floor 0.25). **4 fits.** Control: since32. Predicts an
interior optimum if the event-clock explanation holds (label horizon ≈ 4 events); a monotone
curve points at simple row down-weighting instead. Report WP by since bucket (0–8, 8–32, 32–128,
128+ rows) and by class.

### B02. Hard event mask: loss only on rows within 32 rows of an i0 price event
Changed: floor 0.25 → 0 with a step at since ≤ 32 (weight 1) else 0. Control: since32. This is the hard
version of since-weighting and also cuts the effective rows per epoch. Falsifier: worse than since32 on both seeds
(deep rows still carry signal).

### B03. Since32 × distillation interaction on this runtime
Changed: since32 **and** PR #82 card 02 distillation (PR49 `tl3c_fbc1k_s5` teacher, α 0.5).
Controls: since32 alone and distillation alone (card 02 fits, when run). Predicts a gain well
below the sum (E-351 substitution). Decides whether the production chain needs both or whether
since-weighting can replace the teacher stage cheaply.

### B04. Muon follow-up at LR 1e-3 with a gate-saturation precheck
Changed: Muon recurrent groups at LR 1e-3 (other groups AdamW 3e-4), on since32. Precheck, zero
fit: share of update-gate activations outside [0.02, 0.98] on the K16 and since32 E6
checkpoints over 32 train sequences. If ≤ 10%, skip the fits (mechanism absent). Our evidence:
Muon gain appeared with AdamW at 45–48% saturation and LR 1e-3 (E-OPT25-FAST-PANEL-R1).

### B05. Event-clock auxiliary target
Changed: one auxiliary head predicting the change of i0 mid over the **next k = 4 i0 book-price
events** (computed from future *inputs* inside the loss only; never an input or carried state),
weight 0.3, dropped at export; on since32. Differs from PR81 card 07 (labels at fixed *row*
offsets). Analogue: a Jane Street Real-Time Market Data Forecasting GRU solution gained ≈ +0.001
from auxiliary multi-horizon targets (write-up kept in the author's local workspace). Falsifier: no gain, or gain without lower deep-row loss.

### B06. GRU-D style event-time decay
Changed: append `log(1 + since)` for the i0-price and i1-price clocks as inputs and decay the
hidden state toward a learned baseline by `γ = exp(−relu(W·Δ))` where Δ = rows since the last
event (Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values*,
arXiv 1606.01865; also Phased LSTM, arXiv 1610.09513). Gives the model the event clock
explicitly instead of only through loss weights. Control: since32 (both have since-weighting).

### B07. minGRU with full-sequence gradient horizon
Changed: replace the cuDNN GRU by a two-layer minGRU (Feng et al., *Were RNNs All We Needed?*,
arXiv 2410.01201): gates depend on x only, so training runs as a parallel scan and the gradient
horizon can be the whole 20,000 rows at modest memory. Same width; LR unchanged. Serving remains
a constant-time recurrence. Falsifier: worse than since32 at any horizon; then long BPTT is not
the bottleneck. Pairs with PR #82 card 05 (4,000-row horizon on the standard GRU).

### B08. Volatility-scaled output head
Changed: `ŷ_t = σ̂_t · g(h_t)` where σ̂_t is a causal EMA (half-life 1,000 rows) of |Δ mid| of
i0, floored at its train 5th percentile; g is the existing head. Pooled WP compares predictions
**across** sequences, so cross-sequence scale calibration matters; this hard-wires the "normalize
by volatility" rule used in intraday trading instead of leaving the GRU to learn it. Report the
per-sequence prediction/label scale ratio dispersion. Control: since32.

### B09. Sharpness-aware minimization (SAM, ρ 0.05) for over-fitting under shift
Changed: SAM wrapper around AdamW (Foret et al., arXiv 2010.01412), ρ 0.05, ≈ 2× compute per
update. Motivation: rapid over-fitting reported in PR71 and by other competitors, dropout/SWA
nulls (PR71) and a shifted valid. Control: since32. Report train-valid gap per epoch. Cost is the
main risk; if L4 time is binding, run one seed first.

### B10. Validator null test (integrity, 1 fit)
Changed: train the since32 recipe on labels **permuted across sequences** (each sequence keeps
its own inputs, receives another train sequence's label stream). Expected full-valid WP ≈ 0.
Any value materially above 0 means the pipeline leaks (normalizer, evaluation alignment,
endpoint selection). "Validate the validator on a random walk" is the cheapest guard before a
long wave of small deltas. **1 fit**, no control.

## Budget

B01 4 fits, B02–B09 2 fits each (16), B10 1 fit: **21 new L4 fits**, plus the B04 zero-fit
precheck. Teacher predictions for B03 come from the PR #82 card 02 export.

## Sources

Ledger: E-079 (event-clock label), E-351 (since × distillation), E-118 (online z-score inputs),
E-078/E-080 (online recalibration negative), E-OPT25-FAST-PANEL-R1 (Muon). Papers verified by
title on arXiv: GRU-D, Phased LSTM, SAM; minGRU per its arXiv listing. Practitioner sources (the
author's notes and a Jane Street competition write-up kept locally) motivated B02, B05, B08 and
B10; they are motivation only, not evidence.
