# Optimizers, learning-rate schedules, cyclic ("large-small-large-small") training and stability (2026-10-09)

Root (Claude Code) research note. It answers: is cosine faster, slower, more or less stable than the alternatives;
what a large-small-large-small (cyclic / warm-restart) schedule buys; how Muon, AdamW and friends compare; and
which measurements decide each question in this project. Labels used throughout:
**[measured]** our runs, **[lit]** a published claim (link given), **[hyp]** our reasoning, not yet tested,
**[queued]** an experiment already in a queue. Note: arXiv ids marked from memory (CLR 1506.01186, FGE 1802.10026, SWA 1803.05407, AdamW 1711.05101, Lion 2302.06675, Shampoo 1802.09568) were not re-verified online; verify before citing externally. Verified online this session: SGDR, Snapshot Ensembles, Muon-scalable, LAWA, MiniCPM, Wortsman et al. Companion notes:
[literature on schedules](LITERATURE_SCHEDULES_DISTILLATION_R1.md),
[telemetry design](TRAINING_TELEMETRY_AND_CALLBACKS_R1.md),
[interpretability and blend theory](MECHANISTIC_INTERPRETABILITY_AND_BLEND_THEORY_R1.md).

## 0. Short answers

1. **Cosine vs the rest.** At full scale on our fold1 scratch teachers no schedule beats the one-cycle (cosine-to-zero)
   control beyond seed noise. Small 220-update screens showed WSD +0.013 over one-cycle, and that gain vanished at full
   scale. [measured] Literature agrees that cosine is near-optimal when the horizon is known and fixed, and that WSD /
   linear-decay-to-zero match or beat it in other settings. [lit]
2. **Speed.** "Faster" is only measurable from a learning curve (score vs steps). We had none until today; the telemetry
   runs `tr_base`, `tr_wsd`, `tr_ep6`, `tr_sgdr*` now produce it. [queued]
3. **Stability.** No schedule or optimizer difference is visible in the one stability signal we already had (AMP skipped
   updates, 14-16 per 29,840 attempts for every run, i.e. just initial loss-scale calibration). Real stability evidence needs
   gradient-norm tails, update ratios and LR-response curves from the telemetry arms. [measured null; queued]
4. **Optimizer.** Muon (on recurrent matrices) beat AdamW by about 0.004-0.005 at the same nominal LR in the one pair we
   have (single seed each, LR not tuned for AdamW). The AdamW LR trio and Muon LR trio in waves 5 and 7 decide how much of
   that is LR mismatch. [measured, partial; queued]
5. **Large-small-large-small.** Implemented (`--sched sgdr --restarts K`). In small screens it was slightly worse than WSD.
   Its real selling point is free snapshot-ensemble members, and the measured fact that blend gain depends on prediction
   *decorrelation* (see companion note) means cycle snapshots must be tested for decorrelation, not assumed. [queued]

## 1. Measured results so far (fold1 unless stated, metric `global_wp_clip2_v1`)

Seed noise: identical-recipe scratch GRU256 teachers scored 0.66557 (seed 3) and 0.66675 (seed 4), so one standard
deviation of run-to-run noise is about 0.0006-0.001 and differences under about 0.002 are not distinguishable with one seed.
Scratch baseline for comparison = mean 0.6662.

| Run (scratch teacher, Muon rnn-scope, onecycle, b32, chunk 1000, 4 epochs unless noted) | wp | vs scratch baseline 0.6662 |
|---|---|---|
| GRU256x3 layers | 0.66774 | +0.0015 |
| trapezoid (warmup, stable, linear cooldown 20%) | 0.66744 | +0.0013 |
| GRU384 | 0.66724 | +0.0011 |
| WSD 1-sqrt cooldown (combo_scratch_s3_wsd) | 0.66663 | +0.0004 |
| loss `wmse_gcorr` | 0.66625 | +0.0001 |
| LSTM256 | 0.66581 | -0.0004 |
| MLP encoder | 0.66528 | -0.0009 |
| linear decay to zero (D2Z) | 0.66504 | -0.0012 |
| fixed_mix readout | 0.66490 | -0.0013 |
| `dual` architecture | 0.66275 | -0.0035 |
| **AdamW (lr 1e-3)** | 0.66231 | **-0.0039** |
| GRU512 | 0.65800 | -0.0082 |

Earlier small-screen results (350 fit sequences, 220 updates, folds 0 and 3; not transferable one-to-one):
WSD beat one-cycle by +0.0136 / +0.0134; SGDR slightly worse than WSD; Muon standalone and Huber worse than the control;
FP16 equal to FP32. At full scale for warm-started 2-epoch distillation students WSD and one-cycle tie
(0.66886 vs 0.66882); tail-weight power 1.5 hurt (-0.00055, significant).

Why small screens and full scale disagree [hyp]: a 220-update run is dominated by how quickly the LR reaches a good value
and how clean the final annealing is, so a schedule that spends more steps at a useful LR wins. At 29,840 updates every
reasonable schedule has time to reach a similar basin and the remaining differences are inside seed noise.

## 2. Schedule families and what the literature says

| Schedule | Shape | Needs known horizon | Our flag | Key source |
|---|---|---|---|---|
| One-cycle (cosine to ~0) | short warmup, cosine anneal; beta1 cycled 0.85-0.95 | yes | `--sched onecycle` (default) | [Smith & Topin](https://arxiv.org/abs/1708.07120) |
| Cosine with warm restarts (SGDR) | K+1 cosine cycles, LR jumps back up | per cycle | `--sched sgdr --restarts K` | [Loshchilov & Hutter](https://arxiv.org/abs/1608.03983v5) |
| Cyclical LR | triangular up/down between bounds | no | not implemented | [Smith 2015, arXiv 1506.01186](https://arxiv.org/abs/1506.01186) |
| WSD (warmup-stable-decay) | warmup, constant, short cooldown (1-sqrt, exponential, linear or cosine) | no (cooldown can branch from any stable checkpoint) | `--sched wsd` | [MiniCPM](https://arxiv.org/pdf/2404.06395), [Hägele et al.](https://arxiv.org/pdf/2405.18392) |
| Linear decay to zero (D2Z) | warmup, linear to 0 | yes | `--sched wsd --wsd-decay 1.0 --wsd-final 0 --wsd-shape linear` | [Bergsma et al.](https://arxiv.org/pdf/2502.15938) |
| Trapezoid | warmup, constant, linear cooldown | no | `--sched wsd --wsd-shape linear --wsd-decay 0.2 --wsd-final 0` | [Zhai et al.](https://arxiv.org/abs/2106.04560) |
| Schedule-free | no schedule; interpolation plus averaging | no | not implemented | [Defazio et al.](https://arxiv.org/pdf/2405.15682) |

Claims worth knowing [lit]:
- Cosine is a strong default but couples the whole run to a fixed token budget; WSD matches it while letting the cooldown
  branch from any stable checkpoint, and the 1-sqrt cooldown beat linear in the comparison of Hägele et al.
  Details of what the cooldown does to the model: [Training dynamics of the cooldown stage](https://arxiv.org/pdf/2508.01483);
  a batch-size and horizon agnostic variant: [Power scheduler](https://arxiv.org/pdf/2408.13359).
- Linear decay to zero beat cosine-to-10% in the Bergsma et al. comparison, with a larger benefit when tokens per parameter
  are high. Our D2Z teacher (-0.0012) does not show this; one seed and a short, small-model run are the caveats.
- Warmup mostly buys the ability to use a larger peak LR ([Kalra & Barkeshli](https://arxiv.org/abs/2406.09405)); the
  size of the safe peak LR is what separates "stable" from "unstable" configurations
  ([Wortsman et al.: small-scale proxies for instabilities](https://arxiv.org/abs/2309.14322)).
- Weight averaging recovers part of what annealing does: [latest weight averaging (LAWA)](https://arxiv.org/pdf/2209.14981)
  and [early weight averaging with high LR](https://arxiv.org/pdf/2306.03241). Good news for constant-LR phases.

### Is cosine faster, slower, more or less stable? Decision table

| Property | What to measure | Source of the number | Expectation [hyp] |
|---|---|---|---|
| Speed to a given score | steps until intra-epoch eval first reaches X | `intra_eval.jsonl` of `tr_base` (onecycle), `tr_wsd`, `tr_sgdr*`, `tr_ep6` | WSD/trapezoid lag onecycle mid-run (higher LR, no annealing yet) and catch up only at the cooldown; cosine is "anytime-poor" |
| Final quality at fixed budget | wp at the last checkpoint | paired seeds of each schedule | within about 0.001 of each other |
| Horizon flexibility | can we stop early or extend? | WSD: restart cooldown from the stable checkpoint; cosine: no | WSD wins operationally |
| Stability to peak LR | wp and loss at 0.5x, 1x, 2x LR | `tr_lrh`, `tr_base`, `tr_lr2` (Muon) and `adamw_lr*` (AdamW) | flat response over 4x LR = stable; collapse at 2x = fragile |
| Gradient-norm behaviour | tail of `gn_total`, fraction above clip 1.0, spikes at LR peak | `trace.jsonl` | gradient norms may peak near the LR peak under onecycle; WSD flatter (to be measured, no number assumed) |
| Update scale | `ur_*` per parameter family | `trace.jsonl` state rows | Muon update ratios are roughly shape-independent; AdamW's differ per family |
| Seed variance | std of wp across 3 seeds per schedule | repeats | needed to resolve 0.001 effects |

Rule: never rank schedules by one seed. A schedule claim needs at least two seeds per arm or a paired, same-seed design.

## 3. The large-small-large-small family (cyclic schedules)

What it is: the LR repeatedly rises and falls (SGDR restarts, cyclical LR, triangular, or "Hägele-style" repeated stable plus
cooldown phases). Why people use it [lit]:
1. **Snapshot ensembles** ([Huang et al.](https://arxiv.org/abs/1704.00109)): save the weights at every cycle minimum and
   average predictions; claims ensemble gains at single-run cost.
2. **Fast geometric ensembling and weight averaging** ([Garipov et al., arXiv 1802.10026](https://arxiv.org/abs/1802.10026),
   [SWA, Izmailov et al., arXiv 1803.05407](https://arxiv.org/abs/1803.05407)): high-LR phases move between basins or across a
   flat region and the average of the low-LR points often lands in a wider optimum.
3. **Escaping poor regions** through large-LR phases; also a built-in LR-sensitivity test (each cycle is a free LR probe).

Costs and risks for this project [hyp, partly lit]:
- A restart to a high LR on a warm-started student re-opens forgetting ([Ibrahim et al.](https://arxiv.org/pdf/2403.08763)),
  so cycles are a teacher-side tool, not a student-side one.
- Cycle snapshots share a trajectory and are typically *more* correlated than independent seeds; our measured pair-blend
  theory (companion note) says the blend gain is governed by the prediction correlation. Our independent scratch seeds
  correlate 0.98-0.99, so cycle snapshots are expected to correlate above 0.99 and gain under +0.0005. This is a
  prediction to falsify.
- With 4 epochs the cycles are short (about 7,500 updates each); a cycle that ends mid-annealing yields a worse minimum.

Experiments [queued, Vast wave 6, all with telemetry]:
- `tr_sgdr4` (restarts 3, 4 cycles, minima near each epoch end), `tr_sgdr2` (2 cycles), `tr_sgdr8` (8 cycles, 2 per epoch).
  Cycle-end checkpoints are exported as separate predictions (`vt_f1_tr_sgdr4__ep1`, ...) so the snapshot ensemble, the
  single final model, and an offline weight average (`scripts/weight_soup.py`) can all be scored.
- Decision: snapshot ensemble worth keeping if (a) final cycle model is not worse than the one-cycle baseline by more than
  noise, and (b) the equal-weight blend of cycle-end snapshots beats that single model by more than +0.0007 (the average pair gain of independent seeds).

Ideas not yet implemented [idea]: decaying-peak restarts (each cycle lower peak), restart with a *different seed order* instead of a higher LR
(cheap diversity), cyclical-LR range test at the start of each run, and constant-LR plus periodic cooldown snapshots
(WSD "branching") to get a family of models from one stable run.

## 4. Optimizers

| Optimizer | Idea | Notes for a GRU on market sequences | Source |
|---|---|---|---|
| AdamW | per-parameter adaptive step, decoupled weight decay | default baseline; update size depends on parameter statistics | [Loshchilov & Hutter](https://arxiv.org/abs/1711.05101) |
| Muon | orthogonalize the momentum update of 2D weights (Newton-Schulz), AdamW for the rest | equalizes singular values of updates; matrices are exactly where recurrent dynamics live (our `--muon-scope rnn`); needs an RMS match to AdamW-style LR and weight decay to be stable at scale | [Moonshot: Muon is scalable](https://arxiv.org/html/2502.16982v1) |
| SGD + momentum | plain | tuned LR is narrow; included in trainer as `sgd` | classic |
| Lion, Shampoo, SOAP, Sophia | sign / preconditioned / second-order | not implemented; Shampoo-family is the closest relative of Muon | [Lion](https://arxiv.org/abs/2302.06675), [Shampoo](https://arxiv.org/abs/1802.09568) |
| Schedule-free AdamW | removes the schedule | needs its own LR and weight-decay sweep | [Defazio et al.](https://arxiv.org/pdf/2405.15682) |

Measured [measured]: AdamW 0.66231 vs Muon 0.6662 baseline (-0.004) in teachers with `--optim adamw --lr 1e-3`; but our
earlier 220-update screens had Muon *worse* than the control. Possible reconciliation [hyp]: Muon's benefit appears only
after enough updates for orthogonalized steps to accumulate along the recurrent directions; with 220 updates the
sharper per-step progress of AdamW dominates. The AdamW LR trio (5e-4, 1e-3, 2e-3) and Muon LR trio (5e-4, 1e-3, 2e-3)
settle the question of LR mismatch, and `tr_muonhidden` tests the Muon scope. Decision rule: Muon is better if its best-LR
wp beats AdamW's best-LR wp by more than the two-seed noise (about 0.002) and its LR response is not narrower.

Interaction to watch [hyp]: one-cycle also cycles beta1 (0.85 to 0.95); for Muon the momentum plays a different role, so
the onecycle-with-Muon combination in the combo recipe is itself a confound for "schedule" comparisons. WSD runs keep the
cycled beta1 by default in our implementation; the `beta1` trace column will show exactly what each run did.

## 5. Mapping to a stability dashboard (what we will read from `trace.jsonl`)

| Symptom | Column | Healthy pattern | Action if unhealthy |
|---|---|---|---|
| LR too high | `loss` EMA noisy, `gn_total` tail heavy, `ur_*` above about 1e-2 | smooth loss; `ur` in a stable band (commonly cited rule of thumb is order 1e-3; to be calibrated on our own baseline run, not assumed) | lower peak LR or lengthen warmup |
| LR too low | `ur_*` far below the baseline band, flat loss | steady decrease | raise LR; check `w_*` growth |
| Clipping binds | fraction of steps with `gn_total` > 1 | stable fraction across the run (calibrate on the baseline) | raise clip or adopt percentile clipping |
| Family imbalance | `ur_rnn_hh` vs `ur_head` | within about 3x | per-family LR multipliers |
| Optimizer-state drift | `v_rms_*` (Adam), `m_norm_*` | stable after warmup | revisit beta2 / eps |
| Saturation / dead units | gate statistics (probe tooling, companion note) | gates not pinned at 0 or 1 | regularize or reinit |

## 6. Queue and decisions (all fold1, scratch GRU256x2 unless noted)

| Arm | Question | Wave | Decides |
|---|---|---|---|
| `tr_base` | baseline curve | 5 | speed, grad profile |
| `tr_wsd` | cooldown attribution | 5 | H-298 |
| `tr_ep6` | is 4 epochs saturated? | 5 | H-297 |
| `tr_lr2`, `tr_lrh` (Muon) | LR response | 5 | stability width |
| `adamw_lr1/2/h` | AdamW LR response | 7 | optimizer fairness |
| `tr_muonhidden` | Muon scope | 7 | whether recurrent-only Muon is best |
| `tr_swa` | online weight averaging | 5 | H-301 |
| `tr_b16` | batch noise | 5 | H-302 |
| `tr_sgdr2/4/8` | cyclic / snapshot ensembles | 6 | H-305 (new) |

Paired design reminder: use at least two seeds before stating that a schedule or optimizer difference exists below 0.002.
After these land, rerun the two best configurations on fold0 (replication) before adopting them in a final recipe.

## 7. What would change our plan

- If `tr_ep6` is clearly better than `tr_base` (more than +0.002): budget, not shape, is the lever; extend all teacher fits.
- If AdamW at a tuned LR closes the gap: stop treating Muon as a driver; its apparent edge was LR mismatch.
- If SGDR snapshot blends beat independent-seed pairs: use cycles to manufacture ensemble members in one run.
- If every schedule and optimizer lands within noise: stop sweeping schedules; spend compute on capacity diversity and independent lineages,
  which is where the largest measured blend gain came from (companion note).
