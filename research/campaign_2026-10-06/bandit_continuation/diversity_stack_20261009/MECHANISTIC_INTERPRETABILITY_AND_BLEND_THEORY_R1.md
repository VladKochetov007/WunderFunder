# Mechanistic interpretability tools, blend theory and what we measured (2026-10-09)

Root (Claude Code) note. Goal from the user: use interpretability-style tricks plus training logs to decide which
architectures learn what, which combinations are useful, which models are weak alone but good in a blend, and where each
loss plays a role. Labels: **[measured]** our data, **[lit]** published (link), **[hyp]** untested reasoning, **[queued]**.
Companions: [optimizers and schedules](OPTIMIZERS_SCHEDULES_STABILITY_R1.md),
[telemetry design](TRAINING_TELEMETRY_AND_CALLBACKS_R1.md).

## 0. Headline result (read this first)

On fold1, the best student (`h230_s4_onecycle`, wp 0.66913) blended with **0.4 of a scratch GRU384 teacher** scores
**0.67207 (+0.0029)**; a four-way blend with GRU512 and GRU256x3 scores 0.67225 (+0.0031). All earlier blends of
student-lineage models gained only +0.0003 to +0.0008. The ranking of partners was **predicted in advance by a two-number
rule** (section 1) and the predicted gain tracked the measured gain closely. [measured]

Caveats, stated plainly: the blend weights were chosen on the same exposed fold (mild optimism, perhaps a few 1e-4);
partner selection was among six teachers; the effect is large compared with seed noise (about 0.001) but it still
needs replication. Replication jobs are queued: fold0 GRU384 and GRU512 with the fold1 weights frozen (a true transfer test),
GRU384 seeds 4 and 5 (capacity or luck?), and widths 320 and 448 (where is the optimum?). [queued, Vast wave 0]

## 1. Blend theory that actually works here

Let `r_i` be the weighted Pearson of model `i` with the target and `c` the weighted correlation between two model
predictions (the grader already writes both). Standardize predictions (Pearson is scale free).

- **Marginal test:** adding a little of model `j` to model `i` improves the blend iff `c < r_j / r_i`.
  Derivation: `corr(y, p_i + e p_j) ∝ (r_i + e r_j)/sqrt(1 + 2 e c + e²)`; its derivative at `e=0` is proportional to `r_j - c r_i`.
  So a weaker model helps whenever it is decorrelated enough; a clone never helps.
- **Best two-model blend:** `r* = sqrt((r_i² + r_j² - 2 c r_i r_j)/(1 - c²))` and the equal-weight blend is `(r_i + r_j)/sqrt(2 + 2c)`.
- **Validation against our bank [measured]:** 66 measured equal-weight pair blends on fold1 against the equal-weight formula:
  correlation 0.993, slope 0.69 (the formula overstates the gain by roughly 30%, because our wp is a mean of two per-target
  correlations and the weights make some rows count more). Use it for ranking and a first estimate, not as an exact forecast.
- **Practical rule:** rank candidates by predicted optimal-blend gain with the best current model, not by their standalone score.
  Standalone score decides a model's value only when its decorrelation is equal.

Partner ranking for the best fold1 student (predicted optimal 2-blend gain, then measured best-weight gain):

| Partner | wp alone | corr with student | predicted gain (x0.69) | measured best gain (weight) |
|---|---|---|---|---|
| scratch GRU384 | 0.66724 | 0.971 | +0.0040 (about +0.0027) | **+0.0029 (0.4)** |
| scratch GRU512 | 0.65800 | 0.953 | +0.0033 | +0.0019 (0.2) |
| scratch GRU256x3 | 0.66774 | 0.982 | +0.0024 | +0.0015 (0.4) |
| scratch seed 4 | 0.66675 | 0.984 | +0.0017 | +0.0010 (0.3) |
| trapezoid | 0.66744 | 0.986 | +0.0015 | +0.0009 |
| linear-D2Z | 0.66504 | 0.981 | +0.0014 | +0.0007 |

Interpretation [hyp]: the student lineage (combo distilled from `since` teachers, then distilled again) is one family; an
independent scratch lineage with a *different capacity* is decorrelated more than a new seed of the same capacity
(GRU384 and GRU512 corr 0.95-0.97 vs same-width seeds 0.98). Capacity looks like the cheapest source of diversity.
Open question the queued jobs answer: is it capacity, or independent lineage?

Design consequences: (1) prefer independent teachers of different width/architecture in the final blend; (2) every candidate
screen should report `c` with the current blend and the predicted gain next to its standalone score; (3) keep
weak-but-decorrelated models, they are not "bad".

## 2. Behavioural analysis built today and its results

Scripts: `scripts/regime_blend_analysis.py` (regime slices, class-gated blending), `scripts/weight_soup.py` (weight average
of same-parent checkpoints), `scripts/analyze_training_trace.py` (telemetry plots).

Regime structure of the metric on fold1 [measured, `analysis_regime/regime_fold1_r1.json`]:

| Slice | Share of metric weight | wp range across 8 student-lineage models |
|---|---|---|
| market class 0 | 17% | 0.8134-0.8144 |
| class 1 | 21% | 0.7754-0.7767 |
| class 2 | 12% | 0.7442-0.7448 |
| **class 3** | **50%** | **0.4406-0.4424** |
| position quartiles | 18 / 21 / 54 / 7% | 0.664-0.667, 0.647, 0.671, 0.719-0.721 |

Half of the metric weight sits in class 3 where every model scores about 0.44; the other three classes score 0.74-0.81.
Within a class the eight lineage members differ by at most 0.002, so the models agree about *where* they are good and bad.
Improving class 3 is the main lever for the metric. [measured]

Negative results [measured]:
- **Class-gated blending** (blend weights fitted per market class on half of the sequences, scored on the other half):
  0.68246 and 0.65687 vs the global blend 0.68259 and 0.65698, i.e. no gain (-0.0001). Different models do not win in
  different classes strongly enough to exploit.
- **Monotone recalibration** (power transform of predictions with the exponent fit on one half, scored on the other): the fit
  chose exponent 1.0 for the student and for the student plus GRU384 blend (one fold half chose 1.1 and lost 0.0002).
  Calibration slope of `y` on prediction by *predicted* magnitude is flat, 0.87-1.11 for the student and 0.98-1.09 for the blend,
  so there is no free post-processing gain.
- **Trap noted:** calibration slopes computed by slicing on the realized `|y|` (0.31, 0.57, 1.01, 1.57 from low to high tier)
  are mechanical selection artifacts and must not be read as over- or under-confidence. Always slice on inference-time
  quantities (prediction magnitude, features, position), never on the label.

## 3. Interpretability toolbox mapped to this project

| Tool | What it answers for us | Data / code needed | Priority | Source |
|---|---|---|---|---|
| Fixed points and linearization of the recurrent map | memory timescales, attractor structure, whether architectures (GRU vs LSTM vs bigger) store state differently | hidden-state extractor plus Jacobian eigenvalues along real trajectories (slow modes) | medium | [Sussillo & Barak](https://cris.iucc.ac.il/en/publications/opening-the-black-box-low-dimensional-dynamics-in-high-dimensiona/), [RNNs emulating HMMs: fixed-point topology across vanilla, GRU and LSTM](https://arxiv.org/pdf/2510.25674), [GRUs as continuous dynamical systems](https://paperswithcode.com/paper/the-expressive-power-of-gated-recurrent-units) |
| Gate statistics and per-unit timescale | which units are fast, slow, saturated or dead; GRU update-gate mean gives an effective time constant | forward hooks on GRU gates | high (cheap) | same |
| Linear probes on hidden states | what each model encodes: market class, time since last event, next-change labels (the aux targets), spread, target at several horizons | hidden states on the eval fold plus labels we already have | high | standard; see [CKA vs stitching](https://arxiv.org/pdf/2106.07682) |
| CKA / representation similarity | do two models use the same representation? does it predict blend gain better than prediction correlation? | hidden states of a few models; pair gains from the bank (66+ pairs) | medium | [Kornblith et al. via stitching paper](https://arxiv.org/pdf/2106.07682) |
| Model stitching | can model A's trunk feed model B's head; functional (not statistical) similarity; a route to reuse trunks | linear stitching layer, short fit | low-medium | [Bansal et al.](https://arxiv.org/pdf/2106.07682) |
| Ablation / patching of units, layers, gates | which units carry the score; are solutions distributed or concentrated; do two models rely on the same units | zero/mean-ablate and re-score | medium | standard causal intervention; TSFM version in [Dissecting Chronos](https://arxiv.org/abs/2603.10071) |
| Sparse autoencoders on hidden states | interpretable features; **evidence suggests little superposition in forecasting models** ([Superposition is not necessary](https://arxiv.org/pdf/2605.05151), [TimeSAE](https://arxiv.org/pdf/2601.09776)) | SAE training on 256-d states | low | as cited |
| Effective rank, dead units, weight norm growth (plasticity metrics) | is the network losing capacity during training (relevant to warm-started students and long runs) | cheap from snapshots and trace `w_*` | medium | [Maintaining plasticity](https://arxiv.org/pdf/2306.13812) |
| Linear mode connectivity, model soups, permutation alignment | can same-parent students be averaged into one model with a blend-like gain | `scripts/weight_soup.py` (built), score via the queue | high (cheap) | [Git Re-Basin](https://arxiv.org/pdf/2209.04836), [width and LMC](https://arxiv.org/html/2510.08023v1) |
| Sharpness / edge of stability | is the LR pinned at 2/sharpness; why late annealing helps | Hessian top eigenvalue by power iteration on snapshots | medium | [Universal sharpness dynamics](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c8cd1b78cdae751265c88efc136e5bd-Abstract-Conference.html) |
| Gradient noise scale / critical batch size | whether batch 16 vs 32 matters; compute-efficient batch | two-batch estimator from `tr_base` (b32) and `tr_b16` traces | medium | [McCandlish et al.](https://arxiv.org/pdf/1812.06162), [critical batch size revisited](https://arxiv.org/html/2505.23971v3) |

Two-batch noise-scale estimator [hyp, uses queued runs]: with squared gradient norms `G_B` measured at batch sizes `B_small=16`
and `B_big=32`, `|G|² ≈ (B_big G_big - B_small G_small)/(B_big - B_small)` and `tr(Σ) ≈ (G_small - G_big)/(1/B_small - 1/B_big)`;
the noise scale is `tr(Σ)/|G|²`. Caveat from the literature: it predicts the critical batch only to about an order of magnitude, and
Muon needs a geometry-aware version.

## 4. Questions the user asked and how each gets answered

| Question | Method | Status |
|---|---|---|
| Train more, less, or differently? | intra-epoch eval curves, `tr_ep6`, LR/schedule arms | queued |
| Which architecture learns what? | probes on hidden states, gate timescales, ablations; capacity ladder 256/320/384/448/512 | scripts to build; ladder queued |
| Which combinations are useful? | pair/triple prediction correlation criterion (section 1), CKA vs correlation | criterion measured; CKA to build |
| Weak alone but good in a blend? | `c < r_j/r_i` rule; GRU512 (wp 0.658) is the example, +0.0019 with the best student | measured |
| Which loss plays where? | per-class and per-position wp by loss (`wmse` vs `wmse_gcorr`), `loss_family` share in trace | partly measured (class 3 is the weak spot); class-by-loss table to run |
| Does a regime-gated stack help? | class-gated blend test | measured: no |
| Free post-processing? | monotone recalibration | measured: no |

## 5. Quant brainteaser angles on this metric

- The metric is a weighted Pearson with weights `|clip(y,2)|`, scored from row 99, and 20% of labels are exactly zero (zero weight):
  invariant to a global scale of the prediction, but not to *relative* scale across rows, hence the calibration checks above
  (they came back flat, so the models already choose scale well).
- Weight concentrates in the tails and in class 3 (50% of weight, wp 0.44): a better class-3 predictor is worth more than
  gains elsewhere. A class-3-specialist teacher (trained only on class-3 sequences, `--train-classes 3` exists) is a cheap blend partner to test. [idea]
- Blend gain obeys `c < r_j/r_i`: the objective of teacher selection should be *decorrelation at equal quality*, so diversity
  has to be manufactured (capacity, architecture, input view, loss, lineage), not hoped for.
- Two targets (`t0`, `t1`): the cross-fold blend tool already computes per-target blend weights; check its LATEST output before assuming they help.
- Look for invariances the models share: a feature every model ignores is a candidate for a *different view* (raw-only teachers were
  the strongest diversity source in the teammate recipe).

## 6. Built and queued (this note's actions)

Built: `regime_blend_analysis.py`, `weight_soup.py`, `analyze_training_trace.py`, telemetry flags; soup `soup_f1_s3_a` (5 same-parent
students averaged) queued for scoring. Queued on Vast (existing machine): wave 0 (fold0 and seed/width replication of the headline
result), waves 3-7 (telemetry, schedule, SGDR snapshots, optimizer grid). To build next, in priority order:
1. hidden-state extractor plus gate/timescale statistics plus linear probes for 6-8 models (GRU256 lineage, GRU384, GRU512, LSTM256, dual);
2. CKA between those models and regression of pair blend gain on (prediction corr, residual corr, CKA);
3. class-3 specialist teacher and class-by-loss table;
4. unit ablation importance maps for GRU384 vs GRU512 vs GRU256.
