# Public brief: ideation round toward a 0.700 public score

Date: 2026-10-08. Sanitized for external model ideators. No code, credentials or private paths.

## 1. Task

A competition asks for a causal online predictor of two anonymized targets, `t0` and `t1`, from limit-order-book style rows of two instruments (`i0` predicted, `i1` input only).

- Data: 10,607 training sequences and 1,873 validation sequences. Each sequence has 20,000 consecutive rows, one row per time step. Each row has 112 float features: L11 bid/ask price-like and volume-like columns for both instruments, four trade-price and four trade-volume columns per instrument, and eight additional features. Column index is an identifier, not a depth level.
- Sequence structure: steps 0 to 98 are warm-up (no output). Steps 99 to 19,999 require two finite predictions. Only a mask-selected subset is scored.
- Rules: process rows in order, update state on every row, reset state on a new sequence id, use only the current and earlier rows (strictly causal), and do not use the sequence id as a feature. Test sequences are hidden and streamed row by row to the submitted callback.
- Serving limits (official): one vCPU, 16 GB RAM, 60 minutes for the full test set, CPU only, no internet, submission ZIP at most 20 MB, Python 3.11.
- Metric: global weighted Pearson correlation, computed separately for `t0` and `t1` over all scored rows pooled across sequences, then averaged. Predictions and targets are clipped to plus or minus 2 before scoring. Weights are proportional to the magnitude of the clipped label. Training-target clipping is a separate experimental choice.

## 2. Status

- Our best public score: 0.6660. A teammate team reports 0.6674. The project owner reports that a public score of 0.700 exists on this competition. We have not verified that report.
- Goal: find a mechanism or combination that moves the public score above 0.700 under the same serving limits.
- Compute: GPU for research (Kaggle, a few cloud rentals, one local GPU). Research-time models may be larger than the serving limit, but any served endpoint must run within the 60 minute CPU budget. One GRU256x2 trunk already costs about 33 of those 60 minutes.

## 3. Known findings (our measurements; standalone results are weak evidence)

- Target structure: on one target-class (class 3, about 46% of the metric weight) the labels follow an exact algebraic relation to a latent fair value F, with `t0 = F - ask` and `t1 = bid - F`. The unknown quantity is F.
- The current recipe is a GRU with two layers of width 256, propensity-style sample weights, auxiliary heads, Muon optimizer with chunk length 1000, and bootstrap-style parent targets. Distillation from teacher mixtures helps. A fine-tuned head on a graded correlation objective added about +0.0013 public.
- Two seeds of the same recipe averaged: about +0.0004 public. Ensembling same-recipe seeds is small.
- Attention, TCN, GRU384 and numeric embeddings were weaker as standalone models in our tests. Their contribution as teachers or blend members was not established.
- Known difficulty: quiet rows (small or zero movement) carry large errors; the latent component of the labels is hard to model from the inputs.

## 4. Already tried or proposed (do not repeat unless you add a new mechanism or interaction)

One line each. Names are our internal arm names.

1. Causal attention teacher (segment memory), used as teacher or temporal summary.
2. Transformer predictions transferred into the GRU student; quiet-row transfer; re-anchor interaction.
3. Privileged teacher with future information offline (cross-fitted), and learnable projection of privileged structure.
4. Convergence-aware attention and event-clock context.
5. Cross-fitting of teacher lineages and ensemble targets (a local positive effect, public not confirmed).
6. Compression of heterogeneous teacher recipes into a GRU256 pair (fit packet not yet run).
7. Two-stream teacher; function-preserving widening (Net2Net-style) of the student, including symmetry breaking.
8. Relative-distance, angle-wise and temporal relational distillation; paired-view robust relational distillation.
9. Shared-recurrent teacher families: parallel linear recurrence, causal TCN, block attention, reservoir, liquid retention, sliding DFT, EDMD, random Fourier features, invariant pooling, Gaussian information bottleneck, path signatures, semantic group encoder, causal scattering features.
10. Event clock and quiet-row anatomy; residual memory (diagonal, content, erase-write); quiet-row output holds (failed screen).
11. Stationarity state and sparse feature cliques; interaction grower; neighbor-dependence regime cue.
12. Fixed readout mixture; direct-label contextual readout (a graded positive follow-up); conditional bilinear residual readout (no promotion).
13. Soft value-grid head; window-innovation auxiliary; matched robust loss and target range.
14. Measured denominator (chunk, fixed or shuffled mass); cadence of recurrent gradient credit.
15. Optimizer families proposed but not tested: AdamW, RAdam, AdaBelief, Adafactor, Muon variants, NorMuon, Polar Express, Scion, Shampoo, SOAP. Muon hybrid is our current default.
16. Disagreement-guided student optimization (completed; exposed error location signal, no fit).
17. Coherent affine nuisance augmentation; causal auxiliary-channel anti-shortcut corruption.
18. Raw regime context beside causal normalized innovations; causal window distribution shape; lagged and cross-channel coupling; bounded nonlinear morphology features; monotone calibration before causal contrasts.
19. Several earlier batches of literature-based ideas (microstructure, market efficiency, random matrix theory, optimal execution, contrarian metric ideas, regularization for noisy labels). Most were used as hypotheses; the ones tested are listed in items above.

## 5. Constraints on any idea

- Causality at serving: a prediction may use only earlier rows. Future information is allowed only as offline training targets.
- Data: training and validation data only. Models must not use outside market data or the hidden test rows.
- Serving: the complete endpoint (including the state update on every row) must run within 60 CPU minutes on one vCPU with 16 GB and a 20 MB ZIP.
- Validation: a local comparison is valid only when fitting membership, teacher ancestry and evaluation exposure are controlled. Public results confirm transfer.

## 6. What to return

6 to 10 distinct, testable ideas. For each: mechanism family; the source component or passage it came from; the structural analogy; the intervention; the predicted observable (which score or which row subset should move, and in which direction); the expected effect size with a stated uncertainty; the cheapest discriminating test on a fold-0 panel; the serving-cost impact in CPU minutes; and whether it is novel or a reframing of an item in section 4. At least one idea should be unconventional and transferable, such as a sampling rule, a decomposition, a state representation, a normalization, a diagnostic, an approximation or a control. Say when you found no defensible correspondence rather than forcing one.
