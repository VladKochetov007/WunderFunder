# Independent scout report (Opus, 2026-09-26)

Role: scout, ideator and specification critic on a path independent of `research/tree.md` and
`research/ideas/*` (read only after the hypotheses below were committed). Verification tier B
(private holdout / grouped folds; the test mask and test composition are unobservable).

Tags: **[read]** = read in the primary source at the cited location; **[gk]** = general
knowledge, not re-read in this session; **[ours]** = measured here (tiny read-only checks,
1 process, `scripts/limited`, `WNN_MEM_MAX=2G WNN_CPU_PCT=10`, no training, no GPU).
Probe scripts are in the session scratchpad and are not part of the repo. Each probe is
described in enough detail to rerun it.

## 0. What I measured (new evidence, all [ours])

| id | probe | data | result |
|---|---|---|---|
| P1 | which feature groups change per row | 8 non-holdout valid seqs | i0 book codes change on 87-99% of rows, i0 price codes on 18-60%, i1 on 10-20% (class 3) vs 73-82% (classes 0/1); rows with no i0/i1 change ≤ 0.8%. Skipping rows or updating on an event clock cannot cut inference cost |
| P2 | is `is_scored` a look-ahead rule? HGB on a-columns + change flags | 4 fit / 4 test valid seqs | causal AUC 0.776; + next-row features 0.780; + next 2 rows 0.782; + label features (t0,t1,\|t\|, target-change timing, t at +1/+20) 0.796; labels alone 0.677. Not a next-row time-sampling rule. Weak features, so the +0.02 AUC from labels only hints at outcome dependence |
| P3 | level vs within-sequence headroom, from `cache/seqsums_*` | full valid | ab20 0.6799. Oracle swap of each sequence's weighted mean (model deviations kept): 0.7136. Class 3 0.4235 → 0.5187. Class-3 per-sequence level corr (model vs truth) 0.53 vs 0.84-0.90 in classes 0-2. Class-3 level variance splits into the t0−t1 "direction" part (var 0.709, model corr 0.54) and the t0+t1 "volatility" part (var 0.068, corr 0.40) |
| P4 | WP by step-in-sequence bucket | full valid | 99-300: 0.708, 300-1k: 0.682, 1k-2k: 0.661, 2k-5k: 0.687, 5k-10k: 0.665, 10k-20k: 0.686. No warm-up deficit |
| P5 | calibration of `prop_v1` (causal propensity, fit on devA) | 16 devB seqs, 318k rows, CPU | AUC 0.904. Over-confident at the top: decile 9 mean p 0.653 vs observed 0.485, decile 8 0.273 vs 0.167. Share of loss weight on truly scored rows: p¹ 0.35, p² 0.50 (true mask 1.0, uniform 0.08). By class: mean p / observed = 1.05, 1.03, 1.14, **1.62 (class 3)**. p²·\|y\| puts 77% of the weight on class 3 against a 63% class-3 share of true scored \|y\| mass in this sample. Inside the top deciles, scored rows are less often class 3 than unscored rows (0.76 vs 0.84; 0.55 vs 0.69), so p̂ does not resolve the class. Only 16 sequences; needs a full-valid rerun (R1) |
| P6 | per-class blend weights, GRU ab20 vs baseline×0.25 | chain folds {0,1}↔{2,3} | 0.5 blend 0.6968 / 0.6630; best global w=0.6 0.6970 / 0.6633; per-class w 0.6969 / 0.6633. **No gain.** Blend over GRU alone: +0.0018 in both halves |
| P7 | direction of train↔valid chain links (`cache/chains.npz`) | all | T→V 43 chains, V→T 37, TVT 5. No time order between train and valid; the validation scout found the same |
| P8 | seed-ensemble headroom | `seqsums` | seed3 0.6799, seed4 0.6809, 2-seed average 0.6813 (+0.0009 over the mean single) |

## 1. Diagnosis

### 1.1 Where the score sits

Scored \|y\| mass by class on valid: 0.282 / 0.158 / 0.083 / **0.477**. Within-class WP of the
2-epoch GRU: 0.795 / 0.763 / 0.724 / **0.424**. Global 0.680 exceeds the mass-weighted mean of
within-class WPs (0.607) because between-class levels are predicted correctly. If test has a
class mix closer to train's (43% class-3 sequences vs 37%), the class-3 weight and the drop both
grow. The validation scout showed this alone can explain most of the 0.022 valid→test gap.

### 1.2 Why the recipe saturated (mechanisms, with evidence)

**S1. The information in x is used up under the current objective.** Every "more" axis is flat:
+18% data (train+valid) gives +0.001 on holdout (E-085); width 128/192/256, depth 1/2/3 and
LSTM vs GRU are within noise (E-081/E-083); 2→3 epochs gives +0.0006; 114 causal heuristics add
nothing linearly or as inputs (E-088/E-089); online z-scores add nothing (ab25 0.6702 vs 0.6707
devB). A model near the Bayes limit of its inputs behaves like this.

**S2. Class 3 is at an information ceiling far from moves.** The class-3 residual is not a
calibration problem. P3 shows that the class-3 per-sequence level is dominated by the t0−t1
direction component (variance 10× the volatility component), which is sequence-level drift.
Far-from-move direction has AUC 0.62-0.70 from any snapshot (E-082/E-090). Bouchaud et al.
explain why. For long, self-stabilising best queues the probability that a given queue
depletes first "only depends very weakly on the queue imbalance I, except when at least one of
the queues become small" (TQP §7.3, p.126 [read]). Imbalance is a strong, convex predictor only
near depletion (§7.2 Fig 7.2 p.120, §7.6 (iii)-(v) p.132 [read]). Our class-3 WP (0.42-0.45)
already matches the "perfect break knowledge within 20 rows" oracle (E-018, 0.42). Decoding more
of the label law or the codes cannot add much here.

**S3. Estimation variance is small.** Using the ensemble identity E[(mean error)²] = v/k +
(k−1)c/k (Goodfellow et al., *Deep Learning*, §7.11 eq. 7.50-7.51, PDF p.271 [read]) and P8,
infinite seeds would add ≈ 2 × 0.0009 = +0.0018 over a single GRU, i.e. only +0.0009 beyond the
2-seed average. A different family (the baseline) adds another +0.0018 (P6). The whole
ensembling lever is about +0.001-0.002 beyond what the current blend already has.

**S4. The binding misfit is the scoring selection, and it has had the least engineering.**
Selection is the one lever that ever moved the score by a lot:
- plain → p̂² weights: +0.053 global (ab12 0.619 → ab9 0.672, E-069);
- same devA data and init, uniform → true-mask weights: +0.029 (E-030/E-033);
- same devA data, p̂¹ → true mask: +0.007 (best epochs) to +0.012 (matched epoch 1) (E-036);
- train p̂² then a devA true-mask fine-tune: +0.0057 over either alone (E-045).

Zadrozny (ICML 2004, §2 cases 2 and 4, §3 [read]) shows that when selection depends on x only,
a "local" learner that estimates P(y|x) is asymptotically unaffected, so weights that depend on
x alone should not matter. Here they are worth +0.05. So in her terms our GRU is a "global"
learner: finite capacity, 2-epoch early stopping, errors traded across x-space. Shimodaira
(2000, JSPI 90:227 [read, abstract]) says the same thing: weighting matters exactly under
misspecification. Two practical consequences follow. First, the closer the training weights are
to the true mask, the better: p̂¹ puts 35% of the weight on scored rows, p̂² 50%, and each step
helped (P5; p̂³ lost in E-048 because the weight collapses onto an over-confident top decile).
Second, the gap between the true mask and p̂ on identical data (+0.007..0.012) means the mask
carries information that the x-only propensity does not. That information is either selection
on the outcome or a better-resolved function of x (P5: p̂ misallocates across classes).

The mask pipeline is still the least developed part of the stack: an h64 GRU fit on 741 devA
sequences (AUC 0.90), over-confident at the top, with class 3 over-predicted by about 1.6× in P5.
The final fit (`artifacts/logs/ab21.sh`) still weights **valid rows by p̂², not by their true
masks**.

**S5. The valid→test drop is not ours to fix.** It is common to every model (E-077/E-086) and is
compatible with class mix, mask softness or chain-level noise (validation scout §2.2). Shift
hedges protect rankings. They will not recover the level.

### 1.3 Where the remaining score most likely is

| lever | plausible valid gain | LB gain | evidence |
|---|---|---|---|
| selection alignment (M1-M3) | +0.003..0.008 | +0.002..0.006 if the test mask follows the valid process | S4 |
| variance reduction beyond the current blend (M4-M6) | +0.001..0.002 | similar (blend gains transferred, E-086) | S3, P8 |
| regularization / cleaner targets (M7, M9) | 0..+0.003 | uncertain | flat epoch curve, A17 |
| shift hedges (M8) | about 0 | protects rank | E-094 |

A realistic total is +0.004..0.010 on the LB (0.661..0.667). That brackets the current top
(0.6651). Another +0.03 would need information that x does not contain. The one such source we
have is the valid mask.

## 2. Methods, ranked by expected gain per unit cost

### M1. Supervise valid rows with their true mask in the final recipe
- **Source.** Our own E-030/E-033/E-036/E-045. Zadrozny 2004 §2, case 4 (s depends on x and y):
  only the true selection recovers E[y|x,s=1] [read].
- **Mechanism.** Scored-row predictions should estimate E_w[y|x,s=1]. p̂(x)² weights estimate an
  x-reweighted E_w[y|x]. On identical data the true mask beats p̂ by 0.007-0.012.
  `ab21_final_tv_e2` throws that away for the 1,873 valid sequences.
- **Change.** In `train_rnn.py`, when a sequence comes from valid, use
  `main_valid = scored + eps` (eps 0.03, cf. ab24) instead of `p̂^2 + eps`; train rows keep p̂².
  Variant C: the current fit, then 1 epoch on the valid training folds with `--mask-weight`, lr
  2e-4 (the E-045 stack).
- **Cheapest discriminating test.** Clone `artifacts/logs/ab26.sh` (same 4 chain folds, same
  seed): arm B (mixed weights) and arm C (fine-tune), then `assemble_oof.py` and
  `decide.py compare` against `ab26_tv_oof`. About 1 h GPU per arm. Quick read: fold 0 only,
  about 15 min.
- **Predicted effect.** Pooled OOF +0.003 (range +0.001..+0.008) at λ=0, positive at λ=0.02.
- **Falsifier.** OOF delta < 1 SE, or negative in the λ=0.02 or train-mix cells of the scout's
  panel.
- **Risks.** This is the most mask-specialised option; if the test mask is softer the gain
  shrinks, which is why it needs the panel check. E-045's +0.0057 was measured with devA→devB
  neighbour leakage. The leak is much less worrying than it looks, because plain devA
  fine-tuning (R1) gained only +0.0014 under the same leak. Rules: training on `valid.parquet`
  is allowed (FAQ). Mask-derived weights are already in every submission since ab9 (E-049
  flagged this as uncertain; the user accepted). No latency cost.

### M2. An outcome-aware selection model P(s | x, y) to weight train rows
- **Source.** Zadrozny 2004 §2 case 4 [read]. Importance weighting by the selection probability
  recovers the selected distribution when that probability is known. Here s is observed on valid
  and y on train, so a model of s given (x, y) fitted on valid can be applied to train as a
  training-only weight. This is privileged information in the Vapnik/Lopez-Paz sense
  (arXiv:1511.03643 [gk]).
- **Mechanism.** If the mask selects on future outcomes beyond what x shows (hinted by P2 +0.02
  AUC; P5: inside propensity deciles 0-2, scored rows have mean \|y\| 0.70-0.83 vs 0.58-0.62 for
  unscored), then no x-only weight can reproduce the scored-row target distribution. Weights
  w = P(s=1|x,y) make the weighted train distribution equal the scored one. The same trick
  transfers the true-mask gain of M1 to 6× more sequences.
- **Change.** GBM (monotone constraints optional) on valid rows, coarse-component K-fold. Inputs:
  logit p̂(x) (from M3), class, the spread code, t0, t1, \|t\|, branch code, rows since/until the
  target change, number of target changes in the next 100/1000 rows, next-row a-column deltas.
  Isotonic calibration per class. Weight train rows by P̂(s|x,y) to the power 1 (with
  y-dependent weights only k=1 is unbiased), clipped to [ε, 1].
- **Cheapest discriminating test.** (1) CPU, about 30 min, ≤ 4 GB: held-out AUC and log-loss of
  [x-only] vs [x + labels] on component-held-out valid. Stop if the AUC gain is < 0.005.
  (2) Otherwise one train-only GRU (about 12 min GPU) with the new weights vs p̂², on full valid.
- **Predicted effect.** Selection-model AUC +0.01..0.03; GRU +0.002..0.005.
- **Falsifier.** Label features add < 0.005 AUC, meaning selection is effectively x-only (then do
  M3 only), or the GRU delta ≤ 0.
- **Risks.** Weights fitted on labels can overfit (use K-fold, a small GBM, clipping). Inference
  stays causal. Nothing crosses sequences.

### M3. A better x-only propensity (all valid, calibrated, class-aware) plus class-mass matching
- **Source.** P5; Shimodaira 2000 (misspecified model → weights matter) [read, abstract]; the
  untested H-019 (the class-balance arm ab4_I never finished: `artifacts/logs/ab4.out`).
- **Mechanism.** In P5, p̂ is over-confident at the top and over-predicts class 3 by about 1.6×.
  Classes 0-2 therefore get about half their share of the loss mass, and classes 0/1 are where
  residual structure still transfers (E-092, if it survives the scout's component recheck).
- **Change.** Refit the propensity on all valid via component K-fold (out-of-fold p̂ on valid
  rows), h128, class one-hot input, isotonic calibration per class. Then rescale per-class loss
  mass so that each class's share of Σ w·\|y\| on train equals its share of scored \|y\| mass on
  valid (0.28 / 0.16 / 0.08 / 0.48). Re-tune the power in {1, 1.5, 2}.
- **Cheapest discriminating test.** R1 first: prop_v1 calibration per class on all valid
  (inference only, minutes). If the class-3 ratio is ≤ 1.15 there, drop the class-mass part.
  Then one GRU run.
- **Predicted effect.** +0.001..0.003.
- **Falsifier.** No gain over p̂². E-055 is counter-evidence for the class part (a class 0-2
  expert did not improve classes 0-2).
- **Risks.** Low. It is a prerequisite for M2 anyway.

### M4. Distil the ensemble (GRU seeds + baseline×0.25) into the deployed GRU
- **Source.** Hinton et al. 2015 (arXiv:1503.02531) [gk]. Menon et al. 2020 (arXiv:2005.10419),
  §3 Lemma 1: the Bayes-distilled risk has lower variance than the one-hot empirical risk;
  Prop. 3: bias-variance trade-off for an imperfect teacher [read]. For regression the analogue
  is the tower property: training on E[y|x] instead of y lowers the variance of the empirical
  risk without moving the minimiser.
- **Mechanism.** Per S3 the ceiling is about +0.0018 (seeds) plus +0.0018 (baseline). A student
  recovers part of that at 40 us/row, which frees latency for M5. `cache/pred_baseline_train.npy`
  already exists (C3 in the tree).
- **Change.** `--distill ab20_gru128_e2,ab23_e2_seed4,baseline*0.25` with alpha in {0.3, 0.5}.
  Keep the tilt: weight |y| and target (1−α)y + α·teacher, as the code already does. Store
  teacher predictions for train as float16.
- **Cheapest discriminating test.** One train-only student run (about 15 min) vs ab20 on full
  valid, then blend(student, base) vs blend(ab20, base).
- **Predicted effect.** Student +0.0008..0.0015 over a single GRU; blend +0.0005..0.001 over
  the current blend.
- **Falsifier.** Student ≤ single GRU, or the student blend ≤ the current blend.
- **Risks.** The teacher shares the train-regime bias; the gain is capped by S3.

### M5. Latency engineering so that three members fit
- **Source.** Kalchbrenner et al. 2018 (WaveRNN, arXiv:1802.08435), §1 eq. (1)
  T(u)=|u|·Σ(c(op)+d(op)); §5: for their CPU kernels "the limiting factor for all kernels is
  bandwidth and not flops"; large sparse beats small dense at equal parameters [read].
  Our E-005: 12-15 GFLOPS gemv on one core is bandwidth-bound. Int8 inference: Jacob et al. 2018
  (arXiv:1712.05877) [gk].
- **Mechanism.** At the E-077 ratio (19.5 LB-min for 40.9 local us/row, i.e. about 0.48 min per
  us), the 94.5 us blend is about 45 LB-min. A third member does not fit unless the ONNX baseline
  is moved to native code (exact conversion exists, E-085), the callbacks are fused (tree E1),
  and weights are f16 or int8 (half or a quarter of the bytes; two h128 members stay L2-resident).
- **Cheapest discriminating test.** Rust microbench: fused 2- and 3-member step with f32 vs f16
  weights (F16C convert) on one pinned core; check equivalence (max abs diff, WP on
  `cache/valid_subset20.parquet`).
- **Predicted effect.** Speed-up 1.4-2×; a third member (M4 student or a different-seed GRU)
  adds +0.0005..0.001.
- **Falsifier.** Speed-up < 1.3×, or the quantization WP change exceeds 0.0003.
- **Risks.** The LB CPU is unknown; keep ≥ 20% margin under 60 min. Determinism is fine with
  fixed kernels.

### M6. A model soup of children fine-tuned from one parent (zero latency)
- **Source.** Wortsman et al. 2022 (arXiv:2203.05482): averaging weights of models fine-tuned
  from one initialization "often improves accuracy and robustness" and "improves
  out-of-distribution performance" at no inference cost [read, abstract]. Frankle et al. 2020
  (arXiv:1912.05671): after early training, runs from a shared state are stable to SGD noise and
  linearly connected [read, abstract].
- **Mechanism.** SWA along one trajectory added nothing (E-052), because a OneCycle tail gives no
  diversity. Children with different data orders and ε/lr settings do differ, and the soup
  averages their errors inside one basin.
- **Change and test.** Parent = the 1.5-epoch checkpoint of the final recipe. Four children,
  0.5 epoch each (seed, lr 5e-4/1e-3, eps 0/0.03), about 25 min GPU. Compare the uniform soup,
  the best child, and the output average; first check that loss along the linear path between
  two children has no barrier.
- **Predicted effect.** +0.0005..0.001 over the best child.
- **Falsifier.** Soup < mean child.
- **Risks.** Small; the GRU permutation symmetry is fixed by the shared parent.

### M7. Regularised longer training
- **Source.** Arpit et al. 2017 (arXiv:1706.05394): networks learn simple patterns before
  memorising noise, and tuned dropout degrades noise fitting "without compromising
  generalization on real data" [read, abstract]. Merity et al. 2017 (arXiv:1708.02182):
  DropConnect on hidden-to-hidden weights (weight-dropped LSTM) plus averaged SGD [read,
  abstract]. Gal & Ghahramani 2016 (arXiv:1512.05287), variational RNN dropout [gk].
- **Mechanism.** The labels are very noisy (class-3 far-from-move rows are close to coin flips).
  `--dropout` defaults to 0, and there has been no regularization sweep (wd fixed at 1e-2).
- **Test.** Variational input dropout 0.1 + weight-drop 0.2 on W_hh, 4 epochs, one run (about
  25 min); read every epoch on full valid.
- **Predicted effect.** 0..+0.002.
- **Falsifier.** The best epoch is still ≤ 2 and the gain is < 1 SE.
- **Risks.** Weight-drop needs a custom GRU (cuDNN GRU does not support DropConnect).

### M8. Remove regime crutches found by adversarial validation
- **Source.** Pan et al. 2020 (arXiv:2004.03045), adversarial validation for drift [read,
  abstract]; E-094 drivers: i0/i1 volume-code levels, a5/a6 statistics, a7 mean.
- **Mechanism.** Features that identify the regime rather than predict y do not transfer to a
  new regime. i1 carries little direction information (E-013/E-017) but separates train from
  valid.
- **Cheapest test (inference only, about 5 min GPU).** Permutation importance: shuffle the i1
  volume codes across sequences within a class on valid. If ΔWP > −0.001, stop (the feature is
  unused). Otherwise train one variant with whole-sequence i1-volume dropout (p=0.3).
- **Predicted effect.** Overall ±0.001; +0.002 in the most train-unlike tercile (V2) if it is a
  crutch.
- **Falsifier.** No gain in the novelty tercile.
- **Risks.** i1 activity also drives `is_scored`, so keep it for the mask models (M2/M3).

### M9. Rao-Blackwellised targets for class 3 (privileged teacher)
- **Source.** Menon et al. 2020 §3 (lower-variance distilled risk) [read]; tree F3.
- **Change.** The teacher sees x≤t plus the direction and delay of the next i0 mid move (from
  the tick decoder) and outputs E[|y|y|x,z] and E[|y||x,z]. The student trains with weight
  E[|y||x,z] on target E[|y|y|x,z]/E[|y||x,z]. This preserves the WP-optimal minimiser
  E[|y|y|x]/E[|y||x] by the tower property while removing noise that z explains.
- **Test.** Class 3 only; teacher about 15 min, student about 15 min.
- **Predicted effect.** 0..+0.003 (class-3 share 0.48).
- **Falsifier.** Student ≤ the plain GRU in class 3.
- **Risks.** Teacher bias; the most plumbing of any method here, so it comes last.

### M10. Class-3 "race" features (long shot, cheap)
- **Source.** TQP §7.2-7.3 [read]: direction information concentrates where a best queue is
  small.
- **Change.** min(bid-IC, ask-IC)/its running median, times imbalance, plus time since that
  minimum was last below its 10th percentile (ICA proxies from E-090). Run `lift_probe.py` on
  GRU residuals, class 3, with component halves.
- **Predicted effect.** ≤ +0.001.
- **Falsifier.** Lift ≤ 1 SE (the E-088 depth family was also flat).

**Falsified during this scouting (do not pursue):** per-class blend weights (P6); warm-up or
early-row fixes (P4); per-sequence level recalibration (P3: the class-3 level is directional
drift); event-clock compute savings (P1); a next-row time-sampling explanation of the mask (P2).

## 3. Validation and selection recommendations

V1. **Adopt the validation scout's protocol** (`research/lit/validation.md` §3): the sequence is
the unit, chains/components are the groups, metric sums are pooled, and decisions use paired
deltas. My P7 agrees that there is no train/valid time order, so recency weighting has no basis.

V2. **Add a novelty stratum.** Take the out-of-fold train-vs-valid probability per valid sequence
from the E-094 model (`scripts/adversarial_validation.py`) and split valid into terciles by
train-unlikeness. Report every paired delta per tercile. Test is another regime (AUC 0.99
separability), so treat the most train-unlike tercile as the closest proxy. A method that gains
only in the train-like tercile is fitting regimes. Shift hedges (M8, normalization variants)
must be neutral or positive in the novel tercile. This costs seconds with `seqsums`.

V3. **Mask methods need out-of-fold everything.** Propensity and selection models (M2/M3) must be
fitted on coarse-component folds. A p̂ fitted on devA must not generate claims on devB (49% of
devB sequences have a devA chain neighbour). M1 must use the ab26 folds, with valid rows used for
true-mask training in the training folds only. Report λ ∈ {0, 0.02} and the train-mix cell for
every mask method, because mask specialization is exactly what those cells stress.

V4. **Never read absolute levels.** Class-3 per-sequence levels are drift (P3), so absolute WP of
any subset moves with drift luck (chain SE 0.014, validation scout §2.2). Only paired deltas
carry information.

V5. **Seed noise.** Seed 3 vs seed 4 differ by 0.0010 on full valid (P8). Arms with a predicted
delta < 0.002 need ≥ 2 seeds, or a seed-noise term added to the bootstrap SE.

V6. **Leading indicators.** For M2/M3, selection-model held-out AUC and log-loss per class are
cheap, low-noise, and settle go/no-go before any GRU run.

V7. **LB discipline.** One LB slot per mechanism family, only after V1-V5 pass. The λ-cell does
not identify the mask softness (validation scout §2.2); the LB is the only instrument that does.
If M1 wins at λ=0 and in the train-mix cell but loses at λ=0.02, that is the one submission
worth making to resolve the question.

## 4. Disagreements with the current tree

D1. **Too much budget on decoding and new features** (A1-A2, B1-B5). S1 and S2 say the information
in x is used up and the class-3 residual is directional drift (P3, TQP §7.3). Label-law and code
decoding remain useful for understanding but have low expected LB value. B4 (stale rows) is
exactly where TQP predicts imbalance carries little information.

D2. **The mask branch (A3) is framed as "understand the mask / is the test softer?".** It is the
largest demonstrated lever (+0.053 from p̂², +0.007..0.012 true mask over p̂ on identical data)
and the least engineered component. Promote M1-M3 above C1-C5. The final recipe ignores true
valid masks today.

D3. **D3 recency weighting:** no support (P7 and the validation scout).

D4. **C4 architectures (minGRU/LRU/attention):** capacity is not binding (S1). Pursue them only as
speed enablers or diversity for M5.

D5. **E2 propensity-gated compute:** reject. Every row needs a prediction, a softer test mask would
punish skipped rows, and M5 removes the need.

D6. **"λ=0.02 ≈ LB" as an objective:** use it as a robustness cell only (V3, V7). Tuning to it
would systematically penalize the strongest lever (M1) on an unidentified assumption.

D7. **Agreements, extended:** C3 (distil the blend) → M4 with the S3 ceiling; E1 (fused Rust
blend) → M5 with f16/int8 weights; V3 (K-fold) → M1 runs on those folds; F3 (privileged
teacher) → M9 with a tilt-preserving target.

D8. **F2 (LOB simulator pretraining) and F4 (in-context meta-learning):** poor value per cost
given S1.

## 5. Requests (bigger than my limits allowed)

- **R1.** prop_v1 calibration per class on all valid (inference, minutes). Confirms or kills P5's
  1.6× class-3 over-prediction.
- **R2.** M2 test (1): label-aware selection model vs x-only, component folds (CPU about 30 min,
  ≤ 4 GB).
- **R3.** M1 arms B/C on the ab26 folds (GPU about 1 h per arm; fold 0 first).
- **R4.** Out-of-fold adversarial scores per valid sequence for V2 (minutes).

## Sources

Local:
- Goodfellow, Bengio, Courville, *Deep Learning* (2016),
  `~/Documents/Trading/Кирилл Рыжиков/deeplearningbook.pdf`, §7.11 eq. 7.50-7.51, PDF p.271.
- Bouchaud, Bonart, Donier, Gould, *Trades, Quotes and Prices* (2018),
  `~/Documents/Trading/Trades,_Quotes_and_Prices_...pdf`, ch. 7: §7.2 pp.119-120, §7.3 p.126,
  §7.6 p.132.

Web:
- Zadrozny, ICML 2004, [PDF](https://www.math.kth.se/matstat/gru/sf2935/zadrozny.pdf), §2-3.
- Shimodaira, JSPI 90(2) 2000, [link](https://www.sciencedirect.com/science/article/abs/pii/S0378375800001154).
- Menon et al., [arXiv:2005.10419](https://arxiv.org/abs/2005.10419), §3.
- Wortsman et al., [arXiv:2203.05482](https://arxiv.org/abs/2203.05482).
- Frankle et al., [arXiv:1912.05671](https://arxiv.org/abs/1912.05671).
- Kalchbrenner et al., [arXiv:1802.08435](https://arxiv.org/abs/1802.08435), §1 and §5.
- Merity et al., [arXiv:1708.02182](https://arxiv.org/abs/1708.02182).
- Arpit et al., [arXiv:1706.05394](https://arxiv.org/abs/1706.05394).
- Pan et al., [arXiv:2004.03045](https://arxiv.org/abs/2004.03045).
- Passalis et al., DAIN, [arXiv:1902.07892](https://arxiv.org/abs/1902.07892).
- Kim et al., RevIN, ICLR 2022, [poster](https://iclr.cc/virtual/2022/poster/6034). Noted but not
  ranked: causal online normalization already failed on devB (ab25).
