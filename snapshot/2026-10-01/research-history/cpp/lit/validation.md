# Validation methodology for the Wunder connectome task (scout, 2026-09-26)

Status: literature read + zero-training data checks. Nothing here was trained.
Confidence tags: **[read]** = read in the source text at the given location;
**[skim]** = skimmed, gist only; **[gk]** = general knowledge, not in the local sources,
verify before relying on it; **[ours]** = our own derivation or measurement.
Sources: AFML = López de Prado, *Advances in Financial Machine Learning* (2018), book pages;
MLAM = López de Prado, *Machine Learning for Asset Managers* (2020), book pages;
mlfinlab = mlfinlab 1.5.0 docs (PDF pages); Masters = T. Masters' C++ sources under
`~/Documents/Trading/Masters/`.

## 1. Claims from the literature

### 1.1 Leakage in CV: purging and embargo
- C1 **[read]** AFML 7.3 (p.104-105): k-fold fails in finance for two reasons: non-IID
  observations (leakage) and reuse of the test set (multiple testing / selection bias).
  Leakage needs (X_i, Y_i) ≈ (X_j, Y_j) across the split; X_i ≈ X_j alone, or Y_i ≈ Y_j alone,
  is not leakage. The harm is false discoveries with *irrelevant* features.
- C2 **[read]** AFML 7.4.1 (p.105-107): *purging* removes from training every observation
  whose label interval overlaps a test label interval (three overlap conditions, Snippet 7.1).
  Diagnostic: with leakage, CV score keeps rising as k → T; once purging suffices it plateaus
  beyond some k*.
- C3 **[read]** AFML 7.4.2 (p.107-108): *embargo* additionally drops training observations
  that immediately *follow* a test block (serially correlated features), h ≈ 0.01 T "often
  suffices"; not needed before the test block. Snippet 7.3 `PurgedKFold` assumes contiguous
  test blocks, no shuffling.
- C4 **[read]** AFML 12.2 (p.161-162): embargo is unnecessary in walk-forward (training
  precedes testing) but purging is still needed where label ends (t1) cross into the test set.
- C5 **[read]** mlfinlab 10.1-10.2 (PDF p.107-111): `PurgedKFold`, `CombinatorialPurgedKFold`,
  plus *stacked* variants "used for multi-asset datasets" (one purge per asset stream). Our
  sequences are the analogue of many asset streams.

### 1.2 Overlapping labels, uniqueness, sequential bootstrap
- C6 **[read]** AFML 4.2-4.5 (p.59-65): overlapping outcomes make labels non-IID; average
  uniqueness ū_i (reciprocal harmonic mean of concurrency over the label's span) measures
  redundancy; bootstrapping non-unique observations makes in-bag and out-of-bag nearly
  identical, so OOB accuracy is "grossly inflated". Remedies: max_samples = mean ū, or the
  sequential bootstrap (draw probability ∝ uniqueness given previous draws).
- C7 **[read]** AFML 4.7 (p.70): time-decay sample weights, decaying on *cumulative
  uniqueness* rather than chronological time.
- Mapping **[ours]**: our labels overlap within a sequence (next ~4 book events) and, as
  section 2.1 shows, across the junction into the *next* sequence of the same stream (the last
  rows of A are labelled from B's first book events, a few dozen rows out of 20,000). So the
  resampling unit is the sequence or a block of adjacent sequences, never the row. Row-level
  bootstraps or row-level CV would be the "spilled blood tubes" case of AFML 4.2.

### 1.3 Walk-forward vs CV vs combinatorial CV
- C8 **[read]** AFML 12.2.1 (p.162): walk-forward (WF) tests a single path, is easily
  overfit, and early decisions use little data; "it is as easy to overfit a walk-forward
  backtest as to overfit a walk-backward backtest".
- C9 **[read]** AFML 12.3 (p.162-163): CV backtests trade historical interpretability for
  equal-size training sets and a longest-possible OOS; leakage must be purged.
- C10 **[read]** AFML 12.4 (p.163-166): CPCV: N groups, all C(N,k) train/test splits,
  φ[N,k] = k/N·C(N,N−k) backtest paths; k=2 gives N−1 paths while training on 1−2/N of data.
  12.5 (p.166-167): variance of the mean over paths is φ⁻¹σ²(1+(φ−1)ρ̄), so more,
  less-correlated paths lower false-discovery risk.

### 1.4 Selection bias, deflation, PBO
- C11 **[read]** AFML 11.5 (p.153-154) and MLAM 1.4.2 (p.8-9): test-set overfitting
  (selection bias under multiple testing) is not detected by any backtest; remedies are
  (i) record every trial and deflate, (ii) resample many test sets (CPCV), (iii) Monte Carlo
  on synthetic data. "The purpose of a backtest is to discard bad models, not to improve them."
- C12 **[read]** MLAM 8.5 (p.108-109), AFML 12.5 (p.166): False Strategy theorem:
  E[max_k SR_k] ≈ √V[SR_k]·((1−γ)Z⁻¹(1−1/K) + γZ⁻¹(1−1/(Ke))), γ = Euler–Mascheroni.
  Factor for K = 2/5/10/20/50/100 trials: 0.52/1.19/1.58/1.90/2.28/2.53 **[ours, computed]**.
- C13 **[read]** AFML 14.7.2-14.7.3 (p.203-205), MLAM 8.7 (p.113): PSR and the deflated
  Sharpe ratio: PSR with the benchmark set to the expected maximum under the null.
  Third law of backtesting (AFML p.205): report every result with all trials behind it.
- C14 **[read]** MLAM 8.7.1-8.7.2 (p.114-116): the number of *effective* trials is estimated by
  clustering the trials' return correlation matrix (ONC); correlated variants of one idea
  count as ~one trial; the variance across trials is taken from cluster-level aggregates.
- C15 **[read]** MLAM 8.8 (p.116-118): FWER and Šidák α = 1−(1−α_K)^{1/K} with K from ONC.
- C16 **[read]** MLAM 8.2 (p.105-106): precision of a "significant" finding depends on prior
  odds θ of true strategies; with θ = 1/99, α = 0.05, β = 0.2 about 86% of positives are false.
- C17 **[read]** mlfinlab 2.1 (PDF p.7-10): Harvey & Liu "haircut Sharpe" and "profit
  hurdle" with Bonferroni / Holm / BHY adjustments; MinTRL and PSR helpers (PDF p.13).
- C18 **[read]** AFML 11.6 (p.155-157): CSCV/PBO algorithm (Bailey, Borwein, López de Prado,
  Zhu 2017). Performance matrix T×N (N configurations); split rows into S even disjoint blocks
  (observations inside a block keep their order); for each of C(S, S/2) combinations pick the
  IS-best configuration, find its relative OOS rank ω̄, logit λ = log(ω̄/(1−ω̄));
  PBO = P(λ ≤ 0). Condition: the criterion must be computable on any subset of rows.
  Also: regress OOS on IS performance of the selected configuration (degradation slope).
- C19 **[read]** Masters `CSCV_MKT/CSCV_CORE.CPP`: same algorithm as a combination
  enumerator over even `n_blocks`; relative rank = (#systems with OOS ≤ best's OOS)/(N+1);
  counts combinations where the IS-best is at or below the OOS median; comment: number of
  competitors "should be large enough to reduce granularity". `CRITER.CPP`: criterion is a
  plug-in (mean return or profit factor).
- C20 **[read]** Masters `SELBIAS/SelBias.cpp`: three independent sets: optimise each model
  family on set 1, *choose* between families on set 2, measure the chosen one on set 3.
  Selection bias = (chosen model's set-2 score) − (its set-3 score), averaged over
  replications with a t-statistic. `TRNBIAS/TrnBias.CPP`: training bias = IS − OOS of the
  optimised system over replications.
- C21 **[read]** Masters `MCPT_TRN/MCPT_TRN.CPP`: Monte-Carlo permutation test of the *whole
  training procedure*: permute price changes (endpoints fixed), rerun the full optimisation
  on each permutation; p = fraction of permutations whose optimised score ≥ the original;
  training bias = mean optimised score on permutations (minus the trend component);
  "unbiased return" = original − training bias; skill = unbiased return − trend component.
- C22 **[skim]** López de Prado & Zoonekynd (2024), *Why has factor investing failed?*
  (sections 3.3, 5): specification choices by in-sample fit (R², adjusted R²) favour models
  that condition on colliders; such models look better in-sample and fail under parameter
  shift; "robustness is ... a virtue ... of the causal content". Mapping **[ours]**: the
  scoring mask is a selection variable; a model or a validation rule whose edge depends on
  scored-row structure (propensity weights, scored-row level calibration) is exposed to a
  change in the mask process, which is what the critic found for ab18.
- C23 **[read]** MLAM 6.3.2 (p.81-82), AFML ch.8: feature importance must be out-of-sample
  (MDA under purged CV); substitution effects hide correlated features; cluster features first.

### 1.5 From general knowledge (not in the local sources)
- G1 **[gk]** White (2000) Reality Check, Hansen (2005) SPA, Romano–Wolf step-down: bootstrap
  the maximum of centred performance differences over all K candidates vs a benchmark.
- G2 **[gk]** Varma & Simon (2006), Cawley & Talbot (2010): the CV score of the selected
  configuration is optimistically biased; nested CV removes it. Tibshirani & Tibshirani (2009)
  give a cheap bias estimate from the fold-wise gap between the chosen and fold-best configs.
- G3 **[gk]** Blum & Hardt (2015) "The Ladder" and Dwork et al. (2015) reusable holdout:
  adaptive reuse of a leaderboard/holdout overfits it; only accept a new best when it improves
  by more than a noise-scaled step.
- G4 **[gk]** Roberts et al. (2017, Ecography): block CV for temporally/spatially/hierarchically
  structured data; blocks larger than the dependence range. Bergmeir, Hyndman & Koo (2018):
  plain K-fold is valid for purely autoregressive models when residuals are uncorrelated.
- G5 **[gk]** Adversarial validation (Kaggle practice): classify train vs test; use the
  most test-like training rows as the holdout, or importance-weight by p/(1−p).

## 2. What our data says (zero-training checks, 2026-09-26) **[ours]**

All numbers come from cached validation predictions and a first/last-row scan of both
parquet files. Scripts live in the session scratchpad (`edges.py`, `adjacency2.py`,
`chains.py`, `stage0.py`, `stage0b.py`); nothing was written to `cache/`. Copy them to
`scripts/` before relying on them.

### 2.1 Sequences are NOT independent: they are consecutive chunks of continuous streams
- File order and `seq_ix` carry no time: `seq_ix` are random 31-bit ids and the files are
  sorted by them. Lag-1 autocorrelation of per-sequence statistics (class, target std, event
  rates, a2/a3/a6 means) in `seq_ix` order is |r| ≤ 0.036 (SE 0.023).
- Adjacency *is* recoverable. Score(A→B) = Σ over the 112 features where last-row(A) equals
  first-row(B) of −log p(value). Consecutive rows inside a sequence score median 257 (10% quantile
  68); random cross-sequence pairs score median 5 (99.99% quantile 330). The best successor of
  each validation sequence scores median 238, i.e. like a within-sequence step.
- Over train+valid (12,480 sequences) the mutual-best rule (score ≥ within-sequence 10%
  quantile, margin > 5 nats over the runner-up) gives 9,632 links. Checks: market class
  agrees across 99.99% of links (random pairs 31%); both targets are identical across the
  junction for 86.7% of links (consecutive rows inside a sequence 84.7%; random pairs 5.1%).
- Chains: 2,848 (846 singletons), length median 3, p90 9, p99 28, max 67 sequences (1.34M rows).
  Class-3 chains are longest (median 8). About 2,100 chain ends are ambiguous ties, not missing
  neighbours, so true streams are longer. A looser union (best match above threshold, same class,
  no margin rule) gives 694 "coarse components" (max 666 sequences), an upper bound on grouping.
  A sharper linker exists for class 3: the last rows of A must satisfy the E-079 label law with
  the first 4 book events of B.
- Train and valid are mostly separate blocks, but not separated by one time cutoff. Of 8,249 train
  links, 27 point into valid; of 1,383 valid links, 31 point into train. 56 chains mix the two, in
  both orders (T+V+ 15, V+T+ 9, VT+ 11, T+V 3, V+T 5, ...). Valid therefore looks like
  **blocks of streams**, not "the period after train". The test set is probably more blocks of
  the same kind. That is a guess; the test cannot be linked.
- Class mix of the blocks differs a lot: valid [0.212, 0.185, 0.230, 0.372], train
  [0.123, 0.119, 0.329, 0.430] for classes 0..3.
- Rule reminder (competition scout): no cross-sequence adaptation at inference. Adjacency is a
  *validation-design* fact only, never an inference feature.

### 2.2 What adjacency does and does not break
- Paired model deltas: grouping hardly matters. Bootstrap SE of the full-valid delta
  by sequence / by chain / by coarse component: e2−base×0.25 (+0.0072) 0.0015/0.0015/0.0014;
  ab18−base (+0.0030) 0.0016/0.0015/0.0016; blend−e2 (+0.0017) 0.0005/0.0005/0.0005;
  seed4−seed3 (+0.0010) 0.0004/0.0004/0.0005. Neighbour correlation of each sequence's
  jackknife influence on a *delta* is ≈ 0.01 (shuffled ≈ 0.01). The current decide.py rule is
  sound for ranking on valid.
- Absolute levels: grouping matters a lot. Bootstrap SE of e2's full-valid WP: sequence 0.0066,
  chain 0.0140, coarse component 0.0378. Neighbours share difficulty: per-sequence WP of e2
  correlates 0.64 between chain neighbours vs 0.53 for same-class random pairs. The
  valid→test drop of ~0.022 is 3.3 sequence-SE but only 1.6 chain-SE. It is compatible with
  test composition alone.
- Class mix alone moves the level by a lot: base×0.25 scores 0.6728 at the valid mix and 0.6273
  at the train mix. Three clean LB points (base×1 0.5935, base×0.25 0.6503, ab18 0.6488) are fit
  about equally well by λ=0.02 mask softening (RMSE 0.0033), by a mix 60% of the way to train's
  (RMSE 0.0027), or by both (α=0.2, λ=0.015, RMSE 0.0016). The level shift is **not
  identified**. None of these variants reproduces the ab18<base flip: all predict +0.0014 to
  +0.0026 vs LB −0.0015, so noise is still needed.
- Dev splits leak neighbours. 49% of devB sequences (and 49% of holdout) have a direct chain
  neighbour in devA; 93% (91%) share a coarse component with devA. Anything fitted on devA and
  scored on devB can have been optimistic: the propensity model, symreg terms, GBM residual
  (E-092), and blend weights. Train-only models scored on valid are affected much less, since
  only 188 valid sequences share a chain with train.

### 2.3 Selection bias on valid is small for the current config family
- CSCV (C18/C19) with S=16 blocks, all 12,870 splits, criterion = pooled WP from summed
  sequence sums. Family of 20 configs: cached models, baseline scales 0.2-1.0, blends,
  seed average. Blocks were built from chains, from coarse components, or from random
  sequences; λ ∈ {0, 0.02}. PBO = 0.000 (N=20 and N=7 models-only), ≤ 0.004 (N=9 near-ties at
  the top). Median OOS logit +1.95 to +3.0. The IS−OOS gap of the IS-chosen config is +0.0001.
  The OOS-on-IS slope is ≈ −0.9 to −1.0, but that is an artifact of complementary halves
  (easy sequences on one side means hard ones on the other). Regress deltas vs a fixed
  reference, not levels.
- So the val→test problems were not selection bias among these configs. What is left: test
  composition and mask (levels), and pair-specific fragility (the ab18 flip). This is a
  statement about *this* family. A 50-config hyperparameter sweep would need the check again.

### 2.4 Robustness panel: class mix × mask softness × per-class deltas
Delta at {valid, train, uniform} class mix × λ ∈ {0, 0.02}, plus per-class deltas (λ=0):

| pair | pooled Δ range over the 6 cells | per-class Δ (classes 0,1,2,3) | LB outcome |
|---|---|---|---|
| e2 − base×0.25 | +0.0060 … +0.0072 | +.0089 +.0045 +.0036 +.0040 | final_tv_e2 ahead +0.0045 ✔ |
| blend(e2,base) − e2 | +0.0017 … +0.0029 | +.0021 +.0034 +.0039 +.0032 | blend ahead +0.0021 ✔ |
| ab18 − base×0.25 | +0.0005 … +0.0033 | +.0057 +.0027 **−.0038 −.0034** | base ahead −0.0015 (flip) |

Per-class bootstrap SEs: ab18−base ±.0044 ±.0011 ±.0016 ±.0034 (class 2 is −2.4 SE);
e2−base ±.0038 ±.0009 ±.0016 ±.0028; blend−e2 ±.0012 ±.0004 ±.0007 ±.0011.

The one LB flip is the only pair whose model loses in two classes (2 and 3: 60% of valid
sequences, 76% of train sequences). The two pairs that held dominate in every class. That is 3 of 3,
which is consistent but tiny evidence.

### 2.5 Train sequences are not usable evaluation data without a mask model
Train has no `is_scored`. Kendall τ between the 20-config ranking under the true mask and
under a proxy: λ=0.02 0.947; λ=0.1 0.789; all required rows (λ=1) 0.600; unscored rows
only 0.537. For example, e2−base is +0.0072 on scored rows and −0.0026 on all rows, because
propensity-weighted training gives up unscored rows. A K-fold that *scores* train sequences
on all rows would rank models wrongly. It needs p̂(scored|x) weights, and their fidelity is
untested (Stage 0d).

## 3. Proposed protocol

### 3.1 Units, groups, folds
1. **Unit** = sequence. **Group** = recovered chain; use the coarse component where a stricter
   guarantee is needed (fitting anything on part of valid). Keep group ids in a frozen file
   next to `wnn.splits`. Salt-hash the *group id* instead of `seq_ix` when making halves.
2. **Purging/embargo in our setting.** Label overlap across a junction is only a few dozen rows
   per 20k-row sequence, so row-level purging (C2) is immaterial for recurrent models trained on
   whole sequences. The embargo (C3) becomes "no chain neighbour of a test sequence in training".
   Group folds give this for free. When a chain must be cut (a very long chain, fold balance),
   drop the one sequence on each side of the cut (h = 1 sequence).
3. **Dev halves** (propensity, symreg, GBM residuals, blend weights): split valid by coarse
   component, stratified by class. Re-run every past devA→devB claim that fed a decision
   (propensity model of `train_propensity.py`, E-087/E-092 residual lifts, symreg terms)
   on the grouped halves.
4. **Joint K-fold over train+valid** (for final recipe choice, OOF blend/stack weights, and
   "does valid-period data help"): K=5 folds of chains, stratified by class and by file, so
   each fold holds ~375 valid sequences. Score only the held-out *valid* sequences with the true
   mask. Score held-out train sequences only after Stage 0d passes (p̂-weighted rows, τ ≥ 0.9).
   K=5 because each fit costs ~11 min. CPCV (N=6, k=2: 15 fits, 5 paths, C10) is reserved for a
   final two-way decision, and 2 seeds per arm capture training noise more cheaply.
5. **Forward track stays**: train-only fit, full-valid evaluation. It is 1 fit per config and
   evaluates on the same 1,873 sequences as the K-fold. For ranking recipes it is the default.
   K-fold is for the final fits and blend weights.

### 3.2 Aggregating the pooled metric
- Sum the six sufficient statistics per target over all out-of-fold sequences (`wp_sums`).
  One pooled WP is the analogue of the test metric. Per-fold WPs are only for dispersion.
  Averaging fold WPs is biased under global pooling (class mix per fold, level terms).
- Pooled OOF mixes K models whose levels differ slightly; that costs a little correlation a
  single final model would not pay. Compare arms with identical folds and seeds and read
  *deltas*.
- Uncertainty: deltas by sequence bootstrap (adjacency-insensitive, 2.2); absolute levels by
  chain bootstrap. Quote the component bootstrap as the "test could differ by this much" band.

### 3.3 Decision rule (replaces "2 sd + same sign on devA/devB/holdout")
Adopt B over A only if all hold:
1. Full-valid paired delta > z_K · SE, where z_K is the expected-maximum factor (C12) for the
   number of effective trials since the last LB-confirmed model (K=5 → 1.19, 10 → 1.58,
   20 → 1.90). Count trials from the ledger; merge near-duplicates by clustering per-sequence
   influence vectors (C14). With seed noise present, SE² = bootstrap² + seed variance (use ≥ 2
   seeds per arm, or the known 0.0003-0.0014 seed SD).
2. The sign holds in all 6 panel cells (valid/train/uniform mix × λ 0/0.02).
3. No class has a negative delta beyond 1 per-class SE (class-stratified sequence bootstrap).
   ab18 fails it in class 2 (−0.0038 ± 0.0016); e2 and the blend pass in all classes.
4. Blends and stacks: weights fitted on grouped halves or on OOF, never on the scoring half.
   Do not rank-average predictions: the global metric pools across sequences, and rank
   transforms throw away the class/sequence level (18-20% of weighted variance). Rank-averaging
   *configs across panel cells* is fine as a robust tie-break.

### 3.4 Guarding against selection bias and fitted-probe overfitting
- Run CSCV/PBO (S=16 chain blocks, sums-based criterion, seconds) whenever a new best is
  declared from a family of ≥ 10 configs. If PBO > 0.2, average the top configs instead of
  picking one.
- MCPT (C21) for any fitted corrector (ridge/GBM residual, symreg term, mask model). Permute the
  candidate features across coarse components within class, rerun the full devA-fit/devB-score
  pipeline 50-100 times, and set p = share with lift ≥ observed. Unbiased lift = observed − mean
  permuted lift. Ridge probes take seconds per permutation. The GBM probe (8 HistGBR fits of
  ≤ 250k rows) takes minutes per permutation, so use ~20 permutations or the ridge version.
- SelBias-style estimate (C20) on the ledger itself: choose on devA-by-component, measure on the
  other half. Gap = selection bias of our workflow.
- LB as a ladder (G3): submit only candidates that pass 3.3. Read LB deltas against a
  component-level noise model, not the sequence-level one.

## 4. Costs and staged plan (cheapest decisive first)
Unit costs: GRU 6 min/epoch per 10.6k sequences, recipe 2 epochs ≈ 12 min per fit plus
≈ 2-3 min to predict valid. GBM ≈ 15 min per fit. Sums-based analysis takes seconds.

| stage | what | cost | decides |
|---|---|---|---|
| 0a | freeze chain/component ids; component-hash dev halves in `wnn.splits` | 15 min, CPU | prerequisite |
| 0b | panel + per-class + component bootstrap in `decide.py`; re-score all cached configs | 30 min coding, seconds to run | adopt rule 3.3 |
| 0c | re-run E-092 GBM residual probe and the symreg residual terms on component halves (+ MCPT with ridge) | ≈15 min GBM + minutes for ridge MCPT, CPU | whether the small-tick "transferable" gains (+0.004 class 0) were neighbour leakage. **Most decisive cheap test** |
| 0d | p̂ mask model on component-devA; τ of config ranking under p̂-weighted all rows vs true mask on devB+H | ~20 min CPU | whether train sequences can be scored (6.7× more eval data) |
| 1 | GRU neighbour-leakage test: fixed eval set E = middle sequences of valid chains ≥ 3 long. Arm R trains on train + E's direct neighbours + random fill; arm G trains on train + an equal count from components disjoint from E; score E paired | 2 fits ≈ 30 min GPU | whether grouping matters for our main model (AFML k→T diagnostic, simplified) |
| 2 | joint 5-fold chain-grouped OOF of the finalist recipe, 1 seed (2 if affordable) | ≈ 65 min per recipe per seed | final blend/stack weights; whether valid-block data helps |
| 3 | final fit on all data, ≥ 2 seeds averaged; LB only via rule 3.3 | ≈ 15 min per seed | submission |

Order rationale: 0a-0d cost no GPU and each can overturn a past decision or unlock more
evaluation data. Stage 1 is the only GPU experiment needed to know whether the fold design
matters for the GRU. Stage 2 is spent only on finalists.

## 5. Falsifiers (what would show the protocol is no better than the current split)
- F1 (grouping, probes): Stage 0c lifts on component halves match the hash-half lifts within
  1 SE for every class. Then neighbour leakage was irrelevant for fitted probes, and grouping
  is hygiene only.
- F2 (grouping, GRU): Stage 1 arm R − arm G on E is within ±1 SE (≈ 0.002-0.004 for ~300
  sequences). Then chain grouping does not matter for the GRU, and random-sequence K-fold is
  as good.
- F3 (panel): over the next ≥ 3 LB-tested pairs, the panel rule 3.3 agrees with the plain
  full-valid rule on every pair, or disagrees and the plain rule is the one the LB confirms.
  Then the panel adds nothing. Today it has 3/3, one of them a disagreement it got right.
- F4 (joint K-fold): for ≥ 3 recipe comparisons, the pooled-OOF delta has the same sign and
  a similar SE as the forward-track delta. Then K-fold costs 5× for no decision value; keep it
  only for final fits and blend weights.
- F5 (train sequences as eval data): Stage 0d τ < 0.9. Then evaluation stays valid-only, and
  "K-fold over all 12.5k" is only a training-data scheme.
- F6 (selection machinery): CSCV on the next larger family (e.g., cached predictions of the
  E-083 sweep) again gives PBO < 0.1 and an IS−OOS gap < 0.0005. Then deflation and PBO can be
  dropped to occasional audits.
- F7 (adjacency itself): an exact class-3 label-law check at junctions (A's last-row labels
  from B's first 4 book events) fails for most class-3 links. Then the rarity-score links are
  artefacts of common book states, and sections 2.1-2.2 must be redone.

## 6. Open items
- Whether test blocks are later in time is unknown. The mixed chains (T→V 27, V→T 31)
  argue against a clean cutoff between train and valid, so "later period" simulation by
  recency weighting has no support. Class-mix robustness is the better-supported hedge.
- The organizer baseline's training data is unknown (ideator I-7). If it saw valid blocks
  or their neighbours, "base" scores on valid are in-sample and every "− base" delta is biased
  against our models.
