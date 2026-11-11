# Part A — Ideas

**Framing.** With `w=|y|`, the population optimum is fixed, so a delta can only come from (i) the top-|y| decile (~45% of weight, ρ≈0.43 vs 0.77 elsewhere) or (ii) finite-sample conditioning of the |y|-weighted estimator. A +0.10 tail-ρ move is worth roughly +0.03 pooled by weighted-mix sensitivity — implausible against a 0.0041 leaderboard spread, so treat +0.01–0.02 as the realistic tail prize, still 10–40× any recent loss delta. Rank ideas by tail leverage first, gradient conditioning second; anything that only restates the population target is dead on arrival.

**A1. Per-target loss normalization (not one pooled denominator).**
*Change.* `L = 0.5·L_t0 + 0.5·L_t1`, each with its own `Σw`, instead of pooling both targets in one denominator.
*Why.* If t0 carries 3× the weight mass of t1, t0 receives ~75% of the gradient while the metric gives it 0.5. The model systematically under-optimizes the noisier target; that is an optimization misallocation, not a variance limit.
*Test.* 3 seeds, same fold, matched baseline = 6 fits. Report per-target WP.
*Falsifier.* Per-target WP moves <0.001 on both targets.
*Risk.* Low; mostly a bug-fix. Watch for one target dominating early and destabilizing warmup.

**A2. Explicit Bayes-ratio factorization (two heads).**
*Change.* Head G predicts `|y|·y`, head H predicts `|y|`; serve `p = a + b·G/(H+ε)`, ε≈0.05, clipped.
*Why.* The metric optimum is `E[|y|y|x]/E[|y||x]`. A single head must implicitly learn the denominator, which is the high-variance part (|y| is heavy-tailed). Factorizing gives H a well-posed volatility target (vol clusters, so H is learnable) and keeps G's scale from drifting with local |y| mass.
*Test.* 3 seeds vs matched single head = 6 fits; also compare G alone.
*Falsifier.* No gain over single head, or ratio instability (p99 blowups) despite clipping.
*Risk.* Ratio of two noisy estimates; must clip and monitor H's error separately.

**A3. Causal tail expert / two-regime mixture (high-|y| idea).**
*Change.* Shared GRU trunk + base head and tail head, mixed by a gate built only from strictly causal features (trailing 20-row realized |Δmid|, spread percentile, OFI magnitude); train end-to-end with metric-matched WMSE on `clip(y,±2)`.
*Why.* Tail rows are rare, so minibatch gradients are dominated by whether any land in the batch. A dedicated head with its own normalization lowers that gradient variance and lets capacity specialize; the single-head model is fitting a bimodal conditional distribution with one affine map.
*Test.* Gate-causal MoE, 3 seeds vs baseline = 6 fits; stratify val WP by |y| decile, which you should already log.
*Falsifier.* Top-decile WP unchanged, or base WP drops more than tail gains.
*Risk.* Gate leakage is fatal — gate inputs must be lagged; gate can collapse to one branch (monitor branch usage).

**A4. Importance sampling by |y|^α with unbiased correction (fixes the 0.1×–9.7× mass swing).**
*Change.* Sample chunks with probability ∝ (chunk weight mass)^α, α∈{0.5,1}, divide loss by sampling probability. Chunk-level, not row-level, to preserve state.
*Why.* A 10–100× swing in per-batch weight mass means effective LR and gradient SNR vary wildly between otherwise identical runs — this alone can exceed seed noise. Unbiased importance sampling equalizes conditioning without changing the objective. The EMA-denominator idea already coded attacks the symptom; this attacks the cause.
*Test.* 2 α × 2 seeds = 4 fits, plus baseline; ~free since it is a sampler.
*Falsifier.* Equal wall-clock WP not better, or seed variance increases.
*Risk.* Effective sample size falls → overfitting; needs more steps at equal budget.

**A5. Output saturation matched to the ±2 metric clip.**
*Change.* `p = 2·tanh(z/2)` (or clamp with straight-through) so the served value lives in (−2,2).
*Why.* The metric clips `p`. If the model emits |p|>2 on high-vol rows, extra magnitude is discarded at eval and the ranking information carried by it is destroyed — exactly in the tail class. Soft saturation also bounds gradients there.
*Test.* 3 seeds = 3–6 fits. First, just measure the distribution of unclipped p on val; if p99<1.8 this is a no-op.
*Falsifier.* No tail-decile change.
*Risk.* tanh over-compresses genuine tail spread; compare against hard clamp.

**A6. Causal per-sequence scale normalization (exploits the non-invariance).**
*Change.* Maintain a strictly causal scale `s_t` (EWMA of |Δmid|, spread, or |feature| aggregate over rows 0..t); predict `y/s_t`, serve `p = s_t·f(x)`.
*Why.* WP is invariant to *global* affine maps but not per-sequence/per-state rescaling. Sequences differ in volatility, so a shared output scale is systematically wrong for some sequences; the metric charges for that.
*Test.* 3 seeds = 6 fits; also report per-sequence correlation on high-s vs low-s sequences.
*Falsifier.* Per-sequence correlation unchanged.
*Risk.* `s_t` must not proxy the target; any lookahead (e.g., full-sequence stats) is leakage.

**A7. Auxiliary causal self-supervised head.**
*Change.* Add a head predicting the next-k-row mid-price change (or OFI) from the same hidden state, `L += 0.2·L_aux`, k∈{1,5}.
*Why.* 10.6k sequences × 20k rows is a lot of unlabeled microstructure signal; an auxiliary dynamics objective forces the hidden state to encode order-flow state rather than memorize sequence identity, which is the representation problem behind the 0.43 tail.
*Test.* 3 seeds, matched wall-clock = 6 fits.
*Falsifier.* No gain when normalized for the ~1.2× step cost.
*Risk.* Auxiliary task may be trivially predictable and add no gradient value; can dilute the main signal.

**A8. Unconventional: state-corruption augmentation.**
*Change.* During training randomly reset/zero a fraction of hidden units at random causal cut points, and jitter inputs by ±5%.
*Why.* Serving resets state per sequence and the metric penalizes per-state rescaling; the model is currently fit on one clean state trajectory per sequence. State-corruption augmentation is a variance-reduction regularizer that matches the serving constraint, which is unusual and cheap.
*Test.* 3 seeds = 6 fits.
*Falsifier.* No gain or degradation on the clean val trajectory.
*Risk.* Too aggressive destroys long-memory signal the GRU needs.

**A9. OOF regime-specific blend weights (zero new fits).**
*Change.* Reuse existing OOF predictions; fit simplex-constrained 3-member weights per target and per |y|-regime (base/tail), cross-fitted (fit on 4 folds, score on the 5th).
*Why.* Members disagree precisely in the tail, and the tail carries ~45% of weight; equal 1/3 weights are unlikely to be optimal there. This is the cheapest possible test and it directly targets the tail metric.
*Test.* 0 fits. Cross-fitted to avoid weight overfit.
*Falsifier.* Held-out pooled gain <0.0005, or weights unstable across folds.
*Risk.* Members correlate >0.9, so unconstrained weights are near-singular.

**Order.** A1, A5, A9, A4 first (≤6 fits total, no new architecture, one is free). Then A3/A6. A2/A7/A8 only if the tail panel shows the expected grading.

# Part B — Red team

**B1. Why the loss variants failed — your explanation is half right.**
Population consistency is true and irrelevant to these results. NLL: a clip factor of 0.04 means the global gradient norm was ~25× the threshold, so ~96% of every step was discarded; learned σ² collapses on high-|y| rows and the `(y−μ)²/σ²` term explodes. The mean head never received a clean WMSE-equivalent gradient, and inverse-variance weighting reweights *away* from the metric's |y| weighting anyway. That is an optimization failure, not evidence against distributional heads. Pearson/corr-on-chunks: a per-chunk ratio estimator is biased and high-variance at small n, and the per-chunk denominator is precisely the per-state rescaling the metric punishes. Distillation-loss variants null is expected: same features, same family, same errors — no matching loss can add information the teacher does not have, and α=0.9 means the student's ceiling is a 0.68 teacher. So the correct diagnosis is *conditioning*: weight-mass imbalance (0.1×–9.7× per batch), heavy tails, global clipping, and two targets sharing one denominator. Fix those (A1/A4/A5) before concluding the objective space is exhausted. If you retry NLL: clamp log σ² to [−6,2], β=0.5 with σ detached for the mean term, per-row grad normalization, and require clip fraction <5% as an admission gate.

**B2. The B300 result is probably not "overfitting".**
Train WMSE 0.01 with w=|y| is R²≈0.98 on a noisy financial target. That is not achievable causally — it is memorization or a pipeline leak. Val WMSE 1.55 ≈ var(y), i.e. the model learned nothing transferable. Decisive cheap test: run 1 epoch of a ~100k-parameter model through the identical pipeline; if it also reaches ~0.01 train WMSE, stop and audit (most likely full-sequence normalization stats, row-level rather than sequence-level split, or target lookahead in features). Only then redesign: per-epoch checkpointing with val-WP selection (non-negotiable), 1–3 epochs at 1.5B with warmup+cosine, dropout 0.1–0.2 and weight decay, and never α=0.9 from a teacher whose val WP (0.31–0.41) is far below the GRU student's 0.68. The honest reframing: the GRU ensemble is already a better teacher than the transformer; the transformer's plausible value is as a feature extractor or auxiliary-head trunk, not as a target generator, and any teacher use should be small-α, per-row reliability-weighted, and ensemble-diverse. The destroyed checkpoints are the real loss — the peak was probably epoch 1–3, not 15.

**B3. Most likely wrong assumption: that local CV ranks recipes.**
Local 0.681–0.684 vs public 0.66832 is a 0.013–0.016 gap, 3–4× the entire rank-22→1 spread (0.0041). Either transfer is unusually bad, or the CV is optimistic. Test it: recompute per-contiguous-time-block and per-sequence-block folds; if the gap concentrates in contiguous blocks it is distribution shift (propensity/regime), if it is uniform suspect fold leakage across correlated sequences. Meanwhile the arithmetic is brutal: if the whole competitive range is 0.0041, then effects of +0.0001–0.0004 cannot be ranked by anything you have, and the 0.003 fold spread dominates every decision you are making on F0. Close second-most-wrong: "gains must come from estimation variance" — the tail evidence says representation. Third: treating "high-|y|" as one class; it is probably a mixture of genuinely predictable high-vol states and magnitude artifacts, and 0.43 may be heterogeneous across instruments and time.

**B4. Stop.**
1. Stop 30-epoch runs without per-epoch val checkpointing and best-checkpoint selection.
2. Stop α=0.9 distillation from any teacher below the student.
3. Stop A/B tests powered for <0.001. With σ_seed≈0.001 and paired σ≈0.0005–0.0007, a 0.0003 effect needs ~25–30 paired cells; either run ≥3 seeds × 3 folds for Δ≥0.002, or do not run it.
4. Stop single-fold F0 decisions — fold spread (0.003) exceeds everything you are measuring.
5. Stop per-chunk Pearson/corr losses and Muon.
6. Stop reporting pooled WP only; log per-target, per-decile, per-instrument WP on every run. That instrumentation is free and would have surfaced the tail problem far earlier.
7. Stop another loss-tuning cycle on the same representation. The oracle gap (0.968 vs 0.43) is a feature/tail-modeling signal, not a loss signal.
