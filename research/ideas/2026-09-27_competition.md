# Prior competition winners: transfers to the causal book stream

2026-09-27. Scope: previous completed competitions and author or official writeups only. No current competition participant repositories, leaderboard experiments, training, or new checks were used. Current reference is the two independent EMA-4 GRUs, with pooled `|y|`-weighted Pearson on clipped predictions (`[-2, 2]`). The ranges below are **speculative incremental validation-score gains**, not forecasts of leaderboard movement; reject on held-out chains before considering a submission.

## What the winners actually used

| Competition and primary source | Winning mechanism | Transfer boundary |
| --- | --- | --- |
| [Kaggle Algorithmic Trading Challenge winner paper](https://www.ms.k.u-tokyo.ac.jp/sugi/2013/Kaggle.pdf) | Separate bid/ask models and short-to-long post-shock intervals; close horizons had more predictable response. Order/quote rate, spread, and pre-shock EMA features mattered. The piecewise model beat a single interval. | Our target is rowwise, so use causal *time since a book event* to test phase-specific residuals, not future shock windows. |
| [Optiver Realized Volatility, first place](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/nyanp-1st-place-solution-nearest-neighbors) | Recovered `time_id` order for temporal validation; nearest-neighbor aggregates across similar time IDs/stocks were the main lift, then LightGBM, MLP, and CNN were blended. | The published neighbor construction can use information from later time IDs. A strict row stream can only match previously observed states within its available causal history. |
| [Optiver Realized Volatility, fourth place](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/pksha-life-is-volatile-tentative-3rd-place-solutio) | Also found time-ID neighbors decisive; predicted a ratio to *observed* current-window volatility to stabilize changing volatility levels. | A ratio target is only sensible if its denominator is known at prediction time and its reconstruction preserves the pooled target scale. |
| [Optiver Trading at the Close, first place](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution) | CatBoost/GRU/Transformer blend (0.5/0.3/0.2), causal rolling/relative features, retraining every 12 days, and a metric-aware weighted-mean correction. The writeup reports material test improvement from online retraining. | Our two-GRU pair already beats a weaker tree partner locally; daywise online retraining and simultaneous cross-stock centering require information the row API may not provide. |
| [Jane Street Market Prediction, first place](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s) | Supervised autoencoder trained *inside each purged time-series fold*, five response heads, weights from response magnitude, and seed averaging; final blend also included XGBoost. Fold-local encoder training prevented label leakage. | The useful transfer is an auxiliary representation objective and leak-safe validation. Classification probabilities and XGBoost are a weaker match to this continuous Pearson target. |
| [Jane Street Real-Time Market Data Forecasting official data/API](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data) | A row/time-step evaluation API with `is_scored` and causally released lagged responders. | This is a relevant streaming constraint, **not evidence for a particular winning model**; I did not find a sufficiently detailed official top-team writeup to attribute one. |

## Ranked hypotheses and cheap falsifiers

All checks below use existing out-of-fold predictions and causal feature traces, with fits restricted to one set of chains and evaluation on disjoint chains. Report exact pooled weighted Pearson after clipping, separately for chain folds 0 and 1 and for the current two-member pair. A gain is worth escalating only if both fold signs are positive and the pooled gain is at least `+0.0005`; this is a screening rule, not a significance claim. Never use hidden leaderboard scores for selection.

### 1. Event-phase residual readout on top of the two-GRU pair

**Mechanism.** The 2013 book-shock winner separated near and far response horizons. Here the causal analogue is `a_t`, rows since the latest mid-price change or signed aggressive trade. A single GRU readout may average a sharp immediate response with a quiet-stretch response. Fit only a low-dimensional correction of existing predictions: `p' = p + b_phase + c_phase * (p - mean_train(p))`, with phases `a_t ∈ {0–2, 3–15, 16–63, 64+}`. Pool phase parameters across all classes and shrink them to zero; retain the global scale after clipping.

**Cheap falsifier.** Using cached OOF predictions, fit the eight intercept/slope coefficients by weighted ridge on folds 0/1 and evaluate on 2/3, then reverse the direction. Compare the exact clipped pooled score to the untouched pair; also inspect whether the correction has the same sign in the two held-out groups. Reject if either direction loses or pooled lift is below `+0.0005`. No GRU fit is needed.

**Likely gain / risk.** `0–0.0010`; phase effects may already be in GRU state, and fitted intercepts can spoil between-class covariance. **Overlap:** E-182 already supports *training weights* that down-weight old quiet stretches (`+0.0008` pooled on folds); this tests a distinct prediction-time residual. E-088/E-089 rejected a broad heuristic library, so do not rebuild that library. [Source: winner paper](https://www.ms.k.u-tokyo.ac.jp/sugi/2013/Kaggle.pdf).

### 2. Causal activity gate for the pair's blend weight

**Mechanism.** The Optiver close winner gained from blending different estimators while using recent rolling and relative features. Local E-130 found the old baseline blend helped more in calm sequences than volatile ones; E-177 says the new partner must remain strong. A *small* gate between the two strong GRUs, using only trailing spread/change rate or trade rate, might put more weight on the member whose errors are lower in the current regime. Bound member weights to `0.4–0.6` so the gate cannot become a class-scale recalibration.

**Cheap falsifier.** Bin OOF rows into four quantiles of a prespecified causal EMA-64 book-event rate (quantiles learned on fit chains). For each bin, compute the exact pair score at member-A weights `0.4, 0.5, 0.6`, then cross-fit one monotone four-bin weight vector on chains 0/1 versus 2/3. Reject if the held-out optimal weights are inconsistent in sign, either held-out half loses, or pooled gain is below `+0.0005`. Freeze the bin rule before evaluating holdout.

**Likely gain / risk.** `0–0.0008`; a two-strong-GRU pair may have little useful conditional diversity, and regime binning can overfit. **Overlap:** E-127 refuted *per-class* blend weights, E-179 found within-pair tweaks at `≤+0.0001`; this uses a causal activity axis but starts with a skeptical prior. E-118 refuted online normalization as a standalone model. [Source: Optiver close winner](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution).

### 3. Fold-local supervised bottleneck auxiliary task

**Mechanism.** The Jane Street first-place encoder reconstructed inputs while predicting several related responses, so the latent features were useful for the main target. Our analogue would attach a small bottleneck to the existing GRU state and predict *observable next-event categories* (next mid move sign, trade arrival, book change) alongside the current continuous target. Train the auxiliary head only on each training fold; discard it at inference. This could teach a less noisy state without changing the stream API.

**Cheap falsifier.** Before a full run, freeze the existing GRU and fit a CPU ridge/logistic probe from its cached hidden state to each candidate next-event label on training chains only. Evaluate disjoint-chain AUC/log loss **and** the weighted Pearson of a ridge residual correction from the auxiliary probe's out-of-fold logits, versus the unchanged pair. Reject the task if the residual correction is `≤+0.0005` or changes sign across held-out chain halves; only then is a new training run justified.

**Likely gain / risk.** `0–0.0010`; a predictable auxiliary event can be redundant with the target and increase training variance. **Overlap:** E-102 already found a next-change auxiliary head useful on the older two-epoch recipe, while distillation added little; E-174 found EMA-teacher target bootstrapping harmful (`-0.0031`). This proposal is supervised event prediction, **not** self-generated target replacement. [Source: Jane Street first place](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s).

### 4. Causal nearest-state shrinkage for rare book states

**Mechanism.** Optiver realized-volatility first place used neighboring market states as an empirical prior. For this stream, form a small key from *past/current* causal features (spread bucket, signed imbalance bucket, recent move/quote-rate bucket), then shrink the GRU prediction toward the training-fold average residual for that key. The lookup is fixed before held-out inference; there is no search over unseen test rows or later states.

**Cheap falsifier.** On OOF folds 0/1, build only `2×4×4` predeclared cells, with a minimum of 1,000 scored training rows per cell and ridge shrinkage equivalent to 5,000 rows. Apply the residual table to folds 2/3, reverse, and compare the exact clipped pair score. Reject unless both directions gain and the pooled lift exceeds `+0.0005`; inspect the gain by class to catch a class-3 loss. A coarse table fits well under 2 GB.

**Likely gain / risk.** `0–0.0008`; the GRU likely already learns these states, and class-3 noise can make tables brittle. **Overlap:** E-088/E-089 found no linear lift from 114 causal heuristics, E-092 found GBM residual overfit, and E-093 found a GBM member weaker than the GRU. Those failures make this low priority despite the Optiver precedent. The original winner's recovered-order/nearest-time features themselves are **ineligible** under our causal API. [Source: Optiver RV first place](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/nyanp-1st-place-solution-nearest-neighbors).

### 5. Target as a ratio to a causal local scale

**Mechanism.** Optiver RV's fourth-place team stabilized a shifting target by predicting a ratio to already observed volatility. For our continuous target, define `s_t = max(s_floor, EMA_64(|Δmid| + κ·spread))`, using only rows through `t`, and test `y_t / s_t` as an auxiliary label; reconstruct in original units as `s_t · r_hat` and clip to `[-2,2]`. This may help if the same state-to-return relationship changes mainly by local scale.

**Cheap falsifier.** Before training, set `κ=0.1`, `s_floor` to the fifth percentile of positive training-chain scales, and `s_ref` to their median. Compute `s_t` and the pair's OOF residual. On each fit half, fit the two-parameter weighted ridge correction `p' = p + a·(s_t/s_ref - 1)·p + b·(s_t-s_ref)`. Cross-fit on the opposite half and compare exact clipped pooled score. Reject if either held-out half loses or pooled lift is below `+0.0005`; only then consider ratio-label training. All scale constants come from training chains.

**Likely gain / risk.** `0–0.0007`; ratio targets can amplify noise when the scale is small and can damage global Pearson pooling. **Overlap:** E-118 found per-sequence online z-score inputs neutral, and E-160 found little room for post hoc metric maps. This is a label parametrization, but the cheap interaction probe should reject it quickly if local scale has no residual signal. [Source: Optiver RV fourth place](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/pksha-life-is-volatile-tentative-3rd-place-solutio).

## Explicit non-transfers

- Optiver RV's recovered time order and nearest **future** time IDs would violate the row stream. Its lesson for this project is chain-respecting validation and, at most, causal state matching.
- Optiver close's cross-stock rank/centering and daywise online retraining require concurrent stocks or delayed labels; neither follows from this API. A weighted-mean output correction also has no obvious advantage for globally pooled Pearson, which is invariant to a global affine transform before clipping.
- Jane Street's winner used magnitude-weighted classification for a trading action objective. Here the continuous `|y|`-weighted Pearson and `±2` clip remain the final arbiter; E-160 found weighted MSE already close to its optimum, and E-171 found only a small Pearson fine-tune gain.
- Extra weak models are not automatically useful. E-176/E-177 show a second independent, similarly strong EMA-4 GRU is a better partner than weaker tree or baseline variants; E-181 refuted the conv encoder in the EMA recipe.

Research-history references: `research/tree.md`; `research/experiments.jsonl` E-088, 089, 092, 093, 102, 118, 127, 130, 160, 171, 174, 176, 177, 179, 181, 182, 184. These are internal observations, not external citations.
