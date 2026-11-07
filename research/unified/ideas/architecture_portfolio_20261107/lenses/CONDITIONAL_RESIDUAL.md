# Conditional computation and residual architectures

**Lens:** state-conditioned computation, explicit multiscale state, and metric-aware residual heads  
**Date:** 2026-11-07  
**Status:** independent literature/design proposal only; no data, fits, profiles, or remote resources used.

## Scope and evidence boundary

The corrected assignment describes 10,325 fitting sequences and 1,873 official validation sequences, each 20,000 steps with 112 raw numerical features; the optional 211-input view is those 112 raw features plus 99 derived event inputs. The 2,000-step figure is a possible TBPTT chunk length, not the full sequence length. Under current presets, all training rows after the first 99 enter the fitting mask; evaluation scoring additionally uses the official need_prediction and is_scored mask. Process warm-up and unscored rows to maintain causal state. The sequence identifiers are shuffled. There is no verified physical-time mapping, regime label, or semantic meaning for an anonymous feature. Columns named price/volume indices are identifiers, not ordered depth.

Working inference, not a source fact: the effective independent sample count is much closer to the sequence count than to the hundreds of millions of highly dependent training rows. I would therefore spend capacity on low-rank causal state and small residual modules before another wide dense sequence model. The existing TL3 recurrent checkpoint is the anchor supplied by the assignment; its current score and exact source identity should be rechecked by root before any new comparison. Contract-provided historical validation anchors are TL3 WP 0.692377425 and a fixed 90% TL3 + 10% LSTM WP 0.692502063; treat both as development-exposed validation evidence, not public/private forecasts.

### What the primary sources establish

| Primary source | Source fact | Limit on transfer |
|---|---|---|
| [Soft MoE, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/79fea214543ba263952ac3f4e5452b14-Abstract-Conference.html) | The paper replaces discrete routing with differentiable soft token assignments and reports results on visual recognition. | It does not establish a gain on this dataset. Its sequence-wide token mixing is not a causal time-series router; only the soft, low-capacity expert idea is borrowed here. |
| [Mamba](https://arxiv.org/abs/2312.00752) and [S4](https://arxiv.org/abs/2111.00396) | Input-selective state updates and structured state-space layers are linear in sequence length; S4 reports long-sequence results including length 16k. | This supports testing compact recurrent state on 20k steps, not importing Mamba/S4 wholesale or predicting a score gain. |
| [Temporal Convolutional Networks: empirical evaluation](https://arxiv.org/abs/1803.01271) | Dilated causal convolutions with residual connections beat canonical RNNs on several benchmark tasks in that study. | The tested language, music, and synthetic tasks are not this rowwise weighted-Pearson task. It is a reminder that the baseline family is not universally optimal, not a reason to switch directly to a large TCN. |
| [DLinear / Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) | A simple linear forecasting model beat several Transformer forecasters on nine standard long-horizon forecasting datasets. | The task, labels, and MSE metric differ. I use this only to motivate a linear path alongside a nonlinear one. |
| [FiLM](https://arxiv.org/abs/1709.07871) | Feature-wise affine modulation is a general conditioning mechanism, demonstrated on visual reasoning. | It does not show that learned regimes exist here. A causal hidden-state gate is a testable adaptation, not a known domain feature. |
| [Chronos](https://arxiv.org/abs/2403.07815) and [a cross-domain pretraining study](https://www.microsoft.com/en-us/research/publication/does-cross-domain-pre-training-truly-help-time-series-foundation-models/) | Chronos reports useful zero-shot forecasting results across a benchmark of 42 datasets; the later cross-domain study reports both positive transfer and negative transfer, including effects tied to sampling frequency and evolution patterns. | Anonymous fields, unknown units, unknown sampling cadence, unknown target objective, and no identified source domain make external time-series pretraining a poor default. The literature does not establish domain compatibility here. |

The architecture and cost estimates below are **my testable speculation**. The source facts are the rows above. No feature is called a price, quote, order book, or market regime.

## Ranked architecture cards

### 1. Slow/fast additive residual head on frozen TL3

**Mechanism.** The TL3 recurrent state h_t is the slow/context path. A small target-specific linear path sees current x_t and the causal first difference x_t - x_(t-1); a small nonlinear residual MLP sees h_t. The served output is TL3(x)_t plus this correction. Each target has its own final readout, while the compact hidden projection can be shared. The model uses the exact allowed input at t and previous input only. It keeps one previous 112- or 211-value row per active sequence.

**Why test it.** The linear path gives a low-variance way to use immediate and local changes; the residual MLP only needs to model what the recurrent representation misses. This is an additive slow/fast decomposition, not a claim that one path corresponds to a known domain concept. Per-target heads let each target use the shared state differently without assuming target semantics.

**Prediction.** If TL3 captures persistent context but underuses the immediate input transition, the combined end-to-end WP should improve over the exact TL3 output on both seeds or show a repeatable positive paired mean without a severe per-target loss. A better standalone residual correlation is insufficient.

**Falsifier.** Reject this mechanism if the full TL3-plus-residual endpoint fails to beat the matched TL3 endpoint on both seeds, or if any apparent gain is confined to one target and harms the pooled score. A tiny train-only residual loss with no valid benefit is ordinary fitting, not evidence for a useful fast path.

**Approximate cost.** With H=128 solely as an illustration, the direct branch has about 2(H+2F) weights plus biases, about 700 weights at F=112; a shared H-to-32-to-2 residual MLP adds about 4.2k. Total added parameters are roughly 5k. The online state is the existing TL3 state plus F previous-input floats: about 448 bytes per active sequence for F=112 or 844 bytes for F=211 in float32. Extra work is on the order of 5k multiply-accumulates per row, approximately 100M per 20k-step sequence. Replace H=128 with the actual TL3 width before implementing.

**Serving path.** Run the frozen TL3 causally with an exact reset per sequence. Carry the previous input row, form the additive correction, and pass the complete sum through the unchanged official scorer. Preprocessors and the previous-row state reset at each sequence boundary.

### 2. Low-rank multiscale EMA state plus nonlinear TL3 readout

**Mechanism.** Project each current input into r=32 latent channels using a train-only fitted projection. Maintain four causal exponential states per sequence:
m_(k,t) = a_k m_(k,t-1) + (1-a_k) e_t,
where e_t is the current projected input. Initialize all states to zero at each sequence reset. Use four trainable, globally shared, ordered decay constants initialized to row half-lives 32, 256, 2,048, and 8,192; trainable values are learned on fitting data only. Concatenate the four EMA vectors with the current TL3 hidden state h_t, and predict a small target-specific residual. The final output is frozen TL3 plus that residual. Half-lives are measured in rows, not physical time.

**Why test it.** A small bank of first-order filters is an explicit, cheap state-space basis. It tests whether the GRU's implicit state benefits from a few exposed long and short summaries. S4/Mamba make long-range causal state a defensible model family, but this EMA bank is deliberately much smaller and has no claim to reproduce either paper.

**Prediction.** A gain over card 1 would suggest the exposed long/short summaries carry information that TL3 hidden state and a one-row difference do not. If the decay constants converge together, or the EMA residual is redundant with TL3, its mechanism is not supported.

**Falsifier.** Reject if the full TL3-plus-EMA residual does not improve the matched paired score, if the four learned decays collapse to one effective scale, or if its improvement is not stable across the two seeds. Do not rescue it by searching dozens of decay grids on the exposed official validation.

**Approximate cost.** With F=112/211 and r=32, input projection adds 3.6k/6.8k parameters. The per-sequence EMA state is 4r=128 float32 values (512 bytes), in addition to TL3 state. With H=128, a 256-to-16-to-2 readout adds about 4.1k parameters and about 4.1k MAC per row; input projection adds about 3.6k/6.8k MAC per row. The illustrative total is about 7.7k-10.9k MAC per row or 154M-218M per 20k-step sequence. Actual CPU latency is unknown and must be profiled before any deployment claim.

**Serving path.** Apply the exact fitting-only input normalizer/projection, update four EMA states from current causal input, update TL3 state, form the correction, and emit the full sum. Reset TL3, EMA, and previous-sequence state between sequences.

### 3. Four-expert soft residual head, routed from causal state

**Mechanism.** Add four small dense residual experts to the frozen TL3 hidden state. For H=128, each is H-to-16-to-2; a small linear router maps h_t to four softmax weights at each step; the weighted expert sum is added to the TL3 prediction. The router sees only the TL3 state formed from allowable inputs through t. No sequence-wide pooled summary or state from validation rows is available to the router. All four experts run at every step; this is intentionally a cheap, dense soft mixture, not an efficiency claim about sparse MoE.

**Why test it.** A state-conditioned soft gate can test whether a few local response functions help in different causal contexts. Soft routing avoids discrete top-k assignment and its load-balancing machinery. The analogy to Soft MoE and FiLM is structural only; there is no evidence of discrete regimes or expert specialization in these anonymous data.

**Prediction.** If the conditional-response mechanism matters, the combined output should beat the same TL3 baseline, expert usage should remain non-collapsed on fitting data, and paired gains should recur across seeds. It should be considered only after the cheaper additive heads because it adds a higher-capacity, more selection-sensitive module.

**Falsifier.** Reject if the router collapses to one expert, experts become near-identical on fitting data, or the complete combined endpoint does not improve both paired runs. Router entropy/usage is a fitting-only mechanism diagnostic; it cannot be used to select a model by official-valid subgroups.

**Approximate cost.** Four H-to-16-to-2 experts plus H-to-4 router are about 9k parameters at H=128. Dense execution costs around 8.8k MAC per row, or 176M per 20k-step sequence. The router adds no persistent state beyond the frozen TL3 recurrent state. The parameter count remains small but expert routing can still overfit because there are only about 10k sequence units.

## Bounded initial comparison

The first useful screen should compare cards 1 and 2, the two cheapest structurally distinct mechanisms, each at two predeclared matching seeds against an exact TL3 control. If exact TL3 checkpoints with those seed identities and fitting recipe exist, reuse them; otherwise two matched TL3 controls plus four candidate modules make six fits total. Do not change seeds after seeing validation.

For every seed, grade the **complete output**: frozen TL3 plus the candidate residual, through the unchanged official scoring kernel, all eligible official-valid rows, both targets, and the pooled endpoint. Also report candidate residual alone and paired deltas as diagnostics. Do not choose by residual decorrelation, one target, a 2,000-step subset, or validation-fitted blend weights. These are not substitutes for the complete endpoint. Card 3 remains a later alternative rather than a third simultaneous arm; this avoids a larger matrix and keeps the initial screen at four to six fits.

Run the frozen architectures on a modest L4 lane only if root later admits the recipe and verifies exact controls; these modules are too small to justify consuming B300 capacity by default. No throughput estimate, completion time, or performance expectation is available without a real profile. Save train and full-valid results separately and preserve per-seed outcomes.

## Ancestry, selection, and teacher/student path

- Fit normalizers, input projections, EMA constants, output heads, and any calibrator only from the official training partition. Preserve whole sequences and the official train-to-full-valid boundary. Do not use shuffled seq_ix, random folds, or a valid-fitted gate/blend. The global TARGET_QUANTA constants discovered on validation are ineligible as fit initializers or gate oracles.
- A frozen TL3 feature trunk is a train-only teacher-like representation source, not an excuse to train on valid. If candidate heads learn on base-model in-sample hidden states/predictions, note that train/valid feature shift can result; keep the correction low-capacity and judge its complete official-valid endpoint.
- No new dense Transformer teacher or external pretraining is proposed for the first screen. The source literature reports both useful zero-shot forecasting and negative transfer across domains; the current feature units, cadence, and objective are unknown. If a later teacher/student proposal is admitted, fit the teacher only on official train, keep all pseudo-target/state ancestry train-only and causal, distill into the actual CPU-serving student, and score that complete student (or fixed complete TL3-plus-student output) on full valid. A larger teacher score or low prediction correlation alone is not enough.
- The official valid has already been inspected during development. Two seeds reduce seed noise; they do not create a fresh holdout or justify a public/private score forecast. Record both paired deltas and uncertainty, report the exposed-selection context, and use later public/private feedback only as transfer evidence under the project's submission process.




