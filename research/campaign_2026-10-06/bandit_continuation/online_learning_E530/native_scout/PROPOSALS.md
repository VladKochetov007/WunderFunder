# Online-learning family: independent scout proposals

Scout status: source-grounded hypotheses only. No fits, full-data scans, hypothesis-ledger edits, provider calls, or launch authority. Numerical priors are null. Root owns admission, implementation, grading, and any results.

## Task contract and live information boundary

Objective mode is competition: improve expected unseen Challenge 2 score while keeping a valid deterministic submission. The current-state document reports own public best 0.6660; that number is not a forecast ceiling.

At inference, each callback supplies one row with seq_ix, step_in_seq, need_prediction, and 112 float32 features. It supplies neither t0/t1 nor is_scored. Rows arrive in sequence order; all rows must be processed, including warmup rows where need_prediction is false. Return None only when need_prediction is false and otherwise predict both targets. The feature schema is two instruments' L11 price-like and volume-like features, trade price/volume features, and a0–a7; feature indices are identifiers, not book-depth order. Use only features through the current row. Reset every mutable statistic, posterior, buffer, and fast parameter state when seq_ix changes. Keep static model weights fixed across test sequences. Do not presume a label delay, future row, scoring mask, or cross-sequence test memory.

Grade the complete candidate endpoint with the official global_wp_clip2_v1 metric and its fixed need_prediction AND is_scored mask, clipping, and target-magnitude weights. Keep fitting and checkpoint choice within declared train membership. The validation folds have already been used and remain exposed; they can screen a recipe but cannot confirm fresh generalization. Neither a new agent nor repartitioning makes them fresh. Public transfer remains a separate uncertainty.

## Existing overlap to preserve

The active H230 source has a causal cumulative online_zscore_torch transform and a prefix-conditioned adapter whose context uses up to the first 99 rows. The project has also tried online-normalization members historically. These are required controls for any new normalization or state-adaptation proposal. E-078's old online-learning ceiling used true evaluated labels after an assumed lag; that label feed is absent from the submission interface, so it does not test the label-free inference family. It does not close this broader family.

The proposals below separate (1) label-free inference-time changes to sequence-local state, (2) a feature-driven expert posterior with fixed expert weights, and (3) supervised chronological optimization during training only. The third proposal explicitly freezes model weights at deployment.

## Directed source map

- Sun et al., “Test-Time Training with Self-Supervision for Generalization under Distribution Shifts,” ICML 2020, show the general pattern of using an unlabeled test input to form a self-supervised update before prediction and extend it to an online stream. Their image-rotation objective and results do not establish a useful market objective here. [PMLR paper](https://proceedings.mlr.press/v119/sun20b.html)
- Arik, Yoder, and Pfister, “Self-Adaptive Forecasting for Improved Deep Learning on Non-Stationary Time-Series,” propose backcasting as an input-derived adaptation signal for forecasting. The causal transfer here would use only a fully observed window ending at the current row, never future sequence rows. [arXiv paper](https://arxiv.org/abs/2202.02403)
- Chen et al., “Test-Time Training for Spatial-Temporal Forecasting,” use self-supervised reconstruction cycles without target labels. Their many-location structure creates an effective batch; Challenge 2 has 112 correlated input features, so that assumption is not established. [SIAM SDM 2024 paper](https://epubs.siam.org/doi/10.1137/1.9781611978032.54)
- Wang et al., “Continual Test-Time Domain Adaptation,” use teacher averaging and stochastic weight restoration to reduce error accumulation and forgetting. These are potential safeguards, not evidence that prediction-self-training is safe for this regression task. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)
- Wu et al., “Test-Time Adaptation for Non-stationary Time Series: From Synthetic Regime Shifts to Financial Markets,” is a 2026 preprint. It studies causal forecasting with normalization-affine adaptation and reports that simple statistic updates can be robust while aggressive adaptation may hurt. Treat this as a recent prior, not an independent Challenge 2 result. [arXiv preprint](https://arxiv.org/abs/2602.00073)
- Kim et al., “Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation,” AAAI 2025, uses partially observed ground truth for adaptation. That component is not admissible at this interface; the paper is useful as a boundary case demonstrating why forecast-label availability must be specified. [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/33965)
- Wang et al., “Adaptive Learning of Smoothing Functions: Application to Electricity Load Forecasting,” NeurIPS 2012, combines online supervised smoothing with recursive least squares. Its update uses labels as they become available, so only its training-time sequential optimization pattern transfers here. [NeurIPS paper](https://proceedings.neurips.cc/paper/2012/hash/b571ecea16a9824023ee1af16897a582-Abstract.html)

## Random library passage and transfer attempt

I froze the 14 distinct absolute PDF paths in the active campaign's RANDOM_SELECTION_POOL.txt and sorted them. Using Python random.Random with seed 20261008, I drew one path uniformly, then drew one 1-based PDF page with the same RNG after reading that file's page count. The draw selected /home/vlad/Documents/Trading/Random Matrix Theory.pdf, PDF page 6 of 112, printed page 5 in the preface.

Exact passage: “Any time you need a matrix which is too complicated to study, you can try replacing it with a random matrix”.

Transfer attempt: treat the sequence's rolling covariance of feature innovations as a complicated 112 by 112 matrix, then compare its leading eigenvalue with a train-calibrated block-shuffle null. A dominant eigen-spike could indicate coordinated feature movement worth triggering a small adaptation step; bulk movement could be ignored. This is an unconventional event trigger, not source evidence for predictive value. The book passage motivates a random-matrix comparison only. Real market features are autocorrelated and nonstationary, so a textbook independent-entry null is inappropriate; use block-preserving/shuffled training segments and test the trigger against sham thresholds.

## Proposal OL-1 — Robust two-timescale feature-state normalization

**Mechanism family:** label-free, sequence-local statistic adaptation.

**Intervention:** Keep the frozen predictor weights fixed at inference. For each feature channel, maintain fast and slow exponentially weighted location/scale estimates from the observed rows. Normalize the current row with the state from rows before t, keep the raw view as a parallel input, then update statistics with x_t after forming the prediction. Bound extreme updates with train-fitted innovation caps. Train the model with exactly the same transform and reset state at each sequence boundary. The candidate is a dynamic robust-timescale treatment, not a rebranding of current cumulative z-scores or the first-99-row prefix adapter.

**Information and state:** Prediction at t uses x_0…x_t; normalization uses estimates from x_0…x_(t-1); after prediction, update using x_t even when need_prediction is false. All stats reset on a new seq_ix. No target, mask, future row, or other sequence enters the update.

**Source component and analogy:** Online normalization estimates can adapt without a batch, and recent time-series TTA work motivates comparing modest statistic updates against aggressive weight changes. The structural transfer is only causal state tracking; those papers do not establish that scale drift is the relevant shift in this competition.

**Cheapest decisive comparison:** Replay one existing own-weight checkpoint on a fixed exposed outer group with frozen normalization, the current cumulative online-zscore path, a one-timescale EW control, and the fast/slow capped candidate. Then fit only the small input/readout delta if the replay shows a plausible target-aligned change. Include a row-order-shuffled-statistics sham. Grade the full two-target endpoint under the official metric; do not average per-sequence scores.

**Predicted observable:** The fast/slow candidate should improve the anchor's complete endpoint on sequences whose causal feature location/scale moves, while leaving stable sequences near the anchor. A feature-shift diagnostic alone is insufficient.

**Falsifier:** No complete-endpoint improvement over the current cumulative transform, no relation between the proposed state and cross-fitted anchor residual contribution, or a similar gain under shuffled-statistic controls.

**Teacher/compression path:** If a larger own causal teacher benefits, distill its input-state-conditioned predictions into an independently trained compact GRU using train-only labels and identical causal features; compare the student with and without the dynamic transform.

**Cost and bounded fit path:** Serving state is under 1 KB plus O(112) scalar work per row; measure actual Rust/Python latency. Proposed first fit is one compact own-weight candidate/control seed pair, one declared group split, at most one epoch or one tenth of the current recipe's updates, whichever is smaller, after a fresh resource/profile and root admission. Stop if the added path threatens the 60-minute, one-CPU full-test limit.

## Proposal OL-2 — Sparse self-supervised fast adapter

**Mechanism family:** label-free test-time training with a supervised outer objective.

**Intervention:** Add a rank-4 to rank-8 adapter to the input encoder or final hidden representation, leaving the backbone frozen. Train the adapter to take one regularized inner update that lowers a masked-channel or short-window reconstruction loss on observed inputs and improves the supervised target objective on separate training episodes. At inference, update the adapter sparsely (initial candidate every 128 or 256 rows) from a rolling window ending at the current row. Use a quadratic anchor penalty to the initial adapter and freeze updates when the feature-reconstruction loss is already low. Do not use the model's own target prediction as a target.

**Information and state:** Before predicting t, an update may use only the observed feature window x_(t-W+1)…x_t; any positions reconstructed are within that window. The update objective contains no t0/t1, score mask, later feature row, or cross-sequence example. The supervised meta-objective uses only permitted training groups; validation labels are grade-only. The adapter and window reset on seq_ix changes.

**Source component and analogy:** Test-Time Training and time-series backcasting show that an unlabeled input can generate a task-adjacent adaptation loss. CoTTA motivates a fixed anchor to prevent self-reinforcing drift. The exact feature mask/reconstruction task, adapter size, and task-gradient alignment need to be learned and tested here.

**Cheapest decisive comparison:** Freeze one own backbone and train one small adapter. On one exposed outer group, compare no update, one regularized update at the declared cadence, a random-gradient sham, and unanchored adaptation. Report official endpoint plus per-group deltas, update magnitude, feature reconstruction loss, and latency. If the one-step transfer screen fails, defer episodic meta-training.

**Predicted observable:** Adaptation improves the complete endpoint after feature-distribution shifts while updates remain small and do not degrade stable sequences; update directions learned on training episodes transfer to held-out sequence groups.

**Falsifier:** The update helps its reconstruction objective but not the supervised endpoint; the random-gradient/sham performs similarly; output drift grows with sequence length; or adaptation costs more than the serving budget permits.

**Teacher/compression path:** Train a larger own causal teacher with the same input-only adaptation protocol. Distill its post-update predictions/hidden state into an adaptation-aware GRU student. A student may keep the learned sequence state while eliminating test-time backpropagation; grade its complete endpoint and CPU latency.

**Cost and bounded fit path:** Inner updates are much dearer than statistic tracking. Begin with a frozen existing own checkpoint and one trainable low-rank adapter, one seed, one outer sequence-group split, one epoch or at most one tenth of the parent recipe's updates. Use one admitted GPU worker only after measured actual-shape memory/timing. A Rust implementation or compiled fixed-step update is a delivery requirement if the Python prototype is too slow; do not infer CPU viability from GPU results.

## Proposal OL-3 — Feature-only Bayesian regime filtering over fixed experts

**Mechanism family:** online state inference and conditional mixture; expert weights stay fixed.

**Intervention:** Train a small bank of own-weight experts and a diagonal feature-emission model on training groups. At inference, maintain a K-state posterior over regimes. Update it from current/lagged feature innovations and a learned transition matrix, then mix fixed expert predictions by the posterior. Use only a small, train-defined event/scale vector rather than all 112 raw channels. This changes per-sequence expert weights online without adapting any model parameters or labels.

**Information and state:** At t, the posterior uses x_0…x_t and the previous posterior. Experts predict from their causal recurrent state through t. The filter never sees targets, is_scored, need_prediction as a training signal, or future rows. Initialize from the training prior and reset posterior plus observation history at seq_ix changes; weights and fitted emission parameters are static.

**Source component and analogy:** Continual adaptation research explicitly treats the test stream as changing regimes. The transfer is Bayesian filtering over observable input regimes, avoiding fragile gradient updates. Unlike label-driven online forecasting methods, the filter adapts only its expert mixture from features.

**Cheapest decisive comparison:** First use existing own checkpoint outputs where ancestry permits: fit emissions/transitions on training inputs only, with no validation targets, then compare equal fixed mixture, posterior mixture, constant-posterior, and sequence-order-shuffled posterior on the same exposed outer group. If no suitable diverse experts exist, run one compact expert-bank training cohort only after the filter's cross-fitted state predicts distinct residual contribution.

**Predicted observable:** The posterior changes gradually or at detected input shifts, and the posterior-weighted complete endpoint beats the fixed mixture and shuffled-state controls on held-out groups; gains should not depend on test labels.

**Falsifier:** Feature likelihoods do not predict expert usefulness, the posterior collapses to one expert without gain, or equal/shuffled controls match the full endpoint.

**Teacher/compression path:** A larger own expert bank may serve as a teacher; distill the posterior mixture into a compact recurrent student with a small learned state update. Keep mixture and student endpoints as separate nodes.

**Cost and bounded fit path:** A diagonal K=4 to 8 filter over 8 to 32 features requires O(KF) arithmetic and a few KB of state per sequence. Use existing own experts for the initial CPU screen; if fitting is needed, cap at one compact 4-expert seed, one group split, one epoch or one tenth of recipe updates, then stop for root review.

## Proposal OL-4 — Supervised prequential training, frozen at deployment

**Mechanism family:** training-time online optimization, not test-time label adaptation.

**Intervention:** During training only, predict each chronological row/chunk using the current model, then update a small readout (or a separate shallow model) after its supplied training label is available, with an explicit optional label delay and forgetting factor. Record the final trained weights and deploy them frozen. Reset hidden/context state between sequences. The hypothesis is that chronological online optimization finds a better static predictor for the competition's temporal structure; it does not claim the model can use unseen labels.

**Information and state:** Training update at row t uses only train features through t and labels whose predeclared availability time has passed. At inference, the model uses only features through t and no weight update driven by target labels. There is no cross-sequence mutable test state. The training order and optimizer state are part of the frozen recipe.

**Source component and analogy:** The NeurIPS online smoothing paper uses recursive least squares with a forgetting factor for labeled online load forecasts. The transferable element is a compact update rule under nonstationarity; its live-label premise is explicitly restricted to the training algorithm here.

**Cheapest decisive comparison:** One paired short run from the same own initialization and data membership: prequential chronological readout updates versus the current standard training recipe with equal optimizer updates and compute, plus a chronological-order-shuffled control. Evaluate fixed held-out sequence groups with the official scorer. Do not evaluate per-sequence adapted training weights as if they were deployable.

**Predicted observable:** The prequential run yields a better final frozen checkpoint or better target-weighted validation endpoint at matched training budget, without gains confined to late training segments.

**Falsifier:** It fails to beat both matched controls, its apparent gain requires final weights to continue adapting from evaluation labels, or it worsens independent seed pairs.

**Teacher/compression path:** A strong prequentially trained own model can be used as a fixed teacher for an ordinary independently trained student. Do not copy training labels or per-sequence readout states into test inference.

**Cost and bounded fit path:** No added inference cost. Compare one candidate and one standard control with one paired seed, one fold, equal updates, one epoch or one tenth of the current recipe's updates, whichever is smaller. Use the current GPU admission/profile gates; a single short result remains a screen.

## Proposal OL-5 — Random-matrix innovation trigger for sparse updates

**Mechanism family:** input-only change detection; interaction with OL-2 or OL-3, not a standalone forecast model.

**Intervention:** Maintain a rank-3 or rank-4 online approximation of the rolling covariance of standardized feature increments. Every 64 or 128 rows, compare its leading eigenvalue to a threshold calibrated from block-shuffled train segments. Let a spike trigger a small OL-2 adapter update or temporarily increase OL-3 regime responsiveness; otherwise freeze. This is deliberately a low-cost, unconventional controller, not permission to adapt on every noisy row.

**Information and state:** Update uses only feature increments observed through the current row and resets completely at seq_ix. Fit thresholds from training inputs only. The trigger cannot use targets, score mask, future rows, or information from earlier test sequences.

**Source component and analogy:** The seeded random-library passage motivates comparing a complicated empirical covariance matrix with a random reference. A top eigen-spike may expose coherent cross-feature movement. The source only gives the matrix/random-baseline idea; it does not predict a score gain.

**Cheapest decisive comparison:** Before fitting any adapter, calculate the trigger on a small predeclared set of already-exposed sequence groups and test whether trigger-on windows predict cross-fitted anchor residual magnitude or candidate-minus-anchor value better than an equal-rate randomly timed trigger and block-shuffled null. Include no-trigger control. If this does not separate the groups, stop the trigger component while leaving other online-learning mechanisms eligible.

**Predicted observable:** The trigger is sparse, detects feature co-movement beyond the block-preserving null, and concentrates contexts where the declared online update helps the official endpoint.

**Falsifier:** Similar trigger rate under the null, no enrichment for out-of-fold residual contribution, or trigger-gated performance no better than randomly timed updates.

**Teacher/compression path:** The trigger can gate a high-capacity teacher's update schedule; distill the resulting state-conditioned predictions into a compact student only if trigger ablation passes.

**Cost and bounded fit path:** Rank-4 state is about 2 KB; feature arithmetic is O(112 × 4) at each update boundary. Profile the compiled per-row implementation against a sham and the one-CPU/60-minute full-test envelope. The first screen is CPU-only and limited to a small frozen group sample; no GPU fit until the signal survives.

## Recommended admission order

1. Keep OL-1 as the cheapest dynamic-state control against current cumulative online-zscore and prefix context.
2. Use OL-3 for a low-gradient online state alternative if own diverse experts already exist.
3. Only then profile OL-2's sparse self-supervised updates, with strict source anchoring and an adaptation-aware student path.
4. Keep OL-4 separate as a training recipe; it makes no inference-time label claim.
5. Add OL-5 only as a falsifiable trigger interaction if the rolling feature covariance screen is positive.

All cards have null numerical priors, no measured expected effects, and no launch authority. Existing development folds are exposed. Report per-seed training variation, whole sequence/chain resampling uncertainty, and transfer/public uncertainty separately. Do not convert a proxy result or a confidence interval into a public-score forecast.

