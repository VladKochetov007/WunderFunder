# Independent proposal: loss, architecture and sequence diversity

**Role:** independent Luna/max investigation, 2026-10-09. This is a proposal, not an admission or a result. I did not read other agents’ proposals. No fit, provider action, data materialization, or spending occurred.

## What the code already establishes

The official evaluator is pooled `global_wp_clip2_v1`: scored predictions and labels are clipped at ±2, and each row is weighted by the magnitude of its clipped label. The trainer’s target clipping and loss are separate treatments. `src/wnn/nn.py` already implements weighted per-sequence correlation, pooled weighted correlation, weighted MSE and weighted Huber; `train_rnn.py` exposes `corr`, `wmse`, `mix`, `huber`, `mse`, `gcorr`, `wmse_gcorr`, and `gcorr_pool`. The default `mix` is weighted correlation plus 0.5 weighted MSE. The official metric must remain unchanged.

`scripts/train_rnn.py` exposes AdamW, Muon and SGD, OneCycle/WSD/SGDR, `--arch tcn`, and `--readout-activation fixed_mix`. The TCN is an existing causal residual stack: kernel-two blocks at dilations 1, 2, 4… carry exact convolution tails across TBPTT chunks; its receptive field is `2**layers` rows. The trainer has several other architecture flags, but do not combine them into this initial screen. The current `src/wnn/infer.py` submission-side engine is a row-at-a-time NumPy GRU reader; it does not implement the TCN. A TCN is therefore a research-teacher candidate until an independently validated serving/export path exists.

Sequence-order details matter. The trainer’s `--class-balance` and `--epoch-samples` use replacement sampling. `EpochOrderSampler` can take optional strata and visit each index once, but the trainer currently constructs it without strata. `training_sampling.input_strata` already defines an input-only market-class × activity-tercile label and `stratified_order` spreads those strata across batches. Wiring that option into the trainer is a small, separately reviewed implementation step; it should not silently substitute replacement sampling. Each training example remains a complete sequence with its causal prefix/history intact.

**Interpretation constraint:** previously inspected fold 0 and fold 1 are exposed. They remain useful fixed comparisons, not fresh holdouts. Candidate/control pairs must use identical, component-separated fit membership, full fixed evaluation membership, source-qualified sequence IDs, teacher/initializer ancestry, seeds, optimizer steps, precision and preprocessing unless the named factor is the treatment. Do not reuse a prediction cache unless its checkpoint, split, membership and target ancestry prove that it is an out-of-fold prediction for the intended held-out units.

## Ranked experiment cards

### 1. Huber robustness against the current weighted objective

**Mechanism.** Current `mix` has per-batch/per-sequence weighted correlation and MSE terms. Contrast it with
\[
 L_{H,\delta}=\frac{\sum_{i,t}m_{it}|\tilde y_{it}|\,\rho_\delta(\hat y_{it}-\tilde y_{it})}{\sum_{i,t}m_{it}|\tilde y_{it}|},\quad
 \rho_\delta(e)=\begin{cases}\tfrac12e^2,&|e|\le\delta\\\delta(|e|-\tfrac12\delta),&|e|>\delta.\end{cases}
\]
The existing `--loss huber --huber-delta 0.5` implements this with `tilde y=clip(y,-2,2)` and the same magnitude weights. Huber’s bounded tail influence motivates a robustness test; it does not imply higher weighted-correlation score.

**Pair and ancestry.** Same GRU checkpoint initialization (or same declared fresh seed), AdamW/schedule, complete training groups, batch order, updates and all preprocessing; only loss changes. Compare against `--loss mix`. Use identical exposed evaluation units and no eval-driven early selection. Source fact: [Huber’s robust location estimator](https://doi.org/10.1214/aoms/1177703732). Hypothesis: the robust gradient changes the learned ordering/calibration of rare large residuals enough to improve pooled rank association or blend value.

**Measure / cheap test.** Full official score plus per-sequence score distribution, target-specific slope/intercept on fit-only predictions, tail-bin residuals, gradient clip fraction and measured training time. If validated cross-fit predictions already exist, use them only for no-fit blend triage; they cannot answer this loss contrast. Cheapest training screen is one paired, component-complete subset at fixed updates, followed by full fit only if it changes score or conditional contribution enough to justify compute.

**Falsifier and descendants.** No improvement in complete-endpoint score or cross-fit complementarity across the fixed units, coupled with no meaningful tail/calibration change, lowers priority for this exact `delta`/recipe. If promising, compare a predeclared small delta ladder or `wmse_gcorr` interaction one factor at a time. Cost is near the existing objective; score decision must include endpoint calibration and CPU serving.

### 2. Train against pooled rather than per-sequence correlation

**Mechanism.** The official metric pools rows across sequences before centering. `weighted_corr_loss` centers each sequence separately, while `weighted_pooled_corr_loss` centers all valid rows in the minibatch together. Compare `gcorr` and/or `wmse_gcorr` against `mix`; retain identical clipped targets, `|clip(y)|` weights and official evaluation. `gcorr_pool` keeps a detached exponentially decayed statistic across chunks, but it is an approximation whose memory and batch order are part of the recipe.

**Hypothesis and contrast.** The per-sequence objective may suppress useful between-sequence level/scale relationships that the pooled score rewards. First compare only `mix` versus `wmse_gcorr` with `--gcorr-weight` fixed before looking at the held-out results; consider `gcorr_pool` as a separate follow-up because its decay introduces an additional timescale. Same architecture, initialization, optimizer, sampled sequences, updates and evaluation as card 1.

**Observable / falsifier.** Score targetwise global correlation, per-component contributions, sequence means/variances, output scale and sensitivity to batch composition. No pooled score or conditional blend gain, or large instability under batch/order perturbation, falsifies the rationale for this exact objective. Reusable artifact: per-sequence and pooled weighted sufficient statistics, not only a single score. Compute is similar for `gcorr`; memory-bearing `gcorr_pool` needs a separately bounded profile.

### 3. Training-target clipping as its own factor

**Mechanism.** The existing objectives clamp labels to ±2 before both loss and weight calculation. Test the treatment `L = sum m |clip(y,2)| rho(pred - y_raw) / sum m |clip(y,2)|` against the standard `L = sum m |clip(y,2)| rho(pred - clip(y,2)) / ...`. This preserves the official metric’s target weights while allowing the training residual to retain magnitude beyond 2. Keep the official evaluator’s label and prediction clipping unchanged.

**Implementation gap / test.** There is no general target-clip flag in the current trainer; adding an explicit `--train-target-clip` option (default 2, `none` for raw) and binding it into recipe/checkpoint hashes is needed before this arm is runnable. Pair one loss/architecture only; do not simultaneously change Huber delta or sample composition. Evaluate tail-bin fit-only error and official score. The hypothesis is that unclipped training residuals retain useful ordering information; the counter-hypothesis is wasted gradient on distinctions erased by the metric. Stop this exact treatment if it worsens optimization/calibration without a conditional contribution.

### 4. Causal multiscale TCN as a distinct teacher family

**Mechanism.** Compare the existing GRU with a causal dilated residual TCN using same raw/event feature view, output head, loss, optimizer and sequence membership. Set channels/layers before the run and report both parameter count and receptive field (`2**L`, so `L=15` covers at least 20,000 rows); the initial 99-row prefix is observed causally and must not be reset at chunk boundaries. Do not compare one short-context TCN against a full-history GRU and call it an architecture-only result.

**Source fact / hypothesis.** The source implementation carries convolution tails across training chunks, and `src/wnn/tcn_ref.py` is the rowwise reference. TCN’s published sequence-modeling evaluation makes it a defensible alternative family, not evidence it wins here ([Bai, Kolter & Koltun, 2018](https://arxiv.org/abs/1803.01271)). Hypothesis: parallel local filters plus multiscale receptive field learn event/volatility patterns with different errors than GRU memory.

**Cost and falsifier.** Measure GPU step throughput, peak memory, all-row prediction latency, complete-endpoint runtime, and TCN-vs-GRU residual complementarity. `NumpyRNN` in `src/wnn/infer.py` only implements GRU, so serving cost is presently an unresolved delivery dependency; do not promote on GPU validation alone. If TCN does not beat or complement the GRU after matched budget, or its full endpoint cannot fit the CPU envelope, retain it only as a distillation teacher. A small TCN can teach a GRU student through cross-fitted predictions, but its outputs are not labels for components it trained on.

### 5. Muon/AdamW optimizer diversity

**Mechanism.** Pair `--optim adamw` and `--optim muon` on the same GRU recipe, initialization, complete-sequence order, batch/updates, `mix` objective and OneCycle schedule. In this code, Muon orthogonalizes recurrent gate matrices while other groups use AdamW; it is not an all-parameter optimizer swap. Fix the scope and learning-rate policy in advance, and report optimizer updates and wall-clock time, not epochs alone.

**Hypothesis / measure.** Different update geometry could yield another error mode and improve the later blend even when standalone scores tie. Compare official score, gradient/clip statistics, calibration, compute per update and cross-fit residual covariance. If optimizer performance differs only because one received more effective optimization or a different schedule, the comparison is inconclusive. Only branch into WSD/SGDR after an optimizer effect is visible; do not factorially sweep optimizer × schedule × loss on the first pass.

### 6. Activation diversity only at the readout

**Mechanism.** On a supported plain GRU recipe, compare existing `--readout-activation gelu` with `--readout-activation fixed_mix`, which computes `0.5*GELU(a)+0.5*LeakyReLU(a;0.01)`. It is a parameter-free change at the readout; it leaves GRU gates, input encoder and recurrence untouched. Do not describe it as replacing the recurrent gate.

**Hypothesis / measurement.** A different readout curvature may preserve tail/sign-sensitive combinations while sharing the same temporal representation. Keep all else fixed and measure official score, target slopes, prediction correlation and cross-fit blend value. This is a cheap fit comparison, but only reuse an existing checkpoint/prediction if the activation and ancestry exactly match. No score or conditional contribution means no reason to create a broader activation sweep. Inference needs the corresponding independent readout implementation and parity test; measure it in the complete CPU endpoint.

### 7. Complete-sequence stratified order without replacement

**Mechanism.** On a fixed component-complete fitting pool, compare an ordinary deterministic full permutation to a stratified permutation that spreads market class × input-activity terciles across batches, while visiting every fitting sequence once per epoch. Use only training-input activity for strata; do not use targets, validation inputs or eval-derived quantile cut points. The existing optional `EpochOrderSampler(strata=...)` and `input_strata` support the sampler primitive, but the trainer does not currently wire them. This arm needs a small explicit flag and provenance for the strata source/hash/seed. Do not use `--class-balance` as a substitute: that is replacement sampling and changes within-epoch inclusion multiplicity.

**Hypothesis / measure.** More stable minibatch representation of regimes may reduce gradient noise and improve fixed-update efficiency; it may instead harm natural-frequency fitting. Control membership and total optimizer steps, then log per-batch stratum counts, per-class/group score, throughput and final score. Keep sequence prefixes and all 20,000 rows intact. If it changes batch composition but not score, gradient stability or time-to-score, discontinue. Useful artifacts: exact order identities per epoch and frozen training-only activity metadata.

### 8. Static convex blending chosen by cross-fitted contribution

**Mechanism.** For `K` independent models with source-qualified out-of-fold predictions `p_k`, choose per-target `w` on a training-only cross-fit, `w_k>=0`, `sum_k w_k=1`, maximizing the official weighted pooled correlation (or a predeclared regularized proxy), then freeze weights before outer evaluation. Compare best single, equal weights and fit-selected weights. Do not optimize on the reported outer fold.

**Unconventional transfer.** The previously recorded random-library draw selected Markowitz (1952), which treats covariance alongside individual variance in portfolio choice. That source fact motivates a *diagnostic* here: on fit-only residuals compute weighted covariance/correlation to identify models that may diversify errors. The proposed inference is that residual covariance can prioritize blend candidates; the portfolio result itself does not prove a gain for clipped weighted correlation, and low prediction correlation alone is not success. Source: [Markowitz, “Portfolio Selection”](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.1952.tb01525.x).

**Validity and overfit control.** Each base prediction used to train a blender must be produced by a model whose teacher, initialization and fit components exclude that prediction’s sequence/component. Nest weight selection inside the outer training groups, shrink weights toward equal or single-model weights, and report outer-fold performance plus group-bootstrap uncertainty. Freeze coefficients and calibration using only inner training folds. If no ancestry-valid OOF caches exist, fresh cross-fitted models are required; all-data prediction caches cannot answer this question. Artifact: bound prediction arrays, per-model ancestry, fit-unit weights and frozen blend coefficients. Serving cost is the sum of model costs; benchmark the complete endpoint.

### 9. Causal residual gate/boosting, then distill if it earns CPU time

**Mechanism.** Let `p0_t` be a frozen static blend. Fit `g_phi` to predict an additive residual using only state available at time t: `[x_t, p0_t, p0_{t-1}, …, causal rolling summaries]`; then serve `p_t=clip(p0_t + alpha*g_phi(z_<=t), -2,2)`. Reset the gate state on sequence changes. Train gate on cross-fitted base predictions, not in-sample base outputs. Compare squared-error residual training with a direct differentiable weighted-correlation objective; these are different objectives because correlation is scale/offset invariant while residual MSE also calibrates level and scale. Tune alpha/regularization inside groups and freeze them before outer evaluation.

**Order of cost.** First test a small regularized linear/Ridge residual model; then a low-width causal gate only if the linear model finds stable conditional gain. For an MLP, use shrinkage, early stopping confined to inner fits, and compare against a sham that permutes residual labels within fit-only groups. A neural residual teacher may be distilled into one causal GRU student using cross-fitted gate outputs, but evaluate the student end-to-end against parent/static blend. Distillation can compress serving cost; it does not cure ancestry leakage. Falsify if gain vanishes on outer groups, shrinks to zero, depends on inference-time labels/future values, or adds more CPU time than the endpoint budget allows.

## Minimal orthogonal first matrix

Start with a zero-fit artifact audit: discover which existing predictions have compatible source/fold/membership ancestry and at least two independent OOF sources. If valid, report parent/equal-weight/cross-fitted static-blend comparisons and fit-only residual covariance. Do not infer that compatibility from filenames.

If that audit leaves the mechanism unanswered, the first training matrix should be one frozen component-complete subset and one fixed full evaluation population; all candidates share a parsed-data cache, initialization seed, GRU architecture, AdamW/OneCycle, update count, sequence order and target treatment. Current `--fit-subset-manifest` is explicitly reserved for the E540 numeric-map screen, so do not point these arms at an E540 manifest; use a reviewed general membership receipt or add a separately validated general component-subset interface first. Bind cache consumption to the producer’s `wnn.parsed-sequence-cache` v1 manifest: verify its fold, split hash, source identities, membership artifact hash, and each array hash before any fit. Keep the producer’s measured parse/read telemetry with the arm receipts; OS page-cache hits are not proof of one parse.

| Arm | Sole factor | Control |
|---|---|---|
| A0 | Current `mix` | Reference |
| A1 | Weighted Huber, delta 0.5 | A0: loss |
| A2 | `wmse_gcorr`, predeclared `gcorr-weight` | A0: pooled/global objective |
| A3 | Muon on recurrent matrices | A0: optimizer |
| A4 | Readout `fixed_mix` | A0: readout activation |
| A5 | Input-stratified full permutation | A0: sampler/order |

These are five one-factor comparisons, not a full factorial. Cache reuse requires a hash-bound validated manifest: for each supported arm consume the producer’s parsed sequence arrays by the cohort cache API and record source/split/subset/membership hashes, bytes read, parse time, cache hit/replay and SSD artifact hashes. A claimed “parse once” must be supported by producer telemetry, not OS page-cache warmth. If TCN receives the next training slot, replace a lower-priority arm with a single TCN-vs-GRU pair at matched update budget, and keep its serving path explicitly research-only until CPU parity and latency are measured. The clipping treatment (card 3) is a follow-up because it requires a trainer option.

For any low-fidelity screen, use a precommitted complete-sequence/component subset, preserve every sequence’s full causal history, fit selection only inside the training side, and evaluate all candidates on the same fixed full set. Do not use an individual random-sequence truncation and call it component-complete. Confirm a promising direction at a higher training budget and another already-exposed fold; call that robustness, not freshness. Queue diverse arms by expected information and endpoint value, not raw decorrelation or seeds per hour.

## Recommended order and decision

1. Audit already-produced predictions and provenance; calculate per-component official scores and fit-only residual covariance where valid. This is the shortest useful test and requires no fit.
2. Fit the orthogonal loss/optimizer/readout/order screen only after the parsed-cache and exact subset receipts exist; no new scoring fit is authorized by this proposal.
3. Select a diverse teacher based on both standalone evidence and cross-fitted conditional contribution. TCN is an explicit research teacher candidate; serving/export is an open dependency.
4. Try static convex blend before learned gate; only then test a small causal residual model, with all gate/blend training cross-fitted and shrinkage frozen before outer evaluation.
5. Promote only against the complete deterministic CPU endpoint with measured total time, RAM, state reset, package size and calibration. Existing ~36-minute endpoint calibration leaves uncertain time for new models; sum-and-benchmark rather than assume ensemble capacity.

These are hypotheses about useful diversity, not claims of public-score gain. The current owned public best remains 0.6660 in the brief; this proposal does not support a new expected-public score.
