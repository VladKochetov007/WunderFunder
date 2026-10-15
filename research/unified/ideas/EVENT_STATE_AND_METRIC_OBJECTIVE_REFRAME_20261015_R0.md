# Fresh Challenge2 quality mechanisms: channel age and pooled metric loss

Status at commit: independent proposal draft; existing-idea duplicate audit intentionally pending. These are hypotheses, not measured results or allocated fits. Scope is the root Challenge2 endpoint only.

## Research question

Can the current causal 211-column view and target-only training objective leave useful event-timing information or official-metric structure unused? Current code starts with 112 raw columns, appends causal event features, and exposes only three shared recency counters (`since_i0px`, `since_i0trade`, `since_i1px`). The scorer pools all selected rows per target, clips targets and predictions to ±2, weights each row by the magnitude of the clipped target, and computes weighted Pearson correlation (`global_wp_clip2_v1`).

Known measured context from the active packet: the clean T3 student scored 0.68115536 on repeatedly inspected F0 (468 groups / 141 components), versus 0.68094121 for its exact parent; the paired delta was +0.000214 with 95% interval [-0.003385, +0.003095]. This does not isolate teacher value. Current H384 work and matched alpha-0 follow-up have ownership and admission elsewhere; neither is changed by these proposals.

## Card A — per-channel hold-age state

**Hypothesis ID:** `H-CHANNEL-HOLD-AGE-20261015`. **Family:** causal event-time representation.

**Question and competing explanations.** Does a quote/volume field's own age since its last change carry predictive context beyond the three existing aggregate clocks? The existing GRU may infer it from repeated raw values, or the field ages may be redundant/noisy. Alternatively, individually stale versus newly refreshed levels may distinguish a meaningful update from persistence, and the recurrent model may not learn 112 clocks reliably from its current input.

**Source component and analogy.** Engle and Russell's Autoregressive Conditional Duration work treats inter-arrival duration as a time-varying state in transaction data ([NBER working-paper record](https://www.nber.org/papers/w4966)). Eisler, Bouchaud and Kockelkoren find that order-event impacts can depend on prior order-flow history and that event arrivals themselves respond to earlier events ([primary paper](https://arxiv.org/abs/0904.0900)). These papers support the plausibility of event timing/history; neither establishes that per-field ages improve this competition. A sampled competition writeup also reports duplicate/frequency features, but its transductive static-table mechanism does not transfer directly to this causal stream; see the random-source record below.

**Intervention.** For each raw feature `j`, update `age_j(t)=0` when `x_j(t) != x_j(t-1)`, otherwise `min(age_j(t-1)+1, 10000)`; append `log1p(age_j(t))` to the current 211 causal inputs. Reset all ages at sequence start and initialize row 0 to zero. Keep the original inputs unchanged. Extend a matched parent input matrix with a zero-valued block so the added channels begin function-preserving; validate batch/stream reset and parity before fitting. No labels, masks, future rows or evaluation-group statistics enter the feature.

**Cheapest discriminating measurement.** On the fixed source-qualified evaluation groups already used by the matched parent, calculate official WP and componentwise residual contribution by per-field-age quantile and event type. This is a diagnostic of error concentration, not a quality claim. If it shows structure, run a matched A/control screen on complete sequences and fixed membership; use the same checkpoint, target treatment, optimizer budget, loader order and grader. This screen is exposed-fold development evidence.

**Falsifier.** Age buckets show no stable residual separation across both targets and several event families, or the matched model fails to improve the complete official score and produces no consistent component-level gain. That closes this exact age encoding at the screened fidelity; it does not prove all event-time state is useless.

**Cost and artifact.** Feature state is 112 counters per sequence (about 448 bytes at int32), with O(112) comparisons per row and 112 added model inputs. Training and callback cost rise; there is no extra served head. The reusable artifact is a reset-safe `StreamingChannelAge` transform plus a parity/quality receipt. Quantized size, one-row runtime and competition-host budget must be measured for any promoted endpoint.

## Card B — optimize pooled weighted correlation while retaining a pointwise anchor

**Hypothesis ID:** `H-POOLED-WP-SURROGATE-20261015`. **Family:** metric-aligned learning objective.

**Question and competing explanations.** Does a pointwise regression objective spend updates on amplitude/error reductions that do not improve the pooled weighted-Pearson score, or does the pointwise term provide essential calibration and stable gradients that a direct correlation objective loses? The scorer is invariant to many scale/offset changes before clipping; this makes a correlation term plausible, but clipping, label-magnitude weights, small batch variance and chunk boundaries can make a poorly aligned surrogate noisy.

**Source component and analogy.** The authoritative source here is the competition's own evaluator, [`src/wnn/gmetric.py`](../../../src/wnn/gmetric.py), which defines stable pooled centered-moment `global_wp_clip2_v1`. The trainer already exposes `wmse_gcorr` and `gcorr_pool`; inspect their exact weighting, clipping, accumulation, detach and decay semantics before reusing them. No external paper claim is needed to assert equivalence: the hypothesis is specifically that the training gradient can better track this evaluator when it uses the same clipped targets and absolute-clipped-target weights.

**Intervention.** Freeze all model, target, data-order, optimizer, schedule and checkpoint settings. Compare the matched label-only control with a predeclared hybrid objective `L = L_WMSE + λ(1 - WP_batch)`, where `WP_batch` uses the official clip and row weights. Calibrate one fixed `λ` using only the training split's first 16 batches to match the initial gradient norm of the WMSE term, then freeze it before any evaluated predictions are read. Do not optimize or select λ on evaluation groups. First verify exact analytic/numeric loss behavior on constant predictions, clipped values, zero-weight rows and tiny-variance batches. Keep pure WP as a diagnostic only unless the hybrid indicates its pointwise anchor is unnecessary.

**Cheapest discriminating measurement.** Replay gradients and loss on a fixed train-only batch set and compare gradient direction to a finite-difference official pooled-WP change for small readout perturbations; this is a training-data mechanism check, not a model result. Then fit one matched fixed-budget hybrid/control pair on complete sequences and evaluate the frozen official endpoint on identical source-qualified groups.

**Falsifier.** The hybrid's gradient has no better agreement with actual official-WP change than WMSE, is unstable under low-variance chunks, or the matched model fails to improve official WP/component behavior. A screen failure lowers this specific implementation's priority; a wider interval alone does not kill the mechanism.

**Cost and artifact.** No inference or package cost if the objective changes only training. Extra pooled moments and correlation derivatives add bounded training arithmetic. The artifact is an exact loss/gradient receipt and paired complete-endpoint predictions. The existing `gcorr_pool` is not assumed equivalent without equation-to-code audit.

## Explicit interaction and bounded follow-up

If either singleton screen has useful or unresolved signal, evaluate a registered 2×2 comparison: C=matched parent continuation; A=per-channel age only; B=pooled-WP hybrid only; A+B=both. Reuse exactly the same fitting membership, seed, schedule horizon, target treatment and fixed checkpoint. Zero-extend the same parent for A and A+B. The primary interaction contrast is `(A+B - A) - (B - C)` on the official pooled metric and matched connected-component bootstrap; never add parent gains. Use the project's complete-sequence small-data screen with a fixed screen budget and fixed evaluation membership, then spend full-scale confirmation only if the result or uncertainty justifies it. Keep all inspected folds labeled exposed; confirmation across folds/seeds is robustness evidence, not a fresh holdout. Root owns resource admission and all fits.

**Runnable follow-up owner:** root / designated GPU experiment owner. After the healthy active H384 chain and already-authorized alpha-0 delivery finish, freeze a source-bound parent, use one allowed GPU worker, run C/A/B/A+B with the same seed and a pre-registered complete-sequence screen recipe, export predictions through the unchanged causal serving path, and grade with `global_wp_clip2_v1`. The screen is not authorized by this note and must not displace the active chain. If screen evidence is positive or materially unresolved, root may register a distinct seed/fold confirmation; do not infer public score from local change.

Approximate cost is one matched control plus three treatment fits at the chosen screen budget; each arm has one model-training run, and only A increases serving work. The follow-up should record updates, checkpoint, seeds, ancestry, group/component exposure, per-arm resource use, official paired score, uncertainty, quantized artifact size and recurrent runtime. No numeric priors or public forecast are assigned here.

## Seeded random-source attempt

Selection seed: `20261015`. Procedure: sort all non-README `.txt`/`.md` files under `research/unified/sources/competition-writeups/`; flatten non-heading paragraphs of at least 200 characters; choose one uniformly with Python `random.Random(20261015)`. Selected source: [`rogii_top1.txt`](../sources/competition-writeups/rogii_top1.txt), passage (paragraph candidate 59):

> Below, "default" refers to models using GR, TVT, and z_diff channels without particle-filter or XY-neighbor features. Models also differ slightly in augmentation and training configurations. (for version difference details see version_comparsion.md in project_files section)
> Model  Features  Weight (general / no XY)  CV (3 seeds × 5 folds avg)  Public LB  Private LB
> Weighted ensemble  —  —  4.627  5.980  5.639
> 0719_V1  Default  0.07 / 0.40  5.09  5.648  6.130
> 0729_V3  + Particle filter  0.00 / 0.20  5.53  6.202  6.768
> 0724_V1  + XY-neighbor channels  0.28 / 0.00  4.86  6.095  5.831
> 0803_V2  + XY-neighbor channels + Particle-filter channels  0.28 / 0.00  5.00  6.185  5.778

Transfer attempt: the writeup's particle-filter channels were not a universal winner; weights change with model context and the weighted ensemble is strongest on its reported CV metric. The defensible component is to treat an input mechanism's value as conditional and compare the complete combination. This motivates the predeclared A/B/A+B comparison, not importing particle filtering, XY features, its numeric weights or its public scores. Its features, target and validation setup are materially different from this challenge.

## Evidence and limitations

The current parent/student comparison is one exposed fold and one training lineage. The two cards are proposals only; no new data payloads, fits, model/provider calls, serving changes or public submissions were made by this scout. Any later CV mean must use source-qualified folds and distinct fitting seeds with ancestry explicit.
