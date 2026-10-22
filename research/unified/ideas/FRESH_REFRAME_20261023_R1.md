---
kind: idea
status: proposal
date: 2026-10-23
---

# Fresh reframe: low-cost input representations for the recurrent endpoint

These are proposals, not measurements. The objective is improvement to a complete deterministic endpoint scored by `global_wp_clip2_v1` (fixed `need_prediction AND is_scored` mask, clipped labels and predictions at ±2, clipped-label-magnitude weights, mean over two targets). All local folds are exposed. No fresh holdout, new fit, data scan, or public-score inference is implied. Keep strict FP32 and causal/reset-correct sequence state; any fit must fit the existing allocation and 4–6-total-fit initial-screen policy after complete cost admission. Each candidate should be judged against its matched complete parent, with any parent/control reuse and all seed/fold exposure recorded.

The contract supplied only: about 20,000 rows, 112 anonymous inputs, two continuous outputs, streaming deterministic state; 52 `i1`/`i0` channel correspondences are hypotheses. Team GRUs are strongest. A fixed history-model mixture added small positive deltas in two exposed cells although the history model lost alone; that does not establish history as the cause. The separate online relationship-residual screen has no quality result. These observations motivate input-side alternatives but do not support a numerical prior for any card.

The cards below were drafted from that contract and the cited primary sources. A broad `rg` run accidentally exposed fragments of old idea notes before these cards were committed. I did not open the online residual-fit implementation, and I do not claim a fully blind first proposal set. The exposure is recorded so later duplicate checks and interpretation can account for it.

## 1. Shared pair-set encoder with a raw bypass

- **Source component:** Deep Sets characterizes permutation-invariant set functions with a shared element map followed by pooling and a downstream map; the paper also derives equivariant layers. The theorem requires the target function to have the relevant symmetry.
- **Structural analogy:** If the proposed 52 channel correspondences identify exchangeable pairs, each pair can be processed with a shared small map. Pair-order-neutral statistics (`sum`, absolute difference, product) can feed a pooled interaction summary, while the existing raw-input path preserves channel-specific information.
- **Intervention:** Add one narrow shared pair encoder and pooled 52-pair summary to the recurrent input, retaining the original 112 inputs as a bypass. Compare the proposed correspondence map with a fixed same-size deranged pairing; do not treat anonymous names or the proposed map as established biology.
- **Assumptions:** Useful cross-channel relations recur across pairs; enough signal survives pooling; the raw bypass protects information tied to fixed channel identity.
- **Predicted observable:** The proposed map improves the complete endpoint over its matched parent and over the deranged-map control in both exposed fold/seed cells. No numerical effect prior is assigned.
- **Cheapest informative test:** A narrow paired screen on two source-qualified fold/seed cells: four new fits if both control and challenger must be refit, or two challenger fits only if exact matched parent checkpoints and all training-stage state are reusable. No teacher fits. Grade the fixed complete endpoint with the official mask and metric.
- **Falsifier:** The true-map candidate does not beat raw-only and deranged-map controls, or pooled pair features improve only component diagnostics while lowering the full endpoint.
- **Approximate cost:** One small shared map over 52 pairs per row plus pooling; profile the full train/export path before admission because 20k timesteps multiply even small per-row cost.
- **Reusable artifact:** Frozen pair-index manifest, encoder dimensions, bypass parity check, and matched endpoint packet.
- **Mechanism family:** `input_representation/pair_equivariance`.

The seeded competition passage below reinforces a guardrail for this card: treat pair/channel selection as reversible and interaction-sensitive. It does not establish that any Challenge2 pair is informative.

## 2. Per-channel age since last meaningful change

- **Source component:** GRU-D feeds a per-feature mask and elapsed-time vector to a recurrent model and learns decay rates; its evidence concerns irregular and missing clinical measurements.
- **Structural analogy:** The 112 inputs are observed at each row, so missingness itself does not transfer. A different observable may: causal time since each channel last changed beyond a train-derived tolerance. This can describe active versus stale channels without labels or a target-fitted online head.
- **Intervention:** Retain raw inputs and append a clipped, per-channel age-since-change feature, reset at each sequence boundary. Estimate scales/tolerances from fitting data only; do not impute missing values or read validation targets to define events.
- **Assumptions:** Input activity is bursty enough that elapsed inactivity carries information beyond the GRU's state; a thresholded change is stable under strict FP32 and source-specific scale.
- **Predicted observable:** Age features help where channels are quiet or change-rate shifts; the candidate improves the full endpoint over a matched fine-tuned parent in both cells.
- **Cheapest informative test:** Two paired fold/seed cells with the same parent initialization and update budget: normally four fine-tunes total (age feature versus raw control), no teacher fits. First estimate the complete cost under current caps. An input-only reset/range check may precede fitting once separately admitted.
- **Falsifier:** Nearly every feature resets every row, ages are constant, or the candidate fails to improve the full endpoint on both cells while retaining the parent path.
- **Approximate cost:** 112 absolute-difference checks, counters and appended scalars per row; small memory and parameter increase, but a full-profile is still required.
- **Reusable artifact:** Causal age encoder, fitting-only scale/tolerance manifest, and sequence-reset fixture.
- **Mechanism family:** `input_representation/channel_freshness`.

## 3. Event-triggered recurrent state updates

- **Source component:** Skip RNN learns when to skip recurrent state updates and studies an update-budget constraint; its results on other sequence tasks do not predict this endpoint.
- **Structural analogy:** Some rows may repeat or weakly perturb the observed population state. A deterministic input-novelty trigger could leave recurrent state unchanged on those rows while still reading out a prediction at every required time.
- **Intervention:** Use a frozen, causal novelty score from standardized `x_t - x_{t-1}` to decide whether to run the recurrent update; preserve the normal update path at sequence start and reset all gate state at boundaries. Compare a small fixed threshold ladder, not label-dependent routing.
- **Assumptions:** Target-relevant state changes arrive with input novelty, and skipped updates are sparse enough to reduce actual recurrent work. A held state must remain accurate for rows whose inputs are uninformative.
- **Predicted observable:** At least one nonzero threshold skips a material share of updates, reduces measured end-to-end inference time, and retains the full official endpoint near the zero-threshold parent.
- **Cheapest informative test:** Replay a frozen parent on an already admitted exposed fold over predeclared thresholds, scoring the complete output and measuring actual serving runtime. This uses zero optimizer fits but is only an inference-time screen; a positive result would still need a train-time learned/fixed-gate comparison before a recipe claim.
- **Falsifier:** Few updates can be skipped, score falls with the first useful skip rate, or the gate's own work erases recurrent savings.
- **Approximate cost:** A few complete parent replays; no new teacher or optimizer fits for the first screen. Do not run this replay until separately admitted under the shared resource cap.
- **Reusable artifact:** Fixed threshold/normalization spec, exact skip mask, runtime trace, and causal parity fixture.
- **Mechanism family:** `temporal_control/event_triggered_state`.

## 4. TensorSketch of row-wise channel interactions

- **Source component:** Pham and Pagh's TensorSketch uses CountSketch and convolution to approximate polynomial-kernel features without explicitly materializing the tensor product, with a reported near-linear feature-map cost.
- **Structural analogy:** A recurrent model can miss a useful same-row interaction between anonymous channels unless it learns it from hidden capacity. A low-dimensional degree-two sketch offers a cheap interaction basis without creating all 6,216 channel pairs.
- **Intervention:** Append a fixed 32- or 64-dimensional degree-two TensorSketch of each standardized row, alongside the raw channel path. Freeze and hash both sketch maps; use a predetermined sketch seed, then test hash sensitivity only if the initial signal warrants it.
- **Assumptions:** Some target-relevant instantaneous pair interactions exist, and the sketch dimension retains enough of them without aliasing overwhelming the signal. The random map is an approximation, not a learned biological grouping.
- **Predicted observable:** The raw-plus-sketch candidate improves the complete endpoint over raw-only in matched fold/seed cells; interaction attribution should not be needed for this first claim.
- **Cheapest informative test:** Two paired fold/seed cells with the same strong-parent initialization and training stage: two raw controls plus two raw-plus-sketch candidates (four fits, no teacher fits), unless exact controls can be reused. Compare against a fixed zero-sketch path.
- **Falsifier:** No full-endpoint gain, improvement is unstable to a predeclared second sketch, or actual profile shows feature-map cost dominates.
- **Approximate cost:** A small CountSketch and 32/64-point convolution per row; asymptotic source complexity is not a local GPU runtime guarantee. Measure it under strict FP32 before fit admission.
- **Reusable artifact:** Seeded index/sign maps, float32 operator, dimensional/parity fixture, and inclusive profile.
- **Mechanism family:** `input_representation/sketched_interactions`.

## Seeded passage from the authorized local source pool

The random draw selected [Semenov's competition transcript](../sources/competition-writeups/semenov.txt). The exact passage is in the source receipt linked from the [source index](../sources/README.md). In context, the speaker describes trying feature removal from the opposite direction after a greedy pass; this is a source-specific account, not a general theorem. The transferable detail is to preserve a bypass and permit pair/group re-entry before accepting a one-way importance ranking. This supports a reversible comparison protocol around card 1; it is not evidence for channel value or an endpoint gain.

## Allocation and follow-up

These are four distinct proposals; they should not all be fitted automatically. The current online screen and leading warm/history delivery retain their existing ownership and admission. Choose a challenger only after source and cost review. The novelty-trigger replay offers the cheapest potential information, while the pair encoder and age representation have direct input-side paths to quality. TensorSketch is the most speculative approximate-interaction arm. Since all folds are exposed, any result remains discovery/robustness evidence; report paired deltas and separate model-seed variation from conditional component resampling. Public transfer remains unmeasured.
