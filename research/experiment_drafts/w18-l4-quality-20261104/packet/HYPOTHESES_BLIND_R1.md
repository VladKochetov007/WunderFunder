# Blind hypothesis cards R1 — W18/L4 quality screen

Initial commitment: **2026-11-04 10:33:43 UTC**  
Status: proposal-only, before reading other agents' cards or historical result tables. No fit, profile, or source edit was started by this scout.

## Shared comparison contract

- First resolve one eligible train-only control: audit existing W18/TL3 ancestry, or define a fresh clean GRU256x2 reference. Submitted full-corpus W18/teacher weights are not eligible validation parents.
- Fit only on the official 10,607 earlier training sequences. Preserve whole sequences and reset all state at each sequence.
- Primary endpoint: all 1,873 later official validation sequences, official scored rows, both targets, clipped-label-magnitude-weighted pooled Pearson. seq_ix is shuffled and carries no chronology.
- Validation is repeatedly exposed and therefore development evidence, not fresh confirmation. Fit no normalizer, prior, teacher, initializer, or blend weight on it.
- ROOT selects three of these eight mechanisms for the initial control + three configurations, seeds 1001/1002, maximum 20 epochs. Freeze the endpoint before fitting. Report both the standalone candidate-control delta and the fixed 0.9 * control + 0.1 * candidate delta on the identical scored rows; do not tune the blend.
- The contract's 0.0003 single-model and 0.0002 fixed-blend thresholds are screening gates, not significance guarantees or predicted gains. Strict FP32; no autocast, TF32, or training quantization. ROOT owns all admission and resource decisions.

## Cards

### H01 — Causal innovations alongside raw inputs

**Lens:** Re-encode / representation  
**Change:** Add the existing causal_innovations feature mode: causal slow/fast projected centers and scales plus standardized innovations. Fit its projected prior on official train members only.

**Flags:** --research-features causal_innovations

**Hypothesis:** Within-sequence level and volatility changes can make one static input scale unreliable. Giving the GRU both raw/event features and causal local deviation may expose that shift while retaining absolute level information.

**Predicted observable:** The added channels vary within sequences; gains, if any, should concentrate where feature distributions shift. No numeric gain is estimated.

**Falsifier:** The prior or replay fails ancestry/reset checks, or both paired fixed-blend deltas are nonpositive and the full-valid endpoint misses the frozen screen.

**Cheapest discriminator:** Check train-only prior membership/hash and existing synthetic causal/reset parity, then profile the exact 211+80 input recipe. Quality still requires paired full-train fits and the frozen full-valid endpoint.

**Likely failure and cost:** The 80 added channels may duplicate event or online-normalization inputs, and the train prior may transfer poorly. Existing code; one fitting-only prior pass and moderate input overhead.

**Why it could help the fixed 90/10 blend:** It may correct control errors on regime-relative moves; a weaker standalone model can still help if its residuals differ on scored rows.

### H02 — Age since raw-field update

**Lens:** Instrument / re-encode  
**Change:** Add the existing group_age mode: log1p(rows since last exact field change) for each of seven fixed raw-field groups.

**Flags:** --research-features group_age

**Hypothesis:** A held value may mean something different from a newly updated value. Age gives the model a causal observation-staleness signal.

**Predicted observable:** Train-only ages show nontrivial within-sequence variation and may help around sparse updates. No numeric gain is estimated.

**Falsifier:** Ages are effectively constant on fitting members or both paired fixed-blend deltas are nonpositive and the full-valid endpoint misses the frozen screen.

**Cheapest discriminator:** On fitting members only, count nonzero age values and per-sequence variation per group; use synthetic change/hold cases to verify reset semantics.

**Likely failure and cost:** If values change every row, age collapses near zero; if equality reflects serialization noise, age is misleading. Seven causal scans, seven additional channels, no fitted prior.

**Why it could help the fixed 90/10 blend:** Fresh-versus-stale timing can add a different cue from the raw value representation and correct a subset of the control's errors.

### H03 — Learned structured decay state

**Lens:** Change time / architecture / transfer  
**Change:** Replace the GRU trunk with the repository's structured-state model, using multi-timescale decay initialization and the same 211 inputs.

**Flags:** --arch structured --hidden 96 --layers 2 --dropout 0 --enc linear --decay-init multi

**Hypothesis:** Explicit learned decay channels may retain slow signals while responding to fast transitions more cleanly than a single GRU gate spectrum.

**Predicted observable:** Learned half-lives remain spread, and either standalone quality or fixed-blend complementarity improves. No numeric gain is estimated.

**Falsifier:** The fit is unstable, half-lives collapse to an unhelpful narrow range, or both paired fixed-blend deltas are nonpositive and the endpoint misses the frozen screen.

**Cheapest discriminator:** Static shape/parameter audit, then one strict-FP32 device profile and state reset/replay check before paired fits.

**Likely failure and cost:** The recurrence may underfit feature interactions or the FFT chunk path may be slow. The learned half-life cap is 8,192 rows, below the 20,000-row sequence length.

**Why it could help the fixed 90/10 blend:** Its state dynamics differ from a GRU and may repair errors tied to slow context even if the compact standalone model ranks lower.

**Literature:** S5 motivates structured state-space sequence models, but this repository model is not claimed to reproduce S5 ([Smith et al., 2022](https://arxiv.org/abs/2208.04933)).

### H04 — Separate row and quote-event clocks

**Lens:** Change time / instrument / architecture  
**Change:** Use the existing TwoClockModel: the row GRU updates each row; an event GRU updates only when raw a6 equals the endpoint sentinel, using the row state, segment mean, and elapsed-row count.

**Flags:** --arch twoclock --hidden 256 --layers 2 --event-hidden 128 --dropout 0 --enc linear

**Hypothesis:** If target changes align with sparse quote-move rows, a separate event-clock state can summarize the intervening dense stream.

**Predicted observable:** Any benefit should concentrate near event rows; the event state should add information beyond the row state. No numeric gain is estimated.

**Falsifier:** Fitting-member event support is inadequate, the event state contributes negligibly, or both paired full-valid and fixed-blend screens fail.

**Cheapest discriminator:** Count fitting-member event rows/gaps, check reset and chunk-carry semantics, then profile one exact recipe.

**Likely failure and cost:** The sentinel may be rare or weakly tied to targets; a second recurrent cell adds latency and can overfit the event subset.

**Why it could help the fixed 90/10 blend:** Event timing can supply a sharper transition correction while the reference remains smoother.

### H05 — Dilated causal TCN

**Lens:** Re-encode / change time / architecture  
**Change:** Replace the recurrent trunk with the existing dilated TCN on the same 211 causal inputs. Ten blocks retain a 1,023-row ring history; state resets per sequence.

**Flags:** --arch tcn --hidden 256 --layers 10 --dropout 0 --enc linear

**Hypothesis:** Useful signal may be dominated by local multi-scale patterns rather than all 20,000 prior rows. This deliberately tests a different temporal bias.

**Predicted observable:** A local-convolution model may be more complementary than a second recurrent student; if older context matters, standalone quality should expose that limitation. No numeric gain is estimated.

**Falsifier:** The fixed 1,023-row context misses required signal and both paired standalone and fixed-blend screens fail.

**Cheapest discriminator:** Confirm receptive field/reset parity from the stream reference, then profile exact 20,000-row strict-FP32 memory and throughput.

**Likely failure and cost:** Context may be too short; ten width-256 blocks may consume more memory or compute than expected.

**Why it could help the fixed 90/10 blend:** Its recent-pattern bias can correct errors from the longer-context GRU even if its standalone score is lower.

**Literature:** Bai et al. evaluated generic convolutional and recurrent sequence models and motivated TCNs as a natural alternative; transfer to this competition remains an empirical question ([Bai et al., 2018](https://arxiv.org/abs/1803.01271)).

### H06 — Prefix-routed sequence specialists

**Lens:** Factor / re-encode  
**Change:** Use the existing shared-plus-specialists GRU. A causal router uses the first 99 rows and chooses a sequence-fixed expert; all four small expert GRUs update every row. hidden=288 is a static approximate parameter match to GRU256x2, not a measured compute match.

**Flags:** --arch specialist --hidden 288 --layers 1 --dropout 0 --enc linear --expert-hidden 32 --num-experts 4 --specialist-route learned --prefix-rows 99

**Hypothesis:** Sequences may have stable regimes that merit small recurrent corrections. Prefix-only routing tests that factorization without future inputs or time-varying routing.

**Predicted observable:** Train-only route occupancy does not collapse to one expert, and expert outputs have distinct residual behavior. No numeric gain is estimated.

**Falsifier:** Route collapse, no expert contribution, or both paired fixed-blend deltas nonpositive at the frozen endpoint.

**Cheapest discriminator:** Static parameter count and one bounded profile; record train-only route occupancy and expert contributions before reading validation deltas.

**Likely failure and cost:** The first 99 rows may not identify a stable regime; route can overfit sequence-start quirks, and four cells add per-row work.

**Why it could help the fixed 90/10 blend:** A specialist can repair a subset of sequence regimes that the global control misses, even if average standalone score falls.

### H07 — Cross-chunk pooled weighted-correlation objective

**Lens:** Invert / objective alignment  
**Change:** Keep the model and inputs fixed; replace the clean control's current supervised loss with gcorr_pool and fixed decay 0.98. Verify exact scored-row mask, clipping, and target handling against the official metric.

**Flags:** --loss gcorr_pool --pool-decay 0.98

**Hypothesis:** The official objective is pooled weighted Pearson. The current mixed loss may also reward pointwise error or center within chunks. Cross-chunk pooled sums may better expose between-sequence score contributions during training.

**Predicted observable:** Training pooled correlation improves without variance collapse, with better full-valid official score or fixed-blend delta. No numeric gain is estimated.

**Falsifier:** Mask/clipping/target pooling differs from the official scorer, or after audit the paired full-valid and fixed-blend outcomes fail the frozen screen.

**Cheapest discriminator:** On synthetic tensors, compare with an independent pooled weighted-Pearson calculation; check finite gradients, mask handling, and chunk-memory reset. If admitted, use the same two-seed full-train fits.

**Likely failure and cost:** Prior chunks provide detached memory, so gradients flow only through current rows; decay and chunk order may make the estimate noisy, and near-constant variance can destabilize correlation gradients.

**Why it could help the fixed 90/10 blend:** Correlation fitting can trade squared error for ranking behavior and target-scale differences, making complementary residuals even when pointwise fit or standalone ranking is weaker.

### H08 — Chrono/orthogonal recurrent initialization

**Lens:** Change time / training dynamics  
**Change:** If the audited clean GRU control uses default initialization, change only recurrent initialization to chrono_orth with maximum timescale 20,000 rows.

**Flags:** --rnn-init chrono_orth --chrono-tmax 20000

**Hypothesis:** Default gates may start with timescales too short for a 20,000-row stream. Chrono biases spread update timescales, while orthogonal recurrent matrices may help gradients survive.

**Predicted observable:** Initial gate timescales cover sequence scale while preclip gradient statistics remain trainable. No numeric gain is estimated.

**Falsifier:** The control already uses equivalent initialization, gates saturate and training stalls, or both paired full-valid/fixed-blend outcomes fail.

**Cheapest discriminator:** Audit the control initializer and print the proposed initial timescale distribution; this checks the treatment but does not establish learned retention.

**Likely failure and cost:** Very long initial timescales may adapt too slowly or interact poorly with the optimizer/clipping. Existing flags; negligible parameter and inference cost.

**Why it could help the fixed 90/10 blend:** Different learned temporal horizons may correct control errors that depend on older context.

## Root shortlist

1. **H07 pooled correlation objective.** Highest information per cost: directly tests loss alignment with the official metric without changing model parameters. Admit only after matching mask, clipping, target handling, memory reset, and confirming the control does not already use an equivalent objective.
2. **H01 causal innovations.** Isolates a useful nonstationarity representation while keeping the recurrent architecture fixed. Prior ancestry and exact input cost must be frozen.
3. **H05 dilated TCN.** Strong architecture contrast against full-sequence recurrence; the fixed 1,023-row context makes the mechanism easy to falsify and may yield blend complementarity. Profile before fitting.

All predicted effects above are hypotheses, not measured gains. The eight cards are candidates for ROOT's three mechanism slots; they do not request eight candidate configurations or extra fits.

