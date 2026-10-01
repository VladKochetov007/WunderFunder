# Independent feature and target EDA, fold 0 sample

## Sampling and limits

`selection.json` froze 48 training row groups from the feature-derived chain-excluded fold 0 fit pool and 64 validation row groups from its evaluation pool before reading labels. The sample contains 960,000 fit and 1,280,000 evaluation rows. It is exploratory: fold 0 has been inspected in earlier research, and the selected 64 groups are not a hidden confirmation set. All work ran with one CPU core and a 4 GiB cap; no GPU time was spent. Target-derived classes only stratify diagnostics. The candidate memory reads past and current feature states only.

The target metric throughout is exact-mask, clipped, globally pooled weighted Pearson. The historical OOF reference used below is the available **since/base** recipe pair, not the public **since/since** incumbent. Its original backbones did not exclude every train row group linked to the evaluation chain.

## Feature and target structure

- In the 64-group evaluation sample, 227,562 of 227,576 target-change rows coincide with an i0 price-state change; 159,065 of 386,627 i0 price-state changes leave both targets unchanged. This establishes event locking, not a causal formula for the future target.
- The first two train-fitted principal directions of i0 price-state changes explain 90.4% of pooled change variance. On class 3 events, a two-direction, train-label-fitted linear model explains 97.4% of the held-out **target-jump sum** variance, versus 30.7% of the jump contrast. The same sum model explains 18.2%, 32.0%, and 57.8% in classes 0, 1, and 2. This diagnostic regression uses past training targets and the class stratum; it is not a label-free decoder.
- Exact-mask class 3 accounts for 45.7% of selected rows and 42.5%/46.4% of globally centered, weighted target variance for t0/t1 in this sample. Class 3 rows at least ten rows after the last i0 price event account for 36.8%/40.2% of the total globally centered weighted target variance. This is a measure of target variance, not expected WP improvement.

## Causal event-memory falsifiers

The feature-only memory projects i0 price states onto two train-fitted directions. At event `i`, it retrieves at most 32 earlier states whose subsequent move has already occurred. Content weights use current-to-past state distance. Matched controls use the same completed transitions with a position kernel, a last-event value, or a fitted local mean-reversion vector. A single scalar for each feature forecast is fitted on the selected training row groups using future **feature** changes as the training outcome; evaluation inputs never include future changes.

For the next price event, the content forecast has 0.536 times the zero-predictor mean-squared error across 384,595 evaluation event forecasts. The fitted position and local-reversion controls have ratios 0.678 and 0.703. Class 3 ratios are 0.343, 0.553, and 0.546 respectively. Content retrieval predicts the next observed feature event better than these controls on this sample.

For the mean of the next four price-event states, a separate exploratory screen admits a stored transition only after all four successor events have happened. The content forecast has a pooled MSE ratio of 0.429 versus 0.471 for fitted local reversion and 0.988 for the position kernel. Class 3 ratios are 0.119 versus 0.207 and 1.010; class 1 and 2 favor local reversion. This screen was run after the next-event result and is adaptive exploration, not confirmation.

**The feature forecast gain did not become a target-score gain.** A fixed 0.2-strength weighted-ridge residual readout fitted on 32 fold 1 groups and evaluated on the same 64 fold 0 groups reduced the historical since/base OOF proxy's Global WP from 0.680822 to 0.680162 with the content feature, a delta of **−0.000660**. The unchanged proxy, output-only correction, level plus event age, last-event, position, and local-reversion variants have deltas 0, +0.000000, −0.000426, −0.000493, −0.000605, and −0.000698. Fit and evaluation chain components are disjoint, but the old backbone training provenance limits the strength of this screen. No GPU model fitting or promotion is justified from this evidence.

## Hypotheses and controls

1. **Completed-path content memory, lower priority after target failure.** Store a current price-state key with its *next four-event path* only once the path is fully observed; retrieve by current state. The target-law hypothesis is that matching completed paths may forecast the future reference better during long class 3 plateaus. Falsifier: on clean OOF backbones, a fixed nonlinear readout must beat equal-size position, local-reversion, and last-event memories by at least +0.0005 exact Global WP across the same rows, including class 3 and class-mix robustness. The one-step linear target screen already failed; a four-event feature MSE gain alone is insufficient.
2. **Causal spread-coordinate state, low priority.** Accumulate a train-fitted mapping of the two price-event directions into changes of the target sum, then anchor it to a current-state spread estimate. The mechanism is the near-deterministic class 3 sum change. Falsifier: compare against the existing static price encoder and spread auxiliary with identical parameter and runtime budgets. Reject if improvement is limited to label-derived class 3 or if drift in long plateaus exceeds the static estimator. Earlier single-row class 3 spread decoding was already nearly exact, so little incremental gain is expected.
3. **Plateau hazard distribution, speculative.** Predict a distribution over the next price-event arrival and move from causal event age and current book state, then integrate the expected future reference over four events. This targets the 36.8%/40.2% global target variance mass in class 3 after ten quiet rows. Falsifier: self-supervised next-event timing and direction must improve over the same-size event-age and position controls before any target model; then a clean OOF target head must improve exact WP. Historical phase readouts and future quote-path auxiliary losses were negative, so this is not an unchanged repeat of either.

## Artifacts and commands

- `study.py`, `selection.json`, `results.json`: event structure, price-delta geometry, and exact-mask target mass.
- `memory_falsifier.py`, `memory_results.json`: causal next-event content and matched feature controls.
- `horizon4_falsifier.py`, `horizon4_results.json`: adaptive four-event feature screen.
- `residual_probe.py`, `residual_selection.json`, `residual_results.json`: exact-mask target proxy screen, with the negative result retained.
- `provenance.json`: source hashes, exact capped commands, wall times, exit codes, and the repaired end-of-sequence indexing failure.

No result here supports an expected public score of 0.6660.
