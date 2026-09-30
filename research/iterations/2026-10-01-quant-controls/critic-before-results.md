# Independent specification critique before control implementations/results

I read the new contract and the old QMARK detector, plan and completed results. I have not read either new author's implementation or detailed plan. The prior result is strong prediction of selected code categories; this follow-up must separate what its detector and nuisance priors alone can create. It cannot identify economic order flow from these four coordinates.

## A defensible raw-stream surrogate has a limited preservation claim

A viable bounded control can reorder observed **four-vector raw states or their constant-run tokens**, then rerun the exact original detector and the frozen encoding. The joint vector must move as a unit; independently permuting columns fabricates unseen codes and tests a different representation. The rule must specify block boundaries, first-row handling, neutral/invalid rows, ties, RNG and every fallback before results. If whole `(raw vector, run length)` tokens are permuted, raw-value multiplicities and attached run lengths can be retained, but neighboring identical states may merge and detection still changes. If raw rows are permuted, hold-run structure is deliberately destroyed. Neither version is an exchangeability test or physically valid exchange simulator.

It is not generally possible to promise preservation of raw-value counts, neutral/invalid positions, exact event count, event gaps, retained mark histogram, local frequencies and run topology while also destroying the relevant order with a cheap unconstrained permutation. Choose a clearly limited preservation set and verify it exactly. All changed quantities must be reported rather than silently restored. In particular, rerunning detection will generally change emitted-event count and category frequencies even when the raw multiset is identical. A control with these changes is still useful, but an original-minus-control contrast combines them with changed order; it is not a pure topology effect.

The first4096-event cap is especially important. Surrogate and original may reach it at different raw rows, or only one may reach it. Preserve the registered cap/reset rule and report pre-cap count, retained count, raw-row coverage and category frequencies for both. Do not resample until event counts or losses resemble a desirable baseline. Any rejection sampler needs an outcome-independent finite bound and explicit failed-draw handling; substituting the original order on failure contaminates the null and must not be silent.

Fixed original fit-half centers are defensible as a common measurement scale. Surrogate priors must be declared independently of evaluation outcomes. A prior trained on original order and applied to synthetic paths can be misspecified; this was already visible in QMARK. A matched-fit surrogate transition prior should come only from surrogate fitting chains under the same recipe. Centers, marginal priors and smoothing must stay aligned with the declared question. Holding original `pi` while raw redetection changes mark marginals is another deliberate mismatch, not a perfect matched null. Adaptive F is still necessary, but finite-prefix adaptation does not magically remove every prior/cap difference.

## A no-self-transition baseline is insufficient and can be invalid

The detector excludes **identical consecutive raw-row vectors** only when comparing immediate raw neighbors. It does not universally exclude equal successive emitted marks:

- Distinct raw vectors can map to the same4-bit category and be emitted consecutively.
- `A → neutral → A` emits A twice, while the neutral row emits nothing.
- Invalid rows advance raw state; the first valid row afterward cannot emit, but subsequent changes can. The preceding emitted mark may be several raw rows away.

Therefore setting a category transition diagonal to zero gives impossible probability to legal observations (the old sample has5.79% same-category transitions). A soft no-repeat tendency or conditioning on raw detector state is a different, explicitly declared comparator. A hard zero requires a verified support theorem at the actual prediction unit, not intuition from the word “change.” The transition probability must remain available before observing the next mark; conditioning on its current raw vector, current category or an outcome-dependent support mask leaks the prediction target.

## Exact checks that matter before real execution

1. Raw detector tests cover repeated holds; two distinct raw states in one encoded category; return through a neutral state; invalid/extreme boundaries; sequence reset; first row; consecutive emitted updates; and no-event/one-event sequences. Batch, literal streaming and prefix outputs must match.
2. Test the surrogate's declared preservation properties directly, including a final partial block, repeated states, one-token block and boundaries. Compare raw joint-vector multisets, not only per-column histograms. For any run-token construction also verify exact original reconstruction and the stated handling of merged adjacent tokens.
3. Rerun detection on synthetic output and count events from raw transitions; never treat permuted marks as if they were guaranteed detector outputs. New same-mark transitions can be legal. Zero-support probabilities, NaN losses and empty denominators are recorded failures or explicit missing observations, never fabricated zero loss.
4. For the prior branch, show that `R_ind[i,:]=pi` and `R_only` use no evaluation-fit information; online counts are scored before incrementing and reset each sequence. With fixed `pi`, total Dirichlet-frequency log loss is exchangeable under permutations: this is a correctness invariant, not evidence that the null preserved temporal structure.
5. Preserve the same256 chains, halves, original centers, IDs and baseline encoded hashes. A matched prior is built on the opposite half using its declared surrogate RNG. Every seed is reported; no favorable seed is selected. Paired whole-chain bootstrap must align original and surrogate observations even when retained counts differ and distinguish equal-chain from event-weighted loss.

## Interpretation and stopping

If a raw surrogate reproduces much of the transition advantage, that weakens a richer sequential explanation under this preservation scheme; it does not prove that all observed dependence is a detector artifact. If it fails to reproduce the advantage, it does not identify order splitting or informed flow: changed marginal/run/gap structure, hidden raw states, static representation constraints and prior mismatch remain alternatives. A prior-matched block control can isolate the original-prior mismatch more cleanly, but neither branch establishes additional price-prediction value beyond the existing GRU.

The registered two-recipe limit should remain intact. A failed preservation check invalidates that implementation/control, not the underlying hypothesis. Report the failure and do not append a third empirical null after seeing which assumptions matter. No result automatically authorizes a price model, WP query, coefficient search or upload.
