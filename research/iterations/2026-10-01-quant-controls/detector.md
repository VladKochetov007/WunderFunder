# DETECTOR02: raw run-packet control

The question is whether a declared synthetic raw observation process reproduces the large advantage of causal transition prediction over causal marginal-frequency prediction for the selected four-bit code marks. The old result is 0.7897565537 nats per emitted mark, averaged equally across 256 train-only source chains. It is a result about predicting this representation, without a price target or evidence of additional competition score.

## Frozen recipe

`detector-plan.json` was frozen before new source feature reads; its SHA256 is `59814066828b301ee7098a90903b36aa69dad4dfb21ad239400c1f98336294bb`. Selection, halves, centers, 4096-event caps, detector thresholds and smoothing all come from the original EDA. Three seeds were fixed in advance. No real source features were read during implementation or synthetic tests.

The unit being shuffled is a maximal constant raw run: its joint four-float32 vector **and its attached row duration**. Equality for packet construction uses all raw bits, retaining NaN payloads and signed zero. Consecutive blocks of 128 original packets are permuted, including the initial block and final partial block. Moving complete packets leaves each block's total raw-row length unchanged, so its original raw-row interval remains the same. The joint raw-row multiset, code-duration packet associations, and valid/neutral/invalid occupancy within that interval are preserved. The actual detector still uses its original numeric comparisons, finite/extreme checks and current-row nonneutral rule.

Order, transition topology, positions of neutral/invalid runs, maximal run lengths after mergers, event counts, event gaps, retained mark frequencies and the raw row reached by the cap can change. Adjacent identical packets are allowed to merge. There is no rejection, repair, preservation of favorable counts, or category-level no-self-transition restriction. Two distinct raw vectors can occupy one category; an intervening neutral state can also produce a repeat of the last emitted category.

Both marginal prior π and transition prior R are fitted separately for each surrogate seed using its own fit-half redetected marks. Its opposite half is evaluated with these priors and the same original frozen fit-half center. Original π and R are refitted and compared with the archived result; all original evaluation mark hashes and per-chain loss statistics must reproduce. Every estimator observes only preceding emitted marks, predicts before incrementing counts, and resets for each sequence.

## Estimand and uncertainty

The primary contrast is the equal-chain mean of original F−T gain minus the mean of the three surrogate F−T gains. The paired unit is a source chain. Emitted observations may differ between conditions, so this is neither an event-matched comparison nor a detector-attributable causal fraction. The event-weighted contrast uses each condition's own event denominator before averaging the three seed estimates.

The fixed paired-complete cohort requires at least one retained mark in all four conditions. Excluded chains and empty strata are explicit; no replacement or zero-loss substitution is allowed. If a stratum becomes partially empty, the intervals describe the remaining cohort rather than its original class balance. Conditional 95% percentile intervals use 1000 whole-chain resamples, stratified by original half and class, with centers, fitted priors and synthetic draws held fixed. They omit nuisance-fit, selection and distribution-shift uncertainty.

Per-chain outputs retain all detection/cap denominators, covered raw prefixes, histograms, entropy, same-category rates, raw exact-return rates, gap counts, run mergers, occupancy, raw bitwise order changes and directed raw-boundary transition multiset overlap. Boundary overlap counts matching directed transition instances without claiming their original positions survived. The block clock here is **128 raw constant runs**; the prior-control branch's 128 emitted marks has a different, generally variable raw-row horizon.

## Checks and execution

`detector_control.py --self-test` passes synthetic tests covering bitwise reconstruction and joint multisets (NaN payloads, infinity and signed zero included), partial/singleton blocks, literal/batch/prefix detector parity, same-category legal repeats, no-event/one-event cases, empty-fit smoothing, original prequential probability tests, and unequal event denominators with paired empty exclusions. The first local test used Python 3.9 and failed on `zip(strict=True)` before any data read. Its log is retained; `checked_zip` now asserts equal lengths explicitly. This is a runtime compatibility correction, with no change to the frozen experiment.

The runner verifies the archived parquet stat/footer fingerprint and metadata hashes before reading only the four raw columns, checks whole-chain exclusions, writes the selected-input manifest before feature reads, records raw input hashes, and verifies unchanged source/input identities afterward. Outputs use exclusive creation. Exceptions and internal 870-second timeouts create an immutable failure receipt; externally killed processes may leave only their resource log and partial manifest. Root controls the 900-second, two-thread, 10-GiB CPU execution. There is no GPU job, forecasting fit, target read, WP query, or external publication.

Outcome interpretation remains bounded. Reproducing much of the original gain would weaken richer-order explanations under the preserved constraints of this surrogate; failure to reproduce it would leave gap, topology, representation, cap and simulator differences unresolved. Neither result identifies physical trade direction or price alpha, and neither automatically triggers a model fit.

## Completed frozen execution

The sole real execution completed successfully in 36.88 seconds internally (37.58 seconds wall time including process startup), with peak RSS 1,054,124 KiB, exit status zero. Source/footer and all input identities passed; original mark hashes, original fitted priors, and all original per-chain losses reproduced. The original readonly result remains unchanged. The independent implementation checker also passed against runner SHA256 `362a8fc1c4e1bd092f0495037f0a3775ecd3df3288ce048c3430ff69d0fd11a4`.

All 256 source chains were paired-complete, with 32 in each half×class stratum; no empty condition or exclusion affected the estimand. Both surrogate priors were matched to their own fit condition. Results are in `detector-output/detector-result.json`; `detector-derived-summary.json` contains only transparent aggregations of that frozen output, with its source hash.

| Condition | Retained marks | Equal-chain F−T gain, nats/mark | Conditional 95% CI |
|---|---:|---:|---:|
| Original, exactly reproduced | 681,335 | +0.789757 | [0.781656, 0.798443] |
| Raw-run seed 20261011 | 670,684 | −0.031121 | [−0.033771, −0.028417] |
| Raw-run seed 20261012 | 670,649 | −0.031361 | [−0.033915, −0.028550] |
| Raw-run seed 20261013 | 670,665 | −0.031370 | [−0.033915, −0.028581] |
| Mean of the three surrogates | Separate denominators | −0.031284 | [−0.033773, −0.028550] |
| Original minus mean surrogate | Paired source chains | **+0.821041** | **[0.812124, 0.830342]** |

Event weighting gives original gain +0.786365 and mean surrogate gain −0.024118; their contrast is +0.810483 [0.798710, 0.822298]. Each condition uses its own number of emitted marks. The original point estimate agrees with the archived EDA; its interval differs slightly because this follow-up froze a different bootstrap seed, not because the original observations or model changed.

The outcome is consistent across the registered slices. Original gains by class 0–3 are +0.7252, +0.8281, +0.9406, +0.6652; every surrogate seed has negative gain in every class (class averages across seeds are approximately −0.0568, −0.0129, −0.0314, −0.0240). Both halves agree in sign. Original gain is +0.7949 over the first16 marks and +0.7894 afterward. Surrogates retain only +0.0247 to +0.0285 early gain and have −0.0319 to −0.0322 gain afterward. Thus a small initial benefit from matched transition priors does not turn into a sustained transition advantage in this surrogate.

The complete estimator comparison is sharp: equal-chain F loss rises only from 1.92528 to approximately 1.93080, whereas T loss rises from 1.13552 to approximately 1.96208. Empirical category entropy changes from 1.90475 to approximately 1.90996. These similarities are descriptive and do not establish identical events or marginals. The same-category rate changes from 5.7936% to 18.1689–18.2443%; empirical lag-one mutual information changes from 0.83887 to 0.04383–0.04407. Legal category repeats were retained throughout.

## What moved despite exact raw multiset preservation

Across the 5,120,000 raw rows, all conditions preserve 4,273,024 valid neutral rows (83.4575%), 846,968 valid nonneutral rows and eight invalid rows. Every declared block multiset and attached code-duration packet check passed. The original streams contain 1,090,920 constant-run packets and 8,646 packet blocks. Across the 256 sequences, the median of each sequence's median block raw-row span is **646 rows**, with interquartile range 366.75–934.375 and range 207.5–10,000. These are quantiles of sequence-specific medians including partial blocks, not a pooled-block distribution or a fixed 128-row/event horizon.

The surrogate changes approximately 27.81% of raw row bit-vectors at their positions. Dominant neutral occupancy makes an unchanged-position fraction unsurprising even after packet permutation; this statistic does not measure survival of event order. Directed raw-boundary transition multisets overlap the original by only 19.17–19.22% of original transition instances. There are 87,027–87,519 adjacent packet mergers (7.98–8.02% of original packets), exactly as permitted. This is a substantial alteration of temporal topology while preserving the registered static raw constraints.

| Detection/cap quantity | Original | Three-surrogate range |
|---|---:|---:|
| Detected events before cap | 826,511 | 804,708–804,897 |
| Retained after cap | 681,335 | 670,649–670,684 |
| Chains with more than4096 detected events | 80/256 | 77/256 |
| Fraction of detected events retained | 82.4351% | 83.3212–83.3450% |
| Median covered raw prefix, rows | 19,910.5 | 19,999–19,999.5 |
| Median first retained raw-row index | 10 | 1 |
| Adjacent-event gaps of one raw row | 67.0936% | 72.8556–72.8830% |
| Gaps of2–8 raw rows | 19.3186% | 14.4401–14.4852% |
| Gaps of9+ raw rows | 13.5878% | 12.6592–12.6900% |
| Exact raw-code return after one intervening emitted event | 8.0105% | 3.1788–3.2161% |

Gap fractions use all detected adjacent pairs before cap: original denominator 826,255 and surrogate denominators 804,452–804,641. Exact-return denominators are respectively 825,999 and 804,196–804,385 triples. Retained-event counts fall by 1.563–1.568%, but count similarity does not remove the observational difference: individual covered-prefix endpoint changes range from −2,958 to +5,564 rows. No counts, endpoints, mergers or draws were repaired to match the original.

## Inference supported, and remaining uncertainty

**This raw run-packet surrogate does not reproduce the original transition-prediction advantage.** Original joint raw-code frequencies, their associated durations, coarse local occupancy and the exact detector, combined with condition-matched priors, are insufficient under the registered permutation. The advantage is associated with original temporal organization that this transformation destroys. This is a narrower and better-supported statement than attributing the original effect solely to marginal code frequencies or the mere requirement that raw neighbors change.

It does **not** rule out observation or encoding mechanisms more broadly. A temporally structured encoder, static code-transition grammar, held-state protocol, neutral-run placement or unobserved feed state can still generate the original organization without economically informative trading behavior. The surrogate changes topology, gaps and cap coverage together; its contrast is not a separable causal contribution, a likelihood-ratio hypothesis test for market dynamics, or proof of physical trade-side labels. In particular, negative surrogate gain is not evidence of negative price alpha: the outcome here is the next encoded mark, and the larger adaptive context table can predict it less efficiently than marginal frequency under this synthetic process.

The result keeps a representation/state-mechanism hypothesis alive, while leaving its forecasting value unanswered. A future discriminating study should compare an explicitly causal detector/encoder-state nuisance predictor with the observed transition advantage, or test a preregistered nuisance state against existing model residuals; it must specify its own observation unit and controls before execution. No such new probe, price-target analysis, model fit, WP call or upload was performed in this cycle.
