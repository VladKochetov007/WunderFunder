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

Results pending root-controlled execution.
