# E428 preregistration: end-to-end synthetic gain sensitivity

**Purpose.** E422's real-data gain probe found no state or input-summary
recoverability, but its existing planted-cache test bypasses the earlier-label
gain estimator. E428 calibrates the complete synthetic path through
`realised_gain`, `gain_heads`, later-row scaling, and component scoring. It does
not add evidence about whether the real GRU carries these signals.

**Design.** Generate separate synthetic fit and evaluation sequences with four
prespecified conditions: a strong state-only order cue, a weaker state-only
order cue, a mean-visible input-summary cue with uninformative state, and a null
gain independent of both feature sources. The order cue uses two arrangements
of four `+1` and four `-1` values with identical mean, standard deviation, and
mean absolute first difference. A deterministic causal accumulator at the
anchor distinguishes the arrangements. In the summary condition, gain changes
with a row-level offset visible to the existing mean/std/absolute-difference
summary while state is independent. All four conditions use common fit/evaluation
random streams for nuisance inputs, state noise, predictions and label noise;
only cue placement and planted gain strength vary. Exact equality of the two
prototype input summaries is asserted before any cache is written.

For each sequence, plant target-specific scale factors in an earlier
prediction/label window with fixed additive noise. Derive training targets only
by calling the production `realised_gain()` function. Use those estimated gains
to rescale a later, non-overlapping prediction window. Fit the existing equal
width state, input-summary, and shuffled-label probes on one set of synthetic
components; score on disjoint synthetic components using `feature_gain.run()`
and its paired component bootstrap. No competition data, GPU, model training,
or future features enter the fit.

**Pass criteria.** At the strong planted signal, the correct feature source
must recover positive held-out gain R² and beat both the mismatched source and
shuffled labels; the cue-free null must not show statistically significant
positive gain-MSE improvement over the fit-mean baseline or
positive later WP beyond its component interval. Each strong positive control
also requires a positive later-WP interval versus unchanged, mismatched and
shuffled probes. The weaker signal estimates
the sensitivity boundary. A failure at the strong signal indicates a probe
implementation or power problem; a pass supports sensitivity only for the
tested cue, sample count, gain size, and noise. Gain-MSE uncertainty uses the
same32-component grouping, with2000 draws and seed271832. A slightly positive
point R² in the cue-free null does not itself fail calibration. Crossing-zero intervals are
inconclusive. Report all conditions without selecting a winner.

**Resources and outputs.** CPU only, one thread, one child, 2 GiB hard RAM cap,
no swap, no GPU. Expected runtime is under two minutes. Save synthetic cache
hashes, per-condition reports, the aggregate JSON, stdout, and a plot beneath
`research/campaign_2026-10-04/eda/gain_probe/synthetic/e428/`. Do not run while
the local E424 worker is at its memory cap; admission requires that worker to
finish and local RAM headroom to be verified.

**Interpretation boundary.** This is assay calibration, not a predictive
experiment and not a reason to close sequence-level auxiliaries. It complements
E422's current-data screen and its existing easy planted-state unit test by
exercising noisy earlier-label gain estimation and later prediction scoring.
It constructs eight-row history summaries and a fixed synthetic anchor; it
does not exercise `episode_anchor()` or validate the48-event maturation rule.
The earlier-label oracle diagnoses cross-episode stability only.
