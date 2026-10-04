# H095 disjoint-episode gain probe

`scripts/feature_gain.py` implements Card3 stageA using the current frozen
component-excluded GRU. It prepares one gain target and causal feature vector per
sequence, then fits equally sized state, input-summary and shuffled-gain probes
on one component partition and evaluates the other. Directions may be reversed
and their predictions pooled; hidden coordinates from different backbones may not.

The earlier label episode is fixed at rows1024≤t<4096. Its target is a |y|-weighted
scale of the frozen predictor, with prior strength0.01 toward scale1 and fixed
clipping[0.25,4]. At least32 scored earlier rows are required. The causal feature
anchor is the48th observed a6 lower-endpoint event following that episode. This
buffer comes from the archived30–48-event maturity estimate: it is explicitly an
assumption, not newly established maturity for every target. The later scored
episode starts250 rows after the anchor and uses the original fixed jitter sample.
Insufficient sequences are logged and skipped without adaptive resampling.

State features are the actual frozen head input at the anchor. The matched input
summary contains mean, standard deviation and average absolute differences of
raw inputs up to that same anchor. Both use fixed64-dimensional nonlinear maps,
fit-only normalization and65-coefficient ridge heads. The negative control uses
the identical state map with fixed cross-sequence shuffled fitting gain labels.
Evaluation labels never choose or fit transformations, hyperparameters or arms.

The report contains held-out gain MSE and R² relative to the fitting mean, plus
later-episode official Global WP and paired component intervals for all prediction
arms. The untouched backbone and constant fitting-mean scale are always reported.
An earlier-label oracle applies the already separated earlier episode's gain to
the later episode; it is a non-deployable diagnostic of gain stability and never
enters the causal learned predictor. Its apparent utility cannot establish that
input history can recover the gain. Positive synthetic controls plant input-visible
gain; the same pipeline must recover it while shuffled supervision fails.

These exposed-fold screens retest the hypothesis despite earlier negatives.
Candidate promotion requires predictable gain beyond the input-summary and shuffled
controls and a later-episode gain over the unchanged model. Independent confirmation
is still required. This milestone does not implement the subsequent training-only
gain auxiliary objective or claim that such an objective helps.

Both cache extraction and screening have explicit CPU/GPU draft specs under
`gain_drafts/`. Every input/code byte must be sealed and admitted by the controller
before real-data execution. Cache extraction re-runs the frozen backbone once per
sequence; it is not training. CPU jobs use one core/4GiB; GPU extraction requests
1GiB allocator/2GiB stop with the same weights, fp32 arithmetic and episode policy.
