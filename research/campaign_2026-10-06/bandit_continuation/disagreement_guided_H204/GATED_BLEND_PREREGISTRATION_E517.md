# E-517 preregistration: rowwise disagreement-gated weak-model blend

Registered 2026-10-07 before running E-517 on the prediction sources. This is
one additional user-directed, exposed-development screen. It fits blend
coefficients but trains no neural model and cannot establish a public gain.

## Research question and bandit placement

Can `ab12`, whose standalone score is weak and whose chain-level disagreement
screen was informative for target 0, help the C1 anchor only on scored rows
where its prediction differs substantially from the anchor? E-514 tested
fixed weak-pool scalar rules; E-516 tested a target-wise simplex on the same
pool and collapsed to the anchor. E-517 tests one nonlinear, rowwise routing
rule motivated by E-515's `ab12`/target-0 error-location signal.

The user explicitly requested further changes to this direction after E-516.
This registration narrowly supersedes E-516's no-further-same-pool-search
boundary for this single frozen gate screen. It does not restore freshness:
the candidates, fold-0 data and earlier outcomes are all exposed. This screen
earns no numeric bandit update. Do not run another gate threshold, target,
candidate, score selector or same-pool variation after E-517. A pass can only
motivate a new ancestry-audited candidate cohort and separately registered
confirmation on genuinely unexposed groups; a failure closes this exact gate.

## Frozen inputs and population

- Reuse only the verified E-514 V3 source manifest and fixed three-fold
  complete-chain partition, with SHA-256 digests
  `8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd` and
  `e9e0cb9391789f54e34c1d02836d02c336c5d3a13bde6f9473196bbbc71da1c2`.
- Candidate: `ab12` at scale 1.0; anchor: the unchanged E-511 C1 seed-3/seed-4
  epoch-1 mean. All groups and outcomes are previously exposed.
- Fit on 82 whole chains and evaluate on the held-out 41 chains in each frozen
  outer fold; every one of the 123 chains receives one out-of-fold prediction.
- Target 0 alone receives an `ab12` blend. Target 1 remains the anchor in every
  endpoint. The official mask and warmup are unchanged; training weights are
  `abs(clip(target, -2, 2))`.

## Frozen gate and controls

For each outer fold, clip calibration targets, anchor and `ab12` predictions to
±2. On scored calibration rows, set `d = ab12 - anchor` and set the threshold
to the linearly interpolated 75th percentile of `abs(d)` (`numpy.quantile`,
`method="linear"`). A scored or unscored inference row is gated when
`abs(d) >= threshold`; ties are all included. This is an inference-time
prediction-only gate and uses no target, residual, chain identity or future
sequence information.

Fit `beta_gate` by minimizing the official weighted squared-error surrogate
only on scored calibration rows inside the gate, with `beta_gate` constrained
to `[0, 1]`. The gated prediction is `clip(anchor + beta_gate * I(abs(d) >=
threshold) * d, -2, 2)`. As a matched routing control, fit `beta_global` on all
scored calibration rows with the same objective and constraint, then predict
`clip(anchor + beta_global * d, -2, 2)`. The anchor is the unchanged control.
No intercept, member search or threshold search is fitted. Score all endpoints
with `global_wp_clip2_v1`.

Report pooled and outer-fold endpoints, coefficients, threshold and realized
gate mass. For every evaluation fold report (i) the count and fraction of
officially scored rows routed to `ab12`, and (ii) the fraction of total official
metric weight routed there, where per-row official weight is
`abs(clip(target, -2, 2))`. These are distinct quantities because the gate can
select rows with different target magnitudes. Also report calibration gate row
and official-weight fractions. Report paired, complete-chain bootstrap
intervals stratified by the frozen outer folds for gated-minus-anchor,
global-minus-anchor and gated-minus-global, conditional on fitted fold rules.
Use 2,000 replicates and seed `3772181504`, derived from the first eight hex characters of
`sha256(sensors stdout)` (`e0d6ec0015e849a93ffaeca8df29ef9e21ceaf36575d4156fbd7e94810036161`).
Also save per-chain descriptive official WP and plots under this lineage's
`plots/` directory; per-chain WP is non-additive and not the primary endpoint.

## Decision rule and exposure boundary

The development screen passes only if all conditions hold: (a) pooled gated WP
exceeds both the anchor and global `ab12` blend; (b) the paired-bootstrap 95%
lower bound for **gated minus anchor** is above zero; (c) the paired-bootstrap
95% lower bound for **gated minus global `ab12` blend** is above zero; and (d)
`beta_gate > 0` in at least two outer folds. The `global minus anchor` interval
is reported as a secondary comparison and does not replace either required
bound. Even a pass is not independent confirmation, a public forecast or a
numeric bandit reward. If any condition fails, close this exact gate on this
pool and do not tune it on these chains. Preserve E-514, E-515 and E-516 as
distinct ancestors; do not add their deltas or treat E-515's association as a
score gain.

## Execution envelope and deliverables

Run one CPU core, maximum 2,500,000,000 bytes for any new process, no swap and
no GPU. The explicit notebook action calls
`run_registered_e517(project_root, inputs)`. Its upstream `inputs` cell calls
the frozen E-514 identity-checking loader once; the action reuses that already
verified in-memory object to avoid loading a second several-hundred-megabyte
copy. The runner checks the registered source, partition, row-group order,
dimensions and chain-fold counts, and supplies the registered bootstrap seed.
Standalone runner use without the shared object invokes the same frozen loader.
The Marimo action is gated behind a visible run button that is inactive until
clicked; a persistent one-shot marker prevents accidental same-ID reruns after
the explicit start.
It writes the result JSON, both OOF arrays and per-chain CSV;
the same explicit notebook action writes the preregistered pooled/fold plot in
`plots/`. Capture code/source/output hashes, result timestamp, process exit,
cgroup peak/current memory and OOM counters in a separate terminal resource
receipt after execution. Do not execute this from any other reactive cell or
rewrite the E-516 artifacts. The one permitted reactive analysis cell is the visible,
click-gated E-517 action described above; it must remain inactive until the
explicit button action. No other notebook cell may invoke the registered
runner.
