---
kind: method
status: completed
created_utc: 2026-10-18T01:32:00Z
updated_utc: 2026-10-18T01:32:00Z
hypothesis_ids: [H-ONLINE-DELAYED-PROXY-20261018-R1]
experiment_ids: [E-ONLINE-CAUSAL-PROXY-GATE-20261018-R1]
evidence_paths:
  - recipe.json
  - prepare_recipe.py
  - build_predictions.py
  - grade_gate.py
  - execution_receipt_r1.json
---

# Causal proxy gate recipe audit R1

## Question

Does the frozen implementation implement the preregistered causal proxy-gated
TL3/R3 mixture, preserve source-qualified sequence identity, and keep labels
and the official scoring mask out of prediction generation and gate
reconstruction?

## Frozen method

For current row t, the frozen feature proxy produces q_t = clip(proxy(x_t),
−2, 2), a pseudo-estimate of y[t−1]. The gate compares q_t with each frozen
expert's clipped prediction at t−1, using |q_t| as its pseudo-weight. For each
target it tracks EWA squared error for TL3 and R3, plus the shared normalizer
EWA(|q_t| max(q_t², 0.01)). After 128 observations, the R3 fraction is

  clip(sigmoid(logit(0.25) − 2 × (normalized_R3_error − normalized_TL3_error)),
  0.05, 0.50).

That weight mixes the cached expert predictions for row t. Each sequence
starts fresh; scored row 99 uses the fixed 75/25 prior because no later proxy
observation exists yet. The three declared EWA half-lives are 256, 2,048 and
8,192 rows.

## Recipe-to-code audit

| Requirement | Implementation | Finding |
|---|---|---|
| Freeze source IDs, cache hashes, sidecar hashes, split, memberships and gate constants | `prepare_recipe.py` writes `recipe.json`; current code SHA-256 values are embedded | Match; independent source-only reviewer confirmed current builder/grader hashes and lineage wiring. |
| Proxy must estimate previous-row target from current features | Recipe pins the existing proxy prediction cache and its fit, evaluation, grade and execution receipts | Match; this is pseudo-label feasibility only. Target maturity is undocumented. |
| Causal update and reset | `build_predictions.py::_gate_sequence` updates from proxy[t] and experts[t−1] before mixing expert[t], with local state allocated per sequence | Match; first scored row uses the fixed prior. |
| Prediction builder is label-blind | Builder reads the recipe, proxy predictions, expert predictions and source identities; no dataset target or scoring-mask reader is imported | Match. Its receipt declares that targets and masks were not read. |
| Independent recurrence before scoring | `grade_gate.py::_independent_gate` rebuilds targetwise state with scalar Python updates and checks parity within 2e−7 before constructing `SequenceFile` or reading targets/masks | Match; all six stored endpoints passed parity. |
| Official metric and paired controls | Grader uses `global_wp_clip2_v1`, rows >=99, the official mask, fixed TL3/R3 endpoints, fixed 75/25 control, and conditional component bootstrap | Match. Resampling is conditional on these fixed predictions and this exposed component panel. |
| Source-qualified evaluation units | Grader re-loads the campaign split, checks source sidecars, row-group/component/sequence IDs, and verifies proxy fit/evaluation component disjointness | Match for the frozen 32-component F0 panel. |

## Execution and resource checks

The source check ran with Ruff; all three new Python files passed. A one-group
profile exercised the complete gate builder before the full build. The full
prediction build and independent grade then ran as separate one-CPU durable
attempts, with 1.5 GB and 2.0 GB reservations respectively. Exact argument
vectors, wall times, memory peaks, hashes and event deltas are in
`execution_receipt_r1.json`; prediction payloads remain on physical scratch.
No GPU was used. All three worker-local cgroups recorded zero OOM, OOM-kill
and `max` deltas. During the full-build window the parent-hierarchical `max`
counter increased by 7,090, with zero parent OOM/OOM-kill delta; this parent
counter is not isolated to the worker.

## Deviations and limits

There were no equation, ordering, mask, or prediction-parity deviations from
the frozen recipe. The proxy is not observed label feedback, TL3's exact
training/teacher/initializer ancestry is unknown, and the 32 F0 components
have prior model and feature-selection exposure. This audit verifies
implementation fidelity; it does not establish a clean validation result or
realized online adaptation.

## Source records

- [Frozen recipe](recipe.json)
- [Prediction builder](build_predictions.py)
- [Independent grader](grade_gate.py)
- [Execution receipt](execution_receipt_r1.json)
- [Results](RESULTS_R1.md)
- [Independent evidence review](../../../unified/reviews/CAUSAL_PROXY_GATE_EVIDENCE_REVIEW_20261018_R1.md)
