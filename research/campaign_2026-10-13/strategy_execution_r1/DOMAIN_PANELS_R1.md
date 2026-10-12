---
kind: method
status: proposed
created_utc: 2026-10-12T23:19:09.717860+00:00
updated_utc: 2026-10-12T23:19:09.717860+00:00
hypothesis_ids: []
experiment_ids: [E-STRATEGY-DOMAIN-PANELS-20261013-R1]
evidence_paths: [DOMAIN_PANELS_R1.json]
---

# Initial fitting-defined domain panels

## Question or claim

Can candidate improvements persist when the relative frequency of observable
component domains changes? This freezes three of the six planned panel families
for S30 before new candidate predictions are inspected.

## Mechanism and method

[The frozen specification](DOMAIN_PANELS_R1.json) binds actual input-summary and
source hashes, fitting/evaluation group membership and cutoff values for F0–F3.
It reads only whitelisted feature change rates and distinct-value counts. No
target, target-derived class, or scored fraction enters these descriptors.
Cutpoints use the median of component means over eligible **valid-source fitting
groups only**. This is a declared restricted fitting pool; the partial train
summary is not represented as full training coverage.

Panels measure mean input-column change frequency, instrument-1 change frequency,
and mean log(1+distinct input values). The evaluation component mean determines
its domain, so complete components remain together. Whole-sequence descriptors
are retrospective evaluation labels, not model inputs or serving features.

The fixed stress mixtures give the high domain 25% or 75% of **component
frequency**, sampling components uniformly within a domain and retaining all
their selected rows. They do not force equal row mass across differently sized
components. Original clipped-label magnitude weights and the official selection
mask remain inside each component. Centered sufficient statistics recompute the
pooled correlation; domain correlations are never averaged.

## Evidence and interpretation

All twelve fold-by-panel splits exceed the minimum of 20 components per stratum.
Counts range from 60 to 86, with 141/149/144/145 evaluation components in F0/F1/F2/F3.
Three focused tests check fitting-only cutoffs, component integrity, undersized
strata, and exact agreement with physically repeated complete-component data.
They passed along with Ruff. The wider metric precheck passed 43 tests.

This is engineering and panel-coverage evidence only. No new model result or
private-performance claim follows from it. Independent review is pending.

## Limits and validity

All folds were previously exposed. Domain shifts are constructed sensitivities,
not samples from a verified future distribution. Quiet/transition, source
similarity and cross-fitted scoring-propensity panels remain pending. Their
definitions require separate frozen additions. Paired component bootstraps
condition on predictions and cannot represent fitting variation or transfer.

## Decision and next step

Review the implementation independently and apply these frozen mixtures against
the same clean C bridge when aligned predictions exist. Do not substitute a
different reference per candidate. Preserve the user's ordinary-score sacrifice
limit of 0.001 when the full private-endpoint ranking becomes available.

## Source records

- [Panel constructor](../../../scripts/freeze_strategy_domains.py).
- [Stable mixture evaluator](../../../src/wnn/domain_stress.py).
- [Focused correctness tests](../../../tests/test_domain_stress.py).
- [Actual cache audit](DOMAIN_CACHE_SCHEMA_OBSERVATION_R1.json).
- [Frozen evaluation precheck](EVALUATOR_PRECHECK_R1.json).
- [Coverage matrix](COVERAGE.json), S30.
