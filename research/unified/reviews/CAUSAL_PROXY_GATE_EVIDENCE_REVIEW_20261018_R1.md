---
kind: review
status: pass_narrow_claim
created_utc: 2026-10-18T01:33:00Z
updated_utc: 2026-10-18T01:33:00Z
hypothesis_ids: [H-ONLINE-DELAYED-PROXY-20261018-R1]
experiment_ids: [E-ONLINE-CAUSAL-PROXY-GATE-20261018-R1]
evidence_paths:
  - ../../campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/recipe.json
  - ../../campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/grade_r1.json
  - ../../campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/execution_receipt_r1.json
  - ../../campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/RECIPE_AUDIT_R1.md
---

# Independent causal proxy-gate evidence review R1

## Review scope

The independent reviewer checked the frozen recipe, current predictor and
grader sources, build receipt, official grade artifact and durable attempt
receipts. The review did not rerun code, inspect prediction arrays or parquet
payloads, use target/mask payloads, or edit the repository. It is a claim and
artifact audit, not an independent numerical regrade.

## Findings

- Recipe, code, prediction and grade receipt hashes agree; the build records
  all 32 selected groups, six endpoints and no target or mask reads.
- Timing is causal as implemented: proxy output at t estimates y[t−1], is
  compared against expert predictions at t−1, then weights predictions at t.
  Gate state resets per sequence; row 99 uses the fixed prior; the declared
  warmup is 128 observations.
- The grader checks split, cache sidecar, row-group, component and sequence
  identities. It reconstructs every gate and verifies prediction parity
  before opening target or scoring-mask arrays. It then scores 62,018 rows
  with the unchanged official metric.
- The best adaptive gate, h8192, scores 0.678291 versus 0.678243 for fixed
  75/25: +0.00004795 WP, conditional paired component interval
  [−0.00003098,+0.00012853]. The interval spans zero and all three gates
  remain below TL3 (0.680621). These are fixed-prediction conditional
  comparisons over 32 exposed components.
- All three task-local worker cgroups recorded zero OOM, OOM-kill and `max`
  event deltas. During the full-build window, the parent-hierarchical `max`
  counter increased by 7,090; its OOM and OOM-kill counters did not change.
  This parent-level counter is not isolated to the gate worker, so record it
  rather than claiming zero aggregate `max` events.

## Decision and limits

Do not launch a large B300 study for this specific cached-expert proxy gate:
it shows no reliable complementary gain over the fixed blend, and TL3 is
stronger on this screen. No numerical practical-gain threshold had been
preregistered, so this is an opportunity-cost decision for the exact recipe,
not a formal mechanism falsification. It does not reject broader online
learning.

The evaluation components are repeatedly exposed F0; TL3 fitting, teacher and
initializer ancestry is unknown; the proxy predicts a shifted target rather
than using observed mature feedback; and seed, fold, selection and public
transfer uncertainty are absent. This review does not validate the existing
Kalman teacher/student B300 draft PR #17, which is a separate hypothesis.

## Source records

- [Causal proxy-gate results](../../campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/RESULTS_R1.md)
- [Recipe/code audit](../../campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/RECIPE_AUDIT_R1.md)
- [Grade artifact](../../campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/grade_r1.json)
- [Execution receipt](../../campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/execution_receipt_r1.json)
