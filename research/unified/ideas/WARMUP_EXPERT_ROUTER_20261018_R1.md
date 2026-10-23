---
kind: result
status: complete
created_utc: 2026-10-18T04:13:18Z
updated_utc: 2026-10-18T04:16:53Z
hypothesis_ids: [H-WARMUP-EXPERT-GATE-20261018-R1]
experiment_ids: [E-WARMUP-EXPERT-GATE-20261018-R1]
evidence_paths:
  - ../../campaign_2026-10-18/online_features_r1/warmup_expert_gate_r1/README.md
  - ../../campaign_2026-10-18/online_features_r1/warmup_expert_gate_r1/result_r1.json
---

# Warmup-only routing among related experts

## Question or claim

Do summaries from a sequence's first 99 input rows predict which of four existing endpoints will be strongest later in that sequence?

## Mechanism and method

A standardized ridge gate used mean, log standard deviation and normalized early-to-late drift for 13 input groups. It was trained on per-sequence official scores for four source-qualified F0 experts, with component-grouped four-fold cross-fitting. The frozen recipe and full implementation record are in the [campaign packet](../../campaign_2026-10-18/online_features_r1/warmup_expert_gate_r1/README.md).

## Evidence and interpretation

On 468 repeatedly inspected F0 groups/141 components, the gate scored 0.680833 WP versus 0.681227 for the training-selected fixed parent and 0.680826 for an equal mixture. It lost to the fixed parent in all four held-out component folds. The paired component interval for gate minus fixed parent was [−0.000748, −0.000088], conditional on fitted cross-fit gates. Input-dependent weight dispersion was 0.0121–0.0173 across folds, below the preregistered 0.10 threshold in every fold. The gate stayed near equal weights.

## Limits and validity

All candidate endpoints and labels had prior F0 exposure. Cross-fitting prevents within-screen gate label reuse, not validation exposure, selection, seed or transfer bias. This result narrows this warmup-summary router over these four related experts. It does not falsify richer online adaptation, actual matured feedback, Kalman state features, or teacher/student transfer.

## Decision and next step

The predeclared B300 admission rule failed on all three criteria, so no new B300 draft was created for this router. The existing retained-raw Kalman teacher/student direction remains in [draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17). Broader online-learning work remains open; any new large-scale handoff needs a distinct mechanism and local evidence that can change the decision.

## Source records

- [Frozen recipe](../../campaign_2026-10-18/online_features_r1/warmup_expert_gate_r1/recipe.json)
- [Preflight](../../campaign_2026-10-18/online_features_r1/warmup_expert_gate_r1/preflight_r1.json)
- [Grade](../../campaign_2026-10-18/online_features_r1/warmup_expert_gate_r1/result_r1.json)
- [Execution receipt](../../campaign_2026-10-18/online_features_r1/warmup_expert_gate_r1/execution_receipt_r1.json)
- [Independent review](../../campaign_2026-10-18/online_features_r1/warmup_expert_gate_r1/independent_review_r1.md)
