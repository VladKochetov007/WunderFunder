---
kind: review
status: complete
created_utc: 2026-10-18T08:18:33Z
updated_utc: 2026-10-18T08:30:52Z
hypothesis_ids:
  - H-KALMAN-INNOVATION-GAIN-GRU-20261018-R1
experiment_ids:
  - E-KALMAN-INNOVATION-GAIN-GRU-B300-20261018-R1
evidence_paths:
  - ../ideas/KALMAN_INNOVATION_GAIN_STATE_GRU_20261018_R1.md
  - ../../campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md
---

# Innovation-gain GRU draft: read-only implementation review

## Scope

The independent reviewer inspected the code and planned comparison on the local branch `experiment/kalman-innovation-gain-gru-b300` at commit `527d2e985`. The reviewer did not edit files, inspect competition payloads, run code, fit models, profile a GPU, or recompute scores. This is static criticism, not empirical evidence.

## Findings

- **Causal/chunk state:** no defect was found by inspection. The observer carries the last two encoded measurements and a saturated row counter between chunks; the local-linear prior handles initial rows and one-row chunks. Ordinary training and prediction reset state per sequence. This does not prove numerical chunk/full-sequence parity or serving integration.
- **Precision/provenance:** source inspection found the strict FP32 path and source-hash fields in checkpoint/prediction/cache paths consistent with the draft description. No runtime receipt confirms them.
- **Control meaning:** `context_only` removes innovation only from the gain network's conditioning input. Both observer arms still use innovation in the gain-weighted correction and recurrent residual path. It is a narrow test of innovation-conditioned gain, not a control with innovation removed everywhere.
- **Student-cache blocker:** the frozen recipe's `student_cache.per_cell_source_registry` is null. Exact teacher run IDs, epochs, coefficients, checkpoints and clean-fold ancestry must be bound before student caches or fits can be run.
- **Required prefit checks:** synthetic prefix/chunk/reset/batch-isolation invariants and an actual H512 B300 profile remain unrun. Eight B300 cards are a target, not an allocated or measured resource.

## Disposition

The code draft was suitable for review as a proposal, but it is not a runnable empirical result. No B300 profile, fit, student result, quality claim, runtime claim, submission or PR was produced. The branch remains local and unpushed; its stale PR-body reference to #22 is not evidence that a PR exists. The explicit user stop now closes further work.

## Provenance

Independent read-only critique from role `mft_innovation_strategy`. See the [idea note](../ideas/KALMAN_INNOVATION_GAIN_STATE_GRU_20261018_R1.md), the local draft at commit `527d2e985`, and the [campaign closeout](../../campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md).
