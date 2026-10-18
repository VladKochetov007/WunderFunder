---
kind: research_idea
status: implemented_draft_pending_b300_profile
created_utc: 2026-10-18T01:57:09Z
updated_utc: 2026-10-18T01:57:09Z
hypothesis_ids: [H-2026101801]
experiment_ids: [E-2026101801]
evidence_paths:
  - research/experiment_drafts/causal-predictive-state-gru-b300/README.md
  - research/experiment_drafts/causal-predictive-state-gru-b300/PR.md
  - research/experiment_drafts/causal-predictive-state-gru-b300/recipe.json
  - research/unified/sources/CAUSAL_PREDICTIVE_STATE_SOURCES_20261018_R1.md
---

# Causal predictive-state GRU: B300 experiment draft

## Hypothesis and observed gap

H-2026101801 asks whether training a task GRU to forecast its next canonical
input row encourages a recurrent representation that tracks predictable state
and improves the supervised two-output objective. This direction comes from
the user's interest in online learning, the user-supplied GRU feature-extractor
anecdote, and primary predictive-state/self-supervision papers. Those sources
motivate the test but provide no local quality evidence.

The serving API has no future labels or scoring mask. Consequently, “online”
here means a causal recurrent state updated from each observed input row;
inference-time parameter updates are excluded. The auxiliary task uses only
model inputs from the exact fitting members and does not touch held-out
components. It predicts input row `t+1` from model output at `t`, after
normalizing the target with the cell's fitting-only model mean/std. The last row
of each sequence has no target. SmoothL1 with beta 1 and fixed weight 0.10
reduces the effect of irreducible high-frequency noise.

The output head is the existing shared task readout, which receives both the
GRU output and its current-row encoded input skip. Thus an improvement would
establish value for this shared predictive auxiliary objective, not isolate
the GRU hidden state as the only carrier. A future hidden-only decoder or
OOF-hidden-state student transfer would be a distinct experiment.

## Frozen comparison and execution

E-2026101801 compares cold H512x1 GRUs with identical 211-dimensional
next-input heads, one-step batch shapes, initialization, source-qualified
component membership, fitting-only normalization, optimizer attempts and
checkpoint policy. The control has auxiliary coefficient 0.0; the candidate
uses 0.10. It uses 30 paired cells (F0/F1 seeds1001-1008, F2/F3 seeds1001-1007)
and six fixed epochs, epoch5 checkpoint, strict FP32, AdamW at1e-4, OneCycle
with5% warmup, beta2 .999, weight decay .01, clip norm1, batch8, 2000-row
chunks. Every fold is exposed development data; no fit-time outer evaluation
or checkpoint selection occurs.

Each cell runs both arms sequentially on the same GPU with seed-dependent arm
order. Eight B300 GPUs can process up to eight different pairs concurrently
after the first bounded 50-update profile passes. Each fit is independent, so
DDP is not required. Actual B300 model/driver/software, host memory, device
memory, throughput and paths are not yet inspected; no elapsed-time estimate
or capacity claim is made. See the executable [runner](../../experiment_drafts/causal-predictive-state-gru-b300/run_cell.sh)
and detailed [B300 draft](../../experiment_drafts/causal-predictive-state-gru-b300/PR.md).

## Decision and interpretation

The preregistered primary statistic is the equal-weight mean of four fold
means of per-seed paired candidate-minus-control `global_wp_clip2_v1` deltas.
Report the official metric, component paired bootstrap and shared-seed
bootstrap separately. A provisional +0.0010 absolute-WP mean effect is large
enough to prioritize a downstream representation/student-transfer study; it
is not a hard rejection threshold. A confidence interval crossing zero, lower
standalone score, or weak short profile is unresolved, consistent with the
team's exploration policy. A negative result narrows only this one-step target,
weight, capacity and schedule. It does not close multihorizon forecasting,
coarse predictable-state targets, separate predictive-state pretraining or
input-only test-time state adaptation.

## Status and lineage

- Code is implemented in the modular trainer on the isolated experiment branch.
- Published as [draft PR #18](https://github.com/VladKochetov007/WunderFunder/pull/18), with the [B300 profile and matrix handoff issue #19](https://github.com/VladKochetov007/WunderFunder/issues/19).
- No new fit, B300 profile, independent grade, submission or leaderboard result
  exists; publication is not evidence that a worker claimed or started.
- The next branch, if the result warrants it, is fold-clean hidden-state
  extraction and student transfer measured as a complete matched endpoint.
- The [source note](../sources/CAUSAL_PREDICTIVE_STATE_SOURCES_20261018_R1.md)
  separates arXiv claims and the unverified participant anecdote.

Related source note and draft packet are indexed in the [sources](../sources/README.md)
and [experiment catalogue](../../experiment_drafts/README.md). Stable records:
H-2026101801 and E-2026101801.
