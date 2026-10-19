# Plateau review frontier

Updated: 19 October 2026, Europe/Kyiv. Owner: root/controller.

## Resume instruction

Read the [accepted spec](../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md)
and latest [project state](../../../specs/current-state.md), then inspect live
workers and Git status. Continue the first unfinished gate without reopening the plan.
Record changes and receipts here before context compaction or handoff.

## Actual stage

- Later October19 steering makes4–6 TOTAL new fits the default initial screen
  for future drafts and prioritizes large B300 Transformer teachers→GRU transfer.
  [ADR0043](../../../specs/adr/0043-fast-screens-and-transformer-teacher-scale.md)
  and the [teacher design](../../unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md)
  preserve this direction, its independent review and unimplemented distributed
  training path. Existing B300 results were collected in the linked PR review;
  no B300/local/Kaggle job was admitted. The accepted local plan and unused15h
  allocation remain preserved; reconcile the latest priority before spending.
- Accepted plan saved and linked; no new fits or scientific scores.
- Baseline source: main `9defbcbc0f2a6f1b35941dc657496993b48889c6`. Plan/source audit
  pushed in `372af11fe`; first-comparison code integrated in `0cc795070`.
- Four user files, including `new-meta-review.txt`, are archived verbatim;
  [SOURCES.json](SOURCES.json) binds original bytes and archived destinations.
- First A/B/C comparison: cold H512×1, seed 1001, 10% complete fitting
  components, full exposed F0, 367 equal-width inputs. History/mask implementation
  is integrated with prediction/checkpoint/provenance paths. Ruff and structure
  checks pass; independent static review found no concrete first-route blocker.
  See [implementation record](IMPLEMENTATION.json) and linked review.
- [Prepared recipes](recipes/README.md) contain exact fit/predict/score and profile
  argv. Independent command-source review found no flag/path mismatch; numerical checks, complete source seal,
  actual-shape profiles and admission remain unfinished. No test or GPU job ran.
- Kaggle read-only audit found about 5.02h gross GPU quota remaining, zero provider
  reservations and no active kernels in the recent set. Refresh on admission;
  device availability and other-agent planned reservations remain distinct.
- First F0 subset is frozen: 315 fitting components, 1,118 train + 118 valid
  sequences, full 468-sequence evaluation. See `fit_f0_10pct_s1001.json`.
- The meta-review is reconciled in the spec's dated Amendment 1. Frozen-parent
  anchoring needs a fresh FP32 pair; old FP16 C is not reusable. A newer F3 wide/deep
  study belongs to another stream: collect its evidence before duplicating it.
- Model-fitting start: October 20 Kyiv. No automatic scheduled job is claimed.

## Budget and ownership

New-cycle local GPU time spent: **0h recorded**; allocation: **15h total**.
No local or Kaggle worker has been admitted for this cycle. Inspect the actual
process/provider state before relying on this snapshot. Root owns admission,
source freeze, integration and evidence handoff.

| Role | Owner and scope | Current state |
|---|---|---|
| Implementation | `plateau_i1_inputs_actor`, Luna/max; code only, no fits | Complete, actor commit `3c1936161`, integrated unchanged on main as `0cc795070`; isolated source worktree retained |
| Provider readiness | `plateau_kaggle_readiness`, Luna/max; read-only metadata, no uploads | Complete; indexed snapshot records ~5.02h gross remaining quota |
| Source/code critic | `review_direction_audit`, Luna/max; read-only source/ancestry review | Meta-review, implementation and command-source audits complete; no numerical execution |

The implementation actor may not approve its own scientific result or change
the recipe/budget. Root owns integration and later resource admission.

## Next actions

1. Read the saved implementation/command reviews and confirm the same source.
2. Complete bounded numerical, masking, causal-prefix and checkpoint restoration
   checks before fitting; source review alone cannot establish those properties.
3. Freeze the accepted source/environment, refresh resource controls and Kaggle
   quota/ownership, profile actual shapes and reserve complete paired execution.
4. On October 20, admit bounded local A/B/C and a paired Kaggle F2 cohort only
   after their source/data/resource dependencies are satisfied.

Do not launch a broad loss/history/sampling sweep, restart the old R4 matrix,
count draft PRs as results, or delete historical stop receipts.
