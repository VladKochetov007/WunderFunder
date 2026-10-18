# Plateau review frontier

Updated: 19 October 2026, Europe/Kyiv. Owner: root/controller.

## Resume instruction

Read the [accepted spec](../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md)
and latest [project state](../../../specs/current-state.md), then inspect live
workers and Git status. Continue implementation without reopening the plan.
Record changes and receipts here before context compaction or handoff.

## Actual stage

- Accepted plan saved and linked; no new fits or scientific scores.
- Baseline source: main `9defbcbc0f2a6f1b35941dc657496993b48889c6`.
- Four user files, including `new-meta-review.txt`, are archived verbatim;
  [SOURCES.json](SOURCES.json) binds original bytes and archived destinations.
- First A/B/C comparison: cold H512×1, seed 1001, 10% complete fitting
  components, full exposed F0, 367 equal-width inputs. History/mask implementation
  and exact run recipes are pending.
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
| Implementation | `plateau_i1_inputs_actor`, Luna/max; code only, no fits | Isolated worktree `/home/vlad/development/wunder/research_tmp/plateau_i1_inputs_20261019`, branch `refactor/plateau-i1-inputs-20261019`; focused history/mask and checkpoint integration in progress |
| Provider readiness | `plateau_kaggle_readiness`, Luna/max; read-only metadata, no uploads | Complete; indexed snapshot records ~5.02h gross remaining quota |
| Source critic | `review_direction_audit`, Luna/max; read-only claims/ancestry audit | Complete; FP16-control and newer-F3 corrections recorded in meta-review reconciliation |

The implementation actor may not approve its own scientific result or change
the recipe/budget. Root owns integration and later resource admission.

## Next actions

1. Publish the saved plan, source archives, scoped authority and source audit.
2. Implement history/masking in semantic components and review restoration and
   all feature-assembly paths.
3. Prepare matched recipes and actual-shape profile commands. Before execution,
   refresh resource controls and Kaggle access/quota without touching other agents' kernels.
4. On October 20, admit bounded local A/B/C and a paired Kaggle F2 cohort only
   after their source/data/resource dependencies are satisfied.

Do not launch a broad loss/history/sampling sweep, restart the old R4 matrix,
count draft PRs as results, or delete historical stop receipts.
