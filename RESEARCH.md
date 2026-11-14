# Research evidence index

Use [current project state](specs/current-state.md) for the latest status and
effective scope. The [unified research library](research/unified/README.md) is
the maintained topic index for ideas, methods, reviews, sources, models,
validation, inference and operations.

## Evidence path

Current campaign: [November14 local results and quality follow-up](research/campaign_2026-11-14/README.md). The 18-fit queue produced [three checked submission options](research/campaign_2026-11-14/local_r1/SUBMISSIONS_R1.md). The [resumed quality wave](research/campaign_2026-11-14/quality_r2/README.md) corrected legacy Muon precision, completed strict KD90 s3, and is testing teacher targets and fixed weight averaging. Its [frontier](research/campaign_2026-11-14/quality_r2/FRONTIER.md) records cloud queue state, bounded conditional seed5 confirmation and thermal handling; no public improvement is promised.

Historical campaign: [October24 research, all-PR recovery and live EDA](research/campaign_2026-10-24/resume4h_r1/README.md). Two [checked exploratory quota ZIPs](research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/RESULTS_AND_HANDOFF_R1.md) are delivered; F1's fixed correction shrink is essentially flat. The requested [additional Astra/ultra fundamentals review](research/campaign_2026-10-24/resume4h_r1/astra_fundamentals_r1/README.md) supplies seven deduplicated cards and an actionable no-fit probe. The earlier [ten-reviewer synthesis](research/unified/reviews/astra_ultra_audit_20261023_r1/SYNTHESIS.md) and [all ten reports](research/unified/reviews/astra_ultra_audit_20261023_r1/README.md) remain preserved. Research is resumed within the current bounded allocation; no public gain is promised.

Evening October24 additions: the [independent value/blind-spot review](research/unified/reviews/independent_value_review_20261024_r1/SYNTHESIS_R1.md) (three fresh-context reviewers; endgame mechanics, noise-floor/TL3-anatomy elephants, ranked actions) and the [own-optimizer Muon research](research/experiment_drafts/own-optimizer-muon-20261024/README.md) (proposal, synthetic screen, matched real-checkpoint rank diagnostic, strict-FP32 Newton-Schulz compliance patch). Analysis plus bounded synthetic experiments only; no competition-data fits or uploads.

1. Read the [research objective](specs/research-objective.md) and
   [current state](specs/current-state.md).
2. Follow a topic from the [unified library](research/unified/README.md) to its
   source notes and campaign receipts.
3. Check the append-only [hypothesis ledger](research/hypotheses.jsonl),
   [experiment ledger](research/experiments.jsonl), and the
   [research-tree index](research/bandit_tree/generated/INDEX.md) for IDs and
   lineage.
4. Use the frozen recipe, membership, provenance and grade files in the
   originating campaign to verify a result. A summary or ledger row alone is
   not a complete result receipt.

## Status and evidence limits

The active [October21 cycle](specs/plans/2026-10-21-next-local-kaggle-cycle.md) has 30 local hours with a protected 12-hour production reserve. After the explicit postboot resume, the [online relationship panel](research/experiment_drafts/online-relationships-20261022/README.md) reached its original worker deadline with seven of twelve heads saved. Thermal safeguards kept research at one CPU core with no swap; no allocation amendment was approved. Quality grading remains closed for incomplete coverage. The [frontier](research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md) records the preserved caches/heads and pending budget decision; earlier snapshots below are history.

The user requested implementation of the [accepted October 20 local/Kaggle plan](specs/plans/2026-10-20-plateau-review-local-kaggle.md): 15 local GPU-hours plus verified available Kaggle quota in parallel, without a mandatory 30-run matrix. Preparation is active; the [frontier](research/campaign_2026-10-20/plateau_review_r1/FRONTIER.md) records actual progress and dependencies. The October 18 continuous campaign was stopped; its [closeout](research/campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md) preserves completed evidence. No new fit or score is implied by the plan. Completed deadline probes and released ZIPs remain in the [October 18 overview](research/campaign_2026-10-18/README.md).

The 17 October repository snapshot recorded 14 experiment-prefixed draft branches
and pull requests. They are implementation proposals and handoffs, not 14
completed experiments. Review their source and measurement stage in the
[pull-request list](https://github.com/VladKochetov007/WunderFunder/pulls) and
the [current-state note](specs/current-state.md).

The official leaderboard was read on 17 October 2026: the team had public WP
0.66832 at rank 22 with 97 team submissions, with the leader at 0.67243. The dated observation is
linked to the [official leaderboard](https://wundernn.io/connectome/leaderboard).
It does not identify the exact ZIP or submission ID that produced the current
team best, and it says nothing about private-leaderboard performance.

The 17 October maintenance index also links the
[repository audit](research/unified/reviews/REPOSITORY_AUDIT_20261017.md),
[current model catalog](research/unified/models/CURRENT_MODEL_CATALOG.md),
[validation protocols](research/unified/validation/VALIDATION_PROTOCOLS.md),
and [storage map](research/unified/operations/STORAGE_MAP.md). These maintained
summaries route readers to their hash-bound evidence receipts.

## Where evidence lives

- research/unified/ — maintained topic navigation and synthesis.
- Campaign folders under research/ — dated plans, recipes, execution logs and
  result receipts. Preserve their historical state.
- research/hypotheses.jsonl and research/experiments.jsonl — append-only
  idea and experiment records.
- research/bandit_tree/ — links hypotheses, experiments and combinations.
- artifacts/, cache/ and submissions/ — run outputs, selected caches,
  package manifests and release evidence. Check their receipts before treating
  a path as current or portable.
- specs/ — objective, current authority and durable decisions.

See [AGENTS.md research note organization](AGENTS.md#research-note-organization)
for indexing and provenance requirements. The large
[12 October handoff](research/RESEARCH_COMPLETE_HANDOFF_20261012.md) and the
[current-state history archive](specs/current-state-history-20261017.md) are
dated historical records, not current authority.
