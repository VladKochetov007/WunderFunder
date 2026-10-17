# Research evidence index

Use [current project state](specs/current-state.md) for the latest status and
effective scope. The [unified research library](research/unified/README.md) is
the maintained topic index for ideas, methods, reviews, sources, models,
validation, inference and operations.

## Evidence path

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

Research was explicitly resumed on 12 October and later paused after the
admitted work drained on 15 October. The pause remains the latest recorded
research authority; archived stop receipts and campaign plans do not resume
work. The 17 October trainer integration and experiment drafts are code and
documentation changes. They do not establish new fits, model parity, quality or
throughput results.

The 17 October repository snapshot recorded 14 experiment-prefixed draft branches
and pull requests. They are implementation proposals and handoffs, not 14
completed experiments. Review their source and measurement stage in the
[pull-request list](https://github.com/VladKochetov007/WunderFunder/pulls) and
the [current-state note](specs/current-state.md).

The official leaderboard was read on 17 October 2026: the team had public WP
0.66832 at rank 22 of 97, with the leader at 0.67243. The dated observation is
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
