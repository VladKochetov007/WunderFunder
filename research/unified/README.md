# Unified research library

This is the maintained, human-facing research map. It joins different result types through shared topic pages while preserving the original campaign records and evidence paths.

Latest requested review: [critical assessment and separate public and future private leaderboard strategies](reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md). This review adds proposed refinements and does not resume experimental admission.

## Start by topic

- [Ideas and hypotheses](ideas/README.md)
- [Methods, sampling, and metric](methods/README.md)
- [Reviews and audit receipts](reviews/README.md)
- [Imported source writeups](sources/README.md)
- [Model families and training](models/README.md)
- [Validation and public transfer](validation/README.md)
- [Interactive results dashboard](validation/RESULTS_DASHBOARD_20261012.html)
- [Submission choices and scores](submissions/README.md)
- [Inference, quantization, and runtime](inference/README.md)
- [Execution, incidents, and preservation](operations/README.md)
- [Dated October 12 archive](archive/2026-10-12/README.md)
- [Shared note template](templates/NOTE_TEMPLATE.md)

## How records fit together

The JSONL ledgers remain the structured source of truth: add hypotheses to `research/hypotheses.jsonl`, experiments to `research/experiments.jsonl`, and allocation or archive decisions to `research/bandit_tree/journal.jsonl`. A Markdown note gives readers a concise explanation and points to those stable IDs, the frozen recipe, and the result receipts. Do not rewrite past ledger rows or overwrite dated archives.

Use the shared template for new narrative notes. Set its `kind` and `status`, list related hypothesis and experiment IDs, and link the source receipts. Keep the topic README current by adding a link to the new note. If research is explicitly resumed, update current state and allocation records before admitting work; the existing stop receipt remains in force until then.

## Reading order

1. Read [current state](../../specs/current-state.md) for live status and constraints.
2. Open the relevant domain README above.
3. Follow its links to a dated synthesis, then to the source-qualified receipt or raw artifact.
4. Check the linked H/E record and validation exposure before repeating a claim.

The dated archive preserves the October 12 handoff as it stood when all admitted work finished, including its historical path inventory. Domain pages provide maintained links to current locations. Reviews, idea artifacts, methods references, and imported source writeups have canonical homes under this tree; use those paths directly.
