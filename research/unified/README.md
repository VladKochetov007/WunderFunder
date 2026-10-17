# Unified research library

This is the maintained, human-facing research map. It joins different result types through shared topic pages while preserving the original campaign records and evidence paths.

The [critical assessment and separate public and future private leaderboard strategies](reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md) audits the previous approach and sets out ranked experiments, model and representation scaling, and selection for the public leaderboard versus the user's two-to-four-week future-data scenario. Its proposals are now being implemented under the [subsequent explicit user resume](../campaign_2026-10-13/strategy_execution_r1/README.md).

Independent second review: [first-principles verdict and plan](reviews/FIRST_PRINCIPLES_REVIEW_PLAN_20261012.md), merged from `claude/stoic-euler-70wff3`. This static critique proposes different priorities; it contains no new fit or recomputed score. The [evidence assessment and experiment cards](ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md) distinguish useful compact-model, averaging, symmetry and cross-fitting experiments from unsupported capacity ceilings and policy proposals. The user's existing objective and authorization remain active.

## Start by topic

- [Ideas and hypotheses](ideas/README.md)
- [Methods, sampling, and metric](methods/README.md)
- [Reviews and audit receipts](reviews/README.md)
- [Imported source writeups](sources/README.md)
- [Model families and training](models/README.md)
- [Private HF model archive](models/huggingface/README.md) — checkpoint/run lookup, descriptions, exact local-to-Hub paths and private transfer evidence.
- [Validation and public transfer](validation/README.md)
- [Interactive results dashboard](validation/RESULTS_DASHBOARD_20261012.html)
- [Submission choices and scores](submissions/README.md)
- [Inference, quantization, and runtime](inference/README.md)
- [Execution, incidents, and preservation](operations/README.md)
- [Dated October 12 archive](archive/2026-10-12/README.md)
- [Shared templates](templates/README.md) — narrative notes and experiment draft PR descriptions.

## How records fit together

The JSONL ledgers remain the structured source of truth: add hypotheses to `research/hypotheses.jsonl`, experiments to `research/experiments.jsonl`, and allocation or archive decisions to `research/bandit_tree/journal.jsonl`. A Markdown note gives readers a concise explanation and points to those stable IDs, the frozen recipe, and the result receipts. Do not rewrite past ledger rows or overwrite dated archives.

Use the shared template for new narrative notes. Set its `kind` and `status`, list related hypothesis and experiment IDs, and link the source receipts.

Every useful research item must be linked and described in its corresponding folder's README: reviews, plans, ideas, data and literature sources, evidence, results, analyses, plots, artifacts and operational findings. Explain its contents, relevance, finding or status, provenance and material limitations. Add this entry as part of the same change or handoff, including older unindexed sources when they are used. Each new subfolder needs a README linked from its parent. The full rule is in [AGENTS.md](../../AGENTS.md#research-note-organization).

Frozen campaign artifacts stay in their original locations and are linked here. For large or private data and weights, link their provenance manifest and approved access location rather than copying payloads. The October 12 stop is preserved as history; the subsequent strategy campaign records the current user authorization, allocation and admission checks.

## Reading order

1. Read [current state](../../specs/current-state.md) for live status and constraints.
2. Open the relevant domain README above.
3. Follow its links to a dated synthesis, then to the source-qualified receipt or raw artifact.
4. Check the linked H/E record and validation exposure before repeating a claim.

The dated archive preserves the October 12 handoff as it stood when all admitted work finished, including its historical path inventory. Domain pages provide maintained links to current locations. Reviews, idea artifacts, methods references, and imported source writeups have canonical homes under this tree; use those paths directly.

## Current execution

[October13 strategy implementation and experiments](../campaign_2026-10-13/strategy_execution_r1/README.md) records the subsequent explicit user resume, every actionable review item, full Kaggle/local GPU authority, accepted private-risk preference, owners and evidence. The earlier stop/archive paragraphs describe historical scope; ADR0034 now governs new research. No new model result at admission.
