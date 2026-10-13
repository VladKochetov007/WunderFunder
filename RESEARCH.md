# Research library

Start with the [unified research library](research/unified/README.md) for a topic-based guide, shared note format, and links to the dated archive. The [October 12, 2026 archive](research/unified/archive/2026-10-12/README.md) contains the split final handoff.

The [public and future private leaderboard strategy](research/unified/reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md) reviews the previous approach and gives separate selection plans for the public leaderboard and two to four weeks of future private data, ranked experiments, scaling directions and decision criteria. It is a proposed research plan; its new mechanisms have not been tested.

For a single-file overview, open the [interactive results dashboard](research/unified/validation/RESULTS_DASHBOARD_20261012.html). It includes the public-score climb, protocol-separated local/public plots, recorded local confidence intervals, 45 preserved public records with three missing historic portal events called out, and searchable experiment and idea ledgers.

The [first-principles review proposal](research/unified/reviews/FIRST_PRINCIPLES_REVIEW_PLAN_20261012.md), merged from `claude/stoic-euler-70wff3`, challenges the campaign's allocation and proposes compact ensembles, averaging and cross-fitted teachers. The [experiment assessment](research/unified/ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md) checks its evidence, identifies useful comparisons and explains which proposed policy changes are not adopted.

The current competition status remains in [specs/current-state.md](specs/current-state.md). The user explicitly resumed experiments in the [October 13 strategy campaign](research/campaign_2026-10-13/strategy_execution_r1/README.md), superseding the historical [October 12 stop](research/campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json). CPU/RAM protections and the prohibition on new paid capacity remain in force.

## Find source evidence

- Find uploaded models in the [private Hugging Face archive guide](research/unified/models/huggingface/README.md), including short run descriptions, source checkpoint paths, research IDs and verified upload receipts.
- Search hypothesis IDs in [hypotheses.jsonl](research/hypotheses.jsonl), experiment IDs in [experiments.jsonl](research/experiments.jsonl), and decisions in the [bandit tree](research/bandit_tree/generated/INDEX.md).
- Open a domain README under `research/unified/` for a curated route to the relevant synthesis and receipts.
- Every useful research item must have a link and explanatory entry in its corresponding unified folder's README, including sources and artifacts stored outside the library; see [the indexing requirements](AGENTS.md#research-note-organization).
- Treat append-only ledgers, frozen experiment receipts, and official grade artifacts as evidence. Domain pages summarize and link to them; they do not replace them.
- Use `research/unified/reviews/` for critical reviews and audit receipts, `research/unified/sources/` for imported competition writeups, and the ideas domain for generated proposals.
- The dated archive is a point-in-time record. The unified domain pages are the maintained navigation and synthesis layer if research is explicitly resumed.

## Search examples

```sh
# Find an idea or experiment by its stable identifier.
rg -n 'H-381|E-CLEAN-T3-F0-CONTROL-S4' research/hypotheses.jsonl research/experiments.jsonl

# Find notes about blending, quantization, or transfer.
rg -n -i 'blend|quantization|public transfer' research/unified

# Find all tracked candidate and validation notes.
rg -n -i '0\.6671|Q8|fold|validation' research/unified/{validation,submissions,inference}
```

## Canonical locations

The repository root contains no research compatibility aliases. Use the canonical [competition metric](research/unified/methods/COMPETITION_METRIC.md), [October 5 idea memo](research/unified/ideas/legacy/oct5_ideas.md), and [literature review](research/unified/reviews/legacy/review_2.txt). The old root writeups were sorted by type: imported reports are under `research/unified/sources/competition-writeups/`, generated ideas under `research/unified/ideas/legacy/`, internal reviews under `research/unified/reviews/legacy/`, and controller commentary under `research/unified/operations/legacy/`. The [WunderFunder source audit](research/unified/reviews/legacy/WunderFunder_audit_evidence.md) is filed with the historical reviews.

## Latest authority: strategy research resumed

[Current QAT research plan](research/campaign_2026-10-13/qat_renewable_r1/PLAN.md) implements the user's new quantization-aware training/compression work using local GPU and all renewable Kaggle GPU/TPU allowance, with no campaign GPU-hour ceiling. It specifies exact W4/W8 native export, matched controls, activation/learned-scale extensions and broader endpoint validation; implementation is starting, with no new QAT gain claimed. [Campaign and live progress](research/campaign_2026-10-13/qat_renewable_r1/README.md).

[October13 execution campaign](research/campaign_2026-10-13/strategy_execution_r1/README.md) supersedes the historical experimental stop following the user's approval to implement the review and use all Kaggle allowance/local GPU capacity. It contains the plan, coverage and frontier; existing CPU/RAM and no-paid-spend constraints remain.
