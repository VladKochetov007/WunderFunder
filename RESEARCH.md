# Research library

Start with the [unified research library](research/unified/README.md) for a topic-based guide, shared note format, and links to the dated archive. The [October 12, 2026 archive](research/unified/archive/2026-10-12/README.md) contains the split final handoff.

The current competition status remains in [specs/current-state.md](specs/current-state.md). Research admission is closed under the [stop receipt](research/campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json); this library refactor changes documentation only.

## Find source evidence

- Search hypothesis IDs in [hypotheses.jsonl](research/hypotheses.jsonl), experiment IDs in [experiments.jsonl](research/experiments.jsonl), and decisions in the [bandit tree](research/bandit_tree/generated/INDEX.md).
- Open a domain README under `research/unified/` for a curated route to the relevant synthesis and receipts.
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

## Legacy entry points

The older root entries `METRIC.md`, `oct5_ideas.md`, and `review_2.txt`, plus the `writeups/` and `reviews/` file paths, remain as compatibility symlinks to their canonical locations under `research/unified/`. The unified index also links to the [idea ledger](research/IDEA_LEDGER_20261011.md), [elephants review](research/ELEPHANTS_REVIEW_2026-10-05.md), [review file index](research/REVIEW_FILE_INDEX.md), [research methods](research/RESEARCH_METHODS.md), and the [October 12 handoff compatibility page](research/RESEARCH_COMPLETE_HANDOFF_20261012.md).
