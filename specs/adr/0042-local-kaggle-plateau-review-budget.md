# ADR0042: bounded local/Kaggle plateau-review cycle

Status: accepted user direction, October 19, 2026.

The later same-day [ADR0043](0043-fast-screens-and-transformer-teacher-scale.md)
extends the fast-screen policy to all future drafts and adds the large B300
Transformer-teacher design priority. This ADR's local allocation/start stays in
force; frozen historical protocols remain unchanged until explicitly amended.

## Decision

Implement the [accepted plan](../plans/2026-10-20-plateau-review-local-kaggle.md)
and begin its experiment cycle October 20 Kyiv. Allocate 15 local GPU-hours plus
verified available Kaggle quota in parallel, preserving other-agent reservations.
No paid compute is authorized. Local limits are two aggregate CPU cores, one GPU
worker, 5,000,000,000 bytes aggregate task RAM, 5 GB VRAM, no swap, physical-disk
scratch and resource-aware admission before a job starts.

The user explicitly removed the 30-experiment requirement for this cycle and
requested faster local/Kaggle iteration. This supersedes ADR0040's 30 paired
cells per arm **for this campaign only**. Full-data fits are allowed. Existing
frozen B300/CP30 proposals retain their original recipes until separately amended.
Keep strict FP32, clean ancestry, official scoring, evidence preservation and
independent criticism. Exploratory screens do not become confirmation merely
because a second exposed fold or seed is added.

## Consequences

Start with matched equal-width i0-only, standard and i1-history cold GRUs. Admit
follow-ups by information and complete-endpoint utility; reserve budget for
prediction, evaluation and recovery. Never reinterpret a small-screen loss as a
universal negative or a local gain as a public-score forecast.

The October 18 stop remains preserved. This new scope allows preparation now
and fitting from October 20; unrelated campaign admission remains closed. It
does not create an automatic scheduler or authorize interfering with another
agent's Kaggle jobs. Actual run admission, quota and remaining local budget are
recorded in the [campaign](../../research/campaign_2026-10-20/plateau_review_r1/README.md).

The accepted plan and its frontier are required continuation context after
compaction. Archive imported suggestions separately and record amendments;
do not silently replace the agreed comparison as new reviews arrive.
