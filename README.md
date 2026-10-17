# Wunder Connectome Challenge 2

This repository contains the competition starter solution, research code and
the evidence used to develop and validate submissions.

## Start here

- [Competition quick start](docs/quick_start.md) and [data format](docs/data_overview.md)
- [Submission contract](docs/submission_guide.md) and [official rules](docs/rules.md)
- [Current project state](specs/current-state.md) and [research index](RESEARCH.md)
- [Documentation index](docs/README.md), [architecture](docs/ARCHITECTURE.md),
  and [repository map](docs/REPOSITORY_MAP.md)

## Challenge at a glance

The model processes ordered sequences of 20,000 rows and predicts two targets,
t0 and t1. It must process every row causally and reset its recurrent state
when seq_ix changes. The supplied data has 112 float32 input features; steps
0–98 are warm-up. The scorer uses the pooled, clipped weighted-Pearson metric
described in [the data overview](docs/data_overview.md) and implemented in
[src/wnn/gmetric.py](src/wnn/gmetric.py).

The submission environment is CPU-only, uses Python 3.11, and limits the ZIP to
20 MB and total scoring time to 60 minutes. These serving limits are distinct
from research requirements; see the [submission guide](docs/submission_guide.md)
and [research objective](specs/research-objective.md).

## Current status

The 17 October request resumes research analysis, literature review, descriptive
EDA and bug repair; model fitting remains paused. The
[research reset](research/campaign_2026-10-17/research_reset_r1/README.md) contains
deployed prediction/cache fixes, an existing-checkpoint replay, a full event
census and 25 reviewed proposals. None is a newly trained model. The
[official leaderboard](https://wundernn.io/connectome/leaderboard) showed the
team at public WP 0.66832, rank 22 with 97 team submissions, when read on 17 October 2026. That
dated public snapshot is not a local validation result or a private-leaderboard
score. See [current state](specs/current-state.md) for the evidence limits and
the preserved history.

## Development

The repository pins Python 3.13 for research development; packaged submission
code must remain compatible with the scorer's Python 3.11 runtime. Setup and
routine checks are in [Development](docs/DEVELOPMENT.md). The modular training
components and their contracts are documented in
[src/wnn/training/README.md](src/wnn/training/README.md).

Keep scientific results tied to source, data membership, ancestry and exact
evaluation receipts. A code draft, static check or prepared package is not a
measured model result. Large private data and credentials do not belong in
public notes; follow the repository's provenance and storage instructions.
