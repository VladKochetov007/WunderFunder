# Further experiment code

The following workstreams prepare explicit experiments, rather than launching
fits or estimating a public score. Public anchor remains 0.6660; 0.7000 remains
the research target.

| Workstream | Code and protocol | Required next evidence |
|---|---|---|
| E493/E494 activation and label reanchor | [Materializer](activation_transfer/materialize_specs.py), [activation protocol](activation_transfer/DESIGN.md), [reanchor protocol](supervision/DESIGN.md) | Candidate resource profiles, then matched parent/head fits and held-component comparisons |
| E495 update cadence and gradient credit | [Materializer](cadence/materialize_specs.py), [protocol](cadence/DESIGN.md) | Each update/detach shape must pass its own resource probe; three fixed seed/fold blocks |
| E496 supervision controls | [Materializer](supervision_frontier/materialize_specs.py), [protocol](supervision_frontier/DESIGN.md) | Fixed alpha/horizon cells, direction auxiliary, and genuine fitting-only denominator calibration |
| E497 residual memory | [Workstream](residual_memory/) | Prototype correctness and resource evidence before training integration or native deployment |
| E499 clipping diagnostic | [Diagnostic](activation_transfer/e489_clipping_diagnostic/diagnose.py), [protocol](activation_transfer/e489_clipping_diagnostic/DESIGN.md) | Separate non-scoring investigation of the preserved E489 failed prerequisite; cannot authorize E489 fits |
| E492 raw-count reproduction | [Independent recount](event_semantics/reproduce_counts.py) | Compare raw fit-only counts against existing saved EDA tables |

## Inventory prepared jobs

Run from the repository root. This command reads metadata and writes the queue;
it neither admits jobs nor reads large data arrays into RAM.

```bash
PYTHONPATH=src WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G \
  scripts/limited .venv/bin/python scripts/night_experiment_queue.py \
  research/campaign_2026-10-05/continuation_night/activation_transfer \
  research/campaign_2026-10-05/continuation_night/supervision \
  research/campaign_2026-10-05/continuation_night/cadence \
  research/campaign_2026-10-05/continuation_night/supervision_frontier \
  research/campaign_2026-10-05/continuation_night/event_semantics \
  --output research/campaign_2026-10-05/continuation_night/prepared_queue.json
```

[Queue checker](../../../scripts/night_experiment_queue.py) validates draft
schemas, records actual draft/tool hashes, inventories missing source and data,
separates external dependencies from predecessor jobs, detects conflicting
output producers and cycles, and reports local shape limits and GPU reservation
ceilings. Its internal schema placeholders are never written as executable
sealed specifications. Existing outputs block reuse; use a fresh attempt/output
identity instead of overwriting scientific evidence.

Calibration-dependent `.template.json` files remain unresolved until a real
calibration report passes source, input, split and fitting-membership checks.
They are excluded from the runnable-draft queue. Runtime feasibility, scientific
review and provenance gates still apply even when every input exists.

## Admit a reviewed job

Use the continuation runner at `continuation_night/runner`, which carries the
original budget. Seal real bytes before admission; the inventory cannot replace
sealing. Admit one GPU worker at a time. Full parent/head jobs depend on their
own resource probes and separately reserved confirmation work. The runner checks
live headroom and the October 6, 10:00 Kyiv local launch cutoff. Kaggle has its
own measured quota and headroom; local caps do not become cloud caps.

Record every admitted attempt and terminal outcome in the root MD/JSONL ledgers,
with measured peaks, new OOM counters, failed gates and teardown. Save future
EDA tables and figures inside each workstream's `eda/` directory. The live
marimo continuation notebook displays the prepared queue and current saved EDA.

## Prepared-code validation

Combined tests passed **36/36** using one CPU, 2 GiB RAM and no GPU.
[Receipt](checks/preparation_tests.xml), [summary](checks/PREPARATION_RESULT.json).
The inventory contains **201 schema-valid drafts**, with no dependency cycles or
ambiguous producers, plus **8 unresolved calibration templates**. Independent
scientific review and measured resource admission are recorded separately; these
counts are prepared jobs, not completed experiments. E497 remains a prototype
without trainer or native export integration.

Independent source-review limits: [E495](admission/E495_INDEPENDENT_SOURCE_REVIEW.md) is preparation GO only; [E497](admission/E497_INDEPENDENT_SOURCE_REVIEW.md) is isolated prototype GO only; [E496](admission/E496_INDEPENDENT_SOURCE_REVIEW.md) blocks fixed/shuffled denominator resolution pending a measured-mass fix. See [prioritized untested ideas](NEXT_UNTESTED_IDEAS.md).
