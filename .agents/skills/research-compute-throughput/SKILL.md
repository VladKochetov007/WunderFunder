---
name: research-compute-throughput
description: Improve completed-experiment throughput across local, Kaggle and owned paid accelerators using phase measurements, warmed cohorts and bounded scaleout. Use for idle lanes, repeated data loading or bottlenecked research delivery, not isolated kernel benchmarking.
---

# Research compute throughput

Start from actual live handles, latest user allocation, quotas, creation receipts
and ready recipes. Read [the pipeline](../../../specs/research-pipeline.md),
[execution](../../../specs/experiment-execution.md) and the relevant provider
instructions in AGENTS.md. This skill grants no extra spending, data access or
permission to touch existing rented machines.

## Measure delivery rather than occupancy

For each cohort record provider queue/startup, staging, verification, compilation,
profile, fit, export, evaluation and collection times. Record processed sequences
and completed optimizer updates separately from attempts/skips. GPU memory usage,
RUNNING status and theoretical TFLOPS are not useful throughput measurements.

Count an idea screen only when its declared endpoint/comparator has been evaluated.
Report distinct mechanism comparisons separately from seeds, profiles, failed
startup and proposals. A fresh architecture may deserve longer full training even
when it reduces screens/hour. Preserve a full-budget finalist lane.

For idle capacity, identify its next useful admitted action or preparation owner.
Overlap successor recipe preparation and independent checks with live fitting.
Keep healthy useful fits running. A terminal failure belongs to an owner who
collects diagnostics and prepares an identified correction; a monitor alone does
not deliver results. A transient observation failure is not a terminal job.

## Amortize unchanged inputs

Prefer one finite cohort that verifies immutable inputs, profiles, fits and exports
within the same worker. Reuse existing dataset/cache/streaming tools. Exact path,
hash and membership bindings must survive the handoff; changed stat/path identity
requires the original content-verification fallback. Scope counters to the actual
phase and file: legitimate auxiliary-cache fallbacks are different from unverified
raw-data reuse. Never weaken verification to improve a timing.

Keep data/compilation caches warm where measured memory headroom permits. Memmap or
stream large sources; a parquet file's compressed size is not its RAM footprint.
Share immutable caches, not model/optimizer/RNG/sequence state. Repeated `main()`
calls can retain environment flags and module globals; reset/isolate recipe state
and verify observed features before allowing mixed-recipe warm execution.

Use subprocess workers when that gives reliable state isolation while preserving
the OS file cache. Reprofile materially changed shape/context/batch/precision;
unchanged source input verification need not be repeated merely for a new run name.
Account for export and delivered-artifact verification in the cohort's lifetime.

## Scale from the bottleneck and ready work

Estimate total accelerator work for the next useful comparisons from actual-device
profiles, including control reuse only when its frozen recipe/membership truly
matches. Estimate queue/staging/export costs separately. If the ready queue exceeds
capacity and faster evidence changes useful decisions, quote bounded additional
capacity within current authority. A multi-GPU host can amortize one data transfer
across independent ideas; several hosts can suffer repeated transfer costs. Choose
using effective delivered comparisons per total cost/time, measured reliability,
CPU/RAM/storage/network and teardown allowance, not TFLOPS alone.

Use separate provider headroom: local ceilings do not define cloud capacity.
Kaggle production TPU work requires actual-device numerical/gradient/reload and
representative objective/reduction checks. A synthetic pass is not model quality
or steady-state throughput. Recurrent, convolutional and scan-based architectures
can have different hardware tradeoffs; include eventual student/serving quality.

On paid compute, touch only instances with the campaign's durable creation proof.
Bind quote, input sizes, worker lifetime, upload allowance, collection and cleanup
deadline before creation. Reconcile ambiguous creation without duplicating it.
Keep credentials inside the API client. No bulk cleanup or unrelated instance
operations. Existing spending authorization does not authorize subscriptions or
credit top-ups.

Leave the result in a phase/resource/cost receipt and update the allocation. Scale
again from measured warm cycles and useful queue depth; stop spending on a failed
path when a diagnosed alternative offers better value, without calling that a
scientific negative. Explicit user stop closes admission and drains healthy jobs.
