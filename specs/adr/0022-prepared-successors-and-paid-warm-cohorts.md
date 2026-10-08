# ADR0022: Prepare successors before accelerator completion

Status: accepted, 2026-10-08. Extends ADR0010, ADR0011, ADR0014 and ADR0015;
their budget, ownership, validity and resource boundaries remain binding.

## Basis

The user requires powerful compute to be utilized for most available useful time,
successors prepared before predecessors finish, and a persistent experiment queue.
Repeated data loading and paid idle sessions waste competition time and credit.
The recent local-heavy delivery record demonstrates that having ideas or source
builders alone does not keep cloud accelerators supplied.

## Decision

Prefer capable Kaggle GPUs/TPUs and campaign-owned Vast accelerators for suitable
training. Maintain a rolling queue for each compatible execution lane, with a
prepared successor and an independent fallback whenever possible. Generate or
prepare more recipes before the queue runs dry. Preparation and independent
review overlap the current fit; result collection/grading should not hold up an
otherwise independent, admitted successor.

Distinguish proposed, source-prepared, executable and admitted queue entries.
An executable entry binds source, inputs, preprocessing/training membership,
exact command, endpoint, resource/runtime bounds and output/collection plan.
An admitted entry additionally satisfies that lane's current validity, quota,
cost and ownership conditions. A proposal is not a runnable backup. Keep an owner
and blocking dependency visible for entries that are not ready.

On a valid terminal transition, start the compatible admitted successor without
waiting for a user message. An observation timeout alone is not a transition.
On an execution failure, preserve evidence, diagnose it and use an independent
ready arm while repairing the failed delivery. Reuse completed checkpoints rather
than refitting to repair export or grading code. Do not let one broken lineage
block all available lanes.

Use finite warmed cohorts: stage and verify shared data once, reuse compatible
loaded/memmapped data and compiled shapes, and run several useful recipes in
sequence or safely in parallel. Preserve each recipe's preprocessing, membership,
model/optimizer/RNG independence and provenance. Record startup/loading,
compilation, fitting, export and collection separately. Measure completed useful
comparisons per hour and end-to-end cost, not only GPU occupancy or fit counts.

Before renting Vast, have a bounded executable cohort and successor queue, plus
hard spending/lifetime limits and teardown ownership. Do not pay merely to retain
loaded data while waiting for ideas or code. Prepare next jobs while the current
job runs; if the finite useful queue is exhausted and no admitted successor can
start, collect artifacts and destroy only our receipt-proven instance. Existing
credit, the conservative $20 cumulative all-in ceiling, no top-up and ambiguous
creation safeguards remain binding. Do not blindly create another instance to
work around uncertain ownership.

Exploratory screens, useful controls, teacher fits and declared interactions can
fill a lane without waiting for a predicted gain. Avoid duplicate completed work
or irrelevant compute solely to make utilization appear high. Never remove
resource headroom or validity checks to satisfy occupancy.

## Consequences and review

The controller owns accelerator supply as well as scientific allocation. At each
completion/failure and the two-hour allocation review, inspect actual handles,
queue readiness, owners and phase costs. Explain idle time from measurements and
concrete blockers; a historical access failure is not a permanent restriction.
Scale reviewed warm cohorts within live quota/credit when it reduces turnaround.
