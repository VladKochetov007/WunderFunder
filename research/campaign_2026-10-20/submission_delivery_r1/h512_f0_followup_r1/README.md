# H512 local follow-up and production path

## Active fit

One cold H512×1 GRU fit started at 23:18:35 UTC on October 20 under
`wnn-h512-f0-fit-r1-20261021.service`. The [admission](FIT_ADMISSION_R1.json)
and [actual start](FIT_START_R1.json) record its source, bounds and invocation.
This uses F0, seed 1003, six epochs and fixed checkpoint epoch 5. There are no
new teacher fits. The proposed all-data final fit has separate admission and
remains conditional on results and the remaining budget.

The [exact recipe](H512_FIT_AND_PRODUCTION_PLAN_R1.json) adapts the PR25 wide
arm to execution source `68ff6829`. An independent source review found no core
flag or membership mismatch. This is an explicitly adapted implementation,
not numerical equivalence to the B300 producer. The prior evidence consists
of 12 width/depth pairs across four exposed folds and three seeds; the wider
arm won 9/12. This local seed/fold was also in that prior panel.

## Frozen comparison

The [score screen](H512_F0_SCORE_SCREEN_R1.json) fixes seven endpoints before
training outcomes: oldZ4, the retained WMSE continuation, H512 checkpoint 5,
the equal parameter mean of checkpoints 3/4/5, and three fixed 75/25 blends.
The averaged checkpoint adds no training fit and preserves identical
normalization buffers. It is an additional interaction endpoint, not part of
the original R01 replication. There is no coefficient or epoch search.

Score the same 468 exposed F0 groups with the official metric and 2,000 paired
component-bootstrap draws. These intervals condition on fitted models. This
one local fit does not establish independent replication across folds or seeds.
TL3 has unknown fitting ancestry and cannot be a clean CV comparator.

## Resources and delivery

The [fit log](fit_r1.log) records actual batches, loss, LR and epoch diagnostics.
The measured short profile projects about 4.7 hours for this fit and 4.9 hours
for all-data training; neither is a sustained-training measurement. Root owns
exports, official grading, cumulative 15-hour accounting and native packaging.
No quality result or new submission ZIP exists yet.

The [parent resource enforcement](../h512_replication_profile/AGGREGATE_LIMIT_ENFORCEMENT_R1.json)
now applies 5 GB aggregate task RAM, two CPU cores and zero task swap, replacing
a stale 20 GB runtime ceiling. The worker has one CPU and a 4.2 GB RAM ceiling.

## Sustained thermal limitation

The [early thermal snapshot](thermal_snapshot_r1.txt) reports 89°C, active
software thermal slowdown and SM clocks around 400 MHz. Initial cumulative
throughput declined from 4.8 to 3.4 sequences/s by batch 60. The short-profile
4.7-hour projection is therefore optimistic until sustained timing stabilizes.
Root requested optional physical cooling help and will reassess final-fit
admission from observed epoch timing. The job remains healthy; no power,
clock or fan settings were changed.

[The thermal budget amendment](THERMAL_BUDGET_AMENDMENT_R1.json) extends the healthy fold worker runtime ceiling from six to nine hours within the unchanged 15-hour allocation. Model updates and the fixed six-epoch endpoint are unchanged. At the ceiling, about 3.46 GPU-hours remain; final training and exports require separate admission against actual elapsed time. A shorter all-data continuation is a conditional alternative to the original cold six-epoch production proposal.

## Post-fit delivery amendment

The [post-fit plan](H512_POSTFIT_DELIVERY_PLAN_R1.json) records exact artifact
bindings and commands for the frozen seven-endpoint score screen. It derives
only the predeclared equal mean of epochs 3/4/5 through the canonical
`mean_state` helper, then exports CP5 and SWA345 predictions for the already
frozen F0 comparison. The [manifest template](SWA345_SOURCE_MANIFEST_TEMPLATE_R1.json),
[derivation CLI](derive_h512_swa345.py), and its [source/checkpoint validator](h512_swa345_validation.py)
are post-fit only; none fits or grades a model. The [terminal receipt producer](record_h512_fit_terminal.py)
checks the admitted service, trainer closure, log and checkpoint hashes before
writing a create-only receipt. Root reviews that receipt before derivation.

The same plan contains a conditional two-epoch full-data warm continuation
from H512 F0 CP5, with inherited preprocessing and all supplied valid labels.
It omits `--final-model-adaptation` because that mode currently accepts a
different fit-subset/all-data initializer contract. The continuation remains
`NOT_ADMITTED`; direct packaging of the accepted fold endpoint remains a
possible route if the remaining budget cannot support another fit. Either
production route is production-only, has no heldout claim, and must preserve
the F0 membership and TL3 ancestry limitations above.
