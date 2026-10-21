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

## Prepared post-fit execution

[The separate delivery plan](H512_POSTFIT_DELIVERY_PLAN_R1.json) records exact
commands for terminal acceptance, source-manifest materialization, CPU parameter
averaging, two FP32 exports and the frozen score. It admits no additional fit.

- [Live command binding](FIT_COMMAND_BINDING_R1.json) captures the actual running
  Python process argv, exactly equal to the frozen recipe, and all 191 resolved
  arguments. This prevents accepting a run with optimizer or schedule drift.
- [Terminal receipt producer](record_h512_fit_terminal.py) requires the exact
  user-manager invocation to finish successfully, six completed epochs, fixed
  CP5, complete argument equality, source identity and checkpoint hashes.
- [Manifest template](SWA345_SOURCE_MANIFEST_TEMPLATE_R1.json) and
  [materializer](materialize_h512_swa_manifest.py) fill actual hashes only after
  terminal acceptance, with create-only output.
- [SWA derivation](derive_h512_swa345.py) and its
  [input validator](h512_swa345_validation.py) reuse the canonical arithmetic
  helper, require identical preprocessing and preserve cold-fold ancestry.
  Embedded source checkpoint hashes are removed before writing derived bytes.

These scripts are prepared and statically checked. They have not accepted this
still-running fit, derived weights or produced its quality score. The optional
production recipe is one two-epoch all-data continuation, independently admitted
only if quality and actual remaining compute justify it. Direct packaging of a
supported fold-trained endpoint is also eligible; another full-data fit is not
a prerequisite for delivery.

## Latest runtime and automated handoff

[Runtime amendment R2](THERMAL_BUDGET_AMENDMENT_R2.json) replaces only the9hceiling with12hafter sustained throughput declined to1.9sequences/s. Previous completedGPUjob time9,158.249398seconds plus12hfit,25minpostfit and90secondsstopgrace total53,948.249398seconds, below54,000. No new fit is admitted.

The [finite seven-step sequence](POSTFIT_SEQUENCE_R1.json) runs terminal acceptance, source-manifest materialization, fixedSWAderivation, twoFP32exports, [finite artifact audit](audit_h512_exports.py), and official frozen grading. The [supervisor](run_postfit_when_ready.py) waits for the exact invocation and checks source pins, new outputs, resource limits and cumulative budget. [Execution admission](POSTFIT_EXECUTION_ADMISSION_R1.json) records its conditional status; it admits no final fit or submission. The [service query check](SERVICE_QUERY_CHECK_R1.json) exercised actual user-manager properties without accepting a fit. [Grades](grades/README.md) remain absent until real execution.

The separate [complete native endpoint protocol](../tl3_h512_endpoint_screen_r1/README.md) comparesCP5/SWA345replacements to originalTL3. Its unknown legacy ancestry is explicit; it is not cleanCV.

[Independent static review](POSTFIT_AND_NATIVE_STATIC_REVIEW_R1.json) passed after concrete producer/path/hash corrections; a second canonicalCLI review found no remaining operational mismatch. [Actual watcher start](POSTFIT_WATCHER_START_R1.json) records the live stdlib supervisor, unit `wnn-h512-postfit-watcher-r1-20261021.service`, invocationeff981027d664a728e18b8e0e582471a. Admission is conditional on this exact fit succeeding and all pins/resource/budget/absent-output gates passing. This is a running waiter, not evidence of any completed postfit step.

### First epoch completed — October21

[Epoch0progress](EPOCH0_PROGRESS_R1.json) records the actual first full pass:6,549.8seconds,14,920/14,920completedupdates, zero AMPskips/nonfinitegradientsteps, mean pre-update windowobjective1.3793423, median preclip gradientnorm1.48872 andcliprate86.37%. These are training diagnostics, not a validation score. The fit continues into epoch1of zero-based0–5; the fixedCP5recipe and withheldevaluation remain unchanged.

## Completed fit and page-alignment recovery, October21

The six-epoch fit completed successfully at13:29:06Kyiv. [Terminal acceptance](FIT_TERMINAL_ACCEPTANCE_R1.json) verifies the original frozen recipe and actual artifacts; [SWA source manifest](SWA345_SOURCE_MANIFEST_R1.json) records the three-checkpoint derivative. No validation result yet.

The original postfit launcher stopped before GPU work: Linux rounded the configured5,000,000,000byte cgroup cap down to4,999,999,488bytes. Its exact equality guard incorrectly rejected that effective limit. [Recovery amendment](POSTFIT_PAGE_ALIGNMENT_RECOVERY_R2.json) preserves the cause, original/recovery hashes and cumulative budget. The separate [recovery launcher](run_postfit_page_aligned_recovery_r2.py) checks the exact page-rounded limit and retains all original scientific pins, output guards and budget checks. The original source and failed logs remain unchanged.

Recovery unit `wnn-h512-postfit-recovery-r2-20261021.service` launched the original bounded postfit sequence. At15:21Kyiv, checkpoint acceptance/SWA derivation completed and CP5 prediction export was running, using about3GBRAM without OOM. [Recovery log](postfit_recovery_r2.log) and [sequence log](postfit_r1.log) are mutable operational records, not quality evidence.

## Correction: matched B300 endpoint successfully reproduced

The [matched-cell audit](MATCHED_B300_SCORE_REPLICATION_R1.json) extracts the exact historical B300 PR25 wide F0/seed1003/CP5 score:0.6737706304503658. Our [full-precision local grade](grades/F0_H512_GRADE_POSTFIT_R1.json) is0.6737706291771555, a difference of−1.2732e−9. The earlier chat interpretation that the local recipe had failed to reproduce this endpoint was incorrect:0.68409111 was the historical four-fold mean, not the matching F0 score. No additional fit is needed to establish this observed score agreement.

This supports endpoint-score replication of the matching wide cell, not bitwise weight/prediction parity or an independently repeated width/depth treatment effect. The existing oldZ4/WMSE blend still scores0.6835218031 on the sameF0, versus0.6822209947 for the fixed oldZ4/H512 blend. Thus successful recipe replication has not yet delivered a stronger submission endpoint. The [complete seven-step report](POSTFIT_SEQUENCE_REPORT_R1.json), [export audit](EXPORT_AUDIT_R1.json), and [postfit service terminal](POSTFIT_SERVICE_TERMINAL_R1.json) preserve successful postfit execution; [mutable postfit log](postfit_r1.log) preserves actual timing. Native package grading remains incomplete after a separate coverage-check failure.

## User-authorized exploratory CP5 submission

The [release receipt](EXPLORATORY_CP5_RELEASE_20261021_R1.json) records explicit latest user authority to try this candidate despite lower fixed local blend scores. [Submission folder](../../../../submissions/b300_h512_cp5_tl3_20261021/README.md) contains the17.106MBZIP plus build, ELFABI and actual Python3.11 two-sequence/A-B-A check receipts. ZIP SHAff5d5bdade2d024e0c2bf98237024cef4569664bad0fa3c99d4686e9621f54e2. This is an exploratory deterministic three-member release, not a claimed new best.

The earlier full native replay reached its final coverage guard: expected file size omitted the128byteNPYheader. Its error handler deleted the cache; no full served-endpoint grade was produced. The delivered exact ZIP independently passed finite predictions/reset on40,000actual rows, with5.0576seconds callbacktime. This is not official full-runtime certification.
