# Runtime migration during the matched seed-4 control

The active c20261003_ctrl_s4_f1 training process imported its original trainer
before the loader patches. It continues in its original bounded GPU scope.
Its source fingerprints remain those captured at admission; they must not be
rewritten to imply the training used the new loader code.

Prediction is a later separate Python invocation. It will import the updated
Sequences.batch_tensor_bytes and budgeted prediction loader. Batch size16,
tensor production, order, network computation, checkpoint selection and export
alignment are unchanged. Synthetic zero/one-worker collation/order checks and
schema/RNG checks passed. The checkpoint hash and prediction hash remain in the
export sidecar. Loader-only migration is recorded here because this current job
straddles the infrastructure update; future jobs fingerprint the updated code
and include family/environment provenance from admission.

The watcher migration initially reclassified 18 archived historical failed jobs,
causing the waiting scorer to exit before any scoring ran. The first inline
acknowledgement used stdin through the scope runner and executed no code. The
controller corrected this with a saved script, verified job timestamps older
than 15:30 Kyiv, acknowledged only those incident IDs plus the diagnosed waiting
scorer exit, and preserved the original barrier/logs. No training was stopped,
no completed prediction was overwritten and no scientific result was dropped.
The waiting scorer now waits through resource pauses and was restarted.

Review evidence: resource_review.txt preserves the initial independent FAIL;
runtime_fix_report.txt/runtime_loader_report.txt describe the repaired runtime;
runtime_fix_tests.log (20 tests), runtime_fix_loader_tests.log (7 tests), and
watch_fix_results.json (7 checks) preserve actual validation. Archive validation
is recorded separately. The runtime maintenance pause was retained in logs,
then released after effective nested slice limits and loader checks passed.
