# Observed PR15 recovery and admission evidence

The [first-wave outcome receipt](receipts/pr15-firstwave-outcomes-v1.json) contains all 16 recovered original full-fit endpoints, scored without retraining. Four paired cells per arm use one common seed; these exposed-fold results do not close the CP30 comparisons.

The [sealed queue manifest](runtime/pr15-host-v2/QUEUE_EXECUTION_MANIFEST.json) pins corrected delivery code and unchanged scientific dependencies. [Server tests](state/pr15-v2-acceptance-v1/tests.log) passed 12 checks, the [admission regression test](state/pr15-v2-acceptance-v1/queue_tests.log) passed, and the [real artifact preflight](state/pr15-v2-acceptance-v1/preflight.log) verified the original 16 models and admitted exactly 104 pending cells. `receipts.tgz` retains the transport snapshot; live canonical progress remains on the server.

See the [parent queue record](../README.md) for source bindings, original failure preservation, fixed recipes, inherited resource/deadline limits and observed launch status.
