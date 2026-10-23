[Independent stop metadata audit](ACTUAL_STOP_METADATA_REVIEW.json) confirms original raw/transport/source/acceptance joins and successful-only stop; it performed no remote action or payload reads.

# Successful scorer: one exact owned stop

[EXACT_STOP.json](EXACT_STOP.json) equals the original [raw stdout](STDOUT.raw.json): 1,211 bytes/SHA7ae50e926fb66948a7a3c95ce227c4fc11c88d68e55f2483012275c662bb953e. [Transport](TRANSPORT.json) records one SSH0/empty-stderr action with durable raw capture before parsing. B captured13:29:44.567799 UTC; ROOT and B clock samples are separate. [ROOT score acceptance](../SCORE_ACCEPTANCE.json) was durable first at13:23:06.724383 UTC.

Only score-v3/inv9bd64d6759e8462bb40985b06a9a0ad6 was stopped after exact active/exited/PID0/code1/status0/success checks. It became not-found/inactive/dead/PID0/empty invocation, stop return0 and populated campaign descendants{} before/after. Failed score-v1/v2 were excluded permanently. Acceptance's pending-stop field remains frozen history; this later receipt supersedes it.

The [actual source](REMOTE.py), [source review](SOURCE_REVIEW.json) and [literal binding](AST_BINDING.json) join the exact acceptance, invocation and unit. [ROOT transport](ROOT_CAPTURE_STOP.py) and [binding](ROOT_TRANSPORT_AST_BINDING.json) preserve the conditional source. The initial [transport review](SOURCE_TRANSPORT_REVIEW.json) rejected a generic pre-capture pipe-memory claim; the separate [V2 review](SOURCE_TRANSPORT_REVIEW_V2.json) accepts ROOT's fixed trusted-source/time-bounded scope with remote hard bounds and post-capture limits. It does not claim generic hard local pipe-memory bounds. No replay of scorer, fits, predictions, models or data occurred.
