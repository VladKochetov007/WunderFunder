# CONTROL post-exit restricted verifier V1 — metadata failure

One actual root submission of final reviewed65488619 body passed fresh gates and launched the CPU-only support at09:12:53UTC. [Root preparation](ROOT_ADMISSION_PREPARATION.json), [exact body](REMOTE_SUBMISSION.py), [launch transport](TRANSPORT.launch.json), [raw response](STDOUT.launch.raw.json) and exact [B admission](VERIFIER_ADMISSION.json)/[launch](VERIFIER_LAUNCH.json) bind it.

[Actual terminal capture](capture/README.md) confirms exit1 at09:12:58UTC. [Failure](FAILURE.json) records a checker expectation for absent LAUNCH.body_sha256, while original root acceptance binds the body and exact launch bytes. No verification PASS or CONTROL integrity acceptance exists. Preserve this failed unit/source/output/receipts; never stop/restart/retry it. Native CONTROL weights/training are unchanged and exited; a separately reviewed V2 will repair only that metadata binding and retain full loader/numeric/resource checks.

[Independent failure review](../../../preparation/CHILD_CONTROL_TERMINAL_VERIFIER_FAILURE_REVIEW_V1.json) and [actual launch audit](../../../preparation/CHILD_CONTROL_TERMINAL_VERIFIER_ACTUAL_LAUNCH_REVIEW_V1.json) qualify the metadata boundary only.
