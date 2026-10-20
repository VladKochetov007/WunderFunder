# F0 seed-1001 CONTROL recipe binding

This folder holds one local recipe binding for the canonical source's first full CONTROL fit. The complete proposed argument list, source and input identities, fit schedule, resource limits, and pending gates are recorded in [CONTROL_RECIPE_BINDING_V1.json](CONTROL_RECIPE_BINDING_V1.json). [REMOTE_SUBMISSION_TEMPLATE.py](REMOTE_SUBMISSION_TEMPLATE.py) is the fail-closed source template. [REMOTE_SUBMISSION.py](REMOTE_SUBMISSION.py) binds the exact accepted profile and exact-stop bytes for independent source review; it does not launch anything by itself.

The arm uses the shared 467-input / 25-output child topology. `--representation-mode base` leaves the appended 256 columns at zero, and the WMSE objective leaves the trailing logvariance outputs unused. The child inherits the accepted F0 seed-1001 parent checkpoint and its preprocessing, consumes the accepted homogeneous T3 cache and shared fold-0 dictionary, runs six fixed epochs retaining epoch 5, saves epoch-boundary training state, and does not evaluate during fit.

The all-three profile was accepted and its successful unit was exactly stopped. The local acceptance SHA-256 is `9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045`; the exact-stop raw receipt SHA-256 is `a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce`. The body embeds both exact byte strings and validates their identities before any remote filesystem mutation. Independent source review and fresh admission gates are still required before a launch. The template checks the accepted profile file map against the preserved private `ADMISSION.json`, `LAUNCH.json`, and `unit.log`, and reads the `CHILD_PROFILE_INITIAL_STATE` proof from that log; it does not expect a separate initial-state file.

The binding makes no quality claim. Its expected update count is arithmetic from the frozen fit membership and batch/chunk dimensions; observed counters must come from the eventual canonical trainer run.

The [canonical CLI parse check](CONTROL_CLI_ROOT_CHECK.json) independently parsed the final body argv against immutable source commit `68ff6829`; the full-fit recipe flags passed. It did not call the trainer or read model/data payloads.


The separate [one-shot CONTROL startup observer](startup_health_v1/README.md) binds the fixed unit and run while leaving the invocation unresolved until root launches. It captures only the new CONTROL unit, its small durable receipts/log, own cgroup/RSS, selected GPU metadata, and stat-only run inventory; it does not query predecessor units, accept the fit, or read model/data payloads.
