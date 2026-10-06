# W384 v3r1: bind all fresh fit outputs

The v3 packet was stopped at independent review before launch. Its artifact
manifest glob used the earlier `c20261005night` identity, so it could omit the
newly checked v3 checkpoint and the current fit's provenance and history. The
reviewed correction uses the exact `c20261006night_w384_adamw_s3_f1_serialv3r1_fit`
directory, and assigns the notebook output directory and probe run a fresh
`v3r1` identity as well.

Before any probe starts, the phase runner rejects a pre-existing probe or fit
run directory for the new identity. After the fit and prediction export, it
requires `model_ep3.pt`, `model.pt`, `provenance.json`, `history.json`, and
`training_state.pt` in the exact run directory. It hashes every regular file in
that directory with stable before/after stat identity, then adds the complete
validation prediction array and its JSON provenance. Missing any required file,
unexpected nested/non-file output, or a file changing while hashed stops the
run before completion is recorded.

The historical scientific argv remains unchanged. Only output identities and
artifact-manifest validation change. The correction is covered by a CPU test
that creates a complete synthetic run directory, checks every required file is
manifested, and removes each required file in turn to verify fail-closed
behavior. The pinned W384 loader preflight remains 36/36 CPU tests; the width
transport suite now includes this artifact-binding check.

The immutable packet revision is
`wnn-night-w384-serial-20261006-v3r1`, still provider recovery attempt 2 of 2.
It is prepared for independent source review only. No push, GPU preflight, fit,
prediction, resource probe, or score was launched for v3r1.
