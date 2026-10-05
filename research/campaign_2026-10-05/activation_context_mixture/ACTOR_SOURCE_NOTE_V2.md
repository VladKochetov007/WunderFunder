# E484 actor source v2 receipt

This version preserves the reviewed v1 actor and fixes one device-only
comparison defect exposed by the first actual CUDA probe. After a model is
moved to CUDA, inherited initial-state tensors are now copied to canonical
contiguous CPU storage before exact dtype, shape, and byte comparison. The
comparison remains bit-exact; it does not use a tolerance or change model
values.

The v2 tests add canonical tensor-identity checks, including different
allocations and the signed-zero byte distinction. The remaining actor and
optimizer recipe are inherited unchanged from E484 v1. Seven synthetic tests,
Ruff, and bytecode compilation pass. A bounded one-update-per-arm fit-only CPU
preflight at `preflight_cpu_v5/fit_manifest.json` passes for all twelve cells
and confirms that test-role rows were not loaded.

The failed E484 v1 probe and exact diagnosis are preserved in
`GPU_PROBE_FAILURE_V1.md`; the v1 actor, probe, tests, draft specs, sealed
probe spec, attempt, and runner log are untouched. V2 is a new unsealed source
candidate. It has not run on CUDA and has not been admitted for full fits.
