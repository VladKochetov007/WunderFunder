# R10 R5 external-mount startup repair

R9 reached the bootstrap failure recorded in `R10_R9_FAILURE_BINDING_R1.json`. Its pipeline recorded zero export stages and zero fit workers. The R5 manifest contained 13 external mounts; R7's local `declared_paths` set included only archive members and spec paths, so the existing mount guard rejected the first authorized external key, `artifacts/runs/c20260929_prop_f0/model.pt`.

R10 changes that startup boundary only. Before mounting, it requires the actual manifest's complete `external_mounts` mapping to equal the map embedded in the R7 bindings. It rejects overlap with archive/spec paths and adds only those exact verified keys to the declared set passed to the unchanged `mount_verified_inputs` function. Its absolute-path and `..` checks remain in place. The export specs, checkpoint selection, data identities, worker arguments, resource limits and five-export plan are unchanged.

The fixture uses the exact R5 package whose bytes match R7's sealed hashes: `package_r9`, manifest SHA-256 `5fc7423e…c689` (31,114 bytes), archive SHA-256 `88641b3c…fa549` (246,396 bytes). The separate `package_r6` directory has different bytes and is not the producer version bound by R7. R10 carries copies of the exact `package_r9` manifest and archive under `fixtures/r5_bundle_package_r9/`; these are test fixtures and are not kernel upload files.

`test_r10_transport.py` first checks exact external-map equality, missing/extra/changed entries, overlap rejection, unsafe absolute/traversal paths, pending-release refusal and R10 transport-target binding. Its full-startup fixture then runs the actual embedded R9 worker body with the narrow R10 callsite repair. It verifies and extracts the actual R5 archive, validates and mounts the actual R9 checkpoint manifest and six staged checkpoint/provenance files, resolves all 13 external mappings to their current MAIN source paths, imports the archived R5 driver and validates the five real export specs. It intercepts the driver's first `run_stage` call before the worker subprocess. The fixture records that boundary, creates no prediction files, and starts no GPU worker. A synthetic admitted receipt exists only in the temporary fixture to reach that startup boundary; it is not a ROOT authorization.

The R5 input files are linked into temporary mount paths from their actual MAIN source locations. Large parquet/cache payloads are not read or rehashed by the fixture. The package's existing versioned-source and R7 binding checks remain unchanged; real provider readiness and the R10 ROOT release are still required before dispatch.

Run the bounded local consumer fixture from this worktree:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=650000000 WNN_RUNTIME_MAX=180s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/usr/bin/python3 research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_h390_r10_external_map_repair_20261012_r1/test_r10_transport.py
```

The embedded root release remains pending. This packet has not been uploaded or dispatched; independent review and new ROOT authorization for the R10 transport target remain open. R9's failure is infrastructure evidence, not a negative result for the teacher recipe.

