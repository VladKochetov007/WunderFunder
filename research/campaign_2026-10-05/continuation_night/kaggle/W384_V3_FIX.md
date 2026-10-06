# W384 transport recovery v3: preflight fixture compatibility

The v2 worker stopped in CUDA preflight before any scientific probe or fit. Its
24 failures all came from legacy integration fixtures that create eight-row
Parquet row groups and patch `wnn.data.SEQUENCE_LENGTH` to eight. The transported
`BoundedSequenceFile` had imported the default 20,000-row constant by value, so
its strict metadata check retained a stale value. The failures were validation
compatibility errors; the attempt recorded zero cgroup OOM events and no
remaining child processes. They are not resource-censor or model evidence.

The private loader fix resolves `wnn.data.SEQUENCE_LENGTH` when a file is opened,
matching the baseline reader's runtime module contract. It still checks the
Parquet row-group length before allocation, bounds each decoded batch by that
same expected length, requires the final decoded offset to match, and rejects a
row group containing more than one sequence id. Selected target and mask
columns are decoded in the same row batches and share those checked row
boundaries. The transport derivation also hashes the manifest-bound production
`src/wnn/data.py` and fails unless its top-level contract is exactly
`SEQUENCE_LENGTH = 20_000`; the shorter fixture length is therefore supplied
only by the existing integration test's explicit monkeypatch.

The derived trainer continues to change only the loader import and the existing
post-copy auxiliary-mapping release. No science arguments, split, sampler,
model, optimizer, checkpoint, or prediction code changed. The W384 v3 packet
keeps the prior v2 scientific argv and source/input manifest, changes the
reviewed transport source and packet identity, and gives outputs a fresh v3
artifact identity to avoid any collision with prior attempts.

Validation ran in bounded CPU scopes with CUDA hidden:

- The staged legacy integration and training-state suite passed 36/36 tests,
  using one CPU and an 8 GiB memory ceiling. The tests ran in an isolated
  temporary source tree containing the derived trainer and private loader.
- The private width transport suite passed 16/16 tests under one CPU and a
  2 GiB memory ceiling; it covers full 20,000-row parity, eight-row fixture
  parity, optional target/mask combinations, mixed-sequence rejection,
  production-length pinning, and watchdog behavior.
- Ruff passed for the changed loader, derivation, transport test, temp-tree
  preflight runner, phase launcher, and packet builder.

The first W384 GPU v3 packet received an independent NO-GO: `phase.py` verifies
the fresh model but `artifact_manifest` uses a stale run-name glob and omits
the checkpoint, provenance and history. A corrected immutable packet revision
is being prepared for another review. No provider push, GPU preflight, resource
probe, fit, or score was launched for v3. The v2 packet and its crash
diagnostics remain unchanged; any provider action still needs critic GO and a
root reservation.
