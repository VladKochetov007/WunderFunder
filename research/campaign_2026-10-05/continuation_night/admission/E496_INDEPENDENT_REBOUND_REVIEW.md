# E-496 independent rebound review

Review date: 2026-10-06. This verdict is bound only to the frozen packet
identity below. It authorizes source-preparation handoff for controller review;
it does not authorize calibration, resource profiling, probes, fits, prediction,
scoring, or provider/GPU execution.

## Verdict

**Preparation GO** for the reviewed E-496 source packet. Both prior resolver
findings are fixed in the current code and have negative mutation coverage:

- The resolver recomputes the fit-batch reference mass, extrema, standard
  deviation, and zero count from the recorded per-update masses and verifies
  their count and canonical float64 little-endian SHA-256.
- The resolver requires and hashes each current teacher `model_ep3.pt`; it
  matches that byte hash to provenance and both prediction sidecars. Tests
  reject both a missing checkpoint and changed checkpoint content.

The packet expressly leaves all templates unsealed and `launch_authorized`
false. No calibration or other E-496 research run has occurred. The fixed and
shuffled arms remain unresolved until the fit-only calibration produces a
report that passes this resolver and the controller separately completes
resource admission.

## Frozen identity and verification

The handoff packet
`supervision_frontier/E496_HANDOFF_PACKET.json` has SHA-256
`f36897bdc0ff74191889c8b330a4d7a4df750c204af4c9cc15336e2861774008`.
The packet binds 55 source/design/test files, 139 generated JSON files, and four
verification artifacts. I recomputed every listed digest: all 198 entries
matched current bytes, with no missing or stale files. The packet records both
prior findings and fixes, preserves the frozen `scripts/train_rnn.py` hash, and
does not claim an overall E-496 experiment GO.

Key source identities:

| File | SHA-256 |
|---|---|
| `supervision_frontier/DESIGN.md` | `efd1686736f1d8ca129e2e276999ada0bf784d39b9c8fd392ec3ea7a0cc88777` |
| `supervision_frontier/materialize_specs.py` | `25e6473d77ece00a58e61b33df82ca45874ecd9974ad09739bf1a51d50f7504b` |
| `supervision_frontier/resolve_calibration.py` | `2547e929da8376713bba5224e39a92a3a152039508d0522a14f8927de8431eab` |
| `supervision_frontier/test_materialize_specs.py` | `2fee6e10d7f9fe37d0f2cd919d3db5df379e26139e6e0b8a365a0e37f500fbf3` |
| `scripts/train_rnn.py` | `8a29459e998f7f5d6a03b7a4b87c6f4b569002e65b8c87127c37138360909cbf` |
| `src/wnn/training_calibration.py` | `30923120c1bd9773556556ee0fbbab5d8db4a5f4232c48b81f11fa87a5e74e52` |

I independently ran the resolver and calibration unit suites together under
`scripts/limited` with one CPU, a 2-GiB memory cap, CUDA hidden, and plugin
autoload disabled: 23 passed. Ruff passed on the changed materializer,
resolver, tests, and calibration module. The packet also hash-binds the XML and
log receipts for its 16-test resolver suite and 23-test combined suite.

## Scope boundary

This note clears only the reviewed source-preparation identity. The current
dependency graph still lists missing calibration reports and downstream fit
artifacts; those dependencies are expected at this stage. Each actual run
requires its own current input/source closure, resource admission, and
controller authorization. The earlier NO-GO notes remain historical records
of the specific defects found and do not need to be rewritten.
