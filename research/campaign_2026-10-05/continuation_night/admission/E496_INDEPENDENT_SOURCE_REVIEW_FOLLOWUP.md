# E-496 independent follow-up: mass observations and teacher checkpoint identity

Review date: 2026-10-06. This follow-up records the review of the changed
calibration estimator/resolver identity after the first NO-GO note. It does not
replace that earlier finding or authorize any calibration, probe, fit, export,
score, or provider launch.

## Current verdict

The mass-summary fabrication finding from
`E496_INDEPENDENT_SOURCE_REVIEW.md` is fixed in the reviewed source: the
trainer now emits canonical little-endian float64 update masses with count and
SHA, and the resolver recomputes/checks the mean, extrema, standard deviation,
and zero count before writing a resolved spec. The new scalar-only mutation
test passed independently (1 CPU, 2 GiB, CUDA hidden; 1 passed in 0.17 s).

The fixed/shuffled denominator templates remain **NO-GO** due to the separate
teacher epoch-3 checkpoint binding gap below. The complete E-496 source set
does not yet receive an independent preparation GO.

## Reviewed changed source identities

| File | SHA-256 |
|---|---|
| `scripts/train_rnn.py` | `cf615a40aca822a3bcd131b34feafdc1c74a92ed8112c1519234845baeb06125` |
| `src/wnn/training_calibration.py` | `ac2f49bb0774d8ffd7f3efe1fefce55c1ff3b90beb71d6971d6962b785527682` |
| `supervision_frontier/resolve_calibration.py` | `38b1f26b982f707f44150c6ae067b8b886d768d946ffa46d2d47e6c4e44bf341` |
| `supervision_frontier/test_materialize_specs.py` | `5a3cf715b25196d03aa665cc749246b6e07290014a28d2707e8c35e571f3945c` |
| `supervision_frontier/materialize_specs.py` | `25e6473d77ece00a58e61b33df82ca45874ecd9974ad09739bf1a51d50f7504b` |
| `supervision_frontier/DESIGN.md` | `efd1686736f1d8ca129e2e276999ada0bf784d39b9c8fd392ec3ea7a0cc88777` |

The actor reported 21/21 bounded CPU tests and clean compile/Ruff. I
independently ran the new high-value mutation fixture against these source
files; it rejected a changed scalar with a self-consistent fabricated summary.
The current `E496_HANDOFF_PACKET.json` still contains the older source hashes
and therefore must be regenerated after the final resolver fix.

## Distinct blocker: missing actual epoch-3 teacher checkpoint binding

`_verify_teacher_and_cache_inputs()` hashes the current teacher `model.pt` and
checks it against `provenance.checkpoint_sha256["model.pt"]`. It then compares
prediction sidecars to the string in
`provenance.checkpoint_sha256["model_ep3.pt"]`, but never requires or hashes
the actual `artifacts/runs/{teacher}/model_ep3.pt` file. As a result, the
epoch-3 identity string and both sidecars can agree with each other while
being unrelated to any checkpoint bytes.

I reproduced this with the resolver's synthetic tmp fixture: changed a
teacher's `checkpoint_sha256.model_ep3.pt` to 64 zeroes, changed both
prediction sidecars to the same value, and left the `model_ep3.pt` file
absent. `resolve_template()` nevertheless wrote the downstream draft. This
is a separate content-binding failure from the now-fixed mass-summary issue.

Require and hash the exact epoch-3 checkpoint file; compare that hash with
provenance and the prediction sidecars. Add a negative fixture for a missing
checkpoint and a mismatched checkpoint digest. Keep fixed/shuffled templates
unresolved until this passes and regenerate the handoff packet against the
final source/test hashes.
