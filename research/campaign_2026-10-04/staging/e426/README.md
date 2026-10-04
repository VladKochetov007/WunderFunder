# E426 separated event-clock EMA: staged implementation

This is an isolated source overlay. The root `scripts/train_rnn.py` and
`src/wnn` production sources were not changed. The overlay is not launch-ready
until the root's sealed, resource-capped CPU checks pass; no imports, tests,
compilation, fits, or launches were run while the frozen accelerator jobs were
active.

## Behavior implemented

- `src/wnn/event_ema.py` counts the raw float32 endpoint marker in input-only
Parquet column `a6`, grouped into optimizer-update windows in the exact seeded
batch order. It implements `constant`, `aligned`, and deterministic
within-epoch `shuffled` consistency-teacher clocks. The aligned decay is
`base_decay ** (events_t * updates_per_epoch / total_train_events_epoch)`;
the schedule therefore preserves the constant clock's epoch product.
- Staged `scripts/train_rnn.py` keeps the consistency teacher EMA and fixed
per-update evaluation EMA as independent modules. Consistency targets are
detached; inference/checkpoint selection uses the evaluation EMA. Both states
are independently saved and restored. The legacy shared-EMA training-state
format remains readable.
- Clock counts come from the real, unpadded fit pool: TRAIN plus fit-VALID
row-groups included by `--valid-repeat`, excluding evaluation groups. Repeated
fit-VALID inputs are represented with the same multiplicity/order as the
concatenated training dataset. Before counting, `(split,row_group)` membership
is checked against the fold manifest. Provenance stores the ordered source
identity and hash, the event-count matrix hash/shape, manifest hash, and the
TRAIN row-group 528 exclusion. This is an offline epoch-normalized teacher
clock, not a causal input feature; future-label data are not consulted for
counts.
- E426 requires the explicit seeded epoch sampler and rejects sampling,
partial-epoch, row-clock EMA, and replay settings that invalidate the count
alignment. An epoch with no endpoint events fails closed in the schedule
builder.
- Resume state carries all prior epoch count-schedule identities, so final
provenance records the complete schedule after a restart rather than only the
post-resume epochs.

## Staged verification

`tests/test_e426_event_ema.py` covers marker counting, schedule normalization
and shuffle determinism, separated EMA restart, uniform-count equivalence,
zero-event behavior, source-order/drop-last count mapping, causal future-suffix
invariance, chunked state carry/reset, and detached teacher targets.
`tests/test_e426_trainer_resume.py` exercises the actual staged trainer loop on
synthetic 200-row Parquet/cache inputs (`SEQUENCE_LENGTH=200`, `chunk=100` only
inside the fixture): two epochs, combo-style losses, distinct teacher and
evaluation EMAs, fixed epoch-1 evaluation-EMA export, and bitwise equality
between uninterrupted and epoch-boundary resumed training. It still uses the
actual staged trainer loop, sampler, Muon optimizer, and both EMA update paths;
the 20,000-row production length is unchanged. The test also
checks source-hash closure, explicit TRAIN group 528 exclusion, and resumed
clock-history provenance. Its temporary root includes byte-for-byte copies of
the root WNN source closure because the trainer hashes `src/wnn/run_identity.py`
through its data root. The fixture installs the staged package path before
importing trainer modules, even if another test has already loaded root `wnn`.
Additional zero-event coverage checks FastEMA's first-call copy semantics (the
initial update seeds the average even for decay 1), then verifies later
zero-event updates freeze the teacher. These are test drafts only; they have
not been run.

## Integration closure and remaining checks

Use `scripts/run_event_ema_trial.py` from any working directory. The isolated
entrypoint places staged `src`, root `src`, and root `scripts` in import order,
loads the staged trainer, then sets its data/artifact `ROOT` to the repository
root. Before training, it hashes all root `src/wnn/*.py`, every staged WNN
override, the staged trainer, and the entrypoint. That source-closure record is
included in provenance and the resume contract. The staged
`src/wnn/__init__.py` extends the package path to the frozen root modules.
The staged source closure is:

- `scripts/train_rnn.py`
- `scripts/run_event_ema_trial.py`
- `src/wnn/__init__.py`, `src/wnn/event_ema.py`, and
  `src/wnn/training_state.py`
- both tests under `tests/`

It also depends on the corresponding frozen root source modules: `wnn.nn`,
`wnn.loss`, `wnn.training_controls`, `wnn.training_sampling`,
`wnn.training_dynamics`, data/features and split-manifest loaders, the existing
combo teacher/propensity cache readers, and optimizer implementation. A sealed
execution must hash this full closure and pin the split manifest, train/valid
Parquet, target and teacher caches, propensity cache, parent checkpoints, and
recipe. The actual trainer-resume test and the event/EMA tests still need to be
run under the authorized 1-CPU/2-GiB scope. Only after those pass should a
bounded synthetic/real-data smoke and recipe review determine whether the
experiment can be launched. Endpoint-count pre-read I/O cost has not been
measured.
