# Receiver R2: lower checking overhead and compact outputs

The November10 request authorizes server CPU profiling, followed by full
validation against the served TL3 control. No current server connection or
data mount has been supplied or recovered. No remote model execution has started.
The submission ZIP remains byte-for-byte unchanged:
`766bab1f3205697819af994acbb54cfe60682651977f393d76bec69608a75e3e`.

## Changes

- The launcher defaults to two complete profile sequences; eight remain an
  explicit option. Reset, mask independence, causality and Torch-reference
  checks are retained. This is a smaller profile, not a smaller validation set.
- Full validation calls the actual callback on every one of37,460,000 rows.
  It scores the actual returned candidate output and the same-pass TL3
  component on the canonical mask. Requested outputs remain shape/dtype/finite
  checked. Component finiteness is checked in vectors per sequence; repeated
  float64 mixture verification is confined to the strict profile/state path.
- Save only3,528,273 scored rows: candidate, base and targets as float32 pairs,
  seq_ix as int64, and step_in_seq as uint16 (sequence length20,000).
  These aligned arrays preserve paired-analysis inputs. Payload size is
  119,961,282 bytes plus NPY headers, versus599,360,000 bytes previously.
  The roughly80% reduction is calculated from the storage schema, not measured
  from a completed validation run. No speedup factor is claimed.
- A changed checker invalidates earlier profile receipts through the existing
  receiver/scorer source hashes. The same ZIP/input/recipe identities remain.

## Evidence and next action

Ruff format/check, shell syntax, Python AST and source/ZIP hash checks are static
verification only. Local tests, model imports, inference, training, Docker
build/run and server execution remain unperformed. Independent review checks
the all-row recurrence and scored-row alignment; server parity remains pending.

The original delivery is preserved. The R2 handoff contains the same submission
ZIP and updated receiver. [RUNBOOK.md](RUNBOOK.md) has the commands. Admission
needs an accessible Linux x86-64 server, canonical valid.parquet and a free CPU.
Inspect the short profile before admitting the bounded full run. No training
or portal upload is part of these commands. No existing artifacts were deleted:
the user's deletion reference did not identify a file or directory.
