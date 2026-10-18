# Creative `prop_f0` history trace R1

Status: **the creative propensity checkpoint's producer remains unbound; its likely writer family is narrowed to the legacy trainer schema.** This note follows the exact creative model identity `9e92f26d…b813b21c` through targeted local Git history and one bounded read-only probe of two exact paths in the creative checkout. It does not load or hash model payloads or inspect arrays.

## What is bound to the creative checkpoint

The prior restricted metadata receipt identifies `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/prop_f0/model.pt` as 159,607 bytes, SHA-256 `9e92f26d1704f84f9170d746b372c2ba67f7fb611831865a1fc689d6b813b21c`, with safe metadata `hidden=64`, `n_in=211`, and top-level keys `hidden`, `model`, `n_in`. The two previously copied `propcache_prop_f0` sidecars repeat that model SHA and give row counts, but do not describe training membership, normalization values, source, or selected epoch. Those identity records are pinned in [HISTORY_TRACE_R1.json](HISTORY_TRACE_R1.json).

The similarly named restored Main control is a different artifact: its receipt binds `c20260929_prop_f0/model.pt` to SHA-256 `b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7` (165,623 bytes). That run's recipe and split provenance must not be transferred to the creative `prop_f0` file.

## The nearest committed history is not an identity link

The local append-only experiment ledger has one early propensity precursor, E-031: command `PYTHONPATH=scripts scripts/limited python scripts/train_propensity.py --name prop_v1`, fitted on validation devA (741 sequences) and scored on devB (764). Its recorded `commit` is `44f7eff5…`, whose tree does not contain `scripts/train_propensity.py`; the source first appears in the following tracked commit `9833919b…`. E-031's 342-byte history records four devB AUC values, but no model SHA, command flags beyond the name, or checkpoint-selection digest. It describes `prop_v1`, not `prop_f0`, and cannot identify the creative checkpoint.

The source history explains the minimal checkpoint schema. The initial tracked script fixed Torch seed 0, defaulted to hidden 64 / four epochs / learning rate 0.002, normalized from the first shuffled training batch, and saved only `model`, `hidden`, and `n_in` at the final epoch. Commit `f9167b30…` added optional `--heldout-fold` selection from `valid_targets.npz` and `chain_folds.npz`, retaining that wrapper. Commit `4c260c9a…` added `--dev-all` and selected the saved model by best devB AUC while still saving only those three keys and writing `history.json`. Commit `d1f6e177…` later added explicit split/provenance and checkpoint policy options, and embedded `epoch`, `args`, and `provenance` in the model object. Exact commit and Git blob identities plus SHA-256 values are pinned in [HISTORY_TRACE_R1.json](HISTORY_TRACE_R1.json).

The creative checkout's current `scripts/train_propensity.py` was read from its exact path in the one bounded remote probe. Its 5,034-byte contents have SHA-256 `72fab2d0db66d513aca1e0eab6974eaa5ceb1df0c88fbcf25c842b2fa0a6d1b1`, exactly matching the tracked `4c260c9a…` version. The checkout's exact `research/experiments.jsonl` index is 216,209 bytes with SHA-256 `22385503a0a4972f776a3028265da73fac4380d70ed8cc79c4e90c14a8198357`; a bounded exact-string check found no `prop_f0` or full model-SHA entry in that file. This establishes the current source and the searched index only; it does not show that the current code produced the checkpoint.

The observed legacy source is structurally compatible with the creative checkpoint's minimal metadata wrapper, so it narrows the candidate writer family. It still cannot establish whether the run used default devA/devB, `--heldout-fold 0`, or `--dev-all`. If it used `--heldout-fold 0`, this implementation selected `model.pt` by AUC on the held-out fold; that would expose fold-0 masks to checkpoint selection. This is a conditional source behavior, not an observed recipe for the creative file.

## What remains unknown

No exact command/job record, per-run trainer hash, training history/provenance file, normalization values, or selected epoch is bound to the creative SHA. The creative run's inspected `history.json`, `provenance.json`, and other enumerated metadata paths were not reported as regular files in the earlier [producer trace probe](../tl3_producer_trace_r1/README.md). The remote check here added only the current source and exact research-index query; it did not inspect a new run path or retry any prior missing-path checks.

Therefore the creative checkpoint's fit groups, data identity, training flags, normalization values, checkpoint selection, and exact source revision remain unresolved. Do not equate it with E-031 or the restored Main control, and do not treat matching tensor metadata or source compatibility as proof of clean-F0 ancestry.

## Evidence and limits

[HISTORY_TRACE_R1.json](HISTORY_TRACE_R1.json) binds each supporting local metadata file by byte count and SHA-256, records the exact Git revision/blob fingerprints, and includes the normalized result of the bounded remote probe. [PREPARATION_MANIFEST_R1.json](PREPARATION_MANIFEST_R1.json) pins this packet's files and supporting inputs. No remote mutation, training, inference, checkpoint read/hash, or prediction/propensity array access occurred in this task.
