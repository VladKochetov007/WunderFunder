# R2 B-side CPU preflight utilities

**Current run, 18 October 15:22 UTC:** the root-gated CPU preflight is running
as `preflight-r2-20261018t1520z`. The first tar call omitted `--gzip` and failed
before extraction ([retained failure](records/preflight-r2-20261018t1520z/STAGE_FAILURE.json)).
A separate recovery script then staged into the exact empty root successfully
([stage receipt](records/preflight-r2-20261018t1520z/STAGE_REMOTE_RECEIPT.json)).
The preflight result has not arrived, so no pass is claimed. This run does not
load a checkpoint, predict, grade, train, or use the GPU; root owns admission.

The static reviewer approved only a root-gated preflight. At review time, the
independent review records manifest SHA-256
`1785e8529a2868e1ca40a0c668c57ccc28ae82b04fda7c13bf7a3c8bc32c409b` and
`PASS_STATIC_REVIEW_READY_FOR_ROOT_GATED_PREFLIGHT`; it explicitly says that
no real preflight had run yet. Root compared that digest before this launch;
the resulting raw receipt still needs review against the same pin. The local
launcher rereads the review and overlay pins before each invocation. It does not admit an
export.

## Exact staging layout

The stage command uses the existing SSH alias `beleriand` by default. It
transfers only the frozen source snapshot, the exact R2 manifest-listed source
and support files, the old split, the portable registry, the membership
receipt, and separate R1/R2 review evidence. The R2 review is staged outside
the runtime overlay. It never copies unlisted files into that overlay.

On B, the project root is created only if it does not exist:

```text
/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root
```

An existing path, symlink, failed partial stage, or prior output causes a hard
stop. There is no cleanup or overwrite path, and this utility does not yet
accept a different remote root. Preserve any partial stage for root inspection;
do not retry or remove it automatically.
The root keeps these project inputs at their pinned logical paths:

| Project path | Source path on B |
| --- | --- |
| `datasets/train.parquet` | `/home/grinchenkoeg/computer-vision-homework-20261018/data/datasets/train.parquet` |
| `datasets/valid.parquet` | `/home/grinchenkoeg/computer-vision-homework-20261018/data/datasets/valid.parquet` |
| `cache/chains.npz` | `/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/chains.npz` |
| `cache/chain_folds.npz` | `/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/chain_folds.npz` |
| `cache/train_targets.npz` | `/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/train_targets.npz` |
| `cache/valid_targets.npz` | `/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/valid_targets.npz` |

All six are symlinks. The helper checks literal and resolved link targets and
their sizes. It fully hashes only small caches; the accepted chain-fold link
must point literally to the observed `data/cache/chain_folds.npz` symlink and
resolve to `receipts/chain_folds_derived.npz` at 462,719 bytes with SHA-256
`a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567`. The
frozen portable registry continues to record the old 550,072-byte cache
identity; the separate R2 binding records the reviewed compatibility exception.
No parquet body or target array is read during staging.

The four restored control/propensity model and provenance files are links to
their exact `restored-controls-v1` and `restored-propensity-v1` paths. The
helper checks each file's full size and SHA without deserializing it. `cache/`
is a new writable project directory. Before preflight, the helper verifies
the old split, registry, membership receipt, H390 snapshot manifest, R2 source
and support inventories, and both review records.

The stage plan also binds the exact `remote_stage.py` byte count and SHA-256.
The local wrapper checks that digest on B before invoking the helper for both
staging and launch-time verification, and the helper checks the same pin again
against the staged binding.

The H390 snapshot and runtime overlay must contain no `__pycache__`, `.pyc`, or
`.pyo` files. The bundle is assembled from the frozen manifest and R2 overlay
manifests explicitly; the two reviewer JSONs are separate stage metadata.
Both local and remote checks reject bytecode and any extra file. The runtime
uses `-B` and `PYTHONDONTWRITEBYTECODE=1`; those settings prevent new cache
creation, while the inventory checks prevent stale bytecode from being used.

## Bounded preflight

The selected B interpreter is
`/home/grinchenkoeg/WunderFunder/.venv/bin/python`. The launch sets
`PYTHONPATH` to the reviewed overlay `src` first and the frozen snapshot `src`
second; it passes the old split, registry, and membership evidence explicitly.
The predictor and trainer snapshots remain unchanged.

The loader streams full hashes for about 37.1 GB. No measured full-scan rate
is available, so this is a hard 900-second wall cap rather than a runtime
forecast. Finishing within that cap requires roughly 41.2 MB/s effective
read-and-hash throughput. The unit receives 50% of one CPU, 1.3 GB
`MemoryHigh`, 1.5 GB `MemoryMax`, no swap, and a 10-second stop grace period.
`TMPDIR` points to B's physical scratch directory
`/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp`. The launch
also disables CUDA and caps common numerical-library thread counts at one.
`flock` prevents duplicate preflight units. No process loads the model,
prediction, or target arrays; the run uses no GPU, training, or grading.

The remote result is written by R2's exclusive-create output guard at:

```text
/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2/COMPATIBILITY_RECEIPT_R2.json
```

The launcher stores stdout and stderr as raw local bytes. The capture utility
copies the receipt without JSON reserialization, checks the copied byte count
and SHA against a fresh remote observation, and validates the receipt's
reviewed manifest, split, registry, membership, cache, artifact, and scope
bindings. It records the actual receipt SHA after capture; no output digest is
predetermined. A valid receipt remains evidence for root to inspect, not
export admission.

## Commands for root

Run these only after root records a fresh B resource check, confirms the lane is
clear, and completes the external manifest review check. The wrapper applies
the preflight unit caps but does not measure or gate current B headroom. Use a fresh lowercase attempt ID;
the local record directory and remote project are create-only.

```bash
cd /Users/nb/Documents/wunder/WunderFunder-maintenance/research/campaign_2026-10-18/quality_gap_r1/export_preparation/preflight_execution_r2
python3 stage.py --ssh-alias beleriand --attempt-id preflight-r2-20261018t151500z-a1b2c3d4
python3 launch.py --ssh-alias beleriand --attempt-id preflight-r2-20261018t151500z-a1b2c3d4
python3 capture.py --ssh-alias beleriand --attempt-id preflight-r2-20261018t151500z-a1b2c3d4
```

Each command is a separate operation. Review `records/<attempt-id>/STAGE.json`,
`LAUNCH_STARTED.json`, `EXECUTION.json`, and `CAPTURE.json` before proceeding.
If SSH stops waiting, do not launch a second unit or remove the lock/root;
inspect the recorded unit with B's user systemd state first. An interrupted
stage is preserved for inspection, and the fixed project root will refuse a
retry that could overwrite it.

## Local checks

The small local tests exercise receipt binding, the old-registry/current-cache
separation, the staged helper hash pin, and exclusive evidence writes. They do
not make SSH calls or read large inputs.

```bash
python3 -m unittest -v test_preflight_execution.py
python3 -m py_compile bindings.py local_common.py remote_stage.py stage.py launch.py capture.py
ruff format --check bindings.py local_common.py remote_stage.py stage.py launch.py capture.py test_preflight_execution.py
ruff check bindings.py local_common.py remote_stage.py stage.py launch.py capture.py test_preflight_execution.py
```
