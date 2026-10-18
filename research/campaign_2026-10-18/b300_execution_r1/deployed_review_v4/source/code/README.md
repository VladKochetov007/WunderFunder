# Thin tabdata L4 trainer bridge

## Canonical L4 input reconstruction

The original main-branch split manifests are tracked at `research/campaign_2026-09-29/artifacts/splits/fold{0,1,2,3}.json`; there is no tracked `chains.npz`, `chain_folds.npz`, or target-cache payload. `validate_l4_targets.py` checks target row-group order, every target row's sequence ID and values against the Parquets, valid-mask alignment against both mask sources, and each target index against the sequence IDs in all four exact-hash canonical manifests. Its preflight receipt records whole-file train/valid/valid-mask hashes. Manifest generation is blocked unless the raw train and valid Parquets match the frozen R4 SHA-256 identities.

The graph is regenerated only from the pinned a266 `scripts/sequence_chains.py` and `src/wnn/data.py`. `derive_l4_folds.py` checks the normalized script/data source hashes, the staged executable script hash, the preflight receipt and exact canonical graph size/SHA before deriving folds. A graph hash mismatch is a hard error; matching component labels alone cannot override it. The helper then checks all four original manifests' graph-node components and derives fold membership, rejecting incomplete folds or validation components split across folds. `canonical_manifest_checks.py` is the final gate after fresh manifests are built: it requires exact equality for the full `sequences`, `fit`, `evaluation`, and `excluded` objects in every fold, while allowing attempt-local source metadata and manifest hashes to differ.

From the pinned a266 checkout and a fresh prep root (substitute controller-owned paths), stage the raw Parquets under `<prep>/datasets/`, and create `<prep>/cache/` and `<prep>/scripts/`. Extract the four canonical manifests from the frozen repository revision into `<prep>/canonical/`. Then run:

```bash
mkdir -p <prep>/canonical <prep>/cache <prep>/scripts <prep>/receipts
for fold in 0 1 2 3; do
  git -C <frozen-repo> show main:research/campaign_2026-09-29/artifacts/splits/fold${fold}.json \
    > <prep>/canonical/fold${fold}.json
done
cp <a266-source>/scripts/cache_targets.py <prep>/scripts/cache_targets.py
cp <a266-source>/scripts/sequence_chains.py <prep>/scripts/sequence_chains.py
PYTHONPATH=<a266-source>/src python <prep>/scripts/cache_targets.py train
PYTHONPATH=<a266-source>/src python <prep>/scripts/cache_targets.py valid
python <packet>/validate_l4_targets.py \
  --train-targets <prep>/cache/train_targets.npz --train-parquet <prep>/datasets/train.parquet \
  --valid-targets <prep>/cache/valid_targets.npz --valid-parquet <prep>/datasets/valid.parquet \
  --valid-mask-parquet <prep>/datasets/valid_mask.parquet \
  --canonical-dir <prep>/canonical --receipt <prep>/receipts/input-preflight.json
PYTHONPATH=<a266-source>/src WNN_THREADS=2 python <prep>/scripts/sequence_chains.py
python <packet>/derive_l4_folds.py --graph <prep>/cache/chains.npz \
  --canonical-dir <prep>/canonical --source-root <a266-source> \
  --executed-script <prep>/scripts/sequence_chains.py \
  --input-preflight <prep>/receipts/input-preflight.json \
  --output <prep>/cache/chain_folds.npz --receipt <prep>/receipts/folds.json
PYTHONPATH=<a266-source>/src python <a266-source>/scripts/build_campaign_splits.py \
  --root <prep> --output-dir <attempt>/splits
python <packet>/canonical_manifest_checks.py \
  --canonical-dir <prep>/canonical --fresh-dir <attempt>/splits
```

The canonical graph gate expects 650,142 bytes and SHA-256 `39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766`. The fold output may have new NPZ container bytes; the receipt labels this as `new_attempt_local_reencoded` with a reason. Keep fresh manifests as a separate attempt identity and do not label them with the old R4 manifest hashes. Because manifests are generated against the actual L4 data root and preserve its current metadata, use that root directly; do not add a portable full-file registry pass that would rescan the 35 GB Parquets for each fit. The preflight receipt and membership comparison are required before any profile or fit. The focused CPU tests cover graph-hash rejection, cross-fold component rejection, row-group ordering and canonical sequence-index disagreement; they do not claim that an L4 fit or profile ran.

`train_cell.py` invokes the canonical `a266b036b02e8e012c6eb31072291dad618a169d` trainer through its existing `TrainingEnvironment` and `run_training` API. It does not copy the trainer or alter its fit loop. `command_builder.json` pins the R4 rank-1 recipe hash, the separate canonical 61-file trainer identity, the complete deployed source-tree manifest, wide/deep shapes, training settings, output contract and profile/fit/resume argv additions.

The controller must create `WNN_RUN_DIR` as a fresh empty per-attempt directory before invoking the GPU wrapper. The bridge receives distinct `--source-root`, `--data-root` and `--split-manifest` paths. It sets `TrainingEnvironment.source_root` to the frozen source checkout and `.root` to `WNN_RUN_DIR`; this keeps `artifacts/runs/<name>` and grade outputs local to the attempt. `datasets/` is a directory symlink to the supplied data tree. `cache/` is a local writable overlay with symlinks for `chains.npz`, `chain_folds.npz`, `train_targets.npz` and `valid_targets.npz`, so the grader can write predictions without mutating the shared input cache. Both target indexes are required by the canonical split loader's complete source-identity checks. No parquet or cache payload is copied.

Example argv template for a profile (the host paths and `WNN_RUN_DIR` are supplied by the controller):

```text
WNN_RUN_DIR=<fresh-fit-attempt> python gpu_entrypoint.py train_cell.py \
  --source-root <frozen-a266-source> --data-root <verified-data-root> \
  --split-manifest <verified-split>/fold0.json --arm wide --fold 0 --seed 1002 \
  --mode profile
```

`--prepare-only` builds symlinks and writes `command.json` plus `adapter_preflight.json` without importing PyTorch or fitting. Profile mode adds three finite profile steps and emits `PROFILE` / `PROFILE_UPDATE_EVIDENCE` to the controller log; the producer receipt deliberately says `profile_completed_check_controller_log`, so the controller must validate those records before admitting a fit. Fit mode runs the full six-epoch schedule, fixed epoch 5, strict FP32, batch 8, chunk 2000, seeded ordering, and no fit-time evaluation. It saves `training_state.pt` at each epoch boundary. Resume mode requires the existing state file and the same per-run workspace; the trainer validates the continuation contract.

Before training and export, the bridge checks every regular file under deployed `src/`, `scripts/` and `configs/` against `a266_deployed_tree_manifest.json`, generated from the frozen a266 archive (533 files, archive SHA256 `420395e9fd1206ec0466c8304b80dc095e9d416b97752ebfe39ac53a78e5b2b1`). This catches changes in imported behavior-bearing modules such as `wnn.nn`, `features`, `data`, `optim`, `splits` and `resources`, plus added or missing files. The separate canonical 61-file manifest remains unchanged for exact `training_source_identity` provenance. Both identities and manifest pins are carried through preflight, commands and receipts. The check works without `.git`. Bytecode generation is disabled so importing the frozen source does not create unmanifested `__pycache__` files.

`gpu_entrypoint.py` is the fit/export process wrapper: invoke it around each actor with `WNN_RUN_DIR` set to that actor's unique output root. It requires exactly one visible L4, sets strict FP32, caps the process allocator at 6 GiB, records peak GPU allocation, and sets `WNN_THREADS`, OMP, MKL, OpenBLAS and NumExpr pools to two before importing the actor. The actor also records these environment caps. The export predictor's frozen defaults are batch 16 and 4,000 steps per chunk, so the first completed-fit export is the exact-path resource admission run; inspect its successful `export_cell_resource_receipt.json` and peak allocation before authorizing further exports. Its allocator cap is a per-process limit, not a device reservation.

The bridge also checks the split fold, required input files, data-root separation and exact symlink targets. It pins loader workers to zero, sets a 1 GiB loader budget and TF32-off environment flags, and records any controller-supplied `WNN_GPU_MEMORY_BYTES`.

After a completed fit, `producer_receipt.json` binds the command and adapter hashes, verified trainer source manifest and the matching runtime trainer provenance identity, split/data ancestry, fit membership, epoch update counts, and SHA-256 plus byte size for all six `model_ep*.pt` snapshots, selected `model.pt`, `history.json`, `provenance.json` and `training_state.pt`. It rejects a run whose saved args differ from the frozen shape, fold, seed, optimizer budget or fixed checkpoint.

## Feature-only export after a completed fit

The trainer writes checkpoints and fit-only history. `export_cell.py` invokes the pinned a266 `scripts/predict_run.py` entrypoint, selecting only `evaluation.valid`, `--epoch 5` and `--inference-precision fp32`. That predictor's `FeatureOnlySequences` reads each sequence with `targets=False, mask=False`, and the checkpoint-bound campaign split determines exact output order. The wrapper checks the fixed-epoch fit receipt and checkpoint hash before export, then calls the unmodified R4 grader's `validate_prediction()` / `validate_export_manifest()` against the produced `.npy` and sidecar. It writes `export_receipt.json` with exact code, split, membership, prediction, sidecar and checkpoint hashes.

Example command for one completed fit (paths are supplied by the controller):

```text
WNN_RUN_DIR=<fresh-export-root> python gpu_entrypoint.py export_cell.py \
  --source-root <frozen-a266-source> --run-dir <completed-fit-root> \
  --export-dir <fresh-export-root> \
  --split-manifest <verified-split>/fold0.json --run-name tabdata_l4_r1_wide_f0_s1002 \
  --fold 0 --seed 1002 --grader-script <repo>/research/campaign_2026-10-18/priority25_execution_r4/code/grade_endpoint.py
```

The separate export workspace has a local writable prediction cache and links to the fit's immutable artifacts, datasets and frozen cache inputs. If an export fails, preserve that failed export attempt and retry with a new `--export-dir` linked to the same completed `--run-dir`; do not rerun a valid fit just to regenerate predictions. The first resource-admission export may write a full prediction artifact, but only one actor is admitted until its process receipt is inspected.

This wrapper validates an export but does not score labels; the separate R4 grade command remains downstream. The trainer, predictor, R4 grader and all adapter helper files are content-hashed into command and producer receipts. The bridge packet has only been checked locally; it records no L4 profile, fit, export or score.

This local packet contains bridge design and static checks only. It records no L4 profile, fit, export, score or remote action.
