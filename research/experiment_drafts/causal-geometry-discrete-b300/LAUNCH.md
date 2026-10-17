# B300 launch and input contract

This packet's runner expects a clean checkout at `$PROJECT_ROOT`, exactly one visible CUDA card in each worker, four fold-bound portable manifests, six source inputs, persistent outputs, and a writable physical-disk scratch volume. Preflight checks `/proc/mounts`: all six data/cache inputs and the selected fold registry must resolve to read-only mounts; the persistent cache and artifacts volumes must be mounted read-write at their exact checkout targets, and physical scratch must be read-write; RAM-backed outputs or scratch are rejected.

## Required mounts

Create one clean checkout from the published experiment branch at the source SHA recorded in the PR. Keep the split JSON files from that checkout unchanged. Bind the data as follows:

| Source on the approved B300/data host | Checkout destination | Access | Purpose |
|---|---|---|---|
| supplied `datasets/train.parquet` | `$PROJECT_ROOT/datasets/train.parquet` | read-only | permitted training source |
| supplied `datasets/valid.parquet` | `$PROJECT_ROOT/datasets/valid.parquet` | read-only | permitted training and fixed evaluation source |
| canonical `cache/train_targets.npz` | `$PROJECT_ROOT/cache/train_targets.npz` | read-only file bind | labels, masks and sequence group metadata |
| canonical `cache/valid_targets.npz` | `$PROJECT_ROOT/cache/valid_targets.npz` | read-only file bind | official scoring labels/masks and row order |
| canonical `cache/chains.npz` | `$PROJECT_ROOT/cache/chains.npz` | read-only file bind | source-qualified component linkage |
| canonical `cache/chain_folds.npz` | `$PROJECT_ROOT/cache/chain_folds.npz` | read-only file bind | frozen component fold assignments |
| persistent writable cache volume | `$PROJECT_ROOT/cache/` | read/write directory; overlay the four input files above read-only | prediction arrays and sidecars |
| persistent writable artifacts volume | `$PROJECT_ROOT/artifacts/` | read/write | cell receipts, logs, profiles, run checkpoints and scores |
| portable manifest directory | `/mnt/geometry-inputs/` | read-only | `portable_inputs_fold0.json` through `portable_inputs_fold3.json` |
| physical-disk scratch volume | chosen host path exported as `WNN_B300_SCRATCH` | read/write | temporary files and PyTorch compile cache; must not be tmpfs/ramfs |

The four split manifests remain in the clean checkout at `research/campaign_2026-09-29/artifacts/splits/fold{0,1,2,3}.json`. The adapter checks their committed SHA256. The cache directory must remain writable for predictions, while the four input cache files are individually mounted read-only over it. Do not copy data, target rows, credentials or weights into Git or the evidence archive.

A container runtime may express those bindings directly. For a host-level bind mount, create the writable cache/artifact directories first, bind the supplied source files and directories to the destinations above, then remount the dataset directory and individual source files read-only. Use persistent storage for both outputs and scratch; do not use `/tmp` or `/dev/shm`.

## Portable source manifests

Generate the four registries on the approved source host that already owns the exact source files used by the committed split manifests. Do not build them from a relocated or re-exported dataset. The builder checks the original byte count and modification identity and computes a full content SHA256 for each source; its `--root` path is not stored in the portable registry.

```bash
set -euo pipefail
PROJECT_ROOT=/workspace/wnn_connectome_starterpack
SOURCE_ROOT=/approved/data/source/tree
PORTABLE_INPUT_DIR=/approved/data/geometry-inputs
mkdir -p "$PORTABLE_INPUT_DIR"
for fold in 0 1 2 3; do
  uv run --project "$PROJECT_ROOT" python "$PROJECT_ROOT/scripts/build_portable_manifest.py" \
    --root "$SOURCE_ROOT" \
    --split-manifest "$PROJECT_ROOT/research/campaign_2026-09-29/artifacts/splits/fold${fold}.json" \
    --output "$PORTABLE_INPUT_DIR/portable_inputs_fold${fold}.json"
done
```

Copy the generated small JSON registries to the manifest mount and mount the corresponding data/cache bytes unchanged. The canonical trainer verifies portable content against these registries when it reads the sources. The packet's preflight separately checks each registry's fold binding and source identities.

## Environment and measured profile gate

Run every command below inside the clean experiment checkout. Set these paths to the actual mounted locations on the B300 host:

```bash
set -euo pipefail
export PROJECT_ROOT=/workspace/wnn_connectome_starterpack
export WNN_PROJECT_ROOT="$PROJECT_ROOT"
export WNN_PORTABLE_INPUT_DIR=/mnt/geometry-inputs
export WNN_B300_SCRATCH=/b300-scratch/causal-geometry
cd "$PROJECT_ROOT"
test -z "$(git status --porcelain)"
uv sync --locked
findmnt -T "$WNN_B300_SCRATCH"
```

Run all eight single-card profiles concurrently:

```bash
scripts/geometry_b300/profile_all.sh
```

This invokes two warmup and eight measured optimizer updates on each card for the actual H512×1, 271-input workload. It records device identity, source commit, finite update evidence, timing, CPU RSS and CUDA allocated/reserved peaks. The receipt is sealed only when all eight device IDs are unique B300 cards and their profile windows overlap:

```text
artifacts/geometry_b300/profiles/profile_receipt.json
```

Inspect the receipt and logs before starting a scientific wave. It binds the source commit, H512 workload, strict precision policy, measured profile and the frozen **5,371,320-update** matrix budget (44,760/44,790/44,850/44,640 updates per fit in F0/F1/F2/F3). This is an update budget, not a wall-time forecast. There is no B300 wall-time estimate in this packet. If actual fit updates, precision flags, memory, or hardware differ from the receipt/recipe, stop and diagnose before proceeding.

## Launch the 15 eight-card waves

Each wave contains two pair indices and the four arms for each pair. Across all waves, the runner attempts exactly 120 scientific cells: 30 per arm, with seeds1001–1008 in F0/F1 and seeds1001–1007 in F2/F3.

```bash
set -euo pipefail
cd "$PROJECT_ROOT"
profile=artifacts/geometry_b300/profiles/profile_receipt.json
for wave in $(seq 0 14); do
  pair_a=$((2 * wave))
  pair_b=$((pair_a + 1))
  scripts/geometry_b300/launch_wave.sh "$pair_a" "$pair_b" "$profile"
done
```

Each card sees exactly one physical GPU through `CUDA_VISIBLE_DEVICES`; each worker trains one independent GRU. There is no DDP for these small fits and no queue manager. The launch script assigns four arms for pair A to GPU slots0–3 and four arms for pair B to slots4–7. It skips cells with a complete, matching receipt when a partially completed wave is resumed; incomplete training resumes only from the epoch-boundary `training_state.pt`. A run directory with no resumable state is preserved for diagnosis rather than silently restarted.

The single-cell adapter is also callable directly for a diagnosed missing/failed cell:

```bash
CUDA_VISIBLE_DEVICES=0 WNN_GPU_ID=0 WNN_PROJECT_ROOT="$PROJECT_ROOT" \
WNN_PORTABLE_INPUT_DIR=/mnt/geometry-inputs WNN_B300_SCRATCH=/b300-scratch/causal-geometry \
"$PROJECT_ROOT/.venv/bin/python" -u scripts/geometry_b300/run_cell.py \
  --pair-index 0 --arm combined --profile-receipt \
  "$PROJECT_ROOT/artifacts/geometry_b300/profiles/profile_receipt.json"
```

Select the exact pair/arm from `matrix.json`, and set the card's actual unique physical `WNN_GPU_ID`. The direct command refuses a completed cell unless `--skip-completed` is supplied. Do not relaunch a completed scientific cell as a new replicate.

## Artifacts and archive return

The runner writes per-cell phase commands, timestamps, exit codes and log hashes under `artifacts/geometry_b300/`; canonical training writes history, provenance, scores, checkpoint and resume state under `artifacts/runs/`; fixed-fold predictions and array-hash sidecars stay under `cache/`. Preserve every failure attempt.

Create two separate archives after terminal status:

1. **Evidence archive:** source/profile/cell receipts, command logs, run histories, provenance, score receipts, prediction arrays and sidecars, and portable input-manifest copies. Hash each file and the archive.
2. **Weights archive:** fixed epoch-2 checkpoints and epoch-boundary continuation states, with a per-file manifest and archive SHA256.

Store both in approved private storage and return the immutable location, SHA256 values, terminal source commit and any exclusions. Do not include supplied datasets, credentials, raw target rows or private reasoning logs.
