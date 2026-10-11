# Old-Z4 F0 core crop and HFT successor (R5)

This packet prepares a clean F0 HFT of the existing seed-3 old-Z4 student. It
starts from the actual student CP1, removes only the eight auxiliary family
output rows, then fine-tunes the resulting 23-head checkpoint on the 1,405
fold-eligible `valid.parquet` groups. The fixed F0 evaluation set has 468
groups. Both counts and the split manifest are pinned below.

The crop and training inputs are our own artifacts and source snapshots. The
frozen HFT source is the already reviewed c337 implementation; the runner
imports that exact module and assigns `ROOT` on the module object before
calling its entrypoint. It does not use `runpy` or mutate a returned namespace.
The imported dataloader helper is checked under `WNN_LOADER_WORKERS=0` before
any stage runs.

No fit or export is admitted by this packet. Root owns the short CUDA profile,
profile interpretation, fit release, terminal finite-checkpoint gate and
export release. The runner rejects fit until separate root profile and fit
admission receipts exist, and rejects export until a terminal fit receipt
exists. Receipts and generated artifacts are write-once; retries need a new
stage/run ID.

## Frozen stage commands

Run from the project root and keep the source worktree available for the full
job. `scripts/limited` provides the bounded support process; these commands do
not themselves release a fit.

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=300s \
WNN_LOADER_WORKERS=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_student_core_hft_f0_r5/prepare_core_crop_r5.py \
  --project-root /home/vlad/development/wunder/wnn_connectome_starterpack
```

The crop command is CPU-only. It requires a root-owned
`ROOT_CORE_CROP_ADMISSION_R5.json`, the source checkpoint and provenance hashes
in `CORE_CROP_CONTRACT_R5.json`, refuses an existing destination, and checks
every serialized model tensor before writing a write-once producer receipt.

After root records the one-stage crop admission and admits the CUDA profile,
run the exact profile-only stage. The wrapper requires a root-created
`ROOT_HFT_PROFILE_LAUNCH_R5.json` bound to this profile contract and crop
receipt before it imports and runs the trainer:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=6800000000 WNN_RUNTIME_MAX=1800s \
WNN_LOADER_WORKERS=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_student_core_hft_f0_r5/run_hft_stage_r5.py \
  --stage profile --project-root /home/vlad/development/wunder/wnn_connectome_starterpack
```

The profile has two warmup batches and 25 measured batches at batch 16, chunk
2,000, with gradient statistics enabled. It uses the same F0 parent, source,
optimizer, schedule and objective as the proposed one-epoch fit, then exits
without an epoch checkpoint. Profile output is execution evidence, not a
scientific score or fit result.

For timing context, the existing local `f0_h230_hetero_s3_hft_v1` run processed
the same 1,405 F0 fit groups for 68.5 seconds at batch 16/chunk 2,000, one epoch,
AdamW/onecycle, frozen `inp,gru`, GCorr, fp16 and the same `5e-4` learning rate
and `0.01` weight decay. Its provenance SHA-256 is
`5fb2cc96eb16571616c0569dd2fcca99f312666e4a79f43082b68ddeef06f5fd`. That run
has a different initializer ancestry, so this is a measured planning analogue,
not a duration guarantee; the actual-device 27-batch profile remains the
admission input.

The fit runner uses the same resource envelope, `--stage fit`, and will remain
blocked until `ROOT_HFT_PROFILE_ADMISSION_R5.json` records the actual device
profile and finite update evidence, and `ROOT_HFT_FIT_ADMISSION_R5.json`
releases this exact one-epoch contract. The frozen fit argv and output name are
in `HFT_FIT_CONTRACT_R5.json`.

After a root-owned terminal checkpoint gate, the same runner accepts
`--stage export`; it uses the exact frozen c337 `predict_run.py` with module
`ROOT` rebound to the project root. It predicts the full valid bank for
feature-only downstream use. Grading remains limited to the fixed 468 F0
groups and is previously exposed robustness evidence.

## Validation commands

The no-write runner import check verifies exact trainer/predictor imports,
actual `Sequences.path` resolution and the imported zero-worker helper. The
actual-producer crop check uses the available CP1 but copies it to a temporary
directory before applying the frozen crop primitive; it never writes to the
project artifact tree.

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s WNN_LOADER_WORKERS=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_student_core_hft_f0_r5/run_hft_stage_r5.py \
  --stage validate-import --project-root /home/vlad/development/wunder/wnn_connectome_starterpack

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s PYTHONDONTWRITEBYTECODE=1 \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_student_core_hft_f0_r5/verify_actual_cp1_crop_r5.py \
  --project-root /home/vlad/development/wunder/wnn_connectome_starterpack
```

Both checks are bounded CPU validation only. No GPU fit, profile, export,
provider call, or queue action is part of packet construction.
