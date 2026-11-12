# B300 throughput engineering R1: larger batches with A4 retained

The user's explicit 7 November optimization steering authorizes bounded
throughput/config/source engineering. ROOT owns all SSH/GPU actions and retires
the old partial B300 fits only after preserving their evidence. Healthy L4 work
continues. This packet creates **zero scientific fits** and does not change S3,
the live receiver, precision, data, labels or primary chronological validation.

## First actual-data allocation

ROOT admitted four parallel, exclusive-device engineering profiles:
`labels_s101001_w2_b16_a4`, `...b32_a4`, `...b64_a4`, `...b128_a4`.
Each uses 50 successful updates, first 20 warm and last 30 measured, with a
900-second external bound. The remaining prepared recipes are not automatically
admitted or queued. No new scientific fit starts until ROOT chooses/reviews a
matched new recipe.

[manifest.json](manifest.json) contains 20 explicit configs and exact direct
canonical argv/environment values: both arms, seed101001, world1/2 and
B4/16/32/64/128, all A4. All remote paths are POSIX. The current manifest SHA256
is `251680b6ddad228de3c20f268d861ddecf16be19c581982a76657d4d442a2f0b`;
all 20 config-file hashes and every argv/environment path were checked after
fixing the initial Windows path-separator error. Config bytes were unchanged by
that path repair.

Stage this folder at
`/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/throughput-engineering-r1`.
The scientific profile source is the existing frozen
`source-r3` at `6ab2f679e70928f913b97d5a29f0d1e09ceacaa3`. Call its existing
`scripts/train_sequence.py --profile-updates 50` directly, through ROOT's known
owned-command helper. Do not use S3's frozen four-config wrapper for these new
configs, and do not overlay the optional Muon source preparation onto S3.

```bash
R=/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1
S3="$R/source-r3"
P="$R/throughput-engineering-r1"
PYTHON=/data/epanov_s/tabdata/.venv/bin/python
export PYTHONPATH="$S3/src"
export CUBLAS_EMULATE_SINGLE_PRECISION=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export CC=/usr/bin/gcc CXX=/usr/bin/g++ TORCHINDUCTOR_COMPILE_THREADS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WNN_MONITOR_HEARTBEAT_SECONDS=5
export TMPDIR=/data/epanov_s/tdn7run/throughput-r1/scratch
export TORCHINDUCTOR_CACHE_DIR=/data/epanov_s/tdn7run/compiler/inductor
export TRITON_CACHE_DIR=/data/epanov_s/tdn7run/compiler/triton
mkdir -p -- "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" -m torch.distributed.run --standalone \
  --nnodes=1 --nproc-per-node=2 "$S3/scripts/train_sequence.py" \
  --config "$P/configs/labels_s101001_w2_b16_a4.json" \
  --data-root "$R/project" \
  --split-manifest /data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/data/chronological_primary.json \
  --monitor-reference-root "$R/reference" \
  --output "$P/profiles/labels_s101001_w2_b16_a4" --profile-updates 50
```

Use fresh profile outputs. For complement recipes add
`--reference-cache "$R/reference/fit"`; labels must omit it. The source requires
the monitor reference root whenever the config has monitoring, including profile
mode. Profiles write unapproved engineering receipts and no scientific checkpoint.

## Exact budget changes

Every config retains T768/L24, 12 heads, context1024, chunk256, A4, efficient
compiled attention, BF16/BF16x9, FP32 master/state/loss, original Muon/AdamW,
LR0.0003, five-percent warmup, original weighting/masks and all-fit normalizer.
Larger batch changes global observation groups and optimizer/scheduler update
counts explicitly; it is a new scientific recipe if later admitted, not numerical
equivalence to the old B4 arm. No LR scaling is silently applied.

| Per-rank B | W1 updates/epoch | W1 updates/30E | W2 updates/epoch | W2 updates/30E | W2 global chunk mass |
|---:|---:|---:|---:|---:|---:|
| 4 | 51,640 | 1,549,200 | 25,820 | 774,600 | 32 |
| 16 | 12,920 | 387,600 | 6,460 | 193,800 | 128 |
| 32 | 6,460 | 193,800 | 3,240 | 97,200 | 256 |
| 64 | 3,240 | 97,200 | 1,620 | 48,600 | 512 |
| 128 | 1,620 | 48,600 | 820 | 24,600 | 1,024 |

Counts are `ceil(ceil(10325/world)/B) * ceil(79/4)`, times30 for a full fit.
Every 30-epoch recipe still forwards the same 6.195 billion real rows per fit;
the gain must come from measured hardware throughput and amortized update cost.
The full-warmup schedule recomputes its exact update budget from the new shape.
For peak-LR diagnostics after selecting a shape, `build_profiles.py --peak` writes
a separately named zero-warmup variant; root must separately admit that profile.

At equal global batch, compare W2/B16 with W1/B32, W2/B32 with W1/B64, or
W2/B64 with W1/B128. They retain A4 temporal grouping and the same mathematical
real observation groups, with numerical reduction/batch-shape differences.
W2/B128 needs W1/B256 for a truly equal-global-batch comparison, outside the
initial matrix; admit it separately only if measured memory headroom supports it.
Compare complete four-cell wall time, not isolated utilization or steps/second.

## Read measured results correctly

Use only rank0's globally reduced row counts; summing both ranks doubles them.
Within the last30 updates use total real rows divided by summed window seconds,
and report optimizer seconds/fraction, preclip norm/clipping incidence and all-rank
peak allocated/reserved memory. Retain native/owned total duration separately:
window timing excludes batch loading, while the profile total includes loading
and compilation. First20 updates cover one complete 20k-sequence pass, including
the short final chunk and state reset; last30 span later passes. Preserve warm
and tail diagnostics instead of pretending every update forwards the same rows.
Inspect actual batch-load/startup/compile costs before extrapolating full30E.

No actual GPU result is claimed by preparation. The old ~0.71s/update and ~0.16s
optimizer measurements motivate this matrix; they are not estimates for larger B.

## Optional separately identified Muon source

The actor prepared `muon_batch_by_shape=true` outside frozen S3. It packs the
canonical independent Newton–Schulz work by matrix shape across layers: 96
per-parameter calls covering144 independent blocks become four packed buckets.
Each parameter retains its own FP32 momentum buffer and each Q/K/V block retains
its own Frobenius normalization, five-step polynomial, RMS scale and weight decay.
AdamW is unchanged. Default/omitted flag retains the old path and payload fields.

This source has 14 passing CPU checks including legacy Muon tests, widely
different block scales, missing gradients, state reload and FP32 rejection.
It has **no actual GPU/BF16x9 parity or speed result**. Packing changes GEMM
execution and needs fresh source/config/runtime identity, actual-device parity,
DDP checks and a measured profile before use. Temporary optimizer workspace can
increase. ROOT can defer this source if the measured larger-batch optimizer
fraction is already small; it must not delay the same-S3 shape measurement.
