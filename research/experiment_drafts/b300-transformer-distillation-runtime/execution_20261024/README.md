# PR32: measured Blackwell execution, October 24

The latest continuation is the [R4 batch-throughput amendment](BATCH_THROUGHPUT_20261024.md):
96/GPU for 630M and 64/GPU for 1.54B, selected from real-data measurements through
batch 128 and a separate CPU/transfer diagnostic. The R2 attempt described below
was deliberately superseded at approximately 0.41% of its update budget; its
historical evidence is preserved. R3 comprised engineering profiles only.

The selected production revision is `pr32-blackwell-compiled-r2`, pinned to
[`f1beeb4c210a9da01735e03b1ff6e27be878b873`](https://github.com/VladKochetov007/WunderFunder/commit/f1beeb4c210a9da01735e03b1ff6e27be878b873).
The user authorized the full four-teacher experiment, then explicitly authorized
precision/backend/batch/layout profiling. The earlier four-hour campaign deadline
does not apply. The [precision amendment](../BLACKWELL_THROUGHPUT_20261024.md)
preserves the original strict-FP32 R3 recipe and identifies this as a new execution
arm. [Live progress](https://github.com/VladKochetov007/WunderFunder/pull/32#issuecomment-6377032958)
separates successful optimizer updates from evaluated scientific fits.

There are **four complete training runs**, each 30 epochs: 630M and 1.54B on
F0/seed 101001 and F2/seed 101002. The paired widths use the same split per fold.
Four W2 jobs occupy all eight B300s. Per-rank batch 8 preserves the original
global batch 16; eight accumulated time chunks are not additional independent
sequence slots. Learning rate, schedule, Muon/AdamW settings, masks, fitting-only
normalization, architecture, 512-row TBPTT and exact 1024-row local causal window
are unchanged. Each fit has 112,050 updates on F0 or 112,200 on F2: 448,500 total.

The selected compute mode is BF16 model operations, FP32 master parameters,
gradients, loss and optimizer state, with process-global cuBLAS BF16x9 FP32
emulation, efficient SDPA, block compilation and activation recomputation off.
The same mode applies to endpoint export, whose output arrays are FP32.
BF16x9 dispatch was confirmed by actual `9xbf16gemm` CUDA kernels. This retains
FP32 tensor interfaces and can accelerate Muon matrix work; it is not an
IEEE-identical FP32 reproduction. See [NVIDIA's emulation documentation](https://docs.nvidia.com/cuda/cublas/index.html).

Representative single-GPU profiles used the real 1.54B/F0 data, batch 8 and six
optimizer updates. They are engineering measurements, not completed experiments:

| Mode | Attention | Recompute activations | Loop seconds |
|---|---|---:|---:|
| FP32 | math | yes |65.21|
| FP32 | math | no |52.35|
| TF32 | math | no |26.49|
| BF16 | math | no |29.74|
| BF16 | efficient | no |24.01|
| BF16 + BF16x9 | efficient | no |21.25|

FP32-emulation-only speed was shape-dependent: it improved batch 2 but regressed
at larger batches with math attention. The final selection uses measured whole
updates and four-fit completion time. GPU memory occupancy is not the objective.
Original W8 strict-FP32 profiles measured 47.57 seconds/12 updates for 630M and
65.77 seconds for 1.54B, with the latter reserving 28.39 GiB/GPU.

Final source-bound W2 profiles ran 12 real-data updates per cell, with no errors,
NaNs or compiler-limit fallback. Timing uses the median of aligned per-update
rank maxima after warmup; initialization/compilation are recorded separately.

| Cell | Steady seconds/update | VRAM reserved/GPU | Estimated fit hours |
|---|---:|---:|---:|
|t1280_f0|1.852|29.50GiB|57.66|
|t1792_f0|2.547|58.06GiB|79.28|
|t1280_f2|1.870|29.50GiB|58.28|
|t1792_f2|2.548|58.06GiB|79.42|

The packet estimate is approximately 80 hours of steady fitting and **3–4 days
wall time**, pending calibration from completed epochs. This is a measured-shape
projection, not an observed full-run duration. Allocation: 32 logical CPUs,
512 GiB planned host-memory headroom, approximately 150.24 GiB conservative
staging-plus-profile-RSS estimate, and 200 GiB disk reserve. Hardware inspection
found eight 275040 MiB B300s, NVSwitch, 256 logical CPUs and approximately 2 TiB RAM.
Runtime: Python 3.12.14, PyTorch 2.14.0+cu130, CUDA 13.0.

Numerical checks used the actual 1.54B model/batch 8/chunk 512 path. Future
perturbations left prior predictions exactly unchanged; parameters, gradients,
KV caches, loss and optimizer state passed the declared FP32/finiteness checks.
Compiled/eager synthetic gradient cosine was 0.999995, relative L2 difference
0.00326. Both were similarly close to the same-weight FP32 reference. On the
four actual 12-update comparisons, maximum relative training-loss and preclip-norm
differences were0.000131 and0.000113.

These are execution checks, not quality-equivalence evidence. In particular,
one synthetic Muon update differed by relative L2 0.0877 despite cosine 0.9962;
the update transform is sensitive to small gradient changes. Checkpoint lineage,
precision and compilation remain explicit, and full endpoint validation will
determine teacher quality. No public score or distillation benefit is inferred.

Two implementation corrections were necessary. The legacy optimizer-resume test
needed a deep-copied state fixture; training math was unchanged. The first full
startup on `ad876fa0` then failed admission **before update 1**: live optimizer
identity contained tuples while JSON profiles contained lists. Commit `f1beeb4c`
normalizes metadata at the producer boundary, preserving live optimizer tuples
and the unchanged exact admission validator. A regression covers JSON roundtrip
identity. Final host checks passed **77 tests**, and an additional real two-rank
full-mode admission check passed before entering the training function. The
failed startup's logs and receipts are retained.

Initial block compilation also hit the default recompilation limit. The repaired
path builds an exact additive mask without tracing a mutable Python capability
cache. A bounded 64-specialization limit accommodates KV lengths, row tails,
final partial batches and evaluation; compiler settings are part of identity.
Compiler logs retain graph breaks and recompilations. Fused floating-point
rounding is an explicit part of this arm.

The finite four-job controller runs under a native cron PAM session, with
source/config/split/profile checks before each phase, CPU affinity, explicit GPU
UUIDs, independent per-job deadlines and cleanup restricted to its recorded
process groups. Each job runs fit → held-out prediction → official scoring,
then releases its GPUs. Safe epoch checkpoints and rank logs stay on physical
storage. The seven-day per-job failsafe bounds unexpected stalls; it is separate
from the 3–4 day estimate. Startup is claimed once; failures do not trigger an
automatic retry loop.

The PR progress publisher runs every five minutes locally, writes hourly updates
and updates after evaluated completions or terminal failures, as SemyonEpanov.
Profiles and startup failures never count as scientific completions. Raw
engineering receipts/configs/logs are preserved on the
[evidence branch](https://github.com/VladKochetov007/WunderFunder/tree/experiment/pr32-b300-evidence-20261024/research/experiment_drafts/b300-transformer-distillation-runtime/execution_20261024),
with archive/file hashes. Production run and admission roots are
`/data/epanov_s/tabdata/runs/pr32-teachers-20261024-r2` and
`/data/epanov_s/tabdata/runs/pr32-production-admission-20261024-r2`.

The subsequent r4 batch cohort now has an external
[intermediate validation monitor](validation_monitor/README.md), added at user
request on October 24. It reports same-checkpoint train-probe and full held-out
losses plus official WP from retained epoch snapshots. The first available
snapshots are small-model epoch28 and large-model epoch15. The original fixed
epoch30 endpoint and immutable scientific source remain unchanged.
