# PR32: measured batch continuation, R4

The user explicitly requested larger batches on the eight B300s, then asked to
admit the large model's approximately 13% VRAM headroom and test batch 128 for
630M. This supersedes the operator's initial conservative 15% margin. R4 selects
**96 sequences/GPU for 630M and 64/GPU for 1.54B**, with two GPUs per fit. Global
sequence batches are 192 and 128 respectively. Four fresh fits retain 30 epochs,
original seeds, source-qualified F0/F2 splits, labels-only objectives, causal
1024-row attention, 512-row TBPTT and eight accumulated time chunks.

Scientific source remains pinned to
`f1beeb4c210a9da01735e03b1ff6e27be878b873`. The numerical arm remains BF16 model
compute with FP32 master parameters, gradients, losses and optimizer state,
process-global cuBLAS BF16x9 FP32 emulation, efficient attention, compiled blocks
and activation recomputation off. This is not an IEEE-identical FP32 replication.

## Measurements and selection

Each candidate ran 12 real-data optimizer updates at W2 on the receiving host.
Warmed measurements use steps 6–12, aligned by update and timed by the slower DDP
rank. Row rates use globally reduced forwarded-row counts. Profiles are
engineering work, not completed scientific fits. No held-out scores selected a
batch. Source/config/split/GPU identities and raw profile/operator/log hashes are
bound in `receipts/pr32-batch-r4-selection.json` on the receiving host.

| Teacher | Batch/GPU | Warm rows/second | Peak reserved VRAM/GPU |
|---|---:|---:|---:|
|630M|8|35,553|29.50GiB|
|630M|16|66,580|40.64GiB|
|630M|32|100,400|67.97GiB|
|630M|64|112,639|131.48GiB|
|630M|96|116,930|196.41GiB|
|630M|128|113,551|260.80GiB|
|1.54B|8|25,355|58.06GiB|
|1.54B|16|44,631|77.92GiB|
|1.54B|32|56,699|125.85GiB|
|1.54B|64|60,989|232.25GiB|

All ten candidates completed12 updates without OOM or reported optimizer-state
errors. Batch 128 fits the small model but is approximately 2.9% slower than 96
and leaves only about 6 GiB free. The selected small batch is fastest among the
measured candidates even without a headroom exclusion. Large-model B64 was
repeated on both actual production folds: approximately 61,074/60,982 rows/s,
231.41GiB and13.55% headroom, with no errors.

R4 permits candidates with at least 10% measured VRAM headroom. Within 3% of the
maximum rate, it prefers a smaller batch. This is an engineering admission
choice, not a hardware requirement or proof that every untested shape fits.
The final four-cell profile separately binds the selected recipe and actual
production GPU order. Initial engineering directories named `t*_f2` in the first
two batch rounds are aliases for F0/B16 or F0/B64. In the extension round,
`t1280_f2` aliases F0/B96; both large-model directories are actual F0/F2 cells.
These aliases do not count as fold replications. Final selected-cell profiles
use the actual F0/F2 recipes and seeds.

## CPU and transfer cost

The canonical fitting dataset, `collate_sequences` and
`move_and_validate_batch` were measured independently on one PyTorch/OMP/MKL
thread, with three distinct real fitting batches per size and synchronized
host-to-device timing. The isolated diagnostic used CPUs 64–71 and GPU0 after
the previous GPU profiles exited; temporary device tensors were below 2 GiB.

| Batch/GPU | Median CPU assembly | Median transfer | Tensor payload |
|---|---:|---:|---:|
|64|1.428s|0.051s|595,200,000B|
|96|2.052s|0.083s|892,800,000B|
|128|2.845s|0.101s|1,190,400,000B|

Each whole-sequence batch supplies five optimizer windows. In the actual
four-job R3 confirmation profiles, time outside logged update windows,
including loading, transfers and logging, was approximately 6.7% for B64 small
and 3.5% for B32 large. It is not zero and is included separately from compute
timing; the isolated diagnostic is not an eight-rank loader benchmark. The raw
CPU receipt is `batch-loader-b64-128-r1.json` in the campaign run root.

After the full R4 fits started, 20 consecutive one-second-paused NVIDIA queries
measured mean activity of 98.2–99.7% across the eight GPUs. The raw receipt is
`LIVE_GPU_ACTIVITY.json` in the R4 admission root and the verified evidence
archive. This is a short activity sample, not a measurement of peak FLOPS or MFU.

## Scientific scope and durable execution

R2 was deliberately stopped with 1,842 aggregate successful updates out of 448,500
(approximately 0.41%), zero completed fits and zero official endpoint scores.
Its logs, existing safe checkpoints and supersession receipt remain intact.
R3 selected-batch profiles were engineering checks only; no R3 full cohort ran.
R4 starts from the original seeds rather than resuming the changed R2 schedule.

R4 keeps learning rate 3e-4, weight decay, clipping, Muon/AdamW settings and the
5% warmup/cosine policy. It recalculates the update horizon for the new global
batches: 9,450 updates per small fit and 14,100 per large fit, 47,100 total. Thirty
epochs preserve fitting-data coverage; fewer optimizer updates are a changed
optimization regime. Width and batch now vary together, so endpoint scores
compare complete teacher recipes and do not isolate an architecture-size effect.
Neither the engineering loss checks nor throughput establish teacher quality,
student-transfer benefit or public-score improvement.

The receiving controller validates the exact batch amendment, its own hash,
source/config/data/split/profile identities, disjoint GPU UUIDs and CPU affinity.
Its seven focused stdlib tests pass. The unchanged scientific source previously
passed 77 tests. A separate validator recomputed all ten candidates and the
selection from raw receipts and rank logs. Full fits run under a native cron
PAM session, outside the SSH lifecycle, with bounded per-job deadlines, safe
epoch checkpoints and cleanup restricted to owned process groups.

No OOM occurred in the candidate or receiving-host confirmation profiles. If a
fit fails, the controller stops and releases only its owned GPU pair while the
other fits continue. Logs and existing safe epoch checkpoints remain. There is
no automatic batch reduction or retry. Recovery must preserve the exact resume
contract or declare a newly profiled recipe; changing batch silently is rejected.

R4 admission root:
`/data/epanov_s/tabdata/runs/pr32-production-admission-20261024-r4-batch`.
Full-fit root:
`/data/epanov_s/tabdata/runs/pr32-teachers-20261024-r4-batch`.
The receiving plan and RESOURCE_ESTIMATE are indexed alongside this note. The
four-cell warm fitting projection is 33.64 hours; adding measured time outside
update windows gives 34.85 hours. Cold startup/compilation, checkpoints and final
export/evaluation remain additional, so this is not an observed full duration.
Thirty-epoch fitting has now been admitted through the independent cron session;
the live status and successful-update counters are published in the
[PR32 progress comment](https://github.com/VladKochetov007/WunderFunder/pull/32#issuecomment-6377032958).
It identifies the actual attempt, updates hourly and after evaluated completions or
terminal failures, and separates optimizer progress from completed endpoints.
