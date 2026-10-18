# F1 result collection preparation

Status: **PREPARED_NOT_ADMITTED**. This is a local preparation packet only. No
result snapshot has been collected, no exports or grade were run by this helper,
and the code has not yet received the root's independent review or admission.
The live frozen F1 sequence owns training, epoch-5 exports, and grading; do not
run a second export or grade to satisfy this packet. Wait for the deep terminal
checkpoint to finish and pass its structural verification, then wait for the
frozen sequence's exports and paired grade to finish. Root must prepare a
reviewed arrival seal from the exact receipts before this collector is used.

The collector validates and snapshots existing evidence; it does not train,
export, grade, deserialize checkpoint tensors, or open target parquet. It is
bounded to a 256 MiB snapshot, 4 MiB JSON, 128 KiB sidecars, 16 MiB evidence
files, and 128 MiB prediction arrays. It streams hashes and NPY finiteness
checks, uses project-local no-follow paths, and publishes create-only output.
Keep the collector inside the campaign slice with a 75% CPU quota while the
1.1-CPU fit controller is active (1.85 CPUs combined under the 2-CPU campaign
cap). The bounded run also uses at most 512 MiB, no swap, hidden CUDA, physical
temporary storage, and a 120 second limit. These are admission requirements,
not evidence that the collector has run under them.

## Frozen evidence bindings

The collector accepts only rank 1, fold 1 of 4, seed 1001 and the exact F1
base plan (`96cfb67b3447866651ae04638166f17dccb9729b3d4a9340f1b72b8d7b089f96`),
recovery plan (`f1aa897f0be902e114935f0ac9044b0616e2f3e3a889991651d028823938a111`),
and fold manifest (`d8e9748d770bb3af525943b36b1801e6fd72d78259cc6b823d4661ffd78367d9`).
The wide continuation uses the approved attempt ID
`computer-vision-homework-r01-wide-f1s1001-ep25-v2`; it reads the attempt
record by that ID from the runner's `runtime/attempts/` directory rather than
following the frozen plan's documentary stale `receipt_path` for rejected v1.
The stale path is documented in the readiness audit and is not the runner's
attempt lookup behavior.

F1 has 457 evaluation row groups and 149 components. The expected export is
feature-only FP32 with NPY shape `[457, 20000, 2]`; the 73,120,000-byte raw
payload size is derived from that shape, not an observed artifact size. The
official grade metric is `global_wp_clip2_v1`, with 10,000 paired component
bootstrap resamples. Fit schedule values are static metadata-derived checks;
the collector additionally requires the actual terminal receipts' observed
counters and does not infer convergence from the six-epoch horizon.

Export sidecars are checked against the canonical ordered
`fold1.evaluation.valid` row groups and their membership hash. That export
ordering is distinct from RNG-ordered fit membership and the training-time
evaluation order recorded in checkpoint provenance; the collector deliberately
does not equate these different identities.

## Arrival seal and invocation

The deep terminal receipt hash is intentionally not guessed or frozen here.
Before use, root must create a separate reviewed, create-only
`priority25.f1_terminal_evidence_seal.v2` JSON with `status: "sealed"` and
`review_status: "passed"`. It must bind the exact base/recovery/fold hashes,
both arms' stage1 and terminal collection+verification receipt hashes and
schemas, their helper/dependency maps, the exact current collector module
hashes, the V2/V4 checkpoint support manifests, and both independent passed
review receipts. The validator checks those references against the actual
receipt bytes; a seal is not a substitute for this validation.

After independent review/admission, invoke the existing CLI from the project
checkout with paths to the actual receipts and output names that do not exist.
This is an invocation template, not an authorization or a command to run now:

```sh
systemd-run --user --wait --collect --unit=computer-vision-homework-r01-f1-results.service \
  --property=Type=exec \
  --property=Slice=wnn-campaign.slice \
  --property=CPUQuota=75% \
  --property=MemoryMax=512000000 \
  --property=MemorySwapMax=0 \
  --property=RuntimeMaxSec=120 \
  --property=TimeoutStopSec=10 \
  --working-directory=/path/to/project \
  --setenv=CUDA_VISIBLE_DEVICES= \
  --setenv=OMP_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 \
  --setenv=TMPDIR=/path/to/project/research_tmp \
  /path/to/python \
  research/campaign_2026-10-18/priority25_execution_r4/f1_result_collection_r1/collect_f1_results.py \
  --project-root /path/to/project \
  --base-plan research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/F1_BELERIAND_PLAN.json \
  --recovery-plan research/campaign_2026-10-18/priority25_execution_r4/f1_recovery_r1/F1_RECOVERY_PLAN.json \
  --fold-manifest /path/to/sealed/fold1.json \
  --terminal-seal /path/to/reviewed/F1_TERMINAL_EVIDENCE_SEAL.json \
  --wide-stage1-collection /path/to/wide/stage1/collection.json \
  --wide-stage1-verification /path/to/wide/stage1/verification.json \
  --wide-terminal-collection /path/to/wide/terminal/collection.json \
  --wide-terminal-verification /path/to/wide/terminal/verification.json \
  --deep-stage1-collection /path/to/deep/stage1/collection.json \
  --deep-stage1-verification /path/to/deep/stage1/verification.json \
  --deep-terminal-collection /path/to/deep/terminal/collection.json \
  --deep-terminal-verification /path/to/deep/terminal/verification.json \
  --snapshot-dir result_snapshots/rr18_r01_f1_s1001_e6 \
  --receipt receipts/RANK01_F1_S1001_RESULT_COLLECTION.json
```

Before admission, check the actual export/grade attempt records and outputs,
including both `.npy` and `.json` sidecar collision paths. The helper requires
successful exact-command attempt records, both terminal verification receipts,
the arrival seal and passed reviews, matching grade/export identities, stable
source copies, and absent create-only output paths. It does not impose a
quality threshold or CP30 gate.

## Local checks and limitations

Focused local tests use small synthetic metadata/NPY fixtures; they do not use
the real F1 arrays, checkpoints, parquet, or Torch. Run from this directory:

```sh
python -m unittest -v test_f1_result_collection.py
uv run --no-sync ruff check .
```

The preparation manifest records the exact helper/test hashes and the last
local check outcome. Until a reviewed seal exists and root admits a bounded
run, this packet provides no F1 result snapshot, no score, no evidence of
export/grade completion, and no claim of convergence, resumed/uninterrupted
parity, CP30, serving performance, or public transfer. The current deep
terminal SHA remains unknown and must be bound only when its actual receipt is
available.

The independent implementation review is recorded in
[INDEPENDENT_REVIEW.json](INDEPENDENT_REVIEW.json). The review found no
blocking code defects, and the focused synthetic suite passed all eight tests.
The packet remains
**PREPARED_NOT_ADMITTED** until the deep terminal receipt arrives, its exact
SHA/schema and dependencies are reviewed, and a sealed arrival receipt exists.

See the [readiness audit](../receipts/F1_RESULT_COLLECTION_READINESS_20261018T1240Z.json)
for the prior evidence inventory and the [runbook](../RUNBOOK.md) for current
ownership and operational constraints.
