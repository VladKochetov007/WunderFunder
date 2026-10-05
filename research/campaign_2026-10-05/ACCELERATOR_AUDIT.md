# Accelerator resumption audit — 2026-10-05

Read-only provider and host checks captured 2026-10-04 22:36 UTC (2026-10-05
01:36 Europe/Kyiv; corrected from the initial UTC+2 conversion). No accelerator
kernel was pushed and no fit or accelerator probe was started. The later local
CPU-only wrapper fixture is recorded below. API credentials were not printed.

## State now

- **Local GPU:** `nvidia-smi` reports one GeForce RTX 3060 Laptop GPU, 6,144 MiB
  total and 75 MiB used. The compute-app query returned no process. The reported
  58% utilization may include the desktop and should be rechecked over a short
  quiet interval before any trainer. The runner pause record remains present;
  `systemctl --user` showed the marimo notebook and resource watcher, no WNN
  training service.
- **Kaggle GPU quota:** typed `KaggleApi.quota_view()` reports 27,856.206 of
  108,000 seconds used, 0 reserved, **80,143.794 seconds (22.2622 hours)
  remaining**, refreshing 2026-10-10 00:00 UTC.
- **Kaggle TPU quota:** 134.623 of 72,000 seconds used, 0 reserved,
  **71,865.377 seconds (19.9626 hours) remaining**, same refresh. The currently
  queued TPU kernel has no observed assignment or charge movement in the latest
  saved snapshots; this is not a guarantee it will remain unassigned.
- **E463, the only actual TPU job:** a fresh `kaggle kernels status` still
  reports `KernelWorkerStatus.QUEUED` for
  `vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1-runner`. It has no
  terminal result. Its approved upload inventory is unchanged: private
  TpuV5E8, TPU enabled, GPU/internet disabled, 4,200-second timeout. This is a
  one-device real-data update plus inventory/objective preflight; it is not an
  eight-device training result.
- **W384 remote GPU:** v4 is terminal `ERROR`. Its preserved audit records
  `Unsafe external input mapping` after 1.703 seconds, before phase runner,
  probe, fit, predictions, or score; 0 GPU steps and no OOM. Root cause: the
  wrapper validated 26 intentionally external mount paths against the
  archive-only member allow-list.
- **W384 v5:** only a local held draft exists. The current provider status
  request for the v5 slug is denied/not found, consistent with the saved pause
  record (`W384_v5_pushed: false`). Metadata requests a private NvidiaTeslaT4,
  GPU enabled and TPU/internet disabled. Its worker reuses the v4 scientific
  archive SHA-256 `303dad1b…49cc6338` and the same declared 26 external mounts.
  The independent implementation review and all eight final mount-guard
  fixtures now pass under a 2-core, 2-GiB `scripts/limited` scope. Review
  confirms exact equality of all 26 external descriptors and the immutable
  bundle manifest, and that v5 changes the transport wrapper while retaining
  the pinned scientific archive. It is ready for a controller-owned push under
  the fresh v5 slug; it has not been pushed. See
  [W384_V5_REVIEW_RECEIPT.json](W384_V5_REVIEW_RECEIPT.json).
- **E465 TPU:** no E465 TPU kernel is queued or running. The full-fold
  eight-device FP32/BF16 continuation is a source-only draft. CPU correctness
  evidence and OneCycle schedule parity are recorded as passed; production
  eight-device objective execution, full-shape memory/throughput, and fit are
  still unverified. E465 explicitly depends on terminal E463 evidence and its
  own all-eight-device profile before fit.

The conservative campaign accounting at pause was at most 26.513 device-GPU
hours against an 80-hour ceiling (at least 53.487 hours remaining). Local block
accounting showed 11.0373 of 24 charged GPU-job hours, 12.9627 hours remaining,
and no active local GPU attempt. Reconcile these saved counters against new
ledger entries before admitting jobs; Kaggle provider quota is a separate
budget from device-GPU hours.

## Recommended order

1. Keep the existing E463 TPU identity as the only TPU job. Poll status and
   collect terminal output only. Do not submit E465 while E463 is queued; E463
   is the required TPU/data/objective integration gate.
2. Unblock the already prepared W384 experiment with a small wrapper-only
   review: compare the v5 kernel and metadata to the preserved v4 bundle
   manifest, run the stdlib mount-guard fixture, independently confirm the
   26-entry descriptor map equals the manifest, and confirm the archive hash
   remains `303dad1b…49cc6338`. If those checks pass, admit a fresh v5 identity
   with refreshed GPU quota/reserve and bounded wall time. Require both fresh
   production-shape T4 probes to pass before either fold-1 W256/W384 fit. Keep
   epoch 3 fixed and score both against the same paired rows; do not interpret
   the v4 startup error as a width result.
3. Use the local GPU for the existing E464 sealed head comparison only after
   its saved E425/E426/timing prerequisites and current `scripts/limited`
   guards are revalidated. Keep one trainer, 12 CPU cores, 36 GiB RAM cap and
   at most 5 GB total observed VRAM. The current laptop GPU has 6 GiB physical
   VRAM, so reject a run if sampled total device use approaches 5 GB.
4. After E463 completes successfully, continue E465 from the existing JAX
   staging code: verify its multi-device pooled-numerator/denominator and
   gradient tests on forced two-device CPU; bind an exact input manifest to
   E463; then run only the preregistered two-batch-per-precision,
   eight-device, full 20k-row/chunk-1000 profile. Admit the 7,460-update FP32
   and BF16 arms only if measured HBM, RSS, projected wall time, hashes, and
   distributed parity meet the preregistered limits. Reserve four provider
   TPU hours for recovery and cap both E465 arms plus setup/exports at eight
   provider hours.

This ordering uses the prepared experiments and existing model/JAX code. It
does not warrant another broad integrity-framework rewrite. The highest-value
immediate action is to unblock W384 v5's small transport review while E463
continues its existing queue wait.

## Reproduction probes (read-only unless explicitly noted)

Refresh local device state before each local reservation:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
systemctl --user list-units --type=service --state=running --no-legend | rg 'wnn|campaign|gpu'
```

Refresh Kaggle worker status without pushing or restarting anything:

```bash
.venv/bin/kaggle kernels status vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1-runner
.venv/bin/kaggle kernels status vladkochetov07/wnn-w384-fold1-dualt4-20261004-v4-runner
```

The v5 status lookup is expected to fail until a reviewed push creates that
provider kernel. Refresh typed quota without serializing or printing
credentials:

```bash
.venv/bin/python - <<'PY'
from kaggle.api.kaggle_api_extended import KaggleApi
q = KaggleApi().quota_view()
print("refresh_utc", q.quota_refresh_time)
for name, item in (("gpu", q.gpu_quota), ("tpu", q.tpu_quota)):
    used = item.time_used.total_seconds()
    reserved = item.time_reserved.total_seconds()
    allowed = item.total_time_allowed.total_seconds()
    print(name, "used", used, "reserved", reserved, "allowed", allowed,
          "remaining", allowed - used - reserved)
PY
```

The final W384 fixture probe was run in the bounded CPU scope recorded above;
to reproduce it:

```bash
.venv/bin/python research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v5_transport_review/mount_guard_fixtures.py
```

Primary receipts: `checks/COMPUTE_AT_PAUSE_20261004.json`,
`checks/PAUSE_DRAIN_COMPLETION_20261004.json`,
`kaggle/W384_fold1_dualT4_v4_transport/terminal_audit_v1.json`,
`kaggle/W384_fold1_dualT4_v5_transport_review/kernel_push/`,
`kaggle/jax_tpu_realdata_preflight_v1/provider_prelaunch_gate_v1.json`,
`E463_DRAIN_RESULT.md`, `E465_CPU_CORRECTNESS_REPORT.md`,
`E465_EIGHT_TPU_PRECISION_PREREG.md`, and `staging/jax_tpu_e465/`.

## TPU queue investigation and W384 review at 2026-10-04 22:45 UTC

E463 still reports `QUEUED`, with an empty failure message, 5h22m after its
17:23:26 UTC submission. The private metadata requests `TpuV5E8`, with TPU
enabled and GPU/internet disabled. Typed quota still has 19.9626 TPU hours
remaining and no reservation, so quota exhaustion does not explain the wait.
The provider status response exposes only `status` and `failure_message`; the
installed Kaggle API has no live accelerator inventory or queue-position field.
The prior eight-device probe completed on this same TpuV5E8 shape, but does not
prove current supply. No alternative TPU type is evidenced, so keep the queued
identity and E465's eight-device design unchanged. See
[E463_QUEUE_AUDIT.json](E463_QUEUE_AUDIT.json).
