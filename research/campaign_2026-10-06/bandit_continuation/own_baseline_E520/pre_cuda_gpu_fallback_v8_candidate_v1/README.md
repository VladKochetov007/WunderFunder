# E520 v8 startup GPU-observation repair candidate

This isolated profile-only candidate addresses the exact v7 abort recorded in
[`E520_PROFILE_V7_PHASE_DIAGNOSIS_20261007T1317Z.json`](../E520_PROFILE_V7_PHASE_DIAGNOSIS_20261007T1317Z.json).
The v7 worker verified raw train/valid identities and fold membership, then
aborted at its first synchronous post-CUDA resource check because the
`nvidia-smi` compute-app query did not report a PID allocation. That check
raised before its snapshot was persisted. The terminal receipt records no
timed profile batches, no fit, and no score; it classifies this as a resource
observation failure. Its reported quota delta was 413.308 seconds for 404.543
seconds of worker wall, and physical GPU inventory was unavailable.

The v8 copy keeps the input hashes, split/fold checks, pinned source archive,
raw-only GRU128x2 seed-3 profile, 3 warmup plus 20 timed outer batches,
inference-export probe, external-review gate, and all RAM, VRAM, free-memory,
disk, and headroom thresholds. It changes only GPU observation persistence and
the shorter root-directed worker wall cap: 1,500 seconds, with a maximum
conservative quota reservation of 3,000 seconds. At the last reported E520
balance of 3,109.310 seconds that leaves 109.310 seconds. These are planning
figures only; provider status, quota, actual hardware, and available resources
must be refreshed before any later packet admission.

After CUDA becomes available, the worker first persists an environment
identity containing the actual Torch/CUDA model and the complete physical
`nvidia-smi` inventory, including UUID and memory totals. It also persists the
initial resource snapshot before calling the active guard. If the process table
reports a PID, its allocation remains subject to the unchanged 3,000,000,000
byte cap. If the PID is missing or the process query fails, the worker does not
substitute zero: it uses `device total memory - CUDA-reported free memory` as a
conservative device-wide usage upper bound and requires that bound to remain at
or below 3,000,000,000 bytes, with at least 6 GiB free. Missing or inconsistent
device measurements abort before profile. A physical inventory failure also
aborts before profile; the worker does not assume a T4 or a physical GPU count.

Both `nvidia-smi` commands retain bounded raw stdout/stderr, return code,
status, byte lengths, SHA-256, and truncation flags. The active guard writes
each resource snapshot before enforcing it, and the synchronous failure path
atomically writes both the exact snapshot and abort reason. The existing
pre-CUDA idle state is accepted only while no CUDA allocator observation is
active. At the CUDA handoff, the worker stops its pre-CUDA poller before Torch
initialization, persists and gates one active startup snapshot, then starts a
new poller. This closes the race where a delayed background check could mark
an already initialized CUDA process as idle.

The candidate has no packet, transport bundle, or review approval. It has not
been sent to Kaggle and has not run a model, profile, fit, prediction, or score.
External review and fresh root admission are required before an operator may
build or launch anything.

Focused local validation commands, each bounded to one CPU, 1 GB memory, and
120 seconds:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python -m pytest -q \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/pre_cuda_gpu_fallback_v8_candidate_v1/tests

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/ruff format --check \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/pre_cuda_gpu_fallback_v8_candidate_v1

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/ruff check \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/pre_cuda_gpu_fallback_v8_candidate_v1

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/pre_cuda_gpu_fallback_v8_candidate_v1/prepare_profile_packet.py \
--review-targets-only --repo-root .
```
