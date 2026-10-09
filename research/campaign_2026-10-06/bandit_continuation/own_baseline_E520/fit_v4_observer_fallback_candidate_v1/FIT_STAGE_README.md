# E520 scratch-pair fit v4 observer candidate

This is a code-only successor to the independently reviewed v3 fit worker. It
repairs GPU resource observation and monitor startup after v7's first active
check aborted because nvidia-smi did not list the worker PID. It does not
authorize a packet build, fit, export, score, or provider call.

## Frozen science and execution scope

The scratch recipe, source archive, input identities, digest handoff, checkpoints,
streamed prediction exports, E511 anchor, E520 grader and thresholds are inherited
unchanged from fit-v3. The candidate still means raw-only scratch GRU128x2,
seeds 3 and 4 serially, four epochs, fixed zero-based epoch 3, chunk-normalized
WMSE with pinned training-target clipping, batch 16, chunk 2000, fp16 and
OneCycle/AdamW. Export remains a streaming float32 memmap in 4000-row chunks.
Fold 0 remains exposed development evidence only.

The observer-only delta is:

| Area | v3 | v4 candidate |
|---|---|---|
| Process allocation | Required a matched nvidia-smi PID row | Uses PID bytes when reported; otherwise requires matching CUDA total/free values and enforces total minus free as a conservative device-wide upper bound |
| GPU evidence | PID output and a short identity row | Bounded stdout/stderr, return code, full physical GPU UUID/memory/utilization inventory and selected-device/CUDA total agreement |
| Startup monitor | Changed the existing pre-CUDA guard to active after CUDA init | Stops the explicit-idle poller, durably records the handoff, persists runtime identity, then starts a distinct active guard |
| Synchronous failure | Could raise before saving the failing sample | Saves resource snapshot first, then atomically writes the exact failed snapshot, raw process query and guard reason before raising |

An empty process table never becomes zero. Missing PID is accepted only when
CUDA reports an initialized device, total and free memory agree, device-wide
usage is at most 3,000,000,000 bytes, and free memory remains at least 6 GiB.
If this evidence or the selected physical-device UUID/memory inventory is
unavailable or inconsistent, the worker aborts without retry. Allocated VRAM
remains capped at 2,000,000,000 bytes, reserved VRAM at 2,500,000,000 bytes,
RSS at 8,000,000,000 bytes, and host/cgroup/disk gates retain their v3 values.

The transition writes resource_monitor_handoff.json with pre-CUDA stopped,
active startup, and active-running phases. environment.json contains the raw
physical inventory query and its digest/truncation metadata. resource.json is
written before every full-sample gate; resource_abort.json binds the exact
sample and environment receipt if a synchronous or polled gate fails.

## Validation and exact review targets

The focused synthetic test module covers conservative total-minus-free fallback,
over-cap abort, raw query error retention, selected-device UUID/total binding,
synchronous failure receipt persistence, and pre-CUDA-to-active monitor order.
The builder's review-target-only command prints hashes and creates no packet:

    WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
    WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
    scripts/limited .venv/bin/python \
      research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v4_observer_fallback_candidate_v1/prepare_fit_packet.py \
      --review-targets-only

Focused tests and style checks are bounded as follows:

    WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
    WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
    scripts/limited .venv/bin/python -m pytest -q \
      research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v4_observer_fallback_candidate_v1/tests

    WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
    WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
    scripts/limited .venv/bin/ruff check \
      research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v4_observer_fallback_candidate_v1

    WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
    WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
    scripts/limited .venv/bin/ruff format --check \
      research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v4_observer_fallback_candidate_v1

No packet build/upload, provider action, fit, prediction export, or score has
run as part of this candidate. The v3 source/review bytes remain in their
original directory. This v4 candidate requires its own independent source,
recipe/grader, and transport review plus fresh root admission before use.
