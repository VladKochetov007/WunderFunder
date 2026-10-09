# E520 V9 CUDA/NVML memory-domain profile candidate

This source-only successor addresses V8's measured observer abort. V8 bound the
CUDA device to physical UUID `GPU-2f8814a9-d330-e294-0068-7d7cac59cd21`, but
CUDA exposed 15,636,037,632 usable bytes while NVML reported 16,106,127,360
physical bytes. The 470,089,728-byte difference is an observed capacity-domain
difference, not a different-device identity. V8 stopped before any trainer
profile batches; its input hashes and fold identity passed, and no fit, export,
grade, or score ran.

V9 changes only the observer's device-memory reconciliation. It requires the
exact selected UUID and `0 < CUDA free <= CUDA usable total <= NVML physical
total`, with no tolerance. CUDA property and `mem_get_info` totals must still
agree exactly with each other. NVML free is physical total minus NVML memory
used. The resource gate uses the smaller observed CUDA/NVML free amount for
the unchanged 6 GiB free floor. When the process table omits this PID, the
upper bound is the larger of physical total minus CUDA free and physical total
minus NVML free. That is at least physical total minus CUDA free and remains
subject to the unchanged 3,000,000,000-byte allocation cap. Missing, malformed,
nonpositive, inverted, or UUID-mismatched observations abort before profiling;
empty PID output remains unavailable rather than zero.

The raw train/valid hashes, verified digest handoff, fold/membership checks,
raw-only GRU128x2 seed-3 recipe, 3 warmup plus 20 timed batches, inference
chunk probe, and RSS/VRAM/headroom/disk guards remain unchanged. The worker
wall cap is 1,700 seconds and provisional quota reservation is at most 3,400
seconds under root's V9 conditional allocation. Actual quota, provider status,
mounted input identity, device identity, and free memory are null until a fresh
pre-push admission. This candidate does not authorize a packet build or launch.

No packet was assembled, no provider call was made, and no fit, profile,
prediction, grade, or score was run while preparing this source candidate.

Bounded local checks (one CPU, 1 GB, 120 seconds):

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python -m pytest -q \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/physical_memory_domain_v9_candidate_v1/tests

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/ruff format --check \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/physical_memory_domain_v9_candidate_v1

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/ruff check \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/physical_memory_domain_v9_candidate_v1

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/physical_memory_domain_v9_candidate_v1/prepare_profile_packet.py \
--review-targets-only --repo-root .
```
