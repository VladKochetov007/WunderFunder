# E520 scratch-pair fit v5 memory-domain observer candidate

This code-only successor preserves fit v4 and fixes the same measured V8
observer abort in the serial scratch-fit worker. V8 observed the selected
physical UUID `GPU-2f8814a9-d330-e294-0068-7d7cac59cd21`, CUDA usable memory
15,636,037,632 bytes, and NVML physical memory 16,106,127,360 bytes. The
470,089,728-byte difference is a capacity-domain distinction for the same
UUID; v4's equality check stopped before training.

## Scope

The observer requires the exact selected UUID and `0 < CUDA free <= CUDA
usable total <= NVML physical total`, with no tolerance. CUDA property and
`mem_get_info` totals still must agree exactly. It derives NVML free from
physical total minus memory used, uses the smaller CUDA/NVML free value for
the unchanged 6 GiB floor, and uses the maximum of physical total minus CUDA
free and physical total minus NVML free as the no-PID upper bound. The latter
is at least physical total minus CUDA free. Empty process tables remain
unavailable, never zero. PID allocation remains capped at 3,000,000,000 bytes;
CUDA allocated/reserved, RSS, cgroup/host headroom, disk, runtime, recipe,
checkpoint, prediction exporter, grader, anchor, fold, and digest handoff are
unchanged.

The batch-boundary sampler refreshes CUDA allocator/free values but carries
forward only the latest full resource sample's validated UUID/NVML domain.
Its use is bounded by the existing three-second full-sample interval; an
older sample triggers a new full observation before it can pass a gate.
Unavailable, malformed, stale, mismatched, or inverted memory domains abort
before or during fit and persist the exact resource snapshot.

| Area | Fit v4 | Fit v5 candidate |
|---|---|---|
| Device identity | UUID plus CUDA/NVML total equality | Exact same UUID; separate CUDA usable and NVML physical domains |
| Free-memory gate | CUDA free only | Minimum of CUDA and NVML free; same 6 GiB threshold |
| Missing PID | CUDA total minus CUDA free | Conservative physical-domain maximum bound; same 3 GB process threshold |
| Recipe and outputs | Frozen scratch pair | Unchanged |

This candidate requires independent source/recipe/grader/transport review and
fresh root admission. It is not a packet, and it does not authorize a provider
call, fit, export, score, or public submission. No provider action or model
execution occurred while preparing it.

Bounded checks (one CPU, 1 GB, 120 seconds):

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python -m pytest -q \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v5_cuda_nvml_domain_candidate_v1/tests

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/ruff format --check \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v5_cuda_nvml_domain_candidate_v1

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/ruff check \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v5_cuda_nvml_domain_candidate_v1

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v5_cuda_nvml_domain_candidate_v1/prepare_fit_packet.py \
--review-targets-only
```
