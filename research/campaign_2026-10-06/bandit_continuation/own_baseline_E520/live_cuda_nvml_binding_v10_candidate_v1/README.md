# E520 profile V10: live CUDA-to-NVML identity binding

V10 is a new isolated profile-only successor to V9. V9 completed mounted-input
hashing and fold checks, then stopped before trainer profile because its worker
compared the live CUDA selection with a UUID pinned from an earlier Kaggle
session. The V9 terminal diagnosis records the selected T4 as
`GPU-c6747be0-d0ca-9158-e195-5f8b14866cbd` and the stale expected UUID as
`GPU-2f8814a9-d330-e294-0068-7d7cac59cd21`.

V10 resolves the one `CUDA_VISIBLE_DEVICES` token against the full NVML
inventory at startup: numeric tokens match the exact physical index, while UUID
tokens match exact UUID. It requires one unique complete row, records the
selected index and UUID, and carries that UUID as worker state. Every later
resource snapshot finds that same unique UUID in the fresh physical inventory;
missing, duplicate, or changed identity fails closed. It never infers identity
from row order or persists a UUID across Kaggle sessions.

The physical-versus-CUDA memory-domain ordering, conservative device-wide
upper bound, 6 GiB free floor, 3 GB device-use limit, 2 GB allocated limit,
2.5 GB reserved limit, 8 GB RSS/disk limits, host/cgroup headroom, and all
input, digest-handoff, fold, and profile semantics are unchanged. The root
successor allocation caps this single profile attempt at 1,450 worker-wall
seconds and 2,900 provisional quota seconds, within the 9,000-second cumulative
recovery ceiling and retaining 241.553 seconds after the maximum reservation.

No packet, provider call, profile, fit, export, grade, or score has been run for
V10. Independent exact-source review and static transport review are required
before a fresh live-quota admission and one guarded private push. Fitting is not
included in this allocation.

Run focused checks in the campaign resource slice:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python -m pytest -q \
  research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/live_cuda_nvml_binding_v10_candidate_v1/tests

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/live_cuda_nvml_binding_v10_candidate_v1/prepare_profile_packet.py \
  --review-targets-only --repo-root .
```
