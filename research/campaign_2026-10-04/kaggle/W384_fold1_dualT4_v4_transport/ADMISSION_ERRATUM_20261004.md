# W384 dual-T4 v4 admission erratum

This erratum corrects stale prose in `TRANSPORT_AND_BUDGET_DRAFT.md` and `budget_and_identity_draft.json`. It does not modify or replace the frozen kernel, archive, manifest, specs, launch plan, or their hashes. The packet remains an unadmitted source-only draft; this erratum is not authorization to upload or launch.

## Correct launch limits and budget gate

Use a **7-hour** provider timeout: 25,200 seconds. The in-kernel watchdog deadline is 25,180 seconds, leaving 20 seconds for timeout-record flush and cleanup. With two T4s allocated throughout the maximum run, the device-hours upper bound is **14 hours**, not 18 hours. The source packet's executable watchdog and provider timeout, launch plan, and runtime budget table already contain these values.

The stale statements in the draft that call this a 9-hour reservation/kill, require provider quota for 9 hours, or use `cumulative + 18 + 6 <= 80` are superseded. Admission must instead use a fresh typed provider quota and require provider remaining to cover the 25,200-second maximum plus any separately established provider reserve. Separately, require the latest root device-hour upper bound plus 14 planned device-hours and 6 recovery-reserve device-hours to be at most 80. The recorded 25.9486-hour campaign upper bound would yield `25.9486 + 14 + 6 = 45.9486` hours, subject to recomputing from the root ledger at admission.

The modeled phase total is 23,740.550 seconds (6.595 hours), with 1,459.450 seconds of margin to the 25,200-second provider limit. This is a planning estimate, not a guarantee. A worker's 12,600-second fit timeout applies to its complete two-step plan: four-epoch training followed by fixed epoch-3 full-validation prediction. The separate 1,200-second whole-job allowance is for paired scoring and output closure after both worker plans; it is not an additional prediction timeout.

## Fit semantics

The prose label “AdamW continuation” is incorrect. Each arm is a **fresh random-initialized AdamW fit** using `--train-source train+valid`, four epochs, and its fixed epoch-3 checkpoint. Its command has no initialization-checkpoint, resume, or optimizer-state restore argument. The two frozen teacher checkpoints are distillation inputs only. Replace “W256/W384 continuations” and “continuation” descriptions in the draft metadata with “fresh W256/W384 AdamW fits, paired under the fixed fold-1 recipe.” This wording correction does not alter the sealed scientific recipe.

## Exact provider commands and typed quota query

From the approved private kernel directory, the provider command shape is:

```bash
.venv/bin/kaggle kernels push -p research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4_transport/kernel_push -t 25200
```

The private metadata already declares the kernel identity and accelerator settings. The `-t 25200` option sets the provider runtime limit in seconds. Do not push until root independently admits the source packet and verifies dataset readiness, typed quota, and the device-hour ledger.

Query quota through Kaggle's typed client objects and use `timedelta.total_seconds()`; do not use `str(quota)` or a serialized duration, since those can omit the day component:

```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
quota = api.quota_view()
gpu = quota.gpu_quota
remaining_seconds = (
    gpu.total_time_allowed.total_seconds()
    - gpu.time_used.total_seconds()
    - gpu.time_reserved.total_seconds()
)
print({
    "gpu_total_seconds": gpu.total_time_allowed.total_seconds(),
    "gpu_used_seconds": gpu.time_used.total_seconds(),
    "gpu_reserved_seconds": gpu.time_reserved.total_seconds(),
    "gpu_remaining_seconds": remaining_seconds,
})
```

The last recorded snapshot in `budget_and_identity_draft.json` is historical and is not a substitute for a fresh query immediately before any future admission.
