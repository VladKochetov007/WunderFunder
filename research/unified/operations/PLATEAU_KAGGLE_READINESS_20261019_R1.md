---
kind: incident
status: complete
created_utc: 2026-10-18T22:41:00Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - ../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md
---

# Read-only Kaggle readiness snapshot

Operator: native Luna/max role `plateau_kaggle_readiness`. This records the
operator's read-only SDK/status observations, not an admitted run or a permanent
reservation. No install, upload, kernel modification or fit was performed.

At approximately October 18 22:41 UTC (October 19 01:41 Kyiv), the project `.venv`
authenticated and returned:

| Quota | Used seconds | Reserved seconds | Allowed seconds | Remaining seconds |
|---|---:|---:|---:|---:|
| GPU | 89,923.573 | 0 | 108,000 | 18,076.427 (~5.02h) |
| TPU | 0 | 0 | 72,000 | 72,000 |

The reported refresh is October 24 00:00 UTC. This campaign currently plans GPU
work; the TPU figure is inventory, not a TPU experiment admission. Billing and
multi-device quota consumption must follow the provider's actual semantics.

A status query covered all 29 account kernels last run in the preceding 72h:
27 COMPLETE, 2 ERROR, none RUNNING/QUEUED, no status-query errors. This includes
other-project CASMI kernels, which were not modified. Zero provider reservations
does not rule out another agent's planned reservation. Refresh quota and reconcile
ownership at actual admission; accelerator availability is not guaranteed.

## Existing working interface

The global Kaggle CLI is 2.1.1 and lacks `auth list`. Authentication did work with
`kernels list --mine`. The project `.venv` already contains Python 3.13.9,
Kaggle 2.2.4 and kagglesdk 0.1.37, including `KaggleApi.quota_view` and
`ApiGetAcceleratorQuotaStatisticsRequest`. No upgrade is needed. Reuse the
bounded reader in [poll_provider.py](../../campaign_2026-10-06/cloud_preflight/poll_provider.py)
instead of exposing credentials or inventing CLI options.

[kaggle_bundle.py](../../../scripts/kaggle_bundle.py) seals explicit dependencies
into the private bundle and kernel metadata;
[kaggle_gpu_runner.py](../../../scripts/kaggle_gpu_runner.py) runs the sealed spec.
The plateau source recipe/bundle is not ready yet. The next action is to finish
the common implementation, freeze actual input identities and matching recipes,
then refresh quota and profile on the admitted device. This observation does not
authorize starting a job before the agreed October 20 date.
