---
kind: incident
status: running
created_utc: 2026-10-13T23:47:14Z
updated_utc: 2026-10-13T23:47:14Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: []
evidence_paths: [PROVIDER_OBSERVATIONS_R1.jsonl, QUOTA_AND_PROVIDER_ENTRY_R1.json]
---

# Entry resources and cloud recovery

## Question or claim

Can useful local/cloud work proceed independently of delayed user CPU uploads?

## Mechanism and method

Read actual user-systemd slice limits and GPU/disk state; poll only the two relevant Kaggle kernels and typed quota. Preserve prior failures and queued work. Review the successor before dispatch.

## Evidence and interpretation

The [raw observation](PROVIDER_OBSERVATIONS_R1.jsonl) and [effective quota interpretation](QUOTA_AND_PROVIDER_ENTRY_R1.json) report33,306.065GPU seconds and72,000TPU seconds available, zero reserved, renewing October17. The V8 bridge remainsQUEUED. QAT profileR2 remainsERROR and was already collected/diagnosed in its originating campaign. These are execution states, not quality results. The old poller includes superseded campaign-cap fields; the effective receipt explicitly excludes them from current admission authority.

The parent slice remains5CPU cores,18GB high,20GB maximum and no swap; entry charged memory5.12GB and physical Btrfs free space56GB. GPU was75MiB with no research worker observed. Refresh before admissions. Three native Luna/max tasks started for inference, training delivery and independent review. No external paid model or hardware route was allocated.

## Limits and validity

ProviderQUEUED is not an actual TPU update. Local memory snapshots are not future reservations. CPU submissions await the user, so no shared-slot or target-host timing claims follow.

## Decision and next step

Continue local architecture preparation and theoretical work. The [R3 cache-containment repair](kaggle_profile_r3/README.md) preserves original targets/recipes and is ready for independent internal review. Root retains admission and terminal delivery ownership.

## Source records

[Contract](CONTRACT.json), [narrow provider target list](provider_targets_entry_r1.json), [R2 predecessor](../../campaign_2026-10-13/qat_renewable_r1/kaggle_f0_s4_r2/README.md), [TPU V8](../../campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/README.md).
