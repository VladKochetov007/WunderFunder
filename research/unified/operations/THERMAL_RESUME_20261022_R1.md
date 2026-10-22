---
kind: incident
status: running
created_utc: 2026-10-22T19:33:37Z
updated_utc: 2026-10-22T19:33:37Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-ONLINE-RELATIONSHIPS-20261022-R1]
evidence_paths: [research/campaign_2026-10-21/next_cycle_r1/RESUME_AFTER_THERMAL_R1.json]
---

# Resuming after the laptop thermal pause

## Question or claim

The user explicitly resumes experiments. This supersedes the pause while retaining the two-core, 5 GB aggregate RAM, zero-task-swap and physical-disk scratch limits. The 12-hour production reserve remains protected. Host SSD swap is active; current `/tmp` and zram boot changes remain pending reboot.

## Mechanism and method

The draining F0 warm control completed both epochs and saved fixed CP1; its [terminal receipt](../../campaign_2026-10-21/next_cycle_r1/WARM_GRACEFUL_DRAIN_RESULT_R1.json) pins the checkpoint and 14,920 finite updates. The original four-fit comparison remains unchanged. The [independent resource review](../reviews/receipts/WARM_I1_RESUME_PANEL_RESOURCE_REVIEW_R1.json) holds admission of the remaining three fits until their complete delivery fits the remaining eight-hour stage.

## Evidence and interpretation

The first resumed GPU export could not cool while paused: its resident CUDA context drew about 35.7 W and stayed at 79–83°C. Five-second polling also observed a brief 86°C peak. Root stopped only this incomplete export after 900.128 wall seconds; [R3 cancellation](../../campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/F0_CONTROL_EXPORT_CANCEL_R3.json) preserves the untrusted partial file, guard events and log. No fitting checkpoint was changed. This is execution evidence, with no model-quality result.

The [R4 export amendment](../../campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/F0_CONTROL_EXPORT_ADMISSION_R4.json) uses the same frozen predictor and FP32 checkpoint on CPU, GPU hidden. It starts at half a core and increases to one core after observing 72–74°C; the guard polls each second, pauses at 80°C and resumes at 74°C or below. Its 30-minute wall cap includes pauses. Future paired exports must use the same backend; FP32 device rounding can differ. Exact full-file coverage and sidecar acceptance remain pending.

## Limits and validity

Earlier strong-parent throughput is not a safe current projection. Cooling pauses count toward the budget. No remaining strong-parent fit or online scientific fit has been admitted. Root does not treat a zero-filled or partly written NPY file as prediction coverage.

## Decision and next step

Finish or cost-defer the saved-checkpoint export, then remeasure the complete remaining panel. Separately, the [compiled online feature profile](../../experiment_drafts/online-relationships-20261022/FEATURE_PROFILE_COMPILED_RESULT_R1.json) reduces the empirical one-pass feature projection from 15,818.88 to 523.40 seconds. Its [9.53-second execution receipt](../../experiment_drafts/online-relationships-20261022/FEATURE_PROFILE_COMPILED_EXECUTION_R1.json) records 534,920 KiB peak RSS and zero swaps. This reopens full cost profiling; it does not admit twelve fits. Source-level exactness has an [independent review](../reviews/ONLINE_RELATIONSHIP_NUMBA_FP32_REVIEW_20261022_R1.md).

## Source records

[Resume authority](../../campaign_2026-10-21/next_cycle_r1/RESUME_AFTER_THERMAL_R1.json), [live frontier](../../campaign_2026-10-21/next_cycle_r1/FRONTIER.md), [online packet](../../experiment_drafts/online-relationships-20261022/README.md), [disk swap inspection](SWAP_AND_DISK_TMP_20261022_R1.md). No new public-quality claim or competition upload.
