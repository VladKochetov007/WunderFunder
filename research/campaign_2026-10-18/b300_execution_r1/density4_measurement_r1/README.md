# Fourth-process throughput measurement

The latest user request authorizes testing increased concurrency on the existing eight B300s. Root uses the existing single SSH operator console; the 24 full fits and their frozen recipes remain unchanged.

The bounded `tabdata-density4-v3` unit overlays one canonical PR15 profile on GPU2 and GPU7, using disjoint CPU pairs 56/58 and 61/63. Each profile completes two warmup plus 64 timed outer batches, or 660 optimizer updates and 528 sequences, under strict FP32. GPU2 has a deep R01 fit and two PR15 fits; GPU7 has a wide R01 fit and two PR15 fits. Foreign workloads are preserved.

The measurement compares 125 seconds of existing three-process progress with progress during the fourth process. Existing actor rates come from completed batch increments and log timestamps. Since canonical profiles suppress progress lines, an independent 0.5-second observer records loader-to-completed-update walltime. The added actor's rate includes all warmup and overhead. Aggregate rates include R01 contention; epoch-budget-normalized rates are reported separately. This short engineering test cannot establish sustained eight-GPU throughput or model quality.

Remote evidence lives under `/data/epanov_s/tabdata/state/density4-v3`; executable measurement, observer and postprocessor live under `runtime/density4-v3`. At 2026-10-18 21:17:18 UTC both profile actors had entered training; no concurrency decision or queue handoff had yet occurred. The existing frozen controllers remain unchanged.

Failed startup attempts are retained: v1 omitted a required workspace argument and started no CUDA actor; v2 detected generated Python bytecode before profiling and started no CUDA actor. The two bytecode files created by the first attempt were quarantined with hashes under `state/density4-v1/generated_bytecode`, and v3 disables bytecode writing. No tracked training source was changed.

The separate [successor preparation](../density4_v1/r01_successor/README.md) preserves pending-cell ownership until an exact process-identity and journal/process/workspace reconciliation transfers it. Healthy initial fits retain their original processes, CPUs and inherited deadlines.

The [exact deployed R01 v4 source snapshot](../deployed_review_v4/README.md) supplies the actual controller, actor adapters and postprocessing contracts for successor review, with source pins verified against the server manifest. It deliberately excludes bulk canonical fold payloads.

## Observed measurements and allocation decision

The fourth-PR15 overlay completed both finite 660-update profiles. Its unit failed only in a postprocessor that expected progress lines suppressed by canonical profile mode. The retained [walltime analysis](observed/state/density4-v3/WALLTIME_RESULT.json) uses actual incumbent progress and external walltime instead: aggregate sequences/s rose 5.75% on GPU2 and 2.80% on GPU7, while the incumbent R01 fit slowed 14.30% and 22.76%. This allocation does not address the campaign's R01 tail.

The mixed overlay instead adds a wide R01 to a deep incumbent on GPU2 and a deep R01 to a wide incumbent on GPU7. Both completed 340 finite strict-FP32 updates. Its older a266 profile source omitted a newer resource-evidence marker, so that unit also retains an explicit failed postprocessor status. [Feasibility](observed/state/density4-r01-v1/FEASIBILITY.json) records conservative startup-inclusive aggregate R01 rates of 4.61766 versus 2.64930 sequences/s on GPU2 (+74.30%) and 4.65409 versus 3.35526 on GPU7 (+38.71%). PR15 service slowed approximately 22–24% in those windows.

The [joint fit-queue projection](observed/state/density4-r01-v1/FIT_QUEUE_PROJECTION.json) estimates the remaining R01 tail at 38.46h rather than 50.15h and PR15 at 26.48h rather than 20.52h: approximately 23.3% less time for the slower fit queue. This is a provisional two-GPU engineering projection, excluding export, scoring, repair and nonpreemptive scheduling overhead; it is not an observed campaign duration or a promised ETA.

[Resource reconciliation](observed/state/density4-r01-v1/RESOURCE_RECONCILIATION.json) binds actual profile PIDs, finite update counters, source/recipe/device/CPU identities and normal trainer returns while retaining both failed postprocessing records. Profile peak memory and profile OOM counters were unavailable after their cgroup disappeared and remain null. Original R01 memory events were zero, all 16 original PR15 full fits completed, and the three foreign vLLM clients remained alive. Fresh per-unit hard limits, RSS/OOM counts, host memory and live process-group observations are required for scaleout.

The separately corrected [PR15 queue](../pr15_queue_v2/README.md) recovered all 16 original endpoints without retraining and launched the remaining 104-cell successor at two PR15 workers per GPU. The selected target composition is two R01 (one wide, one deep) plus two PR15 actors per GPU; its actual launch/adoption status belongs in the successor and live-status receipts.

## Eight-GPU follow-up after actual scaleout

The root-owned [read-only observer](observed/runtime/density4-followup-v1/observe.py) samples real batch-index increments and log timestamps every five seconds under a separate1GiB/swap0/32-task CPU unit bounded to30 minutes. It starts no training/GPU actor and sends no process signals. The [complete warm-window receipt](observed/state/density4-followup-v1/WARM_WINDOW.json) contains all24 incumbent actors before scaleout and all32 after scaleout, using a baseline ending before the start gate and a follow-up beginning120 seconds after it. Each actor needs at least two distinct progress points spanning35 seconds. Startup warming and short-window variability remain limitations.

In that all-eight-GPU window, aggregate R01 service rose from24.150 to38.677 sequences/s (+60.15%), while PR15 service fell from51.596 to39.229 sequences/s (about24% slower). This is measured reallocation toward the longer R01 queue, not a60% reduction in total campaign time. Summed raw sequence service across both families rose only about2.9%; their epoch budgets and model work differ.

The [live queue forecast](observed/state/density4-followup-v1/LIVE_QUEUE_FORECAST.json), around22:09 UTC, uses the actual remaining batch counts of the180 frozen cells, excludes the16 already completed original fits, and serializes each measured GPU/arm or GPU/lane slot. It estimates35.19h for the R01 fit tail and28.58h for PR15. First original wide R01 fits had about1.6–1.8h of fitting left, and the active PR15 wave about3.6–3.8h. These estimates exclude export, scoring, repair and phase/scheduling overhead and may change with future contention; they are not completion observations. The all-eight-GPU follow-up refines, rather than replaces the historical evidence for, the earlier two-GPU38.46h projection.
