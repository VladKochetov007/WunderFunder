# Research wrap and submission decision — October 5

## What can be submitted

The strongest measured public predictor is still the complete seed-4 combo: **0.6660**, 37m07s /626MB. Seed3 complete scored0.6655, seed3 head-only0.6654, and seed3 parent-only0.6641. The public improvement belongs to the complete training plus final-head procedure; it cannot be assigned entirely to the parent or added independently for Muon, chunk size and bootstrap. No new activation pilot establishes an expected public0.7000 or provides a verified stronger submission.

See [measured public reflection](../campaign_2026-10-04/PUBLIC_RESULTS_REFLECTION_20261004_LATE.md) and [current ZIP audit](NEW_SUBMISSION_AUDIT_2026-10-05.md). Repaired INT8 packaging is distinct from new model evidence: offline Python3.11 compatibility does not replace measured deployment time. Original Python3.13-dependent ZIPs remain withdrawn. Existing scored packages do not need repeat uploads to rediscover their scores.

## Finished work and what it teaches

| Workstream | Finished evidence | Interpretation and limits |
|---|---|---|
| Complete combo and second seed | Public0.6655 and0.6660; separate parent/head endpoints scored | Real public progress; only two complete seeds, no universal local-to-public mapping. |
| Specialists and conditional adapters | Eight parents plus eight E464 final-head fits;16 fixed complete-predictor comparisons | None beats the seed3 complete combo in the exposed screen. This is evidence about these implementations, not a rejection of expert models. |
| QAT, INT8 and deployment | Recovered QAT fits and native-reference accuracy checks; original public ABI failure documented | Accuracy preservation is useful. Native runtime and Python linkage are separate gates; a ZIP existing is not readiness. |
| Target mechanics | Raw EDA plus fitted and excluded-row strong-predictor diagnostics | Current-event clock is almost necessary for scored target transitions, but hard freezing outputs hurts badly. Difference-mode residual energy dominates; target variance also dominates there, so a new head is not guaranteed to help. Excluded rows still have component links into training. |
| SSM, attention and spline readouts | E467/E469:30 fixed small-model fits at three seeds | Learned diagonal memory beats fixed memory at all seeds. Spline gains over learned-memory MLP stay positive but shrink across seeds. Attention is mixed and topology-dependent. These are matched prototype effects, not gains over the full incumbent. |
| Joint target heads | E468:five fixed fits and all ten between-arm contrasts | None beats its historical GRU128 parent pair. Coarse codebook, frozen trunk and one fitting pass limit this negative. |
| Activation families and optimization | Pointwise, recurrent-candidate, placement, mixture, dropout, LR-schedule, momentum and slope/curvature controls | Effects interact with optimizer and placement. Positive sine effects do not identify periodic structure: linear/cubic controls can reproduce benefits. A first-seed schedule gain failed replication. Full activation accounting and limits are in the linked retrospective. |
| Frozen activation counterfactuals | E486:36 fixed-weight cells/54 comparisons, independently reproduced | Both inference function and training adaptation contribute on the small exposed panel; training-by-inference interactions are positive. Repeated diagonal results are not new independent fits. |
| Muon precision | E487:fixed one-batch crossed replay and all nine saved-delta checks | BF16 orthogonalization fails the registered native/custom step tolerances; FP32 passes this one-step comparison. It is not a convergence or predictive gain. |
| Three-epoch mixtures | E488:12 continuous fits/24 snapshots completed; all24 snapshots independently graded; all27 global/54 target contrasts and2,000 draws reproduced | Compare recurrent versus static/fixed gates at both epochs and retain duration interactions. No checkpoint selection or seed pooling. Epoch3 fixed-mixture-minus-GELU +0.00893/+0.00431/+0.00869; two positive CIs, one includes zero. Learned recurrent-minus-static effects are tiny and all three CIs include zero. See [duration report](context_mixture_duration/REPORT.md). |
| Prepared H128 duration | E485/E489:12 registered trajectories unadmitted | E485 numerical gate failed. New FP32 E489 probe failed import before CUDA/data, and its reservation missed the cutoff by32.527s. Source-only repair is prepared for another window; no scientific result exists. |
| Accelerator support | Private E482 T4 predictive replication completed; JAX scalar FP32 AdamW port check completed | Real GPU predictive evidence and a numerical port check. E463 actual TPU preflight remains provider-queued; no full TPU predictive result. W384 RAM-guard stops and wrapper failures are not width negatives. |

Sources: [October4 retrospective](../campaign_2026-10-04/RETROSPECTIVE_2026-10-04.md), [current research status](CURRENT_STATUS.md), [activation retrospective](ACTIVATION_RETROSPECTIVE_2026-10-05.md), [sequence-component replication](sequence_components/THREE_SEED_REPORT.md), [strong target diagnostics](target_mechanics/strong_baseline/REPORT.md), [joint heads](joint_target_model/REPORT.md), [frozen activation swaps](activation_counterfactual/REPORT.md), [precision replay](muon_orthogonalization_precision/REPORT.md).

## Original review programme is not fully executed

The original review inventory contains proposals as well as completed experiments. E427 remains source-only; E429 L2 engine work is planned; E430 cadence/accumulation has implementation drafts rather than full fits. The registered E431–E446 current-combo interventions have not all been implemented or fitted. These include end-to-end branch readouts, target-specific histories, event-age/activity interactions, thinning, optimizer handoff/clipping, linear-optimizer ARTBP, and cross-fitted teachers. Later small-model screens do not silently count as completing those stronger-model proposals. Consult [the inventory](../campaign_2026-10-04/EXPERIMENT_INVENTORY.md) and the actual statuses in [experiments.jsonl](../experiments.jsonl); source-only, resource-censored and numerically rejected attempts are not scientific negatives.

## Why promising research has not yet produced another ZIP

Most new mechanisms were tested on24 fit components and16 repeatedly exposed evaluation components with much smaller models than the submitted GRU256. This is appropriate for finding controls cheaply, but it measures neither transfer to the complete incumbent nor public improvement. Paired component intervals condition on fixed fitted models and exclude training/search uncertainty. Selecting the largest small-panel gain would therefore overstate expected public performance.

The next stage should spend its main compute on complete-predictor transfer, rather than more minor searches on the same panel. Preserve all candidate/control results and use new training seeds plus component-aware evaluation. Fit the same final-head procedure before comparing against the incumbent. Measure inference time and test evaluator-compatible packaging before calling a model ready.

## Next-window research order

1. First transfer the fixed50/50 GELU/LeakyReLU readout mixture to the incumbent with a matched unchanged readout control and the same final-head procedure. E488 changes the readout activation only; it does not establish an embedding or recurrent-candidate result. Add the frozen training/inference swap control to distinguish function replacement from fitted adaptation. Separately transfer the positive bounded candidate with a slope-matched linear/cubic control; localize its placement and run complete schedules at multiple seeds. Retain all outcomes and measure native inference compatibility.
2. Finish the explicit FP32-Muon H128 duration study only after a newly sealed actual-device probe. BF16 failure is not repaired by renaming the original experiment. Use the already registered epochs and seeds; native-reference/update parity is distinct from expected accuracy.
3. Scale learned memory plus a compact nonlinear/spline readout with a capacity-matched control. Investigate target sum/difference posterior updates and activity-conditioned memory while allowing causal within-run updates; do not deploy the disproved hard output hold.
4. Return to W384 and a useful pair only after honest host-memory admission and CPU runtime benchmarks. Repair actual resident memory demand or stream/chunk buffers; changing only declared metadata is not a repair. INT8/QAT can enable feasible deployment but score preservation alone is not a new best.
5. Add truly cross-fitted teacher/privileged-information comparisons with original label and weights, if budget remains. Component overlap and fitted-versus-held-out population differences must be explicit.

## Resource and execution reflection

New scientific admissions closed at07:00UTC/10:00Kyiv. Grading already completed fits, writing reports and repairing source for a future window continue as drain work; no new scientific GPU fits are admitted. Local hard aggregate limits remain12CPUs/30decimalGB RAM/no swap, with27GB soft headroom. Small activation GPU workers used2CPU/6GiB host RAM/2GiB allocator/3decimalGB observed-GPU stop; all completed activation studies had no new OOM. Background checker remains active and does not restart failed jobs.

Latest provider quota observation in the monitor:19.84GPU hours and19.96TPU hours remaining, refreshingOctober10. These provider credits are not automatically device-GPU hours for two-card kernels. A queued TPU request has no trustworthy completion ETA.

Verification initially checked partial schemas and introduced mismatches between actual sealed file roles and grader expectations. Versioned failures were preserved before test decoding. All224 completed activation fits are now independently graded and numerically reproduced. E488 complete grade charged282.408seconds on one CPU, peak806.474MB, exit0 and verified teardown, no new OOM. The final grader uses an integrated actual fit/probe fixture; active clipping belongs to native-reference parity, not an invented requirement that every real-data update must clip. Future checks should exercise the complete real entrypoint and manifest before reserving a scientific worker. The E489 late admission also requires checking the deadline immediately before reservation, after hashing.

All durable evidence belongs in MD/JSONL, source, small JSON receipts, plots and the marimo notebook. Large tensors/archives remain local ignored artifacts with content identities. No new public score or fresh best is claimed from a pilot.
