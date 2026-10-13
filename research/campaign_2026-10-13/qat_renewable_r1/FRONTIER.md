# QAT implementation frontier

Updated 2026-10-13T16:30:01.115844+00:00. Root owns delivery under the [plan](PLAN.md) and [contract](CONTRACT.json).

- Implemented and independently reviewed uniform recurrent W4, mixed W4/W8, exact native code/scale import and packed transport. Existing checkpoint/CLI paths remain supported. Python and Rust focused checks pass.
- Real CP5 complete fitting-sequence checks pass: packed Q8 predictions/states are bit-identical to native Q8; all cases reset/replay. Archive savings do not reduce resident byte-kernel weights or establish INT4 speed.
- Independent [full-fold clearance](../../unified/reviews/QAT_DEPLOYMENT_R2_FULLFOLD_CLEARANCE.md) is complete. Native Q8 F0 completed 468 groups/141 components with WP0.6811354; FP32 full-fold remains active. Other formats await resource admission. This is an unchanged-parent exposed-F0 baseline, not a QAT gain.
- [Delivery recipe](delivery_r1/README.md) freezes complete CP5/teacher/initializer ancestry and nested component subsets. [Independent 10% fit admission](../../unified/reviews/QAT_RENEWABLE_R1_FIT_ADMISSION.md) passes.
- All three actual-data B16/C1000 FP32 training profiles completed 100 finite optimizer updates. [Local execution packet](LOCAL_10PCT_EXECUTION_R1.json) binds measured resources and matched commands. The 10% floating-control fit is running; uniform and mixed fits follow on the single GPU worker.
- Parent file cache caused two later-profile admissions to be refused before any process started. Receipts preserve those refusals; a bounded cache handoff restored headroom. Physical/CPU/RAM limits were unchanged. Keep fresh admission checks and avoid concurrent support tasks when complete bounds exceed17GB.
- [TPU V8 bridge](tpu_bridge_v8/README.md) is prepared and under independent preflight review. No new provider job has yet been launched. Root owns quota recheck, dispatch and terminal collection.
- Next: finish the matched10% trio, preserve each QAT checkpoint's exact codes/scales in native export, evaluate against matching PTQ controls, grade paired component uncertainty, then allocate30%/full and broader folds/seeds. Activation-aware/learned-scale extensions and complete blend comparisons remain pending.

No complete QAT quality comparison or new strongest package exists yet. Full F0 teacher/initializer exposure makes the small subsets adaptation diagnostics. F3 matching CP5 lineage is not yet available. Public upload remains separate; no paid action is authorized.
