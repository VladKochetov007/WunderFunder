# October 4 retrospective

## Latest public evidence, received during the pause

Seed-4 combo reached **0.6660**, 37m07s /626 MB, +0.0005 over seed 3.
The fixed seed-3 parent/head endpoints scored **0.6641 /0.6654**; their equal
mixture is **0.6655**. The numeric target is reached at portal precision.
Both combo seeds beat the old D256 public point. Full supplied validation ranks
old D256 slightly above seed 4, which reinforces the need to distinguish fitted
validation, excluded reporting panels and public transfer.

The original q8 ZIPs are withdrawn after a libpython3.13 prerun failure.
Packaging-only repairs pass offline Python 3.11 callbacks; runtime remains
held. The research goal stays paused. See [results and reflection](PUBLIC_RESULTS_REFLECTION_20261004_LATE.md)
and [the interactive timeline](eda/leaderboard_climb/leaderboard_climb.html).

## Pause and scope

The user paused new research iterations on October 4. The goal remains paused; later portal evidence reaches the numeric target without resuming it. E425's existing four-epoch soft-routing fit, fixed export and report drained successfully. The submitted E463 TPU preflight remains queued; it has not yielded a TPU predictive result. E464's waiting controller was deliberately stopped before any head fit. There will be no new GPU fits, notebook pushes, timing trials, retries or research-agent iterations while paused. The q8 compatibility repair is bounded packaging maintenance, not a new scientific iteration.

`NO_NEW_RESEARCH_ITERATIONS.json` records the pause. `checks/E464_USER_PAUSE_20261004.json` distinguishes the intentional service stop from a failed scientific experiment. Unrelated historical pause markers were not altered.

## What changed the campaign

The complete combo was finally fitted, packaged, and manually submitted by the user. Seed 3 scored **0.6655**, compared with the previous **0.6641**, and ran in **37m47s /625 MB**. Seed 4 subsequently scored **0.6660**, **37m07s /626 MB**. These are measured complete-predictor gains of +0.0014 /+0.0019, reaching the numeric target at portal precision. They do not separately explain Muon, chunk length and bootstrap, and they do not calibrate a universal local-to-public slope. See [E412 public result](E412_PUBLIC_RESULT.md) and [the new reflection](PUBLIC_RESULTS_REFLECTION_20261004_LATE.md).

The recipe's complete-predictor four-fold CV gain was +0.002414379, with a component interval [+0.001565628, +0.003448539]. Every fold's point difference was positive. These intervals condition on fitted seed-3 models; fold exposure, teacher lineage and search uncertainty remain. Exact public-ZIP replay over all supplied `datasets/valid.parquet` scored **0.700503766**. Because 1,505 of 1,873 supplied validation sequences were used in final fitting, that aggregate is descriptive. The excluded 368 score **0.699822159**, versus prior D256 **0.697777394**, and are historically exposed. See [validation scorecard](VALIDATION_SCORECARD.md).

## Completed empirical results

| Experiment | Actual evidence | Interpretation and limit |
|---|---|---|
| E450 seed-4 replication | Four-epoch parent, fixed head and exports completed. Excluded-panel WP 0.701694557; equal two-seed bag 0.701420130, versus seed 3 0.699822159. All paired intervals include zero. | Fixed seed-4 public 0.6660, runtime37m07s. Public replication supports the complete recipe; equal bag remains unsubmitted. |
| E424 specialists/adapters | Eight complete parent fits and common-FP32 paired scores. Learned hard specialists minus matched GRU +0.000353842; diagonal modulation +0.000768346. All main intervals include zero. | No established architecture improvement. Learned routing did not beat uniform/hash routing. These parent comparisons do not include the incumbent's final head procedure. |
| E426 teacher clocks | Three matched four-epoch fits and fixed FP32 exports. Aligned-minus-constant −0.000045927; aligned-minus-shuffled +0.000081363; both intervals include zero. | The tested consistency-teacher clock has no supported gain on one exposed fold/seed. Event-time features and target-specific histories remain different interventions. |
| E423 EMA horizon | Fresh matched four-epoch control/intervention: preserving decay in rows gave −0.000574096, interval [−0.001796979, +0.000302859]. | The proposed EMA-horizon control was actually tested. It did not improve this screen; it does not isolate gradient accumulation or the distilled-loss normalizer. |
| E419 QAT | Both T4 scientific pipelines completed before a later output-copy failure. Recovered, hashed and rescored. QAT differences from floating continuation +0.000004823 / −0.000006264; intervals include zero. | Close accuracy preservation; no supported QAT gain. Provider ERROR did not invalidate these completed fits. Native deployment is a separate question. |
| E417 reset diagnostic | Reset250 / reset500 differences −0.04217 / −0.03144 with negative component intervals on a small fixed panel. Longer-reset intervals include zero. | The fitted state is useful or sensitive to disruption. Reset is an out-of-distribution intervention, not proof that longer BPTT improves training. |
| E421 categorical readouts | Fixed reciprocal frozen-state heads were substantially worse than the unchanged predictor. Factorized CE −0.088107392. | Negative for this sparse frozen-head fit. The newly fitted wMSE control was also poor; this does not settle a full-data, end-to-end or carefully initialized branch model. |
| E462 fixed INT8 conversions | Full-stream own-reference native parity and the fixed 368-panel score completed. Tiny score differences; paired intervals include zero. | Original q8 ZIPs withdrawn after public prerun exposed a libpython3.13 dependency. Packaging-only rebuild passes offline Python 3.11 callbacks; runtime remains pending. See [incident and repair](E462_PACKAGING_INCIDENT_20261004.md). |
| E465 JAX optimizer port | Actual staged FP32 JAX/Torch scalar AdamW comparison passed 7,460 updates, unchanged tolerances. Parameter max error 4.202127456665039e-6. | Real numerical evidence, but not full-model, distributed TPU or predictive equivalence. The failed v1 NumPy aggregation is preserved. |

Sources: [seed replication](E450_SEED4_PAIRED_REPORT.md), [architecture parents](E424_ALL_PAIRED_REPORT.md), [clock result](E426_CLOCK_RESULT.md), [EMA result](E423_EMA_REPORT.md), [QAT recovery](E419_RECOVERED_QAT_REPORT.md), [reset panel](E417_RESET_REPORT.md), [branch screen](E421_BRANCH_REPORT.md), [INT8 score](E462_NATIVE_PAIRED_SCORE.md), and [actual FP32 AdamW proof](E465_FP32_ADAMW_RESULT.md). E425 also drained successfully: soft WP 0.696672177, +0.000225892 versus hard routing /+0.000171562 versus uniform routing /+0.000579733 versus matched GRU; all paired intervals include zero. All four experts had direct prediction gradients, but useful specialization is not established. See [the final drain result](E425_DRAIN_RESULT.md).

## Where execution lost time

Strict integrity checks repeatedly streamed tens of gigabytes before useful GPU work. The CPU diagnostic measured 84,386,156,018 declared hash bytes over 572 calls; faster CPU-host I/O did not establish a T4-host startup time. Startup failures need to be accounted for separately from scientific null results.

W384 v4 failed about 1.70 seconds into kernel execution because its wrapper checked declared external paths against an archive-only allow-set. All 26 external paths were intentionally outside that set. No phase runner, probe, fit, prediction or score ran. This is no evidence against width. The corrected v5 wrapper is a held draft: eight fixtures ran before a final guard refinement, which remains untested. Exact traceback and audit are preserved under `kaggle/W384_fold1_dualT4_v4_transport/`.

Repeated strict quiet-host timing could not complete while the host was loaded. Those failed trials remain failed. A separate prospective loaded-host instrument records paired ratios, process/thread CPU clocks, perf cycles and interference, but is unsealed and unexecuted. It cannot retroactively turn earlier failures into passes.

Controller status and experiment status drifted. The new explicit pause records and updated ledger distinguish fitted/scored predictors, hardware/implementation checks, source-only work and jobs that never reached training. E427 review caught real pre-launch bugs: bootstrap field mismatch, stale export-loop index, incomplete OOM evidence, and generated bytecode in the code closure. Some repairs are untested and the source inventory is stale. No E427 GPU smoke or fit occurred.

## Lessons for the next session

Finish complete predictors and runtime measurements without letting instrument work block unrelated forecasting fits. Freeze evaluation panels and controls before scoring; report training overlap directly. Preserve negative results at the tested configuration's scope. Require production-shape resource probes and external deadlines before large allocation. Treat source fixtures as narrow evidence, not proof of whole-notebook startup. Keep package readiness separate from scientific score and provider status separate from per-phase results.

The fixed endpoint and seed-4 ZIPs now have measured public results and runtimes. Seed 4 is the current best at 0.6660. Both original INT8 ZIPs are withdrawn; repaired ZIPs pass offline compatibility but still need runtime approval. No unsubmitted candidate is promised to exceed 0.6660. The [historical shortlist](SUBMISSION_SHORTLIST_20261004T2041Z.md) is superseded by [the measured results](PUBLIC_RESULTS_REFLECTION_20261004_LATE.md).
