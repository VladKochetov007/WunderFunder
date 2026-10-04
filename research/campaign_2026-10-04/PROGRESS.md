# Execution progress

Current implementation state is machine-readable in `implementation_state.json`.
Experiment decisions remain in the global ledgers; infrastructure checks are
not predictive results.

## Implemented and checked

- Content-bound local jobs and private Kaggle bundles, CPU/RAM/VRAM admission,
  durable attempts, OOM/monitor failure handling, and epoch recovery.
- Separate optimizer-update, backpropagation and EMA row clocks; pooled loss
  numerators/denominators; fixed and shuffled cached-distillation denominators.
- Opt-in optimizer handoff, module clipping, noise, head learning rate and
  seeded antithetic ordering with an identical normalization pass and dropped tail.
- Fixed complete-predictor combo/head recipes, width and existing structural
  model retests. The 41-card inventory includes additional code still in progress.
- A live marimo notebook with exported historical public-score figures.

## Independent integration checks

The local CPU and CUDA GRU checks passed. The CUDA case included batch16,
1000 rows and hidden256. Its observed device peak was 667,942,912 bytes; no
cgroup OOM events occurred. This is a compatibility fixture, not the full
211-input production training recipe. Raw results are in `checks/`.

The real CUDA trainer test uses synthetic parquet data, actual feature
construction, cached distillation, both auxiliaries, EMA, fused Adam and loader
workers. Model/optimizer/EMA tensors matched interrupted versus uninterrupted
execution. Its first attempt failed a runtime-contract equality check because
initialization changed the CPU thread count (1 versus 4). Reasserting the
configured thread count fixed this: CUDA retry v2 passed all three checks.
The controlled Muon/macrostep case also passed. Failed evidence remains
in `runner/logs/actual-trainer-cuda-v1.log`.

Real systemd preflight rejected an unsupported MemoryOOMGroup property before
starting a workload. OOMPolicy=kill was then verified to set memory.oom.group=1;
the fresh contained CPU attempt passed. We did not remove or reactivate old
campaign pauses. The existing `wnn-codex-research-watch.service` observed the
later test failure and rewrote its legacy pause/RESOURCE_FAILURE markers; these
watcher-generated records are preserved alongside the new runner's incident.

## Remote state

The private raw Kaggle dataset is accessible. The private TPU compatibility
kernel is queued; no TPU training support or HBM figure is yet established.
The first dual-T4 transport attempt stopped before workers started because
Kaggle unpacked the submitted archive. The opaque-file retry passed on both T4s.
Two actual-trainer replay jobs then passed 17 tests each on the Kaggle stack;
these short replay jobs ran sequentially on device0, so they do not establish
concurrent training memory demand.
Provider logs and separate quota records are retained with the remote attempts.

## Next checks and fits

E-410 complete-predictor head fits are running through reviewed, sealed
specifications. Fold0 gives +0.004613 WP (component interval +0.001687 to
+0.007846); this is provisional pending other folds and fresh CTRL reproduction.
E-409 measured W384 at 401.617 kcycles/row for one ordinary head and 452.945
for a duplicated-head sizing control. Neither number is a public-time claim. CPU W384/native timing, quantization, volume-input and conditional
feature implementations proceed in parallel. Historical negatives inform their
matched controls and do not exclude these retests.

No new score gain is claimed. Best recorded public score remains 0.6641.

## Four-fold complete predictor finished

E410 is complete: pooled +0.002414, component CI[+0.001566,+0.003449].
All four fold deltas are positive; fold1/2 class3 each decline ~0.00056.
Fresh CTRL head model tensors exactly reproduce the historical control.
E411 q8 degradation pilot is running on CPU; E412 full combo is being sealed.
The updated trainer passes 20 targeted CPU checks and CUDA replay after the
calibration/jitter/privileged-teacher additions. Current public best remains0.6641.

E412 full-data seed3 started with frozen source 96ca4b1e61823bec3953158bea12124be1efa3ec29493f7902d3eeea5be885a3. Latest CUDA main integration: 8 passed, including paired origin views. H098 teacher code additionally passed synthetic teacher-to-causal-student tests; scientific teacher fits remain pending.

E411 completed: recurrent q8 delta -0.000022729; recurrent+head q8 delta -0.000019261. Both component intervals cross zero. Float32 represented-weight pilot only; native byte runtime still pending. E414 calibration did not launch: its6GiBCPU specification exceeds the current2GiB worker envelope. The next runner revision is being prepared in isolation; the live E412 run remains frozen.

E413 private Kaggle volume-history/current-input pair is RUNNING (provider confirmation02:21:24UTC); exact code digestbbdba0e2810339765ce9b655fcf1dfaa292fb6642734c23eb894509b2972c9ee. CurrenttrainerCUDA checks precede each fixed4epoch/ep3 scientific fit. Worker/device evidence awaits provider logs. TPU stillqueued.

E415 empty sampled sequence raised ValueError; global pause also stopped E412 withSIGTERM. NoOOM. Exact full epoch2 recovery snapshotverified againstevery savedsourcehash; E412retryv2 usesfreshoutput c20261004_combo_s3_final_r1, samecode/recipe/data/runtime. E415sampling-preservingfix/tests pending. SeeE415_EMPTY_SAMPLE_INCIDENT.md.

2026-10-04T02:48:55.008284+00:00: E412 v2 passed strict epoch-boundary recovery and is training epoch2. E415 sampling-preserving fix passed10 CPU checks; A-v2/B-v1 extraction retries are running, with no OOM. Native q8, QAT/ARTBP, factorized recurrence and exact conditional-feature variants remain distinct implementation branches. Historical negative results do not veto these retests.

2026-10-04T03:16:43.837275+00:00: E412 parent/head fits and full-valid exports complete. Package callback/v2-v3/reset checks passed but raw q16x-vs-f32 error0.000415802 exceeds3e-4; officialWPdelta8.745e-8. No package promotion. FP16GPU/native difference needs commonFP32 reference. E416 has a quiet CPU timing window before integration tests resume. SeeE412_NUMERICAL_REVIEW.md.

## 03:46 UTC — calibration and common precision results

E414 finished the actual mixed-fit denominator calibration: mean2947.4312 over7460 updates, SD577.5941 (CV19.60%), nozero-mass updates. Predictive aligned/shuffled/unchanged fits remain pending. The CPU job finished714.68seconds without OOM.

E412 common FP32 exports are complete. Nativef32 versus CUDAFP32 max required/scored error0.000594392, WPdelta−2.1331e−6; q16x versus nativef32 preserves the original max0.000415802 gate failure. FP16 versus FP32 inference max0.0405848/WPdelta−8.8658e−5 explains the large arithmetic-path discrepancy. The remaining smaller nativefloat difference still requires localization. No package cleared.

Selective CUDA v6: three origin tests pass; both thinning checks reached successful exact restart checks but their final normalization assertion compared CPU to CUDA tensors. The isolated test correction is ready; rerun pending. NoOOM or globalpause. E416 clean byte-engine confirmation is complete: q8x smaller but walltime+5.9% D256/+3.2% W384, so no native speedup claim.

## 2026-10-04T06:59:36.915871+00:00 — reconciled live execution

E413 two-T4 volume-history/current-input pair completed. Fixed-epoch3 paired WP gain +0.00096179, component CI[-0.00007997,+0.00205967]; one exposed fold/seed, replication pending. E417 current-student reset panel and all16 E418 literal frozen-state feature screens finished. Neither a frozen-head negative nor old campaign evidence closes the corresponding end-to-end research proposal. E421 both reciprocal categorical branch screens finished; candidate loses unchanged and joint CE in each direction.

E412 i16 alternative passes unchanged max-error/WP gates on20 full streams, sequence-reset replay, and bitwise v2/v3 agreement. Preserve q16x raw-error failure. Nativefloat agrees with independent CPU Torch within2.235e-6; larger CUDA differences concern execution arithmetic. Clean package runtime remains pending, and no competition upload was made.

E422 CPU gain-cache helper exceeded its4GiB child cap because anchor-state views retained full20k hidden arrays. Actual systemd oom-kill evidence is retained. Selected states now copy their small anchor arrays; weak-reference lifetime regression4passed under2GiB. Production cache retry is held while uncheckpointed local training runs. A controller-service namespace mistake and subsequent stale-job reconciliation interrupted two earlier E419 attempts before any epoch checkpoint. Those attempts are preserved; exact fresh rerun E419 FP16 v3 is healthy.

Private E419 dual-T4 QAT kernel is RUNNING; physical-worker logs are not yet available. TPU remainsQUEUED, no memory/core claim. Luna-only delegation respected (three actors); localCPU/GPU jobs remain root-admitted under aggregate resource limits.

## 2026-10-04T08:04:21.521861+00:00 — runtime and precision reconciliation

Clean i16 package timing completed and candidate ready for user upload, public unmeasured. Local FP32−FP16=-0.00001605 with interval[-0.00015023,+0.00008639]. Factor clean timing458.20kcycles versus396.07original q16x and763.03effective dense FP32; v2 CPU overlap preserved. Gain GPU caches retained209/219 sequences and reciprocal screens launched. QATv2 missing-helper preflight failed without fits; correctedv3 private push authorized. TPU queued. Luna-only maximumsix.

## 2026-10-04T08:11:34.072663+00:00 — next predictive fit launched

E422 reciprocal screens completed; state/input gain R² negative, earlier-label oracle worsens later-episode WP. EDA saved under eda/gain_probe and live marimo cell lebv. H093/H094 merged with corrected production auxiliary objectives; root12 integration plus23 optimizer/control tests passed under2GiB. E423 full-schedule EMA-row intervention launched, fresh same-source control/common-FP32 exports/scoring queued behind it in bounded controller. Initial local GPU92%,3168MiB, host11.46GB within caps. Kaggle private QATv3 RUNNING at provider, worker progress pending.

## 2026-10-04T11:00:39.437879+00:00 — EMA result and Kaggle QAT failure boundary

E423 EMA-row intervention vs fresh same-source control=-0.000574 WP, paired component interval[-0.001797,+0.000303],468 sequences/141 components; one exposed fold/seed. Kaggle E419v3 used both T4 workers for preflight, one-batch QAT probes, matched FP32 controls and exports, then failed score lookup on stale t4v2 names. No QAT/PTQ fit ran. Preserved exact failure outputs. Quota26GPU/20TPU hours. TPU queued; width pair staged.

## 2026-10-04T11:07:24.169038+00:00 — recovered E419 control score path

Saved T4 v3 FP32 controls passed corrected single-arm local scoring. Both named controls have identical model tensors and predictions (WP0.681496); this is path validation only. PTQ/QAT fits did not run. Luna is preparing a cached-control QAT retry plan; no additional Kaggle quota spent.
