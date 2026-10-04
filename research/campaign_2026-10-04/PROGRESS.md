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
