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
