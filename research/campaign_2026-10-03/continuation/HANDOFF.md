# Active research continuation — 2026-10-03

**Paused by the user at22:34 Kyiv.** All five agents stopped; no experiment is
running. Current canonical handoff is [RESUME.md](RESUME.md), with completed and
unfinished work distinguished. Consolidation commits preserve research on `main`.
The notes below are historical progress updates and do not override the pause.

At21:00 E-391 independently blocks v3 for Torch RNG byte-state normalization;
NumPy and moment corrections pass. Minimal v4 exact private-engine set/get check
is building, no CUDA receipt. E-392 commonmacrostep CPU credit control passes58checks;
no empirical credit benefit. Independent mechanism-instrument critique blocks
actual AMP update interpretation without an independent decomposition bound;
minimal adapter v2 correction is building, no actual capture/science yet.
Current worker states are `round5/agents.json`. Latest20:50source checkpoint,
310numerical artifacts (785,715,438bytes) and13privatecontext files were actually
restored/hash-verified paused at20:54–20:55; read `CURRENT_RECOVERY.md`. Later
drafts remain durable workspace files and need the next checkpoint. Watcher healthy,
no new OOM/crash, no activeGPU or queuedtraining. Goal still active until23:00.

At20:20 Kyiv, all width320 and combo fold2/3 fits and scoring completed successfully;
no training or GPU job is active. E-362–E-383 preserve mechanism probes and reviews.
`FOLD23_DECISION.json` records all four frozen adoption gates PASS: pooled gain
.00178777, component95% interval[.00092491,.00274712], public-like+.00187316,
pooled class3+.00325245; fold2 class3−.00033410, fold3+.00677881.
The unchanged combo recipe is supported for subsequent controlled work;
individual-mechanism and public-score claims remain unproven.

At20:43 read current `round5/CONTRACT.md` and `round5/agents.json`. E-384 independently
reproduces the negative peer gate; E-385 blocks v2 because NumPy RNG dtype aliases
are silently coerced. Its final moment-dtype amendment is E-389; original E-385
report/spec bytes were recovered exactly into `evidence_versions/E385`, unchanged
ledger hashes verified. E-390 minimal v3 is terminal frozen with66CPUtests and
20protected file hashes verified; independent v3 review active, no CUDA receipt
issued. Controller wrapper selects v3, TERM585/KILL5, outer600s. E-386 independently checks combo gate/member/scorer route,
without bulk real-score recomputation. E-387 freezes a runnable loss-geometry
instrument before actual captures; E-388 audits input clocks/support and does not
rescue the rejected peer feature. Tiny credit-vs-cadence controls and an independent
instrument critique are active. `CURRENT_RECOVERY.md` names the actually verified
20:22 checkpoint, not these later drafts.

Read `round4/REFLECTION.md` for the earlier portfolio. V1 exact-state capture
was independently rejected for optimizer identity, exact clocks, and empirical
CUDA floor gaps. Its evidence is preserved. V2 fixes are frozen with37CPUtests and
six original baseline restores per control; independent v2 review is active.
No CUDA receipt or smoke has been issued. The controller wrapper now selects v2
with TERM585s/KILL5s and campaign outer600s; runtime changes are separate from
frozen actor evidence. Peer innovation failed its preregistered proxy gate and
independent review is active; no filter/horizon search follows that negative.

Deadline remains23:00 Europe/Kyiv. The persistent minute watcher is active with
no new crash/OOM, historical OOM counter11, no degraded monitoring or barrier.
Aggregate limits38.65decimalGB RAM/12CPU, GPU8CPU24GiB/allocator4.8decimalGB/
observed5.5decimalGB; four1CPU2GiB CPU slots. Use wrappers for all compute.
Backups stream numerical artifacts in960MiB parts, including opaque state.pt;
600MiB single-member restore passed with23MB processRSS. New snapshots restore
paused only; datasets, predictions and training checkpoints remain external.
Existing first-round completion statements below are historical, not goal completion.

All ten original independent framings finished as Codex R&D agents using
**gpt-6.1-sol, medium**, with runnable probes, notebooks, figures, provenance and
negative results. The optimizer builder, strong-baseline falsifier and reviewers
also completed. `agents.json` records thread IDs and report hashes. Completed
threads are closed and can be resumed with their retained context.

Read `RESEARCH_SYNTHESIS.md` for the combined interpretation. Best reported public
score remains **0.6641**; the >=0.6660 goal is unproven.

## Results and decisions

Muon + chunk1000 + stopped bootstrap replicated against matched full-schedule
controls on seeds3/4 and folds0/1. The equal-seed bag gains **0.001854** pooled
WP, component95% interval [0.000420,0.003409]. Fold1 is weaker and class3 loses
there. Incremental benefit over Muon alone is unconfirmed. See
`../scores/combo_matched_2seed_report.json`, ledger E-359.

The independent causal-state correction loses0.000237 mean WP on the strong
student despite its weak-ridge gain; close that configuration (E-360). Event-clock,
carry, physical decoding and phase proposals fail their controls. Puzzle reports
expose narrow identities and concentrated recurrence gains; neither establishes
a broad improvement. Preserve their negative evidence.

Optimizer switching, warmup/SGDR, module clipping, head LR, noise and sampling
controls are implemented in `../opt2/`. The trainer patch remains **unapplied**.
The builder's81 CPU tests/six synthetic fits passed. Independent review exposed
normalization confounding and a launch helper bypassing admission; both are
corrected in the prototype with39 focused CPU tests passing. Full GPU/fused/EMA
parity is pending. Exact zoneout and QMC batch integration are deferred. See
`../opt2/PLAN.txt`, `../opt2/CONTROLLER_CORRECTIONS.txt`, `opt2_review/REVIEW.txt`.

## Running work and ETA

`c20261003_w320_s3_f0` is healthy, training four epochs under the repaired launcher.
`wnn-codex-research-queue-v2.service` continues the sequential queue in
`research/campaign_2026-10-01/jobs/queue6.txt`: width320 fold1 next, then frozen
combo seed3 folds2/3. No optimizer jobs were queued.
`wnn-codex-fold23-scores.service` waits and scores each completed pair automatically.
The matched seed4 scoring service completed successfully.

`ETA.json` records measured progress and forecasts in Europe/Kyiv time. Width320
takes approximately65-80min/run from current measured pace; combo folds roughly
45-60min/run from prior matched fits. Export/load variability and pauses extend
ETAs. Each queued GPU job has a6000s execution limit. Recompute with the CPU
wrapper and `update_eta.py` before reporting a fresh forecast.

## Resource control and recovery

Aggregate campaign limit:12 cores,36GiB RAM (38.65 decimalGB), MemoryHigh34GiB,
no swap. GPU monitor/execution share one8-core/24GiB family; four CPU slots each
use1core/2GiB. CPU compute uses `run_cpu.sh`; GPU work uses campaign_job's serialized
admission and `run_gpu.sh`. Observed device VRAM cap5.5 decimalGB, allocator4.8GB;
existing recipes retain4.6GB. Limits include child processes.

`wnn-codex-research-watch.service` persists and checks once per minute: job/terminal
records, user/kernel journals, vanished units and aggregate OOM deltas. Watcher
status and `logs/incidents.jsonl`/`logs/journal_events.jsonl` retain observations.
Durable incident IDs, cursors
and acknowledgements survive restarts. Failures pause new admissions. The watcher
records evidence and pauses; an active controller diagnoses and chooses improved
retry conditions. It does not blindly restart failed training. See
`RECOVERY_POLICY.txt` and `RUNTIME_MIGRATION.md`.

The scorer admission incident ran no Python before failure. Historical failures
were acknowledged explicitly; the scorer was rerun after repair and finished.
Original failures/logs remain intact. No new training OOM was observed in this
continuation. Reconcile actual units before clearing any future barrier.

## Restoring progress

Original Claude session:`7ea48a59-c833-4cf5-b1fe-467c6d7c1ac5` under
`~/.claude/projects/-home-vlad-development-wunder-wnn-connectome-starterpack/`.
`claude_agents.json` inventories31 historical transcripts; the latest11 stopped
on weekly quota. `briefs/` preserves biases/task framing.

Keep the private initial894MB archive at
`snapshots/2026-10-03-153300/research-session.tar` for raw transcripts/older EDA.
`LATEST_SNAPSHOT` names the final bounded portable source/evidence archive; the
same directory has `progress-artifacts.tar` for new arrays/figures/HTML. Both have
checksums and manifests. `RESTORE.md` explains verified paused staging restoration.
Live logs are individual timestamped observations, not an atomic runtime checkpoint.
Dataset/model/prediction files remain external dependencies inventoried separately;
retain their backing files and verify hashes.

## Next step

Inspect watcher status/live units, review completed width320 and combo folds2/3
scores against frozen gates in `FOLD23_CONFIRMATION.txt`, and update ledger/bandit
before another arm. Prefer one narrow optimizer contrast with full-schedule/no-op
controls after GPU parity. Preserve existing native and nested `wunder/` work.
No submission upload occurred.
