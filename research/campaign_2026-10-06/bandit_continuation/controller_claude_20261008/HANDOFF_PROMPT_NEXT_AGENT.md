# Continuation prompt for the next research agent (Codex or other) — written 2026-10-08 19:15 UTC by the Claude Code controller

Paste everything below the line into the next agent. Every "live" fact must be re-verified; this is an index, not proof. When Claude returns it will read `CODEX_CONTINUATION_LOG.md` (see §9) and continue from there.

---

## 0. Mission and authority

You are the root research controller for **Wunder Connectome Challenge2**, repo `/home/vlad/development/wunder/wnn_connectome_starterpack`. Goal: a valid deterministic submission above our public best **0.6660** and teammate Maxim's **0.6674**; the user reports a real public **0.700** exists — plan toward it, try many ideas, do not reject directions because one component was weak (interactions matter). Research continues until the user explicitly says stop. Be autonomous within the rules; keep every paid/quota resource busy with useful work.

Binding rules (read in this order): `AGENTS.md` (= `CLAUDE.md`), `specs/README.md`, `specs/research-objective.md`, `specs/current-state.md`, `research/RESEARCH_METHODS.md`. Key points: our own independently trained weights and independently written code only (teammate code may be read, never copied); official metric `global_wp_clip2_v1` (`src/wnn/gmetric.py`, fixed mask, clip ±2); causal per-row serving with state reset per sequence; 1 vCPU / 60 min / 16 GB / 20 MB ZIP serving limits; public upload is done by the user only; Vast strict ownership — "touch only machines you rented. do not hurt existing, they are not mine"; existing Vast credit only, no top-up; Kaggle within quota.

## 1. Where the notes are

- Session controller dir: `research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/`
  - `CONTROLLER_PLAN_20261008.md` — decisions and timeline (append updates there).
  - `AGENT_RESOURCE_RULES.md` — **mandatory** for you and every subagent (memory caps, no whole-parquet reads, no docker, terse output).
  - `HOST_OOM_SESSION_CRASH_20261008_R1.json`, `SEED9_PARENT_AGGREGATE_BREACH_REVIEW_20261008_R1.json` — the two memory incidents of today and their fixes.
  - `NEW_HYPOTHESES_20261008.md` — H-274..H-283 cards; ideation round `research/ideas/claude_20261008_round/PLAN_TO_0700.md` (H-284..H-290).
  - Scout reports: `PIPELINE_MAP_20261008.md`, `PUBLIC_CV_HISTORY_20261008.md`, `SCREEN_PROTOCOL_20261008.md`, `STUDENT_READINESS_20261008.md`, `KAGGLE_SCOUT_20261008.md`, `WARM_RUNNER_SCOUT_20261008.md`, `ENGINE_TIMING_20261008.json`.
  - `COHORT1_FOLD0_STUDENT_ARMS.json`, `make_core_checkpoint.py`, `slice_filecache_guard_v1.sh`, `USER_AUTHORIZATION_AUTONOMOUS_COMPUTE_20261008_R1.json`.
- Append-only ledgers: `research/hypotheses.jsonl`, `research/experiments.jsonl`. Tree: `research/bandit_tree/generated/INDEX.md` (rebuild: `scripts/limited .venv/bin/python scripts/research_tree.py --root . build`).
- Analysis results today: `research/campaign_2026-10-06/bandit_continuation/blend_lab_H274/BLEND_LAB_RESULT_20261008.md`, `.../delayed_label_H281/H281_RESULT_*.md, H283_*, H284_*, H285_H289_*`.

## 2. Evidence you must not re-derive (fold0 468-group panel = repeatedly exposed development evidence; confirm on fold1 before claims)

| Fact | Number / source |
|---|---|
| Public best | 0.6660 `c20261004_combo_s4_i16` = one GRU256x2 trunk (`c20261004_combo_s4_final@ep3`, Muon+chunk1000+bootstrap, all data) + merged GCorr head-FT (`c20261004_combo_head_s4_final@ep0`), runtime 37m07s |
| Head-FT finishing | +0.0013 public on combo parents (E-457); +0.00083 local on S4 parent |
| H230 distilled students (never head-finished) | public 0.6644–0.6653; same-recipe seed pair +0.0004 public (61m22s runtime) |
| Fold0 students | C1 s3 0.681618, C1 pair 0.682176, H230 hetero+family s3 0.68172, family-off control 0.681658 |
| Blend lab | learned stackers/booster negative; diverse existing architectures (GRU384, attention, PLE, GRU128) add 0; only a 2nd GRU256 trunk helps (+0.0003..0.0008, CI crosses 0) |
| Runtime | single trunk ≈ 37–38 min; 2 GRU256 trunks ≈ 61–72 min (over limit; f16 only 4% faster) |
| H-281 | labels are NOT causally revealed (R² ≤ 0.19 for y_t from features at t+L≤100) — recalibration using past labels is a leakage oracle; invalid |
| H-283 / H-284 / H-289 | label-free post-hoc calibration −0.002..−0.005; class-3 spread identity not found in features (projection −0.011); shrink gate null |
| H-285 | strong-heavy teacher mix +0.0004 (component-CV, CI crosses 0) |

## 3. How to run and monitor each lane

### Local (RTX 3060 6 GB; campaign cgroup 5 CPU / 18 GB high / 20 GB max)
- Guard rules: `scripts/bandit_resources.py` kills a GPU fit if the whole `wnn-campaign.slice` (incl. page cache) exceeds 18 GB. `wnn-slice-filecache-guard-v1.service` reclaims file cache above 15.5 GB. Keep support processes ≤ 2.5 GB each, one at a time, always via `WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2500000000 WNN_RUNTIME_MAX=900s scripts/limited <cmd>`.
- **Warm queue (load-once, run-many):** worker attempt `warm-queue-local-v3-20261008` (unit `wnn-bandit-warm-queue-local-v3-20261008.service`, 2 CPU / 9 GB / 6 h from 18:37Z, idle-exit 30 min) runs `scripts/warm_queue_worker.py` over `research/campaign_2026-10-06/bandit_continuation/runtime/warm_queue_local_v1/` (`pending/ running/ done/ failed/ withdrawn/ logs/ results.jsonl worker_heartbeat.json`, `STOP` file stops claiming). Add work = drop a spec JSON in `pending/` (lowest `priority` first; steps = argv lists). Check: `cat .../worker_heartbeat.json; tail results.jsonl; tail logs/<run>.<step>.log`.
- Queue at handoff: running `z3-combo-s4-parent-student-v1` (student from incumbent parent, ETA ~19:45Z) → `z1c`/`z2c` head-FT on H230 S4/S3 students (core checkpoints) → `z3h` → `f0_combo_s3_parent_hft_v1` (fold0 incumbent analogue) → `f0_h230_hetero_s3_hft_v1` (fold0 student+head-FT) → seed99 faithful student → `z4` → seed9 faithful student → `z5`.
- Relaunching a worker when it exits: new attempt id (ids are single-use), reclaim file cache so slice ≤ 9 GB (`echo "6000000000 swappiness=0" > /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice/memory.reclaim`), then `scripts/limited .venv/bin/python scripts/bandit_resources.py --root $PWD --state $PWD/research/campaign_2026-10-06/bandit_continuation/runtime launch --attempt warm-queue-local-v4-<date> --kind gpu --cpus 2 --reserve-bytes 9000000000 --profile H230-ALLDATA-STUDENT-GRU256-SEED3-FIT --contract research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/WARM_QUEUE_LOCAL_V1_ROOT_AUTHORIZATION_20261008_R1.json --admission-timeout 1800 -- /usr/bin/timeout --signal=TERM --kill-after=30s 21600s $PWD/.venv/bin/python $PWD/scripts/warm_queue_worker.py --queue $PWD/research/campaign_2026-10-06/bandit_continuation/runtime/warm_queue_local_v1 --poll-seconds 20 --idle-exit-seconds 1800` (edit the authorization's attempt_id first). If admission is closed after a breach: write a review, then `bandit_resources.py ... resume --review "<text>"`.
- Head-FT on family-head students: main `scripts/train_rnn.py` has no family-head flags → first `python .../controller_claude_20261008/make_core_checkpoint.py SRC_RUN 1 SRC_RUN_core` (drops the 8 appended family rows of head.2, trunk bitwise identical), then the incumbent head-FT recipe: `--init CORE@ep1 --train-source valid --full-valid --mask-weight --seed 3 --epochs 1 --batch 16 --chunk 2000 --fast --inherit-init-preprocessing --loss gcorr --optim adamw --freeze inp,gru --lr 0.0005 --wd 0.01 --checkpoint-policy fixed --checkpoint-epoch 0 --save-training-state --no-eval-during-fit` (+ `--split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json` for fold0).
- Scoring: `scripts/score_fold0_panel.py RUN@epE --reference RUN@EPOCH --reference "A@ep1*0.5,B@ep0*0.5" --out X.json` (needs `cache/pred_<run>_valid.{npy,json}` from `scripts/predict_run.py RUN valid --epoch E`; aliases by symlink allowed when sidecar run/epoch match). Cohort spec generator: `scripts/make_cohort_specs.py ARMS.json --root R --python PY --queue Q`.
- Monitors: `wnn-campaign-kaggle-poller.timer` (60 s Kaggle status/quota → `research/campaign_2026-10-06/cloud_preflight/provider_poll_log.jsonl`), `wnn-research-monitor.timer`. Your own session monitors die with your session.

### Vast (paid; must never idle)
- Tool: `.venv/bin/python scripts/vast_campaign.py --ledger /home/vlad/.local/state/wnn-campaign/vast/ledger.json {inspect|search|status --attempt|reconcile --attempt|settle --attempt --receipt|run ...}` (credential loaded internally from `.env`; never print secrets/URLs).
- Live: attempt `e529-vast-warm-smoke-c1v2-f0-20261008-r4`, instance **54894762** (RTX 3090, Thailand, $0.3053/h, 64 GB), reserved $2.00, supervisor unit `wnn-vast-warm-smoke-c1v2-r4-v2-20261008`, **hard stop 21:59:24Z**, phase at handoff: `staging_inputs` since 18:49Z (35 GB parquet over plain upload because signed prefetch was skipped for "insufficient runway" — slow). Packet: `research/campaign_2026-10-06/bandit_continuation/vast_warm_cohort/` (smoke spec + intent + admissions r2/r3/r4). Cohort specs V00–V09 (C1v2 fold0 variants + GCorr head-FT V09) and a self-requeueing HOLD spec (keeps the worker from idle-exiting until hard stop − 30 min) at `.runtime/e529-vast-warm-smoke-c1v2-f0-20261008-r4/cohort_v1/`; push them into the remote queue `pending/` over the supervisor's SSH key as soon as staging completes (read host/port via read-only provider instance details for our receipt-owned instance only). Make cohort specs copy grade/pred/provenance into the bundled directory so collection brings them home.
- Monitor: `systemctl --user status wnn-vast-warm-smoke-c1v2-r4-v2-20261008`, `journalctl --user -u <unit> -n 30`, ledger status for the attempt, `vast_campaign.py inspect` (instance list, credit). After terminal: verify `destroyed_verified`, read billing (campaign client GET `/users/me/invoices/` → private file under `~/.local/state/wnn-campaign/vast/`), settle with a root receipt (`settle`; schema in `continuous_execution_E529/vast/SETTLEMENT_*_20261008_R1.json`).
- Lessons: offers vanish within minutes → bind the offer at dispatch time (normalized `vc._offer_snapshot`), admission expires in 120 s, include `spending_authority_sha256` (ADR0015); never retry after an ambiguous/ApiError create (reconcile only); no extend path → choose a long hard stop and long idle-exit at creation; prefer signed Kaggle prefetch for staging (fast, cheap) over laptop upload; worker host RAM ≥ 1.25 × worker ceiling.
- **Next rental (r5, long-lived):** hard stop ≥ 10 h, idle-exit ≥ 2 h, HOLD spec, signed prefetch staging, one 3090/4090 ≤ $0.35/h (or two machines with disjoint queues), reservation ≈ $4–5. Ledger committed was $1.79 before r4; credit ≈ $17.9. Queue it with the Vast cohort list in §5 (≈ 20–30 runs) so it stays busy for many hours.

### Kaggle (GPU quota ≈ 11 h until 2026-10-10 00:00 UTC; TPU ≈ 20 h)
- CLI `.venv/bin/kaggle` 2.2.4 (credentials configured). Warm-cohort kernel code `scripts/kaggle_warm_cohort/` (merged), source dataset `vladkochetov07/wnn-warm-cohort-src-20261008`, data `vladkochetov07/wunder-funder` (33.7 GB raw parquet), assets `wnn-c20261004-fold0-assets`, `wnn-c20261004-fold1-assets`.
- Live: smoke `vladkochetov07/wnn-warm-cohort-smoke-20261008-v1` v1 and v2 ended ERROR with EMPTY logs and null failureMessage (never started); probe kernel `vladkochetov07/wnn-warm-cohort-probe-20261008-v1` (no datasets, trivial script) queued to isolate mount vs account/kernel-setting causes. Compare `kernel-metadata.json` with the last COMPLETED kernel `h232-gru384-b8-amp4096-conditional-profile-fit-v4` (packet `continuous_execution_E529/h232_conditional_v4/packet`). If the probe completes, reduce mounts (fold assets + valid only) and stage train parquet differently; if it fails, check accelerator/visibility settings and account phone verification.
- Monitor: `kaggle kernels status <slug>`, `kaggle kernels output <slug> -p <dir>` (via scripts/limited), poller log. TPU V6 failed on an over-strict input-inventory check (`continuous_execution_E529/E463_TPU_V6_TERMINAL_OUTPUTS_20261008_R1/`); JAX TPU path is unproven — low priority.

### Marimo
- Server `wnn-marimo-blend-lab-20261008` was stopped to free 1.7 GB for the GPU fit; agents could not discover its session. Notebook to continue: `research/campaign_2026-10-06/bandit_continuation/blend_lab_H274/blend_lab.py`; the real analysis lives in `blend_lab_analysis.py`, `blend_lab_architecture.py`, `delayed_label_H281/*.py`. Restart only when no local fit is near the memory limit: `systemd-run --user --unit=wnn-marimo-blend-lab-<date> --slice=wnn-campaign.slice -p MemoryMax=3000000000 -p CPUQuota=50% -p MemorySwapMax=0 --working-directory=$PWD .venv/bin/marimo edit --headless --no-token --host 127.0.0.1 --port 2718 research/campaign_2026-10-06/bandit_continuation/blend_lab_H274/blend_lab.py`, then open http://localhost:2718 (a browser tab must connect before a kernel session exists). Save plots in the lineage `plots/` folders.

## 4. Submission candidates and decision rule
- Packager output: `submissions/c20261008_<zN>_combo_i16/` (pkg, submission.zip, `RELEASE_CHECK.json`). Members are `CORE@ep1|HFT@ep0`, merged heads, i16, incumbent native builds and solution.py. Gates: tensor check, `scripts/check_submission_abi.py`, `scripts/check_submission_python311.py` (py311 env in the packager's scratchpad; rebuild with `/usr/bin/python3.11 -m venv` + numpy/pyarrow if missing), timed replay on `cache/valid_subset20.parquet` (incumbent 2.21 s/seq ↔ 37m07s).
  - Z1 = H230 S4 student core + head-FT; Z2 = H230 S3; Z3 = student from incumbent parent `c20261004_combo_s4_final@ep3` + head-FT; Z4/Z5 = faithful seed99/seed9 Muon-parent students + head-FT.
- Decide with fold0 analogues: `artifacts/runs/f0_h230_hetero_s3_hft_v1/grade_fold0.json` (student+head-FT) vs `f0_combo_s3_parent_hft_v1` (incumbent analogue: parent+head-FT). If student+head-FT ≥ parent+head-FT (paired component bootstrap), recommend Z3 first (strongest parent), then Z4/Z1; else keep incumbent and prioritize teacher/capacity work. Give the user ZIP path + sha + RELEASE_CHECK; the user uploads (≤5/day). Record public results as receipts.

## 5. Experiment backlog (register each as hypothesis/experiment; one-flag matched controls; promote only after fold1)

**Local queue additions (after current queue):** head-FT variants on the best student core (lr 2e-4/1e-3; 2 epochs; `--loss gcorr_pool`; `--loss wmse_gcorr`); 3-head merge (parent head + GCorr head + wmse-FT head) packaged and timed; fold0 incumbent analogue seed4 (`c20261003_combo_s4_f0` + head-FT); strong-heavy teacher cache variant (0.73125 strong / 0.08125 full, H-285) + one all-data student; organizer-baseline teacher slot (H-279; `cache/pred_baseline_*`).

**Vast long cohort (fold0, inputs: combo_s3_f0/s4_f0 inits + teacher preds + propensity + split + valid_targets; ~18–25 min each on a 3090):**
1–10. V00–V09 (C1v2 control, seed4, alpha 0.5/0.9, lr 2e-4, 3 epochs, chunk 1000, boot-h 8, Muon student, GCorr head-FT on V00).
11. V00 + head-FT with lr 1e-3; 12. head-FT 2 epochs; 13. head-FT `gcorr_pool`; 14. `wmse_gcorr` student loss; 15. EMA/SWA of student ep0+ep1 then head-FT; 16. distill-alpha far/near schedule (`--distill-alpha-far`); 17. student from combo_s3_f0 with 4 epochs at lr 5e-5 (longer finish); 18. batch 32 + lr ×1.4; 19. `--update-rows`/detach variants (truncated BPTT 500); 20. seed4 versions of the best three arms (pairing for confidence); 21–24. fold1 confirmations of the two best arms (upload fold1 assets: `wnn-c20261004-fold1-assets` contents); 25. capacity teacher GRU512 fold0 (4 epochs, Muon, chunk1000; ~2–3 h) for later distillation; 26. GRU384 student distilled from the same teachers (capacity in the student; check runtime with H-282).

**Kaggle (once a kernel actually starts):** mirror Vast arms 11–20 on the two T4s, or run fold1 confirmations.

**Bigger ideas (need code; delegate to builders):** H-282 int8/AVX2-VNNI native kernel to fit 2 trunks (then a diverse pair or GRU320 student); H-280 diverse-ensemble distillation into the served GRU (teachers: GRU512/768, attention, TCN — teachers only); H-286 i0/i1 cointegrating-residual feature; H-288 low-rank wide student (`--recurrent-rank`); H-290 late optimizer switch; H-287 rank-4 innovation basis; cross-instrument lead–lag, order-flow imbalance, event-time features; class-weighted loss matching public class mix [0.183,0.16,0.198,0.459]; born-again self-distillation; teacher-disagreement sample weights; pair-to-single distillation.

## 6. Subagents (Codex harness)
Follow AGENTS.md "Agent roles and invocation": spawn with `collaboration.spawn_agent`, distinct task name, `model="gpt-6-luna"`, `reasoning_effort="max"`, `fork_turns="none"`; ≤ 3 active subordinates; no Sol/Astra without explicit user confirmation. Give each a self-contained assignment (basis, scope, owner, deliverable, budget) and paste `AGENT_RESOURCE_RULES.md` rules verbatim. Suggested roles: (1) **cloud operator** — owns Vast r4→r5 lifecycle (push cohort, hold spec, collect, destroy, settle) and Kaggle probe/fix/cohort; (2) **packager/grader** — packages Z-candidates, runs gates, grades fold0/fold1 contrasts with `score_fold0_panel.py`; (3) **builder** — one code task at a time from §5 "Bigger ideas" (start with H-282 microbenchmark or the strong-heavy cache variant). Root keeps admission, spending approval and synthesis; agents never approve themselves.

## 7. Safety checklist (every hour)
`systemctl --user list-units 'wnn-*'`; slice memory `cat /sys/fs/cgroup/.../wnn-campaign.slice/memory.current` and `memory.events` (oom_kill count was 16 at handoff); `nvidia-smi`; `df -h /home` (**98% full, ~32 GB free** — keep big outputs on cloud, ask the user before deleting anything); Vast `inspect`; Kaggle poller tail.

## 8. Immediate next actions (in order)
1. Verify local worker v3 is running and the queue is advancing; when Z1c/Z2c/Z3H land, make sure the packager produces ZIPs and RELEASE_CHECKs.
2. Watch r4 staging; the moment it reaches the worker phase, push HOLD + V00, V09, V01 … over SSH; collect before 21:59Z; verify destroy; settle.
3. Prepare r5 (long-lived) with the §5 Vast cohort and signed prefetch; dispatch right after r4 is collected.
4. Resolve the Kaggle empty-log failure via the probe; then run arms 11–20 or fold1 confirmations.
5. When both fold0 head-FT grades land, apply §4 and hand the user the best ZIP.

## 9. Catch-up protocol for Claude
Append every action/result (UTC time, what, where, outcome) to `research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/CODEX_CONTINUATION_LOG.md`; append durable results to `research/experiments.jsonl`; update `specs/current-state.md` at meaningful results; commit scoped files (never `git add -A`; ~1270 unrelated dirty entries exist). Keep attempt ids, unit names and receipt paths in the log so Claude can resume without re-deriving.
