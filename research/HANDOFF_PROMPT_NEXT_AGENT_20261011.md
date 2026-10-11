# HANDOFF PROMPT — continue the Wunder Connectome Challenge 2 research (written 2026-10-11 ~12:30 EEST)

You are taking over a running research campaign from another Claude Code session. **Read this whole file first, then do the "First 30 minutes" checklist, then run the "Loop pass" every time the user says `status? utilization? ...` (or every ~15-20 min if working autonomously).** Do not skip the rules in section 2: each one exists because something broke or was wasted.

> **UPDATE 2026-10-11 ~15:50 (reviewer):** (1) Vast **auto top-up is now DISABLED** (see CLAUDE.md); the "~$1/hour auto top-up / cap 70" statements in this file are SUPERSEDED: collect artifacts early, no top-up, no new rental, no blind retries. (2) CLAUDE.md now says **do not select submissions from fold 1 alone**: rank complete endpoints by cross-fold evidence with distinct seeds and clean ancestry (ADR0031); the owner agent holds the live state. (3) The public results of the fold-1-held-out ZIPs and the review are in `diversity_stack_20261009/review_public_gap_20261011/REVIEW_PUBLIC_GAP.md` (read sections 11-13 first; the corrections there override sections 4, 6, 7).

Working directory: `/home/vlad/development/wunder/wnn_connectome_starterpack` (git repo, branch `main`, remote `origin` on GitHub; commit AND push every meaningful step). Use `rg` not `grep`. Reply in a terse style (the user likes caveman-short status). Code, commits and security text are written normally.

---------------------------------------------------------------------
## 0. Mission and how success is measured

* Task: Wunder Connectome Challenge 2 — causal per-row regression on market streams, metric `global_wp_clip2_v1` (pooled weighted Pearson, weights |clip(y,2)|, rows >= 99, both targets t0,t1). Never change the metric or its clipping.
* Objective: **raise the PUBLIC leaderboard score.** Public best (user-verified) = **0.6668** (`4_z1c_z4_pair_q8vnni.zip`); leaderboard top ~0.6717; the user wants top-1 / >0.70 as active goals. Fold-1 validation (component-disjoint, 457 sequences) is *exposed* local evidence for mechanisms; it is not a forecast. Never add a local delta to a public score; label expected gains as judgment with uncertainty.
* Selection rule (user): **select on the average over seeds, not on a CI lower bound** (memory file `select-on-average-not-lower-bound.md`). Seed sd is ~0.001-0.002 (3-epoch) and up to 0.005 (4+ epochs); **no claim from one seed; need n >= 3 for "recipe A > recipe B"** (we retracted two single-seed epoch-ladder claims).
* The user uploads submissions to the portal (5/day) and tells you the scores. You prepare ZIPs and a ranked recommendation; you never claim a public score and never upload.
* Compute (all must stay supplied with useful experiments until the user says "stop"): local GPU (RTX-class, 5 GB VRAM cap, fits run ~35-45 min each), Kaggle 2xT4 kernel, **Vast lane B** (one rented 2xRTX24GB host; the ONLY Vast instance you may touch). User allowed ~$1/hour Vast spend via auto top-up; lane-b tick cap is 70. Do not rent new Vast machines.

---------------------------------------------------------------------
## 1. Where everything is (read in this order)

1. `CLAUDE.md` (repo root) — project rules, roles, skills table, Kaggle/Vast ownership rules. It is binding.
2. This file.
3. **Deep dive of the current best and all findings:** `research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/DEEP_DIVE_CURRENT_BEST_20261011.md`
4. **Chronological log (read the last ~10 passes):** `research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md`
5. **Idea ledger (status of every idea A-F):** `research/IDEA_LEDGER_20261011.md`
6. **Hypotheses (append-only, `H-xxx`, parents/status/evidence):** `research/hypotheses.jsonl`
7. **Incident log + lessons:** `.../diversity_stack_20261009/CRASH_POSTMORTEM_20261011.md` (read all of it once).
8. Research index: `research/RESEARCH_NOTES_INDEX_20261010.md`; generated tree: `research/bandit_tree/generated/INDEX.md` (rebuild: `scripts/limited .venv/bin/python scripts/research_tree.py --root . build`); methods: `research/RESEARCH_METHODS.md`; specs: `specs/README.md`, `specs/current-state.md`, `specs/research-objective.md`, `specs/experiment-execution.md`, `specs/idea-generation.md`, `specs/agent-contract.md`; ADRs `specs/adr/`.
9. Topic notes (all under `.../diversity_stack_20261009/`): `SCALING_DEEP_DIVE_20261011.md` (does scaling width/epochs/LR work), `DISCORD_TRICKS_DEEP_DIVE_20261011.md`, `LR_AND_DATA_SCALING_LITERATURE_20261011.md`, `eda_residuals_20261011/RESULTS.md` (residual surprises), `distillation_deep_dive_20261011/{LITERATURE.md,CEILING_AND_PLAN.md,CACHE_ONLY_DIAGNOSTICS.md,ideation/}`, `gbm_teacher_20261011/RESULTS.md`, `blend_probe_20261011/RESULTS.md`, `PUBLIC_SCORES_20261011.md`, `SCORER_CPU_SPEC_20261010.md`, `solution_report/CURRENT_BEST_SOLUTION_EXPLAINED.md` (how the model/data/training work, with plots), `external_ideation_20261010/` (Grok/OpenCode/Freebuff rounds).
10. Native serving: `native/INT8_VNNI_REPORT.md`, packaging: `packaging_q8vnni_20261010/` (`build_upload3.py`, `run_gates_upload3.sh`, `RESULTS.md`), ZIPs in `submissions/upload_2026-10-11/` (git-ignored; `SHA256SUMS`).
11. Memory (auto-loaded context, also readable): `/home/vlad/.claude/projects/-home-vlad-development-wunder-wnn-connectome-starterpack/memory/MEMORY.md` and files it lists.
12. Skills (read `SKILL.md` before use): `.agents/skills/{conduct-creative-research,research-portfolio-control,research-compute-throughput,research-recipe-audit,research-evidence-review,research-experiment-delivery,research-reframe,python-standards,wnn-connectome-top1,vastai,vastai-sdk,claude-autoresearch-skill}/SKILL.md`.

### When to use which skill (proactively)
| Situation | Skill |
|---|---|
| Starting a new direction / writing a contract / deciding how to test an idea | `conduct-creative-research` (contract, falsifiers, actor/critic split, lenses: re-encode, factor, mine surprises, instrument, transfer, invert, relax) |
| Choosing which lineage gets GPU time, renewing a stale direction, interaction nodes | `research-portfolio-control` |
| Lanes idle, slow loading, deciding cohorts / fan-out | `research-compute-throughput` |
| Before any fit that claims to replicate a method/recipe (equation-to-code, exact command) | `research-recipe-audit` |
| Before believing a surprising gain (leakage? wrong comparison? exposure?) | `research-evidence-review` |
| Launching and repairing remote/local jobs end to end | `research-experiment-delivery` |
| Fresh-context rethinking when a family plateaus | `research-reframe` |
| Anything touching Vast | `vastai` (CLI) / `vastai-sdk` — but ONLY on the receipt-proven instance (see section 4) |
| Python changes | `python-standards` |
| Metric / serving constraints | `wnn-connectome-top1` |
| EDA | marimo-pair (`~/.agents/skills/marimo-pair/SKILL.md`), save PNG under the lineage's `plots/` |

---------------------------------------------------------------------
## 2. RULES. Prohibitions first (each one is a past incident)

DO NOT:
1. **Do not run HPO** (user stopped it; `.runtime/hpo_disabled` exists). No hyper-parameter sweeps of any kind.
2. **Do not rent, stop, destroy, reconfigure or connect to any Vast instance except the lane-b instance via `controller.py`** (`VT_LANE=b`). Matching labels are not proof of ownership. No bulk cleanup. No new instances.
3. **Do not copy teammate code.** Teammate methods may be read for ideas and re-implemented independently (ADR0012). Do not use teammate weights.
4. **Do not change the official metric or clipping**, and do not make anything non-causal (serving must be causal, deterministic, reset per sequence).
5. **Do not claim a result from one seed.** Never write "X > Y" unless n >= 3 seeds each or the effect is >> 0.003 and reproduced on a second machine. Never say "new best" without separating "local fold-1 candidate" from "public score".
6. **Do not edit any file under `scripts/`, `src/wnn/` or `submission/` while a local-queue item is between reservation and start** (the local runner hashes ALL of them; a change fails the item with "frozen source changed"). Put new analysis scripts under `research/.../` (not `scripts/`), or batch edits and then requeue.
7. **Do not put large files in `/tmp`** (it is RAM-backed tmpfs and counts as non-reclaimable memory in `wnn-campaign.slice`; 3.5 GB of scratch files closed local admission once). Use research dirs.
8. **Do not run CPU jobs/agents above 1 core / 3 GB RAM each**, aggregate local limits are 5 CPU cores, 18 GB working RAM (20 GB hard), 5 GB VRAM. Run analysis through `scripts/limited` (see 5.6). Keep <= 3 sub-agents alive; Haiku subagents are fine for mining; Astra needs explicit user OK.
9. **Do not hand-rename Vast pending job files** except with the `p00_`-style prefix (the controller strips `^p\d\d_` only; other renames cause duplicate shipping).
10. **Do not delete data you cannot regenerate.** The local `cache/pred_vt_*_valid.npy` files are now the ONLY copy for ~90 runs (host copies were removed). The five 13 GB `cache/*distill*`/`bd26_*` directories can be deleted only if the user says so.
11. **Do not use `sleep` chains or poll loops in the foreground** (the harness blocks them); use `timeout N bash -c 'until ...; do sleep 10; done'` with a bound, or `run_in_background`.
12. **Do not use `rm` with an unset variable path** (use `rm -rf "${VAR:?}/x"`); the safety hook blocks otherwise.
13. **Do not print, commit or upload secrets** (`.env`, Vast key, Kaggle credentials). Do not commit large binaries (ZIPs, npy, checkpoints); check `git status` before `git add -A` on broad directories.
14. **Do not leave a lane without queued work**, and do not fill it with uninformative repeats. Each queued run must have a hypothesis id and a decision rule.
15. **Do not treat provider "RUNNING"/`ready` as success.** Verify the log, the output sizes, the job names in the kernel status JSON.
16. **Do not re-open closed directions without a new mechanism:** long scratch training / small-LR 20-30 epochs, HPO, next-row pretraining, highway head, anti-curriculum, time-warp at 3 epochs, clockbank/dualtower/dssm, GBM windowed teacher, data-size / wd-rescale, batch-size rules, lwfloor on top of prop0, equal-weight blends with weaker partners (full list with numbers in the DEEP_DIVE file section 4).

ALWAYS:
1. Run `scripts/alerts_unread.sh` FIRST in every pass and react to every ALERT (a wrong "stalled"/"missing timer" alert is also a bug to fix in `scripts/watchdog.py`, see 6).
2. Write evidence to the notes the same hour (LOOP_LOG pass entry, ledger row, hypothesis line, post-mortem for incidents); then `git add <specific files>; git commit; git push`. Commit message ends with the attribution line required by the harness.
3. Correct the record when you were wrong (write a "RETRACTION" paragraph; update the hypothesis `status`).
4. Report to the user: status of every lane (GPU %, queue depth, spend), results landed (with n seeds), what you changed, what is blocked on them, in a few lines.
5. Preserve unrelated dirty files (`git status` shows many log files modified by services; do not revert them, do not `git add -A` blindly).

---------------------------------------------------------------------
## 3. FIRST 30 MINUTES CHECKLIST

1. `date`; `git status --short | head`; `git pull --rebase` only if clean (otherwise skip; services keep logs dirty).
2. `scripts/alerts_unread.sh` (prints unseen ALERT/RECOVERED rows; cursor in `.runtime/alerts_cursor`).
3. Read the last 150 lines of `LOOP_LOG_20261010.md`, `IDEA_LEDGER_20261011.md` section F, tail of `CRASH_POSTMORTEM_20261011.md`.
4. Lane snapshot (commands in section 5): local GPU+queue+barrier, Kaggle status, Vast dash/status/pending/running/disk.
5. Score anything that landed since the last log entry (section 5.4) and log it.
6. Make sure each lane has queued work (section 5.7); ensure the all-data finals `fin_mu256p0e4_s3/s4/s5` actually started (see section 8, item 1).
7. Tell the user the status in <= 15 lines.

---------------------------------------------------------------------
## 4. CURRENT STATE SNAPSHOT (12:30 EEST 2026-10-11; verify live, then update)

Best: public 0.6668 (pair of two H230 all-data students, int8). Best fold-1 servable single: **h256 GRU, Muon on ALL 2-D weights (`--muon-scope all`), `--prop-power 0`, 4 epochs: 0.66976 / 0.67195 (mean 0.6709, n=2)**; 3-epoch version 0.6690 (n=2). Best distilled student 0.67103 (3 muon-all h384 teachers, alpha 0.9, 6 epochs). Details and every number: DEEP_DIVE file.

Running/queued (names are job files; arm names `vt_f1_<id>` in `artifacts/runs/` on the host):
* Vast B running: `w17_dm_a95_e6`, `w17_dm_a9_e10`, `w18_dm_pm_a9_e6` (student with muon-all h256 parent), `w18_dm_ba_a9_e6` (born-again), `w19_kd_ch02`, `bl_muonall_h384_e3_s10`, `bl_data8400_e4_s3`.
* Vast B pending: `fin_mu256p0e4_s3/s4/s5` (**all-data finals, highest priority**), `w19_kd_ch052/msepear/pearson/run05` (KD-loss variants from the v10 trainer: diff-channel weight, Pearson, run-level), `bl_muonall_h384_e3_s8/s9`, `w14_ood_*` x5 (OOD/augmented-path distillation, v9), `w15b_dm_h384_t3`, scale_b64 x2.
* Kaggle `vladkochetov07/wnn-kaggle-long-20261010-a` version 6 running: `kg_mu256_prop0_s5/s6/s7`, `kg_mu256_prop0_e4_s5`.
* Local queue (`local_queue_r1/pending`): `muonall256prop0` s5, `muonall256prop0lwf` s3 (fit+export each).
* Open user decisions: (a) delete five old 13 GB distill caches in `cache/` (local `/home` is 90% full, 132 GB free)? (b) which ZIPs to upload (you propose, they decide). Vast spend authorized (~$1/h, cap 70).
* Session-only cron jobs and `/loop` of the previous session die with it: **the user will re-send the loop prompt `status? utilization? reflect on research results landed, write to research notes, continue and make robusltly better and research promising directions or what i have made`; each time run the section-7 loop pass.**

---------------------------------------------------------------------
## 5. HOW TO OPERATE (copy-paste recipes, all verified in this campaign)

### 5.1 Local GPU queue
* Queue a fold-1 3-epoch run (fit+export cloned from the control template): `python3 scripts/queue_local_experiment.py --tag NAME --seed 3 --extra "--muon-scope all --prop-power 0" --note "why"`. The export epoch follows `--checkpoint-epoch` in `--extra`. Export only: `--tag NAMEx --seed 3 --export-only RUNNAME --epoch N`. Fits reserve 10.5 GB RAM (admission needs measured slice RAM <= 7.5 GB; the runner reclaims page cache and retries every 2 min — a gap of 2-4 min between launches is normal). **Local lane hosts h256 only** (h384 does not fit).
* State: `research/campaign_2026-10-06/bandit_continuation/local_queue_r1/{pending,done,failed,blocked,retired}`, log `local_queue_r1/runner_log.jsonl`, per-attempt records `runtime/attempts/<attempt>.json` (`failure`, `peak_ram_bytes`), logs `runtime/logs/<attempt>.log`. Timer: `systemctl --user status wnn-local-queue-r1.timer`.
* **Admission barrier**: `runtime/STOP_ADMISSION.json` exists => nothing launches. Read `reason`; fix cause; then reopen with a written review:
  `WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s scripts/limited .venv/bin/python scripts/bandit_resources.py --state research/campaign_2026-10-06/bandit_continuation/runtime resume --review "what broke, why safe now"`
  (fails if an attempt is still active). A failed item moves to `failed/` with its export in `blocked/`; requeue with a NEW tag (attempt names must be unique).

### 5.2 Vast lane B (controller)
```
cd research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1
VT_LANE=b python3 controller.py dash      # GPU util/mem, spend, pending/running, done list
VT_LANE=b python3 controller.py status    # remote status JSON
VT_LANE=b python3 controller.py feed --n 4   # move best 'ready' items of backlog/IDEA_BACKLOG.json to refill_jobs_b/ and ship
VT_LANE=b python3 controller.py queue     # ship new refill_jobs_b/*.json to the host (idempotent)
VT_LANE=b python3 controller.py reorder   # family-interleave pending
VT_LANE=b python3 - <<'EOF'
import controller as c
print(c.ssh("ls /root/run/refill/running; ls /root/run/refill/pending; df -h /root|tail -1").stdout)
EOF
```
* Host layout: `/root/wnn` (code mirror: `scripts/`, `src/wnn/`, `artifacts/runs/<run>/{model_ep*.pt,history.json,provenance.json}`, `cache/pred_<run>_{valid,train}.{npy,json}`), `/root/run/refill/{pending,running,done,held}` (job JSONs; daemon `remote_refill.py` runs up to MAX_JOBS=7 jobs, needs >= 7000 MiB free on a GPU to start one, retries CUDA-OOM up to 3 times as `<name>.tryN`), `/root/run/logs/` (`<run>.fit.log`, `<run>.export.log`, `pipeline_<arm>.log`, `<job>.pipeline.log`).
* Job file = JSON `{name, steps:[{label,timeout,argv}]}`; completed files in `done/` are named `<job>.rcN.json` (rc0 success, rc1/rc2 failure — read the log). To jump the queue prefix the pending file with `p00_`.
* Add an experiment: append an item to `backlog/IDEA_BACKLOG.json` (copy a similar item; `status: "ready"`, `change` dict of train_rnn flags, `rationale` with the hypothesis id) and `feed`; or write the job file directly into `refill_jobs_b/` (see `make_refill_jobs.py`, `w19_*.json` for pipeline jobs) and `queue`.
* Deploy code to the host: scp into `/root/wnn_stage_v9/`, `python -m py_compile`, back up the old file, then `mv` into place (atomic). Never copy files directly onto a path a running job imports.
* Fetch a small file: `c.scp(["/root/wnn/artifacts/runs/<run>/model_ep3.pt"], "<local dir>", upload=False)`; **do not pull 286 MB prediction files** (the link is ~250 KB/s; pull timer is disabled). Score on the host instead (5.4).
* Watch spend: `dash` prints hours and $ (about $0.85/h); the tick service holds queued jobs at its `--cap` (70) and DESTROYS the owned instance when everything is drained — keep the queue non-empty or accept the teardown.
* Disk: host has 150 GB; distillation caches are 13 GB each and `build_distill_cache` needs 20 GB free beyond its output (>= 33 GB free to build one). Free space by deleting finished `cache/dc_*` dirs and host copies of `pred_*_valid.npy` that exist locally with identical byte size.

### 5.3 Kaggle (2xT4 kernel, ~1 h per h256/h384 fit, 4 arms per kernel version)
```
cd research/campaign_2026-10-06/bandit_continuation/kaggle_long_20261010
# 1) edit ARMS in build_long_packet.py (flag dict per arm; SNAP = trainer source dir, now h230_source_v8_arch)
cd <repo root> && .venv/bin/python research/campaign_2026-10-06/bandit_continuation/kaggle_long_20261010/build_long_packet.py
cd research/campaign_2026-10-06/bandit_continuation/kaggle_long_20261010
kaggle datasets version -p dataset -m "vN what"           # then poll: kaggle datasets status vladkochetov07/wnn-kaggle-long-bundle-20261010  (until 'ready'); check 'kaggle datasets files ...' size equals dataset/bundle.wnn
kaggle kernels push -p kernel                              # kernel vladkochetov07/wnn-kaggle-long-20261010-a
kaggle kernels status vladkochetov07/wnn-kaggle-long-20261010-a
mkdir -p output_vN; kaggle kernels output vladkochetov07/wnn-kaggle-long-20261010-a -p output_vN   # when COMPLETE; out/status.json lists per-job rc and seconds, out/pred_kg_<arm>_valid.npy
```
Score locally: copy `pred_kg_*` to `cache/`, run 5.4 local scoring. A kernel takes ~2-4 h; a new push while one RUNS is not allowed, wait for COMPLETE.

### 5.4 Scoring and blending (never score on a laptop what the host already holds)
* Host: `cd /root/wnn && PYTHONPATH=src python scripts/score_prediction_remote.py 1 NAME1 NAME2` (names without `pred_`/`_valid`). Blends: `python scripts/blend_score_remote.py 1 "a+b+c" "a:2+b"`. Headroom diagnostics: `scripts/distill_ceiling_diag.py 1 SINGLE ENS1+ENS2+... [PARENT]` (run-level swap, channel split, parent-delta correlation).
* Local: `WNN_THREADS=1 WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_MEM_MAX=4000000000 timeout 250 scripts/limited .venv/bin/python scripts/score_prediction_remote.py 1 NAME` (works for local `r3_*` and Kaggle `kg_*` preds in `cache/`).
* Always report mean +- n for a recipe, and compare to the matched control (3-epoch plain h256 ctrl ~0.6660, h384 plain ~0.6685).

### 5.5 Distillation pipeline (host script `/root/wnn/scripts/distill_pipeline.py`; canonical copy `vast_teachers_r1/host_scripts/distill_pipeline.py`)
`python /root/wnn/scripts/distill_pipeline.py --arm NAME --fold 1 --teachers T1,T2,T3 --parent RUN@epN --teacher-epoch 2 --epochs E --alpha A [--lr --wd --hidden --batch --sched] [--extra-fit-args "--distill-channel-weights 0.5,2"] [--no-teacher]`
* `--parent RUN@epN` loads `model_epN.pt`: 3-epoch scratch runs only have ep0..ep2 => `@ep2`; the combo parent `c20261003_combo_s3_f1` has ep3 => `@ep3`. `--teacher-epoch 2` for 3-epoch teachers. Student output = `vt_f1_dsq_<arm>`; predictions land on the host automatically.
* The distillation cache is shared per (teacher set, epoch) (`cache/dc_<sha1[:8]>`), built once, reused, not deleted. Control for any student variant: same recipe with `--no-teacher`, and the matched alpha/epochs run (e.g. `dm_a9_e2` 0.67018).
* OOD students: `scripts/ood_student_pipeline.py` (v9 source `continuous_execution_E529/h230_source_v9_ood`); KD-loss flags: v10 source `h230_source_v10_kd` (`--distill-channel-weights SUM,DIFF`, `--distill-loss {mse,pearson,mse+pearson}`, `--distill-runlevel-weight`; README there).

### 5.6 Resource-limited local commands
`WNN_THREADS=1 WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_MEM_MAX=3000000000 WNN_RUNTIME_MAX=120s scripts/limited <cmd>`; for 1 core/3 GB analysis. Check slice memory: `systemctl --user show wnn-campaign.slice -p MemoryCurrent` and `grep -E '^(anon|file|shmem) ' /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice/memory.stat`.

### 5.7 Keeping lanes busy
* Vast: pending depth >= ~10 jobs or >= 3 h of work; Kaggle: push the next kernel as soon as one is COMPLETE; local: >= 4 items. New work comes from (in order): open hypotheses with a decision rule (ledger), replicates of promising single-seed results (n<3), cache-only follow-ups, new ideas from section 9.
* Seeds: use 3,4,5,6,7,8... consistently; record seed in the name.

### 5.8 Building submission ZIPs (the part that moves the LB)
* All-data final of a recipe: same flags as the fold-1 run but `--valid-fold -1 --train-source train+valid --include-heldout-valid`, no `--split-manifest`/`--eval-source`/`--train-seqs`; with `--prop-power 0` no propensity is needed. Reference command: `research/campaign_2026-10-06/bandit_continuation/runtime/attempts/h230-all-data-student-gru256-seed3-v1-20261008.json` (key `command`).
* Pull `model_ep<final>.pt` of the finals (9 MB), build int8 Q8Vnni ZIPs with `packaging_q8vnni_20261010/build_upload3.py` (copy the pattern: `build(name, members, precision, ...)`), run gates `packaging_q8vnni_20261010/run_gates_upload3.sh` (CRC, ABI, Python 3.11 offline, determinism/reset, timing). Portal budget: ~13 portal-minutes per h256 model (variance 2x); keep <= 3 members; pair/triple must be checked on fold-1 siblings with `blend_score_remote.py` first. Write `SHA256SUMS` and a short README of members/expected direction; give the user a ranked list.

---------------------------------------------------------------------
## 6. WATCHERS: what exists, how to set up, how to debug

### Services (systemd --user; check with `systemctl --user list-timers 'wnn-*' --no-pager`)
| Unit | Role | Notes |
|---|---|---|
| `wnn-local-queue-r1.timer` (2 min) | launches local queue items under admission rules | must be active |
| `wnn-vast-tick-lane-b.timer` (5 min) | `controller.py tick --cap 70 --no-pull` (drop-in `~/.config/systemd/user/wnn-vast-tick-lane-b.service.d/headroom.conf`: MemoryMax 2G) — spend cap, auto-feed when pending < 3, destroy-when-drained | one `unit_failed` on an ssh TimeoutExpired is transient |
| `wnn-watchdog.timer` (~3 min) | `scripts/watchdog.py --once --notify` writes ALERT/RECOVERED rows to `research/campaign_2026-10-06/bandit_continuation/watchdog/ALERTS.jsonl` + desktop notification | read with `scripts/alerts_unread.sh` |
| `wnn-campaign-kaggle-poller.timer` | read-only Kaggle status polling | |
| DISABLED on purpose | `wnn-grade-bank.timer` (re-graded 240 banked preds at 5.9 GB RAM, closed local admission), `wnn-vast-pull-lane-b.timer` (300 MB scp over slow link) | do not re-enable without a fix |

### Setting up / changing a watcher
* Checks live in `scripts/watchdog.py` (`EXPECTED_TIMERS`, `SPEND_WARN=60`, Vast checks around `vast_{lane}_job_stalled`, `vast_job_failed:*`, `*_queue_low`, `disk_low`, local barrier/failed/blocked/queue-low). An alert is keyed `kind:detail`; it becomes RECOVERED when the condition clears. **Editing `scripts/` while a local item awaits start breaks that item (rule 6) — requeue it.**
* False positives seen: "job stalled" for pipeline jobs (log is `logs/pipeline_<arm>.log`, fixed by globbing `pipeline_*.log` for `w13-w16` names — extend the `case` in the watchdog if you add `w2x_` pipeline jobs); "timer missing" for disabled timers (remove from `EXPECTED_TIMERS`).
* `systemctl --user cat <unit>`, `journalctl --user -u <unit> -n 30 --no-pager` to debug a failing unit; `systemctl --user daemon-reload` after editing drop-ins.

### Debug playbook (symptom -> cause -> fix). Full detail in `CRASH_POSTMORTEM_20261011.md`
| Symptom | Cause | Fix |
|---|---|---|
| Vast job `rcN` failure, no GPU use | read `/root/run/logs/<job>.pipeline.log` or `pipeline_<arm>.log` / `<run>.fit.log` tail | fix flag/path, requeue under a NEW name |
| `CUDA out of memory` at start | too many jobs on a GPU | daemon needs 7000 MiB free + retries x3; lower MAX_JOBS file `/root/run/refill/MAX_JOBS` if repeated |
| `FileNotFoundError ...model_ep3.pt` | 3-epoch runs have ep0..2; `--parent X@epN` / `--teacher-epoch` index | use `@ep2` / `--teacher-epoch 2` |
| `distillation cache needs 13.4 GB; available budget ... 5.5 GB` | host disk (150 GB) full | delete finished `cache/dc_*`, host pred copies that are also local; shared cache already deduped |
| `--checkpoint-epoch must be within the fitted epochs` | `--pretrain-next-in-budget` subtracts pretrain epochs | checkpoint-epoch = epochs - pretrain - 1 |
| local `resource budget breach ... exceeded its 9000000000 byte profile reserve` | fit peak includes page cache; or non-reclaimable tmpfs in slice | reserve 10.5 GB (default now); free `/tmp` files; check `memory.stat shmem`; reopen barrier with review |
| local `aggregate working RAM exceeded 18 GB` | another process in `wnn-campaign.slice` (grader, agent, ideator) | find with `ps -eo rss,args --sort=-rss | head`; stop it; reopen |
| local item `preflight_failed: frozen source changed` | edited `scripts/`/`src/`/`submission/` during admission | requeue as new tag |
| local export crash `model_ep2.pt missing` or wrong-epoch score | old helper hard-coded export epoch 2 | helper fixed; use `--extra "... --checkpoint-epoch N"`; re-export with `--export-only` |
| GPU 0% locally with pending items | barrier closed, or 2-4 min admission retry gap, or fit loading | check `STOP_ADMISSION.json` + `runner_log.jsonl` tail |
| duplicate Vast pending files | hand-renamed with non-`p\d\d_` prefix | delete the duplicates from `/root/run/refill/pending/`, keep originals in `refill_jobs_b_shipped/` |
| tick service stuck for tens of minutes | slow scp pull | tick now `--no-pull`; do not re-enable pulls |
| `kaggle datasets status` says ready instantly | stale; mount not proven | compare `kaggle datasets files` size to `dataset/bundle.wnn`; confirm new job names in `out/status.json` |
| Kaggle kernel COMPLETE but job missing | job rc!=0 | read `out/logs/<arm>.fit.log` |
| score looks too low/high | wrong epoch exported, wrong fold, truncated pred (`.partial`) | check `export` argv epoch, file size 286 MB for fold-1 valid, rescore |

---------------------------------------------------------------------
## 7. LOOP PASS (do this every time the user sends the standing prompt)

1. `scripts/alerts_unread.sh` -> react (fix cause, requeue, reopen barrier, patch watchdog false positives).
2. Lane snapshot: local (`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader`; `ls local_queue_r1/pending | wc -l`; `STOP_ADMISSION.json` exists?), Kaggle (`kaggle kernels status ...`), Vast (`controller.py dash`; running/pending/disk; newest `cache/pred_*_valid.npy`).
3. Score anything new (5.4); compare to the matched control and sibling seeds; update the running mean (n!).
4. Write: LOOP_LOG pass entry (what landed with n, interpretation, retractions, ops), ledger row/status, hypothesis lines (`status`, `evidence`), incident -> post-mortem with a RULE. Commit + push (specific paths).
5. Refill any lane that will run dry in < 2 h (5.7). Prefer: replicate promising n<3 results; execute the decision rule of queued hypotheses; start the next idea from section 9.
6. Reply to the user with: lanes (GPU %, queue depth, spend), landed results (n seeds), decisions/blocks for them, next actions. Short.

---------------------------------------------------------------------
## 8. IMMEDIATE TODO LIST (priority order)

1. **All-data finals** `fin_mu256p0e4_s3/s4/s5` (Vast): confirm the first log (`/root/run/logs/vt_all_fin_mu256p0e4_s3.fit.log`) passes argument parsing and data loading ("train+valid", all 12,480 sequences); if it errors, copy the working flag set from the all-data student command (5.8), fix job files, requeue. Then pull final checkpoints, build int8 single/pair/triple ZIPs, run gates, rank, hand to the user. Also consider all-data finals of the best distilled student recipe after `w18_dm_pm` reports.
2. Collect `w19_kd_*` (control `dm_a9_e2` 0.67018; need >= +0.002 or a 2nd seed), `w14_ood_*`, `w17`/`w18` student results; decide what the next student recipe is.
3. Replicate the h256 recipe: Kaggle v6 + local runs (n >= 4 for muon-all x prop0 at e3 and e4).
4. Teacher-count test (s3..s10 muon-all h384): does a student from 6 teachers beat 3? Does the 3-seed teacher ensemble (0.67254) get distilled further with KD-loss variants?
5. Not-started, ordered by expected value (all have written falsifiers in the ledger / EDA / distillation notes): (a) privileged-lookahead teacher R^2 check (H-371: `vt_f1_lead*` oracles exist in the bank), (b) class-3 objective / head fine-tune (H-361), (c) event-change auxiliary head probe (H-362: logistic probe AUC < 0.60 kills it; re-verify the "labels change only at book events" claim on dense train), (d) multi-lag aux heads (H-363), (e) (u,v) loss reparameterisation (H-365), (f) residual-decorrelation member (H-364), (g) masked pretraining, (h) GBM on GRU hidden state/residual, (i) hybrid 256+384 pair test.
6. Keep the research index/tree fresh: rebuild `research/bandit_tree/generated/INDEX.md` after big changes; refresh the archive tarball when the user asks.

---------------------------------------------------------------------
## 9. HOW WE GENERATE IDEAS WITHOUT LOSING STRUCTURE (copy this style)

1. **Start from measured surprises, not from the literature.** Run a residual/EDA pass on cached fold-1 predictions (what fraction of weight/error sits where? which slice has the lowest local correlation? is the error a level offset? lead/lag? shared across members?). Each surprise -> a hypothesis with parent links, a predicted observable, a *cheapest falsifier* and a cost. Examples already done: class 3, run-level offset, 1-row lead, difference mode.
2. **Cache-only diagnostics before GPU** (minutes): swap probes (replace component X of model A with B's), channel/run-level splits, correlation of deltas, oracle bounds. Only run fits whose probe says there is headroom.
3. **Diverse independent ideators with a PUBLIC facts file** (never private data): write `FACTS.md` (setup, numbers, constraints, what failed) + `TASK.md` (ask for N distinct methods with mechanism, loss/schedule, signal, data source, falsifier, expected gain, citation). Run Grok (`~/.grok/bin/grok --cwd <workspace> --model grok-4.7 --no-subagents --permission-mode plan --max-turns 4 --prompt-file TASK.md`), OpenCode (`~/.opencode/bin/opencode run --dir <workspace> --model opencode/<free-model> --format json 'Perform the task in the attached file.' --file TASK.md`; put the message BEFORE `--file`), and a web-research sub-agent for the literature. Wrap each in `scripts/limited` (<= 1 core, 3 GB, `timeout 12m`). Keep raw outputs under `.../ideation/`. Treat all citations as unverified until opened.
4. **Convert to flags + tests with an implementer sub-agent in an isolated git worktree**: new source dir `continuous_execution_E529/h230_source_vN_*` copied from the previous one, default-off flags, CPU unit tests incl. bit-identity with flags off, README with deploy list. Merge the branch, deploy via stage + atomic `mv`, queue arms with a control.
5. **Record in four places every time**: LOOP_LOG (chronology), ledger (status per idea), `hypotheses.jsonl` (id, parents, statement, status, evidence), post-mortem (incidents + rules). Commit + push after each. The index/tree and memory files point to them. This is how a fresh context can resume with no loss.
6. **Portfolio discipline**: parallel branches; kill a direction only by its recorded falsifier; prefer new mechanisms over seeds; seeds only to confirm candidates; exploration reserve for unconventional ideas (privileged teacher, OOD paths, event-gated state).
7. **Self-skepticism**: when a single-seed result looks great (or terrible), queue replicates before writing a claim; audit the export epoch, fold, file size, and comparison control (we found three such errors).

---------------------------------------------------------------------
## 10. WHAT TO TELL THE USER AT THE START

Report in <= 15 lines: lane utilization, which of the section-4 jobs finished, results with n, whether the all-data finals started, ETA to submission candidates, and the two open decisions (old caches deletion; which ZIP to upload). Then continue the loop. Be honest about uncertainty; say "not yet" when the LB has not moved.

---------------------------------------------------------------------
## 11. MANAGING ALERTS (the user said: "react to them, it is crucial")

Alert rows: `ALERT <key> | message` and `RECOVERED <key>`. Keys seen: `vast_job_failed:b:<jobfile>`, `vast_b_job_stalled:<job>`, `vast_b_queue_low`, `vast_b_spend_high`, `vast_b_disk_low`, `local_admission_barrier_closed`, `local_failed:<item>`, `local_blocked:<item>`, `local_queue_low`, `unit_failed:<unit>`, `timer_missing:<timer>`.
Procedure for EVERY alert (do not just acknowledge):
1. **Classify** in 2 minutes: real failure / transient / false positive / expected consequence (e.g. `local_blocked` export after its fit failed).
2. **Real failure** -> read the log named in section 6 playbook, find the root cause (not the symptom), fix it, requeue under a new name/tag, then add a post-mortem entry: symptom, root cause, fix, **RULE** to prevent recurrence. Examples of root causes we found: wrong checkpoint index, disk full, tmpfs memory, frozen-source hash, reserve too small, duplicate shipping.
3. **Barrier closed** -> never reopen before you know why; reopening needs the written `--review` text.
4. **Transient** (single ssh timeout, one `unit_failed` followed by successful runs) -> note it, no action; if it repeats 3 times, treat as real.
5. **False positive** -> fix the checker in `scripts/watchdog.py` (and say so in the loop log) so it does not train you to ignore alerts; remember rule 6 (editing `scripts/` can fail a local item waiting to start: check `local_queue_r1/failed` afterwards and requeue).
6. **Queue-low / spend** alerts -> refill lanes (section 5.7) / check the cap is still above spend (cap 70, warn 60).
7. After fixing, confirm the matching `RECOVERED` row appears on the next watchdog pass (~3 min). If a check itself breaks, the watchdog alerts on that too: "silence means every check ran".
8. Mention in the user report only alerts that changed state or cost time; list the fix in one line.

## 12. KEEPING ALL GPUs UTILIZED CORRECTLY (without wasting or crashing)

* **Utilization is measured per lane**, not by the busiest one: local GPU % (`nvidia-smi`), Vast `controller.py dash` (two GPUs, util and MiB, number of compute processes), Kaggle kernel `RUNNING` plus job names in its status JSON. A lane at 0% with pending items means a gate is closed (barrier, admission retry gap 2-4 min is normal, disk, frozen source) — find the gate, do not just add more items.
* **Never fill a lane with uninformative repeats.** Every queued job needs: hypothesis id, comparison control, decision rule. Priority: (1) jobs on the critical path to the LB (all-data finals, packaging inputs), (2) replicates of promising n<3 results, (3) decision-rule experiments of open hypotheses, (4) exploration (new mechanisms).
* **Depth targets**: Vast pending >= ~10 jobs (7 slots; each fit 1-1.5 h under sharing; students 1.5-3 h), Kaggle: next kernel pushed the moment the previous is COMPLETE (4 arms ~2-4 h), local >= 4 items (fits ~35-45 min). Use `controller.py feed` (backlog) and add backlog items before the queue is short.
* **Capacity rules that prevent crashes**: Vast GPU memory — the daemon starts a job only with >= 7000 MiB free and retries OOMs (3x); do not raise MAX_JOBS above 7; check host disk (>= 33 GB free before any new distillation cache). Local — h256 only, 10.5 GB RAM reserve per fit, one GPU job at a time, aggregate slice <= 18 GB; nothing big in `/tmp`; analysis through `scripts/limited`; keep sub-agents within 1 core/3 GB. Kaggle — one kernel version at a time; mount check by job names.
* **Spend**: ~$0.85/h on lane b is within the authorized ~$1/h; the tick holds jobs at the cap and destroys the instance when drained, so a long idle queue = teardown. If you intentionally drain, say so to the user first.
* **Do not starve the critical path**: use `p00_` prefix to put urgent Vast jobs first (e.g. all-data finals), and note it in the log.
* **GPU contention changes speed**: with 7 jobs a student runs at ~12 seq/s (vs ~25 alone); ETAs in the dashboard are per-job estimates, multiply queue depth by that.

## 13. INTERPRETING RESULTS (how not to fool yourself)

1. **Matched control or it did not happen.** Compare against the same recipe/fold/epochs with the same export epoch. Controls: plain h256 3 ep ~0.6660, plain h384 ~0.6685, muon-all h256 ~0.6674, muon-all h384 ~0.6696, student no-teacher 2 ep 0.66861, dm_a9_e2 0.67018.
2. **Noise model**: one-run sd ~0.001-0.002 at 3 epochs, 0.002-0.005 at 4+ epochs. A difference < 0.002 on n=1 is nothing. Report mean over n seeds and n. A claim needs n >= 3 or an effect >> 0.003 reproduced on a second machine. First-three-seed estimates regress to the mean (muon-all h384: +0.0022 -> +0.0014).
3. **Check validity before interpreting**: right fold (`vt_f1_*` = fold 1; `vt_f0_*` fold 0 has offset +0.013), right export epoch, pred file size (fold-1 valid = ~286 MB), not `.partial`, not a failed/short run, parent/teacher ancestry not containing evaluated sequences (students are warm-started from combo/teacher runs trained on fold-1-excluded data).
4. **Large improvement = suspect first** (leakage, wrong comparison, wrong epoch). Large degradation = suspect export epoch / truncated pred before concluding "the idea is bad".
5. **Fold-1 vs public**: fold 1 has been inspected many times; use it for mechanism ranking. Public transfer is only known after the user uploads; the pair/ensemble gains and int8 losslessness are the only quantities that transferred reliably so far. Never convert a local delta into an expected LB number; say "candidate expected to help; uncertainty ~0.002-0.003".
6. **Distinguish scientific from infrastructure failures.** A crashed job is execution evidence, not a negative result for the idea. Record it as a failure with cause and requeue; keep the idea open.
7. **Read curves, not endpoints**, when possible (per-epoch checkpoints exist for long runs; score ep3/7/11/... on the host) — this showed the 24-epoch runs peak at epoch 11 and then overfit.
8. **Ensembles**: equal-weight blends help only among members within ~0.001-0.002 of each other; a weaker partner dilutes. Use cache-only blends (5.4) to test before building ZIPs.
9. **Distillation reading**: student gain over the matched no-teacher continuation = value of the teacher signal; gap to the teacher-ensemble score = remaining headroom; a student cannot be expected to beat its best single teacher unless the target carries new information (diff-channel/run-level targets, OOD paths).
10. **Write the interpretation with its uncertainty** and the next decision it triggers; update the hypothesis `status` (proposed / queued / supported_Nseed / refuted_for_X / retracted).

## 14. WHEN TO USE SUB-AGENTS AND EXTERNAL IDEATORS (OpenCode / Freebuff / Grok / web research)

Use them for **breadth and independence, not for facts about our own data** (they have no access to private data; they hallucinate citations).
* **Use an external ideator round when**: a direction plateaued (e.g. distillation gave +0.0008), a new class of mechanism is needed, or after a big diagnostic (EDA, cache probes) produced a clear problem statement. Give a public `FACTS.md` + `TASK.md`; ask for 4-6 DISTINCT methods each with mechanism, loss, signal, data, falsifier, expected gain. Run Grok-4.7 + 2-4 OpenCode free models in parallel (different providers => different priors), compare, pick 2-3, write falsifiers, test cheaply. Typical yield: 1-2 useful ideas per round (Grok's "event-run level" and "privileged lookahead" ideas were the best; free OpenCode models give shallower lists and sometimes empty output — check the `.json`/`.md` is non-empty; put the message before `--file`).
* **Freebuff** (`~/.npm-global/bin/freebuff --cwd <workspace>`) is interactive; use it only for drafts/code review in an isolated workspace, never with private data. Do not invent headless flags.
* **Use a web-research sub-agent (general-purpose, WebSearch extended)** for literature surveys with a concrete mapping to our pipeline and a ranked experiment table; ask it to mark which sources it actually opened.
* **Use an implementer sub-agent (isolated worktree)** when an idea needs a new trainer flag/loss: give it the exact file to copy, default-off requirement, tests required (flag-off bit-identity), README, no GPU, <= 1 core/3 GB. Review its report; merge; deploy yourself. A stuck implementer should be pinged with SendMessage to wrap up.
* **Use a critic/auditor sub-agent** (research-evidence-review / recipe-audit skills) before a big claim or before building submission ZIPs (leakage, wrong comparison, ABI). The proposer must not approve its own result.
* **Do NOT use sub-agents for**: routine scoring, status passes, queue management (you do it directly — it is faster and avoids RAM spikes), anything that needs `scripts/` edits during local admission, or more than 3 concurrently. Do not delegate decisions about spend, uploads or deleting data.
* **Always record**: tool/model, prompt files, raw output path, which ideas were adopted, measured result (even negative) in the ledger section F-style table. Do not archive private reasoning transcripts.
