# Controller plan — Claude Code session, 2026-10-08 (started ~16:55 UTC)

Mutable controller notes. Raw receipts stay authoritative; this file indexes decisions.

## User instructions this session (latest wins)

1. Root = planning/monitoring agent; most tasks to Haiku subagents (≤3 active).
2. Continue until explicit stop; long-running research agent.
3. Use compute efficiently: load data into machines/notebooks once, feed queued runs; manage with research notes + multi-armed bandit.
4. /loop every 30 min (job 624c1fa4, :13/:43): status, reflect, what to submit, have we beaten previous research and Maxim on local/Kaggle/Vast CV.
5. Full use of Vast and Kaggle authorized for climbing LB: check hypotheses, validate, re-implement old ideas with new framing.

Binding rules unchanged: independent code/weights, official metric, causal reset, human upload gate, $20 Vast cumulative ceiling/no top-up, campaign-receipt-owned instances only, local 5 CPU / 18 GB high / 20 GB max / 5 GB VRAM / one GPU worker.

## Live state at session start (16:55–17:10 UTC)

- Public best 0.6660 (`c20261004_combo_s4_i16`, seed4 parent+head equal mix, 37m07s). Maxim 0.6674 (equal pair of GRU256x2 distilled students). Latest ours 0.6653 (H230 all-data pair, 61m22s — at the 60-min limit).
- Local GPU was idle. **Launched** `h230-strong-parent-seed9-muon-local-v1-20261008` 17:03:51Z (exact seed9 recipe V2 science; only seed/name differ from completed local seed99). 8.9 seq/s, ETA ~18:40Z + exports. Authorization `continuous_execution_E529/H230_SEED9_LOCAL_ROOT_FIT_AUTHORIZATION_20261008_R1.json`. Kaggle seed9 V8 must not be pushed while this is admitted.
- Stopped futile local v7r5 Vast supervisor loop (316 restarts, pre-provider stale-receipt exit 2, attempt absent from ledger). Receipt `continuous_execution_E529/vast/h230_seed54354_v1/H230_V7R5_FUTILE_RESTART_LOOP_STOP_20261008_R1.json`.
- Reclaimed 2×1.44 GB partial tmpfs test copies charged to campaign slice (`TMPFS_SCRATCH_RECLAIM_20261008_R1.json`); slice 10.86→7.96 GB.
- Vast read-only inspect 17:05Z: 0 instances, credit $17.89. v7r6 supervisor still reconciling ambiguous ApiError create (ledger needs_review, instance null, cleanup deadline 23:42Z). Ledger conservative reservations $18.80/$20 block new rentals until ambiguous holds ($2.80 h217-v5r2, $2.80 v7r6) are settled with evidence.
- Kaggle: TPU V6 QUEUED (no device evidence); all GPU V4–V7 seed9/capacity packets ERROR before updates. GPU quota 39,759 s remaining until refresh 2026-10-10T00:00 (unused quota is lost at refresh); TPU 71,621 s.

## Strategy

Critical path to beat 0.6674 (faithful method, ADR0023): seed9+seed99 own Muon strong parents → students (warm start, +2 ep, heterogeneous teacher mix) → fixed equal pair → package (runtime check: pair ran 61m22s; need margin) → human upload. Local GPU is the reliable lane for this.

Exploration (bandit): short screens in warm workers so the data is staged once per machine; screen many distinct mechanisms on a fixed fold0 panel, confirm promising ones at full scale/second fold. Arms ranked by expected information per GPU-hour, not seed count.

Compute lanes:
- Local GPU: critical path (seed9 parent → students). Screens only in gaps.
- Kaggle GPU: finite warm cohort kernel (data mounted once, sequential queue of screens) built on the last transport that actually completed a fit (H232 V7B shortslug). Use remaining quota before 10-10 refresh.
- Vast: one owned 3090-class worker, data staged once, warm queue fed over SSH, destroyed when queue empty. Requires settling ambiguous holds first (evidence: 0 instances, unchanged credit).

## Delegations (Haiku)

| Agent | Scope | Deliverable | Status |
|---|---|---|---|
| pipeline-mapper | read-only faithful pipeline status | PIPELINE_MAP_20261008.md | running |
| warm-runner-scout | read-only warm runner feasibility | WARM_RUNNER_SCOUT_20261008.md | done |
| public-history | read-only public/CV history | PUBLIC_CV_HISTORY_20261008.md | running |

## Controller bandit (session-level; numbers only from measurements)

| Arm | Lane | Status | Evidence | Next |
|---|---|---|---|---|
| Faithful seed9/99 Muon parents → students → pair | local | seed99 parent done; seed9 running | Muon fold0 +0.0030 vs control (muonfull_report) | students after seed9 |
| Served-head GCorr on parent | local | graded +0.00083 vs parent (exposed fold0) | H272_S4 grade | fold for packaging later |
| Warm screen queue | Kaggle/Vast/local | design | — | builder task |
