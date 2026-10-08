# Handoff prompt for the next research agent (living file — last updated 2026-10-08 ~19:00 UTC)

Paste everything below the line into the next agent. Re-check every "live" item; this file is an index, not proof.

---

You are the research controller for **Wunder Connectome Challenge2** in `/home/vlad/development/wunder/wnn_connectome_starterpack`. Goal: a valid deterministic submission above our public best **0.6660** and teammate Maxim's **0.6674**; a real public **0.700** exists, so pursue it. Research continues until the user explicitly says stop. Be autonomous, plan, delegate bounded tasks, verify, record evidence.

## Read first (binding)
1. `AGENTS.md` (= `CLAUDE.md`): rules — independent code & weights (no teammate weights/code copies), official metric `global_wp_clip2_v1` (`src/wnn/gmetric.py`), causal per-row serving with state reset per sequence, human-only public upload, Vast strict ownership ("touch only machines you rented"), no top-up.
2. `specs/README.md`, `specs/research-objective.md`, `specs/current-state.md`, `research/RESEARCH_METHODS.md`.
3. Controller notes for this session: `research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/` — `CONTROLLER_PLAN_20261008.md` (decisions/timeline), `AGENT_RESOURCE_RULES.md` (MANDATORY for any subprocess/agent), `NEW_HYPOTHESES_20261008.md` (H-274..H-283), `PIPELINE_MAP_20261008.md`, `PUBLIC_CV_HISTORY_20261008.md`, `SCREEN_PROTOCOL_20261008.md`, `STUDENT_READINESS_20261008.md`, `KAGGLE_SCOUT_20261008.md`, `WARM_RUNNER_SCOUT_20261008.md`, `ENGINE_TIMING_20261008.json`, `COHORT1_FOLD0_STUDENT_ARMS.json`.
4. Ledgers (append-only): `research/hypotheses.jsonl`, `research/experiments.jsonl`. Tree: `research/bandit_tree/generated/INDEX.md` (rebuild with `scripts/research_tree.py --root . build`).

## Hard-won operational rules
- **Memory:** campaign cgroup `wnn-campaign.slice` = 5 CPU, 18 GB high / 20 GB max, no swap; the local GPU worker guard (`scripts/bandit_resources.py`) KILLS fits when the whole slice (incl. page cache) exceeds 18 GB. Two incidents today: a subagent's unwrapped 15.8 GB python caused a host OOM that killed the whole session; page cache + support scopes killed the seed9 fit. Mitigations live: `wnn-slice-filecache-guard-v1` (file-cache reclaim above 15.5 GB). Every python/CLI via `scripts/limited` with `WNN_SLICE=wnn-campaign.slice WNN_MEM_MAX≤2.5e9` while a GPU fit runs; never read whole parquet; long work only as systemd units.
- **Local GPU launch:** `scripts/bandit_resources.py --root $PWD --state $PWD/research/campaign_2026-10-06/bandit_continuation/runtime launch --attempt <NEW id> --kind gpu --cpus 2 --reserve-bytes 9000000000 --profile <id> --contract <auth json> --admission-timeout 1800 -- <argv>` (run it via scripts/limited). Attempt ids are single-use (a refused launch burns the id). Launch needs measured slice ≤ 9 GB: reclaim file cache first (`echo "<bytes> swappiness=0" > /sys/fs/cgroup/.../wnn-campaign.slice/memory.reclaim`). Support CPU quotas count toward 5 cores.
- **Local warm queue (load-once, run-many):** worker attempt `warm-queue-local-v3-20261008` runs `scripts/warm_queue_worker.py` over `research/campaign_2026-10-06/bandit_continuation/runtime/warm_queue_local_v1/` (pending/ → running/ → done|failed/, `results.jsonl`, `logs/`, `worker_heartbeat.json`, STOP file). Add work by dropping spec JSON into `pending/` (lowest priority first). Generate fold0 cohort specs with `scripts/make_cohort_specs.py ARMS.json --root ... --python ... --queue ...`. Worker idle-exits after 30 min empty.
- **Scoring:** `scripts/score_fold0_panel.py RUN@epE [--reference RUN@EPOCH | blend "A@ep1*0.5,B@ep0*0.5"] --out X.json` — official metric on the frozen fold0 468-group panel (repeatedly exposed: development evidence only; confirm on fold1 before claims). Reproduces E511 grades exactly.
- **Packaging:** `scripts/package_submission.py NAME "PARENT@epN|HEADFT@ep0":1.0` (merged heads on one trunk = incumbent format); gates: `scripts/check_submission_abi.py`, `scripts/check_submission_python311.py` (py311 env built by packager in its scratchpad), timed replay on `cache/valid_subset20.parquet`. Runtime: single trunk ≈ 37–38 min, two GRU256 trunks ≈ 61–72 min (over 60-min limit; f16 only 4% faster).
- **Vast:** `scripts/vast_campaign.py --ledger ~/.local/state/wnn-campaign/vast/ledger.json {inspect,search,run,status,settle,...}`. Run path needs intent + root admission (120 s expiry, fresh offer snapshot via `_offer_snapshot`, `spending_authority_sha256`, ADR0015) + dispatcher receipt; then `systemd-run` supervisor (see `research/campaign_2026-10-06/bandit_continuation/vast_warm_cohort/`). Offers vanish within minutes — bind at dispatch. Never retry after ambiguous create. Settle finished attempts with root receipts (billing via campaign client GET `/users/me/invoices/`). Credit ≈ $17.9, committed ≈ $1.79 + live reservations. No extend path for hard stop: set long hard stop + long idle-exit at creation; keep paid GPUs busy (hold spec trick: self-requeueing low-priority wait spec).
- **Kaggle:** CLI `.venv/bin/kaggle` (2.2.4). Warm-cohort kernel code on branch `worktree-agent-abc36427758e3823c` (`scripts/kaggle_warm_cohort/`, receipt `research/.../kaggle_warm_cohort/PUSH_01_SMOKE_20261008_R1.json`); private datasets `vladkochetov07/wunder-funder` (raw parquet), `wnn-c20261004-fold0/1-assets`, `wnn-warm-cohort-src-20261008`. Quota: GPU ~11 h until 2026-10-10 00:00 UTC (lost if unused), TPU ~20 h (TPU V6 failed on an over-strict input-inventory check; JAX path unproven). Most past pushes failed on transport validators — smoke first.
- **Agents:** Haiku subagents only (user), caveman-terse, minimal tool calls, ≤3 active, each prompt must include AGENT_RESOURCE_RULES.md. Root approves admissions/spend; agents never approve themselves.

## State at handoff (verify live)
- Local queue order: seed9 strong-parent epoch-3 resume → Z1b/Z2b GCorr head-FT on H230 S4/S3 all-data students → Z3 student from incumbent parent `c20261004_combo_s4_final@ep3` (+Z3H head-FT) → faithful seed99/seed9 Muon-parent students (+Z4/Z5 head-FT) → fold0 evidence: `f0_combo_s3_parent_hft_v1` (incumbent analogue), A00 (C3 analogue, never-run heterogeneous teachers + family head on fold0) + A00 head-FT, A01, A04.
- Packager agent packages `c20261008_<zN>_combo_i16` (merged-head combos) and writes `RELEASE_CHECK.json`.
- Vast r4 smoke: attempt `e529-vast-warm-smoke-c1v2-f0-20261008-r4`, instance 54894762 (RTX 3090 TH $0.305/h), hard stop 21:59 UTC; cohort V00–V09 (C1v2 variants + head-FT) prepared at `.runtime/e529-vast-warm-smoke-c1v2-f0-20261008-r4/cohort_v1/`.
- Kaggle smoke `vladkochetov07/wnn-warm-cohort-smoke-20261008-v1`: running at handoff.
- Ideation: `research/ideas/claude_20261008_round/` (Grok 4.7, OpenCode Muse, Nemotron raw outputs; PLAN_TO_0700.md when synthesized).

## Evidence so far (fold0 468 panel unless noted)
- Public: combo_s4 (Muon/chunk1000/boot parent + GCorr head merged) 0.6660; H230 students 0.6644–0.6653 (never head-finished); same-recipe pair +0.0004; head-FT +0.0013 public on parents.
- C1 s3 0.681618, C1 pair 0.682176; H272S4 hidden 0.681363; best equal 2-trunk pair 0.682231.
- Blend-lab: learned stackers/booster negative; diverse existing architectures add ~0; only a 2nd GRU256 trunk helps (+0.0003..0.0008).
- H-281: labels are NOT causally revealed (R² ≤ 0.19 for y_t from features at t+L ≤ 100); recalibration using past labels is a leakage oracle (+0.06..0.18) — invalid, but shows per-sequence target bias/scale tracking is valuable if done label-free (H-283 testing).

## Idea backlog (register each as a hypothesis/arm before running; one-flag matched controls; promote only after fold1)
Finishing/serving (cheap, near-term):
1. Student × GCorr head-FT merged combo (Z1–Z5) — main candidate.
2. 3–4 merged heads on one trunk (GCorr, wmse-FT, multiplicative readout, per-class head), weights by component-CV.
3. Head-FT objective variants: gcorr_pool, wmse_gcorr mix, per-market-class GCorr, longer FT (2–3 ep), lr sweep {2e-4,1e-3}.
4. EMA/SWA of student checkpoints (ep0+ep1) before head-FT.
5. Int8/AVX-VNNI or better-blocked native kernel → 2 trunks or GRU320/384 within 60 min (H-282).
6. f16 storage + pruning low-norm hidden units to cut runtime.
Distillation/teachers:
7. A00–A13 fold0 arms (alpha 0.5/0.9, family head on/off, raw vs clip2 family targets, normalized vs source teacher sum, lr, epochs 3, chunk 1000, boot-h 8, Muon student).
8. Organizer baseline in the 0.0625 teacher slot (H-279; `cache/pred_baseline_*` exist).
9. Capacity teachers GRU512/768 fold0-clean on Vast 3090/4090 → distill into GRU256 (packets exist: gru512_768_muon_fit_v5r3).
10. Teacher ensembles of different seeds + architectures (attention/TCN as teachers, not served) — H-280 via distillation.
11. Distill a 2-trunk pair into one student (pair-to-single).
12. Born-again / iterated self-distillation (student → teacher → student).
13. Teacher-target temperature / amplitude rescale; teacher disagreement as sample weight.
14. Relational/structural distillation of hidden states (H196 code exists, unadmitted).
Representation/features:
15. Label-free per-sequence scale/bias tracking features (H-283): EW volatility of price/spread/volume, own-prediction running stats.
16. Cross-instrument lead–lag features (i1 → i0), order-flow imbalance, microprice, queue imbalance.
17. True raw-feature teacher family (H-273, raw-weight parity fix).
18. Numeric embeddings (PLE) on selected features only; feature-group dropout.
19. Market-class routing / class-conditional heads; class-weighted loss matching public class mix [0.183,0.16,0.198,0.459].
20. Event-time (activity clock) inputs: time since last trade/quote change, event counts.
Objective/optimization:
21. Muon vs AdamW at student stage; whole-matrix vs per-gate Muon; LR multipliers.
22. WSD schedule / cooldown, SGDR restarts + noise (H-266/267).
23. Loss: wmse vs gcorr vs mix; chunk-level vs global correlation surrogate; |y|-weighting exponent.
24. TBPTT chunk 1000 vs 2000 vs 4000; stateful chunk carry; detach jitter.
25. Larger batch + LR scaling on 3090/4090; gradient accumulation.
26. Auxiliary targets: future multi-horizon targets, realized-target statistics (offline-computed) as aux heads.
Data/validation:
27. Fold1 confirmation of every promoted arm (fold1 assets exist on Kaggle).
28. Train on more data epochs (5–6) for parents; all-data final fits for winners.
29. Small-data screens (`specs/plans/2026-10-08-small-data-screening.md`) for fast ranking, then full confirmation.
Bold bets:
30. Much larger teacher (GRU1024 / Transformer) on rented 4090 + aggressive distillation.
31. Native engine rewrite to fit 3 trunks (diverse ensemble).
32. Test-time per-sequence adaptation that is causal and label-free (e.g. normalization stats updated online, entropy-style objectives).

## How to decide a submission
Fold0 analogue gain vs the incumbent analogue (`f0_combo_s3_parent_hft_v1`) and vs C1, consistent across 2 seeds, then fold1, then package single trunk merged heads, pass ABI/py311/determinism/reset/timing (<55 min projected), give the ZIP path + RELEASE_CHECK to the user for manual upload. Report public result receipts in `continuous_execution_E529/` or controller notes.
