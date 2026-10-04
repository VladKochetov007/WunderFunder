# Campaign contract 2026-10-04 — combo transfer + width decision

Status: **DRAFT — pending user authorization.** This is the new dated reviewed
contract required by `research/campaign_2026-10-03/continuation/RESUME.md`
before any resumption. Nothing is launched by its existence. Derived from
`research/ELEPHANTS_REVIEW_2026-10-03.md` (second-pair-of-eyes review).

## 1. Target claim

Primary: ship the full-data Muon/chunk1000/SBD combo (d256, seed 3) and
measure its public transfer. Interim expectation: public ~0.6655 (transfer
slope 0.8 x local +0.0018), honest PI ~[0.6645, 0.6660]. The campaign-level
target E[public] >= 0.6660 stays open and is NOT claimed by this contract.

## 2. Deliverables

- D1: full-data combo d256_s3 checkpoint + packaged ZIP + provenance + both
  metrics recorded; upload decision stays a human gate.
- D2: width decision — W384 confirmed/rejected on fold1 (+folds 2/3 if alive).
- D3: updated frontier + ledger entries for every run, including negatives.

## 3. Baselines

- Public incumbent 0.6641 (`D256_single_distilled`, registry).
- Local pooled references: CTRL d256_s3 0.6723 (folds 0/1 panel),
  0.7055 (folds 2/3 panel, per FOLD23_DECISION.json).
- Combo local deltas: +0.00185 bag f0/f1; +0.00179 f2/f3 (CI > 0, gates passed).

## 4. Metrics and invariants

- Primary metric: `global_wp_clip2_v1` (pooled, clipped), paired connected-
  component bootstrap, seed 271828, n_boot 2000. Dual-report the legacy
  sequence-mean block metric for every packaged candidate (contract E6).
- Invariants: causal inference only, state reset per sequence, predictions
  from step 99, 1 vCPU / 16 GB / 60 min / ZIP <= 20 MB, fixed membership and
  weights (no runtime governor), deterministic repeat output, native/v2/v3
  parity within predeclared tolerance.
- Class guard: no fold's class3 delta < -0.002 under pub143 weighting
  (class3 ~= 0.459 of public weight).

## 5. Falsifiers and disqualifying evidence

- H-A (combo): public delta <= 0 -> recipe downgraded; transfer slope for
  distill/Muon family re-estimated from the result.
- H-B (W384): fold1 delta <= 0, or fold0 gain fails to replicate direction,
  -> width family closed in this context.
- H-I (branch head): fails to beat the same trunk's own wMSE head on
  class3-under-pub143 -> closed regardless of pooled WP.
- Any leakage finding (target-derived inference inputs, oracle spread, mask
  leakage) -> immediate abort of that branch, preserved as negative.
- Any resource incident -> pause, preserve evidence, diagnose before retry
  (runtime_incidents policy).

## 6. Allowed scope

- Data: existing local competition data and caches; no new external data.
- Jobs: only through `gpu_queue.sh`/`run_gpu.sh` with the minute watcher;
  no background shells; existing frozen job scripts (`dcurve_fold.sh`,
  FOLD23_CONFIRMATION job lines) preferred.
- Code mutations: new flags only where a frozen script already exposes them;
  trainer patch remains unapplied (opt2 stays out of phase 1).
- No instrument work (state drift, AMP, ARTBP) in phase 1.

## 7. Budgets

- Phase 1 GPU <= 6 h: combo full-data s3 (~1 h), W384 f1/f2/f3 (~3 h),
  contingency ~2 h. One GPU job at a time; 12 CPU / 36 GiB / VRAM <= 5.5 GB
  observed, allocator <= 4.8 GB.
- CPU scoring as needed (screens run in ~12-20 s).
- Phase 2 (only after review of phase 1): engine -15%, gen-3 teacher,
  option 4/3 forks, opt2 screens — separately contracted, <= 10 GPU-h.
- Structural arm reserve: 10-20% of GPU budget for H-I.

## 8. Verification tier

- Local pooled screens: tier B, but on previously exposed folds — treat as
  selection-prone; require direction agreement across panels and class guard.
- Public transfer: tier C until an upload returns; LB noise ~0.0006; one
  submission resolves most uncertainty (H-A or TE07 arbiter first).
- Model-to-reality gap: local folds differ from public composition (37% vs
  ~46% class3 sequences); pub143 reweighting is a proxy, not the public set.

## 9. Stop / abort / rollback

- Abort: leakage, metric mismatch, prediction mismatch, runtime over gate.
- Pause: any resource incident or watchdog failure; diagnose, preserve.
- Reframe trigger: after H-A + H-B resolve, if no path to >= 0.666 exists
  from alive families, fork into the representation arm (H-I) instead of
  tuning scalars.
- Rollback: every packaging step reversible; incumbent ZIP untouched.

## 10. Roles

- Controller: this agent (contract, queue, synthesis; does not grade own runs).
- Actors: frozen queue jobs (fixed scripts, fixed flags, no interim peeking).
- Critics: existing prereg gates + class guards + the user for uploads.
- Reproducer: second seed or second panel (folds 2/3) for any promoted delta.
- Sub-agent depth: none in phase 1 (jobs are prewritten and frozen).
