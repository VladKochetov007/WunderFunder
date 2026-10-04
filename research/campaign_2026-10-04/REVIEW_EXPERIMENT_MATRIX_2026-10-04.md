# Review-to-experiment matrix and ETA — 2026-10-04

**Historical planning snapshot as of12:14UTC.** Live evidence and current jobs are in `CURRENT_STATUS.md`; the tables below preserve the original cost estimates and specifications. The former20:00Kyiv hard stop was explicitly superseded by continued user authorization. E412 publicscore is now0.6655, E419scientificQATscreen completed, E424is7/8fits, E426is implemented/checked/queued and E449legacywidthv2retry is running. E430now has3fresh cadence/credit arms; adding that control and the2legacywidth fits moves the approximate79-fit portfolio to82fit screens (32reviewdirections plus the separatelegacyreplication). This is a planned inventory, not completedfit count. CPU framework/code checks are not counted as candidatefits.

Original planning context: This matrix joins the Oct 3 and Oct 4
Elephants reviews, the supplied second-opinion directions, and the critic
follow-ups. “Mentioned” is not treated as “tested”: each actionable direction
has a concrete experiment ID, an explicit status, and an estimated cost. The
estimates are planning bands, not guarantees. All local GPU work remains
single-worker; Kaggle experiments may run on separate T4s after E419 releases
them. The 80 device-GPU-hour ceiling is cumulative and includes failures and
retries. TPU quota is separate.

## Current state

| Work | Status | Current ETA / boundary |
|---|---|---|
| Full-data combo package, E412 | Ready at `submissions/c20261004_combo_s3_i16/submission.zip`; score not measured | New public point requires a manual upload. Planning center is about0.6654, with a broad plausible range0.663–0.666; not a calibrated promise. |
| E417 reset-lag screen / H-099 | Complete | 113.7s; reset250/500 disrupt sampled WP, longer intervals remain uncertain. This is a state-disruption result, not a gradient-credit result. |
| E422 real-data gain screen / H-095 | Complete | Reciprocal component screen is negative for its fixed scalar-gain protocol; it does not close auxiliary learning. The reports were corrected from “pending.” |
| E419 precision/QAT | Kaggle v4 RUNNING; local FP16 comparison complete | At12:21Z, Kaggle showed25.23/30 GPU-hours remaining and20/20 TPU-hours remaining. Provider charging confirms execution, but worker logs, per-GPU progress, output and memory remain unavailable. V4 has an8.5 device-GPU-hour cap; reliable completion ETA is not visible yet. |
| E424 specialist/adapters | Local first arm running, epoch3 started | Eight arms are sequential on the sole local GPU. Last log was epoch3 batch0/378; the worker is at its24-GiB cgroup ceiling, primarily reclaimable file cache, with zero worker OOM events. A20:00 Kyiv hard stop is armed, so the full eight-arm run may spill into the next local block. No second local process will start while this worker is active. |
| E425–E427 critic designs | Prepared; no fits started | Seven fits total. E426 is blocked on distinct teacher/evaluation EMA states and resume tests; E425/E427 need the critic amendments below. |
| E428 synthetic gain sensitivity | Registered and coded; not run | CPU-only, under2GiB and2min; wait until E424 releases local memory. It complements the existing easy planted-cache unit test by using the real noisy gain estimator and later scoring path. |
| TPU probe | QUEUED | 0/20 TPU-hours used; no TPU assigned, so hardware timing/fit ETA cannot yet be given. |

The independent timing audit measured two E423 four-epoch combo fits at2309s
and2410s each (about0.64–0.67 device-GPUh), before export/scoring. Kaggle
charging moved by0.77 GPUh between11:36Z and12:21Z. Empty provider logs prevent
any defensible worker-level estimate for E419.

## First tranche already admitted: 12 programs, about 37 fitted models

These are the existing finite tranche, not the complete review portfolio.
The width screen now includes width320 as a current-generation re-test, so its
fit count is37. Its cost is about **23–31 device-GPU-hours of fit time**, plus1–3 wall-hours
for prediction export, scoring and reconciliation. With three continuously
fed devices the fit-only theoretical wall time is8–12 hours; current E419 and
E424 occupancy, implementation work, hard deadlines and provider opacity make
**2–4 calendar days after the active jobs clear** a more realistic estimate.

| Program | Fits | Current state / review coverage |
|---|---:|---|
| E424 H093/H094 specialists and adapters | 8 | Running locally; includes causal prefix specialization and matched-capacity controls. |
| E419 precision/QAT | 2 | Running on Kaggle v4, with FP32 controls already recovered. |
| H073 chunk-mass objective | 3 | Prepared; actual distilled objective and aligned/shuffled/historical controls. |
| H079 randomized-origin training | 2 | Prepared; paired same-seed view-variation test. |
| E413/H078 volume inputs | 2 | E413 completed with +0.000962 (CI crosses zero); second fold/seed replication remains planned. |
| H072 common-macrostep credit | 2 | Prepared; matched forward history and pooled numerator/denominator. |
| H066 width × optimizer | 6 | Prepared; current-generation width256/320/384 × optimizer screen. Pair/W384 deployment still depends on E429 runtime work. |
| H098 origin-preserving privileged teacher | 3 | Prepared; distinct from generation-3 teacher. |
| H067 second combo seed / pair candidate | 2 | Prepared; native two-member runtime must pass before a public package. |
| E425/H106 soft specialist mixture | 1 | Await exact parameter count, expert-gradient and exposed-panel amendment. |
| E426/H107 event-clock consistency teacher | 3 | Await separate teacher EMA/evaluation EMA implementation and resume/eval tests. |
| E427/H108 volume-gated recurrent state | 3 | Await exact causal current-cycle, stale-cycle and no-op controls. |

Historical screens are controls, not exclusions. When a new idea uses an
already-tested model family, it will keep its named mechanism distinction and
will not inherit a claim from a frozen-head or proxy-only predecessor.

## Remaining review directions: second tranche

These **20 programs add about42 fitted models**. At the E423 reference rate,
they need roughly26 device-GPU-hours before wider models, optimizer prototypes,
or retry overhead; budget **26–35 GPU-hours**. Most reuse a verified seed3
combo control, but only after checkpoint, recipe, data and prediction hashes
match. Exploratory screens on already exposed components remain labelled
exploratory; a positive lead gets a separate seed/component replication.

| ID | Review idea made testable | Planned fits | Cost / gate |
|---|---|---:|---|
| E429 | Build and measure the existing L2-resident engine patch; benchmark pair-bag and W384 kcycles and parity before further width spending | 0 GPU; CPU/Rust build and benchmark | 1–2 engineering days. No fit until real kcycle evidence; preserve existing engine as fallback. |
| E430 | Chunk1000 two-chunk gradient accumulation to match optimizer cadence, plus row-horizon-preserving EMA decay `sqrt(0.999)` | 2 | 1.3–2 GPUh. Pool loss numerators/denominators exactly; distinguish optimizer cadence from BPTT length. |
| E431 | Modern end-to-end causal branch/U categorical readout against wMSE and capacity control | 2 | 1.3–2 GPUh. E071/E072 and E421 are relevant negatives, not the same modern end-to-end configuration. |
| E432 | Regime-conditioned intercept/amplitude and four-head readout on current strong trunk | 2 | 1.3–2 GPUh. Causal regime uses only the first99 rows; class-weight attribution is not multiplied into a pooled-score promise. |
| E433 | Target-specific row/event history: t0 row clock, t1 event clock, both-clock matched control | 3 | 2–3 GPUh. New strong-baseline test after a small causal feature preflight. |
| E434 | Activity-conditioned EMA interactions and event-age × EMA memory, each with matched stale/shuffled control | 4 | 2.6–4 GPUh. Causal features only; no physical-column interpretation. |
| E435 | Lagged i1 residuals with identical i0-lag and shifted-i1 controls; separately isolate statistical transition structure from physical spread decoding | 4 | 2.6–4 GPUh. Pair by component and preserve causal row availability. |
| E436 | Cycle8-then-EMA against row EMA and cycle7 placebo | 3 | 2–3 GPUh. Tests dense cycle smoothing, not phase replacement. |
| E437 | Refit nonlinear t1 offset on the strong combo with component-disjoint calibration | 0 GPU; CPU/head fit | Under1 CPU-hour; E360's strong-GRU result is a control, and intervals crossing zero remain inconclusive. |
| E438 | R11 label-exact thinning/view averaging with retained-mass control | 2 | 1.3–2 GPUh. Separate from randomized-origin H079. |
| E439 | Late Muon→AdamW handoff and module-clipping controls | 3 | 2–3 GPUh after fused/EMA parity gates. |
| E440 | ARTBP finite-window credit with unclipped SGD/momentum against matched truncated SGD/momentum | 2 | 1.3–2 GPUh after exact gradient and CUDA resume checks. |
| E441 | LSTM diversity partner and same-cost GRU pair control | 2 | 1.3–2 GPUh plus native runtime. E345 informs but does not veto a current recipe. |
| E442 | Teacher row-overlap CPU diagnosis, then one generation-3 teacher/student comparison | 3 | 2–3 GPUh after the lineage audit; public TE07 upload remains manual. Distinct from H098 privileged information. |
| E443 | Current-spread auxiliary plus sentinel-handling intervention on the modern combo | 3 | 2–3 GPUh; use matched equal-work controls and class guards. |
| E444 | Content-addressed memory / two-block structured state against position/decay matched controls | 3 | 2–3 GPUh after H095 sensitivity and E427 state-gate work. |
| E445 | Stale incoming-state refresh versus equal-work sham | 2 | 1.3–2 GPUh; not the E417 fixed-predictor reset test. |
| E446 | Common-increment coordinate and connected-component/teacher-overlap audit | 0 GPU; CPU analysis | Under2 CPU-hours before any model allocation; report the frozen-fit incident and source identities explicitly. |
| E447 | Distillation-alpha interaction on the current Muon/chunk/SBD combo (`alpha0.7` and `alpha0.85`; reuse the exact `alpha0.5` control) | 2 | 1.3–2 GPUh. New optimizer×alpha interaction; report as exploratory on the exposed paired components. |
| E448 | Inference-only blend/weight-soup comparisons after the second current combo seed exists | 0 GPU; CPU scoring | Under1 CPU-hour after E067. Compare complete predictions, exact alignment, single model, equal prediction blend and weight-average where tensor provenance permits. |

The exact fit count is approximate because an unchanged control can be reused
only when its identity is proven. These rows deliberately include variants of
previously negative ideas when they test a different model generation,
information path, or control. A zero or inconclusive result remains a result.

## Complete ETA and budget

- Current E424 eight-arm block: approximately5–6 local GPU-hours from its first
  fit, with the first arm already in its third epoch; the20:00 Kyiv hard stop
  may leave queued arms for the next local block.
- Current E419 QAT kernel: ETA is unknown while worker logs stay hidden; the
 8.5 device-GPU-hour cap is the hard bound for this kernel. TPU queue has no
  provider ETA.
- First 37-fit tranche: 23–31 device-GPU-hours, then2–4 calendar days after
  active jobs are reconciled.
- Remaining review tranche: another27–32 device-GPU-hours, then4–8 calendar
  days for implementation, staged admission, scoring and review.
- Full actionable review queue: **about79 fitted models across32 programs**,
  roughly50–63 device-GPU-hours, plus CPU/Rust work. Including about8.3
  device-GPU-hours already charged in this continuation and a6-hour recovery
  reserve gives an estimated62–79 of the authorized80 hours. Heavy retries or
  a slow W384 route can exceed this band; no branch may overrun the hard cap.
- With current three-GPU capacity, idealized fit time is about17–26 hours of
  elapsed GPU-running time across the two tranches. Allow **about8–14 calendar
  days from now (Oct 12–18)** for the full review queue, because E424 is
  sequential, E419 has no visible progress, TPU is queued, implementations and
  sealed manifests precede each launch, and Kaggle quota refreshes Oct10.

These dates are for the initial registered screens, not all possible positive
replications or the Nov15 competition deadline. Any replication, new
experiment, failure retry, public score, or generated ZIP is appended to the
same registry with its own cost before it starts.

## Consistent-generation rule

All score comparisons use the same named combo generation: identical teacher
and propensity identities, train-plus-validation rows, epoch policy, seed,
optimizer schedule, and final head procedure. A change tests one named
mechanism; matched controls reuse weights only with verified provenance. Report
parent and complete-predictor results separately. Component intervals are
conditional on the fitted models and exposed panels; they do not erase
training-seed/search uncertainty. The first pass may use one fold/seed to
generate information, but any promoted lead gets a new seed or disjoint
component replication. The user receives every valid deterministic ZIP and its
unmeasured/bounded public-score forecast; no public score is inferred from a
local WP sum.
