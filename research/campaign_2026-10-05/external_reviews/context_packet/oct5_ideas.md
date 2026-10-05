# Stuck research agent review and new directions

**Session ID:** ses_ef6f41bf4ffeCpSIVZc3KLKe1g
**Created:** 10/5/2026, 1:32:16 AM
**Updated:** 10/5/2026, 1:35:06 AM

---

## User

Looks like our previous research agent got stuck and cant produce ideas. Take a look, make some analysis and find elephants in the room and underestimated ideas or ideas rejected for nothing. Research on its previous ideas and results. Make new directions and what to prioritize 
- research/SECOND\_PAIR\_OF\_EYES.md — competition explanation, prioritized reading list, each agent’s research framing, evidence locations, and a ready-to-copy review prompt.
- research/REVIEW\_FILE\_INDEX.md — complete clickable file list, grouped by directory with purposes.
- research/REVIEW\_FILE\_INDEX.json — the same list for automated traversal.

The index covers **2,274 files**, including **419 Markdown files, 1,517 JSONs, 149 images**, plus JSONL ledgers, TXT reports and notebooks. **1,081 files belong to today’s research.** Large numerical artifacts remain local and are separately inventoried.

The competition predicts two future-price indicators from sequential market features. Predictions must be causal and run offline on one CPU within 60 minutes; the repository uses Global Weighted Pearson correlation. Best recorded public score: **0\.6641**.

The guide distinguishes completed results, failed hypotheses, historical reports, and unfinished ideas. No new experiments were started. 

---


## Why the agent got stuck

It chased mechanism purity while the public gain came from engineering.

* Oct3 `RESUME.md` + `SECOND_PAIR_OF_EYES.md` froze best at **0.6641**. Registry now shows Oct4: `combo_s3 0.6655 (10tq248b)`, `combo_s4 0.6660 (2e5j6bnt, 37m07s/626MB)` — target reached at display precision, research **paused** (`campaign_2026-10-04/CURRENT_STATUS.md`, `RETROSPECTIVE_2026-10-04.md`, `PORTAL_RESULTS_20261004_LATE.json`, `PUBLIC_RESULTS_REFLECTION_20261004_LATE.md`).
* Round2-6 spent cycles on warm/synthetic toy oracles with subthreshold effects: state-drift `DETECTABLE_SUBTHRESHOLD` (hidden 0.236%, pred RMS 1.6e-5, `round6/real_probe_2222r/comparison.json`), AMP descriptive residuals 0.00045/0.00019 (`amp_linearity_audit_v2`), ARTBP 7-case ≤1024-mask toy (`artbp_credit/`). Correct verdicts, zero score path.
* Meanwhile the actual +0.0014/+0.0019 came from: full-data fit + fixed head procedure + seed. No mechanism isolated. Agent kept asking “what is credit?” instead of “what is the head doing?”

## Current state / index

* `research/REVIEW_FILE_INDEX.md/json`: 2,274 files at `1bae9cf`, 419 md / 1517 json / 149 png. Oct3 pause guide is stale on scores.
* Oct4 live state: `campaign_2026-10-04/CURRENT_STATUS.md`, `RETROSPECTIVE_2026-10-04.md`, `VALIDATION_SCORECARD.md`, `FURTHER_RESEARCH_PLAN_PAUSED.md` (authoritative paused plan), `experiment_catalog.json` / `hypotheses_current.json` (H-065…H-110, 46 cards), `EXPERIMENT_INVENTORY.md` (H-065…H-105), `REVIEW_EXPERIMENT_MATRIX_2026-10-04.md` (79-fit, 50-63 GPU-h plan — this queue paralysis is part of “stuck”).
* Oct4 completed: E410 CV pooled +0.002414 [+0.00156,+0.00344] (`E410_complete_predictor.md`); E412 s3 public 0.6655; E450 s4 public 0.6660 + paired report (all local intervals include zero); E424 8 arch parents all null; E425 soft-routing null (+0.00022/+0.00017/+0.00058, all CI⊃0); E426 clock null (-0.000045/+0.00008); E423 EMA-rows -0.000574 [-0.00179,+0.00030]; E419 QAT ±6e-06 null (accuracy preserved); E417 reset250/500 -0.042/-0.031 (disruption, not credit); E421 branch -0.088 (frozen-head only); E462 q8 withdrawn `libpython3.13` (`E462_PACKAGING_INCIDENT_20261004.md`); E465 JAX AdamW parity 4.2e-06 (scalar only).

Decomposition (`PORTAL_RESULTS_20261004_LATE.json`, reflection):

```
parent_s3 0.6641 -> head_s3 0.6654 = +0.0013  (same trunk, ep0 head)
head 0.6654 -> combo_s3 0.6655 = +0.0001 (nonlinear, no weight lesson)
combo_s3 0.6655 -> combo_s4 0.6660 = +0.0005 (seed)
```

## Elephants

1. **Head is the gain, mechanism unknown.** All E424/E425 parents lack the incumbent head procedure (`E424_ALL_PAIRED_REPORT.md:16,27`). H-109 (8×1-epoch fixed heads, `E464_COMPLETE_ARCHITECTURE_PREREG.md`) is cheapest, highest-information, still `source_preparing` / stopped.
2. **chunk1000 confounded.** Doubles steps/epoch, changes EMA counts (`.999` vs `.999^2`), denominators. E430 (2-chunk accum vs chunk2000, pooled numerators), H-072 macrostep, H-073 normalizer (E414: mean 2947, CV 19.6% — predictive fits pending) never ran. Don’t claim “longer credit won”.
3. **Seed variance >> forecast.** Local s4-s3 +0.00187 [-0.00097,+0.00693] vs public +0.0005. Bag full-file 0.70110 > s4 0.70077 but excluded 0.70142 < s4 0.70169 (`VALIDATION_SCORECARD.md`, `E450_SEED4_PAIRED_REPORT.md`). No seed distribution. Bag unsubmitted; likely >60min — needs measured runtime, not score arithmetic.
4. **t0 vs t1 + class3.** Excluded: t0 0.6876 / t1 0.7120 diff +0.024 (`VALIDATION_SCORECARD.md:11`). H-084 (t0 row / t1 event adapters) never ran end-to-end; E426 tested *teacher* clock, not input representation. `signal/econometrics` consistently show t1-event signal.
5. **Full-file vs excluded sign flip.** Full: D256 0.70079 > s4 0.70077 > s3 0.70050; excluded: s4 > s3 > D256; public: s4 > s3 > D256. Fitted aggregate would have killed the winner.

## Rejected-for-nothing (do bounded retests, not blanket revival)

Correctly killed as *universal law / submitted member*, wrongly orphaned as *feature*:

* `hft`: 91.5% joint jump agreement, modal 66.2% vs 43.1% shuffled; regime constant already 0.317 WP. Missing: rows0-99 prefix regime intercept + per-regime EMA, hold-between-events (`puzzle_a`: 0 target changes on 97,928 unchanged-i0 transitions, 8409 full-block 0 vs 693 shifted).
* `linear/statphys`: activity×EMA, age×EMA gates from causal running moments. Explains class heterogeneity without quantum routing.
* `olympiad`: current 0.464 vs carried-a6 0.347 (+0.116, 17/20 seqs); p/dp hold costs 0.137. Missing: p-only lookup + freeze-between-events.
* `signal/econometrics`: own-clock failed universally (halves +0.036/-0.001) because class `.6569` -0.057 vs others +0.036/+0.039/+0.051. Missing: split heads t0-row / t1-event (H-084).
* `fractal`: row16 +0.037 vs raw; event16 -0.055 (horizon mismatch flagged by authors). Missing: horizon-matched rate-normalized span.
* `puzzle_b`: lag4 +0.0079 concentrated in rg54/1385 (+0.09/+0.06); t0 +0.021 vs t1 -0.002. Missing: input-only lag4 gate on reserved comps.
* `nonlinear`: weak-ridge offsets +0.022/+0.037 killed on strong GRU -0.000474 (`continuation/reproduction/REPORT.txt`). Correct kill of transport, not of strong-residual t1 corrector with component-disjoint cal (H-083).
* E424 diagonal modulation +0.000768 [-0.00008,+0.00173] — largest adapter point, just crosses zero, no head. Retest via H-109, not new fits.
* W384 f0 +0.0020 [0.00072,0.00332] single-fold; v4 died in 1.7s on allow-set bug, v5 held draft — no science evidence (`RETROSPECTIVE:48-49`). H-066 still open.
* Volume28 E413 +0.00096 [-0.00008,+0.00205] one fold/seed — needs second fold, not closed (H-078).
* Teachers: TE07 public 0.6636 kills teacher-ensemble *as member*, not H-096 crossfit / H-097 gen3 / H-098 privileged-future8 questions.
* `state_drift_response` (fixed lags 0/8/32/96/256/end from *stored* packets, ≤32MiB, no GPU) — cheapest discriminator, `INCOMPLETE_PAUSED`. Do first.

Correctly dead — do not reopen unchanged: post-hoc calibration/scales/smoothing/soup (`E-287/288/292/293/295/309/332`: max bound 6e-05), EMA6 public -0.0006/-0.0010 (`1ykwb95u`, `246401vj`), TCN -0.0044 / LSTM -0.00086 (`dtcn/dlstm_f01_report.json`), online-norm (zero-cache incident), merged-GRU additive (negative at every weight), runtime governor, DTW/routing grids, w320-in-AdamW-context.

## What to prioritize on resume

P0 (cheap, decisive):
1. H-109 complete heads for 8 E424 arms + controls. Resolves whether all nulls are head artifact.
2. `state_drift_response` from stored packets. Transient vs persistent, grad vs update.
3. Seal repaired-q8 runtime (loaded-host packet for exact repaired hashes). Packaging, not science. Keep s4 37m07s as anchor; no engine -15% needed.

P1 (mechanism separation):
4. E430 cadence: chunk1000 accum vs chunk2000, pooled numerators/denominators, optimizer/EMA counts fixed.
5. H-073 aligned vs shuffled vs historical normalizer (calibration E414 done).
6. H-066 W384 fold1 + H-067 bag runtime measurement (or bag-as-teacher student if >52min).
7. H-078 volume28 second fold/seed.

P2 (representation, strong baseline + matched capacity):
8. H-084 target clocks, H-085/H-087 activity/age×EMA, H-086 i1-residual vs i0-lag/shifted-i1, H-088 transition head, H-089 cycle8-then-EMA vs cycle7, hold-between-events inference-only (0 GPU).
9. H-082/H-083 regime heads/offsets with within-outerfold crossfit.

P3: H-096/097/098 teachers, H-070 factorized / H-069 QAT-student, H-091/092 content/structured memory with position/decay + donor/shift controls, H-100 ARTBP only after blind-critic cross-review.

Rules: same generation (teacher/prop/source/epochs/seed/head) per contrast; parent vs complete predictors reported separately; component CIs ≠ seed uncertainty; one fold/seed to screen, new seed/partition to promote; no LB tuning on +0.0001 blends.


---

