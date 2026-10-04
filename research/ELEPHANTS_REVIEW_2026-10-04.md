# Second-opinion review — 2026-10-04

Read-only audit. Every claim below is traceable to a file path, a line number, a
score-report key, or an experiment ID in `research/experiments.jsonl`. No
experiment was launched; the queue pause markers
(`research/campaign_2026-09-29/STOP_NEW_JOBS`, `.../2026-10-01/QUEUE_PAUSE`)
remain in force and only the user can lift them.

Position: public 0.6641 (`D256_single_distilled`, registry), leader 0.6677,
target ≥ 0.6660, deadline **Nov 15 2026** (`docs/timeline.md`) = 42 days,
5 public submissions/day (`docs/rules.md:15`), last upload **2026-10-02**.

---

## 1. Corrections to `ELEPHANTS_REVIEW_2026-10-03.md`

The previous review is directionally right (ship the confirmed combo, use the
submission quota, don't orphan W384). It has six concrete errors that change
priorities.

**C1 — "one w384 trunk (~1.5x compute = ~43 min) fits the gate" is wrong.**
Lines 71–72 and 155–157. The GRU itself is not the dominant cost; *weight
streaming* is (`engine/STATUS.txt:56`: "about 190 kcyc per member is a floor
for 16-bit weights on this machine", at 12.4 B/cyc for a 2.2 MB member).
Parameter count for `RNNModel` (`src/wnn/nn.py:14-57`, n_in = 112 + 99 = 211)
is `14H² + 226H`:

| hidden | params | weight bytes | streaming floor (kcyc) | est. total q16x (kcyc) | est. public min |
|---:|---:|---:|---:|---:|---:|
| 256 (measured) | 0.98 M | 2.2 MB | ~190 | **233.8** (STATUS.txt:10) | 28.8 |
| 320 | 1.51 M | 3.4 MB | ~293 | ~360 | ~44 |
| 384 | 2.15 M | 4.7 MB | ~418 | **~500–540** | **62–66** |

The 2× scaling is *quadratic* (both GRU layers and `head.0` = 512×512 scale as
H², and STATUS.txt:60 names `head.0` as 23% of the bytes). Gate = 423 kcyc
(52 min); hard rule limit = 60 min ≈ 488 kcyc. **W384 in q16x is estimated at
62–66 min — over the hard limit, not "43 min".** In i16 it would be ~79 min.

**C2 — U10 ("c3on/c3off class-3 distillation screen … dropped silently") is a
misreading.** Those four jobs are the C3 `since-weight × distill-alpha`
2×2 factorial (`queue6.txt:29-34`: `dcurve_fold.sh … --since-weight 0 /
--distill-alpha 0`), reported in full as **E-351** with verdict "substitution
as predicted by H2". The quoted "+0.0015, p=0.054" is exactly
`ON_A0_minus_OFF_A0` = 0.001549, p0 = 0.0540 in
`campaign_2026-10-03/scores/c3_report.json`. It is neither a class-3 screen nor
an orphan. The same claim is repeated at `campaign_2026-10-04/FRONTIER.md:39`
and should be deleted from both.

**C3 — the dual-metric claim (E6) is wrong.** The 10-03 score reports do carry
`legacy_sequence_mean_not_official` alongside the official metric; the legacy
number is present and labelled.

**C4 — W320 was not "a different context".** Both widths ran the same AdamW arm
(`dcurve_fold.sh {fold} n 3 --hidden {384|320}`): W384 f0 = **+0.002009**,
CI [0.00072, 0.00332], p0 = 0.0000, one fold only
(`c1_report.json` → `W384_minus_S100s3`); W320 = **+0.000072**,
CI [−0.00121, 0.00131], p0 = 0.43, two folds
(`width320_prereg_report.json` → `WIDTH320_minus_CTRL`). Same recipe, same
control, different width — i.e. the effect grows between 256 and 384 and is
zero at 320. That is a real capacity signal, not a contextual difference.

**C5 — the submission quota is not scarce, it is idle.** 12 uploads total,
0 on 10-03, 0 on 10-04, last one on 10-02 (`research/lb/registry.json`).
At 5/day and 42 days left, ~200 slots remain. Reviewing uploads as a scarce
resource caused the campaign to *estimate* the transfer slope (0.66–0.8 from
the GRU128-since family) when it can be *measured* directly, for free.

**C6 — neither review noticed that W384 was never compared to the combo.**
Both W384 and the combo are measured against the same distilled AdamW control.
Muon×width and combo×W384 are untested and cost one flag each
(`dcurve_fold.sh … --optim muon --hidden 384`).

---

## 2. Elephants in the room

**E-1 — Runtime is now the binding constraint on every remaining lever, and
the thing that would relax it is a half-finished uncommitted patch.**
`engine/l2_partial_uncommitted.diff` (536 lines: `bench_engine.rs`,
`approx.rs`, `half.rs`, `python.rs`, `rnn_fast.rs`) implements the "keep ~1 MB
resident in L2, stream the rest non-temporally" plan estimated at **15–25%**
(`STATUS.txt:61`). It was never applied, never built, never measured. It is the
gate on *both* remaining deployable width/ensemble routes:

| payload | today (q16x) | with −15..25% | verdict |
|---|---:|---:|---|
| 1× GRU256 | 233.8 kcyc / 28.8 min | 199 / 24.5 | fits |
| 2× GRU256 (pair bag) | 463.9 / **57.1 min** | 394 / **48.5 min** | **unlocks the gate** |
| 1× GRU384 | ~500–540 / **62–66 min** | 425–459 / 52–56 min | marginal, needs it |

The engine is not an engineering side-quest; it is the precondition for H-B
(width) and the pair-bag route to ≥ 0.6660. Est. 1–2 days of Rust work.

**E-2 — The only significant width result is at an undeployable width, and
nobody has measured it.** W384 fold0 is the largest single lever in the
ledger (p0 = 0.0000, CI entirely above zero) and it may not fit in 60 minutes.
`bench_engine.rs` already accepts a `wK` head spec and has a kcyc counter
(`STATUS.txt:63`) — **the measurement needs no training and no GPU**, it only
needs a checkpoint-shaped weight file of the right dims. Measuring it before
spending 3 GPU fits on W384 folds 1–3 is the highest information-per-dollar
action available.

**E-3 — The confirmed recipe's `chunk1000` component has an untested
confound: optimizer cadence.** The default is `--chunk 2000` with a per-step
schedule (`dcurve_fold.sh:13`); `chunk1000` doubles the number of optimizer
steps per epoch at the same LR trace family. The campaign itself flagged this
("earlier chunk experiments changed the update count (confounded)") and built
the macrostep-credit instrument (U3/H-G, ~8 GPU-h) to de-confound it — but
never ran the *cheap* version: **one gradient-accumulation run at chunk1000
with 2× fewer effective steps** (~1–2 GPU-h). If the gain survives, it is
credit and H-G is worth the 8 GPU-h. If it vanishes, chunk1000 is a cadence
knob, and `chunk500`/`chunk4000` become free one-flag sweeps on top of the
combo. Either way the answer is one paired run, and it also decides whether
the reset's Card 4 (long-horizon credit) is worth anything — note the observed
direction is *shorter* credit helping, which argues against Card 4 as written.

**E-4 — The only measurement that converts local evidence into a submission
decision is not being used.** Transfer slope for the distill/Muon family is
unknown; everything (combo expected +0.0011–0.0016 public) rests on a slope
extrapolated from a different family (GRU128-since). We already own
diversity-packed, gate-passing ZIPs that bracket the question and have never
been uploaded:
`submissions/c20261002_q16x_D256` (precision control for 0.6641),
`c20261002_q16x_T2D256` (2-member payload), TE07s3 (packaged, E-339).
Five slots a day, ten already burnt.

**E-5 — Class 3 is 46% of the weight and 61% of the loss, and it is the only
lever capable of a step change.** Local class WP 0.81 / 0.77 / 0.74 / 0.43
(E-313: class3 loss share 0.605, weight share 0.477; public mix
[0.183, 0.160, 0.198, 0.459]). Every surviving lever is worth ≤ +0.0016
pooled. A **+0.02 class-3 WP is worth ~+0.009 pooled** — five times the whole
remaining plan. E-313 concluded "only lead: decode class-3 queue race from
volume-code history"; E-321 did that *outside* the network (deployable blend
class3 +0.00079, global +0.00022, CI [−0.00013, 0.00055]) and explicitly left
open "optional cheap GRU-input test of 28 run-relative volume features".
**That follow-up has never been run.** It is the only surviving mechanism
aimed at the 46%-weight class.

**E-6 — `research/hypotheses.jsonl` cannot be used as a work queue.** Five
rows carry live-looking statuses for resolved work: H-054 `running`, H-055
`queued` (refuted by E-291), H-056 `implemented` (rejected by E-293),
H-057 `tool_ready` (rejected by E-292), H-058 `queued` (answered by E-345).
H-060 and H-064 have **no status field at all**. H-046 `proposed` is the only
genuinely open row, and it is the highest-upside one (T-3 below). Anyone
triaging off this file will spend GPU on refuted ideas.

---

## 3. Rejected, orphaned, or dropped for no good reason

| # | Item | Evidence it was never properly closed | Cost to close |
|---|---|---|---|
| **T-1** | 28 run-relative volume features as **GRU inputs** (not a post-hoc blend) | E-321 verdict itself: "optional cheap GRU-input test … expected ≤ +0.0005"; never ledgered | data build + 1–2 folds |
| **T-2** | R11 thinning augmentation (label-exact, view-averaged) | E-314: `insufficient_evidence` at +0.0002, "P(pass) ≈ 0.2", proposed 2 GPU-h — never run | ~2 GPU-h |
| **T-3** | **H-046 branch/U categorical head** (H-I structural arm) | status `proposed` since creation; oracle branch readout 0.968 (H-043, `supported`); never one fit | ~6 GPU-h |
| **T-4** | TE07 / gen-3 teacher: 2-fold +0.001145, CI [0.00026, 0.00216] (E-337) vs holdout −0.000762 (E-339) | contradiction *diagnosed but never tested*: E-327 shows the teacher's own class-3 WP is 0.468 fitted vs 0.413 held-out, i.e. the fold-bound teacher is optimistic exactly where the student's rows overlap. If that is the cause, gen-3 ≈ 0 and the family dies for free; if not, it is a real +0.0011 | CPU row-overlap analysis, then 1 upload |
| **T-5** | W384 folds 1–3 **and** Muon×width | C6: orphan, not rejected; one flag each | 3–4 folds + E-2 |
| T-6 | TE07s3 ZIP packaged and never uploaded (`artifacts/submission_quota_2026-10-03/`) | free | 1 upload |

Things that were correctly killed and should stay killed: post-hoc
calibration, class scales, output EMA, physical spread decoding, DTW, weight
soup, class-routed parent/FT, merged-GRU additive members, cross-member
bilinear heads, sequence-latent family (E-314: oracle headroom is overfit
noise — episode cross-fit −0.021/−0.012, best shrinkage +0.0005).

---

## 4. Dedup: items the 10-02 research reset re-proposed that are already closed

`RESEARCH_RESET.md` was written without full dedup against E-313/E-314/E-315
and before the 10-03 continuation rounds. Re-ranking its eight cards:

| Card | Verdict now |
|---|---|
| Card 8 — input usage map | **Done.** `campaign_2026-10-03/continuation/round2/usage/REPORT.md`: "No representation-changing training proposal is justified." Plus round3 `matched_usage` / `washout_usage` / `usage_review`. |
| Card 3 — sequence-level state aux | **Pre-killed.** Its premise (oracle per-sequence D level +0.027) was shown to be ~97% overfit by E-314 R02. Do not spend the GPU. |
| Card 4 — long credit horizon | **Disfavored by the campaign's own result**: shorter credit (chunk1000) is what helped. Resolve via E-3's accumulation control instead. |
| Card 6 — self-normalised chunk loss | = U2 / H-F. Alive, instrument-ready, ~2 GPU-h. Keep. |
| Card 1 — efficiency weighting | Alive but bounded: E-351 shows `since-weight` ON−OFF = +0.00035 at α=0.5 (CI crosses 0), i.e. the knob is nearly redundant *for students*. Cheap CPU probe; expect small. |
| Card 2 — fresh views of same sequences | Alive, overlaps E-314 R11 (T-2). One fold. |
| Card 5 — head-dominated clip | Minor, instrument-flavoured. Low. |
| Card 7 — experimental design / power | Adopt as a rule, not an experiment. |

So the reset's own priority line ("card 4 probe > C3 > card 2 > card 8 > card 6
> card 5") should be rewritten; three of its top six are closed or inverted.

---

## 5. Does the plan add up to 0.6660?

Honest arithmetic from 0.6641, using transfer slope 0.66–0.8 unless noted:

| contribution | local | expected public | running total |
|---|---:|---:|---:|
| base | — | — | 0.6641 |
| H-A combo, full-data, one trunk | +0.00179 pooled (E-359 gates passed) | **+0.0011 … +0.0015** | 0.6652 … 0.6656 |
| pair bag (2×GRU256), *needs E-1* | +0.0004…0.0008 (E-333/E-342/H-064) | +0.0003 … +0.0006 | 0.6655 … 0.6662 |
| W384 on top, *needs E-1 + E-2 + 3 folds* | +0.0020 (1 fold) | +0.0010 … +0.0016 | up to 0.6672 |
| gen-3 teacher | +0.0011 (2 folds) vs −0.0008 (holdout) | −0.0005 … +0.0008 | — |
| class-3 mechanism (T-1 or T-3) | unmeasured, oracle branch 0.968 | unbounded, variance ±0.004 | — |

**Conclusion: combo alone lands at ~0.6655 and misses the target.** Reaching
≥ 0.6660 requires combo **plus** at least one of {pair bag, W384}, and both of
those are gated on the engine (E-1). That is the single most important
structural fact in the project right now.

---

## 6. Priorities

### P0 — this week, no new research questions

1. **Train, package and upload the full-data combo** (H-A). One fit, one
   upload. It simultaneously delivers the score *and* measures the transfer
   slope for the new family. Everything else is an estimate until this lands.
2. **Measure W384 kcyc** with the existing `bench_engine.rs` counter and a
   dummy-weight checkpoint (CPU, hours). Kill or confirm E-2 before any W384
   fold is launched.
3. **Use the 5/day quota**: combo (P0.1), TE07s3 or `q16x_T2D256` as a
   2-member/payload probe, and one precision control. Zero-cost information.

### P1 — next, gated on P0

4. **Finish and land `l2_partial_uncommitted.diff`** (E-1). Budget 1–2 days.
   Unlocks the pair bag (57 → ~48 min) and makes W384 marginal→feasible.
   Success gate: 2×D256 q16x ≤ 423 kcyc with all parity gates intact.
5. **Chunk cadence vs credit control** (E-3): 1 accumulation run. Decides
   H-G's 8 GPU-h, the chunk sweep, and Card 4 in one shot.
6. **Class-3 queue-race as GRU input features** (T-1): the only surviving
   mechanism aimed at the 46%-weight class; E-321 explicitly proposed it.
7. **Reconcile the ledger** (E-6): one pass over `hypotheses.jsonl` to set
   terminal statuses from `experiments.jsonl`, and delete the C2 claim from
   `FRONTIER.md` + `ELEPHANTS_REVIEW_2026-10-03.md` U10.

### P2 — the reserve (10–20% of budget, per the bandit policy)

8. **H-I structural branch head** (T-3): the biggest untested structural
   lever (oracle 0.968), never fit. ~6 GPU-h.
9. **W384 folds 1–3 + Muon×width + combo×W384** — only after P0.2 and P1.4.
10. **Gen-3 teacher** — only after the CPU row-overlap diagnosis (T-4); the
    current evidence is consistent with a measurement artifact.

### Do not spend on
sequence latents / FiLM / online recalibration, post-hoc calibration, class
scales, DTW, physical spread decoding, weight soup, cross-member heads,
class-routed experts, long-horizon credit (until P1.5 says otherwise),
card 3 or card 8 of the reset.

---

## 7. One-line summary

The campaign's plan is sound but is estimating three things it could measure
(transfer slope, W384 runtime, chunk mechanism), while the two levers that
must both land to reach 0.6660 — the pair bag and width — are gated on a
536-line engine patch nobody has finished, and the only lever big enough to
change the score by a step (class 3, 46% of the weight) has exactly one
surviving lead that has never been run.
