# Fresh-eyes review B — 2026-10-05

Second independent pass over the Challenge 2 campaign, commissioned after the
previous reviewer stalled. Read-only: no fits, no uploads, no file modification
outside this document. Scope is the root campaign tree only.

New work in this pass, beyond what the Oct-5 guide, elephants review and two
audits already contain:

1. an exact cell-level **headroom decomposition** of the strong-baseline
   residual report (reproduces the reported global WP to 6 decimals before any
   counterfactual is applied);
2. a **ledger census** of `research/hypotheses.jsonl` (113 rows) and
   `research/experiments.jsonl` (481 rows) by status and by cost;
3. a **harness status check** of the Oct-4 input-feature screening code;
4. an **arithmetic derivation of the 0.7000 milestone** against the only two
   measured local-to-public slopes.

Everything below is quoted from files already in the tree; file paths are given
so any line can be checked.

---

## 1. Elephant one — the 0.7000 milestone has never been costed against anything

`RESEARCH_CONTRACT.md` records the milestone as *"0.7000 honest expected public
score"*, and every subsequent document repeats that it is unmet. No document in
the tree derives what it would take. Doing the arithmetic:

| Quantity | Value | Source |
|---|---:|---|
| Best public score | 0.6660 | seed-4 combo, `2e5j6bnt` |
| Milestone | 0.7000 | `RESEARCH_CONTRACT.md` |
| Required public gain | **+0.0340** | subtraction |
| Recorded field leader | 0.667720100448 | `artifacts/leaderboard_snapshot_2026-09-29.json`, captured 2026-09-28T21:18:14Z, 3,725 participants |
| Gap to recorded leader | **+0.00172** | subtraction |

Two local-to-public slopes have actually been measured:

* **head-procedure slope ≈ 0.58.** E410's complete-predictor CV gain was
  **+0.002414375** (CI `[+0.001566,+0.003449]`, 4 folds, 1,873 sequences,
  579 components); the corresponding public move was seed-3 complete 0.6655 vs
  previous D256 incumbent 0.6641 = **+0.0014**. Ratio ≈ 0.58.
* **seed-choice slope ≈ 0.27.** `NEW_SUBMISSION_AUDIT_2026-10-05.md:39`:
  seed-4 minus seed-3 was **+0.001872398** on the excluded panel
  (CI `[−0.000975354,+0.006931454]`) while the public difference was
  **+0.0005**. Ratio ≈ 0.27.

So the required *local* gain for the milestone is **+0.034 at slope 1.0 down to
+0.126 at slope 0.27**. The largest complete-predictor local gain ever measured
is +0.002414 (E410). The milestone therefore asks for **14× to 52× the best
measured lever**, or **24× to 52×** if you restrict to the two slopes that were
actually observed rather than the unattainable slope 1.0.

The same arithmetic says something the tree never states: **beating the recorded
leader is a different-sized problem from 0.7000.** +0.00172 public needs only
**[+0.0030, +0.0064] local** — one E410-sized lever plus one smaller one, or a
partial transfer of the activation result (E488's fixed-mixture effect was
+0.00893 / +0.00431 / +0.00869 across three seeds on an exposed panel).

**Consequence.** Every priority list in the tree mixes these two problems into
one ranked column. They should be split and budgeted separately:

* **Track A — beat 0.6677.** Honest expected public ≥ 0.668. Size: +0.003 to
  +0.006 local. Several registered, unrun items are in range.
* **Track B — 0.7000.** Size: +0.059 to +0.126 local. **Nothing currently
  registered is within an order of magnitude.** Track B can only be pursued by
  attacking the headroom map in §2, and each Track B proposal must state which
  cell it attacks.

---

## 2. Elephant two — where the error actually is (new computation)

`target_mechanics/strong_baseline/residual_report.json` publishes per-cell
weighted moments for 24 `class_event_age` cells. Summing the per-cell weighted
covariances and second moments reproduces the reported global figures exactly
(reconstructed **0.742063 / 0.806768** vs reported **0.742063 / 0.806768** for
t0/t1), so the moment algebra is sound. Re-running the same arithmetic with one
cell replaced by its oracle (`f := y` inside that cell, everything else held
fixed) gives an exact first-order value of perfection per cell.

Fixed predictor, fitted 16-group panel, complete (0.5 parent + 0.5 head),
global WP = 0.774416:

| Set to perfect | cells | weight share | ΔWP t0 | ΔWP t1 | mean ΔWP |
|---|---:|---:|---:|---:|---:|
| **class 3, all ages** | 6 | **26.5%** | **+0.1273** | **+0.0776** | **+0.102** |
| class 3, age ≥ 8 | 3 | 20.7% | +0.1140 | +0.0674 | +0.091 |
| class 3, age ≥ 32 | 2 | 16.0% | +0.0942 | +0.0540 | +0.074 |
| cell `3/128+` alone | 1 | 8.0% | +0.0502 | +0.0293 | +0.040 |
| cell `3/32-127` alone | 1 | 8.0% | +0.0440 | +0.0247 | +0.034 |
| all classes 0–2 | 18 | 73.5% | +0.1475 | +0.1254 | +0.136 |
| all age 128+ (any class) | 4 | 9.4% | +0.0531 | +0.0320 | +0.043 |
| all age 0 (any class) | 4 | 25.9% | +0.0407 | +0.0384 | +0.040 |

*(weight shares are the mean of the t0 and t1 shares; `3/128+` is 9.5%/6.5% and
`3/32-127` is 7.3%/8.7% for t0/t1 respectively)*

Read it as headroom *density*. Class-3 rows at age ≥ 8 carry 20.7% of the weight
and 0.091 WP of headroom — **0.44 headroom per unit weight**. All of classes 0–2
carry 73.5% of weight and 0.136 headroom — **0.185 per unit weight**. The
old-class-3 cells are **2.4× more headroom-dense than everything else**.

Cross-check against the tree's own earlier estimate: `ELEPHANTS_REVIEW_2026-10-04.md`
quotes E-313's public-mix linearization, *"+0.02 class-3 WP is worth ~+0.009
pooled"*. Extrapolated to the full class-3 headroom that gives ≈ +0.046 pooled
on a public-like mix — larger than my +0.102-on-fitted-mix figure would suggest
under a naive rescale, because the public mix carries ~46% class-3 weight. The
two agree on the direction and on the conclusion: class 3 is where the headroom
is, and it is not by itself enough.

Three things follow.

**(a) The milestone, priced in cells.** At slope 0.58, +0.0340 public needs
+0.0586 local — **57% of the entire class-3 headroom in first order**, i.e.
roughly the difference between today's class-3 WP (~0.42/0.45 for t0/t1) and
something near 0.70, with nothing else moving. At slope 0.27 it needs +0.126
local, which **exceeds all class-3 headroom combined (+0.102)** — class 3 alone
could not reach 0.7000 even if predicted perfectly. That is a fact worth writing
into `RESEARCH_CONTRACT.md`: at the measured seed-choice slope, the milestone
requires simultaneous class-3 *and* class-0/1 gains.

**(b) The class-3 weight understatement.** These shares are for the fitted
16-group panel, which is a *fitted* cohort, not a public-like mix.
`strong_baseline/REPORT.md` further notes that its 16-group excluded
counterpart *"contains classes 0/1/3 and **no class2**"* — so the class-3
weight share and the class-3 WP reported there (0.418856 fitted / 0.553540
excluded) are both measured on a class-2-free mix. E-298 independently
estimates the public class-3 weight share near 46% against 26.5% here. On a
public-like mix the class-3 headroom is worth materially more than +0.102.
Any Track B proposal should report its effect under **both** mixes, and should
not treat either 16-group panel as representative.

**(c) The proxy panel is real but narrow.** `PUBLIC_RESULTS_REFLECTION_20261004_LATE.md`
records both panels for the same three models:

| Model | full-file replay | excluded-368 | public |
|---|---:|---:|---:|
| previous D256 | 0.700792587 | 0.697777394 | 0.6641 |
| combo seed 3 | 0.700503744 | 0.699822159 | 0.6655 |
| **combo seed 4** | **0.700771292** | **0.701694557** | **0.6660** |

The excluded-368 panel reproduces the public order; **the full-file aggregate
inverts it** (D256 0.000021 above seed 4, while public ranks seed 4 0.0019
higher). The reflection itself says the fitted aggregate "would have
discouraged a useful" submission. So one panel agrees with public and one
disagrees — a two-outcome comparison, not a validated proxy. It should not be
the only gate on a submission decision, and the disagreement should be
revisited rather than cited selectively.

---

## 3. Elephant three — the metric-aligned objective test fell out of the synthesis

`review_20261005/MECHANISM_AUDIT.md` proposes five discriminating tests. Test 2
is *"train against the actual weighted-Pearson objective"*. It does **not**
appear in the Oct-5 elephants priority table, and it does **not** appear in
`RESEARCH_WRAP_2026-10-05.md`'s "Next-window research order" (items 1–5 are:
activation transfer, FP32-Muon H128 duration, memory+spline+posterior updates,
W384, cross-fitted teachers). It has been dropped by two successive syntheses
without a recorded rejection.

The evidence that it is worth running is already in `strong_baseline/REPORT.md:89`:

> "The head increases pooled WP but worsens pooled absolute-target weighted MSE
> for both targets in both panels."

Globally: parent WMSE `[0.76194, 0.60155]` → head `[0.78164, 0.61607]` (t0/t1),
while parent WP 0.771045 → head 0.776015. Prediction std 0.980/1.054 against
target std 1.286/1.303 — the surrogate is not even the right dispersion. The
incumbent *already* moves in the direction WP prefers and WMSE disfavors.

This is the one registered idea that changes the **estimand** rather than a
shape knob, and it is the only one whose mechanism is directly supported by a
published number. Cost: 2 fits at complete scale. Falsifier: no gain on 2 seeds.

Pre-declare the loss exactly, because WMSE and WP coincide under ideal
conditions (shared affine-optimal predictor); the divergence must come from
clipping to ±2, the |y| weight asymmetry, or finite-sample dispersion. The
test must specify: pooled over the sampled sequence, clip ±2, weight = |clipped
y|, per target, with a fixed stabilizer, matched against `wmse` + auxiliary
structure on identical parents.

---

## 4. Elephant four — a finished, free, never-run input-feature screen

This is the largest unused asset in the tree.

`campaign_2026-10-04/kaggle/FEATURE_SCREEN.md` and `FEATURE_SCREEN_V2.md`
define a **reciprocal, component-excluded, capacity-matched head screen** over
nine fixed feature families (v1) and eight exact conditional families (v2):

| ID | Family | Prespecified controls |
|---|---|---|
| H082 | regime heads (4 warmup-routed 32-coordinate state heads) | hash-routed heads, one universal head |
| H083 | regime affine (4 intercepts + 4 slopes/target) | hash routing, global affine |
| H084 | **target clocks** (t0 row EMA; t1 event EMA α 2/33) | swapped, common, row-both, event-both, lag4096 |
| H085 | **activity interactions** (history × EMA event rate/variance) | additive, stale128 gate, shuffled gate, history alone |
| H086 | **i1 innovations** (conditional raw i1 residual, lag 8) | current-row i1, lag-8 own-i0 residual, i1 lag136 |
| H087 | **event age** (history × log age of last i0 update) | additive, stale128 gate, shuffled gate, history alone |
| H088 | transitions (current−previous state, coordinatewise products) | stale128 transition, current raw input |
| H089 | cycle smoothing (trailing 8-row mean + EMA) | cycle7, row-EMA, phase8/phase7 |
| v1 | + volume28, activity clocks, multiplicative aux/state proxies | equal-width capacity control |

Status, verified by direct inspection:

* `cache/c20261004_feature_f0_a.npz` (19.3 MB) and `_b.npz` (16.7 MB) **exist**
  — the reciprocal caches were extracted on Oct 4.
* `research/campaign_2026-10-04/specs/feature_f0_{a,b}_v{1,2}.sealed.json`
  **exist**.
* `kaggle/feature_v2_drafts/` holds **32 sealed-capable CPU/GPU drafts**
  (8 families × 2 directions × 2 backends) plus histories and router drafts.
* **No result file exists for any of them.** The only `feature_screen` entries
  in `REVIEW_FILE_INDEX_2026-10-05.json` are the unit-test module. The five
  `feature_screen_tests_v*.log` files are test runs, not screen runs.

Cost: the screen is a 99-column ridge head on frozen cached states — **CPU
minutes, zero GPU hours, zero admission risk.** The GPU variants are the
follow-up, not the gate.

Why this is the right next thing rather than one more item on a list:

* H084/H085/H087 are exactly the mechanisms the headroom map points at (event
  age × activity, target-specific histories). `FEATURE_SCREEN.md:44-46` states
  explicitly that the proxy tests *"do not fulfill or close those exact
  proposals"* — they were built to be run, not to be argued about.
* H086 (`i1_innovation`, H-086) and H087 (`event_age`, H-087) are both still
  `open` in the ledger; H-084/H-085 likewise. The screen answers four open
  hypotheses at once with matched controls.
* It is the only listed direction that is *cheaper than the time it takes to
  argue about it*.

The Oct-5 wrap does not mention it. The Oct-5 elephants table ranks
"complete the low-cost open controls" last.

---

## 5. Elephant five — runtime gates the only ensemble lever, and the RAM block on the volume gate is file cache

**(a) The bag.** `NEW_SUBMISSION_AUDIT_2026-10-05.md:29`:

> "Fixed seed-3/seed-4 equal bag — No public ZIP/submission. The saved
> exposed-panel score is 0.701420130 versus 0.701694557 for seed 4. **Its
> local exposed-panel point is below seed 4**; there is no untouched
> public-transfer evidence or runtime approval."

and the full-file replay is 0.701106873 vs 0.700771292 for seed 4 alone. So the
two panels **disagree in sign**, and all paired intervals include zero. The bag
is *not* an obviously positive lever — it is an *unmeasured* one, and the local
evidence is genuinely mixed. Two trunks at incumbent per-member cost are
~74 min against a 60-minute limit. Nothing in the tree reports a timing
measurement of the bag.

**(b) q8.** `NEW_SUBMISSION_AUDIT_2026-10-05.md:33` gives bitwise-level parity
(i16 0.699821638 vs q8r 0.699816586, Δ −0.000005052, CI
`[−0.000043098,+0.000030580]`) and states plainly: *"Neither report supplies
timing clearance or public-transfer calibration."* H-068 is still `probing`
with `current_reason` = "compressed runtime integration remains pending".

**(c) The decisive measurement is CPU-only.** Neither (a) nor (b) needs GPU,
admission, or the science cutoff. One quiet-host timing run of (i) the bag and
(ii) a q8 package converts the single most uncertain deployment question in the
campaign into a number. Until that number exists, every "ensemble is infeasible
under 60 min" statement in the tree is an assumption, and ensembling is the
only lever with a multi-member averaging prior in its favour.

**(d) E427 (the volume28 gate) is blocked by inactive page cache.**
`E427_RECOVERY_AUDIT.md:36` records the Oct-5 sample as **24,063,160,320 bytes
of file cache (22,195,871,744 of it inactive)** against **1,236,369,408 bytes of
anon** — the process is not resident-memory hungry, the box is full of clean
cache from a prior 29.24-GB Parquet scan. The audit itself names the remedy and
defers it: *"An exact-input, read-only `POSIX_FADV_DONTNEED` pass could release
some cache, but its effect must be measured."* Nobody has measured it.

There are two incompatible framings of this block in the tree, and neither has
been checked against a measurement. `CURRENT_STATUS.md:65` states the block as
hard arithmetic: *"its 24 GiB worker plus 4 GiB support reserve exceeds the
actual 30 decimal GB aggregate cap"* (25.77 + 4.29 = 30.06 GB > 30 GB, so the
reservation genuinely does not fit). `E427_RECOVERY_AUDIT.md` instead says
headroom *"is insufficient for a conservative 24-GiB worker reservation **until
exact file-cache release and a fresh admission sample are made**"* — i.e. the
24-GiB figure is a reservation, not an observation. `memory.stat` puts observed
anon at **1.24 GB**, not 24 GB. So the 24-GiB reservation — the number the
capacity argument rests on — is itself the unverified assumption, and one
CPU-only command would resolve which document is right. Until then the volume
gate is blocked by a *model of* memory rather than by memory.

That matters because E413's pooled lead, small as it is (**+0.0009617902**,
bootstrap SD 0.0005622273, CI `[−0.000079965,+0.002059670]`, `p_le_0` = 0.039),
is **concentrated exactly where the headroom is**. Its per-class values from
`e413_paired_report.json`:

| class | volume-history WP | volume-current WP | Δ |
|---|---:|---:|---:|
| 0 | 0.819236 | 0.818066 | +0.0012 |
| 1 | 0.761705 | 0.762848 | −0.0011 |
| 2 | 0.752101 | 0.753239 | −0.0011 |
| **3** | **0.422082** | **0.418667** | **+0.0034** |

The pooled number is small because gains in class 3 and class 0 are cancelled
by losses in classes 1 and 2. This is the clearest class-3-concentrated input
result in the tree, and it is a direct fit with the headroom map in §2. Caveats: these are class-level **point differences with
no per-class interval reported**, one fold, one seed, and no feature-less arm
(see §7 row 1).

---

## 6. Elephant six — a finished idea backlog, not a shortage of ideas

The premise that ideas are exhausted is contradicted by the ledgers.

`hypotheses.jsonl`, 113 rows: **73 are in a non-terminal status** (38 `open`,
12 `probing`, 5 `proposed`, 4 `unknown_pending_evidence_audit`,
3 `proposed_not_fit`, 2 `queued`, 2 `screened_uncertain`, and 1 each of
`inconclusive`, `running`, `implemented`, `tool_ready`,
`blocked_by_frozen_fit_gate_failure`, `cpu_cache_and_patch_pending_independent_grade`,
`cpu_implementation_pending_independent_grade`). The remaining 40 are terminal:
18 `refuted`, 16 `supported`, 5 free-text verdicts, and one row whose `status`
is a wrap summary rather than a state. Of the 73, **10 carry
`implementation_status: checked`** — i.e. code verified, control specified,
waiting for a fit: **H-066, H-071, H-072, H-073, H-074, H-075, H-076, H-078,
H-079, H-098**. (An eleventh checked row, H-065, is already `supported`.) Nine
more non-terminal rows carry a `current_reason` containing "queued / pending"
— H-066, H-068, H-072, H-073, H-074, H-075, H-076, H-079, H-098.

`experiments.jsonl`, 481 rows: **E-431 … E-445 — 15 rows, all dated 2026-10-04
— carry `planned_gpu_fits` summing to exactly 38, and `result: null` and
`scientific_fits_completed: null` on every one**, i.e. **0 scientific fits
completed**. Thirteen are `planned_implementation_required`, E-437 is
`planned_cpu_analysis` (0 GPU fits, "under 1 CPU-hour"), E-442 is
`planned_cpu_diagnosis_then_fit` (3 GPU fits after a CPU diagnosis). Summing
their own `cost_estimate` bands gives **25.0–38.0 device-GPU-hours** (the
`26–35 GPU-hours` figure in the matrix header is the band for the *20-program*
second tranche, E429–E448). E-430 sits in a further-along state —
`three_arm_implementation_drafts_and_synthetic_discriminator_complete` — with
**3 more planned GPU fits**, also unrun.

Meanwhile `H-114` records **118 activation fits complete, 106 officially
graded** (the wrap says 224 completed activation fits across the programme),
and the wrap's verification section confirms all of them independently graded
and reproduced.

So the actual ratio this window is: **38 preregistered, mechanism-targeted,
costed fits — 0 run** against **224 externally-motivated activation fits — all
run.** The Oct-4 review's ranking was displaced by an arXiv-triggered sweep and
no document records the displacement or re-costs the displaced tranche.

Two smaller ledger defects, both cheap to fix and both actively harmful:

* **Status zombies.** H-071 is still `probing` with `current_reason` quoting a
  completed null: *"delta −0.000574, CI[−0.001797,+0.000303]. No positive
  signal"*. H-093/H-094 are `screened_uncertain` while their own
  `implementation_status` reads `production_fits_and_controls_completed` (the
  reasons cite eight E424 fits and conditional-adapter deltas with all intervals
  including zero). H-060/H-062/H-063/H-064 are
  `unknown_pending_evidence_audit` with `current_reason` = *"Historical record
  omitted status; no scientific outcome is inferred."* Anyone triaging from
  `hypotheses.jsonl` will spend GPU re-running things that already returned
  nothing.
* **H-112 and H-113** are `open` with a `design` pointer
  (`sequence_components/PREREG.md`) but **zero experiment rows reference them**,
  even though `E-467` (`complete_ten_fixed_GPU_fits_and_seven_paired_component_comparisons`,
  10 planned fits) and `E-469` (`complete_all_twenty_fixed_seed_replication_fits`,
  20 planned fits) — the two experiments that preregistration exists for — are
  both recorded as complete. Thirty finished fits are unlinked from the
  hypotheses they were built to test.

### 6.1 None of this is new — which is itself the finding

`ELEPHANTS_REVIEW_2026-10-04.md` already carried both an elephant and a table
for exactly this:

* **E-6** — "`research/hypotheses.jsonl` cannot be used as a work queue. Five
  rows carry live-looking statuses for resolved work: H-054 `running`, H-055
  `queued` (refuted by E-291), H-056 `implemented` (rejected by E-293), H-057
  `tool_ready` (rejected by E-292), H-058 `queued` (answered by E-345). … Anyone
  triaging off this file will spend GPU on refuted ideas."
* **§3 "Rejected, orphaned, or dropped for no good reason"** — T-1 volume GRU
  inputs, T-2 R11 thinning, T-3 H-046 branch head, T-4 the gen-3 teacher
  contradiction, T-5 W384 folds + Muon×width, T-6 a packaged-and-never-uploaded
  ZIP ("free — 1 upload").

Of those six, only T-1 (volume, as one clause of priority 6) survives into
`ELEPHANTS_REVIEW_2026-10-05.md`'s priority table. T-2 is still `open` as
H-080, T-3 is still `proposed` as H-046 with E-431 unrun, T-5 is still
unresolved, T-6 is unverified. E-6 was half-patched: H-060 and H-064, which Oct-4 flagged as having no
status field at all, now carry `unknown_pending_evidence_audit` with a reason
that explicitly infers nothing. The class of defect is unchanged — those rows
still cannot be triaged — and the ledger now holds 73 non-terminal rows two
days later, with more of them.

The pattern to name: **each review re-derives a good "dropped items" table, and
the following review drops the table.** That is the mechanism by which the
campaign runs 224 externally-motivated fits while 38 preregistered ones wait.
The fix is not a seventh review; it is a single living artifact that both
reviews append to and the controller reads as a work queue.

Related, still-live free items worth a one-minute check rather than an argument:
`artifacts/submission_quota_2026-10-03/` still holds four packaged ZIPs
(reference D256, teacher TE07, precision q16x, chunk500 ensemble) untouched
since Oct 3 22:47, and the campaign has a 5-submissions/day quota with the last
upload on Oct 4. Whether any of those four is an unspent *measurement* should be
answered by inspection, not by another review.

---

## 7. Elephant seven — rejections that were never tests

Each of these is cited in a review or a status file as settled. Each has a
stated design that does not answer the question it is credited with answering.

| # | Claimed result | What was actually run | Gap |
|---|---|---|---|
| 1 | E413 is read as answering the volume question (Oct-5 review: *"It is no longer 'never tested'"*, then asks only for replication) | `e413_paired_report.json`: HISTORY (`--research-features volume28`, run-relative history) vs CURRENT (`--research-features volume28_current`, current-row volume), fold 0, seed 3, `n_boot` 2000 | **Neither arm is feature-less** — there is no no-volume arm at this scale, so the contrast run was history-vs-current, not volume-vs-none, and the CI crosses zero by 8e-5. The designed definitive test is E427 (`aligned`/`stale`/`noop` arms, volume projected into the GRU update-gate), which is blocked (§5d). Replication without a feature-less arm cannot distinguish "volume helps" from "current-row volume hurts". |
| 2 | "target-specific clocks are refuted" | E426: three teacher-consistency EMA clock fits, Δ −0.000046 | That tests a *teacher timing*, not an input/history adapter. E433 (target-specific row/event clocks, 3 fits) is `planned_implementation_required` and unrun, and the H084 target-clock family in the ready-to-run screen (§4) was never run either. |
| 3 | "joint heads fail" | E468 `joint_target_model/RESULT.json`, one fold, all arms vs the parent-pair baseline: joint_tilted **−0.000618**, regression_anchor **−0.000555**, shuffled placebo −0.000550, independent −0.001354 | Every arm — including a shuffled placebo and a plain regression anchor — lands within ±0.0008 of every other and below the baseline; **shuffled beats joint_tilted**. The protocol could not separate joint structure from a placebo, so it refutes the *protocol*, not the joint-head idea. |
| 4 | "EMA horizon is null" | E423, one fold, CI `[−0.0018,+0.0003]` | Point estimate negative, CI width ±0.001, declared on one fold. |
| 5 | H-046 branch/U head "refuted for this backbone" | CE-vs-WMSE single runs | The end-to-end arm **E431 (2 fits) was never fit**. The 0.968 oracle is conditioned on target-derived class and a physical decoder that later failed — privileged, not deployable. |
| 6 | E-256 / E-264 / E-233 rejected | point deltas of −0.0001 / +0.00015 / +0.00004 against bootstrap SDs of 0.0015 / 0.00115 / 0.00037 — i.e. **0.07–0.13 SD**, on fold 0 or one pure-chain panel | A promotion gate larger than the noise cannot detect an effect this size. These are "not detected", not "absent". |
| 7 | E-357 "close this" | CI `[−0.00266,+0.00015]` | Crosses zero; the instruction was to close. |
| 8 | Width is settled | W384 has repeatedly ended in a resource guard (E449: `..._paired_fits_stopped_on_host_RAM_guard_no_complete_checkpoints_or_scores`; E422 OOM evidence retained). E409 measured W384_head1 at **401.617 kcycles/row vs D256_head1 195.424 = 2.06×** | Width has produced neither a completed matched score nor a runtime clearance. It is unresolved on cost *and* on score, and is cited as if settled. |
| 9 | Longer training is settled | Fixed at 4 epochs / fixed epochs everywhere; E417 itself says reset results *"do not establish whether longer training credit improves predictions"* | The registered duration study (E485/E489, 12 trajectories) never ran — both probes failed before any scientific fit. Duration has never been swept at complete scale. |
| 10 | Metric-aligned loss | Not run (§3) | Dropped from two syntheses without a rejection. |

None of these should be re-litigated on the same evidence. Items 1, 3, 5 and 10
have concrete, costed designs already in the tree.

---

## 8. Priorities

Ordered by expected information per unit cost, with the cost stated. Track A
targets beating the recorded leader; Track B targets 0.7000 and is explicitly
labelled as such.

### P0 — no GPU, this window (do these before spending anything)

1. **Run the nine-family + eight-family feature screen** (`FEATURE_SCREEN.md`,
   `FEATURE_SCREEN_V2.md`) on the existing `feature_f0_{a,b}` caches. CPU
   minutes. Answers H-084, H-085, H-086, H-087, H-088, H-089, H-082, H-083 and
   the volume28 control family with prespecified capacity and placebo controls.
   Gate: any family must clear the harness's own `+0.0005` screen threshold
   *and* beat its named control on both reciprocal directions before promotion.
2. **Time the seed-3/seed-4 bag and a q8 package on a quiet host.** CPU only.
   This is the single highest-value unknown in the deployment path. Report
   minutes/ZIP, not a projection.
3. **Measure `POSIX_FADV_DONTNEED` on the exact E427 input FDs, then re-sample
   the parent slice.** If the guard then admits, E427 is unblocked; if not,
   record the number and close it. Either way it stops being an open question.
   (The audit mandates exactly this sequence and nobody has executed step one.)
4. **Repair the ledgers.** Move the completed-null hypotheses (H-071, H-093,
   H-094, H-060, H-062, H-063, H-064, and any other
   `unknown_pending_evidence_audit` whose experiment finished) to terminal
   status; attach the finished E467/E469 rows to H-112/H-113. ~30 minutes,
   prevents repeated GPU spend.

### P1 — Track A, cheap and registered (target: +0.003 to +0.006 local)

5. **Train against the actual metric** (§3). 2 fits. Pre-declare the loss.
6. **E413 done properly**: add a matched **feature-less** arm at the same scale,
   fold and seed, so the contrast becomes volume-vs-none instead of
   history-vs-current; then repeat on a second seed. **2–4 fits.** This is the
   cheapest path to a confirmed class-3 input gain: the existing differential
   is already +0.0034 in class 3 (§5d), but nobody can currently say whether
   that is history helping or current-row volume hurting. **Then run E427** —
   the designed three-arm test (`aligned` / `stale` / `noop`, 3 fits), volume
   projected into the first-layer GRU update gate — which is exactly what P0.3
   unblocks. E413's replication without a feature-less arm cannot settle the
   question; E427's `noop` arm *is* that arm.
7. **Complete the E431–E445 tranche as a block, or formally retire it.** 15
   experiments, **38 planned GPU fits (`planned_gpu_fits` sums to exactly 38),
   0 scientific fits completed**, costed **25.0–38.0 device-GPU-h** by summing
   their own `cost_estimate` bands, all dated Oct-4, never run. E430 sits one
   step further along — its three implementation drafts and synthetic
   discriminator are complete, its **3 GPU fits** are still unrun. Highest
   prior on any batch in the tree precisely because it was preregistered
   *before* the activation programme displaced it. Sub-order by the headroom
   map: E434 (activity/event-age interactions) and E435 (i1 innovations) first
   — both attack class-3/old cells — then E431 (branch readout), E433
   (target-specific row/event clocks).
8. **H-098 origin-privileged teacher** (3 fits per the Oct-4 matrix,
   `implementation_status: checked`, `current_reason` ending *"scientific
   teacher/student fits pending"*; its only linked row, E-461, is the CPU
   bridge check, not the fits). Training-time privileged information with a
   causal inference path — the one registered idea whose information asymmetry
   exists only during training, so a positive result is directly usable.

### P2 — Track B, explicit headroom attacks (target: class-3 WP 0.42 → 0.70)

Every proposal in this track must name the cell(s) from §2 it attacks and
report class-3 WP separately on both mix panels. Anything that cannot name a
cell is Track A wearing a costume.

9. **Joint conditional sign+amplitude head / transition-posterior-update branch**
   — this is what `strong_baseline/REPORT.md` itself recommends, with matched
   capacity and the complete head-training procedure, and it is the only
   proposal in the tree aimed at the 46–50% of class-3 squared error that comes
   from *wrong sign* despite 82–91% sign accuracy.
10. **Metric-aligned objective** (§3), extended to 2 seeds if P1.5 is positive.
11. **Activation transfer** (wrap item 1) — keep it, but run it as a *pair*
    with a matched unchanged-readout control and report it against the
    class-3 cells; E488's effect is readout-shape, so it may not touch class 3
    at all, and that is the first thing to know.

### Explicitly deprioritized

* Further activation-variant sweeps at H64. 224 fits have already
  characterized this family; the wrap's own item 1 correctly converts it to a
  transfer question.
* Any new idea review. Four elephants reviews, two audits, a guide and an
  index now converge on the same three conclusions. The next document that
  helps is a run log, not a ranking.
* Re-testing any item from §7 on the same design.

### Stop rules

* P1 items 5 and 6 are single-block experiments: two seeds each, then stop
  regardless of sign, and write the result into the ledger either way.
* The E431–E445 tranche is all-or-nothing: either fund all 38 fits in priority
  order within one window, or record a dated retirement. Partial runs that stop
  after two items reproduce the current failure mode.
* Track B gets a cap. If no class-3 cell moves by ≥ +0.01 WP after items 9–11,
  the honest conclusion is that 0.7000 requires information the campaign does
  not yet have, and the contract should say so in writing rather than remaining
  silently unmet.

---

## 9. Summary

The campaign does not have an idea problem. It has a **conversion problem**:
38 preregistered fits and a finished nine-family feature screen are sitting
idle while 224 externally-motivated fits were executed; ten hypotheses are
`implementation_status: checked` with no fit; a five-test mechanism audit lost
its best test in synthesis; a volume gate is blocked by clean page cache; the
one lever with multi-member averaging is gated on a CPU timing number nobody
has measured; and a milestone of 0.7000 has never been priced against the two
slopes that were actually measured — after which it turns out to require
**14× to 52×** the best lever ever demonstrated (24×–52× using only the two
observed slopes), or an attack on class-3 old rows that alone carry only
~+0.10 local WP.

Split the goal into "beat 0.6677" (in range) and "reach 0.7000" (requires the
headroom map), run the free screen and the free timings first, then spend the
GPU on the tranche that was already reviewed and already costed.
