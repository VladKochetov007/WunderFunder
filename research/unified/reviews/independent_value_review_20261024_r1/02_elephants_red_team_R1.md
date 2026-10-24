---
kind: review
status: complete
created_utc: 2026-10-24T17:21:00Z
lens: elephants-and-misleading-beliefs
author: independent subagent (fresh context)
---

# Elephants and misleading ideas — 2026-10-24

One-shot, read-only red-team audit. All claims carry a source path; "measured" =
number read from a repo receipt; "inference" = my judgment. Overlaps with the
2026-10-23 Astra synthesis and the 2026-10-03/05 elephant reviews are marked
explicitly; the load-bearing findings below go beyond them.

## Main judgment (bullets)

- **The noise floor is measured and the celebrated effects sit under it.** The
  repo's own records put full-endpoint seed noise at ~0.0007–0.0014 WP
  (E-176, P8, forensic panel), while the flagship recent positives are
  +0.0003-scale (i1 mixture, Kalman screens, 0.1-shrink). The portfolio of
  small positives is direction evidence, not established gains — and it is
  partly selected noise until a predeclared replication lands.
- **TL3 is not a black box, it is misallocated attention.** Members, weights and
  even the w18-family recipe are documented; what is missing is the FTT/
  tiedlead source (263 inputs incl. 52 lead features, 0.66 of the weight).
  Measured public evidence says the student slot we keep replacing is worth
  only +0.0006; the score mass sits in the two members we cannot retrain.
- **The final event is private-only and fed from previously submitted solutions
  (docs/rules.md).** The two checked quota ZIPs are unuploaded; ~22 days
  remain; the runtime ceiling (60 min, unknown final-set volume) has accepted
  packages as slow as 57:27 with one engine timeout on record. Final-event
  logistics, not more +0.0003 screens, is the largest unmanaged risk.
- **0.700 is not reachable by blending this model family.** Team public scores
  plateau at 0.6680–0.6683 across many compositions; the entire rank-22→rank-1
  spread is 0.0041. The only repo-evidenced 0.700-scale route is structural
  decoding of the high-|Y| target classes (oracle branch readout 0.968), a
  direction abandoned in September.
- **Teammate overlap:** our transformer-teacher ladder and 52-target
  future-input dynamics head substantially duplicate the already-50%-trained
  PR32 teachers and the FTT "52 lead features". Exploiting theirs is cheaper
  than building ours.

## Ranked elephant table

| # | Belief | Why suspect | Evidence for | Evidence against | Decisive check | Severity |
|---|---|---|---|---|---|---|
| 1 | "Recent +0.0003 results are real gains" | Below measured seed noise; sign-flips across seeds documented | i1 mixture positive in 2 cells; PR26/27 +0.0009 at 10% | seed noise 0.0007–0.001 (tree.md:197,217; opus_independent V5; forensic:195); fast/slow seed3 +0.00005 vs seed4 −0.00001 | One predeclared 2-seed replication of the i1 mixture on a third cell | Critical |
| 2 | "TL3 is unknown; our job is better students" | Members known; student slot worth only +0.0006 publicly; FTT holds the score | Two-parent 0.66771 vs TL3 0.66832 (measured, PUBLIC_RECEIPT_R1.json); H512 replacement 0.66667 | FTT source unlocated (TEAM_METHODS_AND_EVIDENCE_R1.md:15) | Ask teammate for FTT training push / lead-feature code; grade an FTT-feature student | Critical |
| 3 | "Public best anchor aligns with the prize" | Final ranking solely private, from previously submitted solutions (docs/rules.md:24) | rules.md is explicit; robust-candidate-must-be-submitted noted in 2026-10-12 strategy review:97 | Repo continues optimizing around TL3's public 0.66832 | Upload the two quota ZIPs now (quota exists: 5/day, 99 used total) and fix a final-selection plan | Critical |
| 4 | "Runtime is fine; callback fits" | 60-min cap with unknown final-set volume; observed 50–58 min packages and 1 engine timeout | catalog: TL4 50:07, QAT 54:26, CPU C 57:27, CPU B timeout; public_receipts: 3 timeouts | TL3 itself 39:12; pair only +24.1% callback (quota handoff:69) | Establish final/private row-count bound and a hard self-imposed ≤45-min package gate | High |
| 5 | "0.700 public is an achievable near-term target" | Gap 0.0317 ≈ 8× the whole top-22 spread; blending plateau measured | User objective (specs/research-objective.md); no ceiling proven | 0.6680–0.6683 across 6+ diverse compositions (CURRENT_MODEL_CATALOG.md:9-17); leader 0.67243 | Test one structural class-3 decode arm (Sep-era oracle 0.968) vs more blending | High |
| 6 | "Small positives on exposed folds aggregate into progress" | Adaptive selection on repeatedly exposed panels inflates the portfolio | Each receipt honestly labels exposure | Follow-ups designed after earlier grades (shrink after F0; Kalman expanded after first; PR26/27 10%→25% reversal) | Publish a count of selected-after-looking comparisons; preregister next panel before grading | High |
| 7 | "We train/emphasize the metric tail correctly" | ~43–47.7% of weight is class 3 where WP is 0.43 vs 0.74–0.81 elsewhere; unexamined since September | lit/validation.md:154; ELEPHANTS 10-03 E2; EV12 47.69% | Weighting is population-optimal under WMSE (astra claim review) | Per-weight-decile decomposition of TL3 vs H390 students on the quota sample (zero new fits) | Medium-high |
| 8 | "4–6-fit screens tell us what to do next" | Their own intervals cross zero; power ~nil vs 0.0007 noise | ADR0043 admits "direction, not significance" | Decisions still consume +0.0003 directions as evidence (i1, Kalman, PR26/27) | Screen only where effect ≥2× plausible noise, or use fixed-prediction paired grades (zero fits) | Medium |
| 9 | "We and teammates build complementary things" | PR31 ladder ≈ PR32 (630M/1.54B, 50.37% trained); our 52-target dynamics head ≈ FTT's 52 lead features | ALL_PR summary; FUTURE_INPUT_DYNAMICS_AUXILIARY vs catalog:26 | PR32 has zero evaluated endpoints; no consumer plan recorded | Schedule PR32 evaluation + GRU-transfer plan before designing another teacher | Medium |
| 10 | "0.66832 is our true public quality" | Winner's-cursed: best of 97–99 adaptively ordered submissions | public_receipts.json: 65 scored, max 0.6682; 97→99 submissions | Selection was composition-level, not coefficient-fitted (repo policy) | None feasible; treat 0.66832 as upper bound, expect private regression to mean | Medium |

## Deep dives (top 3 elephants, evidence-rich)

### Deep dive 1 — The noise floor vs the celebrated effects (elephant 1, 6, 8)

Measured noise (repo receipts, September–October era):

- `research/tree.md:197,217` — "seed noise: FT ~0.0008"; "seed noise measured:
  s4 − s3 pooled +0.0003 ± 0.0007 (E-176)".
- `research/lit/opus_independent.md:301` — "Seed 3 vs seed 4 differ by 0.0010
  on full valid (P8)"; `research/lit/opus_arch.md:425` — "paired full-valid
  deltas carry about 0.001 of seed noise".
- `research/panel/forensic.md:195` — "C3 shows seed noise of finals at about
  0.0007"; independent retrain error ±0.0015.
- `research/ELEPHANTS_REVIEW_2026-10-03.md:81` — fold0 seed noise ~±0.001.

Celebrated recent positives (measured): i1 fixed 90B/10C +0.00036/+0.00027,
one interval crossing zero (campaign_2026-10-21 i1 GRADE_R1); Kalman
replacement +0.00034 then +0.00012 (ideas README:47); 0.1-shrink +0.00013 F0
but −0.000005 F1 (campaign_2026-10-24 synthesis §heads); proxy gate +0.00005;
checkpoint mean +0.00001. Every one is ≤0.5σ of endpoint seed noise.
(My inference: for fixed-prediction blend comparisons the relevant uncertainty
is the component bootstrap, which the repo reports honestly — but the
underlying B/C/H390 members are single fits per cell, so a +0.0003 blend gain
is equally consistent with "that seed draw happened to complement" — exactly
the ordinary-diversity alternative the Astra synthesis raised.)

Sign-flip precedents on record: groupwise fast/slow seed3 +0.00005 vs seed4
−0.00001 (ideas README:160, "much smaller than seed-to-seed movement");
PR26/27 innovation +0.0009 at 10% collapsing to ≈0 at 25% (campaign synthesis
§PR evidence); spline-head lead shrinking +0.039→+0.013 across three seeds
(THREE_SEED_REPORT.md:32-40). The portfolio pattern — initial positive,
replication fades — has occurred at least three times. (Overlap note: the
Astra synthesis flagged "ordinary seed diversity not priced"; the noise-floor
quantification and portfolio-level selection accounting above go further.)

Exposure aggregation (elephant 6): the repo never counts how many comparisons
were selected after looking. Documented instances: Kalman panel 2 "selected
after the first" (ideas README:47); F1 shrink "designed after F0" (synthesis
§limits); quota strata "sampling strata, not clean holdouts" (quota handoff
§limits); i1 F2 diagnostic after F0 interim. (My inference: with ~6+ adaptive
selections per lineage on the same 4 folds, an aggregate soft inflation of
+0.0001–0.0003 per "positive" is plausible; the honest aggregate statistic is
a preregistered ledger of comparisons run vs comparisons reported.)

### Deep dive 2 — TL3 anatomy: we are optimizing the wrong slot (elephant 2, 9)

Measured/documented anatomy (`research/unified/models/CURRENT_MODEL_CATALOG.md`,
`research/campaign_2026-10-18/online_features_r1/DATA_AUDIT.md:22`):

- TL3 = 0.34 `w18final_distilled_s5@ep1` + 0.33 `fttall_tlead_s5@ep3` +
  0.33 `fttall_tlead_s6@ep3`; public 0.66832, 39:12, portal `1lmehfi3`.
- The w18 distilled family is essentially fully mined:
  `TEAMMATE_RECIPE_DIGEST.md` records student recipe (GRU256×2, 2 epochs,
  AdamW 1e-4, chunk 2000, alpha 0.7, propensity² eps 0.03, aux-struct/next
  0.3/0.3) and the teacher cache weights summing to 0.8125 (unnormalized —
  a ~19% shrink of the distillation target, "observation to verify").
- The FTT/tiedlead family — 0.66 of TL3's weight — is the actual unknown:
  263 inputs = 211 + **52 lead features**, shared instrument encoder, scratch
  parents, EMA; "a fresh remote/branch/PR/fork search did not locate the later
  FTT training push" (TEAM_METHODS_AND_EVIDENCE_R1.md:15).

The decisive public decomposition (measured, `quota_retained_pair_r1/`):
removing the student entirely (two FTT parents at 0.5) scores **0.66771**
(−0.00061 vs TL3) and runs 10m41s faster; replacing the student with our H512
scores **0.66667** (−0.00165). (My inference: the student slot's total public
value is ≈+0.0006 and our replacements have so far destroyed it; the ~0.003+
"TL3 gap" that all local reproductions trail lives in the FTT members, whose
52 lead features are a feature family our entire local 211-input lineage has
never contained.)

Yet the October effort allocation: i1-history screens, online heads, shrink
diagnostics, direction probes, batch-mass audit, and the two new quota ZIPs —
all operate on student-slot or own-lineage models. The quota ZIPs again swap
the 0.34 student (H390 S3/S4, +0.0037 F0 on exposed strata). Nobody has
scheduled: (a) recovering FTT source from the teammate (ADR0035 authorizes
asking/sharing), (b) retraining an FTT-feature member, or (c) even grading
whether our H390 students' local F0 advantage survives when the FTT members'
errors are removed. (Overlap note: the Astra synthesis treated TL3 ancestry as
a caveat, not as the largest attackable gap; the 10-05 review's "compare with
the champion" elephant was about prototypes, not about the FTT/two-parent
decomposition, which only became measurable on 10-22.)

Duplication corollary (elephant 9): our proposed future-input dynamics head is
"52-target" (`FUTURE_INPUT_DYNAMICS_AUXILIARY_20261021_R1.md`) and FTT's extra
features number exactly 52 (catalog:26) — almost certainly the same mechanism
(teammate got there first, trained it, and it is sitting inside TL3). Our PR31
teacher ladder (0.2/0.68/1.65B) overlaps PR32's already-running 630M/1.54B
teachers at 50.37% (23,725/47,100 updates, zero evaluated endpoints;
`PR_REFRESH_INTERPRETATION_R3.md`).

### Deep dive 3 — The final event: private scoring, submission eligibility, runtime ceiling (elephants 3, 4, 10)

Rules (measured fact, `docs/rules.md:24-25`): "Your final position will be
determined solely by your score on the Private Leaderboard… you will be asked
to choose some of your *previously submitted and scored solutions* to be
re-scored on the Final Dataset." Tie-break: earlier submission ranks higher.
`docs/submission_guide.md`: 1 vCPU, 60 minutes for the entire test set, 20 MB
ZIP (note: 20 MB, not the 5–16 MB sometimes quoted; all current packages fit).

Consequences the repo has under-managed (my inference from documented facts):

1. **Unuploaded candidates are final-event-ineligible.** The two quota ZIPs
   (16.2 MB / 12.5 MB, technically checked, `quota_submission_r1`) and every
   future candidate must actually pass through the portal before 2026-11-15.
   The 2026-10-12 strategy review:97 already flagged this ("a robust private
   candidate also needs a successful submission before the deadline"); no
   operational upload schedule exists in the 10-24 close-out. Quota is not
   scarce (5/day; 99 total used), so the blocker is process, not allowance.
2. **Runtime headroom is thinner than "60 min" suggests.** Observed portal
   runtimes: TL3 39:12, H512 50:07, TL4 50:07, xteam 50:16, QAT 54:26, CPU C
   57:27, CPU B **timeout** (catalog:11-17, :34); 3 timeouts among 65 scored
   receipts (`public_receipts.json` summary). The new pair callback is +24.1%
   vs TL3 (quota handoff:69) — projecting ≈48–49 min if TL3 is 39:12.
   (My inference at 1.24×: acceptable for the current test set, dangerous if
   the final set has more rows.) The 10-12 review:95 already computed that
   2× rows would put a 73-min package over the cap; the recommended
   "establish the private row-count bound" remains open three weeks out.
3. **Public-best is a selection statistic.** 0.66832 is the max of 97–99
   adaptively ordered team submissions; private re-scoring regresses such a
   max toward the family mean (~0.668). Combined with the user's
   future-data premise for the private set (strategy review:17), the
   compact/robust candidate deserves equal billing with the mean-best one.

## Misleading-ideas table

| Idea | How it misleads | What to do instead | Repo evidence |
|---|---|---|---|
| "i1 mixture is our strongest recent evidence" | True-but-weak: +0.0003 under 0.0007–0.001 seed noise | Call it a direction; demand one predeclared replication before further family spend | i1 GRADE_R1.json; tree.md:217 |
| "quota ZIPs +0.0037 F0 ⇒ likely public gain" | Student slot worth only +0.0006 publicly; F0 stratum exposed | Expect ≈0 public change; upload as transfer measurement, not as contender | quota handoff:44-52; PUBLIC_RECEIPT (two-parent) |
| "TL3 gap ≈ 0.003 needs a better student" | Gap lives in FTT members we never built | Recover FTT source; train a lead-feature member | NATIVE_HOST_GRADE_R1 (−0.0074); catalog:26 |
| "H390 = wider students" | H390 is an experiment id; members are H256×2 | Always name architecture explicitly | quota handoff:34; strategy review:61 |
| "0.700 via ensembling/teacher-distillation increments" | Compositional plateau measured at 0.6680–0.6683 | Treat 0.700 as requiring a new information source (class-3 structure) | catalog table; ELEPHANTS 10-05 §2 |
| "exposed folds are OK because intervals are reported" | Conditional bootstrap omits training, selection and transfer variance | Pair every promoted claim with a seed-level or preregistered check | quota handoff §limits; synthesis §limits |
| "boundary censoring 0.074% is a representation lead" | It affects <0.1% of field-row pairs; potential score mass is tiny | Rank age/clock screens below tail-weight and final-event work | eda README; ELEPHANTS 10-03 E2 contrast |
| "our transformer-teacher plan is the team's scale path" | Teammate PR32 already 50% trained at similar scale | Become PR32's evaluation/transfer consumer | PR_REFRESH_INTERPRETATION_R3.md |
| "5–16 MB is the size limit" | Official limit is 20 MB | Use the doc number; headroom exists for one more member | docs/submission_guide.md:91 |

## What is NOT an elephant (explicitly cleared beliefs, with evidence)

- **"Is the data really market data?"** — Officially documented: "two trading
  instruments i0/i1, orderbook- and trades-related features… future price
  movement of instrument i0" (`docs/data_overview.md:3-6`). "Connectome" is
  branding; the repo already reasons in microstructure terms. Only caveat:
  rules.md forbids pre-trained models containing external market data.
- **Metric implementation doubts** — `src/wnn/gmetric.py` verified: clips
  predictions and labels to ±2, weights = |clipped label|, zero-weight rows
  dropped, mean of two target correlations. Matches every receipt's usage.
- **"TL3 composition is a total mystery"** — Overstated. Members, weights,
  public receipts and the w18 recipe are documented (catalog, DATA_AUDIT,
  TEAMMATE_RECIPE_DIGEST). The true hole is FTT ancestry/membership only.
- **ZIP size pressure** — All packages 8.9–16.8 MB vs 20 MB limit; not binding.
- **"Screens are believed to be conclusive"** — ADR0043 explicitly says two or
  three cells "estimate direction, not significance". The practice gap is the
  elephant (#8), not the stated belief.
- **Kaggle quota scarcity** — ~108k GPU-s unreserved (fact base) and 5
  submissions/day; neither is the binding constraint on final-two-week value.

## Sources inspected

specs/research-objective.md; specs/current-state.md (top 120 + LB section);
src/wnn/gmetric.py; docs/rules.md; docs/submission_guide.md;
docs/data_overview.md; research/campaign_2026-10-24/resume4h_r1/
CAMPAIGN_SYNTHESIS_R1.md, quota_submission_r1/RESULTS_AND_HANDOFF_R1.md,
ALL_PR_RESEARCH_SUMMARY_R1.md (excerpts), PR_REFRESH_INTERPRETATION_R3.md;
research/unified/reviews/astra_ultra_audit_20261023_r1/SYNTHESIS.md,
EVIDENCE_MAP.md (excerpt); research/unified/ideas/README.md;
research/unified/models/CURRENT_MODEL_CATALOG.md;
research/campaign_2026-10-20/submission_delivery_r1/tl3_member_adaptation/README.md;
research/campaign_2026-10-22/quota_retained_pair_r1/{README.md,PUBLIC_RECEIPT_R1.json};
research/campaign_2026-10-13/team_models_r1/TEAM_METHODS_AND_EVIDENCE_R1.md;
research/campaign_2026-10-06/.../TEAMMATE_RECIPE_DIGEST.md;
research/campaign_2026-10-18/online_features_r1/DATA_AUDIT.md (excerpt);
research/unified/archive/2026-10-12/pages/03-submissions.md;
research/unified/reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md;
research/unified/reviews/audit-2026-10-17/public_receipts.json;
research/ELEPHANTS_REVIEW_2026-10-03.md; research/ELEPHANTS_REVIEW_2026-10-05.md;
research/tree.md, research/panel/forensic.md, research/lit/opus_independent.md,
research/lit/validation.md (noise/class evidence, via targeted search);
specs/adr/0043-fast-screens-and-transformer-teacher-scale.md.
