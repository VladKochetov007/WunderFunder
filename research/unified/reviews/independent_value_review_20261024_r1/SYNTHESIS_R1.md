---
kind: review
status: complete
created_utc: 2026-10-24T20:10:00Z
updated_utc: 2026-10-24T21:30:00Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/unified/reviews/independent_value_review_20261024_r1/01_value_allocation_R1.md
  - research/unified/reviews/independent_value_review_20261024_r1/02_elephants_red_team_R1.md
  - research/unified/reviews/independent_value_review_20261024_r1/03_questions_and_directions_R1.md
  - research/experiment_drafts/own-optimizer-muon-20261024/RESEARCH_PROPOSAL_R1.md
---

# Independent value, blind-spot and questions synthesis — 2026-10-24

## Question or claim

The user asked three fresh-context independent reviewers to think about how to add
value to our ideas, which questions we should be asking, which directions to look
around, and whether there is an elephant in the room we are missing — explicitly
suspecting that "we may be losing something or some ideas are misleading". A
fourth work stream (user-requested the same evening) researches making our own
optimizer, Muon-inspired, including bounded experiments. This synthesis compares
the independent outputs, reconciles their disagreements, and states the combined
recommendation. It is a tier-C source/literature/document review plus one bounded
local synthetic screen: zero competition-data fits, zero provider jobs, zero
uploads, zero git mutations.

## What was done

Three independent fresh-context subagents with distinct lenses, run in parallel
(at most three active subordinate roles, per policy), each writing its own report
before seeing the others:

1. [Value allocation and endgame review](01_value_allocation_R1.md) — where
   expected final-placement value is concentrated given 22 remaining days
   (deadline 2026-11-15, `docs/timeline.md`).
2. [Elephants and misleading ideas red team](02_elephants_red_team_R1.md) —
   adversarial audit of load-bearing beliefs, with repo evidence for/against.
3. [Questions and external directions scout](03_questions_and_directions_R1.md) —
   20 ranked questions, 8 transferable-component cards, 3 wildcards, with a
   logged literature scan (arXiv abstracts + web).

A fourth scout produced the [own-optimizer research proposal](../../../experiment_drafts/own-optimizer-muon-20261024/RESEARCH_PROPOSAL_R1.md)
(8 candidate cards C1–C8) from the user-supplied Muon transcript, repo
archaeology and an optimizer-literature scan; a bounded synthetic CPU screen
(4 optimizers × 3 seeds) implements its first mechanism probe and is recorded in
the same draft folder. Provider instability caused several launch failures; every
report on disk was verified, and one overwritten draft was explicitly
incorporated by its successor (noted inside the reports).

The three lenses reached overlapping conclusions **independently** — the
strongest signal this review offers. Agreement here is still same-harness,
same-model-family agreement; treat it as convergent analysis, not independent
measurement.

## The convergent judgment (what all three lenses found separately)

1. **The final event, not the next experiment, is the largest unmanaged risk.**
   `docs/rules.md`: final ranking is computed **privately, solely from
   explicitly chosen, previously submitted and scored solutions**; ties break to
   the earlier submission. Therefore: (a) an endpoint never uploaded is worth
   exactly zero for final placement; (b) the two technically checked, delivered
   but **unuploaded quota ZIPs** (TL3+H390 pair, +0.0037 F0/+0.0002 F2 vs TL3 on
   exposed strata) currently have zero placement value; (c) upload champions
   early. Quota is not scarce (5/day, ~102 used); the blocker is process.
2. **The celebrated +0.0003-scale effects sit below the repo's own measured
   noise floor.** Documented full-endpoint seed noise is ~0.0007–0.0014 WP
   (`research/tree.md`, `research/lit/opus_independent.md`,
   `research/panel/forensic.md`); the flagship recent positives (i1 mixture,
   Kalman screens, 0.1-shrink) are ≤0.5σ of it, and the repo records at least
   three initial-positive-then-fade precedents. The small-positive portfolio is
   *direction evidence, partly selected noise* — an aggregate accounting problem
   (how many comparisons were selected after looking?) that no single receipt
   exposes.
3. **TL3 is misallocated attention, not a black box.** TL3 = 0.34
   w18-distilled student + 0.33+0.33 FTT/tiedlead teachers (`research/unified/models/CURRENT_MODEL_CATALOG.md`).
   Measured public decomposition: removing the student entirely scores 0.66771
   (−0.0006 vs TL3); our H512 student replacement scores 0.66667. **The student
   slot we have spent weeks optimizing is worth ≈+0.0006 publicly; the score
   mass sits in the two FTT members (0.66 of weight, 263 inputs incl. 52 lead
   features) whose training source was never located** — and whose 52 lead
   features are almost certainly the same mechanism as our proposed 52-target
   future-input dynamics head. Exploiting FTT (source recovery, lead-feature
   member) is cheaper and bigger than another student variant.
4. **PR32 teammate teachers are the only remaining +0.005-scale lever** (630M/
   1.54B × F0/F2, ~50% trained, 0/4 evaluated, no artifact channel). Value
   reaches the leaderboard only through distillation into ≤20 MB CPU students;
   the distillation packet should be frozen *before* the teachers land, and
   ~108k unreserved Kaggle GPU-seconds are exactly that budget.
5. **Public 0.700 is not reachable by 2026-11-15 through any measured path.**
   The team plateau is 0.6680–0.6683 across many compositions; the entire
   rank-22→rank-1 spread is 0.0041; the leader sits at 0.67243. Keep 0.700 as
   the standing research objective (ADR0009 — an allocation statement, not a
   kill switch), but do not let it drive endgame selection into all-upside bets.
   The only repo-evidenced 0.700-scale route is structural decoding of the
   high-|Y| target classes (class 3: ~43–48% of weight, WP 0.43 vs 0.74–0.81
   elsewhere; oracle branch readout 0.968) — a direction abandoned in September
   and, honestly, a post-competition-scale research arm.
6. **Runtime and verification are live final-selection risks.** Public replays
   already reached 3227/3600 s (one timeout on record; H390 pair callback is
   +24.1% vs TL3 → projected ~48–49 min); the private/final-set volume is
   unknown. Prize verification requires training code with fixed seeds within 7
   days of results — TL3's FTT ancestry gap is a **forfeiture risk** if TL3
   stays in the final set. Both need owners now (see action list).
7. **Marginal value of our (non-B300) effort has shifted from member science to
   assembly, grading, transfer testing and packaging.** Three weeks of admitted
   science produced +0.0001–0.0004 diagnostics; one day of zero-fit member
   assembly produced a +0.0037-scale exposed package gain. Recommended effort
   split [judgment]: ~55% assembly/grading/packaging, ~25% transfer testing and
   upload orchestration, ~15% PR32 distillation readiness, ~5% bounded riders.

## Elephant ranking (from the red-team report, condensed)

| Rank | Elephant | Severity | Decisive check |
|---|---|---|---|
| 1 | Recent +0.0003 results treated as established gains; below measured seed noise | Critical | One predeclared 2-seed third-cell replication of the i1 mixture before any further family spend |
| 2 | Optimizing the student slot while the FTT members hold the score | Critical | Ask teammate for FTT training push/lead-feature code; grade an FTT-feature student |
| 3 | Public-best anchor vs private-only final event; unuploaded candidates | Critical | Upload the two quota ZIPs with a pre-declared decision rule; verify portal pick-count mechanics |
| 4 | Runtime ceiling (60 min) with unknown final-set volume; 3227 s already observed | High | Self-imposed ≤45-min package gate; measure pair runtime before shortlisting |
| 5 | 0.700 anchor driving endgame risk posture | High | Treat as aspiration; endgame selects on expected placement |
| 6 | Adaptive selection on repeatedly exposed folds inflates the positive portfolio in aggregate | High | Publish a selected-after-looking comparison count; preregister next panels |
| 7 | Metric tail under-emphasis: class 3 holds ~43–48% of weight where our WP is 0.43 | Medium-high | Zero-fit weight-decile decomposition of pair-vs-TL3 gain (Q4) |

Explicitly **cleared** as non-elephants (with evidence, see report 02): metric
implementation (verified in `src/wnn/gmetric.py`), data semantics (officially
trading-instrument LOB data per `docs/data_overview.md` — "connectome" is
branding), ZIP size (20 MB limit; all packages 8.9–16.8 MB), Kaggle quota
scarcity, TL3-composition-as-total-mystery (members and weights are documented;
the hole is FTT ancestry only).

## Which current ideas mislead, and what to do instead (condensed)

- "i1 mixture is our strongest recent evidence" → call it a direction; one
  predeclared replication before further spend.
- "Quota ZIPs +0.0037 F0 ⇒ likely public gain" → expect ≈0 public change
  (student slot worth +0.0006 publicly; local deltas compressed 3–4× with one
  sign flip and one −0.026 collapse in transfer history); upload as a transfer
  *measurement* with a pre-declared rule.
- "The TL3 gap needs a better student" → the gap lives in FTT members we never
  built; recover FTT source instead.
- "0.700 via ensembling increments" → compositional plateau measured at
  0.6680–0.6683; 0.700 requires a new information source (tail structure), not
  more blending.
- "Screens are believed direction-only" is stated correctly (ADR0043) but
  practice still consumes +0.0003 directions as evidence — pair every promotion
  with a seed-level or preregistered check.
- Our transformer-teacher ladder duplicates teammate PR32; our 52-target
  dynamics head duplicates FTT's 52 lead features — become the consumer of both.

## The questions worth keeping in active memory (merged shortlist)

The three reports contain 30 questions total (20 in report 03, 5 each in 01/02).
The decision-changing core, in priority order:

1. What exact portfolio of previously scored ZIPs will we offer the private
   re-score — and are the two quota ZIPs uploaded before anything else? (Q1/03;
   needs portal pick-count verification by the user.)
2. What decision rule (written *before* uploading) does each possible public
   receipt of the pair/S3 trigger? (Q2/03.)
3. Where does official weight mass sit by |y| decile and target (t0 vs t1), and
   where does the pair's +0.0037 F0 gain live — mass rows or tail rows? Zero-fit
   on frozen arrays. (Q3/Q4/Q5/03; connects to elephant 7.)
4. What is the marginal local WP per minute of callback, and what member count
   fits a ≤45-min self-imposed gate under unknown final volume? (Q7/03.)
5. Is the PR32 evaluation+distillation consumer packet frozen now, before
   artifacts land? (Q8/03; action 3 below.)
6. Can we pass prize verification with the current asset mix (FTT provenance)?
   (Q2/01.) Who owns the packet?
7. What is the endgame objective — expected rank, P(top-8), or P(top-1)? The
   user should fix this before Nov 8; it changes final-set risk posture. (Q4/01.)
8. Are t0/t1 one signal or two (zero-fit cross-target analysis), and is training
   per-target balance right given the official score averages two separate
   correlations while `main_loss` pools both in one denominator? (Q5/Q9/03,
   verified in `src/wnn/gmetric.py` vs `src/wnn/training/losses.py`.)
9. Which anonymization-invariant LOB quantities (group sums/differences, spread
   proxy, imbalance) are already inside the 211 inputs, and does event-feature
   value concentrate in disequilibrium states? Zero-fit first. (Q10/Q11/03.)
10. How is team quota/upload scheduling coordinated with teammates near the
    deadline? (Q5/01.)

## New directions admitted by this review (cards, condensed)

From report 03 (full research-reframe cards with falsifiers there): a
closed-form RLS-style comparator bank as the online-family control (zero-fit);
state-dependent event×book-state interaction features (LOB Hawkes transfer);
normalization component decomposition (RevIN parts) extending the batch-mass
audit; a preregistered diversity-aware fixed blend-weight family; per-target
loss balancing (mass already measured even; signal balance open);
anonymization-invariant group-relative LOB encoding; a **non-recurrent diversity
member** (GBM/ridge on causal window summaries — our pool is 100% GRU-family;
M5 precedent); wildcards: submission quota as information design (incl. a
repeat-ZIP determinism check), per-sequence output standardization as a
serving-side transform (zero-fit first), cross-instrument delay estimation.

From the own-optimizer stream (report: [RESEARCH_PROPOSAL_R1.md](../../../experiment_drafts/own-optimizer-muon-20261024/RESEARCH_PROPOSAL_R1.md)):
C1 strict-FP32 Newton-Schulz + scaling audit (our `src/wnn/optim.py`
`orthogonalize()` **casts to bfloat16 on CUDA — every historical GPU Muon fit
violated the strict-FP32 policy**; the CPU branch is FP32); C2 per-group hybrid
scopes (hh-only/ih/per-gate vs rnn/hidden vs AdamW); C3 two-stage NS
coefficients; C4 batch-mass-corrected optimizer (the `loss_norm="running"` EMA
branch already exists untested in `losses.py`); C5 matched effective-rank
diagnostics on existing checkpoints (cheapest, zero-fit); C6 metric-aware update
shaping; C7 "Adamuon" second-moment blend on orthogonalized updates; C8
optimizer sandwich via existing `optim_switch`. The synthetic CPU screen running
tonight is the mechanism probe for C2/C3/C5-style questions; it is explicitly
NOT competition-quality evidence.

## Recommended immediate actions (ranked, zero-fit unless stated)

1. **Upload the two quota ZIPs (pair first, then S3) with a pre-declared
   decision rule** (e.g., "pair public ≥ TL3−0.0005 keeps the pair as
   finalist; otherwise revert to S3"). Two submissions; makes them
   private-selectable; records transfer evidence. No tuning from the receipt.
2. **Fix the CUDA bf16 Newton-Schulz cast to FP32 in `src/wnn/optim.py`**
   (~20-line compliance patch) and expose `ns_steps`/`rms` as CLI flags. This is
   policy compliance regardless of any quality effect.
3. **Run the C5 matched effective-rank diagnostic** on the D02 AdamW/Muon
   checkpoint pair (minutes, one core, ~600 MB): either supplies the mechanism
   for every Muon arm or removes the Parallax/effective-rank justification.
4. **Freeze the PR32 evaluation+distillation consumer packet** (code-only) with
   its cost model so Kaggle spend starts the day the teachers evaluate; assign a
   named owner for PR32 artifact collection (escalate by ~Nov 1 if the channel
   stays empty).
5. **Ask the teammate for the FTT/tiedlead training push and lead-feature code**
   (ADR0035 authorizes sharing): it is the score mass of TL3, the missing prize-
   verification provenance, and our duplicated 52-feature idea, all in one ask.
6. **Verify portal final-selection mechanics** (how many solutions may be
   chosen, when choosing opens) — a user action; set construction depends on it.
7. **Zero-fit weight-decile/tail decomposition** of the pair-vs-TL3 gain and of
   official weight mass (Q3/Q4) — decides whether the final assembly needs tail
   coverage members.
8. **Runtime budget engineering**: self-imposed ≤45-min gate; measure the pair's
   full replay; decide size-vs-member-count (quantization has negative history).
9. **If compute allows one bounded rider**: the C4 batch-mass CPU screen (paired
   batch-vs-running normalizer, 3 seeds) — the only optimizer-direction test that
   fits the local envelope and is unique to our setting.
10. **Write the final-set plan** (anchor TL3 + best local assembly + optional
    B300-derived package; TL3-family anchored per public roster) with the Nov 8 /
    Nov 13 / Nov 15 milestones from report 01.

## Reconciled disagreements

- **Report 01 parks all mechanism science past Nov 15; reports 02/03 propose new
  fits (FTT member, interaction features, C2 Kaggle panel).** Reconciliation:
  zero-fit diagnostics and *preparation* always pay; new fits only when they (a)
  feed the final set through the PR32/FTT path or (b) cost ≤ a few local CPU
  hours with a predeclared replication design. The C2 optimizer Kaggle panel is
  post-competition unless C4/C5 produce a surprising positive before ~Nov 5.
- **Report 03's Card 8 (GBM diversity member) vs report 01's member-count/runtime
  risk.** Reconciliation: run the zero-fit complementarity proxy first; a GBM
  member enters only through a distilled student or a proven native path —
  never as a raw serving risk in the final two weeks.
- **Report 02 treats the 0.700 objective as potentially misleading; the
  objective spec treats it as achievable.** No conflict after ADR0009 framing:
  0.700 stays as the research north star; endgame selection optimizes expected
  placement under the private-event rules.

## Limits and validity

All three reviewers share one harness and model family; convergence is
analytical, not statistical. Every repo number cited is a dated receipt read,
not a recomputation; public scores are user-verified snapshots (leaderboard
2026-10-17; roster 2026-10-22) and the leaderboard fetch failed (JS-only page).
All local strata are exposed; nothing here creates fresh validation. The
optimizer synthetic screen is a mechanism probe on synthetic data at tiny scale
(3 seeds/arm) — it cannot establish competition-quality gains and is labeled
accordingly in its own report. No fits on competition data, no uploads, no
ledger-free claims: new hypotheses from this review were appended to the
append-only ledgers with explicit zero-fit/pending status.

## Source records

- [01_value_allocation_R1.md](01_value_allocation_R1.md) — endgame mechanics,
  headroom buckets, portfolio dispositions, 10-action plan, milestones.
- [02_elephants_red_team_R1.md](02_elephants_red_team_R1.md) — noise-floor
  receipts, TL3 anatomy, final-event/runtime risks, misleading-ideas table.
- [03_questions_and_directions_R1.md](03_questions_and_directions_R1.md) — 20
  questions, 8 cards, 3 wildcards, full literature log.
- [Own-optimizer proposal](../../../experiment_drafts/own-optimizer-muon-20261024/RESEARCH_PROPOSAL_R1.md)
  and its [experiments folder](../../../experiment_drafts/own-optimizer-muon-20261024/experiments/)
  (synthetic screen script, grid logs, EXPERIMENT_R1.md).
- Prior audits preserved: [October23 ten-reviewer synthesis](../astra_ultra_audit_20261023_r1/SYNTHESIS.md)
  (mechanism/identification lens; this review is additive) and the [October24
  fundamentals report](../../../campaign_2026-10-24/resume4h_r1/astra_fundamentals_r1/FUNDAMENTALS_AND_MISSING_TESTS_R1.md).
