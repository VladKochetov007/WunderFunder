---
kind: review
status: complete
created_utc: 2026-10-24T17:58:00Z
lens: value-allocation-and-endgame
author: independent subagent (fresh context)
---
# Value allocation and endgame review — 2026-10-24

One-shot, read-only, value-allocation lens. Zero fits, zero repo mutations, one attempted leaderboard fetch (failed: `https://wundernn.io/connectome/leaderboard` renders only a title client-side; no team names/scores retrievable — ranks below remain the dated 2026-10-17 user-verified snapshot in `specs/current-state.md`). "Measured" = repo receipts; "judgment" = my recommendation. This report is additive to the Oct 23 ten-reviewer synthesis (mechanism/science lens); it reallocates that portfolio under a 22-day endgame constraint.

## Main judgment (bullets)

- **The endgame mechanic dominates everything: final ranking re-scores explicitly chosen, previously submitted and scored solutions on a hidden dataset** (`docs/rules.md`: "you will be asked to choose some of your _previously submitted and scored solutions_"). An endpoint never uploaded is worth exactly zero for final placement; exact ties break toward the earlier submission. The two delivered-but-unuploaded quota ZIPs — our largest measured package-level local gain (+0.0037422 F0 vs TL3, `research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/RESULTS_AND_HANDOFF_R1.md`) — currently have zero placement value. Getting them uploaded is the highest-value zero-fit action available. [judgment]
- **The only remaining +0.005-scale arm is PR32 teacher consumption** (630M/1.54B × F0/F2, 50.37% of updates at 2026-10-24 13:40 UTC, 0/4 evaluated endpoints, no artifact channel yet — `research/campaign_2026-10-24/resume4h_r1/ALL_PR_RESEARCH_SUMMARY_R1.md`, `PR_REFRESH_INTERPRETATION_R3.md`). Its value reaches the leaderboard only through distillation into ≤20 MB CPU students; that conversion path (Kaggle ~108,000 GPU-s) must be prepared before the teachers land. [judgment]
- **Marginal value of our (non-B300) effort has shifted from member science to assembly, grading, transfer testing and packaging.** Three weeks of admitted science produced +0.0001–0.0004 diagnostics; one day of zero-fit member assembly produced a +0.0037-scale exposed package gain. No family testable on a 1-core/1-GiB laptop can decide placement.
- **Public history is a transfer prior we are underusing.** Measured portal roster (user-pasted 2026-10-22, `research/campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json`): every TL3-family upload scored 0.66771–0.66832 (TL3 0.66832, TL4 0.6683, xteam-H4 0.66808, two-parent 0.66771); every non-TL3-family upload scored 0.65839–0.66667. Local fitted-sample deltas compressed or flipped publicly (two-parent +0.00257 local → −0.00061 public; wide four-fold mean 0.684 → public 0.65839). The final set should be TL3-family assemblies. [judgment on the conclusion; roster is measured but not independently portal-verified]
- **Runtime margin is a live final-selection risk, already measured:** public replay took 2352 s (TL3), 3007 s (TL4), 3227 s (H512) of the 3600 s limit (`docs/submission_guide.md`; receipt above). The H390 pair's callback costs +24.1% vs TL3 (176.348 s vs 144.858 s on the 64-sequence replay, quota handoff) → projected ~2600–2900 s public, thin margin. This must be budget-measured before the pair is shortlisted. [projection is judgment]
- **Prize verification is a strategic risk:** top-8 must supply training code with fixed seeds plus a technical report within 7 days (`docs/prizes.md`). TL3's retained teachers have unresolved fitting ancestry (quota handoff). If TL3 stays in the final set, its provenance must be reconstructible or a win can be forfeited post-hoc.
- **Public 0.700 (+0.0317) is not reachable by Nov 15 through any identified path** — no measured mechanism exceeds +0.004. Keep it as the standing research objective (ADR0009: this is an allocation statement, not a kill switch); do not let it drive endgame selection into all-upside bets. [judgment]

## Gap decomposition and realistic headroom

Position (dated 2026-10-17, `specs/current-state.md`): us 0.66832, rank 22, leader 0.67243 → −0.00411 to top-1; +0.03168 to 0.700. Prize band: top-8 (`docs/prizes.md`). Roughly 21 rank steps span 0.00411 → ~0.0002/rank average near our position [judgment]. Transfer noise at package scale is comparable: observed public-minus-local discrepancies range −0.0006 to −0.026 (two-parent; wide). Any local delta ≲0.003 is inside observed transfer error.

| Bucket | Measured evidence (exposed/fitted scope unless noted) | Can it decide top-1? |
|---|---|---|
| +0.0001–0.0005 "tweaks" | i1 fixed 90B/10C mixture +0.00031549 two-cell mean (`specs/current-state.md` Oct 22 entry); fixed 0.1 shrink +0.00013 F0, flat F1 (`resume4h_r1/feasibility_scout/f1_mean_shrink_r1/RESULTS_R2.md`); checkpoint mean +0.000011; Kalman input +0.000116, CI spans zero (`research/unified/ideas/README.md`) | No. Five stacked successes ≈ +0.0015, below transfer noise; i1 is additionally unshippable today (367-input checkpoint fails the native layout gate, `ALL_PR_RESEARCH_SUMMARY_R1.md`) |
| +0.002–0.005 "assembly" | H390 S3 +0.00351 F0 / −0.00003 F2; S3/S4 pair +0.00374 F0 / +0.00021 F2 vs TL3 (quota handoff); PR27 innovation +0.000946 at 10% fidelity, ≈0 at 25% | Decides rank movement inside the 0.668–0.672 band; plausibly closes most — or none — of the −0.00411 gap depending on transfer. This is where our effort pays. |
| +0.005–0.02 "new strong members" | Nothing measured. PR32: 4 teachers at 50.37% updates, 0/4 evaluated, ~34.85 projected fit-hours, BF16-emulated (not strict-FP32 identity) | The only credible top-1 lever left; reaches placement only via distillation into compliant students. High variance. |
| +0.02–0.032 "structural" (→ 0.700) | Nothing on any measured ladder | No. Not achievable by Nov 15 from our assets. [judgment] |

Metric structure supports this: `global_wp_clip2_v1` (`src/wnn/gmetric.py`) is a weighted Pearson over ±2-clipped targets; the recent clipping/geometry/batch-mass audits found no cheap metric artifact left to harvest (`feasibility_scout/direction_probe_r1/RESULTS_R2.md`, `batch_mass_audit_r1/RESULTS_R1.md`). Headroom must come from better predictions, i.e., members.

## Portfolio audit: what can pay off before Nov 15

Disposition under current reality (1-core/1-GiB laptop, ~108k s Kaggle GPU, teammate B300s, human upload gate): **deliver** = plausible complete endpoint by Nov 15; **park/out** = cannot deliver in time or negative at exact scope — an allocation verdict, not a scientific kill.

| Family | Latest measured status | Endgame disposition |
|---|---|---|
| Teacher scaling (PR32) | 50.37% updates, 0/4 evaluated, no artifact destination | **Deliver (teammate-owned)** — the only +0.005-scale option; gating risks are evaluation and artifact access, not training |
| Distillation / student refresh (PR39, PR30; warm-KD recipe) | Warm-KD removal harms (−0.0143, evidence map in Oct 23 synthesis); PR39 8-fit design exists only on paper | **Deliver** — required converter from PR32 teachers to a ≤20 MB causal ZIP; this is what Kaggle GPU is for |
| Ensemble assembly / member replacement | H390 pair +0.0037 F0 / +0.0002 F2; runtime +24.1%; 16.2 MB of 20 MB | **Deliver** — cheapest measured large gain; needs transfer test + size/runtime budgeting |
| i1 history mixture | +0.00032 two-cell mean; one component CI crosses zero; not natively shippable | **Park as rider** — fold into final student-generation targets only if it rides existing student fits [judgment] |
| Online relationship heads | All six F0 heads below parent (~−0.0005); F1 shrink flat; calibration <0.1% of score slope (`feasibility_scout/RESULTS_R2.md`, direction probe) | **Park** — negative at exact tested scope, no laptop capacity to re-run; saved arrays remain for post-competition analysis |
| Age clocks (per-field/group) | Teammate shortscreen −0.00035 (10%) / −0.000085 (25%); faithful panel ≈6.18 serial B300-h (`ALL_PR_RESEARCH_SUMMARY_R1.md`) | **Out** — slightly negative at both fidelities; the decisive panel doesn't fit our compute |
| Innovation vs moments | +0.00095 at 10%, ≈0 at 25% (24 fits) | **Park; flag to teammates** — unresolved reversal, but resolution is B300-scale, not ours |
| Kalman input features | +0.000116, CI spans zero | **Park** — inconclusive; no capacity |
| Geometry/discrete | Partial matrix below baseline; full R4 est. 113.8 GPU-h, stopped | **Out** — structurally undeliverable in time |
| NLL/variance objectives | No advantage; PR37 paused at epoch-0 states | **Out** |
| Checkpoint averaging | +0.000011, already graded (Oct 22 entry) | **Out** — duplicate task per Oct 23 synthesis |
| Width/depth | Wide +0.00078 partial (PR25); already consumed in H512/H390 usage | **Out** — no new action |
| Initial-state family, TBPTT, superbatch WP replay, mirror symmetry, Mamba/Transformer handoffs, Chronos transfer, meta-learned adaptation, group-masked pretraining, predictive-state self-supervision | Code-only/proposal-only; zero fits, zero artifacts (PR table, `ALL_PR_RESEARCH_SUMMARY_R1.md`) | **Out** — none can produce a complete evaluated endpoint in 22 days under current constraints |
| Inference-time skipping / smoothing | Documented negative or unvalidated; deprioritized by Oct 23 synthesis | **Out** |

Additive claim versus the Oct 23 synthesis: its priority-1–10 discriminating screens (Z/A control, ordinary-diversity-vs-history, response-speed probe, etc.) remain good science, but **under the endgame constraint their best case is a +0.0005-scale member tweak and their worst case is spend with no deliverable** — they no longer change any placement-relevant decision and should be explicitly deferred past Nov 15. [judgment]

## Underused assets

1. **The two delivered, unuploaded quota ZIPs** (`submissions/quota_20261024/`; byte hashes in the quota handoff). The only ready instrument that tests assembly-gain transfer; uploading also makes them *selectable* for private re-scoring. Cost: 2 submissions of ~90 remaining (5/day per `docs/rules.md`; team total 102 as of 2026-10-22 per `PUBLIC_RECEIPT_R1.json`).
2. **Kaggle ~108,000 unreserved GPU-seconds (~30 h)** — largest uncommitted controllable resource (verified 2026-10-24 10:58 UTC, `resume4h_r1/FRONTIER.md`). Reserve for teacher-cache builds + 2–4 student distillation fits + exports once PR32 lands; not for mechanism screens. Historical Kaggle failures were in the age-profile delivery path (checksum/membership/timeout, `kaggle_age_r3/README.md`, `kaggle_age_r4/`), not student fitting per se.
3. **PR32 teachers when they finish.** At the observed comment cadence (17,773 → 21,745 → 23,725 updates across Oct 24 refreshes) the remaining 23,375 updates plausibly complete within days, but the 34.85 fit-hour projection, BF16 caveat and absent artifact channel make ETA and *access* the real risks; someone must own collection. [ETA is judgment]
4. **TL3 as an object of study.** TL3 beats every replacement graded locally (H512 −0.0074, B300 CP5 −0.0033 exposed; `specs/current-state.md` Oct 22 entries) and its ancestry is unknown. Two separate asks: (a) *mechanism* — why it wins — is the cheapest big scientific unknown, but is post-competition work; (b) *verification provenance* — training code + seeds for TL5/TL6 — is endgame-critical and must be resolved before final selection. Buy only (b) before Nov 15. [judgment]
5. **The 102-submission history as a transfer-calibration set.** Eight public points (receipt above): TL3-family 0.66771–0.66832; non-TL3 0.65839–0.66667; runtimes 1711–3227 s; memory 626–672 MB (no memory risk). Implications: family-stable gains transfer better than novel replacements; local four-fold means can mislead (0.684 → 0.65839); expected public delta for the H390 pair is plausibly small-positive, not +0.0037. [judgment, n small]
6. **H390 S3/S4 students and saved parent caches.** Already exploited in the ZIPs; caches reusable for grading/head work at ~400 MB per bounded run under the 1-GiB cap.
7. **Teammate shortscreen results (PR26/27).** Information, not assets: they calibrate expectations for B300-derived members (low-fidelity positives may vanish at full data) and argue against re-testing age on Kaggle.

## Science vs delivery: where our marginal value is

Measured contrast this week: six scientific diagnostics ≈ +0.0001–0.0004 each (intervals crossing zero); one assembly action = +0.0037 (F0 stratum, fitted-sample). Member generation is teammate-owned (B300); member *selection*, *assembly* and *packaging* are ours, cheap and now decisive. Packaging is also the only place hard constraints (20 MB, 60 min, determinism) can still destroy value late — public runtimes already reach 3227/3600 s.

Recommended split of our (non-B300) effort for 22 days [judgment]:

- **~55% assembly + grading + packaging:** fixed ensemble search over {TL3 members, H390 S3/S4, distilled students}, full-strata grading, size/runtime budget engineering, verification packet.
- **~25% transfer testing and submission orchestration:** upload the two ZIPs, record receipts honestly, keep the selectable pool aligned with local evidence.
- **~15% delivery readiness for PR32 output:** distillation packet prepared and smoke-checked offline *now* so Kaggle spend starts the day teachers evaluate.
- **~5% bounded science rider:** i1-mixture targets inside the final student fits only; no standalone screens.

## Endgame plan and submission-quota strategy

**Mechanics (from `docs/rules.md`, `docs/submission_guide.md`):** final ranking solely private, from explicitly chosen previously-submitted-and-scored solutions; 5 submissions/day; tie-break to earlier submission; ZIP ≤20 MB; 60-min CPU-only replay; deterministic; Python 3.11 offline. Consequences: (a) anything not uploaded is unselectable; (b) upload champions early, not on the last day.

**Quota arithmetic [judgment]:** 22 days × 5/day ≈ 110 theoretical remaining (102 used as of Oct 22); binding scarcity is information value and human-gate bandwidth. Recommended spend: 2 now (quota ZIPs), 2–4 in the Nov 8–13 window (final candidates incl. any distilled-student package), ≥5 held for the final 48 h contingency. No near-duplicate variants, no coefficient sweeps (public-guided tuning prohibited by standing policy).

**Final set construction [judgment]:** 2–3 selections hedging the hidden-dataset risk: (1) proven anchor — TL3, already submitted at 0.66832 and selectable; (2) best locally-supported assembly — H390 pair, pending its public receipt; (3) a B300-derived package only if it beats existing members on complete strata grading before Nov 13. The public roster argues the whole set stays TL3-family.

**Milestones — what must be TRUE:**

- **By Nov 8:** PR32 has ≥1 officially evaluated teacher endpoint and a working artifact channel (escalate ownership if still absent); distillation packet smoke-checked offline end-to-end (cache → student fit CLI → native export); both quota ZIPs uploaded with public receipts recorded; portal final-selection mechanics (pick count, UI timing) verified by the user; TL3/teacher training-code provenance located or the verification gap explicitly accepted.
- **By Nov 13:** final member set frozen with complete F0/F1/F2 strata grading; packages pass 20-MB size, ≤60-min runtime with margin (pair projected ~2600–2900 s; measure, don't assume), determinism/reset checks; champion uploads done (tie-break favors earlier); verification packet (training code + seeds + report draft) assembled.
- **By Nov 15 23:59 UTC:** every wanted candidate already submitted and scored; explicit final selection made in the portal; package freeze after the last full replay check — a broken final submission can lose everything; an unchanged 0.668-class package cannot.

**Risk register:** runtime creep (+24% pair callback; public 3227 s already observed at 90% of limit); 20-MB ceiling (pair 16.2 MB — more members likely force quantization, which has negative history: decide size-vs-members early); PR32 artifact access/ETA; human-gate availability near deadline; overreacting to noisy public receipts (transfer evidence only, no tuning); BF16 caveat on PR32 quality claims; i1/innovation gains concentrated on subsets may not survive full data (PR26/27 precedent).

## Sequenced action plan (≤10 actions)

| # | Action | Why now | Cost | Decision it changes | Main risk |
|---|---|---|---|---|---|
| 1 | User uploads H390 **pair** ZIP, then **S3** | Only test of assembly transfer; makes them selectable; anchor ordering | 2 submissions; 0 fits | Whether H390 replacement is in the final set; calibrates all later local→public reads | Public noise misread — record receipt, no tuning |
| 2 | Teammate: evaluate PR32's 4 teacher endpoints on the official metric as they land; own artifact collection | 0/4 evaluated is the biggest unknown on the only +0.005 ladder | 0 our fits | Whether teachers enter distillation | Artifact channel empty; BF16 identity caveat |
| 3 | Prepare + offline-smoke-check the distillation packet (teacher-cache builder, student fit CLI, native export path) before teachers finish | Converts Kaggle GPU from "available" to "spendable" the day teachers evaluate | 0 fits; timebox 2 days | Kaggle spend shape and start date | Prep-without-admission loop |
| 4 | Run 2–4 student distillation fits on Kaggle GPU from the best-evaluated teacher (optional i1-mixture target rider) | The only route to a new member under our control | ~50–90k GPU-s of 108k | Whether a new member joins the final ensemble | Kaggle delivery failures; use validated packet paths only |
| 5 | Fixed ensemble search + full-strata grading over {TL3 members, H390 S3/S4, new students} | Selection evidence for the final set | 0 fits; grades ≈ 4–90 s each under current cap | Final 2–3 package composition | All strata exposed; avoid over-reading |
| 6 | Package engineering: size (≤20 MB) and runtime (≤60 min, measured margin incl. pair projection) budget; quantization fallback decision | Public runtimes already at 90% of limit; 16.2/20 MB used | 0 fits; bounded replays | Maximum member count of final ensemble | Quantization quality loss; ISA variance |
| 7 | Resolve TL3/TL5/TL6 training provenance; assemble verification packet (training code, seeds, report draft) | Prize verification is 7 days post-results; unresolved ancestry is forfeiture risk (`docs/prizes.md`) | 0 fits | Whether TL3 stays in final set; verification pass/fail | Teammate memory loss — ask early |
| 8 | Verify portal final-selection mechanics (pick count, UI timing) with the user | Rules say "some solutions"; count unknown from docs | 0 | Final-set size and upload schedule | Assuming Kaggle-style 2 picks wrongly |
| 9 | Upload final candidates Nov 8–13, champion first | Selectability + earlier-submission tie-break | 2–4 submissions | Selectable pool | Last-day rush; human-gate unavailability |
| 10 | Explicit final selection in portal by Nov 14–15 after final full replay check; freeze all packages Nov 13 | Unselected/unsubmitted work is worth zero | 0 | Final placement | Late-change breakage — freeze discipline |

Critical path: 2→3→4→5→6→9→10. Actions 1, 7, 8 are independent — start immediately.

## What NOT to do

- **No new mechanism screens on the laptop** (age F−G, online-head recovery, Z/A control, response-speed probe, batch-normalization arm): none can change final placement; the Oct 23 priority list is post-competition material under this deadline. [judgment]
- **No Kaggle GPU spend on screens or profiles** — it is the distillation budget.
- **No public-guided coefficient or member tuning**; receipts are transfer evidence only.
- **No re-grading of already-graded items** (checkpoint mean, old geometry cells) — duplicate-task removal confirmed by the Oct 23 synthesis.
- **No final-set bet on un-uploaded packages**; no all-upside selection that drops the 0.668 anchor in pursuit of 0.700.
- **No package mutation after the final replay check**; no new libraries/dependencies in the final ZIP.
- **Do not let PR32 arrive into a dead channel** — if no artifact destination exists by ~Nov 1, escalate collection ownership.
- **Do not treat stale statuses as current** — PR states here are as of 2026-10-24 ~13:40 UTC; the leaderboard position is as of 2026-10-17.

## Five questions we should be asking

1. **Exactly how does final selection work in the portal** — how many solutions may be chosen, when does choosing open, and can choices be revised until the deadline? Set construction (anchor + upside hedging) depends on this number; it is not in `docs/rules.md`.
2. **Can we pass prize verification with the current asset mix?** If TL3's teachers cannot be reproduced with training code and fixed seeds, does any TL3-family final set survive the 7-day verification window — and who owns assembling the packet?
3. **What is our standing local→public transfer calibration, and is one more deliberate probe worth it?** Observed compression ~3–4× with one sign flip and one −0.026 collapse (wide); should S3-vs-pair on separate days be spent to size the final-set hedge?
4. **What is the endgame objective function — expected rank, P(top-8), or P(top-1)?** The 0.700 aspiration and the rank-22 reality imply different final-selection risk postures; the user should fix this before Nov 8. [user decision]
5. **How are team quota and upload scheduling coordinated with teammates?** 5/day is one team resource (102 already used); near Nov 13–15, teammate uploads, our candidates and the human gate could collide — an agreed calendar costs nothing now and prevents a missed champion upload.

## Sources inspected

- `specs/research-objective.md`; `specs/current-state.md` (incl. Oct 17 leaderboard snapshot, public receipts)
- `research/campaign_2026-10-24/resume4h_r1/`: `CAMPAIGN_SYNTHESIS_R1.md`, `FRONTIER.md`, `ALL_PR_RESEARCH_SUMMARY_R1.md`, `QUOTA_SUBMISSION_AUTHORITY_R1.json`, `quota_submission_r1/RESULTS_AND_HANDOFF_R1.md`, `astra_fundamentals_r1/FUNDAMENTALS_AND_MISSING_TESTS_R1.md`, `PR_REFRESH_INTERPRETATION_R3.md` (via synthesis/frontier)
- `research/campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json`
- `research/unified/reviews/astra_ultra_audit_20261023_r1/SYNTHESIS.md`
- `research/unified/ideas/README.md`
- `docs/rules.md`, `docs/submission_guide.md`, `docs/timeline.md`, `docs/prizes.md`, `docs/faq.md`
- `submissions/` inventory (incl. `submissions/quota_20261024/`)
- Leaderboard fetch attempt at `https://wundernn.io/connectome/leaderboard` (2026-10-24): JS-rendered, no scores visible — reported as a failed observation.
