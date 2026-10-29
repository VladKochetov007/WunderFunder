---
kind: method
status: complete
created_utc: 2026-10-29T00:40:00Z
updated_utc: 2026-10-29T00:40:00Z
hypothesis_ids: [H-OPT25-MUON-FP32-R1, H-TABM29-GRU-R1, H-NORMUON29-R1]
experiment_ids: [E-TABM29-CEILING-R1]
evidence_paths:
  - research/campaign_2026-10-26/README.md
  - research/campaign_2026-10-26/diagnostics_r1/README.md
  - research/campaign_2026-10-29/tabm_gru_r1/README.md
  - research/unified/reviews/independent_value_review_20261024_r1/SYNTHESIS_R1.md
---

# Reflection: telling useful ideas from not-useful ones (October 29)

User request (paraphrased from Russian): reflect on what helps us tell useful ideas from less
useful ones, how this fits the skills, which changes are worth making, and which questions we
should be asking ourselves.

## 1. What the last four sessions actually taught us

| Episode | Cost | What a cheaper check would have told us |
|---|---|---|
| Strict-FP32 Muon: +0.0036/+0.0033 cold (three replications), **+0.00026** on the warm production student | ~10 local + 4 Kaggle fits, 2 production retrains | Zero-fit gate saturation of the production parents (10%, vs 45–48% for cold AdamW) predicted the warm null: the mechanism Muon fixes was absent in the artifact we ship. Measured *after* the transfer fit. |
| Muon student vs AdamW sibling prediction correlation 0.9995 (seed pair 0.994) | zero-fit | Prediction correlation with existing members is the cheapest blend-value bound; it should precede packaging. |
| Loss family (NLL/Pearson/Huber/KD) | several fits before the analysis | |y|-weighted MSE is Bayes-aligned with pooled WP, so distance changes cannot move the target. An analytic invariant would have closed the family at zero cost. |
| Class-3 up-weighting, i1 lead block, hand microstructure features | 8+ fits | Negative; information audits (AUC 1.0 quote arithmetic, held-i1 fingerprint) were the decisive cheap evidence. |
| TabM ensemble ceiling (today) | zero-fit | Mean of two Muon seeds +0.0011 over the mean single; four Muon siblings +0.0013/+0.0015 over the best single (F0/F2, intervals > 0). Any implicit ensemble of this recipe is bounded near +0.002 before transfer losses. |
| Stale 1-GiB slice cap from Oct 24 (today) | ~15 min idle, an OOM | A session-start check that live cgroup caps equal the latest user allocation. |

Independent value review (Oct 24) already put numbers on the frame: public seed noise
0.0007–0.001, TL3 student slot ≈ +0.0006 publicly, PR32-teacher consumption the only
+0.005-scale lever, public 0.700 not reachable by blending.

## 2. What distinguishes useful ideas here

An idea is useful in proportion to **expected change of the served endpoint per GPU-hour**,
which factors into terms we can usually estimate before fitting:

1. **Ceiling (C).** The best case computable from existing artifacts: oracle blends, seed-ensemble
   gain, per-class error budget, information audits. Today's TabM ceiling ≈ +0.002.
2. **Mechanism presence (P).** Does the deficit the idea fixes exist in the *production* artifact
   (saturation, dead units, calibration slope, class-3 error)? Muon: present cold, absent warm.
3. **Transfer multiplier (T).** Measured ratio of effect along teacher → parent → student →
   int8 → blend → public. Muon: 0.00026 / 0.0035 ≈ 0.07 through the warm stage. Student-slot →
   public ≈ 0.0006 per full student replacement.
4. **Detectability (D).** Ceiling relative to the panel standard error (≈0.0004 per fold, one seed)
   and to public noise (≈0.0008). Ideas with C·P·T < 0.0005 cannot be confirmed locally with
   two cells or publicly at all; they need a different justification (cost, robustness) or a
   larger panel.
5. **Decision linkage.** Which concrete action (package, retrain, close a family) changes on
   each outcome? If no outcome changes an action, the experiment is curiosity.
6. **Diversity value.** Prediction correlation with current blend members, which bounds blend
   gain independently of standalone score.

Expected value ≈ C × P(mechanism present) × T × P(works) − cost. Most of our negatives were
predictable from C, P or analytic invariants; most of our real wins (distillation, TL3
members, Muon cold) had a large C and a measured P.

## 3. How this fits the skills

- `conduct-creative-research` asks for falsifiers, cheapest discriminating tests and tiers.
  It does not ask for a **ceiling** or a **transfer path**; add them to the hypothesis card.
- `research-portfolio-control` allocates bandit-style but uses qualitative priors; the
  C/P/T/D terms give it a numeric, auditable prior without becoming a kill switch (ADR0009 still
  applies: an uncertain arm can run if information value is high).
- `research-reframe` generates distant ideas; the triage should apply *after* first commitment so
  ideation stays independent.
- `research-recipe-audit` / `research-evidence-review` stay as is; add a "mechanism present in
  production artifact" check to the evidence review before any transfer fit.
- `wnn-connectome-top1` already states noise and exposure; it should state the measured transfer
  multipliers so new agents do not re-derive them.

## 4. Changes worth making

1. Add a five-field **triage block** (ceiling and source, production-mechanism check, transfer
   path with measured multiplier, detectability vs SE, decision on each outcome) to the
   hypothesis card in `research-portfolio-control`. *Applied in this change (one short section).*
2. Order checks by cost: analytic invariant → zero-fit on existing predictions/checkpoints →
   2-cell cold screen → stage-matched transfer test → package. Never skip to the transfer fit
   when a zero-fit mechanism probe exists (Muon lesson).
3. Session-start resource check: compare live `wnn-campaign.slice` caps with the newest user
   allocation; fix before launching (operational lesson today).
4. Keep a small table of measured transfer multipliers in the frontier, updated when a new
   stage pair is measured.
5. Shrink always-loaded instructions: `AGENTS.md` carries many historical stop/resume
   sections; archiving superseded ones behind a link would reduce context cost. *Proposed only;
   it changes shared policy and needs the user's agreement.*

## 5. Questions to ask ourselves (and current answers)

1. **Where does the remaining error live?** Class 3 scores 0.39–0.44 against 0.73–0.80 for the
   others on the cold panel; any idea that is neutral on class 3 competes for a small residual.
2. **Which stage does the idea act on, and is that stage still trained in the final chain?**
   Optimizers act cold; the shipped student is warm-distilled. Cold-only gains require retraining
   the teacher chain.
3. **What is the best case from data we already have?** (TabM: ≈ +0.002; student slot: ≈ +0.0006
   public.)
4. **Is the effect detectable with the planned panel?** If not, enlarge it or stop pretending the
   screen decides.
5. **What would we do differently after each outcome?**
6. **Is the idea chosen for its ceiling or for its novelty?** TabM and NorMuon were chosen partly
   for interest; their honest ceilings are small. That is acceptable as bounded exploration
   (≈10–20% of compute), not as the main line.
7. **What is the largest unexploited lever?** Still teacher quality (PR32 / FTT source and lead
   features), not student optimization.
8. **What cheap measurement would falsify our favourite explanation?**

## Decision and next step

Apply the triage block to the portfolio skill; run the TabM/NorMuon screen as bounded
exploration (4 Kaggle + 2 local fits) with the ceiling stated up front; prioritize
teacher-side levers for the next allocation.

## Source records

[Oct 26 campaign](../../campaign_2026-10-26/README.md) ·
[gate saturation](../../campaign_2026-10-26/diagnostics_r1/README.md) ·
[TabM/NorMuon screen](../../campaign_2026-10-29/tabm_gru_r1/README.md) ·
[independent value review](../reviews/independent_value_review_20261024_r1/SYNTHESIS_R1.md) ·
[loss review](../../campaign_2026-10-25/loss_ideation_r1/README.md).
