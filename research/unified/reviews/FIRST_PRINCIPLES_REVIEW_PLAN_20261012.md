---
kind: review
status: proposed
created_utc: 2026-10-12T23:35:00Z
updated_utc: 2026-10-12T23:35:00Z
hypothesis_ids: [H-096, H-317, H-335, H-156, H-042, H-045]
experiment_ids: [E-061, E-079, E-083, E-094, E-095, E-104, E-149, E-154, E-164, E-168, E-181, E-188, E-191, E-230, E-264, E-302]
evidence_paths:
  - research/panel/forensic.md
  - research/panel/skeptic.md
  - hpo/VALIDATION_PROTOCOL.md
  - research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/review_public_gap_20261011/REVIEW_PUBLIC_GAP.md
  - research/campaign_2026-10-06/bandit_continuation/reviews/MAXIM_REPLICATION_REFLECTION.md
  - research/lb/registry.json
  - specs/current-state.md
---

# First-principles review of the Challenge 2 campaign: verdict and plan

This note is the approved plan for an independent review of the campaign summarized in the [unified research library](../README.md), the [October 12 archive](../archive/2026-10-12/README.md) and the [current-state log](../../../specs/current-state.md). It states the verdict and the deliverables that follow. The full review and the ranked idea cards will be separate notes, linked here when they land.

**Status:** proposal and critique only. No fits, predictions, uploads or provider actions were run; the review container has no data and no GPU. Historical ledger rows and frozen archives are not rewritten.

## Question

Why did roughly four weeks of research, with a large share of the effort after October 5, move the public score only from 0.6622 to 0.6671? What should change in the 34 days before the November 15 deadline?

## Core verdict

Each point cites the evidence it rests on.

### 1. There are two different projects in this repository

**September 15–28 was real, quantitative science.**
- The team read the metric: rescaling outputs ×0.25 so the ±2 clip stops biting was 85% of all leaderboard gain (E-061; [forensic post-mortem](../../panel/forensic.md)).
- It recovered the target law: t0 = F − ask, t1 = bid − F. The targets are piecewise constant and change only at i0 quote moves (E-302).
- It characterized the scoring mask: activity-driven, about 9.4% of rows, propensity AUC about 0.91–0.93.
- It found the stream structure: 3,288 chains (E-095).
- It wrote honest post-mortems ([forensic](../../panel/forensic.md), [skeptic](../../panel/skeptic.md)).

**From about October 6 it became a compliance machine.**
- 33 ADRs, a 258-line `AGENTS.md`, and a 645-line append-only telemetry log presented as the [current state](../../../specs/current-state.md).
- About 36,000 files under `research/` and about 320,000 lines of Python.
- 179 copies of `train_rnn.py` (178 frozen snapshots plus `main`). The main trainer cannot reproduce the production H390, Z4 or T3 members, because their flags exist only in snapshots.
- Kaggle R6–R9 all failed before any useful export:
  - R6: export exceeded the host budget.
  - R7: a literal `{python}` was left in the command.
  - R8: a required mount was missing.
  - R9: the startup mapping was wrong.
- Vast lanes cost roughly $50.

### 2. The scoreboard

Public scores by date ([registry](../../lb/registry.json), [archive submissions page](../archive/2026-10-12/pages/03-submissions.md)):

| Date | Public score |
|---|---|
| 09-27 | 0.6622 |
| 10-04 | 0.6660 |
| 10-10 | 0.6668 |
| 10-12 | 0.6671 |

- The most expensive week bought +0.0011.
- The teammate's simple GRU256 distilled pair scored 0.6674 on October 5 ([replication reflection](../../campaign_2026-10-06/bandit_continuation/reviews/MAXIM_REPLICATION_REFLECTION.md)). The whole October 6–12 campaign never beat it.
- On the October 9 leaderboard the top score was 0.6717, the top-10 cutoff was 0.6677, and we were about 19th.

### 3. The campaign ignored its own power analysis

[hpo/VALIDATION_PROTOCOL.md](../../../hpo/VALIDATION_PROTOCOL.md) (October 10) measured the noise:
- A one-fold, one-seed A/B comparison has sd 0.0022, so the minimum detectable effect is 0.0064.
- A +0.002 lever needs 4 folds × 3 seeds to detect.
- Its own conclusion: "No positive lever is distinguishable from noise."

Days later, single-fold, single-seed deltas of 0.00003–0.0008 were still ranked and reported as findings:
- F0 trio: +0.0000256.
- Old/new half blend: +0.000113.
- C vs B: +0.0003, while near-identical public uploads spread with sd about 0.0008 ([public-gap review](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/review_public_gap_20261011/REVIEW_PUBLIC_GAP.md)).

Fold 1 graded more than 200 models, and its local winner did not transfer: 0.67195 locally, 0.6616 public.

### 4. Effort went to the wrong levers for this problem

The problem is limited by variance and noise, not capacity:
- There are about 28 effective episodes per sequence ([physicist panel](../../panel/physicist.md)).
- Width and depth are flat (E-083).
- Longer training falls off a cliff: e5/e6 lose about 0.02.
- Averaging three seeds (+0.0019) beats every recipe tweak.

Effort nonetheless went to teacher, alpha, epoch, HFT and core-crop details:
- The 0.8125 teacher scale was carried over from a teammate's unnormalized sum of teacher weights.
- Teacher targets are in-sample on the rows the teachers were fitted on. The cross-fitted-teacher hypothesis (H-096) was never run.

EMA weight averaging gave E-154/E-164 +0.0037/+0.0033 offline and 0.6601 → 0.6618 public. It appears in only 22 of 350 recorded production attempts.

Two cheap levers were never run:
- **Cheaper ensemble members.** A one-layer GRU scored no worse than two layers (E-083: 0.6675 vs 0.6669/0.6672).
- **Bid/ask symmetry augmentation (H-317)**, which came from the user's own October 9 ideas.

Correction to an earlier draft: the MLP/conv input encoder *was* tested. It was positive on the 2-epoch recipe (E-168), then refuted under EMA (E-181) and as an ensemble partner (E-264). It is not an unexplored lever.

### 5. Governance disabled statistical discipline

- ADR0009 and ADR0001 were used to keep every arm "unresolved" and to withdraw practical noise thresholds ([public-gap review §11](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/review_public_gap_20261011/REVIEW_PUBLIC_GAP.md)).
- ADR0014 ("keep every lane busy"), combined with many admission gates, turned compute time into infrastructure churn.

### 6. The objective is mis-specified

**A public score above 0.700 is not credible.**
- The field's top score is about 0.672.
- Public scores sit about 0.03 below local holdout levels ([skeptic](../../panel/skeptic.md)), because the test set is heavier in class 3:
  - estimated public class-3 weight share ≈0.556 ([validation protocol](../../../hpo/VALIDATION_PROTOCOL.md));
  - validation share 0.477 (E-188).

**The real target is private rank.** The biggest private risk is runtime on a final dataset of undisclosed size. Candidate C uses 36.6 of the 60 minutes; with twice the rows it would take about 73 minutes and time out.

### 7. The October 12 strategy review repeats the pattern

The [October 12 strategy review](CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md):
- offers another long, hedged menu: 6 mechanisms, 4 refinements, 7 ladders and 9 papers;
- proposes a teacher width ladder up to 768 on a laptop GPU;
- has no calendar against November 15, no fix for throughput and no list of things to stop doing;
- misses the points in item 4.

Worth keeping from the whole campaign: the official scorer, component grouping of sequences, the native Q8 runtime, the honestly recorded negatives, and the early exploratory data analysis.

## Planned deliverables

### 1. Full review

`FIRST_PRINCIPLES_REVIEW_20261012.md` in this folder. Sections, in order:
- TL;DR.
- Scoreboard: public timeline and where the gains came from.
- Big-picture diagnosis.
- Technical critique:
  - evaluation and statistical power;
  - the loss and weighting stack, including the unreconciled finding that prop0 beat p²;
  - what the distillation setup actually does;
  - pipeline depth and reproducibility;
  - training dynamics;
  - serving and runtime;
  - how the leaderboard was used.
- Contradictions in the record:
  - E-079 (97.5%) versus E-302 and E-230;
  - stale "supported" status on H-042 and H-045;
  - the unsupported claim that valid is a later period;
  - the wrong claim that F2/F3 were held in reserve;
  - fold 1 has 457 sequences, not 468.
- What the October 12 review got right and wrong.
- A 34-day calendar with weekly gates.
- A proposed lean operating contract (proposal only, not applied).
- Questions for the organizers: final-dataset row count, time limit, number of final picks.

### 2. Ranked idea cards

`../ideas/FIRST_PRINCIPLES_IDEAS_20261012.md`.

| Priority | Idea | Notes |
|---|---|---|
| P0 | Evaluation reset | Pooled 4-fold out-of-fold score with at least 2 seeds; promote only effects above the minimum detectable effect; a cheap proxy calibrated against the 28-recipe F0/F1 bank; the teammate's composite veto (validation fold plus a train holdout with a propensity mask), which matched 5 of 5 public signs |
| P1 | More ensemble members per CPU-µs | One-layer or narrower GRU members against H256×2 at equal serving cost; restore EMA/SWA |
| P1 | Deterministic runtime safety valve | Count rows processed, not wall-clock time: after N sequences, switch to a cheaper sub-ensemble |
| P2 | Bid/ask mirror equivariance augmentation (H-317) | Run the CPU equivariance diagnostic on an existing model first |
| P2 | Cross-fitted teacher targets (H-096) | Only if distillation stays; also normalized vs 0.8125 target scale |
| P3 | Reconcile prop0 vs p² on 4 folds | — |
| P3 | Event-time / learned-decay memory | Transfer test of the earlier small-panel lead |

**To stop:**
- teacher width ladders;
- relational knowledge distillation;
- path signatures;
- the privileged lookahead teacher;
- Group-DRO;
- public-score regression;
- more rounds of external-LLM ideation.

**Already tested and negative:**
- class-mass reweighting (E-104);
- MLP/conv encoders (E-181, E-264).

### 3. Ledger rows

Proposed hypothesis rows appended to [hypotheses.jsonl](../../hypotheses.jsonl), so each card has a stable ID.

### 4. README links

The full review and idea cards get their own description and limits in the [reviews](README.md) and [ideas](../ideas/README.md) README files.

## Limits

- This is a static review of recorded evidence. Numbers were traced to the cited files but not recomputed from prediction arrays.
- Public scores are as transcribed by the user.
- The premise that the private set covers two to four future weeks comes from the user, not an official statement.
- Leaderboard ranks are from dated snapshots in the repository.

## Decision and next step

Write the full review and the idea cards on branch `claude/stoic-euler-70wff3`, then link them here and in the domain README files. No experiment is authorized by this note.
