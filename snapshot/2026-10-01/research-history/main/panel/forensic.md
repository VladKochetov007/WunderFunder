# Forensic report: research process post-mortem, E-060..E-144 (2026-09-26)

Scope: how the score moved from 0.5935 to 0.6600 on the LB, what cost time and returned nothing,
which beliefs were wrong, and what to change next. Sources: `research/experiments.jsonl`,
`reflection-2026-09-26.md`, `findings-2026-09-26.md`, `tree.md`, `lit/*.md`, `ideas/*.md`,
`artifacts/logs/` timestamps, git log. Tags: [ledger] = quoted from the ledger or docs;
[ours] = measured in this audit on cached arrays (read-only, `scripts/limited`, 3G, 10% CPU).
Status: DRAFT, AWAITING USER APPROVAL.

## 0. Headline
1. 85% of the LB gain came from reading the metric (baseline x0.25, E-061). Model work after that
   added +0.0097 in total, spread over five submissions.
2. The decision rule cannot tell arm effects from seed noise. With the recipe unchanged, seed 3 vs
   seed 4 passes `decide.py` in every cell [ours, C1]. Most "passing" arms were +0.001-0.0015, about
   the same size as one seed swap.
3. The "3x" fine-tune claim in E-144 comes from which proxy cell was used as the forecast. On the
   exact LB pair, plain lambda=0 predicted +0.0011 (folds) and +0.0012 +- 0.0004 (holdout); the LB
   gave +0.0014. The transfer was about 1:1, not 3x.
4. The stress panel was fit to 3 LB pairs, one of them a -1.4 sigma result (critic, agents doc).
   On the two pairs without retrain noise (blend, fine-tune) it was worse than plain valid.
5. All 51 GPU runs since 09-25 were GRU h128 variants. There was no other sequence family, because
   the training script offers only `gru|dual` and the native engine runs only GRU. Tooling set the
   search space, and no one decided that on purpose.
6. The biggest late lever (valid true-mask fine-tune, +0.0014 LB) was measured on 09-15 (E-045). It
   sat in a stale `frontier.md` "decisions pending" list until an Opus scout brought it back
   (E-098/E-124). Two LB submissions went out without it.

## 1. Attribution

### 1.1 LB gain by idea source (+0.0665 over the organizer baseline)
| source | lever (ledger) | LB | share |
|---|---|---|---|
| own metric anatomy | output scale x0.25 vs +-2 clip (E-060/061) | +0.0568 | 85% |
| own RNN program | 2 epochs, wMSE, train+valid (E-071/083/085) | +0.0045 | 7% |
| standard ensembling | 0.5 GRU + 0.5 baseline (E-084/086) | +0.0021 | 3% |
| own RNN | propensity floor eps 0.03 (E-097) | ~+0.0008 | 1.2% |
| haiku (H2-4) | next-target-change aux head (E-097/102) | ~+0.0004 | 0.6% |
| Opus independent M1 | true valid masks as weights (E-098) | ~+0.0003 | 0.5% |
| Opus shift / own | blend weight 0.65 (E-131/140) | ~+0.0001 | 0.2% |
| Opus M1 variant C (own E-045, 09-15) | 1-epoch valid true-mask fine-tune (E-124/141) | +0.0014 | 2.1% |

The E-144 bundle's +0.0017 is split in proportion to the valid arm sizes (floor 0.0020, aux 0.0010,
masks 0.0008, blend 0.0003). The split is approximate because arms are not additive and each
carries seed noise. The reflection groups E-088/089/093/112 as refuted haiku feature ideas. In fact
E-088/089 were our own heuristic library (B1), E-093 came from the 09-25 ideator, and E-112 from
opus-micro. Only E-097b, E-107 and E-130 trace to haiku files.

### 1.2 Ledger entries by source and outcome (88 rows: E-097 split into 4 arms)
| source | entries | passed/adopted | refuted | observation | GPU runs (~min) | hit rate of decision tests |
|---|---|---|---|---|---|---|
| own EDA / metric / diagnostics | 40 | 2 (061, 131 weak) | 10 | 27 (+1 corrective, 074) | 0 (CPU minutes) | 2/12; one hit worth +0.057 |
| own RNN A/B + finals | 19 | 12 (6 of 13 hypothesis arms) | 7 | 0 | 35 (~306) | 6/13 arms |
| haiku random literature | 3 | 1 (097b) | 1 (107) | 1 (130) | 2 (~24) | 1/2 tested; 1 of ~108 generated |
| Opus / LLM scouts (09-25 team + 5 docs) | 15 | 4 (084 rule, 098, 110 proxy, 124) | 9 | 1 (+1 corrective, 118) | 14 (~148) | 4/13 |
| literature validation scout (books) | 3 | method (chain folds) | 0 | 1 | 0 | 2 correctives (095, 096) |
| tooling / packaging | 5 | 5 | 0 | 0 | 0 | n/a |
| LB readouts | 3 | | | | | |

Refuted: own EDA 062 064 066 073 080 088 092 109 127 138; RNN 072 081 089 115 121 125 141; Opus 093
097d 103 104 112 116 119 120 123. GPU total: 51 runs, about 8 GPU-h. 29 of the 51 (57%) went to
refuted arms. Of about 158 LLM ideas generated (108 haiku, ~50 Opus), 18 were tested and 5 adopted.

### 1.3 Where time went without return (cost order)
1. Same-family knob arms after E-083 declared saturation (09-25 18:45): ab23 x4 (e3, seed4, h192,
   lr2e3), ab24_eps001, ab30_eps010, ab31 x2, ab32, ab36 x2. 11 GPU runs (~2.2 h), no adoption.
   This broke the skill's own plateau rule (SKILL.md §9: no more tuning until a
   representation-changing hypothesis has been tested).
2. Distillation line: E-097d passed, then ab28 x2, ab30_distill and ab34 were run. 4 GPU runs; the
   final showed no gain (E-116: 0.6914 vs 0.6916). A false positive of the seed-blind rule (§3 B1).
3. Oracle-law readout line: E-064..067, 072, 073, 074. Six CPU probes plus 3 GPU CE-head runs
   (ab16_ce, ab17 x2), all -0.016 or worse. The idea rested on oracle quantities (true spread,
   branch) that contain future information.
4. Heuristic/feature line: E-088, 089, 092, 093, 112. 1 GPU run plus GBM fits; zero transferable
   lift. The class-0 "gain" in E-092 was chain leakage (E-096).
5. Diagnostics with no decision: E-126..E-139 (14 CPU entries between 08:50 and 12:00). The GPU sat
   idle for about 3 h of the session (last run q10 at 08:50; next ab39 at 14:05). 12 of the 14
   entries were observations; the two tests (E-127, E-138) repeated a known refutation (§2 W10).
6. Haiku rounds: 9 files, ~108 ideas, triage time, 2 GPU runs; about +0.0004 LB.

## 2. Wrong beliefs and what they cost
| # | belief (where) | how it was shown wrong | cost |
|---|---|---|---|
| W1 | a `decide.py` pass means a real arm effect (E-084 rule) | [ours] C1: seed swap passes every cell | distill false positive (4 GPU runs); arms around +0.001 adopted without replication |
| W2 | test responds ~3x more to valid-like training (E-144) | [ours] C3: exact pair on holdout +0.0012 +- 0.0004 (lambda 0), +0.0008 (lambda .05); E-141 full-lambda-0 cells +0.0015/+0.0007; LB +0.0014 | none yet; at risk of overspending on "valid-like" arms because of a 3x multiplier |
| W3 | the stress panel is the calibrated LB proxy (E-110) | 3 constraints, 2 knobs, terciles picked post hoc (opus-shift §2); one constraint is -1.4 sigma noise; out of sample worse (§3 table) | forecasts off by 0.0013-0.0018; fine-tune forecast as +0.0005; eps 0.10 debate |
| W4 | dev halves / holdout are clean | E-095: 49% of devB and holdout have a devA chain neighbour | E-045 +0.0057 (clean: +0.0016); E-092 class 0; prop_v1 claims; holdout optimistic for finals |
| W5 | the target law holds in every class (E-064) | E-074: the check was vacuous (targets only change at book events) | oracle readout (E-066 +0.015) and CE heads: ~3 GPU runs, 6 probes |
| W6 | holdout ranks blends correctly (E-085, E-142) | blend -0.0008 on holdout vs +0.0021 on LB; finals train on neighbours of holdout, so the GRU scores in-distribution there | nearly under-weighted the baseline; overridden by LB history |
| W7 | selection is the binding misfit: M1 worth +0.003 valid and +0.002..0.006 LB (opus-independent S4) | measured: +0.0006-0.0010 (E-098) + 0.0014 LB (fine-tune); M2 and M3 refuted (E-103/104) | scouts ~2-3x optimistic (they used leaky E-045 and pre-floor baselines); M2/M3 cost 2 GPU runs + CPU |
| W8 | distillation adds +0.0013 (E-097) | E-102 (fold 0 negative), E-116 | 4 GPU runs |
| W9 | GRU seed blends add nothing (E-111, findings) | E-111 blended eps_aux with an e2-recipe seed (cross-recipe). [ours] C2: same-recipe 2-seed average +0.0009 alone, +0.0005 inside the baseline blend vs a random seed; ~0 vs the holdout-selected seed (C3) | a cheap variance reducer was dropped; latency allows a third member |
| W10 | segment-specific blend weights recover ~0.0005 (E-126) | already refuted by opus-independent P6 and opus-shift S11 before E-127/E-138 | 2 re-tests (CPU); dead-end memory failure |
| W11 | the dense-scored regime is the highest-value branch (E-139) | E-135: every model loses ~0.14 there, so it is a data property. Weight share shows importance, not headroom | 8 CPU entries, no test of headroom yet |
| W12 | test mask is softer (lambda 0.02-0.05, critic 09-25) | not identified (validation.md §2.2: class mix fits the LB level equally well); pair 5 lands at lambda ~0 | pushed arms toward mask-robustness cells; a bias against valid-mask-specialised arms |
| W13 | the 09-15 fine-tune result had been carried forward | frontier.md stale since 09-15; ab21 used p^2 on valid rows (opus-independent M1) | ~+0.0014-0.002 missing from 2 LB submissions (0.6548, 0.6569) |

## 3. Bias diagnosis

### Measurements made in this audit [ours]
- C1: `decide.py compare ab23_e2_seed4 ab20_gru128_e2` (same recipe, seed only). Full: lambda 0
  +0.0010 +- 0.0004, lambda .05 +0.0010 +- 0.0003. Stress: +0.0011 +- 0.0005, +0.0013 +- 0.0004.
  All 4 folds positive in every cell, `pass=True`. A seed effect is global, not fold-specific, so
  the same-sign-on-every-fold rule gives no protection against it.
- C2: 2-seed average of the e2 recipe vs each single, full valid lambda 0: +0.0014 / +0.0004.
  Inside the 0.7/0.3 baseline blend: +0.0009 / +0.0001. Same pattern in the stress cells.
- C3: holdout (368 sequences, 0.65 GRU + 0.35 baseline): s4 - s3 +0.0007 +- 0.0008; 2-seed average
  vs s4 +0.0001 +- 0.0004, vs s3 +0.0008 +- 0.0003; ft - s4 +0.0012 +- 0.0004 at lambda 0.

### Proxy calibration by pair type (valid proxy minus LB)
| LB pair | noise type | plain full, lambda 0 | stress, lambda .05 | LB |
|---|---|---|---|---|
| base - ab18 | two independent models, floorless GRU | -0.0029 (sign wrong) | +0.0017 | +0.0015 |
| final_tv - base | independent, floorless GRU | +0.0086 (0.0072 + 0.0014 OOF) | +0.0049 | +0.0045 |
| blend - final_tv | deterministic (same GRU) | +0.0017 | +0.0039 | +0.0021 |
| s4_base35 - blend | independent retrain (seed noise ~0.0007-0.001) | +0.0030 | +0.0033 | +0.0017 |
| s4ft - s4 | shared parent | +0.0011 (holdout +0.0012) | +0.0001 | +0.0014 |

My reading: stress was right only where a floorless p^2 GRU was compared with the baseline (pairs
1-2). Once the floor was in (pairs 4-5) and in the deterministic blend (pair 3), plain lambda 0 was
within 0.0004 whenever retrain noise was absent. The remaining error of about +-0.0015 on pair 4
matches the seed noise of independent final fits (C1, C3). The lambda knob most likely measured how
fragile floorless models are, not how soft the test mask is.
Alternative: the organizer baseline gets a test premium from training data closer to the test
(ideator I-7, never audited). But a premium would have lifted the pair-3 blend gain toward the
stress value, and it did not.

### Verdict on the suspected biases
- B1 Evidence standard is seed-blind (confirmed, C1). The ">2 SD, same sign" rule is loose for
  training noise, since a seed swap passes. It is strict for real small effects with few folds:
  E-141 failed, then gave +0.0014 on the LB. The critic's rule controlled sequence sampling and was
  used as if it controlled everything. The skill requires multiple seeds at tier B (§6); no arm had
  them.
- B2 The stress panel penalises valid-like training (confirmed for mask-specialised arms). Pair 5:
  stress +0.0001 vs LB +0.0014. It also drops a third of valid, which widens the SE and makes small
  real effects fail more often. The panel is still useful as a veto for floorless or baseline-share
  decisions.
- B3 Too many small A/B arms (confirmed). 51 GPU runs; the median passing effect (+0.0012) is about
  one seed swap. Arms were compared one at a time against one reference seed, so every "stack"
  inherited that reference's noise. The expected LB value per knob run after E-083 was about zero.
- B4 Too little architecture diversity (confirmed as a process fact, untested as a lever). Zero
  non-GRU sequence models; opus-arch A3/A5/A10 and tree 2b/2c/F1-F5 never ran. Root cause is the
  layer below: `train_rnn.py --arch {gru,dual}` and a GRU-only native engine, so any new family
  means a Rust port before it can ship. Caveat: the only diverse member, the organizer baseline, is
  itself a 2x128 GRU (residual corr 0.956; `baseline/solution.py`). The diversity we saw came from
  how that model was trained, not from its architecture.
- B5 Streetlight on cheap CPU diagnostics: 27 of 40 own-EDA entries were observations. The
  dense-regime chain (8 entries) never ran a headroom or oracle test. Importance was mistaken for
  tractability.
- B6 Scout magnitudes were optimistic: haiku promised +0.005-0.015 per idea, Opus-independent
  +0.002-0.006 LB. Opus-shift was the exception: it predicted LB 0.659 and got 0.6586.
  Predictions were not pre-registered (`hypotheses.jsonl` stops at H-046 on 09-25; only E-144 has
  a `predicted` field), so calibration by source cannot be scored.
- B7 Narratives from single points: "3x", "test closer to valid", and "stress orders all 3" each
  rest on 1-3 LB pairs with known noise of about 0.0004 (shared parent) to 0.001 (retrain).

## 4. Five process decisions, ranked

### D1. Replace the decision rule with a seed-aware, pair-type-aware standard
- Decision: run 3 seeds of the incumbent once (~36 GPU-min) to estimate sigma_seed of paired deltas.
  Accept an arm when Delta > 2 sqrt(SE_boot^2 + sigma_seed^2 / n_seeds_per_arm), with 2 seeds for
  every arm whose predicted Delta is below 0.002. Primary cell for recipe-internal changes: plain
  full valid, lambda 0. Keep stress/lambda as a veto only for changes to the baseline share or the
  weighting floor.
- Evidence: C1 (a seed swap passes every cell), W1, W8 (distill false positive), E-141 (false
  negative), and the proxy table (plain lambda 0 within 0.0004 on pairs without retrain noise).
- What would change my mind: the 3-seed ladder shows sigma of the paired delta below 0.0003, so
  single seeds are fine. Or 2 further LB pairs where stress beats plain on a recipe-internal
  change.

### D2. Move GPU from knob arms to diversity members, and build the deployment path first
- Decision: cap knob arms at 10% of GPU. Spend at least 40% on 2-3 members that change
  representation: a non-GRU family (TCN or attention over the last k events), a different target
  parameterisation (F and spread, tree 2c), or bid/ask mirror augmentation (F1). Gate each with one
  seed: residual corr below 0.975 with the incumbent (opus-arch D7 method) and a plain lambda-0 blend Delta
  above 2 sigma_seed. Only then port it to native. Budget: ~23 min of scorer headroom, about +6-8
  min per GRU-sized member.
- Evidence: B3/B4. 11 post-saturation knob runs returned nothing. Blending is the only lever that
  kept transferring (+0.0021 LB, and part of +0.0017). Own-recipe variants sit at residual corr
  0.99+, and a third member added +0.0002 (E-121).
- What would change my mind: two different families both land at residual corr above 0.985 with
  the incumbent. Then diversity is not in architecture; switch to data and target views (for
  example a member trained only on valid-like sequences).

### D3. Use the LB only for pre-registered, noise-controlled pairs (the user uploads and decides)
- Decision: each upload is a candidate that differs from the current best in one factor, with a
  shared parent or a deterministic transform (blend weight, fine-tune dose, seed average), so that
  seed noise cancels (~+-0.0004 instead of ~+-0.001). Before upload, log the predicted Delta from
  plain lambda 0 and from stress. Use at most 2-3 of the ~5 daily slots and keep the rest for
  confirmations. Accept a new best only if the LB Delta is above 0.0005 and the sign matches the
  forecast (ladder rule, validation.md G3). Best-informed next pairs, if the user agrees: baseline
  share 0.35 vs 0.45 on the same GRU (tests baseline premium vs floor fragility, §3); fine-tune dose
  of 1 vs 2 epochs from the same parent (tests the valid-like hypothesis with a dose response
  instead of a 3x claim).
- Evidence: pair 5 (shared parent) was the cleanest LB measurement so far; pair 4 (independent
  retrain) had error +-0.0015; C3 shows seed noise of finals at about 0.0007.
- What would change my mind: the public LB turns out to be a small slice (deltas below 0.001
  unreadable), so only candidates predicted at +0.002 or more are worth uploading. Or the user
  prefers pure candidate submissions.

### D4. Change the idea-source mix
- Decision: stop random-book haiku rounds; about 108 ideas yielded one adopted aux head worth about
  +0.0004 LB. Keep Opus scouts, but aim them at one named bottleneck each, require a measured probe
  (as opus-shift and opus-independent did), and require a pre-registered predicted Delta. Keep 15%
  of GPU for low-prior branches with explicit falsifiers (tree F1-F5 had 0%). Keep own metric and
  scorer analysis first in line (+0.057), but every diagnostic chain must end in a test within 3
  entries. For the dense regime: first an oracle-headroom test (true scoring density as an input,
  or a dense-only specialist); drop the branch if the oracle gains less than 0.002 on dense.
- Evidence: §1.1-1.2 hit rates; B5, B6; E-135 (the gap is a data property).
- What would change my mind: 2 of the next 4 tested haiku ideas pass D1. Or an untargeted Opus round
  produces a lever of +0.002 or more.

### D5. Fix research memory: hypothesis cards, dead-end index, live frontier
- Decision: every GPU arm gets a card in `hypotheses.jsonl` (predicted Delta and interval,
  falsifier, source) before launch. Keep a dead-end index keyed by mechanism so the next test
  checks it first (segment blend weights, per-class calibration, online normalisation, soups
  without a shared parent). Replace the stale `frontier.md` with the current best plus open
  decisions, and turn each resolved "pending decision" into an action the same day.
- Evidence: W10 (E-109/E-127/E-138 repeated P6/S10/S11), W13 (E-045 lever lost for about 10 days
  and 2 submissions), B6 (no pre-registered predictions to score).
- What would change my mind: the re-tests were deliberate reproductions. They were not labelled as
  such in the ledger.

## 5. Stop doing
- Single-seed arms with predicted Delta below 0.002. Knob sweeps on the GRU recipe (width, depth,
  epochs, lr, chunk, eps variants, aux horizons).
- Forecasting LB deltas from stress or lambda cells for arms trained on valid masks.
- Segment- or class-specific blend weights (refuted 3 times). Per-class offsets (refuted 3 times).
- Taking multipliers ("3x") or rankings ("orders all 3") from 1-3 LB pairs as established.
- Using the holdout to rank blends for finals trained on dev rows (it favours the GRU, W6).
- Random-literature idea generation not tied to a named bottleneck.

## 6. Decision points for the user
1. My diagnosis: E-144's surprise is a proxy-cell artefact (lambda 0 and holdout predicted
   +0.0011/+0.0012, LB +0.0014), not evidence that the test is closer to valid. The alternative
   (the test really is more valid-like) is consistent with the data too, but not needed to explain
   it. Which do you believe? The dose pair in D3 separates the two.
2. Attribution is complicated. The stress panel came from opus-shift, but we adopted it (E-110) on
   3 points against the scout's own warning ("stress cell, not an objective"). The fine-tune lever
   was ours (E-045), was blocked by a pending rules question, and was brought back by a scout.
3. Proposed fixes D1-D5 assume the seed-noise diagnosis. Confidence: high for C1 (measured),
   medium for "lambda measured floor fragility" (2 out-of-sample pairs), low for the
   baseline-premium alternative (never tested).
4. Please approve before these lessons go to the other panel members.

Monitoring to add: a standing 3-seed incumbent ladder (sigma_seed refreshed on each new recipe),
a forecast-vs-LB table per pair type, and a dead-end index checked before any new arm.
