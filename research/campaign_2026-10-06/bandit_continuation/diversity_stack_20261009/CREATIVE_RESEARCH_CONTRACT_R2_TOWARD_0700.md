# Creative research contract R2: toward public 0.700 (2026-10-09/10)

Controller: Claude Code root. Skill: `conduct-creative-research` (invoked by the user on 2026-10-09). Companions:
[research plan](RESEARCH_PLAN_NIGHT_20261009_R1.md), [ideas note](TEACHER_UPPER_BOUND_AND_GENERATIVE_IDEAS_R1.md),
[blend theory](MECHANISTIC_INTERPRETABILITY_AND_BLEND_THEORY_R1.md), [optimizers and schedules](OPTIMIZERS_SCHEDULES_STABILITY_R1.md).

## 1. Contract

| Field | Value |
|---|---|
| Question | What training, data, loss and architecture changes (including scale-then-distill) raise the **unseen** weighted-Pearson score of a serving-size model (GRU256-class, 1 vCPU, 60 min, ZIP < 20 MB), toward public 0.700? |
| Objective mode | **competition**: expected unseen-score gain under serving and rule constraints. Novel or high-risk lineages are explicit arms, not required of every candidate. |
| Metric | `global_wp_clip2_v1` (weighted Pearson, weights |clip(y,2)|, scored rows >= 99). Never altered. |
| Baseline / best known | Public: own best 0.6662 (Z4, Z3); teammate 0.6674 (distilled h256 pair); leader 0.6717; top-10 cut 0.6677. Local fold1 best single student 0.66913, fold0 0.68172. |
| Verification tier | **B/C mix**: local folds are historically exposed (no untouched holdout); the public board is one noisy, small-sample observation (seed noise about 0.0005 public, 0.001-0.0015 local under Muon). Claims need >= 2 seeds per arm, two folds, and a transfer check; public is confirmation, not selection. Model-to-reality gap: test has a different class mix (class 3 about 0.55 of weight vs 0.37 on valid) and the transfer loss is concentrated in class 3 (teammate probes). |
| Invariants | Causal, deterministic inference with per-sequence state reset; independent training (no teammate weights; teammate code is read for ideas only); only supplied data; no use of test features for training; ZIP < 20 MB and runtime < 60 min on 1 vCPU (a single GRU256 student with head finishing is already about 39 min). |
| Falsifiers | An arm is not supported if its two-seed mean gain is below the seed noise on fold1 **and** it does not transfer in sign to fold0. A blend gain that rests on a partner much weaker than the student (corr rule `c < r_j/r_i` fails or gap > 0.004) is not deployable evidence. |
| Budget | Two Vast lanes (about $1.76/h together, accounts 1 and 2; account 2 has $25 credit with auto top-up), local GPU (5 GB), Kaggle GPU quota exhausted until reset. Queue holds new jobs gracefully at $12 (A) and $10 (B); running jobs always finish. |
| Stop / plateau | Pause an arm after two seeds on two folds without gain; reframe when three diverse families plateau; abort if the evaluator or leakage guard breaks. |
| Human gates | Public submissions are uploaded by the user (5 per day); money beyond the stated credit; any use of unlabeled test features. |

## 2. Evidence ledger from the teammate branch (read-only; ideas, not code)

From `research/iterations/2026-10-02-slow-fast/README.md` on `origin/research/creative-iteration-20260930` (all small-model h128/d256 unless noted).
Treat as strong priors, replicate before trusting.

| Finding | Evidence | Implication here |
|---|---|---|
| **Muon with a large LR multiplier** (x6 relative to AdamW LR, rms-matched; x12 diverges) | h128: +0.0013..0.0022 over AdamW, d256 x6 .6921 vs .6906; true gain about +0.001 after seed calibration | We run Muon at multiplier 1. New flag `--muon-lr-mult` added; scan x3/x6 on GRU256/384 |
| **Stopped bootstrap h8** | +0.0010 mean on d256; long horizon (32) hurts | Our combo recipe has it; the Vast scratch teachers do not (needs a distillation teacher) |
| **EMA-evaluated weights** | combo recipe uses EMA teacher plus eval-ema; Muon EMA .9995 variant hurt (-0.003) | Test the combo setting (decay 0.999) on scratch teachers |
| **Class 3 is data-limited, not capacity-limited** | half the train data: class-3 -0.009; class-3-only expert: -0.14; exact extra class-3 data: +0.003 on class 3 | Specialists are a dead end (removed our class-3-only job); more *real* data or better priors is the lever |
| **Synthetic-label augmentation hurts** | instrument swap, rebuilt labels (90% exact), noisy-student pseudo-labels all lost | Generative label synthesis is low prior; consistent with our assessment |
| **Transfer gap lives in class 3** | public probes: classes 0-2 lose 0.005, class 3 loses 0.024; test class-3 weight share about 0.55 | Evaluate candidates also with a class-3-heavy weighting; class-3 within-sequence ranking is the target |
| Organizer baseline blended at effective coefficient about 0.06 | public +0.0004..+0.0021 across packages; train-holdout +0.002..+0.003 | Only supplied-baseline lever with public evidence; check provenance rules and packaging |
| Raw-only input view is the main diversity source for blends | +0.0005..0.0012 as 0.2 blend member | Queued as `rawonly` teacher |
| Dropout 0.1/0.2, many HPs, larger student widths (h512 < h256), small Transformer, valid bagging, rank transforms, class-gated stacking, far-horizon aux, global readout refit | all null or negative | Dropout and TCN arms in our queue are expected null; keep one probe each |
| Context distillation (predecessor-chain state) | +0.0003 mean, +0.003 on early rows | Marginal; not prioritized |
| True mirror symmetry | block-swap is not a mirror; code graph has no exact automorphism (84 vs 75 edges); deferred | Open high-risk item |

## 3. Hypothesis cards (committed before results)

Ranking = expected unseen-score gain x probability / cost. Mechanism family in brackets. Each has a cheapest test and a falsifier.

| ID | Lineage | Hypothesis | Prior | Cheapest test | Falsifier | Status |
|---|---|---|---|---|---|---|
| R2-1 | recipe fidelity [optimizer] | Muon LR multiplier 3-6 gives +0.001 on GRU256/384 | medium-high (teammate, 3 seeds) | `vt_f1_mu3`, `mu6` (seeds 3,4), fold0 `vt_f0_mu6`, GRU384 `mu3` | two-seed mean not above multiplier-1 baseline by more than 0.001 | queued |
| R2-2 | recipe fidelity [weights] | EMA-evaluated weights lift scratch teachers (they are part of the combo recipe) | medium | `vt_f1_ema`, `mu3_ema` | no gain over the matched control | queued |
| R2-3 | scale-then-distill [scale] | A stronger/wider teacher improves the serving student more than the teacher-student gap alone predicts | unknown (teammate: wider *students* lose) | Vast pipeline: teacher T in {GRU256, 384, 448} -> train predictions -> 1-teacher cache -> matched H230 student | student(T384) <= student(T256) within noise | **build** |
| R2-4 | scale [regularize] | Large models (512/768) fail from under-regularization or LR mismatch, not capacity; with lower multiplier, EMA and more epochs they beat 384 | low-medium | `gru512` with multiplier 0.5-1.5, EMA, 8 epochs (after R2-1 shows the sign) | 512 never reaches 384 under any of three LR/EMA settings | queued (part) |
| R2-5 | blend theory [diversity] | Pair rule + decorrelation selects partners that beat score-ranked partners on a second fold | validated in-fold | rank candidates by closed form on folds 1,2,3 | rule fails to predict gain on a new fold | running (folds 2,3) |
| R2-6 | baseline [external prior] | Blending the supplied organizer baseline at effective coefficient about 0.06 raises public by 0.0005-0.002 | medium (public evidence from teammate) | package Z4 + baseline; check runtime and rules | runtime > 55 min or rules forbid | **investigate** |
| R2-7 | structure [architecture] | A diagonal selective state-space (parallel-scan) teacher trains on the full 20,000-row sequence without truncation, is decorrelated from GRUs, and scales to larger widths | low-medium, high novelty | small SSM in the trainer behind `--arch`; compare corr and blend gain vs GRU384 | no decorrelation beyond TCN and no standalone competitiveness | idea (high-risk) |
| R2-8 | data [symmetry] | An exact bid/ask mirror (decoded event codes, t0<->t1 swap, exact labels) doubles class-3 data | low (prior attempts failed approximate) | decode code tables; verify equivariance on trained models | no automorphism within 5% edge mismatch | idea (high-risk) |
| R2-9 | information [context] | Class-3 ranking limits come from missing context; chain-predecessor warm start or longer BPTT improves class 3 | low-medium | warm-start state from predecessor in a teacher; evaluate class 3 only | gain only on early rows (already seen +0.003) | idea |
| R2-10 | objective [metric] | Optimize a metric-aligned, class-3-heavy surrogate (global moment matching, class-share-corrected weights) | low (many loss variants null) | `wmse_gcorr` done (0.66625, null); class-mix reweight seen -0.0011 | no change vs control across 2 seeds | closed-ish |
| R2-11 | distribution [loss] | Predict a distribution (mixture or quantiles) and use its mean/sign structure; direction t0-t1 is the crux per teammate | low-medium, novel | add a sign-classification auxiliary or mixture head to the teacher | no standalone gain and no blend gain | idea |

## 4. Attack before compute (critic pass, done by the controller; independent critic pending)

- R2-1: divergence risk at x12 for h128, x8 for d256 (teammate); GRU384 may diverge earlier. Start at x3, only then x6. Single-seed screens are not evidence (sd 0.0012-0.0015 under Muon): two seeds per arm.
- R2-3: confounds: student recipe (parent init, epochs, batch/chunk) must match across teachers; the student inherits the parent's lineage, so the teacher effect may be small. Control: teacher = GRU256 scratch.
- R2-5/R2-6: composite validation overstates blend gains (teacher blends rose +0.0008 locally and vanished publicly).
- Anything using the baseline needs a recorded provenance and rule check; the local evaluation of baseline-containing models is contaminated if it saw valid data.
- Public score differences below 0.0005 are noise.

## 5. Execution plan (this round)

1. Queue recipe-fidelity arms (R2-1, R2-2) on both folds, interleaved with the existing mechanism-diverse queue.
2. Build the scale-then-distill pipeline (R2-3) as a chained Vast job: export train predictions -> `build_distill_cache.py` (single teacher) -> matched student fit -> export.
3. Check the baseline-blend submission route (R2-6) against the rules and the packaging tool.
4. Design the SSM arm (R2-7) and the mirror decoding (R2-8) as separate, bounded build tasks after steps 1-3 report.

## 6. Status log

- 2026-10-10 early: R2-6 checked locally and **not supported**: adding the organizer baseline (kept in files `baseline/baseline.onnx`, scores 0.598/0.604 alone) at the teammate's
  weights (main 1-w, baseline 0.25 w) lowers our H230 students on fold1 (-0.0009 at w=0.25, -0.0003 at 0.15) and fold0 (-0.0013, -0.0005): they already carry baseline information in their
  targets. Public gains reported for it came from non-distilled members. A Z4 + baseline probe stays optional and unpredicted.
- Queue (both Vast lanes, interleaved by mechanism family): R2-1/R2-2 Muon multiplier and EMA arms with paired controls (lane B), fold0 transfer (lane A);
  R2-3 scale-then-distill chain `distill_pipeline.py` with four matched student arms: scratch GRU256 teachers, scratch GRU384 teachers, mixed 384+448, and a no-teacher continuation control (lane A, where the teachers live).
- Retired by prior evidence: class-3-only specialist (teammate -0.14), `since-weight` 8, `prop-power` 1/3 sweeps.
- Next builds: R2-7 selective state-space teacher (parallel scan), R2-8 exact mirror via decoded event codes, bootstrap-h8 option in the chain (needs distillation).

- 2026-10-10 public: Z5 (seed 9, Muon parent) **0.6664** (own best, +0.0002), Z2c (b16 parent) 0.6649; 5/5 daily quota used. Z-series pattern: Muon parents 0.6662/0.6664/0.6662, b16 parents 0.6649/0.6649.
  Parent lineage is worth about 0.0013 public while seed spread inside a lineage is at most 0.0002 (H-320). Consequence: parent-side levers (R2-1 Muon multiplier, R2-2 EMA, bootstrap) rank above student-side ones.
  Plan for the next five uploads: all-data Muon parents with the lever winners (Vast), same student plus head finishing, package with the existing gates.
