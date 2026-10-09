# Hyper-parameter optimization plan (Optuna): design, reasoning, budget, operation

Status 2026-10-10. Code: `hpo/` (space, store, lanes, orchestrator), `scripts/score_run.py`, tests `tests/test_hpo.py`.
Study config: `hpo/studies/teacher_v1/config.json`. Companions: [creative research contract](../research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/CREATIVE_RESEARCH_CONTRACT_R2_TOWARD_0700.md),
[optimizers and schedules](../research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/OPTIMIZERS_SCHEDULES_STABILITY_R1.md).

## 1. Honest expectation

- Where it helps: the levers that moved results are interaction-heavy (public: parent optimizer lineage +0.0013; teammate: Muon LR multiplier,
  schedule, budget). Learning rate x Muon multiplier x schedule x epochs x width x loss cannot be explored one factor at a time.
- Where it will not: one-factor sweeps by the teammate found most of the standard knobs flat ("current HPs at the optimum"). Realistic
  teacher-level gain: +0.001..+0.003 over the best recipe, attenuated through distillation to the serving student, so about
  +0.0005..+0.0015 public. It will not reach 0.70 on its own; architecture/data/information ideas must carry the rest.
- By-product that may matter more than the optimum: every trial is a trained, scored, **kept** teacher. Decorrelated good ones are
  blend/distillation partners (the pair rule `corr < r_j/r_i`). Predictions are therefore kept by default.
- Main statistical risk: winner's curse. Fold-level recipe deviation has sd 0.0016 (measured on 28 recipes, fold1 vs fold0, Pearson 0.81,
  Spearman 0.73, level shift +0.0131) and seed noise is about 0.001. With N trials the best observed score is optimistic by about
  0.001 x sqrt(2 ln N), i.e. about 0.003 at N=100, as large as the effect. Counter-measures: fold-wise racing, final confirmation on all four folds,
  and the public board as the last check (5 per day).

## 2. Validation scheme (what "robust cross-validation" means here)

Facts: `seq_ix` is shuffled and carries no time order (docs/data_overview.md), so time-blocked or forward-chaining splits cannot be
built from ids. Sequences are linked into chains (`chains.npz`); folds are component-disjoint (`chain_folds.npz`), which already
blocks chain leakage. Folds differ in difficulty (fold0 is 0.013 easier), and the test distribution differs from valid
(teammate probes: transfer loss concentrated in class 3). So:

1. **Grouped K-fold by chain component** (existing folds 1,0,2,3) is the unit of evaluation. No time split exists to be had.
2. **Fold-wise racing** (cross-validation pruning): every trial trains on fold 1; scores are reported cumulatively; a median pruner
   (10 startup trials) rejects clear losers; survivors run fold 0; only trials in the top 40% of completed trials after two folds spend folds 2 and 3.
   Objective = mean official WP over the folds completed. This buys 4-fold robustness for the finalists at about 1.9 fold-runs per trial instead of 4.
3. **Regime sensitivity is logged, not optimized**: per-class WP (class 3 = half of the metric weight at WP 0.44) and t0/t1 are stored per stage
   so post-hoc analysis can pick configurations that are good where the test is hard.
4. No fidelity ladder (fewer epochs / smaller data): the earlier 220-update screens flipped sign at full scale, and a one-cycle schedule
   makes partial-training scores uninformative. Racing across folds, not across training length.
5. Finalists (top 5): all four folds and one different-seed rerun (a robustness check, not a search dimension), then blend analysis with the incumbent.

## 3. Objective: single or multi-criteria

- **Main study: single objective** (fold-mean official WP). Rationale: per-class and t0/t1 scores are strongly collinear with it; adding objectives
  would fragment the budget and rules out CMA-ES and Hyperband-style pruning.
- Secondary values (class WP, t0/t1, finite fraction, AMP skips, train time) are stored for every stage.
- **Diversity** is not baked into the main objective; it is a second, smaller study later (NSGA-II over WP and `-corr` with the incumbent blend,
  scored from the kept predictions), because blend value is what a teacher is for. 20-25% of budget, after the main study.

## 4. Sampler and algorithm

| Phase | Sampler | Why |
|---|---|---|
| 0-1 (exploration, mixed categorical/continuous/conditional, 25 params) | **TPE**, multivariate + group, constant-liar for parallel asks, 24 random startup trials, 5 anchor configurations enqueued first | handles conditional and categorical spaces and noise; parallel-safe; no model fitting cost; GP-BO degrades with 20+ mixed dimensions |
| 2 (local refinement, ~10 continuous params, categoricals frozen to the top-k) | **CMA-ES** (IPOP restarts, population 8 = parallel width) | strong for continuous, correlated parameters (lr, Muon multiplier, wd, warmup, decay); noise-tolerant with large population |
| 3 (diversity archive) | **NSGA-II** (genetic) | natural for Pareto sets; used for the two-criteria diversity study only |
| Not chosen | plain GA for the main study | less sample-efficient than TPE at 80-150 trials; Bayesian GP (BoTorch) deferred: mixed conditional space and parallelism make it fragile |

Anchors (enqueued first, defaults for all other parameters): recipe default, Muon x3, Muon x3 + EMA, GRU384 + Muon x3, chrono init.
They validate the whole job path with known-sane configurations and give TPE good reference points.

Search space (core mode, v1; `hpo/space.py`): optimizer scope, base LR, Muon multiplier, weight decay, schedule family
(one-cycle, WSD sqrt, WSD linear, SGDR) with its parameters, epochs, width, depth, recurrent init (+ timescale), loss family (+ weights),
class-3 weight, averaging (none, EMA, SWA), dropout. Full mode adds head LR, clip, warmup, batch, chunk, cell type, readout activation, input
encoder, since-weight, propensity power and the two auxiliary weights. Seeds are never searched. Trials estimated above 2.6 h
are marked infeasible for the sampler (constraint `cost_hours`).

## 5. Failure handling: retry or skip

| Failure class | Detected by | Action |
|---|---|---|
| infra (host/ssh/job lost, kill signal, aggregate-RAM kill, disk) | rc -9/-15/137/143, "killed", "ssh failed", "job not found", timeout | retry up to 2 times (new job name), then skip with an alert |
| oom | "out of memory" | retry up to 2 times (the daemon places the job on the emptier GPU), then skip |
| numeric (divergence) | NaN/inf loss, non-finite predictions | **no retry**: record the trial as a penalty value (worst completed minus 0.02) so TPE learns to avoid the region |
| invalid (argparse, ValueError, unsupported combination) | error text in the log | **no retry**, trial FAIL, warning alert (a space bug to fix), combination not re-tried |
| unknown | anything else | one retry, then skip with an alert |

Rule of thumb: rerun only when the failure says nothing about the configuration; never rerun a configuration that failed on its merits.

## 6. Durability (laptop crash, killed process)

- `hpo/studies/<study>/events.jsonl` is the source of truth: append-only, fsync per line, committed to git. It holds asks, stage submissions
  (lane, job name, attempt), scores, retries, failures and tells.
- `study.db` (SQLite, git-ignored) holds the sampler/pruner history. If lost, `orchestrator replay --study NAME` rebuilds it from the log.
- The orchestrator is **stateless per tick** (a systemd timer, every 5 minutes). It takes a file lock, so overlapping ticks are harmless.
  Stages still marked in flight but unknown to the lane are treated as lost and retried. Jobs live on the Vast hosts independently of the laptop,
  so a laptop outage only delays ingestion.
- Guards: `STOP` file in the study directory (stop asking, drain), `max_trials`, per-lane spend caps (queue holds new jobs, running jobs always finish),
  keep-alive file so the lane drain check never destroys a host the optimizer still needs, watchdog alerts for stalled ticks or studies.

## 7. Routing across resources

- **Vast lanes A and B** (2 x RTX 4090 each, 6 concurrent jobs per host): all four folds' assets are staged on both. Jobs are
  `fit -> export -> score` chains; only a ~1 KB `score.json` travels back per trial. Lanes are filled up to `target` in flight; the HPO jobs join the
  family-interleaved queue, so they start as the existing experiment queue drains (they share slots meanwhile).
- **Local GPU** (5 GB): a third lane for configurations that fit (hidden <= 256, <= 2 layers); not wired yet; the local queue stays on student experiments.
- **Kaggle**: GPU quota exhausted for the campaign; a burst lane (fold-1 stage only, kernel overhead about 15 min, T4 about 10x slower) is possible later; CPU kernels
  are better used for data exploration. No routing now.
- Stage placement: fold-1 stages on any lane; later folds on the lane with free slots. Priority goes to older trials' next folds before new asks.

## 8. Budget and phase gates (rates: lane A $0.94/h, lane B $0.83/h; about 5 fold-runs/h per host; about 1.9 fold-runs per trial)

| Phase | Trials | Cost (approx.) | Gate to continue |
|---|---|---|---|
| 0: anchors + 19 random | 24 | $10 (about 6 h) | best fold-1 score minus median >= 0.003, else the space is flat: stop and move compute to architecture/data ideas |
| 1: TPE | to 80 total | $20 more (about 11 h) | best-so-far improved by >= 0.001 over the last 20 trials, else stop |
| 2: finalists | top 5, all folds + seed check | $5 | gain survives on folds 2,3 |
| 3: CMA-ES refinement | 30 | $10 | optional |
| 4: students (distillation recipe, top teachers, teacher mixtures) | 30 | $10 | optional |

Credit: lane A about $11 left (no top-up; cap 11), lane B $25 with auto top-up (cap 33). The queue of the earlier experiment waves occupies
the first hours; HPO ramps as slots free up (raise `target` in `config.json` to 6 per lane once the queues drain).

## 9. Operation

```
python -m hpo.orchestrator tick   --study teacher_v1      # one pass (the timer does this)
python -m hpo.orchestrator status --study teacher_v1      # states, best value and params
python -m hpo.orchestrator replay --study teacher_v1      # rebuild study.db from events.jsonl
touch hpo/studies/teacher_v1/STOP                          # stop asking, drain running trials
cp hpo/units/wnn-hpo@.* ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now wnn-hpo@teacher_v1.timer
```

Weekend rule: look only at watchdog alerts (`hpo:` keys) and `status`; do not edit `space.py` during a study (new space version = new study).

## 10. After the search

Top-k configurations feed (a) the Vast scale-then-distill chain as better teachers, (b) a student-stage search (alpha, LR, epochs, family weights, label weighting,
Dirichlet teacher mixing, random teacher per stage), (c) the diversity study. The student stage is a separate study with its own space and job type.
