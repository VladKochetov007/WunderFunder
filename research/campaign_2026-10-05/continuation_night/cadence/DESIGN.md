# E-495: cadence, pooled mass, and recurrent credit

Status: registered preparation only. No resource probe, fit, prediction export,
or score has been launched. This continues H-155 and the historical E-430
cadence question with fresh strong-recipe parents and clean held-component
lineage.

## Question and contrasts

Do fewer optimizer updates at the same processed-row EMA horizon help the
complete predictor, and does a longer recurrent gradient path add value after
matching update count and pooled loss mass?

Three fresh arms share the same recipe, row order, data, model, objective,
optimizer, schedule, precision, teacher caches, and fixed epoch-3 checkpoint:

| Arm | Micro rows | Update rows | Detach rows | Intended contrast |
|---|---:|---:|---:|---|
| `update1000` | 1,000 | 1,000 | 1,000 | Fresh control |
| `pooled2000` | 1,000 | 2,000 | 1,000 | One pooled two-microchunk loss and update, with state values carried but gradient cut after 1,000 rows |
| `connected2000` | 1,000 | 2,000 | 2,000 | Same two-microchunk update window with recurrent gradient carried across the boundary |

The primary operational cadence contrast is `pooled2000` versus `update1000`.
It includes fewer optimizer, scheduler, clipping, and weight-decay applications.
The credit contrast is `connected2000` versus `pooled2000`: update width and
loss mass match, while the graph remains connected for 2,000 rather than 1,000
rows. `connected2000` versus `update1000` is reported as a joint change.

All arms use the existing `--update-rows`, `--detach-rows`, and
`--ema-reference-rows` options. The recipe sets `--ema-decay 0.999` and
`--ema-reference-rows 1000`, so the per-update EMA factors are 0.999 for a
1,000-row update and `0.999**2 = 0.998001` for either 2,000-row update. This
matches nominal parameter retention per processed row; EMA trajectories can
still differ because students update at different times. AMP-skipped attempts
advance the existing EMA and scheduler clocks, so reports must show attempted
steps, successful optimizer updates, skips, and EMA updates by epoch.

The controlled trainer path pools weighted numerator and denominator across
each update window. Loss weights continue to use the original clipped labels;
distillation changes targets, not their mass. The 2,000-row arms must preserve
the final rows: the 20,000-row sequence length is divisible by both update
widths, yielding 20 or 10 update windows with no dropped tail. Any runtime or
source change that removes or changes these option semantics invalidates the
prepared recipes; this workstream does not add flags or modify the trainer.

## Recipe, cohort, and endpoint

`scripts/continuation_recipes.py` supplies the four-epoch strong combo recipe:
H256, two-layer GRU, batch 32, WMSE, Muon over the RNN, LR 0.001, weight decay
0.01, propensity power 2 and floor 0.03, since weighting 32 with floor 0.25,
structural/next auxiliaries 0.3/0.3, the two fold-matched since teachers at
alpha 0.5 with stopped bootstrap horizon 8, EMA teacher weight 1, and evaluation
from EMA weights. Each arm adds seeded batch order and the registered cadence
flags only. Checkpoint policy is fixed at epoch 3; no evaluation occurs during
fitting.

The screen is seed 3 / fold 0. Promotion requires confirmation at seed 4 / fold
0 and seed 3 / fold 1. All three arms are specified at each block so every
contrast has a fresh matched control. Fold manifests provide the fitting and
evaluation component sets; no component or ancestor crosses from a fold's
evaluation role into its fit role. The 24-fit / 16-test small cohort is reserved
for small diagnostics where it was designed for that purpose. It does not
substitute for a resource profile or a full-data fit here.

Parent endpoint: each arm's fixed epoch-3 model. Complete endpoint: the fixed
equal mixture of its epoch-3 parent and its one-epoch head trained on the same
fold's permitted validation fitting pool, using the existing head recipe and
fixed epoch 0. No mixture weight or checkpoint is selected from evaluation
results. Score parent and complete endpoints separately with the frozen
`global_wp_clip2_v1` scorer on the fold's held components. Use 2,000 paired
component bootstrap draws for all three pairwise arm comparisons. Report each
target, market-class and event-age breakdowns, and the exact evaluation
component count. Any PUB143 reweighting is a labeled inferred scenario, not an
official forecast.

## Admission order and limits

The drafts contain one full-production-shape resource probe per arm for seed
3 / fold 0. Each uses the real data loader, objective, teachers, EMA, optimizer,
batch 32, and two measured profile steps after one warmup. Probes are diagnostic
attempts and never become candidates. The connected arm has the highest
activation-memory risk and must pass its own probe.

Remote limits follow the current continuation convention: one scientific
worker, two CPUs, 18,000,000,000-byte process-tree RAM cap, 4,200,000,000-byte
allocator cap, and 4,800,000,000-byte observed-device stop. Effective host RAM
must be at least 25,715,000,000 bytes for the 70% target; stop before 15%
available headroom, and use the stronger host or provider limit if present.
Loader workers are zero. Probe limit is 900 seconds; fit/head/prediction draft
limit is 14,400 seconds. Current provider quota and active jobs, root budget,
local cutoff, and independent source review remain controller-owned gates.

No full fit is admitted by these drafts. Each job names all raw data, cache,
teacher, propensity, split-manifest, code, and design inputs so the runner can
hash them immediately before reservation. Missing parent checkpoints in head
and prediction drafts are expected until their prerequisite fit completes.
Inputs and inherited historical artifacts are read-only.

## Claims and limits

The initial comparison is one exposed clean fold/seed screen, not a stable
generalization estimate or public-score forecast. Any held-component effect
comes only from a properly bound clean fold and must be independently scored.
Do not describe in-sample training traces, the small cohort, a resource probe,
or the inferred PUB143 scenario as a held-out or public result. Candidate
promotion requires the predeclared confirmations and the campaign's complete
reporting gates. E-495 preparation itself claims zero fits and zero score delta.
