# Prepared first-comparison commands

These JSON files use the existing `scripts/kaggle_sequence.py` finite-step
schema. They are **prepared arguments, not admitted jobs**. Feature implementation
is integrated on main at `0cc795070`. Root must freeze source/data identities, complete bounded
engineering checks and an actual-shape profile, and record the resource/budget
admission before execution from October 20. No scheduler is installed.

- [Local A](local_a.json): cold H512×1 with all explicitly i1-derived channels masked after
  common preprocessing and zero history slots.
- [Local B](local_b.json): same initialized model, standard raw/event inputs,
  zero history slots. This is the shared comparison control.
- [Local C](local_c.json): same model and standard inputs plus 156 causal i1
  history channels. Same fitting membership, order, schedule and fixed ep2.

Each finite plan fits one endpoint, exports predictions on the complete F0
evaluation panel and calls the existing official scorer. Independent comparison
grading and the fixed 90/10 blend are separate decisions using saved predictions;
no plan automatically starts another arm or selects a winner.

The [recipe](../recipe.json) supplies exact profile argv, environment variables,
artifact paths and Kaggle prerequisites. Its first 10% component subset has
1,236 fitting sequences: 154 batches per epoch at batch 8, ten updates per full
sequence batch, three epochs = 4,620 expected updates. The existing sampler
drops four tail sequences per epoch; record the actual consumed order. The fit
membership itself remains component-complete.

## Execution interface after admission

Inside the existing admitted worker scope, with the recipe environment and root
working directory, use the finite plan through the existing runner:

```sh
.venv/bin/python scripts/kaggle_sequence.py \
  --plan research/campaign_2026-10-20/plateau_review_r1/recipes/local_a.json
```

Substitute `local_b.json` or `local_c.json` only for its separately admitted arm.
This invocation alone does not enforce the campaign budget or acquire the shared
GPU lock. Those belong to the existing durable controller/resource wrapper; do
not bypass it. The plan's report must be new. On failure inspect retained logs,
checkpoint and phase report, then prepare a source-bound recovery attempt using
the canonical trainer's `--resume-state`; do not rerun completed fits.

Kaggle must use a sealed dependency bundle and provider profile. The observed
5.02h gross quota is not permission to start a full control unless its matched
treatment and exports fit the available quota on that actual device. An explicit
fidelity amendment is preferable to an unusable unpaired control.
