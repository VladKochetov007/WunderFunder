# H099 current-student state reset panel

`scripts/feature_reset.py` prepares a fixed component-hash panel before reading
scores, then runs the same current checkpoint under no reset and GRU-state resets
every250/500/1000/2000 rows. Each lag also has a same-boundary0.5 state-retention
control. The first99 warmup rows are preserved: reset boundaries are99+k×lag,
k≥1. A reset acts before processing the boundary row. All raw/event/volume
features retain their full causal history; only carried GRU state changes.

The panel includes at most16 complete evaluation components and32 row groups,
selected by SHA(component,271828), without label or score lookup. Predictions
use the same prespecified jittered rows and scoring-mask intersection as E415;
empty sequences are logged and skipped without resampling. The sample and fixed
checkpoint are content-bound. Full-final checkpoints that fitted panel components
require an explicit `--allow-fit-overlap` declaration and report the overlapping
components. Fold-excluded checkpoints do not need that diagnostic exception.

The output reports every reset against no-reset and its attenuation control with
official Global WP and paired component intervals. This measures fixed-weight
state-value disruption. It does not establish the usefulness or length of training
gradients, and it is not a clean final holdout. Historical negative reset results
are controls, not a reason to omit this current-checkpoint retest.

```bash
python scripts/feature_reset.py --root . \
  --manifest research/campaign_2026-10-04/kaggle/feature_plan_f0/direction0_fit.json \
  --output research/campaign_2026-10-04/kaggle/reset_panel_f0.json plan
python scripts/feature_reset.py --root . \
  --manifest research/campaign_2026-10-04/kaggle/reset_panel_f0.json \
  --output research/campaign_2026-10-04/feature_reset_results/f0_v1 run --device cuda
```

These commands require controller admission for execution. No fits or reset-panel
scores were run in the implementation milestone. Tests run tiny actual GRUs on
CPU, checking no-reset parity, all fixed boundaries, warmup and prefix causality,
unchanged weights, and complete-component membership.
