# Class-3 public-mix weighting screen — 2026-10-25 R1

User: "utilize gpu, experiment!". Question: does `--class-weight 1,1,1,1.43` (class-3
weight matched to the E-298 public-fit class mix) improve the strong warm H256×2 student?

- [Recipe](RECIPE_R1.json): F0 seed3 candidate vs **reused exact control**
  `overnight_nll_f0_s3_wmse_e2_r1` CP1 (0.6813131585); F1 seed4 control + candidate
  queued. 3 new fits total, fixed CP1, frozen source `overnight-nll-adaptation-20261020`.
- Execution: `wnn-c3mix-f0` (fit), `wnn-c3mix-queue` ([queue.sh](queue.sh): exports,
  F1 fits, scoring), `wnn-c3mix-thermal` ([guard](thermal_guard.sh): freeze ≥97 °C,
  thaw ≤88 °C). Limits: ≤14 GB RAM, no swap, ≤1.5 CPU, ≤4.5 GB CUDA allocator.
- Grading: [F0 screen](screen_f0.json) / [F1 screen](screen_f1.json) via
  `scripts/screen_score.py` with the E-298 public-fit class-mix diagnostic.
  F0/F1 are exposed folds, one seed per fold: direction evidence only.
- Status: running. Results will be appended here.

## F0 result (2026-10-25 01:25 UTC) — [grade](grade_f0_R1.json)

| Arm | Official WP | Class 3 WP | Public-mix (E-298) WP |
|---|---:|---:|---:|
| control (reused) | 0.6813132 | 0.41737 | 0.64258 |
| cw143 | 0.6812545 | 0.41717 | 0.64249 |

Paired component bootstrap (141 components): official −0.0000586 [−0.000131, −0.0000015];
public-mix −0.0000930 [−0.000176, −0.000023]. Class weight did not move class 3 at all in
this warm LR3e-5 two-epoch continuation; the regime is insensitive to row-measure changes.
The unstarted cw2.0 dose arm was cancelled (amendment R2). F1 pair continues.
One seed, exposed fold: direction evidence only, not a refutation of class weighting in
teacher/cold training (H-360).

## F1 execution failure and stage-6 replacement (amendment R3)

The queued F1 control failed at startup: `ValueError: distillation cache metadata exceeds
its 512000-byte read bound` (the F1 four-family cache metadata is larger than the frozen
source's bound). Execution evidence only. Because the warm LR3e-5 regime proved insensitive,
F1 was not repaired; instead a more sensitive matched strict-FP32 stage is running
(`wnn-c3mix-queue3`, [queue3.sh](queue3.sh)): init `c20261003_combo_s3_f0@ep3`, 6 epochs,
LR 1e-4, fixed CP5, [control](stage6_control_f0.argv) vs [cw143](stage6_candidate_f0.argv),
graded by [screen_stage6_f0.json](screen_stage6_f0.json) (also vs the historical FP16
parent as a reference).

Note: the cancelled cw2.0 arm had already started (queue2 began at 01:25:32, right after
the F1 failure) and was stopped about 15 minutes in, before any checkpoint; its empty
run directory was removed. No partial result exists or is claimed.
