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
