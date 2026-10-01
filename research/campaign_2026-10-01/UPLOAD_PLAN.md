# Upload plan (2026-10-02 00:15) — not uploaded; user decision

All archives are our own code (own Rust int16/fp32 engine, own training scripts); teammate ideas only.
All passed: ZIP ≤ 20 MB, root `solution.py`, fixed members and weights (no runtime governor), warm-up returns None,
state reset on new `seq_ix` (A→B→A bitwise), finite outputs, repeat-deterministic, v2/avx2/v3 identical, offline
Python 3.11 scorer-image run, int16-vs-fp32 parity ≤ 1e-4.

Forecasts: 50/50 mix of holdout route (public Δ = 0.657 × holdout Δ) and clean-fold route (selection-shrunk,
transfer slope 0.75 ± 0.35), plus ±0.0006 leaderboard noise. Reference: our public 0.6622; NaiveBayes 0.6641.

| order | archive | what it tests | runtime | expected public | 80% range | P(> 0.6641) |
|---|---|---|---|---|---|---|
| 1 | `submissions/c20261002_R4D_i16/submission.zip` (sha256 `c474138f…fc76`) | class-routed since + distilled GRU256 | ~45 min | 0.6634 | 0.6623–0.6645 | 0.22 |
| 2 | `submissions/c20261001_S4D_i16/submission.zip` (sha256 `ab0f5287…ec00`) | since + distilled GRU256, merged heads, no routing | ~48 min | 0.6632 | 0.6621–0.6643 | 0.15 |
| 3 | `submissions/c20261002_D256_i16/submission.zip` (sha256 `c493f042…0e32`) | distilled GRU256 alone (merged heads) | ~38 min | 0.6629 | 0.6618–0.6640 | 0.09 |
| 4 | `submissions/c20261001_RSP2_f32/submission.zip` (sha256 `2783eea9…bdc20`) | head-mix routing only, incumbent trunks | ~29 min | 0.6627 | 0.6617–0.6637 | 0.04 |

Why this order uses the quota well: 1 vs 2 isolates class routing; 2 vs 3 isolates the since trunk next to the
distilled GRU256; 4 vs the 0.6622 incumbent isolates the parent+FT head mix. Each public result recalibrates the
next decision. None is expected to reach 0.6660.

Next candidate with a real chance to pass NaiveBayes: the team-candidate recipe rebuilt in our code (GRU256
teachers → distilled GRU256, GRU128 since s3/s4 and chunk-500 on train + all validation, head-only FT), training in
`jobs/queue3.txt` (~4.5 GPU-h). Its anchor is the teammates' measured 0.6640–0.6641 for that recipe.

Copies with priority prefixes: `submissions/upload_2026-10-02/` (SHA256SUMS inside).
D256 measured 296 kcyc/row ≈ 37.6 public min; R4D 354 kcyc/row ≈ 45 min.
