# i1 lead-block screen — 2026-10-25 R1

**Hypothesis (H-I1LEAD-HELD-R1).** TL3's score-carrying FTT members use 52 extra inputs that
the [fingerprint](../class3_reframe_r1/FTT_LEAD_BLOCK_FINGERPRINT_R1.md) identifies as i1 raw
values held at the most recent i0 book event (std-ratio match RMSE 0.025, corr 0.987; lags
and EMAs fail). Our inputs lack this snapshot (`d_i1px` is only the one-row i1 change), so
"i1 moved while i0's book stood still" — the class-3 reframe's i1-as-twin-of-F signal — is
not explicit. Prediction: adding the block improves WP, most in class 3.

- Feature: [`src/wnn/i1_lead_block.py`](../../../src/wnn/i1_lead_block.py) registered as
  research mode `i1_held_i0book` (commit `96d31cf4d`; NumPy/torch/streaming parity,
  causality, reset and zero-control tests pass). Frozen worktree
  `research_tmp/i1-lead-screen-20261025` at that commit.
- Design: 4 cold fits on the reviewed fast recipe (H512×1, 10% frozen memberships, 3 epochs,
  fixed CP2) — F0/s1001 and F2/s1002 × {control: block present but zeroed
  (`--research-feature-control all`), lead: active}. Same width, init and data order.
  [Plans](PLANS_R1.json), [runner](run_panel.sh), screens [F0](screen_f0.json)/[F2](screen_f2.json)
  with class and E-298 public-mix breakdowns. Exposed folds, one seed per fold.
- Units `wnn-i1lead`, `wnn-i1lead-thermal`. Status: running (~1 h).
