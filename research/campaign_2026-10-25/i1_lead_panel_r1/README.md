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

## Execution note and F0 result

Export initially failed: the current `predict_run.py` rejects `--loader-workers` (accepted by
older frozen trees); the flag was removed and the completed fit reused. F0 grade
([grade_f0_R1.json](grade_f0_R1.json)):

| Arm | Official WP | class 0 | class 1 | class 2 | class 3 | public-mix |
|---|---:|---:|---:|---:|---:|---:|
| control (zero block) | 0.659530 | 0.7944 | 0.7344 | 0.7227 | 0.3913 | 0.6195 |
| lead block | 0.654645 | 0.7880 | 0.7326 | 0.7210 | 0.3821 | 0.6155 |

Paired **−0.004885 [−0.007007, −0.003018]**: the held-i1 block hurts in every class. Because
the input projection is linear, a "move since i0 event" difference variant carries the same
information and is not queued. Likely explanation: 52 near-collinear inputs add variance in
this 10%-data cold fit; FTT's benefit may come from its tied i0/i1 encoder or full-data
training rather than the block itself. F2 pending.

## F2 result and closure ([grade_f2_R1.json](grade_f2_R1.json))

F2: control 0.683091, lead 0.681479, paired **−0.001612 [−0.003001, −0.000259]**; class 3
0.43985 → 0.43819; public-mix 0.65852 → 0.65725. Both cells negative with intervals excluding
zero: **H-I1LEAD-HELD-R1 closed negative** for concatenated inputs in the cold GRU. FTT's
advantage, if causal, must come from its tied i0/i1 encoder, full-data training or ensemble
role — not from adding this block to our architecture.
