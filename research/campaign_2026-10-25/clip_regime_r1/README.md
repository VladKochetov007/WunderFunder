# Clip-regime panel — 2026-10-25 R1

Global-norm clipping at 1.0 is active on 94% of steps in the fast cold recipe. With AdamW,
per-step clip factors down-weight large-norm steps inside the moment estimates — an implicit,
untuned batch reweighting (see [theory note](../optimizer_panel_r1/OPTIMIZER_THEORY_R1.md)).
Never varied before in this repository. Four fits: `--no-gradient-clip` and `--clip 0.3` ×
F0/s1001, F2/s1002 vs the reused clip-1.0 controls. [Plans](PLANS_R1.json),
[runner](run_panel.sh), screens [F0](screen_f0.json)/[F2](screen_f2.json). Queued after the
optimizer panel. Direction-finding screen, exposed folds.

## Result ([F0](grade_f0_R1.json), [F2](grade_f2_R1.json))

| Arm vs clip 1.0 | F0 Δ [95% CI] | F2 Δ [95% CI] |
|---|---|---|
| no clip | −0.001796 [−0.004366, +0.000606] | **−0.003208 [−0.005229, −0.001231]** |
| clip 0.3 | −0.001106 [−0.003432, +0.000592] | +0.000114 [−0.000904, +0.001165] |

Removing clipping hurts (mean −0.0025, F2 interval excludes zero); tightening to 0.3 is
neutral. Because AdamW is per-step scale invariant, clipping acts only by reweighting steps
inside the moment estimates — down-weighting high-norm (high-|y|-mass) batches. That
implicit robust reweighting is beneficial and saturated by clip ≈1.0. Keep clip 1.0.
Two interrupted fits (two resets, one restart) were discarded and refit from scratch.
