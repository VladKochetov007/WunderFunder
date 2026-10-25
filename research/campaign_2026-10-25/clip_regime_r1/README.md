# Clip-regime panel — 2026-10-25 R1

Global-norm clipping at 1.0 is active on 94% of steps in the fast cold recipe. With AdamW,
per-step clip factors down-weight large-norm steps inside the moment estimates — an implicit,
untuned batch reweighting (see [theory note](../optimizer_panel_r1/OPTIMIZER_THEORY_R1.md)).
Never varied before in this repository. Four fits: `--no-gradient-clip` and `--clip 0.3` ×
F0/s1001, F2/s1002 vs the reused clip-1.0 controls. [Plans](PLANS_R1.json),
[runner](run_panel.sh), screens [F0](screen_f0.json)/[F2](screen_f2.json). Queued after the
optimizer panel. Direction-finding screen, exposed folds.
