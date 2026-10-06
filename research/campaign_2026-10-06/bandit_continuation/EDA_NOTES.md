# Historical teacher diagnostics

The marimo notebook `teacher_eda.py` samples 64 exposed fold-0 evaluation groups with seed 20261006, every 40th row after warm-up. There are 2,682 scored sampled rows. These are old cached predictions, not new fits or a public forecast. The notebook has no training side effects.

Row order and finite sampled predictions were checked for four caches. Full-file hashes remain an experiment-admission responsibility; the EDA manifest records sidecar hashes without claiming they were independently verified.

Weighted clipped-label residual correlations exceed 0.9927 in this sample. Residual noise can dominate that statistic, so it does not establish that heterogeneous teachers are redundant. The disagreement-versus-error plot is an association diagnostic only. It is not monotonic: in this sample the highest-disagreement bin has lower mean absolute error than several middle bins for both targets. Do not assume disagreement is a calibrated hardness/uncertainty signal; label magnitude, market class and sequence composition may confound this association. The registered C1–C4 controls will test whether preserving family signals helps training.

- [Residual overlap](plots/teacher_residual_correlation.png)
- [Disagreement versus error](plots/teacher_disagreement_vs_error.png)
- [Sampling and cache provenance](TEACHER_EDA.json)
- [Numerical plot data](TEACHER_EDA_STATISTICS.json)

Run `marimo edit research/campaign_2026-10-06/bandit_continuation/teacher_eda.py` from the repository environment. Existing local session: http://localhost:2721.
