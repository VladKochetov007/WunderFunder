# October 13 strategy execution

This folder records the authorized first cohort of clean fold-3 H384/Muon-all
teachers. The R1 packet defines the frozen recipe and source lineage. R2
preserves its first quota-bounded Kaggle attempt, including the successful T4
profile and the RAM-admission failure before fitting. R3 is a corrected,
quota-bounded successor that reuses that exact profile and stages the fit plus
fit-train prediction first; fit-valid and full-valid exports remain assigned to
a checkpoint successor after terminal collection and a fresh quota review.

- [R1 frozen recipe and first packet](first_cohort_f3_teachers_r1/README.md) —
  source, split, and producer bindings for the scratch-initialized seed-3
  teacher.
- [R2 terminal attempt](first_cohort_f3_teachers_r2/README.md) — preserves the
  successful resource profile and the pre-fit worker-admission failure.
- [R3 fit and fit-train recovery packet](first_cohort_f3_teachers_r3/README.md)
  — repairs host RAM measurement/admission, verifies the reusable profile
  against terminal output hashes and the R1 recipe, and defers unmeasured
  prediction stages to a checkpoint successor.

- [F3 R4 conservative successor](first_cohort_f3_teachers_r4/README.md) preserves R3, binds the completedR2profile andfreshquota, and passesrootarchive/bootstrap/admission checks. Rootnowowns delivery; no newscience/provideractionyet.
