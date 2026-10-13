---
kind: experiment
status: proposed
created_utc: 2026-10-13T18:31:46.281621+00:00
hypothesis_ids: [H-QAT-LSQ-20261013]
experiment_ids: [E-QAT-LSQ-10PCT-F0S4-20261013-R1]
---

# Seed-4 learned-scale confirmation

The seed-3 LSQ screen had a small positive but uncertain native endpoint delta. This matched pair tests whether it repeats with the clean-T3 seed-4 CP5 parent and fitting RNG, while preserving the same component-complete adaptation subset and full F0 evaluation membership.

[Execution recipe](EXECUTION_R1.json) pins the existing independently reviewed LSQ source, exact seed-4 parent checkpoint/provenance, matched fresh AdamW one-cycle settings and profile/fit commands. [Subset manifest](fit_components_10pct.json) contains 315 components, 1,048 train and 104 fitting-valid sequences. Its member hash matches seed3 deliberately; both arms therefore have 72 batches and 1,440 updates in one epoch. New experiment identity does not imply new validation exposure. Both selected first-head Q8 blocks and recurrent W4 row steps are learned in the candidate; the control recomputes nonlearned row absmax steps after each update.

Root owns independent successor review, live admission, actual-shape 100-update profiles, matched fits, native full-trajectory export/evaluation and grade. These commands are prepared, not admitted. Host reservation remains 6GB based on the measured seed3 profiles/fits; live total committed memory must remain below17GB, with one local GPU worker. No source/hardware/shape change is hidden.

The seed4 parent/cache identities come from the [Kaggle seed4 ancestry packet](../kaggle_f0_s4_r1/README.md), and the independent [allocation recommendation](../../../unified/reviews/QAT_LSQ_SCALE_CONFIRMATION_ALLOCATION_R2.md) explains why missing F3 ancestry does not block this feasible seed replication. The result will measure fitting-seed variation on an exposed fold; cross-fold validation remains separate.
