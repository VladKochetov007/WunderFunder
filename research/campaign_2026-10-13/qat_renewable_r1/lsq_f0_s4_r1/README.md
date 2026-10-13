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

[Root profile allocation](PROFILE_ALLOCATION_R1.json) admits two sequential 100-update profiles under live guards after current work. The [independent recipe review](../../../unified/reviews/QAT_LSQ_F0S4_PROFILE_RECIPE_REVIEW_R1.md) passes checkpoint/source/subset and exact-command identity checks; measured resource and finite-update evidence remain required before either fit.

[Profile acceptance and fit allocation](PROFILE_ACCEPTANCE_AND_FIT_ALLOCATION_R1.json) records both 100-update finite passes (4.10/4.04GB host, 2.17/2.18GB observed VRAM) and admits the frozen matched pair through the [guarded fit sequence](FIT_SEQUENCE_R1.json). The first learned-profile admission was refused before startup at17.024GB projected memory; the [cache handoff](CACHE_HANDOFF_AFTER_FIXED_PROFILE_R1.json) and [R2 recovery](LEARNED_PROFILE_RECOVERY_R2.json) preserved the same source/command/reservation and restored headroom. No profile model is selected as a result.

Runtime profile records: [fixed success](../../../campaign_2026-10-06/bandit_continuation/runtime/attempts/qat-lsq-f0-s4-profile-fixed-r1.json), [learned R1 pre-start refusal](../../../campaign_2026-10-06/bandit_continuation/runtime/attempts/qat-lsq-f0-s4-profile-learned-r1.json), and [learned R2 success](../../../campaign_2026-10-06/bandit_continuation/runtime/attempts/qat-lsq-f0-s4-profile-learned-r2.json); [fixed log](../../../campaign_2026-10-06/bandit_continuation/runtime/logs/qat-lsq-f0-s4-profile-fixed-r1.log) and [learned log](../../../campaign_2026-10-06/bandit_continuation/runtime/logs/qat-lsq-f0-s4-profile-learned-r2.log) retain complete finite-update evidence.
