# Execution, incidents, and preservation

This domain preserves run delivery, resource safety, incident diagnosis, and the final stop state.

## Read these sources

- [Private HF model archive transfer](../models/huggingface/README.md) — user-authorized private model uploads only, actual access checks, exact-byte verification and per-batch completion receipts; no research admission or new training.
- [October 12 strategy review contract](../../campaign_2026-10-12/strategy_review_r1/CONTRACT.json) — records the retrospective review's deliverable, independent role allocation, evidence limits and zero training/spending budget. It authorizes the review and proposed plan only; experimental admission remains closed.
- [Execution, incidents, and durable preservation](../archive/2026-10-12/pages/08-execution-and-preservation.md)
- [Stop authority](../../campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json) and [global admission barrier](../../campaign_2026-10-06/bandit_continuation/runtime/STOP_ADMISSION.json)
- [Kill prevention plan](../../campaign_2026-10-12/KILL_PREVENTION_PLAN_20261012.md)
- [Temporary artifact preservation receipts](../../campaign_2026-10-12/research_archive_inventory_r1/)
- [Approved October 13 strategy execution](../../campaign_2026-10-13/strategy_execution_r1/PLAN.md) and [first-cohort F3 clean-T3 teacher packet](../../campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1/README.md) — identifies the missing exact F3 H384/Muon-all teachers, binds the canonical split and F3 propensity inputs, and stages seed 3 as a locally sealed profile→fit→export Kaggle packet. Its [entrypoint binding](../../campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1/ENTRYPOINT_BINDING_R1.json) fixes the exact kernel/archive hashes and title-derived slug; the [offline bootstrap fixture](../../../tests/test_f3_teacher_packet_entrypoint.py) executes the real bootstrap against temporary mounts without GPU/data loading. Current quota and resource estimate are recorded in the packet README. Root review is pending; no private dataset upload or kernel push has occurred. The predecessor October 12 stop was explicitly superseded by this newer user authorization, which does not authorize paid capacity or change local resource bounds.
- [Quota-bound seed-3 F3 teacher packet successor](../../campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r2/README.md) — preserves the unlaunched R1 packet while correcting seed-level FP32 export arithmetic, binding a quota-derived wall deadline with finite profile/fit/export reserves, and testing the exact generated bootstrap against nested dataset-qualified mounts and the actual split loader. R2 remains root-review pending; no private dataset upload, kernel push, or worker launch has occurred.

The October 12 archive remains historical evidence. The later October 13 strategy-execution contract supersedes that stop for the admitted campaign scope; it does not authorize paid capacity or change the local CPU/RAM/no-swap bounds.
