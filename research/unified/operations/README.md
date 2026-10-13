# Execution, incidents, and preservation

This domain preserves run delivery, resource safety, incident diagnosis, and the final stop state.

## Read these sources

- [Private HF model archive transfer](../models/huggingface/README.md) — user-authorized private model uploads only, actual access checks, exact-byte verification and per-batch completion receipts; no research admission or new training.
- [October 12 strategy review contract](../../campaign_2026-10-12/strategy_review_r1/CONTRACT.json) — records the retrospective review's deliverable, independent role allocation, evidence limits and zero training/spending budget. It authorizes the review and proposed plan only; experimental admission remains closed.
- [Execution, incidents, and durable preservation](../archive/2026-10-12/pages/08-execution-and-preservation.md)
- [Stop authority](../../campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json) and [global admission barrier](../../campaign_2026-10-06/bandit_continuation/runtime/STOP_ADMISSION.json)
- [Kill prevention plan](../../campaign_2026-10-12/KILL_PREVENTION_PLAN_20261012.md)
- [Temporary artifact preservation receipts](../../campaign_2026-10-12/research_archive_inventory_r1/)

The October12 stop and its terminal-state archive remain preserved as historical records. The user explicitly resumed the strategy campaign on October13; current admission state is owned by the shared resource controller and linked campaign receipts below.

## Resumed strategy execution — 2026-10-13

[Strategy execution campaign](../../campaign_2026-10-13/strategy_execution_r1/README.md) implements the full October12 review under the user's subsequent explicit resume. It links the approved plan, full proposal coverage, resource authority, frozen comparisons and live frontier. All Kaggle allowance and actual local GPU capacity are authorized; CPU/RAM protections and no-paid-spend remain. R3 P1 completed and the admitted P2 continuation is actively fitting: see the [P1→P2 resume-gate receipt](../../campaign_2026-10-13/strategy_execution_r1/target_amplitude_f0_s3_r1/P1_P2_RESUME_GATE_R1.json), [P2 attempt receipt](../../campaign_2026-10-06/bandit_continuation/runtime/attempts/clean-t3-f0-amplitude1-s3-a9-r3-p2-20261013.json) and [P2 worker log](../../campaign_2026-10-06/bandit_continuation/runtime/logs/clean-t3-f0-amplitude1-s3-a9-r3-p2-20261013.log). The [P1 worker-local receipt](../../campaign_2026-10-06/bandit_continuation/runtime/attempts/clean-t3-f0-amplitude1-s3-a9-r3-p1-20261013.json) ended successfully at the validated epoch boundary. No amplitude score exists yet. Existing folds remain exposed.

- [Custom-output prediction alias workflow](../../campaign_2026-10-13/strategy_execution_r1/PREDICTION_ALIAS_WORKFLOW_R1.json) and [validation helper](../../../scripts/alias_grade_prediction.py) — the full-valid exporter writes under its custom `--output-name`, while the frozen grader consumes a run-name path and checks `sidecar.run`. The helper validates the actual NPY header, streamed prediction hash, CP5 checkpoint hash, split hash and FP32 metadata before creating no-overwrite hardlinks for the NPY and identical sidecar. A verify-only check passed on the actual matched-control producer and its two existing aliases; the new amplitude/S14 outputs still require their own post-export checks.
