# Target-only frozen input adapter

H-ADAPTER-TARGETONLY-20261018-R1 / E-ADAPTER-TARGETONLY-20261018-R1. This is an explicit objective ablation after the negative inherited-objective adapter, not a claim that gradient conflict was observed.

## Frozen implementation handoff

- [Recipe](recipe_target_only.json): same parent, full 60 feature block, seed 3, sample budget/order, fixed epoch, optimizer and schedule as R2. The intervention skips auxiliary losses while retaining inherited flags, model heads and state shapes. Local F0 remains exposed; no alpha search in this arm.
- [Patch](adapter_objective.patch) and [source hashes](source_and_patch_hashes.json): three small modules from exact current root source snapshots. Default inherited behavior remains unchanged. Root must verify original pins before integration.
- [Actor fidelity receipt](objective_fidelity_receipt.json): AST and code-path checks only; no checkpoint, data or model execution. Independent review is separate.
- [Execution receipt](execution_receipt.json): static-check commands and a setup-bound deviation. Worktree checkout used the wrapper defaults for 5.165 seconds; no payload was read and no peak memory was measured. Subsequent checks used explicit 20% CPU / 500 MB limits. Parent limits were not raised.

Historical handoff status: at the time this packet was prepared, root still owned integration, profile, admission, fit, export and grading. Those steps subsequently completed; the result and independent grade are recorded below. F0 remains exposed.

### Actual-shape profile admission

[First profile failure](profile_startup_failure.json) records an argument-validation rejection before fitting: profiling exits cannot save epoch continuation state. [Revised profile command](profile_command_v2.json) removes only that incompatible output flag; the full training recipe retains it. Both [first log](profile.log) and [revised log](profile_v2.log) are preserved. This is an execution repair, not scientific evidence against the objective.

[Profile receipt](profile_receipt.json) passed 60 actual-shape finite updates with 3.830 GB cgroup peak and no OOM. The frozen target-only recipe was admitted under the [continuous runtime](../runtime/README.md), reserving 4.5 GB and one GPU worker.

## Completed run and independent grade

- [Independent review](INDEPENDENT_REVIEW.md), [reproducible grade driver](grade_target_only_independent.py), [result](independent_grade_result.json), [receipt](independent_grade_receipt.json) and [pre-score identity freeze](independent_grade_frozen_inputs.json) bind the exact checkpoints, caches, sidecars, fold and official scorer. Candidate and R2 scores reproduce their saved score records exactly; the parent reference also reproduces its recorded F0 WP.
- Target-only scores **0.680145482657** versus **0.680449904025** for the matched R2 full-feature adapter and **0.681226513222** for the parent. Conditional paired component-bootstrap intervals are [−0.00053839, −0.00009132] versus R2 and [−0.00181516, −0.00044532] versus parent. These fixed-endpoint intervals do not account for training-seed variation or adaptive selection.
- Candidate and R2 use identical source-qualified F0 groups, source-separated fit membership, parent checkpoint, feature layout, seed, sampler and training arguments; only run name and adapter objective differ. Parent predictions were aligned by group ID from a full-valid cache. Valid-fit group membership is disjoint from F0, but F0 is exposed and adaptively reused.
- A preliminary overlap check incorrectly combined row-group IDs from separate train and valid parquet namespaces. The corrected valid-to-valid check finds no F0 fit overlap for candidate, R2 or parent; the original false flag is preserved and explained in the result.
- The saved artifacts do not prove bitwise old-head invariance: the objective receipt is static AST evidence, and no parameterwise checkpoint comparison was recorded. No Torch/model payload load was performed for this review.
