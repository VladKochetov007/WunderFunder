# ADR0020: explicit precision in paired representation tests

Status: accepted, 2026-10-08. Owner: root. Scope: E540 numerical-representation comparison and interpretation of its failed endpoint gate.

E540 V6 verified its inputs, passed both FP16 profiles with316 completed updates and zero skips, and trained the linear control through72996 optimizer attempts. Dynamic AMP skipped29 updates, leaving72967 completed updates. The packet's strict full-fit equality/zero-skip predicate then rejected the fixed epoch0 checkpoint before training PLE or exporting predictions. Downloaded control parameters are finite. There is no paired scientific result, and this outcome does not refute either representation.

Prepare an identified FP32 successor that independently trains both arms on the same frozen subset, order, objective, normalization policy and fixed epoch. Do not relabel the previous FP16 control as a matched FP32 control. Profile the actual FP32 path with symmetric gradient diagnostics, require complete finite-gradient evidence and exact update counts, and project both fits and exports inside the bounded cohort deadline before admission. Include diagnostic overhead in measured throughput. AMP is disabled in this treatment; an unused configured initial scale is not active AMP evidence.

Preserve V6 source, contracts, logs, checkpoint and all actual skip counts. The full-fit zero-skip requirement belongs to this declared comparison, not a universal validity or arm-rejection rule. Other FP16 experiments may explicitly permit dynamic-scaler skips and report unequal completed updates and their limits. A finite checkpoint alone does not establish a finite loss/gradient history or competitive score.

Keep resource admission, numerical evidence and scientific allocation separate. Parameter and prediction finiteness checks, official scoring and exposed-membership reporting remain binding. Root must independently review the new recipe, changed evidence path and source hashes before an identified cloud push. This decision grants no additional spending or unrelated-instance access.

Evidence: [downloaded control check](../../research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/E540_V6_ROOT_DOWNLOADED_CONTROL_CHECK_20261008_R1.json); raw V6 artifacts are under the campaign's `continuous_execution_E529/downloads/e540_v6` directory.
