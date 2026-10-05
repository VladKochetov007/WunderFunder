# Separate numerical reproduction of E477 and E480

Root implemented `../reproduce_activation_exports.py` without importing either scientific grader or campaign metric helpers. It uses additive raw sums around fixed shared anchors, rather than the graders’ component centering and pairwise merges. Registered draws are regenerated from seeds and numeric-sorted component IDs; the E480 exported draws match exactly.

All12 E477 target/global scores and18 global contrasts, and all24 E480 target/global scores and44 global contrasts reproduce. Maximum discrepancy is3.39e-15. Both endpoints of each95% interval, bootstrap SD and nonpositive-tail fraction are checked. This is numerical reproduction on existing exports, not an independent data cohort, checkpoint inference replication or public confirmation. E477’s target-specific contrast intervals are outside this particular checker’s scope.

A first helper invocation failed before opening any scientific artifact because systemd defaulted to `/home/vlad`; an explicit working directory corrected the invocation. Successful source is hash-bound in activation_export_reproduction.json. The bounded successful helper used one CPU,31.1MB peak RAM and no GPU; runtime0.655s. No fit was rerun.
