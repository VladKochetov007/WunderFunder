# E-518 independent pre-score review

- Reviewer: independent native subagent, `gpt-6-luna` with max reasoning.
- Scope: implementation and preregistration alignment only; no E-518 execution or result inspection.
- Decision: **PASS** after correction.

The corrected selector computes the global maximum across all 101 calibration
scores and chooses the first beta whose score is within `1e-12` of that maximum.
It no longer compares against a moving threshold. The zero-target-weight branch
returns the declared `(beta, score)` tuple `(0.0, 0.0)`. Focused regression tests
cover both cases.

The reviewer confirmed that implementation and registration agree on the
candidate, target-0 gate and global controls, 101-point beta grid, 75th-percentile
linear threshold with ties routed, clipping and float32 scoring, warmup and
scored mask, disjoint whole-chain folds, official endpoint, bootstrap seed and
replicates, and pass rule. The target-0 fitting objective is equivalent to the
two-target mean because target 1 is unchanged. Focused tests passed: 4/4 in
0.13 seconds under the bounded CPU/memory wrapper.

This review admits only the one registered E-518 exposed-pool screen. It gives
no numerical prior, public-transfer claim, or permission for additional
same-pool tuning.
