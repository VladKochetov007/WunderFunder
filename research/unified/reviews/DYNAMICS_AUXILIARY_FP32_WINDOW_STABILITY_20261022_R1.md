# Dynamics auxiliary FP32 window stability

A bounded CPU numerical probe found that the dynamics target builder's global
FP32 energy prefix lost small changes after an earlier large pulse. In a
20,000-row fixture with a 1,000-unit early pulse and later alternating
0/0.01 values, the expected log mean squared change at row 16,000 was
`-9.2102404`; the original prefix-subtraction path returned `-18.420681` for
both configured horizons. The source observation and numeric values are in
[`DYNAMICS_NUMERICAL_PROBE_R1.json`](../../campaign_2026-10-21/next_cycle_r1/DYNAMICS_NUMERICAL_PROBE_R1.json).

The code amendment retains the existing FP32 normalized input, feature-group
definition, 16/256-step targets, epsilon placement, and validity masks. Signed
mean changes use the normalized endpoint difference. Energy uses per-step
group mean-square changes and positive block-local prefix/suffix sums, so a
window never subtracts a distant cumulative prefix. The target formula and
training loss are unchanged. New tests compare the long-pulse case and
non-divisible/block-boundary windows against direct float64 window oracles,
including compressed-length masks. Root owns test execution and the source
quality checks; this note records no fit, GPU profile, or quality result.
