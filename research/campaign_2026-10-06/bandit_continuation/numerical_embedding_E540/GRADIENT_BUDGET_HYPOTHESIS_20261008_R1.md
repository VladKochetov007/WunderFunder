# Gradient budget as a representation interaction

Basis: actual E540 V8 linear-control epoch0 producer GradStats,72996finite updates on the declared512-component fit subset. Global clip1 activated94.92% of steps; median total norm5.11971, mean clipping factor0.289. Head gradients contributed93.23% of mean squared family norm, recurrent hidden/input families0.08%/0.31%. These are training diagnostics, not a validation score or evidence of a public gain.

Hypothesis: frequent global clipping coupled to a dominant output head may change the time pattern of representation updates. Test whether separating recurrent/input and head clipping budgets helps the representation learn; include the interaction with PLE rather than assuming a linear-control observation transfers.

Counterargument: Adam can largely cancel constant gradient rescaling through its moments. A high clipping rate or head norm share alone does not prove an optimization defect; family geometry, variable per-step scales, epsilon, decay and changing gradients matter. Longer ordinary training may explain the same observation. Preserve this competing explanation.

Next bounded contrast, after the V9 original paired endpoint lands: same fit groups, seed, initialization, sampling, objective, precision and fixed update count; compare globalclip1 with one declared grouped-clipping recipe (encoder/recurrent and head separate thresholds), alongside an ordinary-duration contrast if allocated. Record pre/post clipping family norms, optimizer-update norms and loss/official exposed-fold predictions. Freeze thresholds and budget before fitting; no threshold sweep on evaluated results, no numerical prior/public forecast manufactured. Check finite gradients and causal serving; do not change officialmetric or labels.

This arm is eligible exploration, not a reason to modify healthy H218 or repeat the existing control. Source: `kaggle_v8_terminal_20261008_r1/downloads/experiment/artifacts/runs/e540v8-linear-s540-fp32/provenance.json`; original SHAef342abcc11b94eabed66bd4d1e29009fc645079bb17bb95f5f63d83b7e82794.
