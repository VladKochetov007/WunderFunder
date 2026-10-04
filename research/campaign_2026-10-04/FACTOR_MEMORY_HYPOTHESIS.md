# Wider state with cheaper mixing: mechanism to test

This is a mathematical rationale and proposed interpretation, not a predictive result.

A GRU retains a diagonal carry path: h_new = z * h_old + (1-z) * n. Its state Jacobian contains diag(z), in addition to terms involving gate/candidate derivatives and recurrent mixing. Low-rank recurrent matrices therefore do not force the entire state transition Jacobian to have the same low rank. They reduce learned cross-coordinate mixing while retaining384 independently carried state coordinates. This does not guarantee long memory, identifiability, or better forecasts.

The native factor backend now implements exact saved U/V inference and passes full-stream parity. It makes the separate question experimentally accessible: can384 recurrent coordinates with rank128 mixing improve the score/runtime trade-off relative to256 dense coordinates? The useful comparison is against both a dense384 continuation from the same parent and a parameter-budget matched dense model; the latter is not initialization-matched.

For the inspected211-input,23-output,2-layer GRU with linear projection and concatenated projection/GRU readout, dense parameter count is14H²+248H+23. Width384 has2,159,639 parameters. Replacing four3H×H recurrent matrices by rank128 factors gives1,176,599 parameters. Dense281 has1,175,165 (−0.122%); mean/std buffers are excluded from learned counts. Timing must slice to the two scored outputs and report actual storage, not derive runtime from parameter counts.

Prespecified next contrasts remain: zero-update SVD versus parent as approximation context; same-parent dense/factor continuations with fixed epoch and commonFP32 inference; dense281 as separate equal-budget architecture control; measured full-callback CPU costs. No outcome on three E409 runtime streams can replace full component-based predictive evaluation.
