# E485 native-vs-custom update gates (source-only preregistration)

Reviewed with the independent specification critic before probe edits. The
existing native/custom output, state and input/parameter-gradient tolerances
remain unchanged. The post-step check uses **deltas from the same pre-step
snapshots**, not absolute parameter values.

- For each of the four `gru.weight_*_l*` matrices separately, require
  `RMS(delta_custom - delta_native) / max(RMS(delta_native), 1e-7) <= 0.10`.
- For `inp`, `gru.bias_*`, and `head`, apply the same 10% RMS-difference limit
  to each element-weighted AdamW family aggregate; this keeps the GRU biases,
  which use AdamW, distinct from its four Muon matrices.
- For each GRU matrix, cap maximum absolute delta error at
  `0.5 * RMS(delta_native_matrix) + 1e-7`. For input/head parameters, cap each
  tensor's maximum absolute delta error at `0.5 * RMS(delta_native_family) +
  1e-7` for the AdamW families above.

The 10% bound limits update distortion relative to the operation being
compared; the per-matrix and per-family denominator avoids unstable
coordinate-wise relative error near zero. The absolute cap constrains isolated
coordinate outliers. These are fixed gates, not observed tolerances. No gate
may be widened after a CUDA result.

Separately clone identical models and optimizer states, assign identical
synthetic gradients with global norm greater than one, and compare the local
clip/update helper with frozen E474 `apply_optimizer_update`. Require byte-
exact parameters and optimizer-state tensors after the clipped MuonAdamW step.
This tests implementation equivalence, including the registered CUDA BF16
Muon orthogonalizer; it does not replace approximate native-GRU/custom-scan
parity.
