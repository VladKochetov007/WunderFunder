# E479 source and CPU preflight receipt

The implementation follows `E479_DESIGN.md`: a two-layer H64 reset-after GRU
with `n = tanh(v) + alpha_l * sin(pi * tanh(v))` and learned coefficients
`alpha_l = 0.2 * tanh(raw_alpha_l)`. The scripted recurrence receives each
coefficient as a tensor, so gradients remain connected. Learned scalars have a
dedicated AdamW group with zero weight decay; Muon remains restricted to the
four three-block GRU matrices.

`fit_recurrent_activation_adaptive.py` supports the complete registered 16-fit
recipe, and its only CPU preflight mode runs one update per cell. It loads only
the 24 fit-role groups. `gpu_probe.py` contains the actual-fit-data CUDA parity
and one-update probe for parent review; it was not launched for this task.

Synthetic CPU tests passed for the alpha-zero incumbent comparison (outputs,
states, input and parameter gradients, and clipped AdamW update), matched
fixed/learned initializations, chunk carry, reset and future isolation,
coefficient bounds and monotonicity, nonzero learned-alpha gradients, and
disjoint optimizer groups.

The bounded run `cpu_preflight_v4/fit_manifest.json` reports all 16 cells
passed on one fit-only B4×256 batch each. Every one of the eight learned cells
had finite nonzero gradients and actual updates for both layer coefficients.
The receipt records 61,058 fixed versus 61,060 learned parameters, the sampled
fit row groups, losses, and optimizer-family telemetry. No test rows were
loaded and no fit was completed.

Final source SHA-256 values:

| File | SHA-256 |
|---|---|
| `fit_recurrent_activation_adaptive.py` | `6a11025735bb8df7d88b08bdcf7dfd9d8879992a82c1c64d40b43ada7456c7bf` |
| `tests.py` | `88ce5a39aa8c8a8af869689f5394ba1eae4225c008aaef61e6679de592994b17` |
| `gpu_probe.py` | `293fe14c00cc7f76d13daa44f9a3e9f99fbdb1b083a5a0045a020c99e1047f39` |
