# E479 source review

**Decision: GO for the guarded real-device update probe on the current v4 source identity.** This is source admission only. I did not launch the CUDA probe, run the fit matrix, or inspect test-component scores.

The implementation matches the registered recurrence and model layout. The TorchScript scan splits GRU gates in PyTorch reset/update/candidate order and implements the reset-after hidden candidate bias as `tanh(input_candidate + reset * hidden_candidate)`. The state has shape `(2, batch, 64)`, each layer receives its own tensor-valued `alpha[layer]`, and both hidden states are carried across fit chunks and reset between batches. The learned coefficient remains on the autograd graph through `0.2 * tanh(raw_alpha)` and the scripted candidate map; no Python scalar conversion or detachment appears in the forward path. The inverse initialization for 0.1 is `atanh(0.5)`. The fixed arms add no parameters; the learned arms add exactly two.

The optimizer partitions are disjoint and cover every trainable parameter. Muon selects only the four stacked recurrent matrices with three gate blocks; learned `raw_alpha` is in its own zero-weight-decay AdamW group. The shared incumbent update function globally clips `model.parameters()`, which includes the learned scalars, before either optimizer steps. Every update checks parameter finiteness. The fixed-step telemetry records the raw alpha gradient and update as well as the layer coefficients.

The actor's synthetic suite passed all five groups: scalar-map bounds/monotonicity, alpha-zero output/state/input-gradient/parameter-gradient and clipped AdamW parity, fixed-versus-learned initialization parity, chunk carry/reset/causality, and optimizer parameter partitions with nonzero alpha gradients. I independently ran a bounded CPU check for the missing Muon parity case: the alpha-zero candidate matched the incumbent outputs/states, maximum parameter-gradient difference was `7.45e-9`, and post-global-clipping Muon update difference was `2.46e-7`. This verifies the same update path for Muon; the actual-device probe still needs to confirm CUDA execution and finite updates for every cell.

The current source and `cpu_preflight_v4/fit_manifest.json` agree on all three source hashes, both frozen parent-source hashes, and the dataset/cohort identities. The 16 one-update CPU receipt contains only 24 fit groups, reports `fit_only=true` and `test_rows_loaded=false`, and its selected B4 row groups match the seed-specific shuffle for every arm. Both seeds reuse the same order across all four conditions and both optimizers. Fixed/learned parameter counts are 61,058/61,060. The earlier v3 receipt records a prior trainer hash; the v4 receipt is the matching current one and should be used for sealing.

Validation: current trainer unit tests passed under a 1-CPU/2-GiB scope; Ruff and `py_compile` passed. The CPU receipt source/order audit passed. No full fit or GPU operation was run by this reviewer.

Reviewed hashes:

- Trainer: `6a11025735bb8df7d88b08bdcf7dfd9d8879992a82c1c64d40b43ada7456c7bf`
- Tests: `88ce5a39aa8c8a8af869689f5394ba1eae4225c008aaef61e6679de592994b17`
- CUDA probe: `293fe14c00cc7f76d13daa44f9a3e9f99fbdb1b083a5a0045a020c99e1047f39`
- Matching CPU preflight v4 receipt: `92fcb33ea416527dc742911605da9059eb1d7c728a4a594ed9e5a65dae5bbb66`
