# Finite-aware soft-route gradient diagnostics

The passed v2 diagnostics source is preserved verbatim under
`passed_v2_source/`; its recorded SHA-256 is
`1337b008ef1e352be9232a023e099a25b5d0017709611d7d2ae89d0d5c0d8822`.
The active diagnostics module adds only finite-aware accounting. It does not
change model outputs, gradients, optimizer steps, losses, or trainer code.

For each expert and optimizer attempt, the module checks the expert GRU and
head gradients before computing their norm. If any element is nonfinite, it
increments `expert_prediction_gradient_nonfinite_steps` and excludes that
expert's whole attempt from the finite norm sum and denominator. It reports
the denominator as `expert_prediction_gradient_finite_steps`; if no finite
gradient attempt exists, the corresponding mean is `null`. It still records
the full optimizer attempt and measures the actual post-call parameter delta.
For a skipped optimizer step, unchanged parameters contribute zero movement.

These are gradients of the complete training objective flowing through the
expert predictions: main supervision, enabled auxiliary supervision,
distillation, and EMA consistency. They are measured after GradScaler
unscaling and the trainer's gradient-dynamics/clipping hook, immediately
before the optimizer call. They are not isolated main-label gradients. AMP
skip and completed-update counts stay in the trainer's existing
`EpochUpdateCounter` fields `amp_skipped_updates` and
`optimizer_completed_updates`; this diagnostic does not replace or normalize
those counters.

`tests/test_e425_diagnostics_finite.py` injects one infinite expert gradient
while route probabilities remain finite, then simulates a skipped optimizer
call by keeping model parameters unchanged. It checks finite-step denominators,
nonfinite-step counts, null for the expert with no valid gradient norm,
finite means for the other experts, unchanged gradient data, one optimizer
attempt, and zero measured parameter movement. The test is prepared for the
root's bounded CPU run and was not run locally.
