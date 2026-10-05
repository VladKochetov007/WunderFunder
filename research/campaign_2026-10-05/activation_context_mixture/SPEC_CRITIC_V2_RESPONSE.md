# E484 design v2 critique response

Reviewed design v2 SHA256: `e2657b85e31c10d0bab3edab066ba71faa6f2eee79c5cddfd9aff86c6f89b398`.

The v2 design resolves the earlier context ambiguity: `z_t` is now explicitly the existing activated projection `GELU(inp(x_normalized))`, with no added projection. That matches the proposed context source and preserves the current encoded recurrent input.

Correction to my v1 critique: pure GELU is a distinct zero-added-parameter anchor, so it does **not** initially match the mixture. The five mixture arms—fixed 50/50, learned-static, recurrent-context, projection-context, and head-self-context—must produce the same initial mixture output. All six arms must share identical baseline parameter bytes and RNG/order; the GELU arm is the named activation anchor. Learned/static/dynamic gates remain zero initialized without consuming RNG.

The source tests must use nonzero synthetic gate coefficients as well as zero-init checks. Otherwise zero-initialized gates equal 0.5 and can hide a wrong context source, detached context, or incorrect broadcast. For each dynamic source, verify the gate numerically against an independent calculation with nonzero coefficients; verify finite, nonzero gradients to the gate coefficients and through the context tensor. Run future-row and batch-element independence tests with these nonzero coefficients so the checks exercise the actual routing path. Retain the all-zero-context epsilon case and exact 64-channel RMS formula test.

Capture the FP32 backend policy metadata from inside the execution context; do not serialize restored defaults as the policy used for a fit. Keep the distinction that Muon CUDA orthogonalization still uses its frozen BF16 path. With these corrections recorded, the design conditions are resolved for implementation. This response does not inspect or approve implementation source.
