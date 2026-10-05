# Independent review: E487 saved-delta verifier v1

Reviewed source SHA-256 `f01a76bf542dc54666bc9ec15dee694fb53f6d5ebe282d835d72bdfb40bf6180`
and tests SHA-256 `0c59cd3a13e081c8010a9bd74e00a63e8ae41d8ee1f5ae0fc7ef016c39e8738c`.
This review did not run the verifier on the real probe result or open its
saved-delta archive.

The bounded synthetic suite passed (4 tests), and Ruff passed under a 1-CPU,
2-GiB `scripts/limited` scope. The checker recomputes all nine delta cells,
tests finite arrays and inventories, preserves expected strict-gate failures,
and verifies shared-gradient/clipped-update receipt consistency. It is an
arithmetic verifier, not a provider-attempt/source admission gate; the
controller must separately validate sealed source hashes, fit-only loader,
actual run, resource caps, and teardown as planned.

**NO-GO for real verifier execution on v1** because:

1. `_npz_cell()` widens saved float32 deltas to float64 before the verifier
   subtracts candidate/reference or BF16/FP32 arrays. The probe computes those
   subtractions on float32 tensors, then widens for RMS accumulation. The
   verifier's nested-metric tolerance (`rel=2e-13`, `abs=2e-15`) is too tight
   to safely treat these different arithmetic paths as identical; the current
   float64 synthetic fixtures miss the rounding case.
2. The verifier does not compare the `parameter_sha256_before` receipts across
   the crossed replays. Recomputed delta differences only establish the
   registered update comparison if those cells start from the same snapshot;
   require the four crossed starting-state hashes to match before interpreting
   their gate results. This is distinct from the controller's source and
   attempt validation.

Preserve v1 and its synthetic result as the first review attempt. A new
version should preserve NPZ dtype through subtraction and widen only for
accumulation, add a float32-rounding fixture, and require equal crossed
pre-step state hashes. No real verifier run was performed.
