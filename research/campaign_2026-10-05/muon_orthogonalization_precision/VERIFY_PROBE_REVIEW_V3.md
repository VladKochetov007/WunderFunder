# Independent review: E487 saved-delta verifier v3

Reviewed `verify_probe_v3.py` SHA-256
`eda4e2dd94568964389940bb5bd6753f71dbb5533f738f4b2381f77071b86a29` and
`verify_probe_tests_v3.py` SHA-256
`0e98cf8e076ef0921c84da83a97ce07dafe87820769b2be1743d361c39ab6691`.
No real E487 result or saved-delta archive was opened by this review, and the
verifier was not run on those actual artifacts.

The v3 delta reader preserves the NPZ float32 dtype for subtraction, then
widens for RMS accumulation, matching the producer's update-metric arithmetic.
The synthetic float32 fixture exercises a case where subtract-before-widen
differs from widening both operands first. V3 also requires a nonempty and
identical `parameter_sha256_before` for all four crossed replay cells, which
binds the paired update comparison to one starting snapshot. The other fixed
checks remain: exact 4+5 archive inventory, finite arrays, recomputed strict
matrix/family gates and BF16/FP32 sensitivities, same-gradient receipts,
frozen-E474 shared delta equality, and preservation of expected numerical
gate failures as diagnostic outcomes.

Bounded synthetic validation passed independently under `scripts/limited`
with one CPU and 2 GiB: all six tests passed; Ruff and `py_compile` passed.
This is the arithmetic verifier only. The controller must still validate the
sealed E487 probe source/spec, successful attempt, raw-fit provenance,
resource/teardown receipts and output hashes before interpreting the
independent arithmetic result. It does not validate actual gradient tensors,
which are intentionally not exported; it checks their producer receipts and
recomputes all nine exported delta cells.

**GO for the controller to run v3 once on the sealed E487 result/archive.**
Do not use v1 or v2 for the actual verifier run. A verifier pass will establish
that the persisted delta arithmetic agrees with the producer receipts; it
does not by itself admit any training experiment or change E485's failed
status.
