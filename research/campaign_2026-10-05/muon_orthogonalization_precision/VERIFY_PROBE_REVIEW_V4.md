# Independent review: E487 saved-delta verifier v4

Reviewed `verify_probe_v4.py` SHA-256
`f717d0d30d1c6e1456b80bdd31d76896633a588ea22e5b9243652af5f92391e6` and
`verify_probe_tests_v4.py` SHA-256
`9f6e4f2cda9f0300420af84c4bb36cef80187bc0665c4f924deae153b6af68c7`.
No actual E487 result/archive was opened or passed to the verifier.

The v4-only change correctly matches `_jsonable_shared_control()`'s fixed
three-key schema for both precision maps. BF16 requires `native`, `custom`,
and a real `frozen_e474` receipt; FP32 requires the same exact key inventory
but insists `frozen_e474` is `null` and skips only that marker. It still
requires the five actual shared-control delta arrays exactly once; the NPZ
inventory remains 4 crossed + 5 shared arrays. This addresses the preserved
v3 serializer/schema mismatch without weakening replay validation.

V4 retains v3's two arithmetic/provenance fixes: it preserves float32 through
delta subtraction before widening for RMS, with a synthetic rounding test;
and it requires all four crossed replay receipts to carry the same nonempty
pre-step parameter hash, with a mismatch rejection test. The strict numeric
gate and expected-failure status logic are unchanged. Bounded independent
validation under `scripts/limited` at 1 CPU/2 GiB passed all seven tests; Ruff
and `py_compile` passed.

**GO to run v4 once on the already sealed E487 result and NPZ.** Preserve v3's
actual failure as the schema-mismatch incident; do not alter the probe or
retry it. This GO covers only the independent arithmetic verification. The
controller must still bind the actual artifact hashes to the sealed attempt
and confirm its existing source/resource/teardown receipt before interpreting
the verifier output. The verifier does not consume test labels, and this
review did not read evaluation outputs.
