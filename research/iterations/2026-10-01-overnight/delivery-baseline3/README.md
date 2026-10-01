# Validated fixed3 fallback, pending final selection

Final artifact: `submissions/overnight_baseline3/baseline3_fixed.zip` (3,796,019 bytes).

SHA256: `9471821e81e7c3f80a088147512183b72e0ab52bc593a1136d2becc229d0d098`.

The ZIP contains root `solution.py`, three weight NPZs, `members.json`, and CPython3.11 x86_64 native binaries for v2/AVX2/v3. The fixed equal-weight composition is `fin128_s3h2`, `fin128_s4h2`, `finc500b_s3h2`, all exact final H2 epoch1 checkpoints. There is no wall-clock governor. Every exported float32 array, including only the first two final-head outputs, was proved byte-identical to its corresponding member in the archived V1 ZIP before packaging. Original checkpoints and ZIP were preserved; verified checkpoint copies live in the isolated checkout.

The raw packaging provenance is `baseline3_fixed.manifest.json`; its build-time status is retained unchanged. Completed validation is recorded in `baseline3_fixed.validation.json` and `delivery-receipt.json`. Frozen native binaries and wrapper hashes match the independently reviewed `critic/native_order_v2` manifest.

Fresh extracted-ZIP tests ran from an empty working directory with the scorer's Python3.11/glibc2.36 environment. Each ISA branch passed exact archived-native comparisons on 2,048 synthetic rows and 2,560 actual rows; v3 additionally matched all100,000 actual rows. Warmup returns None, subsequent requested predictions are two finite float32 values, reset/repeat and future-prefix tests pass, and false prediction flags still advance state. A harness import guard prohibited Torch. The packaged source is unchanged by tests; ISA discovery was mocked only inside the correctness harness.

Three fresh-process100,000-row complete-callback timings were67.87,68.63,66.92µs/row. The one-million-row trace took67.7116seconds, or67.71µs/row, projecting to44.46minutes for39.4million rows. All1.3million timed rows matched the archived reference; allruns passed the declared SMT sibling58 utilization bound of<5%, with observed aggregate siblingload1.17%,0.29%,0%,0.13%. Each process used one thread pinned toCPU18 on XeonGold6230. Maximum RSS was90,880KiB (88.75MiB), including the mapped rawfixture; import plus model initialization took0.0188–0.0195seconds. There were no noisy runtime retries.

The long trace cycles five actual20,000-row sequences ten times, with fresh sequence IDs and correct state resets. It is not a claim about a single uninterrupted million-row hidden state. Runtime qualification applies to v3 on this host; fallback ISA branches are correctness-tested but not qualified for the time limit, and actual public hardware runtime remains unmeasured. No targets, competition score, training, GitHub upload or public submission were used. This artifact is a baseline fallback pending the root agent's final selection, not a new SOTA claim.

An initial reference-harness whitespace SyntaxError occurred before imports or inference. Its original script and receipt are preserved as `make_reference.failed-01.txt` and `harness-failure-01.json`; the package was never changed. The corrected harness passed all checks.
