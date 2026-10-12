# Independent review of causal moments and innovations R1

**Verdict: implementation review passes for the declared CPU/streaming paths; no blocking correctness defect was found.** The full-sequence recurrence correction addresses the recorded long-path parity failure. This review does not establish model quality, a GPU fit, or a production-hardware profile.

## Scope and exact sources

I reviewed the implementation introduced in `bae0e2ee40b532e3588653919753c86918f22fa4` and the recurrence correction in `ab97a9c30d17cd2519f9a392dfc5975ecf491dad`. The current reviewed file hashes are recorded in the accompanying JSON receipt. I traced feature construction, fitting-prior creation, parent input expansion, checkpoint metadata, batch prediction, one-row streaming, and ONNX export.

The fitting prior is formed from deterministic samples of source-qualified training membership (`train_idx` plus `extra_valid` where those groups are admitted for fitting), reads only `features`, and records its selected `(split,row_group)` members and membership digest. With a campaign split, `assert_fit_groups` checks the training pool against the declared fit members. Evaluation groups do not enter this prior through the reviewed path. The checkpoint carries the validated prior; prediction and export restore the same prior and control mode.

The feature calculation is causal: the innovation uses the previous slow moments, then the current projected row updates the fast/slow first and second moments. Prior shrinkage advances on the per-sequence row clock. Batch blocks carry the preceding shrunk moments across block boundaries, while the streaming graph resets them on the first-row flag and includes the current row before returning its feature. New channels are inserted as zero input weights with mean zero/std one when extending the own initializer, preserving its initial function. The `newest` control retains width while zeroing the intended suffix.

## Verification

Under `scripts/limited` with 1 CPU, a 2.0 GB memory cap, and a 120-second runtime bound, I ran:

```text
.venv/bin/python -m pytest -q tests/test_causal_moment_features.py tests/test_streaming_graph.py
14 passed in 4.73s
```

This includes the real 20,000-row supplied sequence batch-vs-NumPy check, future-perturbation causality checks, control-width and initializer-function-preservation checks, trainer input construction, Torch streaming/reset parity, and an ONNX state-input round trip. There were only ONNX deprecation/tracing warnings; no assertion failed.

I separately exercised the exact Torch per-row moment-state step used by `StreamingGraph` across the same 20,000-row training sequence, using group 0 only as the engineering prior fixture and group 1 as the sequence fixture. Relative to `research_features_numpy`, the maximum absolute error was `2.5510788e-5`; the maximum normalized tolerance ratio under `atol=5e-5, rtol=3e-4` was `0.14946`. This fills the serving-state-path coverage gap left by the stored R2 receipt, which reports batch-vs-NumPy and NumPy-stream-vs-NumPy parity. Both operations were CPU-only and no targets were read for feature construction.

The earlier [R1 receipt](../../INNOVATION_FULL_SEQUENCE_PARITY_R1.json) showed batch parity failure (`max_scaled=2.2179`) while NumPy streaming passed. The fix in `ab97a9c3` rounds the exponential coefficient to float32 before computing the block scan and streaming coefficients, so its complement follows the same float32 recurrence as the NumPy reference. The stored [R2 receipt](../../INNOVATION_FULL_SEQUENCE_PARITY_R2.json) reports full-sequence batch-vs-NumPy `max_scaled=0.4723` and batch-vs-stream `max_scaled=0.5216`; both remain below the unchanged tolerance.

## Limits and disposition

The numerical check covers one 20,000-row sequence and an engineering-only prior fixture. It does not cover every source-qualified component, GPU arithmetic, actual training batch/chunk memory, a model fit, or a local grade. The new representation adds 64 or 80 channels, so an actual-device profile remains necessary before a fit under the active resource contract. Treat the feature family as an untested research treatment until matched-quality evidence lands.

**Disposition:** pass implementation parity review for the checked CPU batch, Torch streaming, and ONNX round-trip paths. Preserve R1 as the failed numerical record and R2 as the corrected record. No fit, export of a candidate checkpoint, grade, or provider action was performed by this reviewer.
