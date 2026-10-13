# Activation-aware QAT profile readiness review R1

## Verdict

**No-go for using the reviewed profiler as an actual-data or fit-admission
profile.** It is a useful synthetic `torch.compile(fullgraph=True)` forward and
backward smoke at a nominal sequence shape. It does not exercise the frozen
CP5 parent, actual fitting inputs or training objective, an optimizer update,
or QAT-cache refresh. A root-admitted synthetic compiler smoke can be run and
reported under that limited label, but it cannot clear the activation-aware
fit.

This review covers actor source revision
`2a5c9c89b040b7a2e52a97ae1d286dc056aa07fc`, whose implementation commit is
`7939e64c0a14cba9e0c1eaaf5822787c89dd54bf` on base
`af49c2fc1fa4f02cf4bdfa64f4268236b0478fd4`. It does not approve a training
recipe, GPU resource admission, score, or quality claim. No GPU, Torch,
checkpoint tensor, or prediction run was performed by this reviewer.

## Evidence and technical findings

The finite-input quantizer uses nearest-even rounding and signed symmetric
codes `[-127,127]`, computes a per-final-axis-vector scale, and returns zero
codes/scale for all-zero vectors. This matches the finite, nonzero-vector rule
in `native/src/vnni.rs`. The scan uses PyTorch's `[reset, update, new]` gate
order and candidate `input_new + reset * recurrent_new`, including the
recurrent candidate bias; its update mixes the unquantized FP32 prior state.
The first and later-layer GRU inputs and each recurrent previous-state operand
are quantized at the intended GRU matvec boundaries. This keeps the recurrent
state FP32. The reported native fixture is deliberately limited: zero input
projection isolates recurrent A8, and the four actor tests include fused-GRU
output/state/gradient parity with A8 disabled, scalar code/scale checks,
FP32-state checks, and a small controlled native-Q8 state comparison. The
receipt's native error and A8 sensitivity support that fixture only.

The profile is not the required actual-shape training profile. Its defaults
instantiate a randomly initialized 112-input model and random `[16,1000,112]`
tensor. The frozen deployment manifest instead binds `inp.weight` to shape
`[256,211]` and `mean`/`std` to `[211]` (manifest SHA256
`8894af708dcba1500fc8c1b431d30468a29a9999531dd7133ec982c3b1d19556`). Its
objective is mean squared predictions plus a small final-state term, rather
than the frozen fit's distillation and auxiliary losses. It performs backward
repeatedly but never calls `optimizer.step()` or `invalidate_qat_cache`; thus
compile and backward can be exercised, but actual update memory, step cost,
and represented-weight refresh are not measured. The receipt has no data,
checkpoint, cache, or fit-recipe hashes. The actor's profile script does not
load actual data or model parameters.

The extracted `qat_weight_replacements()` reuses the existing fixed and
learned-scale substitution branches, so the scan can consume the configured
represented matrices. However, this profile configures only fixed scales;
compiled interaction with learned scales has not been tested. The new wrapper
is not connected to `scripts/train_rnn.py` or its training loop, so fit
integration, checkpoint/resume, optimizer updates and cache invalidation still
need a separately reviewed implementation before a fit can be launched.

One finite-domain edge remains explicit: `native::vnni::quantize` returns zero
codes/scale if the maximum is zero or non-finite and maps NaN entries according
to its scalar `max/min` behavior when the vector maximum is finite. The Python
implementation's `amax` propagates NaN and its `fmax/fmin` path can emit
`-127` codes for NaN/Inf positions at zero scale; the identity STE can also
produce NaNs on non-finite values. The docstring restricts the formula to
finite vectors, but the public forward does not reject non-finite inputs or
states. A real-batch finite preflight and an explicit finite-domain guard (or a
separately tested native-equivalent non-finite policy) are required before
calling the implementation native-style beyond its finite domain.

Finally, the new implementation row in `research/experiments.jsonl` reuses
`E-QAT-DEPLOYMENT-20261013-R1`, which already names the W4 deployment comparison
under `H-QAT-W4-20261013`, then records activation-aware work under
`H-QAT-A8-20261013`. Keep the historical row unchanged and append a distinct
activation-aware experiment ID; do not silently reparent the deployment ID.

## Requirements for the next profile gate

Before using a profile to admit an actual fit, bind it to the frozen CP5
checkpoint/model and normalization, use an actual bounded B16-by-1000 fitting
batch from the frozen cache/loader, and run the same distillation plus
auxiliary loss as the fit. Perform at least one disposable AdamW update,
invalidate the QAT cache afterward, and verify finite gradients plus a changed
parameter/update counter. Record compile/first-step, warmed forward/backward
and update timings, CUDA allocated/reserved peaks, and SHA256 identities for
the code, parent checkpoint, data/cache and recipe. Validate finite inputs and
states before entering the compiled recurrence. This is still a profile only;
the activation-aware path must also be integrated into the actual trainer and
reviewed before fitting. Root retains resource admission and any fit/quality
decision.

## Source bindings

- Actor implementation and focused-test receipt: [activation-aware backend](../../campaign_2026-10-13/qat_renewable_r1/ACTIVATION_AWARE_BACKEND_R1.md) and [CPU/native receipt](../../campaign_2026-10-13/qat_renewable_r1/ACTIVATION_AWARE_CPU_NATIVE_RECEIPT_R1.json).
- Reviewed source SHA256 values and machine-readable findings: [review receipt](receipts/QAT_ACTIVATION_AWARE_PROFILE_REVIEW_R1.json).
- Native A8 implementation: `native/src/vnni.rs`; native dynamic GRU path: `native/src/rnn_fast.rs`.
- Related hypothesis: `H-QAT-A8-20261013`. The W4 deployment experiment remains `E-QAT-DEPLOYMENT-20261013-R1`; activation-aware work needs its own experiment ID.
