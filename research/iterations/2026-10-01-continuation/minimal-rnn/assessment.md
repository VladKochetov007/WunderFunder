# Reserve architecture: positive-state minGRU256×2

**Feasible as a separately allocated architecture/cost experiment; not yet a training-ready or native-serving replacement.** The isolated CPU prototype passes the recurrence, gradient and state-continuity checks below. It keeps the existing211→256 input projection/GELU and512→256→output head, and replaces only the two recurrent layers. No feature, residual, convolution, layer-normalization or forecast-recipe changes are bundled with it.

The [primary paper, §3 and AppendixB](https://arxiv.org/html/2410.01201v3) removes previous-state dependence from gates/candidates. We use its positive candidate: `g(v)=v+.5` for nonnegative v, sigmoid otherwise, with input-only sigmoid update gate. Its §4.1 speed comparison uses plain-PyTorch recurrent implementations, not our cuDNN baseline. We do not transfer those speedups. Two implementation ambiguities matter: Algorithm6's affine transform of `-x` differs from negating its output when bias exists; Listing8 applies `log_g` to the supplied initial state. Our implementation follows the stated recurrence, uses `-gate_logits`, and carries the actual h0 unchanged, including exact zero reset. These observations concern the published pseudocode, not evidence of forecasting benefit.

`minimal_rnn.py` uses blocks64 and an associative log-domain affine scan. Pair composition is `(a2*a1, b2+a2*b1)`; logaddexp evaluates the positive sum. This avoids subtracting large cumulative log totals. The incoming state contributes linearly, so zero h0 never requires log(0), and gradients with respect to zero h0 remain defined. Carry stays FP32 and is not detached inside the model. Arbitrary external chunk boundaries therefore preserve the mathematical recurrence; tiny floating-point differences are expected.

| Member | Serving parameters | Dense MAC/row | State floats |
|---|---:|---:|---:|
| GRU128×2 |258,434|256,640|256|
| GRU256×1 student |580,866|578,816|256|
| **minGRU256×2** |**449,282**|**447,744**|**512**|
| GRU256×2 |975,618|972,032|512|

For hidden H, L minimal layers, input D and output O, parameters are `(2L+2)H²+(D+O+2+2L)H+O`, dense MACs `(2L+2)H²+(D+O)H`. The table uses D211/O2 and excludes auxiliary outputs. Training O23 gives454,679 parameters. The candidate cuts54.0% dense MAC versus d256 and22.6% versus the shallow student, while preserving512 recurrent state floats. It is still1.74× an h128 member. Gate nonlinearities, state arithmetic, feature construction and callback overhead are excluded; no time-limit qualification follows from these counts.

Generated CPU checks passed45 comparisons: length2000 versus independent FP64 recurrence, FP32 streaming, positive/zero h0, irregular splits/prefixes, future perturbation, gates±80, logits/state VJPs and complete-model parameter/input VJPs. Largest absolute discrepancy was1.53e−5. Full h256×2 B1/T2000 scan took0.088s and Python streaming0.862s on one CPU; these are descriptive checks, not a native/CUDA benchmark. The wrapper's23-output factory, state shape and `inp,cells` freeze passed; only metadata fields were added to its configuration. See `generated-01/`.

The important risks are reduced recurrent expressivity (no hidden-to-hidden gate/candidate mixing), positive-state bias and changed optimization/initialization. Stacking makes the second layer's gates depend on a contextual first-layer input, but does not restore a full GRU. The blocked implementation stores autograd intermediates and can cost more training memory than cuDNN. Current finite/nonnegative state checks cause CUDA synchronization; current `step` is FP32 whereas an autocast forward can project gates in FP16. **CPU parity qualifies neither AMP nor native parity.** AMP agreement and deployment integration remain unresolved before a forecasting fit; the later isolated FP32 native prototype below does not establish either.

At B32/T2000/H256, one full FP32 hidden-shaped tensor is62.5MiB. Six doubling levels per block retain several such tensors per layer; a conservative source estimate is roughly2–3GiB of saved recurrent activations across two layers, plus projections/head/gradients, full20k resident inputs, prefetched raw data and an EMA forward. This is an estimate, not a measured CUDA peak or an upper bound. A single fixed B32 generated GPU probe was allocated to measure this envelope, without an automatic smaller-batch fallback; its actual result is recorded below.


## Generated GPU feasibility result, 1 October

The fixed B32/T2000/H256×2 probe completed on GPU7 in 7.84 seconds: warmup 2.349 s, timed steps 0.309/0.301 s, all tensors finite and GradScaler unchanged. Peak allocated memory was 4078.62 MiB, reserved 4222 MiB, observed NVML 4441 MiB; host RSS 1.65 GiB. The probe included resident raw/prefetch buffers, fused AdamW, EMA and auxiliary synthetic losses. See `gpu7-generated-01/`. The prior GPU3 attempt was rejected before execution because another user's job occupied the device; it allocated no GPU work.

This establishes feasibility within the allocated memory envelope, not AMP/stream equivalence or forecast quality. Its autograd scan is slower than our cuDNN GRU student, so the paper's training-speed premise does not transfer to this implementation. No forecast fit or native port is allocated while the promising shallow GRU is being completed.


## Isolated native FP32 core

A separate C ABI prototype now matches the actual Torch FP32 recurrence in80 generated checks: ordinary maximum absolute error3.58e-7, extreme-gate error1.53e-5; native reset/split/future-prefix and23-to-2output export comparisons are byte-identical. Source copies and all generated inputs/results are local+SSH under `native-prototype/`; no active or closed serving engine changed.

Its measured cost is80.15 microseconds per row inside the C++ loop,94.07 through a per-row ctypes call, for one minGRU256x2 member. This excludes raw112-to-211 feature construction, submission ABI and any other ensemble members. It uses FP32 weights/exact activations, so comparing it directly with the production INT8 ensemble would confound architecture, precision and invocation boundary. There is no measured end-to-end serving advantage, forecasting-quality result or AMP equivalence result. No forecasting allocation follows automatically from this feasibility check.

## Direct affine scan reserve: failed numerical qualification

The separate [direct-scan prototype](direct-scan-prototype/README.md) completed its one fixed CPU matrix:186/194 comparisons passed. Full-model ordinary cases and all complete-model VJPs passed, but slowly decaying nonzero states at gate−17 failed at T2000 and20000. At20,000 rows, direct scan differs by0.00729 from FP64 and0.01431 from the literal FP32 recurrence. This includes coefficient rounding and an additional accumulation-order effect; switching from logarithms to positive affine composition did not solve the long-memory numerical problem. No GPU or forecasting promotion follows.

Two other failures isolate an existing exact-zero boundary discrepancy in the original `log_g`: its observed candidate derivative is0 at v=0, while `positive_g` chooses derivative1. The original generated45-check result did not cover this exact boundary. Source and tolerances remain unchanged; the scalar explanation is retained separately from the original result. A future training implementation must deliberately reconcile this branch convention and validate AMP/stream parity. No real-model frequency or forecast impact was measured.
