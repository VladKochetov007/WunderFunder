# Independent SCALARPLE01 specification checks

2026-10-01, architecture scout. This note and `scalar-spec-oracle.py` were written from `scalar-input-proposal.json`, `scalar/spec.md`, and `amendment-02.json`, **without reading scalar_model.py, custom autograd, trainer, or native adapter implementation**. Only generated CPU values were used. No competition observations, GPU, target, metric, quantile estimation, or new mechanism was used. The oracle result is a self-check of the independent reference, **not an implementation approval**.

Bound specification SHA256: `c1d45b3c36279c0c8697b17982cf3ba45fb1691d32a796ede37d9621d4c549f5`. Proposal: `f03523c237f5df00791cc101e31b007810fac63c0271c2d03a05174807f7d99e`. Amendment02: `b7a0c5d2c33d84122500663f4d0320cd3a597a90cbe387331cbb63ac36022303`.

## Finding and scope

The interpolation and first-order gradient contract is implementable as written. Twenty generated checks pass in the independent stdlib oracle. The useful remaining review targets are preserving its exact operation order, dispatching custom-autograd storage correctly, and keeping the knot/count/table checkpoint tuple intact. No mechanism or numerical tuning is proposed.

The initial baseline claim should mean equal finite numerical outputs, ordinary gradients, and recurrent state on the supplied fixtures. Universal byte identity is too strong: IEEE `float32(-0 + +0)` is positive zero. The oracle records baseline bits `0x80000000` and result bits `0x00000000` with numerical equality. This is a claim boundary, not a request to special-case zero residuals. Nonfinite inputs are already rejected; extreme finite tables can still overflow the specified `b-a` operation, so zero projection alone does not protect a branch that has generated infinity/NaN. The trained finite-model gates must remain in place.

## Adversarial cases for implementation review

| Case | Required observation |
| --- | --- |
| All 16 proposed knots collapse after float32 conversion | `count=1`; direct selected vector; fraction zero; one gradient contribution, not double counting the shared lower/upper index. |
| Quantile values differ in float64 but round to the same float32 | Collapse after conversion, canonicalize signed zero first, and pad with the last valid knot. |
| Exact interior knot and last knot | Right tie selects the interior knot as lower; last knot uses a direct return. Padding never changes lookup. |
| Adjacent float32 normal/subnormal knots | Retain strictly positive float64 gaps. Do not use an epsilon merge or float32 reciprocal that overflows. CPU subnormal claim assumes FTZ/DAZ off. |
| Knots `[-F32_MAX,+F32_MAX]`, input zero | Float64 denominator is finite and fraction is exactly 0.5; float32 subtraction would overflow. |
| Strictly interior input in a huge span | Fraction can round to exactly zero or one. Do not assert strict interior fraction bounds or infer index from rounded fraction. |
| Feature values `[2^24,1,-2^24,1,0,...]` | Ordered float32 sum is 1; real-number sum is 2. Tree/double reduction changes the declared function. |
| Many samples in one interval and count-one features | Scatter accumulates every contribution; unused padding receives zero gradient. Compare parallel GPU scatter with an appropriate tolerance, not CPU accumulation bytes. |
| Nonzero table derivative with upstream negative values | `left=g-round32(g*f)` differs from `round32(g*round32(1-f))`; use the specified order. |
| Nonzero table entries with multiplication/addition cancellation | Contracting the product and addition changes the scalar result; native code must preserve separate roundings. |

The oracle exposes `locate`, `interpolate`, `encode`, and `table_backward` for independent implementation comparisons. `encode` returns a 16-vector and the 44 `(lower,upper,fraction)` tuples. `table_backward` consumes batches of those tuples and upstream 16-vectors, applies the frozen normalization scale, and performs a literal sample/feature/coordinate ordered scatter. It implements the declared first-order surrogate, not a derivative of floating-point rounding. Finite-difference claims should use benign values and suitable tolerance; no input/knot gradient is specified.

Concrete stored witnesses:

- `g=-10.973383903503418, f=0.022747419774532318`: declared left contribution `-10.723767280578613`, rewritten contribution `-10.72376823425293`.
- `a=-92.94637298583984, b=30.59819221496582, f=0.632328987121582`: separate result `-14.8255615234375`, contracted product/add result `-14.82556438446045`.
- Finite table endpoints `[-F32_MAX,+F32_MAX]` at fraction 0.5 produce positive infinity under the required three-operation interpolation. This is a finite-domain limitation of the specified arithmetic, not a recommendation to replace it.

## Storage, autograd, optimizer, and AMP traps

PyTorch runs a custom `autograd.Function.forward` without recording its internal operations. Consequently, consulting `torch.is_grad_enabled()` **inside** `forward` to decide whether to save indices is not a valid way to distinguish training from an outer no-grad call. The training/no-grad dispatch decision should be carried from the enclosing call or handled by an equivalent verified route. Conversely, a parameter can still have `requires_grad=True` under outer no-grad, so that property alone does not establish that saved lookup state is needed. This is a source-review target, not an assertion about unread implementation. [Official PyTorch custom-function tutorial](https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html).

Initialization must leave the global CPU and every relevant CUDA RNG unchanged after the ordinary model is constructed. A standalone zero parameter and a private CPU generator can satisfy this. A randomly constructed layer followed by zeroing cannot. Counts must not affect draw count, including unused padded table rows. EMA construction and checkpoint loading must not inadvertently redraw or substitute tables.

The specified residual cast before addition is necessary to preserve the ordinary AMP dtype path. Record actual tensor dtypes and backend TF32 policy; FP32 storage alone does not imply full-mantissa GPU matrix multiplication. The scalar oracle intentionally makes no GPU projection or whole-model bitidentity claim.

At the initial zero projection, table gradients are zero, while the projection can have nonzero gradient on a nondegenerate fixture. With AdamW, a zero gradient tensor can still permit weight decay to alter table values; do not assert first-step table immobility merely from zero gradient. Also, the additional projection gradient contributes to global gradient clipping. If clipping engages, ordinary parameter **updates** may differ from the control even when their initial unclipped gradients match. The specification promises initial gradient parity, not universal first-update parity. These are expected consequences of the fixed ordinary optimizer, not grounds to change it.

## Immutable state and causality

Treat padded knots, integer counts, selected-feature order, table, projection, and fit hash as one checkpoint schema. A count mismatch can silently route values into a padded row while all tensor shapes remain plausible. Revalidate sorted unique valid knots, positive float64 gaps, canonical zeros, repeated-last padding, counts in 1..16, dtype/shape, and fit identity on every load/export. The oracle rejects representative count/padding mismatches.

EMA must copy/preserve immutable knots and counts exactly; they are not floating model parameters to average. H2 must retain the parent's active transform while freezing all adapter parameters. A loader must not pair EMA table/projection with newly fitted knots or constructor-default counts. Strict key loading alone does not prove semantic knot/count consistency or fit provenance; bind the arrays/hash tuple explicitly.

With fixed training-only knots and pointwise lookup, the adapter adds no temporal state. Prefix/reset/chunk tests should compare the same already-loaded tuple and raw-before-normalization input. Fitting or adapting knots from an inference sequence, batch-dependent binning, applying lookup after normalization, or mutating buffers during evaluation would break the specified semantics. No such behavior has been inspected or alleged here.

## Receipt and handoff

`scalar-spec-oracle-result.json` records all 20 generated checks, concrete witnesses, the float32 scale bits, and bound source/spec hashes. Runtime is below one second on local CPU. No claim of native, CUDA, checkpoint-loader, or full-model implementation correctness is made. The independent delivery critic can now compare the implementation against this reference without changing the experiment.
