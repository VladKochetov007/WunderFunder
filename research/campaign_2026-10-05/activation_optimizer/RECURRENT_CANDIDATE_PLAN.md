# A bounded recurrent activation follow-up

Status: proposal, **not fitted**. E470 changes embedding and readout activations
only. It does not test different GRU candidate activations or justify closing
that direction.

Keep the GRU reset/update gates sigmoid and change only its candidate:

```
u = tanh(input_candidate + reset * recurrent_candidate)
candidate = u + alpha * sin(pi * u)
hidden = update * previous_hidden + (1 - update) * candidate
```

For fixed `alpha` in `{-0.1, 0, 0.1}`, the derivative with respect to `u` is
`1 + alpha*pi*cos(pi*u)`, at least `1 - 0.1*pi > 0`. The map is monotone on
[-1,1] and preserves its endpoints, so candidates remain in [-1,1]. With an
initial hidden state in that interval and sigmoid update gates, hidden values
also remain there. This is a bounded alteration of curvature and gradient
flow, not evidence of periodic market dynamics or globally stable gradients.

A later learned variant could constrain `alpha = 0.1*tanh(a)` and initialize
`a=0`; fixed/frozen-scalar controls would separate extra parameters from shape
adaptation. That variant must not replace the fixed comparison retrospectively.

First implementation gate: reproduce PyTorch's GRU equations at alpha zero,
including where the reset multiplier applies and both biases. Check forward,
chunk/state carry, reset, backward gradients and optimizer updates against a
matched implementation. Compare custom-backend alpha-zero controls, rather
than attributing a backend numerical difference to activation shape. Start
with three fixed alpha values × AdamW/Muon, same initialization, six fits,
with a real production-shape memory/time probe before admission.

The cuDNN GRU path cannot accept an arbitrary candidate nonlinearity. A custom
compiled loop or JAX scan needs a separate performance and correctness gate;
TPU availability alone is insufficient. No change to a submission's recurrent
equations is allowed without a matching inference implementation and full-
sequence parity against its own reference. This proposal is separate from
the completed sixteen-arm E470 study and has no public-score forecast.
