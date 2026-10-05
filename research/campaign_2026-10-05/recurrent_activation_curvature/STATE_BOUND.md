# What bounded candidate shapes guarantee

For the exact reset-after GRU equation used here,

`h_next = (1-z) * g(tanh(v)) + z * h_previous`, with `0 <= z <= 1`.

If a scalar map `g` maps `[-1,1]` into `[-1,1]`, and the initial hidden state is
in that interval coordinatewise, the next hidden state is also in that interval.
This follows directly from convexity. Induction applies to every coordinate,
time step and recurrent layer, regardless of the finite weight values.

The fixed sine maps have `g(u)=u+alpha*sin(pi*u)`, endpoints ±1 and derivative
`1+alpha*pi*cos(pi*u)`. For `|alpha|<=0.2`, the derivative is at least
`1-0.2*pi>0`, so the map is monotone and preserves the interval. E479's learned
parameterization `alpha=0.2*tanh(raw_alpha)` preserves that condition at every
finite parameter value. It therefore preserves the hidden-state interval in
exact arithmetic, even while learning the candidate coefficients.

E481's cubic has `g(u)=u+0.1*pi*u*(1-u*u)`. Its derivative is
`1+0.1*pi*(1-3*u*u)`, with minimum `1-0.2*pi>0`; its endpoints are also ±1.
The same induction applies to its exact recurrence. The baseline is immediate.

This is a hidden-state amplitude bound, **not a gradient-stability guarantee**.
The recurrent Jacobian also contains weight matrices and derivatives of gates,
and products of those Jacobians may grow or shrink. Finite floating-point
arithmetic, intermediate logits, parameter updates and actual gradients still
need explicit finite checks. The argument does not bound output heads or prove
predictive benefit. Sampled state telemetry can detect violations at the
recorded steps; four samples cannot prove the invariant over an actual run.

This clarifies the earlier scalar-map caveat: candidate bounds do extend to
hidden-state amplitude through the convex GRU update, under the stated initial
state assumption. They do not establish stability of the full learning system.
