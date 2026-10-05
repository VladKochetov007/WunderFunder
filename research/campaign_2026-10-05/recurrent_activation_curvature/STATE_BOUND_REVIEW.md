# Independent review of `STATE_BOUND.md`

**Finding: the hidden-amplitude argument is correct under its stated exact-arithmetic and bounded-initial-state assumptions.** The E481 implementation uses reset-after GRU equations and the same convex update; no counterexample arises from its two-layer state ordering.

For each layer, coordinate, and time step, the implementation computes `r=sigmoid(i_r+h_r)`, `z=sigmoid(i_z+h_z)`, then `u=tanh(i_n+r*h_n)` and `h'=(1-z)g(u)+z h`. With finite exact real preactivations, `u∈[-1,1]` and `z∈[0,1]`. If `h∈[-1,1]` and `g([-1,1])⊆[-1,1]`, the update is a convex combination of two values in the interval, hence `h'∈[-1,1]`. The initial state created by the model is zero. A caller carrying state across chunks preserves the same invariant by induction; arbitrary initial state requires the coordinatewise bounded-state assumption in the proof.

The range claims are analytically established:

- **Baseline:** `g(u)=u`, immediately in range.
- **Sine:** `g(u)=u+α sin(πu)`. Endpoints are `g(±1)=±1`. For `|α|≤0.2`, `g'(u)=1+απ cos(πu) ≥ 1−0.2π > 0` because `|cos(πu)|≤1`. Thus `g` is increasing and its endpoint values bound its entire image to `[-1,1]`. For E479, `α=0.2 tanh(raw_α)`, so every finite raw parameter has `|α|<0.2` (and the closed bound also holds by continuity).
- **Cubic:** `g(u)=u+0.1πu(1-u²)`. The endpoints remain ±1 and `g'(u)=1+0.1π(1−3u²)`. Since `u²≤1`, this derivative is minimized at `u²=1`, giving `1−0.2π>0`. Therefore it too is increasing and maps the interval into itself.

For the two-layer scan, layer 0's outputs are each bounded by the same recurrence induction and are valid inputs to layer 1. Layer 1 has its own bounded state and candidate update. Concatenating the final-layer sequence with the embedding for the output head does not affect recurrent state. The argument applies independently to every batch coordinate and sequence step; it does not depend on weight magnitude in exact real arithmetic.

The note correctly distinguishes an amplitude invariant from gradient or optimization stability. It does not bound derivatives of the complete recurrent map, products of recurrent Jacobians, finite-precision overflow, head outputs, or predictive quality. In floating-point execution the invariant is subject to roundoff and overflow, so runtime finite/state checks remain useful. The note's statement that four sampled telemetry points cannot establish an all-step runtime invariant is also correct.

No edits to fit sources, checkpoints, or score artifacts were made for this review.
