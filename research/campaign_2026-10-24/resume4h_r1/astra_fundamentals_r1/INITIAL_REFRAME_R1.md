# Independent first reframe — 2026-10-24

Status: source-informed proposals, before reading the requested feasibility reports, PR synthesis or old idea archive. No scientific execution. Owner: the single user-authorized native Astra/ultra research-reframe agent. Initial basis: objective, agent instructions, metric implementation, raw task documentation and the five observations supplied by root.

Independence limitation: mandatory `specs/current-state.md` and the active frontier expose a short summary of the earlier ten-reviewer synthesis (recover online contrasts, history versus diversity, residual mismatch and response speed). I have not opened that synthesis or its individual reports. This is an independent formulation from restricted context, not a statistically independent evaluation or a claim of blindness to those four topic labels. All existing evaluation folds remain exposed.

## Initial assumptions worth challenging

1. **A better residual head means new information.** A correction may mostly repair output calibration, clipping or a parent's systematic error along an existing direction. All raw heads being worse and all shrunken heads gaining similar tiny amounts is compatible with this explanation, but does not establish it. “Mean-only” must be read in the code: an online input mean is not a constant prediction. First compare marginal score directions and nuisance directions before attributing improvement to relationships.
2. **The natural time coordinate is the row number, and equality means stale information.** Rows may interleave multiple observation processes. A column staying unchanged can reflect no event, coarse quantization, invariance of the representation or an update that preserves its value. Event-time state, observation hazard and genuinely elapsed physical time are different objects. Field/group clock disagreement motivates an observation-process test, not automatic adoption of an age feature.
3. **Smaller training fidelities preserve mechanism rankings.** A 10% versus 25% reversal may concern sample diversity, sequence histories, optimizer progress, capacity or saturation of an already-learned feature. A shortscreen that changes several at once need not predict the full-data endpoint. Characterize the fidelity intervention before promoting or rejecting innovation.

## Distinct lenses and provisional tests

| Lens | Reframe | Cheapest discriminator |
| --- | --- | --- |
| Metric geometry | The score is weighted correlation of clipped values, not generic prediction error. The target-dependent weights tilt the optimal prediction problem. | Analytically decompose each fixed correction into its first-order score derivative, curvature and clip crossings; compare with simple affine-parent directions. |
| Observation process / change time | Treat which features changed, and in what order, as a separate stochastic process from feature values. | Compare target-free event/transition structure and residual response at matched causal states; distinguish individual change, grouped change and time since previous *observed* change. |
| Anonymous representation / factor | Stable column identity does not guarantee numerical depth order, physical scale or a common feature coordinate system. | Source audit of every ordering, subtraction and symmetry assumption; conditional event/value perturbations that preserve legal structure. Do not blindly permute named sides or invent economic units. |
| Selection / invert | The useful research unit may be a change to the deployed mixture, not a standalone model. | For a proposed replacement, evaluate the complete fixed endpoint and its covariance with endpoint residuals, using exact clipping order. |
| Fidelity / instrument | A resource budget can accidentally define a different learning problem. | Matched masks and update counts for 10%/25% runs; separate data exposure from optimization and history fidelity. |
| State / change search unit | A recurrent state can miss information either because it forgets or because it has never learned a representation. A readout failure does not distinguish them. | Controlled history interventions on fixed weights, then one prospective comparison isolating event/value state from an equal-budget nuisance state. |

## Preliminary priority

First ask whether existing correction predictions contain any score-relevant direction beyond parent calibration and the shared mean-only component. This can use frozen prediction sufficient statistics with zero new scientific fits. It informs whether the next investment should be optimization/calibration, a missing representation, or a genuinely complementary endpoint. It cannot prove public transfer, establish equivalence from an interval crossing zero, or invalidate an entire online-learning family.

Tentative prospective screen: three matched treatment/control cells, six new small-head fits total, with the same clean frozen parent, same training groups, same strict-FP32 recipe and fixed checkpoints. The treatment should add one precisely defined observation-process representation to an equal-capacity control. Select the exact intervention only after inspecting the actual existing code and prior ideas, so an already-completed comparison is not renamed as new.
