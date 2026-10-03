The cheapest dangerous false positive is a replay that matches aggregate clocks
but reconstructs the wrong recurrent history or optimizer mapping. An EMA export
cannot supply the missing paired raw state. Hidden tensors belong to distinct
raw/lagged-EMA trajectories, even when current weights happen to coincide.

The teacher prediction must be a detached value from the lagged EMA forward at
the captured boundary. Updating EMA weights later must not retroactively change
that target. RNG belongs before the stochastic forward, not after it. Optimizer
moments need stable parameter names and exact group order, not merely compatible
shapes. AMP nonfinite steps may skip optimizer updates while surrounding clocks
continue; reproduce actual ordering rather than imposing preferred semantics.

TESTS.json freezes acceptance and adversaries before trainer inspection. This
first pass will not inspect builder files or approve a GPU candidate. CPU fixtures
prove detector sensitivity only; actual fast/Muon/fp16 combination replay needs
its own measured backend parity. Source-derived observations will be appended
with file/line provenance separately from these precommitted risks.
