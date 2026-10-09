# Independent E530 online-dynamics review

Reviewed 2026-10-07. This is an independent method/source review, not actor approval or experiment admission. No source, protocol, recipe, runtime, or scientific ledger was changed.

## Finding

**Pass as a causal recursive-ridge reference; not evidence of task benefit, a fitted predictor, or production serving readiness.** The implementation's update order and direct-ridge test agree. The remaining requirements are a reset/warm-up-aware caller, a causal training path for any readout, longer forgetting-factor stability checks, and an actual task-endpoint experiment before any performance claim.

`OnlineDynamics.observe(x_t)` computes the transition residual against the pre-update matrix, `x_t - W_{t-1} x_{t-1}`, then uses that completed transition to update `W` and its inverse precision before returning `W_t x_t` as the forecast for the next feature row. Thus both outputs are available at row `t`; neither labels nor future rows enter the update. `test_each_prefix_matches_direct_weighted_ridge` checks every prefix for forgetting factors 1.0 and 0.98 against the corresponding exponentially weighted least-squares solution with prior `lambda**n * ridge * I` and prior mean `W0`. This confirms the recurrence's orientation, weight ages, and update-before-next-forecast semantics.

The state reset restores `W0`, precision, and the previous feature. Input is copied and checked for shape/finiteness, and the source weight matrix is made read-only. Prefix-change and replay tests support causal isolation and reset behavior. The online API is row-wise and does not key state by sequence ID or implement the scorer's warm-up contract; its integration must call `reset()` at every sequence boundary, update on warm-up rows, and return `None` exactly while `need_prediction` is false. `utils.DataPoint` supplies current features, sequence ID, row index and this warm-up flag; it does not supply labels or `is_scored`. The proposal's hidden-target/no-scoring-mask premise is accurate, with the warm-up flag worth making explicit.

The recorded protocol and tests are narrow synthetic evidence. The test suite covers a 20,000-row constant stream only at forgetting factor 1.0; forgetting 0.98 is checked on 40 random rows. The timing uses independent Gaussian vectors, zero prior weights and forgetting 1.0. It does not test correlated/ill-conditioned input, long-stream covariance stability with forgetting, or a fitted projection/readout. Add a long-stream non-unit-forgetting and conditioning check before consuming real data. The reported 28.05/34.58 microseconds per row at widths 16/32 are NumPy reference timing only; even a linear extrapolation to 39.4 million rows is roughly 18/23 CPU-minutes for this update alone, before projection, recurrent parent, task head, callback and serving overhead. No native timing or Rust integration is present.

The proposal correctly requires training any fixed readout on causally generated dynamics states, and treats auxiliary feature prediction only as a diagnostic. No such projection/readout fit or complete official-metric comparison exists in the reviewed source. The online-dynamics protocol reports five focused tests passed, CPU-only, with 48,377,856-byte peak RSS; its `independent_review` is still pending. Independent rerun below passed five tests. Public score forecast remains null.

## Reviewed artifact identities and verification

| Artifact | SHA-256 |
|---|---|
| `src/wnn/online_dynamics.py` | `15276c6879f770dcbb89f3dd7b10f916b7db7790f5564c7ca5fd690f25c2722b` |
| `tests/test_online_dynamics.py` | `cf4fb44fef309d17baee88c0f0fb86f45f0ce519d313a9186eccf6122808091e` |
| `CONTROLLER_PROPOSALS_V1.md` | `acead0002e992a4731b78b8656fa52550271b042a14fceb5f984a18bd1b778b7` |
| `ONLINE_DYNAMICS_PROTOCOL_V1.json` | `d42255857844b1aea0ab7f5ce3f11cb67ff2d1d2a4ed4785dd76c4c6299ea5e4` |

Executed `tests/test_online_dynamics.py` and `tests/test_conditional_readout.py` together with one pinned CPU, no GPU, and no supplied data. **10 passed in 3.57 seconds; maximum process RSS 870,528 KiB.**
