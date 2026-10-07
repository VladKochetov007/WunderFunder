# Random library source for weak-model complementarity reframe

Source: *Momentum has its Moments (with Code)*, Vertox, 2023, pp. 5–7. The full local source path, PDF digest, selection pool and seed are in [SOURCE.json](SOURCE.json).

The article's small transferable component is to estimate a changing risk scale from past observations and adjust exposure inversely, with a cap. The local excerpt says, “Forecasted Volatility is a good predictor of future returns,” and later describes setting weight to “vol_target / vol_predicted.” Those are the source author's claims and code choices, not verified facts for this competition.

Possible analogy: use causal model disagreement or a separately trained uncertainty head as a forecast of *which expert is locally reliable*, then adjust expert weights with a bounded gate. This analogy remains speculative. Disagreement is not prediction error, the source concerns financial returns rather than this benchmark, and serving legality must be checked before any use of target history. The official comparison must remain `global_wp_clip2_v1`.
