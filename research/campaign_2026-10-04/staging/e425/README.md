# E425 soft specialist route staging

This staging tree does not modify `src/wnn/specialist.py` or
`scripts/train_rnn.py`, which are inputs to the active E424 run. The staged
`src/wnn/specialist.py` adds a `soft` route: it combines all expert outputs with
the learned prefix-router softmax probabilities. The `learned` route retains
its existing straight-through hard top-one behavior; `hash` and `universal`
are unchanged. All four routes retain identical trainable tensor shapes.

`patches/train_rnn.py.diff` contains the one-line CLI choice update needed to
expose `--specialist-route soft` after the active run. `tests/` contains CPU
unit tests for the exact production parameter budget, causal prefix and chunk
continuation parity across row 99, direct prediction-gradient support for all
experts, and equality with universal averaging when router logits are uniform.

Tests are intentionally not run while E424 occupies the constrained local
worker. No scientific fit or GPU launch is part of this staging change.
