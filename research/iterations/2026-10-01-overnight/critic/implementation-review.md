# Independent implementation review before empirical selection

## AUX02

PASS at source SHA256 `f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f`. The only scientific change is a common zero/one multiplier on existing structure and next-change losses. Shapes, private output rows, optimizer parameters, schedule, fixed parent and head endpoint stay matched. The copied trainer also protects fresh output directories and skips evaluation while preserving its DataLoader RNG draw.

`check_aux.py` extracts the actual two auxiliary AST blocks from the base and actor trainer. Scale1 losses and output gradients are exactly equal; scale0 is exactly main-only while auxiliary output gradients remain zero tensors. An independently constructed frozen RNN retains byte-identical trunk parameters after AdamW, with trainable head gradients. Iterating an actual sequential two-worker evaluation DataLoader consumes exactly the emulated one CPU integer draw. Fixed prediction output uses checkpoint epoch/fold checks, original model construction and isolated arrays with NaNs outside the predicted fold. No data or candidate score was read by these tests.

## F/T features and native deployment

The actor's feature schema is causal: raw detector memory advances on every row; current event probabilities use past counts; current mark is observed before future-price prediction; all state resets per sequence. Nuisance fitting reads only four raw columns on other-fold validation chains, checking held-chain exclusion and preserving raw input hashes. Extra projection starts at zero without changing initialization RNG. It enters before the existing head GELU and leaves the original recurrent inputs unchanged.

The separate native implementation was tested against an independently written Counter-based oracle and the actor batch implementation. Its results are in `native-transition-check.json`; a separate actor-authored C++ review is recorded at the campaign root. This division matters: my native numeric tests alone are not an independent review of my own C++ implementation. Mixed precision is explicit: existing dense recurrent/head matrices may use fp16 storage, while the new extra projection stays float32 and nuisance arithmetic stays float64.

The new native feature path is technically feasible; it is not a serving-time certification. Fixed4 ensemble cost on the measured host remains much larger than60minutes. See the full runtime receipts and independent optimization branch before any final artifact decision.

## ENSHEAD01

PASS at trainer SHA256 `7e8fb8b567af132c09980b11b09b41ad579b5f1bdf6875bd1729aa3f22a24894`, support SHA256 `016234399859554dd0f09313393220e45682b92278ffe033f04d383760c769d8`. Review requested and obtained per-entry source-order/chain-map/position checks for partner caches, plus required parent initialization. Cache generation uses the matching fold's H2 control, predicts only allowed training rows and reads no labels or scoring mask. In-sample partner overfit remains a statistical risk, not a code leakage finding.

`check_ensemble.py` executes the actual main-loss AST against a separately written target-wise centered formula. Gradient difference is at most1.86e-9. The fixed multiplier4 restores the own-slot standalone gradient when all partners are numerically identical. Only two log-scales train in the scale control; nonidentity folded export agrees within2.98e-8 and original tensors stay byte-identical after AdamW. The fixed endpoint is exported as an ordinary model, so native inference needs no additional scale operator. The positive-scale constraint and fixed-zero offsets are part of the registered recipe.

## Frozen grader

The upgraded wrapper SHA256 `800801305721251dda0c26320c1cf2f1ffc468e2843d7d8b61ea06a4f20c70b0` adds immutable target/split/class metadata hashes, pure-chain integrity, prediction receipt alignment, checkpoint hashes and finite selected-array guards. It calls the unchanged pooled scorer, preserving arithmetic order and blend-before-clip semantics. Independent synthetic direct correlation agrees with target-specific sufficient-statistic scores within1.11e-16; adversarial examples distinguish incorrect sequence averaging and clip-before-blend. The first formula receipt predating the provenance upgrade is preserved separately; the current receipt matches the exact upgraded wrapper.

No critic test adds an empirical fit or candidate WP query. These correctness checks justify executing the frozen exploratory studies; they do not establish competition improvement, an untouched holdout, physical market structure or hidden-test transfer.
