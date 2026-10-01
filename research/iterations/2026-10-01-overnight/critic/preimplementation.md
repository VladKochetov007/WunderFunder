# Independent specification critique before actor inspection

Recorded UTC: 2026-10-01T00:27:01.518584+00:00. Read contract, CURRENT_STATE, prior quant-controls proposed study, official metric, immutable fold_eval.py/paired_seed_eval.py/gmetric.py. No overnight actor implementation or outcomes read before this file was created. This file is an initial commitment, not approval of unseen code.

## Primary estimand and comparators

The F-versus-T feature study asks whether a causal conditional-history representation adds competition utility beyond an equal-size marginal-history representation in a fixed V1-like serving ensemble. Its nuisance mark loss is motivation, not promotion evidence. A matched no-feature two-epoch head anchor is also necessary to distinguish extra-feature usefulness from regression relative to V1. One-epoch C is a historical diagnostic, not a replacement comparator. Candidate/control training must restart from exactly the same parent checkpoint and copy all original head weights; frozen GRU must remain byte-identical after training. New input weights must begin at zero. Exact initial predictions should equal parent outputs, including auxiliary heads if altered.

Freeze folds, seed pairing, selected head epoch, per-target serving weights, reference partner checkpoints, clipping location, allowed readouts, and useful-delta threshold before scores. Report individual replaced-slot deltas, paired seeds, and combined-serving estimand separately. Averaging seed deltas does not equal scoring mean predictions. Use two-epoch fixed endpoint unless preregistered otherwise; best-on-fold checkpoint selection adds exposure. An aux-off study is a separate configuration and must not silently enter one feature arm.

## Causality and split attacks

- Fit centers, category thresholds, pi/R, feature normalization and nuisance hyperparameters from allowed training chains for that fold only. Explicit row-group/chain intersections must be empty, and input hashes recorded. Full-sequence target-derived market class must never be an inference input.
- New seq_ix must reset counts, previous raw detector state, previous emitted category, age, cached feature, recurrent state and governor per intended existing policy. No assumed continuity between sequences even when statistical chain labels suggest adjacency.
- Define raw equality as the original numerical detector, including neutral/invalid bridging. First row, unchanged row, changed-but-neutral row, invalid row, next valid row and same-category/different-code transition are separate test cases.
- Probability for current emitted mark uses prefix counts before incorporating that mark; observed current raw code is legal before predicting its future target. Emit features/predict/update counts in the declared order. Never score nuisance log loss using updated counts.
- Freeze non-event behavior: held feature versus recomputed next-event probabilities differ. Age units must be rows or emitted events explicitly. No access to is_scored, target, whole future sequence, final event count, or look-ahead normalization.
- Prefix invariance: identical prefixes followed by arbitrary different future inputs must produce bitwise-equivalent prefix features and predictions. Streaming versus offline/chunked extraction must agree across a chunk boundary at every detector case.

## Numerical and implementation checks

Use an independently written scalar 16-category oracle for F/T probabilities, count increments and feature outputs on planted sequences, not a copy of actor vectorized code. Cover row0, long constant code, all-neutral/invalid prefix, alternating two states, uniform/unseen states, self-category transitions and neutral bridges. Require finite outputs, positive smoothed probabilities summing to one, bounded probability/residual channels, explicit surprise clipping/scaling and no counter overflow for 20,000 rows. Both arms have equal dimensions and common category/age history. Residual channels contain no extra information beyond one-hot and p, so any benefit is representational rather than new-information evidence.

Verify modified head tensor ordering by constructing a nonzero single-feature weight with known sign and checking CPU/native output; zero-initialization parity alone cannot detect unused/swapped new features. Test frozen-backbone byte identity, feature-gradient nonzero, all intended head parameters trainable, optimizer contains no backbone parameters, and parent model eval/dropout/normalization handling matches the control. Head-only training with auxiliary losses may change shared head parameters; exact graph matters.

## Metric and uncertainty audit

Current scorer computes target-wise globally pooled Pearson using weights abs(clip(y,-2,2)), clips the ensemble prediction after averaging members, applies need_prediction AND is_scored, then averages the two correlations. It must not average per-fold or per-sequence correlations. Independent toy examples should distinguish those definitions, weighted versus ordinary centering, target clipping before weights, and clip-before versus after blend. All 99 warm-up rows update state but are excluded from scoring.

Pure-chain filters remove adjacent training leakage but do not create an untouched holdout: all folds/valid have previous exposure. Bootstrap whole chains within fold, paired across arms and conditions; only fixed-model chain uncertainty is claimed. Training seeds are separate replications. Recompute pooled sufficient statistics from the deployed blend, including target order, prediction shape and row-group order. Freeze source and model provenance; fold_eval.py lacks strong nonfinite/shape/source-overlap guards by itself, so the campaign wrapper must add them.

## Serving falsifiers and fallback

A gain is ineligible if it cannot be served within 20MB compressed ZIP, 1 CPU, 16GB, 60 minutes over 39.4M rows in the organizer Python3.11 Linux image. Native feature prototype and export parity must happen before large final fitting. Test extracted ZIP in a fresh process at another working directory with no repository imports or absolute paths. Test no-argument PredictionModel, root solution.py, warm-up None, exactly two finite values, new-sequence reset, repeated deterministic runs and inference on every row. Time at one pinned CPU with numerical-library threads one; include compilation/load overhead and governor behavior. Extrapolation is a proxy, not guaranteed remote runtime. Keep exact known-good V1 ZIP unchanged as fallback.

If F and T both regress versus matched anchor, stop this recipe; do not rescue by choosing a seed/blend after outcomes. If T beats F but not V1 blend, mark representation effect without promotion. If only one fold improves, apply preregistered failure gate and retain evidence. A new physical quote interpretation is not required for practical competition promotion, but no price-alpha claim follows from code prediction alone.
