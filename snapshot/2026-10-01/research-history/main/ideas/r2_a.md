# Haiku R2 Ideas: Model Validation & Selection (Agent A)

Focus: SELECT winning models by fixing validation artifacts + unconventional CV schemes.
Problem: validation overstates test by ~0.02; unknown time order of sequences; test is likely later period than valid.

---

## Idea 1: Hardened Purged Walk-Forward CV (A5 "recover global time order")
**Concept** (mlfinlab ch10.1 + Advances in AFML §6.2 "labeling over non-stationary data"):
- Purged CV (embargo on lookahead labels) + walk-forward rolling window assumes time order
- When time order is UNKNOWN (our case), simulate "in-order" sequences by rank-ordering them via global token ID / i0 + i1 identity recovery, then apply standard walk-forward splits
- Test: does ranking by (i0_id, i1_id, row_ix) yield monotonic feature evolution?

**Analogy to our data**: Train seqs are mixed time periods; valid seqs are 2,076 later; test is unknown. If we can assign pseudo-time via sequence identity (same i0/i1 pair appearing in different periods), then valid sequences become a true OOS period, not a random split.

**Testable hypothesis**: When ranked by recovered instrument identity, valid WP under walk-forward (train → valid window) should be <= cross-validation WP computed on random splits. If they match, time order is irrelevant; if walk-forward drops WP, we have a time-order effect to exploit.

**Cheapest test** (1 min): Extract i0/i1 identity from first 100 rows of each train/valid sequence. Group sequences by identity. Refit GRU (E-083 recipe) on time-ordered splits: early seqs → valid seqs of same instrument. Compare valid WP vs. random-split WP. 
Cache: cache/valid_targets.npz (reshape by sequence grouping).
Script: extend scripts/decide.py to accept split_type = "walk_forward_by_identity".

**Falsifier**: If walk-forward WP is identical to random-split WP, time order does NOT explain validation leakage.

**Research tree branch**: A5 / D1 (walk-forward architecture)

---

## Idea 2: Soft-Label Uniqueness Penalty (D2 "label uniqueness")
**Concept** (mlfinlab §5.3 "sample weights" + AFML §4.8 "tail sets"):
- Problem: is_scored is 9.4% → many rows get same label (intra-sequence stationarity). Models overfit to label patterns.
- Solution: in loss, down-weight rows whose |target - seq_mean(target)| < threshold_k. Prioritize "unique" targets (outliers signal real moves).
- Analogy: mlfinlab's sequential bootstrapping upweights rare events to prevent overfitting to the crowded center.

**Analogy to our data**: Class 3 is bottleneck partly because 46% of metric weight clusters in few sequences with large moves. Within each sequence, most targets repeat. Down-weighting duplicates should push model to capture the signal structure, not memorize seq-level constants.

**Testable hypothesis**: Adding label-uniqueness weights to wMSE (already in E-083) yields WP >= E-083 on validation, with lower variance-of-residuals per sequence (less "stale" predictions).

**Cheapest test** (1 GRU epoch, 6 min): Compute per-row uniqueness score = 1 - exp(-|target - seq_rolling_mean(target, window=100)|). Integrate into loss as sample_weight. Run 1 epoch on E-083 (gru, h128, wMSE). Report ∆ valid WP and residual per-seq entropy.
Cache: cache/valid_targets.npz (compute rolling mean offline).
Script: scripts/train_rnn.py --loss_weights=label_uniqueness.

**Falsifier**: If WP drops or variance increases, signal is driven by label clustering (not overfitting to duplicates).

**Research tree branch**: D2 / C1 (training signal)

---

## Idea 3: Adversarial Validation Within Class (A5 "adversarial validation within class")
**Concept** (mlfinlab §10 "Combinatorial Purged CV" + scikit-learn covariate shift):
- Train a binary classifier (valid vs. heldout test, not real test but simulated forward roll) on features conditioned per class.
- Class 0: classifier AUC > 0.55 → distribution shift within small ticks.
- If class-conditional AUC > 0.6, train separate models per class to exploit the shift structure. Benchmark blend of class-specific models vs. global model.

**Analogy to our data**: Train is 24% classes 0/1, valid is 40% classes 0/1. A classifier trained to distinguish train distrib from valid distrib within class 0/1 should capture the real covariate shift.

**Testable hypothesis**: Class-specific GRU models (separate h128 per class, trained only on that class) stack to higher WP than a single global model, DESPITE class imbalance.

**Cheapest test** (2 GRU epochs per class, 15 min total): Train class-0 GRU on rows where class==0 (24% of train). Eval on valid class-0 rows. Compare to blending global + class-0 specific model. Report class-0 WP delta and best blend weight. Repeat class 1.
Cache: class-specific row indices.
Script: scripts/train_rnn.py --class_filter=0 --output_tag=class0.

**Falsifier**: If class-specific WP is worse, the model is already learning class-conditional features and class imbalance is not a distribution shift issue.

**Research tree branch**: A5 / C2 (diverse members)

---

## Idea 4: Metric Softening Adversarial Threshold Hunt (A3 "test mask softer than valid?")
**Concept** (tree note: lambda=0.02 softens mask; validates like LB):
- Our softened-mask metric weights unscored rows at 0.02. But optimal lambda for VALIDATION may differ from LB.
- Hypothesize: test mask is even softer (lambda_test = 0.001?). Run Bayesian optimization to find lambda that minimizes (valid WP - test WP estimate).
- Test via double-hold-out: split valid into val_A / val_B. Fit lambda on val_A vs. val_B pair. 
- If best lambda ≠ 0.02, your leakage model changes with mask softness.

**Analogy**: Our 0.02 was hand-tuned. If test uses a different mask (unknown), validation ranking of models will be brittle.

**Testable hypothesis**: Optimal lambda (via paired val_A/val_B bootstrap) is < 0.02. When we re-rank GRU vs. baseline by new lambda, ranking flips.

**Cheapest test** (30 sec): Load cache/valid_targets.npz + cache/heur_valid.npz. Subsample 70% → val_A, 30% → val_B. For lambda in [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1], compute WP on val_B using scoring metric trained on val_A. Plot lambda vs. WP stability (variance across random subsamples).
Script: scripts/decide.py --metric_optim=lambda --subsamples=100.

**Falsifier**: If all lambdas yield same ranking of models, the mask softness is not a ranking lever.

**Research tree branch**: A3 / D1 (validation robustness)

---

## Idea 5: Propensity-Score Reweighting to Match Test Marginals (B2 + A5)
**Concept** (causal ML + mlfinlab §5.3):
- is_scored is predictable (AUC ~0.9, tree note E-091). Train a propensity model: p_scored = P(is_scored | features).
- Down-weight high-propensity rows (oversample low-p rows) in training to simulate a test with lower is_scored rate.
- If test has ~5% scored (vs. valid 9.4%), this rebalancing should tighten the valid→test gap.

**Analogy**: Importance weighting to account for selective labeling. Scored rows are not MCAR (missing completely at random); they depend on book state. Reweight training to match hypothesized test scoring process.

**Testable hypothesis**: Retrain GRU (E-083 recipe) with is_scored propensity weights (down-weight p > 0.3). Valid WP <= E-083, but valid-to-LB gap shrinks.

**Cheapest test** (1 GRU epoch, 6 min): Fit logistic regression: is_scored ~ features (on train). Compute propensity. Invert for sample_weight (high p → low weight). Train GRU 2 epochs. Report valid WP, LB WP if submitted.
Cache: propensity scores pre-computed (2 min sklearn).
Script: scripts/train_rnn.py --sample_weights=propensity_inverse.

**Falsifier**: If gap widens, scored rows are not the bottleneck; the leakage is elsewhere.

**Research tree branch**: B2 / A3 (label bias correction)

---

## Idea 6: Extrapolation-Aware Feature Engineering (F1 + A2)
**Concept** (AFML §6.2 "fractal differentiation" for stationarity):
- Current features are likely "forward-looking" due to feature-label lookahead or non-stationary encodings. 
- Test: compute per-feature autocorrelation in train vs. valid. If feature ACF flips sign, it's non-stationary.
- Build synthetic "extrapolation-aware" features: apply inverse transforms (e.g., de-trend, de-seasonalize) that match train ACF structure. Score via GRU on these vs. raw features.

**Analogy**: Our class mix shifts (24% → 40% classes 0/1). Features may over-fit to train's statistical properties. Fractal differencing or robust scaling should generalize.

**Testable hypothesis**: A GRU trained on fractionally differentiated features (AFML recipe) yields valid WP <= E-083 but more stable across held-out class subsets.

**Cheapest test** (offline + 1 epoch): Compute autocorr(features, lags=1..20) on train and valid separately. Identify 10 most non-stationary features. Apply AFML's fractional differencing (d=0.3). Compute correlation of new features vs. targets on valid. If improved, train GRU.
Script: scripts/probe_class3_hgb.py (feature quality probe).

**Falsifier**: If differenced features have lower correlation or model WP drops, stationarity is not the constraint.

**Research tree branch**: B2 / A2 (feature robustness)

---

## Idea 7: Cross-Instrument Meta-Learning (F4 "in-context meta-learning")
**Concept** (modern meta-learning + A2 "instrument identity"):
- Each sequence is a different instrument (possibly). Train a meta-learner: given first K rows of a new sequence (support set), output a vector → predict remaining rows' class/target offset.
- Test: on valid, freeze GRU, train a 2-layer meta-head on first 100 rows of each seq, then predict remaining 19,900 rows.
- If per-sequence meta-adaptation lifts WP by >0.003, instrument-specific structure is worth learning.

**Analogy**: Class 3 next-move direction is hard far from moves (AUC 0.62) but easy near moves (0.96+). Each sequence may have its own "activity regime." In-context learning (prompt-tune on first rows) should adapt.

**Testable hypothesis**: A meta-learned per-sequence offset (trained on first 100 rows, applied to all 20K rows) lifts valid WP on class 3 alone by >=0.003.

**Cheapest test** (6 min, lightweight architecture): For each valid sequence, train a 1-layer adapter (20 dims → 1 output) on first 100 scored rows. Apply to all rows. Compute per-seq WP. Compare to no-adapter baseline.
Script: scripts/train_rnn.py --meta_head=per_sequence_adapter, loss_only_on_first_100_rows_per_seq=False.

**Falsifier**: If WP drops on held-out rows, the per-sequence structure is memorization, not generalization.

**Research tree branch**: F4 / C4 (adaptive capacity)

---

## Idea 8: Combinatorial Purged K-Fold Cross-Validation with Embargo (D2 + mlfinlab 10.2)
**Concept** (mlfinlab §10.2 "Combinatorial Purged CV"):
- Instead of random K-fold, use CPCV: for each fold, embargo a temporal window around train/test boundary.
- Test: run 5-fold CPCV on valid sequences (rank-ordered by pseudo-time). Report WP distribution & overlap with LB.
- If CPCV WP < random-CV WP, we have temporal leakage in the random split.

**Analogy**: Our random CV (seed 42, fixed split) may mask leakage. CPCV ensures no temporal information flows train → test.

**Testable hypothesis**: 5-fold CPCV yields mean WP within 0.003 of single-split walk-forward (idea 1), validating both methods.

**Cheapest test** (2 GRU epochs, 12 min total): Implement CPCV generator (embargo +/- 500 rows). Loop 5 folds, train GRU each, report mean/std WP. Compare to E-083 (fixed split) WP.
Script: scripts/train_rnn.py with --cv_type=combinatorial_purged.

**Falsifier**: If CPCV WP is much lower, the random split actually captures signal; embargo destroys it (implies label is "near-future" deterministic).

**Research tree branch**: D1 / D2 (purged cross-validation)

---

## Idea 9: Selection Bias Detector via Causal Forest (A5 + unconventional)
**Concept** (Athey & Wager, Causal Forests for heterogeneous treatment effects + our scoring mask):
- Train a causal forest to estimate heterogeneous selection effects: how does P(is_scored) vary by features X?
- Rows with high ~selection bias~ (high propensity residual after controlling for structure) are either (a) hard to predict or (b) highly predictive if masking is random.
- Test: up-weight high-bias rows in training. If WP improves, selection bias is masking a signal.

**Analogy**: We know is_scored ~ (a3, i1 activity, a5, book state). But within this, are there unmeasured drivers of scoring? Causal forest finds heterogeneous effects → rows with unusual scoring patterns.

**Testable hypothesis**: Rows with high "unexplained" propensity residual have higher |target| (wider distribution). Up-weighting them improves WP on class 3 (the hardest class).

**Cheapest test** (sklearn CausalForest, 2 min): Fit causal forest on is_scored ~ features. Extract propensity residuals. Segment rows by residual quartile. Compare target distribution & model WP per quartile.
Script: scripts/decide.py --analysis=causal_forest_propensity.

**Falsifier**: If target distribution is uniform across residual quartiles, selection bias is orthogonal to predictability.

**Research tree branch**: A5 / B2 (selection mechanism)

---

## Idea 10: Multi-Horizon Calibration (D1 "recency weighting")
**Concept** (forecast calibration + mlfinlab §12 "bet sizing"):
- Train a model to predict target on rows 100-200, 200-400, 400-1000, 1000+ steps ahead within each sequence.
- Use this to compute a per-row "recency weight": rows far from scored rows get lower weight (they're harder, more prone to extrapolation error).
- Retrain GRU with recency weights. If WP holds or improves, recency structure is learned.

**Analogy**: Our sequence is 20K rows, scored rows are sparse (9.4%). Rows near a scored row are "in-context" and easier. Rows 5000 steps from any scored row are extrapolation.

**Testable hypothesis**: Adding recency weights (exponential decay from scored rows) to training yields same or higher valid WP, with lower prediction std on distant rows.

**Cheapest test** (1 GRU epoch, 6 min + offline compute): For each row, compute distance to nearest scored row. Assign weight = exp(-distance / 1000). Retrain GRU E-083 with these weights.
Script: scripts/train_rnn.py --sample_weights=recency_decay --decay_scale=1000.

**Falsifier**: If WP drops or std increases, distant rows are solvable with current capacity.

**Research tree branch**: D2 / D3 (row weighting)

---

## Idea 11: Leak Detector via Information Bottleneck (unconventional)
**Concept** (Tishby, Information Bottleneck principle):
- Hypothesis: validation WP > LB WP because valid labels contain lookahead info (future price changes). 
- Test: train two models: (a) standard GRU on full features, (b) GRU on "bottleneck" features (embed features through 10-dim layer, force compression). 
- If (b) WP drops < 5% on valid but (a) WP drops > 10% on test, the 5% gap is due to lookahead leakage in unused feature dimensions.

**Analogy**: Lookahead leaks through fine-grained feature structure. Compression forces the model to use only robust features.

**Testable hypothesis**: Bottleneck WP on validation ≈ bottleneck WP on test (tighter alignment), suggesting squeezed model has less lookahead.

**Cheapest test** (1 GRU epoch, 6 min): Add a bottleneck layer (features → 10 dims → GRU). Train for 1 epoch. Compare valid WP vs. unbottlenecked (E-083). If gap shrinks and model is simpler, report as evidence of lookahead.
Script: scripts/train_rnn.py --bottleneck_dim=10.

**Falsifier**: If WP drops > 10%, bottleneck is too aggressive; lookahead is not the issue.

**Research tree branch**: F3 / D1 (label quality audit)

---

## Idea 12: Stability of Predictions Under Bootstrap Resampling (D1 + nonparametric validation)
**Concept** (mlfinlab §2.1 "Haircut Sharpe Ratios" + Rademacher haircuts):
- Bootstrap resample valid sequences with replacement 100 times. For each resample, compute predicted targets on the full valid set (using predictions from resamples, not retraining).
- Measure prediction stability: for each row, compute std of predictions across bootstrap resamples.
- Test: rows with high prediction std are overfitted (model output is noisy). Down-weight these rows in final model training.
- Hypothesis: prediction std is negatively correlated with test-time performance. Rows with stable predictions transfer better.

**Analogy**: Bootstrap aggregating (bagging) is a form of validation: if a single model's prediction on a row is fragile under resampling, it's unreliable.

**Testable hypothesis**: Sorting valid rows by prediction stability (low-std first) yields a ranking correlated with per-row test error (if we had access). Rows with high stability (std < median) have better per-row WP.

**Cheapest test** (1 GRU trained + 100 bootstrap predictions, 10 min): Train E-083 model once. For each valid sequence, resample w/ replacement. Compute predictions on full valid set using features from resamples. Measure per-row prediction std. Bin rows by std; report per-bin WP.
Script: scripts/decide.py --analysis=bootstrap_stability, cache predictions.

**Falsifier**: If prediction std is uncorrelated with per-row WP, the model's stability does not predict generalization.

**Research tree branch**: D1 / D2 (prediction reliability)

---

## Summary Table

| Idea | Concept | Hypothesis | Cost | Tree Branch |
|------|---------|-----------|------|-------------|
| 1 | Purged walk-forward by identity recovery | Time order explains leakage | 1 epoch | A5 |
| 2 | Label uniqueness penalty | Duplicates hurt generalization | 1 epoch | D2 |
| 3 | Class-specific adversarial models | Class shift within train/valid | 2 epochs | A5 |
| 4 | Metric softening Bayesian search | Lambda != 0.02 on test | offline | A3 |
| 5 | Propensity reweighting | Scoring mask is mechanism | 1 epoch | B2 |
| 6 | Fractional differencing | Features are non-stationary | 1 epoch + offline | B2 |
| 7 | Per-sequence meta-learning | Sequences have local regimes | offline adapter | F4 |
| 8 | Combinatorial purged K-fold | Temporal leakage in random CV | 5 epochs | D1 |
| 9 | Causal forest propensity | Selection bias hides signal | 2 min + offline | A5 |
| 10 | Recency weighting | Distance to label matters | 1 epoch | D2 |
| 11 | Info bottleneck leak detector | Features carry lookahead | 1 epoch | F3 |
| 12 | Bootstrap prediction stability | Fragile predictions fail OOS | 1 model + offline | D1 |

All tests avoid heavy GPU compute; most use cached valid data + single GRU epochs (~6 min each).
