SOURCE_OCR.txt absent → read SOURCE.txt (per PROMPT fallback).

```json
{
  "source_assessment": {
    "source_id": "vertroxquant Identifying Small Mean Reverting Portfolios — An Analysis (Part 2, with code)",
    "extract_used": "SOURCE.txt (SOURCE_OCR.txt not supplied). 3135 chars; SOURCE.json says pages 7 and 10 of 10 were sampled.",
    "what_the_extract_actually_contains": [
      "A backtest loop compounding mean-reversion and momentum portfolio values from feature-matrix returns (P_mr/P_mom recursion).",
      "Definition of the Johansen procedure and L(1) series (difference once to stationarize).",
      "Greedy Search for a sparse mean-reverting portfolio: minimize portfolio predictability subject to <= k nonzero weights that sum to 1; stated NP-hard; heuristic = brute-force the most mean-reverting pair, then greedily add the single asset giving the most mean-reverting triplet, repeat to k.",
      "The extract asserts greedy is fast and 'more robust' than the exact optimum."
    ],
    "transferable_atom": "Coarse-to-fine greedy forward subset selection under a cardinality (<=k) + simplex (sum-to-1) constraint, driven by a residual-variance/stationarity objective — divorced from the finance portfolio headline.",
    "limitations": "Only ~2 of 10 pages present; no objective equations, no Semidefinite Relaxation detail, no code. Any numeric claim (e.g. 'more robust') is the source's untrusted assertion, not verified. My mappings below are analogies, not source claims; I did not read inaccessible pages and do not claim to.",
    "source_trust": "treated as untrusted reference data; embedded links/instructions ignored."
  },
  "ideas": [
    {
      "title": "Greedy Sparse Mean-Reverting Feature Cliques (per L11 group)",
      "mechanism_family": "Sparse forward subset selection / cointegration eigen-combination transfer",
      "source_component_and_location": "SOURCE.txt p.9 'Greedy Search': brute-force best pair, then greedily append one asset to k, minimize predictability with <=k nonzero weights summing to 1.",
      "task_mapping": "Assets -> features inside one L11 book/trade group; simplex weight vector -> sparse feature combination; minimized residual variance -> a (near-)stationary, mean-reverting component treated as a causal input feature.",
      "proposed_change": "In addition to the 112 raw features, feed each group's greedy k-sparse simplex combination (k=2..5) into the predictor; weights chosen only on group-disjoint training folds.",
      "assumptions": "Some feature groups admit sparse near-stationary linear combinations, and stationarity of a combination of causal features is a usable predictive signal.",
      "predicted_observable": "Weighted Pearson correlation rises, with gains concentrated on rows carrying large |clipped target| (high metric weight).",
      "cheapest_discriminating_test": "One historical fold: build greedy k=2..5 simplex combinations per group, concatenate to frozen penultimate features, fit an OLS/logistic head, compare weighted Pearson to controls. CPU-only, seconds.",
      "matched_control": "Same-cardinality top-k univariate-correlation features, and equal-size random feature subsets per group.",
      "falsifier": "Greedy sparse components fail to beat both controls on a held-out fold, or each component's standalone correlation with either target is ~0.",
      "resource_estimate": "CPU seconds-minutes per fold; deployed weights precomputed, trivially inside serving envelope.",
      "suggested_parent_direction": "Augment inputs to the served recurrent blend with sparse mean-reverting group components.",
      "genuinely_new_part": "Applying the cardinality+simplex greedy forward search with a mean-reversion objective to grouped order-book features instead of ranking features independently.",
      "possible_interactions": "Sensitive to feature normalization/stationarization; shared features may let one greedy clique serve both instruments; interacts with group-disjoint split definitions.",
      "expected_public_gain": null
    },
    {
      "title": "Pair-Seeded Stagewise Interaction Grower",
      "mechanism_family": "Stagewise/boosting forward selection over second-order interactions",
      "source_component_and_location": "SOURCE.txt p.9 greedy recipe: seed with the exhaustive best 2-element combination, then grow by one element at a time (coarse-to-fine seed).",
      "task_mapping": "The source's 2-element 'pair' seed becomes a cross-group feature product/ratio; growth becomes greedily appending single features to a sparse interaction term up to a small budget k.",
      "proposed_change": "Construct a k<=4 sparse interaction term per target by seeding on the best cross-group pair then greedily appending the feature maximizing held-out weighted correlation.",
      "assumptions": "Each future-price indicator depends on low-order feature interactions not recoverable by main effects under the current blend.",
      "predicted_observable": "Held-out weighted correlation improves specifically on rows where a main-effects-only residual is large.",
      "cheapest_discriminating_test": "One fold, frozen features, linear head: compare pair-seeded greedy interactions vs main-effects-only head. CPU minutes.",
      "matched_control": "Equal-count random interaction terms and a full pairwise-ridge on the same frozen features.",
      "falsifier": "Interaction term is no better than main-effects-only and no better than the random-interaction control.",
      "resource_estimate": "CPU minutes per fold (pair scan dominates); fine at training scale.",
      "suggested_parent_direction": "Extend the recurrent blend's input set with a sparse stagewise interaction constructor.",
      "genuinely_new_part": "Reusing the pair-seed-then-grow greedy as an interaction-construction operator, where the source applied it to selecting assets.",
      "possible_interactions": "May overlap with tree/kernel heads already capturing interactions; depends on feature scaling; per-target budgets may need decoupling.",
      "expected_public_gain": null
    },
    {
      "title": "Greedy Sparse Stationarity Scout for Causal State Management (unconventional)",
      "mechanism_family": "Regime-change detection via sparse cointegration residual (transfer to state timing, outside portfolio construction)",
      "source_component_and_location": "SOURCE.txt p.9 mean-reversion objective ('most mean reverting portfolio') and the L(1)/Johansen stationarity framing.",
      "task_mapping": "The greedy sparse combination per group defines a residual expected to be stationary; when that residual's unit-root behavior re-emerges (starts trending), it flags a structural break -> a causal trigger to decay/reset recurrent state.",
      "proposed_change": "Maintain greedy sparse group residuals online; monitor a rolling mean-reversion statistic; when it crosses a threshold, decay/reset the recurrent state (or down-weight that group) at that timestep only.",
      "assumptions": "Causal regime breaks manifest as loss of stationarity of feature combinations, and detecting them earlier than a fixed schedule helps.",
      "predicted_observable": "Correlation gains concentrated on rows immediately following detected breaks; reduced stale-state errors after breaks.",
      "cheapest_discriminating_test": "Replay one sequence strictly causally: fixed state vs stationarity-gated reset; compare pooled weighted Pearson on that sequence. CPU seconds.",
      "matched_control": "Fixed-schedule resets, random-timing resets, and a plain rolling-variance threshold trigger of matched rate.",
      "falsifier": "Stationarity-gated resets do not beat the matched-rate controls on the replayed sequence(s).",
      "resource_estimate": "O(#groups) online per step; negligible within one-CPU serving budget.",
      "suggested_parent_direction": "Add break-gated state decay to the recurrent predictor's state-management layer.",
      "genuinely_new_part": "Repurposing sparse mean-reversion selection from a return-construction objective (source) into a causal state-timing/regime signal.",
      "possible_interactions": "Trigger rate interacts with warm-up length and scoring mask; thresholds must be set group-disjointly.",
      "expected_public_gain": null
    },
    {
      "title": "Simplex-Sparse Greedy Correction of the Served Blend",
      "mechanism_family": "Greedy sparse ensemble reweighting under a simplex/cardinality constraint",
      "source_component_and_location": "SOURCE.txt p.9 constraints: <=k nonzero weights that sum to 1.",
      "task_mapping": "Blend members / per-instrument heads -> 'assets'; a small sparse simplex perturbation of blend weights is the 'sparse portfolio'.",
      "proposed_change": "Post-hoc, greedily select k members and non-negative weights summing to 1 that maximize held-out weighted correlation, rather than the current dense blend.",
      "assumptions": "The served blend is near-optimal, so only a small sparse correction (or member pruning) remains, and correlated members make sparsity useful.",
      "predicted_observable": "Small but consistent weighted-correlation gain, stable in sign across folds; a genuinely sparse optimum (few nonzero weights).",
      "cheapest_discriminating_test": "Cache member predictions on one held-out fold; run greedy k-sparse simplex selection; compare weighted Pearson to the current blend. CPU seconds.",
      "matched_control": "Equal-weight blend, dense (unconstrained) reweighting, and random k-member subsets.",
      "falsifier": "Sparse k-subset is no better than equal-weight or dense fit on held-out folds.",
      "resource_estimate": "Seconds given cached member outputs; zero serving cost beyond fixed weights.",
      "suggested_parent_direction": "Replace/augment the served blend combiner with sparse simplex weights.",
      "genuinely_new_part": "Transplanting the source's cardinality+simplex constraint onto ensemble combination, where the source applied it to asset selection.",
      "possible_interactions": "Interacts with per-sequence state resets and per-target metric weighting; overfitting risk under few folds.",
      "expected_public_gain": null
    },
    {
      "title": "Greedy Group-Spanning Cointegration to Define Group-Disjoint Fitting",
      "mechanism_family": "Group-level greedy subset selection for split/regularization design",
      "source_component_and_location": "SOURCE.txt p.9 greedy 'add one asset that gives the most mean-reverting' set-building step.",
      "task_mapping": "Assets -> feature groups; greedily build a minimal spanning set of groups whose combination is most mean-reverting, then use that set to structure group-disjoint folds and to regularize reliance on non-selected groups.",
      "proposed_change": "Derive group-disjoint splits by greedy cointegration clustering and add a penalty concentrating model weight on the greedy-selected groups.",
      "assumptions": "Group-disjoint generalization is a real bottleneck, and some groups are jointly near-stationary across the two instruments.",
      "predicted_observable": "Reduced train/validation gap and improved group-disjoint held-out weighted correlation.",
      "cheapest_discriminating_test": "One fold re-run with greedy-derived group splits vs random group splits; compare weighted correlation and gap. CPU minutes.",
      "matched_control": "Random group partitions of equal size; unregularized model on identical splits.",
      "falsifier": "No gap reduction and no held-out gain versus random-partition control.",
      "resource_estimate": "CPU minutes per fold; training-scale only, no serving impact.",
      "suggested_parent_direction": "Strengthen the parent's group-disjoint fitting/validation protocol.",
      "genuinely_new_part": "Using greedy cointegration selection to design the split and regularizer rather than the predictor itself.",
      "possible_interactions": "Couples with the shared-feature structure and with any group-disjoint gains reported elsewhere; may change fold difficulty.",
      "expected_public_gain": null
    }
  ]
}
```
