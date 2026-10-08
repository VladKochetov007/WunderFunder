# Grok grok-4.7 raw public output (lens: microstructure, target law, quant brainteaser)

Public stdout captured verbatim from the bounded native CLI call. No private reasoning.

The brief is the only source I will open. I will read it and then write the ideas in plain text.The recurrent student saturates because the served targets are a low-dimensional function of a latent fair value, while a GRU256x2 spends about 33 of the 60 CPU minutes relearning that function from nonstationary, anonymously indexed book columns. On the class-3 rows that carry about 46% of the metric weight, the labels obey t0 = F − ask and t1 = bid − F, so one scalar F determines both outputs. The useful symmetry is translation in price, conservation of a cointegrating residual, and a short eigen-expansion of book innovations. A coupled recurrence is a costly way to represent those objects.

1. Algebraic one-factor head
Mechanism family: target-law decomposition.
Source: section 3 of the brief. No defensible correspondence to either passage. Passage 1 is a covariance expansion, and passage 2 is an alpha-search ablation. The identity comes from the measured labels.
Structural analogy: once the current touch is observed, both scored outputs are a marking of a single latent position F.
Intervention: on rows that training labels identify as class 3, fit only F. Serve t0 = F̂ − ask and t1 = bid − F̂, with ask and bid mapped from the price-like columns by a causal affine into label units. Other classes keep the existing head, selected by a causal class probability.
Predicted observable: the class-3 contribution to weighted Pearson rises. The identity residual t0̂ + t1̂ − (bid − ask) on those rows goes to zero. The complement moves only through the gate.
Effect size: a gain of 0.05 on the class-3 correlation is about 0.02 on the average because that class is 46% of the weight. Uncertainty is wide. The split of today’s 0.666 across classes is unmeasured here, and the public gap to 0.700 can remain if F itself is the hard object. Interval easily includes a gain below 0.005 if the current two heads are already coherent.
Fold-0 test: from training labels, flag rows where ask + t0 and bid − t1 agree. Report the current model’s identity residual and its weighted Pearson on the flag and its complement. Then replace the two heads by an oracle F on flagged rows only and record the ceiling. No new fit is required for the ceiling.
Serving cost: about 0 minutes of extra runtime. The head is a handful of adds. The ZIP impact is negligible.
Status: novel relative to the soft value-grid and readout mixtures in section 4.

2. Hold F, mark every row to the current touch
Mechanism family: control rule on the latent, with an algebraic readout.
Source: passage 2, “Minimum number of holding days H 20 days.”
Structural analogy: that backtest holds a position for a minimum horizon while prices still move the mark. The corresponding object is F. The served targets must keep moving with bid and ask inside the hold.
Intervention: update F only after a causal seed statistic has kept the same sign for H book rows. Between updates, freeze F and recompute both targets from the current touch. H is chosen on the training fold from a small grid such as 1, 5, 20, 100.
Predicted observable: quiet rows, where the touch jitters and F should be sticky, improve. Rows with a real fair-value jump inside the hold get worse if H is too long. Net movement should show up on the low-activity half of class 3.
Effect size: 0.003 to 0.015 local on the quiet class-3 slice, with a one-standard-error band that can include zero. Public transfer is unproven. This is smaller than the 0.034 gap to 0.700.
Fold-0 test: compute a frozen microprice-style F seed, apply the hold, score quiet versus active rows separately against the unheld seed and against the current GRU. Same fit membership, no second seed.
Serving cost: well under 1 minute. State is a scalar and a counter.
Status: a new mechanism on top of the failed quiet-row output hold in section 4. The held object is F.

3. Frozen seed formula, recurrent model only on the residual
Mechanism family: decomposition into a closed-form alpha plus a small residual learner.
Source: passage 2. Initializing the pool with the seed formulas of reference [10] is the “Non-Init” contrast, and widening the operator set moved average IC by +0.0034 in the original space and −0.0016 in the expanded space, both inside the seed standard deviation.
Structural analogy: further recurrent operators are the expanded search space. The missing piece is the seed formula for F. Same-recipe seeds already added only about +0.0004 public, which matches a search that is inside noise.
Intervention: freeze three causal candidates for F. Volume-weighted touch (microprice), a depth-weighted price after sorting price-like columns, and a trade-VWAP from the four trade-price and trade-volume columns. Learn a nonnegative combination on training rows. Train a narrow GRU, or a linear head, only on F − F_seed, under the same algebraic readout.
Predicted observable: class-3 weighted Pearson moves toward the seed. The residual head helps only the slice where the seed’s error still correlates with later labels. Active-book rows should move first.
Effect size: if one seed already matches the latent, the class-3 gain can be 0.02 to 0.08 and the average 0.01 to 0.04. If every seed is only a covariate of F, expect 0.002 to 0.010 on the average. Uncertainty is one-sided and large. The test exists to separate those two regimes.
Fold-0 test: score each frozen seed and their convex combination with the algebraic head, before any recurrent fit. Promote a residual network only if the best seed leaves a weighted correlation in the residual above a pre-registered bar, for example 0.01, on the training membership.
Serving cost: the seeds are a few dozen arithmetic operations per row, under 2 minutes for the full test stream. A residual GRU of width 32 is a fraction of the current 33 minutes. Replacing the 256-wide trunk is the point.
Status: a reframing of morphology features and shared-recurrent families in section 4. The new rule is that the network is forbidden from re-predicting the part the seed already explains.

4. Conserved cross-instrument residual as the bottom eigenmode
Mechanism family: symmetry / conserved quantity.
Source: passage 1. A Gaussian vector is m plus a sum of λ_i ξ_i φ_i, and the covariance of that sum is B. The smallest eigenvalue of the joint level covariance is the direction whose coefficient has the least variance.
Structural analogy: i0 and i1 price-like columns share a random-walk component. The tradable latent is the cointegrating residual, which is the bottom mode of B. F for i0 moves with that residual, while the GRU otherwise spends state on the shared level.
Intervention: maintain a causal 2×2 covariance of a causal i0 mid and an i1 mid, with a long discount. Take the unit-norm bottom eigenvector β, form s = mid0 − β mid1, and let a cheap filter of s be the candidate F in label units. Reset the covariance at each sequence boundary.
Predicted observable: sequences where the two instruments co-move should gain on both targets. Sequences where i1 is unrelated should be flat if the gate falls back to the i0-only seed. The gain should concentrate on slow displacements, which carry weight through |clip(y)|.
Effect size: 0.005 to 0.02 on the average if i1 is a real hedge, and about 0 if the in-sample hedge is unstable out of sample. A confidence interval from a few dozen sequences can cover zero even when the in-sample residual looks stationary.
Fold-0 test: report the causal residual’s autocorrelation and its weighted Pearson against the class-3 F built from labels, using only past rows to form β. Compare with the raw i0 mid as a control. No GPU fit.
Serving cost: under 1 minute.
Status: novel. Cross-asset inputs exist in the task, and section 4 does not contain an online cointegrating coordinate.

5. Rank-k innovation basis with independent scalar filters
Mechanism family: approximation, plus a diagnostic that can retire the coupled trunk.
Source: passage 1. After the rotation, the coordinates λ_i ξ_i are independent scalar Gaussians, and that independence is what produces the density. The covariance eigenfunctions φ_i are the objects a recurrent net would otherwise have to learn.
Structural analogy: the book innovation at each row is a draw from B. A handful of modes carry the fair-value shock. Coupling all 112 channels inside a GRU is wasted if the score-relevant coordinates are independent after this rotation.
Intervention: on a reduced set of about 16 causal inputs (sorted distances to touch, imbalances, trade surprises, the cointegrating residual), keep a discounted covariance, its top four eigenvectors, and four independent one-dimensional filters of the coefficients. Predict F from those four filtered scalars. As a diagnostic, measure residual cross-correlation among the coefficients on held-out rows of the same fit membership.
Predicted observable: if residual cross-correlation is near zero, the four filters match a much larger GRU on class 3. If cross-correlation stays material, the basis is only a preprocessor and the gain is a small feature improvement. Quiet rows should improve when discarded eigenvalues are microstructure noise.
Effect size: replacing a 0.66-quality coupled model by the diagonal filters could add 0.00 to 0.03 on the average. The upper end requires the class-3 fair shock to be linear in the top modes. Uncertainty dominates. Treat a gain under 0.005 as a reason to keep the trunk and use the basis only as an auxiliary channel.
Fold-0 test: fit the discounted covariance on the training membership only. On fold-0, compare a linear readout of the four coefficients with the current GRU scores on the same rows. Also publish the off-diagonal correlation of the coefficients. This is a CPU screen of minutes, not a training run.
Serving cost: a 16×16 rank-4 update is a small constant per row, on the order of 1 to 3 minutes for the full test set. It is compatible with shrinking the GRU from 33 minutes toward a much smaller residual model. A full 112×112 update every row is the version to avoid.
Status: novel as an interaction. Section 4 lists Gaussian information bottleneck, invariant pooling, and diagonal residual memory as architectures. Here the basis is estimated from the stream, the filters are independent by the passage’s criterion, and the diagnostic decides whether the coupled student is necessary.

6. Translation-invariant book in tick units
Mechanism family: state representation.
Source: no defensible correspondence. Passage 1 rotates a fixed vector, and passage 2 searches formula trees. The symmetry here is a reindexing of the book by price, which those texts do not state.
Structural analogy: absolute price-like columns shift when the market ticks, and column position is an identifier rather than a depth. The invariant state is volume placed at an integer number of ticks from a causal reference price.
Intervention: estimate a tick as the dominant gap among pairwise differences of price-like columns inside the sequence so far. Each row, place bid and ask volume into bins at k ticks from the current mid, for a small fixed window such as k from −10 to 10. Pair volume with price by the known column roles. Feed the histogram, plus the four trade aggregates in the same tick units, to the algebraic F head.
Predicted observable: performance becomes stable across sequences that sit at different price levels. The largest gain should be on rows just after a touch change, where an absolute-price encoding moves every channel at once.
Effect size: 0.005 to 0.02 on the average if the current features are in raw price units and the tick is stable. Near zero if the supplied columns are already relative. That distinction is the result of the test.
Fold-0 test: plot the spectrum of first differences of each price-like column and the occupancy of the rebinned histogram. Score a linear F readout on the histogram against a linear readout on the raw 112-vector, same rows, causal binning only.
Serving cost: a sort and a histogram of the level set is under 3 minutes. It can stand in for a large part of the recurrent input.
Status: novel relative to monotone calibration and semantic group encoders. The state is rebuilt from a measured tick every row.

7. Trade sub-clock inside the book row
Mechanism family: causal-clock decomposition.
Source: no defensible correspondence strong enough to lean on. Passage 2 does use a holding horizon and a backtest calendar that differ from the alpha’s native clock, but it never decomposes an aggregated print. Forcing that into a trade-path estimator would be decorative.
Structural analogy: four trade-price and four trade-volume columns are an aggregate sitting inside one book row. The fair-value shock can occur on the trade clock, while the scored row is a book time. A recurrence that steps once per row has to infer an intra-row path it never sees.
Intervention: treat the four trade slots as an unordered set of prints when their prices differ, ordered by a causal rule such as distance from the previous mid. Update F print by print with a Kyle-style increment, ΔF = λ̂ · signed volume, where λ̂ is a causal regression of mid changes on lagged signed volume. Emit the book-time F only after the four updates. If the four prices are identical, treat them as one print and do not invent a path.
Predicted observable: rows with dispersed trade prices inside the four slots improve. Rows with a single repeated trade price stay flat. The class-3 latent should track cumulative signed trade more closely than book imbalance on that subset.
Effect size: 0.003 to 0.015 on the trade-dispersed subset, and a smaller amount on the average, depending on how often the four slots actually disagree. Uncertainty includes zero if those columns are four summaries (for example open, high, low, last) rather than four prints. The test is required to see which encoding is present.
Fold-0 test: measure the fraction of rows where the four trade prices take more than one value, and the weighted Pearson of cumulative signed volume against the label-implied F. Compare ordering-by-distance with leaving the columns in index order.
Serving cost: under 1 minute.
Status: novel. Event-clock context in section 4 changes the step between rows. This changes the update inside the row.

8. Top-K enter rule matched to the metric weight
Mechanism family: sampling / control rule at inference.
Source: passage 2, “Top K enter threshold Eth 0.0,” together with the report that seed-to-seed differences stayed inside the standard deviation.
Structural analogy: Eth = 0 lets every weak alpha into the book. The competition weight is proportional to the magnitude of the clipped label, so large |F − touch| rows dominate the score. A zero threshold spends the same state update on rows the metric almost ignores.
Intervention: from the training membership, set Eth so that the seed’s |F_seed − touch| exceeds Eth on about the top half of metric weight. When the seed is below Eth, serve a shrunk displacement, F̂ = touch, which is a prediction of zero for both targets after the algebraic map. When it is above Eth, serve the seed plus any admitted residual. State still updates on every row, so causality and the required per-row update are intact.
Predicted observable: weighted Pearson on the high-weight tail rises or holds, and the low-weight body contributes less noise to the pooled correlation. An equal-weight correlation can fall. That split is the signature of a real effect.
Effect size: 0.002 to 0.010 on the official weighted metric, consistent with the +0.0034 IC move in the passage and with the small public gains already seen from same-recipe ensembling. A move of this size does not by itself reach 0.700. Uncertainty of a few thousandths is realistic on one fold.
Fold-0 test: sweep Eth by weight mass, for example 20%, 50%, and 100% of training weight, and publish weighted and equal-weight correlations on the untouched fold-0 panel. Pre-register the selection on training weight only.
Serving cost: a comparison and a scale, far under 1 minute.
Status: a reframing of propensity weights and of the failed quiet-row hold. The threshold acts on the latent displacement at inference, and the official weight defines Eth.

The two passage-backed families are the eigen-expansion (ideas 4 and 5) and the seed-alpha search result (ideas 2, 3, and 8). Ideas 1, 6, and 7 have no defensible correspondence to the random excerpts. The cheapest screen is idea 1’s identity residual plus idea 3’s frozen seeds on one fold-0 panel. If the heads are already coherent and every seed is a weak covariate of F, the saturation sits in the latent itself, and ideas 4 through 7 are the structures worth a fit. If a frozen seed is already close, the 33-minute trunk is the wrong place to spend the CPU budget.
