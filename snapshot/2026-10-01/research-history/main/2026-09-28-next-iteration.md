# Next research iteration: forecast the event that reveals the payoff

## Mechanism established on pure chains

Class 3 carries 46–47% of scorer target weight. Its current pair scores
0.411/0.436 within class on the two pure development folds. A noncausal
true-sign oracle for that class alone adds 0.111/0.108 to *global* WP;
providing true magnitudes with the wrong signs lowers global WP. The first
future decoded mid-price move agrees with target direction on 98.55/98.56%
of weighted eligible episodes in the 128-sequence snapshot. On scored rows,
the current pair contrast correlates with the target at 0.460/0.478, while
the same trained pair after the next book event reaches 0.755/0.733. That
future output uses future input. It is an information-arrival diagnostic,
not a serving trick.

The pair is already extracting some intra-event information: freezing its
prediction at episode entry yields class-3 WP 0.119/0.161 on those same
scored rows, versus 0.415/0.436 at the actual rows. A direct current-sign
auxiliary increases weighted AUC 0.793→0.824 on fold 0, but barely changes
threshold accuracy or pair WP. A causal snapshot booster with raw features,
the pair output, and up to 64 past book events improves next-jump AUC on
only one fold. The missing gain is an **incremental conditional expectation
of the next quote path**, especially during long quiet stretches and rare
large trend breaks.

## Completed arm and falsifiable next ideas

1. **Training-only future quote-path auxiliary (rejected E-251).** Decode
   class-3 price paths using the train-fitted transition map. Add 8- and
   24-book-event mean displacement targets, each observed only during
   training, to the existing GRU. Preserve the ordinary weighted MSE and
   two-output inference path. Evaluate the fixed epoch-3 fold-0 pair against
   its matched parent: at least +0.0005 WP at both exact mask and λ=.02,
   chain bootstrap, then fold 1 and reserved chains. The future-label
   contrast correlations 0.95/0.93 are strong but proved redundant for this
   recipe. Its pure-chain pair replacement lost 0.00144 ±0.00105 exact-mask
   WP and 0.00133 ±0.00100 at λ=.02. Stop before fold 1.
2. **Branch-conditioned soft payoff mixture (next architecture hypothesis).** The prior sign×positive-size
   head forced symmetric magnitude and lost score; its strict gate lost
   0.012 global WP. A genuinely different head would estimate
   `p_up`, `E[(y0,y1)|up]`, and `E[(y0,y1)|down]`, then output the soft
   expectation `p_up*mu_up + (1-p_up)*mu_down`. Keep the main metric-aligned
   weighted MSE, with small class-3 branch losses and no hard threshold.
   Compare parameter-matched fold-0 pair replacement with the same gate.
   The extra branch channels must earn their inference cost before a native
   port.
3. **New causal intra-event order-flow information (highest information priority).** Use train-only decoded
   queue removal/refill and bid/ask depletion *relative to the expected
   six-event cycle* as candidate signals for the next-event side. Integrate
   activity across the current quiet interval and test a signed imbalance
   or hazard-weighted CUSUM, not just raw volume shocks. Earlier raw shock,
   seasonal return, cycle-mismatch, and quote-volatility proxies failed.
   Require weighted target-side AUC improvement and partial residual
   correlation at least +0.08 in **both** pure folds before another GPU
   model. Check at all scored rows, not only event snapshots.
4. **Scorer-aligned validation and variance reduction.** Use the exact
   clipped, pooled WP, true scoring mask, disjoint chains and a paired
   bootstrap. The 1% highest-weight episodes contain 42–44% of sampled
   class-3 target weight; local lifts below about 0.0005 are fragile.
   Favor training changes that strengthen both current ensemble members
   without increasing serving time. Preserve a measured native CPU margin.

Residual ARMA/GARCH describe realized errors but need past targets that are
unavailable at inference. The six-event quote oscillator is real, yet a
phase correction adds only 0.00006 WP. The 12-lag duration forecast explains
about 25% of event-gap variance but changes global WP by under 0.0002.
Bid/ask VECM mostly re-expresses one drifting price component and bounded
spread. Those results guide feature or loss design; they are not substitute
predictions.

Primary evidence: [current frontier](current_frontier_2026-09-27.md),
[marimo notebook](../notebooks/brainteaser.py),
[experiment ledger](experiments.jsonl),
[next-event oracle](artifacts/brainteaser_scored_next_event_oracle.json),
[class allocation](artifacts/brainteaser_class_oracle_allocation.json), and
[quote-path label check](artifacts/brainteaser_quote_path_label_quality.json).
