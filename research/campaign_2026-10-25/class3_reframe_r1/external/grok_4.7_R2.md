Class 3 is a three-way branch problem wearing a regression loss. The stated law makes \((t_0,t_1)\) a function of spread and of where \(F\) sits relative to the quotes, and a branch oracle at weighted Pearson (WP) 0.968 means almost all score is in that branch. The live GRU is at ~0.44, with branch AUC ~0.55–0.60 until the last 20 rows before a price change. Correlation falls from 0.82 at a label-run’s start to 0.60 after 16 rows, and 94% of the residual is a constant offset on each run. Quote moves already announce label changes (AUC 1.0). What is missing is \(F\)’s move relative to those quotes, then the discipline to hold that inference while the label stays constant. Shared GRU residuals (96–99.9%) say another GRU will not supply it.

### 1. Predict absolute \(F\), emit labels by algebra
**Target change, class 3 only.** \(t_0=F-\mathrm{ask}\) and \(t_1=\mathrm{bid}-F\) mix a slow latent with a jumpy quote. Every quote update rewrites both labels even when \(F\) did not move. A GRU that outputs the differences re-solves that identity on every row, then shrinks it. Maintain \(\hat F\) in tick units. Update it only when the encoded quote changes. Set \(t_0,t_1\) from the closed-form branch law and the current spread.

**Diagnostic.** On held-out sequences, recover integer spread by clustering encoded ask−bid (one global monotone encoding; the smallest positive cluster is one tick). Project current predictions onto the nearest legal \((t_0,t_1)\) for that spread and the three branches. Report WP before and after projection, and the fraction of rows already off-lattice.

**Falsifier.** Projection does not raise WP, and branch-implied labels disagree with realized labels on more than a few percent of rows.

**Cost.** A few comparisons and a latch per row. No extra net.

### 2. Latch the post-event forecast for the whole run
The age curve is the signature of a forgetful state scored against a piecewise-constant target. Within a run the label does not move, so any within-run wiggle in \(\hat y\) lowers pooled correlation. GRU forget gates do exactly that, and per-row \(|y|\)-weighted MSE rewards shrinking toward 0 on the long, uncertain rows that dominate the run.

**Diagnostic.** Zero-fit, causal: at each label change, store the model’s prediction; copy it forward until the next quote-encoding change. Compare WP to the live path, split by run age 0, 1–15, and 16+.

**Falsifier.** The held path loses to the live path on held-out sequences. Later rows then contain real level information, and the decay is not self-inflicted.

**Cost.** One stored vector. Training change if you also want it learned: one loss term per run (run-level concordance), updates on event rows only.

### 3. Depletion race, not imbalance EMA
On a large tick the next quote change is a race between two queues. Direction is the hitting probability of bid size versus ask size under recent consume and cancel rates (Cont–de Larrard), not an EMA of \((q_b-q_a)/(q_b+q_a)\). Far-from-event AUC ~0.55 is what you measure when that functional is missing: the book’s information is a rate ratio, and it is weakest exactly where this model family sits near chance.

**Diagnostic.** Trailing per-side depletion rates from trades and size decreases, no fit beyond counts. Score
\[
P(\mathrm{up})=\bigl(1+\tfrac{\lambda_a}{\lambda_b}\tfrac{q_b}{q_a}\bigr)^{-1}
\]
only on rows with age ≥ 16, against the GRU’s branch score.

**Falsifier.** Formula AUC ≤ the GRU’s AUC on those rows, inside the same spread bucket.

**Cost.** A handful of counters. CPU-trivial.

### 4. i1 as the fine-grained twin of \(F\)
Class 3 is low-trade and slow. If i1 is the active, smaller-tick name on the same latent, i1 moves while i0’s quote is stuck, which is precisely “\(F\) relative to i0’s quotes.” Nothing in the i0 aux heads (spread, branch, class, queue EMAs) forces the net to use i1 as a leader, and a shared GRU residual is what you see when every member ignores the same cross-section.

**Diagnostic.** Zero-fit lead test: sign of i1’s encoded-price change and i1’s trade sign over the last \(k\) i1 events, scored against i0’s current branch and against i0’s next branch. Residualize on i0 spread, i0 queue ratio, and run age (partial AUC). Stratify by whether i0’s own quote just moved.

**Falsifier.** Partial AUC ≤ 0.5 in every age bucket.

**Cost.** A few EMAs already available in a two-instrument stream, plus a three-way gate. Negligible next to the GRU.

### 5. Reload state for hidden size
Visible size on a large tick overstates residual liquidity when an iceberg refills. A queue that drops and is refilled is not in a depletion race; a queue that drops and stays down is. Raw size and a size EMA confound the two, so the race statistic in (3) is biased exactly on the long runs (34% zero labels, low trade rate) that carry the age decay.

**Diagnostic.** Define a reload as a best-level size increase with no encoded price change, within a few rows after a size decrease. Inside a fixed visible-size decile, compare P(run survives another 16 rows) and branch AUC with versus without a reload on that side.

**Falsifier.** No separation inside size deciles.

**Cost.** Two counters per side (last depletion row, refill count).

### 6. Event-key codebook instead of a continuous net
Large-tick transitions live in a small discrete set: quote-jump sign, spread before and after, trade print at bid / ask / through (equality tests in encoded space), reload flag, i1 jump sign, queue-ratio tertile. A GRU averages over keys that should not be averaged, which yields one shared residual. A table does not.

**Diagnostic.** Hash those bits on train sequences. For keys with count ≥ 50, store mean \((t_0,t_1)\) of the following run. Apply the table on held-out event rows and hold it across the run as in (2). Compare WP to the class-3 base rate and to the GRU.

**Falsifier.** Key means do not beat the class-3 mean out of sample. The combinatorics are empty and the signal is elsewhere.

**Cost.** A few thousand keys, one lookup per quote event, hold between events.

### 7. Short-horizon Hawkes surprise, separate from the race level
The race in (3) is a slow level of intensity. Clustered cancels and trades are a fast residual: a burst of ask-side cancels raises near-term ask intensity above its own trailing rate. That surprise is what the last-20-row AUC of ~0.83 is made of, and a single queue EMA does not represent it.

**Diagnostic.** Per side, count events in the last \(k\) rows minus the trailing EMA of that count. Partial AUC for the branch, given the race probability from (3) and given run age.

**Falsifier.** Partial AUC ≤ 0.5 once (3) is in the conditioning set. Then the near-event AUC is just the queue level, and the extra state is wasted.

**Cost.** Four EMAs.

### Assumption most likely wrong
That branch AUC ~0.55–0.60 far from a price change means the direction is not in the data yet. The same brief says the label is piecewise constant, announced by the quote move at AUC 1.0, and predicted at 0.82 on the first row of the run. Scored against a sticky regime, a forgetful net produces ~0.55–0.60 in the middle of a run by losing a value it already had. Those two readings imply opposite fixes: invent early-race alpha, or stop forgetting a revealed level. The age curve supports the second. I would also re-derive the outside branch before trusting it: under \(t_0=F-\mathrm{ask}\) and \(t_1=\mathrm{bid}-F\), a one-tick step through the touch is \(\mathrm{sp}+1\), while the brief states \(\mathrm{sp}+2\). An off-by-one there quietly caps the “branch oracle 0.968” and shows up as a level offset.
