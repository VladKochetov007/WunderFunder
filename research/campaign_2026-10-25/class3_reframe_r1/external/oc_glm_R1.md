# Class-3 reframe: seven mechanisms from an independent read

The target law makes both labels functions of **one hidden scalar**, `U = F − bid`, plus the observable spread `sp` (`t0 ≈ U − sp`, `t1 = −U`, pinned to `±(sp+2)` on jumps). Since members share 96–99.9% of residual, ensembling is dead; the class-3 deficit (~46% of metric weight at 0.44 vs 0.74–0.81) is information-structure, not averaging. Every idea below exploits that structure.

---

## 1. Virtual-label state machine (changes the target representation)

**Structural reason.** Between quote moves, y is exactly constant; at a quote move with fixed F, `Δt0 = −Δask`, `Δt1 = +Δbid` — fully computable. So a large share of the target stream is *mechanically reconstructable* online; the only genuine unknowns are (a) the run-start branch and level U, (b) rare hidden F-updates. Today the GRU re-estimates a slowly varying level every row through global rank-Gauss encodings, where tick arithmetic (`bid − ask`, runs of `sp`) is not linearly exposed — the exact law's determinism is wasted. Replace with: decoded tick-space quote features + an explicit "virtual label" carried across rows by quote deltas + a small head that predicts only the run-start scalar (branch, U).

**Cheap diagnostic.** Per held-out class-3 sequence: monotone-rank decode the quote ladder; verify run-y constancy and the quote-delta identity row by row (report exact fraction satisfied — should be ~100%); then a ridge/logistic probe on run-start features (each-side size, imbalance, idleness, time since trade) predicting run-level U/branch; compare probe AUC to the GRU's 0.74.

**Falsifier.** Law reconstruction < ~90% of rows (in-run drift exists), or probe AUC ≤ 0.74 as-is.

**Serving cost.** O(1) integer bookkeeping + one small head; deterministic, correct reset per sequence.

## 2. Branch-first classification with exact jump constants (locked-book identity included)

**Structural reason.** With |y|-weighted Pearson, rows where F exits the spread carry |y| ≈ sp+2 — the dominant metric mass — and their targets are *exact constants*, not regressions. That makes class 3 a cost-sensitive 3-way classification (inside / above / below) with known payoffs, plus a small within-branch level regression. Sub-split the weights: if `sp = 0` or negative (locked/crossed book), the "inside" branch is *impossible* — branch is partially known for free on those rows.

**Cheap diagnostic.** Frequency of locked/crossed states in class 3 and their realized branch distribution (conditional oracle); loss-calibrated tri-action decision on held-out (choose per-row in {inside, up, down} a fixed action minimizing weighted loss) vs the GRU's continuous prediction; isotonic recalibration of predicted jump probabilities.

**Falsifier.** Recalibrated/gated tri-action yields ΔWP ≤ +0.005 over the current endpoint, or the locked-branch conditioning flips sign under exposure checks.

**Serving cost.** Head swap + lookup tables; ~zero.

## 3. Queue-depletion race with Hawkes-style trade clustering

**Structural reason.** Large tick ⇒orders sit queued at the tick with few trades; branch outcome is a depletion race: does the bid (ask) queue empty before the far side's quote re-anchors? Visible top-of-book size changes *without* quote moves reveal cancellation pressure. Class 3's low trade rate makes each trade carry more per-event information, and trade arrivals cluster (Hawkes) — a burst at one side is a depletion precursor.

**Cheap diagnostic.** Zero-fit: bin next-run branch by simple trailing rules — signed trade imbalance over last k events, trade-rate z-score, bid/ask visible-size ratio, count of size-decrease events without quote change — and read monotonicity of the bin curves on held-out sequences; then a ~10-feature logistic probe.

**Falsifier.** Bin curves flat (no monotone gradient) or probe adds no lift over the GRU's own branch head on matched rows.

**Serving cost.** Trailing-window counters, causal; ~zero extra latency.

## 4. i1 lead–lag common-factor conditioning

**Structural reason.** Two instruments on the same venue share a common factor with one-sided lead–lag. A tick move in i1 (or a burst of i1 trades) shifts the conditional distribution of i0's next branch/level; inequality is asymmetric when one instrument is thinner. The GRU sees i1's *encoded* levels as generic inputs; the mechanism isolates i1's **signed deltas and event timing**, which the global nonlinear encoding hides.

**Cheap diagnostic.** Cross-correlation/slope table on held-out runs: i1 Δ-features at lags {−5 … +5 events} vs i0's run-level branch and U; spike structure at a specific lag (and stronger where i1 changes between i0 events) is the signature. Also lagged mutual information `MI(branch; i1 features at lag)`, and state-specific curves (gate on whether i1 led the last i0 move).

**Falsifier.** Flat response across all lags (|effect| below ~0.02 on branch prob), or signal survives only contemporaneously (undistinguishable from a global time factor — check against shuffled-i1 controls).

**Serving cost.** Maintain two causal EMAs of i1 returns and one i1 event timer; O(1) per event; deterministic.

## 5. Regime-split: change-anticipation specialist vs carry specialist

**Structural reason.** The brief's own numbers — AUC ≈0.83 within 20 rows of a change vs 0.55–0.60 far away — imply two different inference problems: a **transient** one dominated by events forcing reallocation, and a **steady-state** one where y is nearly constant ("carry"). One GRU with shared capacity compromises between them; also the reported error structure (offset corr priorities 0.82 at run start falling to 0.60 after 16+ rows) points to regime-dependent miscalibration rather than a single bias.

**Cheap diagnostic.** Bucket held-out residuals/deltas by run age and by distance-to-next-unobservable event (allowed only causally at serving — gate on *time since previous* event, not time until the next); fit per-bucket multipliers/offsets and simulate the gated mix's WP. Look for systematic sign flips: e.g., transient underprediction of jump magnitude vs steady-state overprediction.

**Falsifier.** No per-bucket offset pattern (deltas < 0.005 weighted scale) unexplained by noisy branch cases; no replicate stability across seed splits.

**Serving cost.** Trivial — two read-only state vectors + a gate on an existing timer.

## 6. Hidden-liquidity refill signature (unconventional feature channel)

**Structural reason.** In large-tick books, visible top-of-book size often *recovers* within a few rows after dropping — refill consistent with iceberg/refreshing liquidity.Persistent refill at a side means orders sit behind the top; that side is more resilient than visible imbalance suggests, directly changing the branch odds. The current features are levels/EMAs; the **sequence-shape** of size around drops without quote moves (drop magnitude, refill latency, refill fraction) is not represented.

**Cheap diagnostic.** On held-out runs, classify drop events {with-trade, without-trade} and their refill behavior within a few rows; tabulate next-run-branch rates per class (confusion-free: these are summaries of sizes, not of y); compare to the GRU's conditioned branch probabilities on the same slices — look for slices where the simple rule beats the model, and quantify the covered probability mass.

**Falsifier.** In class 3, top-of-book sizes almost never move between quote events (nothing to classify), or refill classes carry no branch-odds gradient across two folds.

**Serving cost.** A couple of ring buffers + counters; O(1); deterministic.

## 7. Generative Bayesian filter over latent branch/level state (unconventional inference)

**Structural reason.** Treat the problem as *filtering with near-exact physics*: latent state = quantized U level and branch (a small grid, e.g., ±(sp+2), sp, few interior levels); transitions encode "F updates roughly one event after it changes", quote events update observations exactly per the law; the GRU supplies the observation likelihood/proposal (data-driven), but the **exact law is imposed as hard structure**, not learned. Enforces run-level consistency the GRU lacks, propagates uncertainty in the metric's idiom (a distribution over U, integrated to predictions), and roughly maps to how the labels were generated — the strongest form of inductive bias available here.

**Cheap diagnostic.** Posterior inference with **no simulation**: take the GRU's raw branch/level head outputs as pseudo-observations on held-out sequence, run the small state-space filter over discrete U states, and integrate the posterior to `p0, p1`. No training needed if heads already emit per-branch probabilities. Whiten residuals per sequence to verify no remaining serial structure.

**Falsifier.** Filter needs unrealistically heavy transition noise (U jumps nearly every event — i.e., the law's persistence assumption fails in class 3) or adds < +0.003 WP over the raw head. Cheap enough to falsify first.

**Serving cost.** Small — state space ~4 levels × 3 branches; a vector-matrix multiply per event; strict-fp32 CPU; deterministic and reset-safe.

---

## Assumption most likely wrong

**The framing that the missing information is the run-start latent (branch/level) — i.e., that y is piecewise constant within a label run.** The brief is internally tense: it says 94% of residual is a *per-run constant* offset, yet error *grows* with rows since the last change (0.82 → 0.60 correlation decay). Both can hold only if F keeps moving relative to quotes *between* label changes — which breaks `t0 = U − sp` in-run exactness and means the branch/level latent is mis-specified: part of the deficit is **within-run drift**, invisible to every run-boundary fix (mechanisms 1, 2, 7 as stated). The decisive zero-fit check: verify directly on held-out sequences whether y is exactly constant between detected label changes, or within-run drift is nonnegligible — and if it is, whether a "quote-adjusted carry" prediction (last run's value mechanically adjusted) explains more residual than per-run branch assignment.

---

**Cheapness ranking if run sequentially:** law-replay decode (gates mechanisms 1, 2, 7) → carry-vs-branch prediction comparison → per-bucket recalibration → locked-book oracle statistics → i1 lag scan → size-drop/refill event study → Hawkes probe → full filter. Only the last two (Hawkes probe support, filter) require more than trivial compute; the gates first.
