# Brainteaser idea filter after event EDA

Four gpt-6-sol-medium literature scouts sampled selected books in
`~/Documents/Trading/` from microstructure, time-series, quantitative research,
and random trading-note angles. The parent agent performed the measurements;
the scouts did not edit or run experiments. This table preserves both useful
and high-variance suggestions, with the new EDA acting as the filter.

**Public-winning idea types so far.** A: exact scorer/deployment parity (the
prediction clip fix was +0.0568 public). B: allocate learning to scored-like
states (propensity, true-mask fine-tune; smaller recent gains). C: reduce fit
variance without extra serving work (EMA was +0.0017 public). D: add an
independent strong predictor (baseline blend +0.0021 early, two GRU seeds later).
The mixed six/four result was −0.0006 public and a **local exact-mask tie**;
λ=.02 manufactured +0.00065 of apparent lift. Exact mask and chain-paired
replacement scores are therefore the primary gate.

The strongest new allocation result is class 3: 46–47% of target weight,
0.411/0.436 within-class WP, and +0.111/+0.108 *global* fold WP from a
noncausal true-sign oracle in that class alone. A true-magnitude oracle with
the pair's wrong signs loses global WP. A direct sign head raises weighted
AUC to 0.824 from the pair's 0.793 on fold 0, yet its incremental
cross-fitted Pearson correction is flat. The research target is therefore
**class-3 conditional future movement with calibrated uncertainty**, not
another hard sign gate or blanket size boost.

| Idea | Type | EDA/ledger filter | Cheapest useful next test |
| --- | --- | --- | --- |
| Train-only class-3 tier head | C/B | High-weight tiers concentrate in few episodes. The matched .1 auxiliary gave only +.000040±.000374 exact-mask pair lift on pure fold 0. | **Stopped at predeclared gate**; no fold-1 fit. |
| Train-only observable 8–24-event quote-path head | C | Eight-event future mean reconstructs ~92–93% of reference prices per episode, but the matched 8/24-event auxiliary **lost 0.00144 ±0.00105 exact-mask pure-pair WP** on fold 0 (E-251). The future-next-event pair output's correlation .460/.478→.755/.733 is a noncausal information-arrival oracle; 98.55% weighted side agreement does not provide an advance signal. | **Stopped before fold 1.** The high-fidelity teacher was redundant or destabilizing for a current-input forecast; require a new causal residual signal before another quote-path head. |
| Joint side × continuation head | C | Winner side ≈ first future move, but a matched sign × positive-size GRU lost −.000499 in the exact pair as a soft gate and −.012041 as a strict gate. A direct current-target-sign auxiliary improved the single +.000656 but gave only +.000468±.000712 as a partner, below gate. Its unused logit ranks direction better (AUC .824 versus .793) but adds essentially zero chain-cross-fitted Pearson lift. | **Stop both sign-head variants**; an observable future-path continuation head would need a distinct mechanism. |
| Scoring-mask auxiliary on shared trunk | B | Matched fold-0 mask BCE gave only +.000168±.000299 exact-mask pair gain, below the fixed +.0005 gate. | **Stopped before fold 1**; no mask-loss tuning. |
| Train-source group robustness | B/C | Train/valid adversarial AUC ~.99; simple source terciles and class routing are unstable. | Predefine causal input-summary groups on train, train one capped group-weight arm, check positive exact-mask lift in each held-out block. |
| Compact competing-risk survival head | C | Event-gap AR(12) has ~.255 held-out R² and ~.15 scored partial timing correlation after age, but its exact global correction gains only +.00012–.00018. Existing aux-next already supervises time and direction. | **Low priority** unless a new directional cue appears in the rare long trend-break runs. |
| Event-clock seasonal correction | D | Raw class-3 price codes recur at lag six on 69.6% of sampled events, but a 12-return-lag residual correction reverses sign across folds and a phase-only correction adds +.00006. | **Stop readout tuning**; the GRU already tracks the routine oscillator. |
| GARCH error-volatility gate | D, wild | ARMA/GARCH fit hindsight residuals well, but past true errors are unavailable at inference. Past observable quote volatility has opposite-sign association with error variance across folds. | **Diagnostic only**; require a causal variance proxy with matching signs before training a head. |
| Trend-break first-passage expert | D/C | Just 2.8–2.9% of high-error runs lasting 6+ book events contain 92–94% of scorer-weighted class-3 contrast error. A three-event seasonal-innovation statistic has negative weak residual correlation; raw lag-six price-code mismatch count has partial correlation only .015/.078 across pure folds. | Search for an *observable pre-break* signal in book queues/trades; require partial residual correlation ≥+.08 in both folds before GPU work. |
| Event-transition innovation auxiliary | C, new | ARMA(1,1) nearly whitens event errors retrospectively, whereas row labels overlap heavily. However the future-next-event oracle jumps in accuracy only after the event's information arrives; a derivative loss could demand an unpredictable jump. Pure differencing would also suppress long weighted error runs. | Lower priority than a quote-path supervision screen. If tried, keep weighted level loss and preregister one small auxiliary against a matched fold-0 control. |
| Past event-rate and reversal inputs | D | Corrected out-of-fold one-snapshot tier probe showed AUC +.019/+ .042, but the three-feature exact-row residual correction on all class-3 sequences changed score +.00176/−.00056 across folds; drift-only was −.00056/+.00095. | **Stopped** as a posthoc correction; a future input-feature arm needs a new mechanism and untouched validation. |
| Causally decoded depletion/refill ratio | D | Raw i0/i1 volume-code shocks lost 0.01–0.04 side AUC in earlier event probes; literal queue meaning is not guaranteed. | Train-only transition dictionary, then event-level residual AUC ≥+.03 and partial corr ≥+.08 in both folds before GPU. |
| Phantom-wall cycle count | D, wild | A completed add/remove-looking volume-code cycle might flag adverse selection; it could also be a code artifact. | Matched current code, age, and GRU side score; require stable side/tier residual across chain halves. |
| i1 innovation, not i1 level | D, wild | Earlier lead effect shrank to weak/unstable; i1 transition decoding has low coverage. | Restrict to fully decoded transitions and test surprise residual on both folds; reject if coverage or sign fails. |
| Event-pooling second clock | D/C | Existing dual-clock GRU was weaker; may still create a distinct error source, but extra serving cost matters. | Parameter-matched event-pool vs event-count input on one fold, then CPU runtime gate. |
| Randomly offset BPTT boundaries | C | Chunk changes have mixed/weak results; a long-quiet class-3 state could still suffer boundary bias. | Equal-step randomized-offset control, then occasional two-chunk backprop; stop if pair score is flat. |
| N-BEATS-style history backcast | C, wild | A current GRU state may represent history poorly, but reconstructing past could become a copying task. | Auxiliary history prediction vs current-input-only control, matched params. |
| Counterfactual non-price-event deletion | C, wild | Could encourage invariance to nuisance events; could erase order-flow information. | Compare to ordinary input dropout at the same deletion rate; check scored event rows separately. |
| Episode-overlap graph weighting | B/C | Earlier inverse-sqrt episode weighting increased effective counts but lost −0.00083 on fold 0. | First show a new graph changes independent-outcome geometry versus E-212; no retraining on another scalar weight without new evidence. |
| Online pseudo-label / recalibration | A/C | Four-event reconstruction misses large payoff weight; eight-event exact weight is only 63–71%. E-080 online updates already hurt. | Do not try until a causally matured label reaches >95% *metric-weight* accuracy and latency makes recalibration useful. |
| Distributional class-3 output replacing wMSE | C | Outcome-code CE already lost to weighted MSE. | Consider auxiliary only; keep metric-aligned point head. |
| Packaging path fix | A | Old ZIP uses both GRUs on v2 and v3; mixed ZIP serves one on v2. Local fallback penalty −.00215 exact-mask reserved. Public CPU path unknown and local same-host runtime ratios do not identify it. | Require an inference package that serves intended members under the organizer CPU path and fits a measured time margin. |

Source prompts, not empirical support: Cartea–Jaimungal–Penalva,
*Algorithmic and High-Frequency Trading* §§4.4, 10.4, 12.2; Harris,
*Trading and Exchanges* ch. 6 and §13.7.2; Paleologo, *The Elements of
Quantitative Investing* §§8.1–8.2; Nielsen, *Practical Time Series Analysis*
ch. 11; N-BEATS; and selected PDFs in `~/Documents/Trading/Naive/` on alpha
pipelines, order-book visualization, stationarity, and time-frame crowding.
The exact competition target law is undisclosed; 8-event and 20-event fits
are descriptive future-oracle operators, not known production targets.
