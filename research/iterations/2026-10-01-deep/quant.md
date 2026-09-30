# Quant branch: what is recoverable, and what would actually be useful

**Completed decision:** the fixed 11-line dictionary failed its preregistered geometry gate: only 6.91–11.93% of changes have two-atom reconstruction error at most10%, versus80% required. Do not allocate the proposed inverse-coordinate feature training from this result. A literal queue-depletion model remains underidentified; a useful causal coordinate is a separate, still permissible hypothesis. Failure to name a coordinate “best ask volume” does not disqualify a useful feature, and successful reconstruction would not identify a physical queue. The detailed outcome and positive-control limits are in section8.

This dossier replaces the preliminary scout's overly strong implication that physical identification must precede every predictive feature. It also narrows its dictionary claim: the proposed computation checks a particular cheap global linear representation, not whether the anonymization is invertible in principle. The diagnostic plan is [quant-plan.json](quant-plan.json); implementation is [quant_geometry.py](quant_geometry.py). No validation score or new training is part of this round.

## 1. What the inputs actually warrant

The organizer describes two L11 books and explicitly identifies the 11 bid and 11 ask volume-like groups. Within a group, column index is a persistent representation component, **not depth rank**. Neither a linear mixing matrix, monotonicity, physical zero, nor an inverse transformation is specified. Thus splitting the 22-column block by side is justified; calling the first column the best queue is not. [Official data overview](https://wundernn.io/connectome/docs/data_overview)

`src/wnn/data.py` encodes precisely these side slices. The old [brief](../../brief-2026-09-25.md) says features *look* rank-Gaussian and records a −5.19934 sentinel. That is a distributional observation, not a recovered transform. If `z=Phi^-1(F(q))`, applying `Phi(z)` gives a percentile, not the original quantity `q`; the unknown `F^-1` remains. If a mixing precedes componentwise transforms, even that coordinatewise percentile does not isolate queues. We will not silently apply a guessed inverse CDF.

E090 supplies a useful constraint: volume levels are not approximately one-dimensional (8 components per side for 90% variance), and many columns change together. But PCA rank and ICA kurtosis neither demonstrate a linear mixing nor uniquely identify a best queue. ICA imposes independent latent components; real queue levels can be dependent. The reported ICA/raw direction AUCs, .6202/.6202, and imbalance .6214 give no evidence that a level rotation alone adds useful information. The original scratchpad scripts were not found in the repository/server checkout during this audit; these exact measurements are ledger evidence, not independently reproduced measurements.

## 2. The sparse-increment premise, stated mathematically

Let `z_t^s in R^11` be the observed volume-code vector on side `s`, and `q_t^s` an unobserved physical queue vector. A fixed affine code would imply

`z_t = A q_t + b`, hence `d_t = z_t-z_(t-1) = A (q_t-q_(t-1))`.

If exactly one queue changes, nonzero `d_t` lies on one of the 11 lines spanned by columns of `A`; two simultaneous changes lie in a two-column span. This is the precise reason a line dictionary is worth testing. Full-rank covariance alone does not test it.

There are several distinct ways this premise can fail:

| Issue | Consequence for observed increments | Consequence for interpretation |
|---|---|---|
| Componentwise nonlinear transformation | Locally `d ≈ J_g(Aq) A Δq`; directions depend on the starting state | A global straight-line dictionary can fail even if a nonlinear inverse exists |
| State/sequence-dependent normalization | Changing level or scale can move many coordinates with no corresponding queue event | A stable global decoder cannot be presumed |
| Several updates between rows | `Δq` sums additions, cancellations and executions | Endpoint differences cannot count separate events; a zero net change can hide high activity |
| Sentinel transition, rounding, tiny differences | Creates artificial rays or unstable relative-error denominators | Missingness and precision must be reported separately |
| Price movement / depth relabeling | A coordinate at depth `j` can refer to a different price after the move | An apparent volume jump need not be a physical insertion/removal |
| Correlated latent updates | Many dictionaries reconstruct nearly equally well | Sparse reconstruction alone does not make factors unique |

The diagnostic therefore trains only on observed **price-code equality** between adjacent rows. This removes obvious price-representation changes; it is not a theorem that an unchanged code uniquely fixes every latent absolute price. Price-code-changing rows are a separately reported stress set. No crossing event is labeled an insertion, cancellation or market order.

Even with a perfect invertible `A`, one retains permutation, scaling and sign ambiguity, and increments do not identify the offset `b`. In particular, integrating increments cannot locate physical zero, and clipping an arbitrary factor at zero does not recover a depleted queue. Stable unnamed coordinates can still be prediction features, provided their orientation is fixed once from training and their semantics are not overstated.

## 3. What the queue papers require that we do not have

The queue-reactive model's sections 2.1 and 2.3.2 use queue sizes at identified price offsets, classified additions/removals/trades, and inter-event durations in **seconds**. Its conditional arrival-rate estimates use waiting times and event types; section 2.5 allows dependence on both queue sizes. Recordings restart when the reference price changes. These requirements explain why an EMA of anonymized endpoint changes is not automatically an estimate of its physical intensities. The paper itself treats order-size simplification and relabeling explicitly. [Huang–Lehalle–Rosenbaum, full text §§2.1, 2.3.2, 2.5](https://arxiv.org/html/1312.0563v1)

The simpler queue-race model assumes a one-tick spread, equal-sized events and independent arrival processes, and resets to a new queue-size distribution upon depletion. Its reduced state combines cancellations and market orders because both remove queue size. This makes a removal-versus-addition feature plausible without distinguishing cancellation from execution; it does **not** remove the need for valid size coordinates or correct treatment of reset events. [Cont–de Larrard, full text §2.2, pp.4–5](https://arxiv.org/pdf/1104.4596)

These are modeling assumptions and motivating mechanisms, not evidence that this competition is generated by those models. Event-clock positive/negative code flow is allowed as a predictive feature; it should be called that, rather than a fitted exchange-time queue intensity.

## 4. The bounded diagnostic and why its controls matter

Q-DEEP-01 selects one representative per chain, 16 chains from each historical market class, then partitions whole chains into two halves. Class metadata only balances a diagnostic sample; it is never an inference input. All data are previously exposed. At most 256 quiet increments per sequence/side are retained, with each sequence contributing total fitting weight one. This prevents an active chain from defining the entire dictionary. It intentionally answers a geometry question, not the target-weighted competition objective.

The exclusion threshold is fixed: nonzero increment norm must exceed `1e-7`; both endpoints must be finite and have absolute volume-code values below `5.19`. Counts of tiny/zero updates and extreme/sentinel exclusions are recorded; excluded rows do not enter the reconstruction denominator. A sentinel threshold can also discard genuine extreme values, so its exclusion fraction is material. Nonzero simultaneous changes on both sides are counted without declaring them synchronous exchange events.

Each fitting half supplies its own uncentered RMS scaling and second-moment PCA basis. The candidate uses 11 unit line atoms, PCA initialization and exactly 20 hard-assignment/eigenvector updates. There is no restart search. The other half is projected onto each individual atom and all 55 atom pairs. Near-singular pairs retain the one-atom solution. Raw axes, the PCA basis and one fixed random orthonormal basis have the **same atom budget and free reconstruction coefficients**.

The proposed gate combines accurate sparse reconstruction, cross-half atom stability, conditioning, and an advantage over the stronger raw/PCA control. If raw coordinates already have the same sparse geometry, a new learned decoder has not earned a downstream fit. If all bases work because changes occupy a small subspace, the same conclusion holds. Per-class, state-magnitude and trade-indicator distributions distinguish global failure from a pattern confined to one regime; they do not authorize fitting a new per-class dictionary after the result.

A failed gate can reflect a bad local optimum, simultaneous updates, nonlinear encoding, unstable scaling or simply no advantage over cheap coordinates. It rejects this **one affordable representation recipe**, not all possible factor models. A passed gate only nominates a prediction experiment: deterministic transforms add no information to the full causal input history, though they may make useful nonlinear statistics easier for a finite trained model to estimate.

## 5. A concrete later candidate/control study, conditional on the gate

The following is a proposed next study, not executed or fully registered in this round. Freeze the selected global dictionary without validation feedback. With `D` the 11-by-11 atom matrix in scaled space, let

`u_t^s = (D^T)^-1 S^-1 (z_t^s-z_(t-1)^s)`.

Use a fixed orientation: the largest-magnitude loading of each atom is positive; preserve a deterministic atom order. At each valid, price-code-unchanged volume event update, for each coordinate,

`r_(t,+) = (1-a) r_(t-1,+) + a max(u_t,0)`,
`r_(t,-) = (1-a) r_(t-1,-) + a max(-u_t,0)`, where `a=1-2^(-1/32)`.

Between volume events hold these values. At a price-code change or invalid endpoint set these **feature states** to zero and flag the reset; do not reset the GRU. This prevents a price-level switch being integrated as flow, at the cost of losing cross-interval memory. Reset all state at the sequence boundary. Negative/positive here means coordinate direction, not economic depletion/refill.

The candidate adds these 44 side-by-coordinate half-wave flows and one shared validity/reset flag to the existing reference inputs. A matched control has the same dimensions and update rules in raw scaled coordinates; a second control uses the fixed PCA coordinates. The model width, train data, steps, loss, checkpoint rule and seed pairs stay fixed. Raw input and existing event features remain present in every arm. This is a three-recipe allocation, so it must be explicitly budgeted before training; do not select a favorable control afterward.

Why half-wave accumulation matters: an invertible linear map alone leaves the span of a linear readout unchanged. Applying `max(±u,0)` **before** accumulation makes separate opposing flows different from the failed net displacement `z_t-z_anchor`. Equal-dimension raw/PCA controls ask whether learned directions, rather than just extra history or parameters, are useful. A GRU already has nonlinear gates and sees volume differences, so the prior for incremental improvement is modest.

For a future predictive screen, fit each fixed readout on opposite exposed folds, report all three recipes, and judge full eligible-row pooled WP plus contribution to fixed C, not retrospective quiet-interval midpoints or geometry alone. If a training study is warranted, use matched seeds rather than treating rows or ensemble members as recipe replications. C is an exploration reference; it is not silently identical to the public V1 two-epoch-head package.

Serving costs are bounded but unmeasured: two 11-by-11 transforms on volume changes, 44 EMA states and 45 additional model inputs. Dictionary/state storage is kilobytes, well below the 20 MB package limit; added input weights and row latency still require native measurement and parity. Do not add a fifth full model to C to host this feature. Exact geometry passing does not guarantee the proposed latency target.

## 6. What historical negatives actually exclude

| Evidence | Tested treatment | Implication here |
|---|---|---|
| E090 | ICA rotation of volume levels; small class-3 direction probe | No demonstrated benefit from level rotation; no proof of latent queue identity |
| E215 / `probe_i1_flow_shock.py` | Raw 22-code net changes over 4/16 volume events, opposite-fold ridge, completed long-interval midpoints | Strong reason to avoid another net-displacement arm; does not test separate half-wave flow in stable learned directions |
| E088/E089 | Broad rate/depth/history library, residual interactions and GRU inputs | Extra causal statistics are often already captured; requires matched controls and current C, not an old weak baseline |
| E112 / `micro_features.py` | Price-impact-split MO0/MO1 trade EMAs | Do not revive taxonomy EMAs because earlier small samples looked positive |
| Later `quant` / `quanti1` | Since-price-code-event volume displacement and trade-code sums | Full training negatives lower the prior; the proposed state must materially differ from cumulative raw code sums |

The E215 midpoint is selected using a completed interval's future endpoint. That is acceptable as a retrospective diagnostic slice but cannot be the submitted sampling policy; passing a revised midpoint test would not establish all-row score utility. E112's helper identifies trade rows using a whole-sequence median, while its one-row-delayed impact classification is causal at the delayed row. Do not import that median helper into a new serving feature. This implementation caveat does not turn the negative into positive evidence.

## 7. If the cheap dictionary route fails: conditional flow surprise

There is one better-defined fallback worth keeping separate: predict a reproducible **trade-code mark** from its past and use the realized innovation after the mark arrives. It needs no recovered physical queue. Define the mark from the four current trade-volume code signs relative to fixed train-only centers; at most 16 categories. Estimate a fixed multinomial predictor from past mark-frequency EMAs at 4/16/64/256 trades. Compute probabilities before updating with the current mark; reset at sequence start.

For one-hot mark `m_t` and pre-trade probability `p_t`, the innovation is `u_t=m_t-p_t`; surprise is `-log(max(p_t[m_t],1e-6))`. Their interactions with current book state, not mark-prediction accuracy alone, would need to improve a fixed future-price predictor. Compare against a same-capacity raw-mark/frequency control. A linear innovation without a nonlinear interaction is only a reparameterization of its inputs.

The relevant empirical paper uses actual signed market orders and a DAR predictor based only on prior trades, then studies conditional penetration, quote revision and impact. Its §3.5 explicitly separates transaction-level behavior from longer-horizon predictability. Our marks are anonymized categories, so its economic interpretation transfers only as motivation. [Taranto–Bormetti–Lillo, full text §§3.1–3.5 and Appendix A](https://arxiv.org/html/1403.0842v2)

Before any target screen this fallback requires one train-only prequential mark-log-loss comparison to stationary frequencies, with frozen detector, centering, penalty and unknown-mark handling. The current diagnostic budget does not include that comparison. It is less dependent on physical identification than queue intensities, but remains low priority: recent raw trade-code and MO0/MO1 screens failed, and the reference GRU may already estimate conditional flow. Do not label it a discovery of informed trading or metaorders.

## 8. Status and decision boundaries

The single real-data diagnostic completed under root's scheduling at frozen commit `a4eeecc`, in **4.31 seconds**, about **1.04 GiB peak RSS**. No targets, prediction caches or validation WP were read. Inputs and chosen chain IDs are in `quant-inputs.json`; all controls/slices are in `quant-result.json`, resources in `quant-run.log`. No real-data retry or changed recipe followed. Full Parquet hashing is deliberately not claimed; the manifest records source stat/footer and code/plan/metadata hashes.

| Side / fitting half | Learned best-two coverage at error≤.1 | Raw control | PCA control | Median learned error | Dictionary condition |
|---|---:|---:|---:|---:|---:|
| Bid / A | 6.91% | 0.78% | 0.00% | .269 | 6.12 |
| Bid / B | 7.29% | 1.57% | 0.00% | .282 | 11.71 |
| Ask / A | 8.97% | 0.65% | 0.00% | .267 | 247.03 |
| Ask / B | 11.93% | 1.50% | 0.21% | .239 | 121.83 |

All four coverage values miss the fixed 80% gate by a wide margin. Bid atoms match across halves in 10/11 cases, ask in 7/11; the ask dictionary also fails the condition<100 requirement. Every per-class slice is below the coverage gate: class2 is relatively easier (18–29%); class3, the original queue-race motivation, is only 3.4–11.8%. Low previous-state-magnitude slices reconstruct better than high ones, roughly 11.8–18.4% versus 1.4–3.7% coverage. That is compatible with state-dependent geometry, but does not identify its cause. There were zero extreme/sentinel exclusions in this sample, so that exclusion does not explain this failure. Before subsampling, there were 591,556 bid and 661,319 ask eligible quiet increments; 418,455 rows changed both sides (70.74% of bid updates, 63.28% of ask updates). This demonstrates substantial simultaneous **observed code** movement, not a count of exchange messages or proof that two physical queues changed.

The learned basis **does** capture nontrivial structure: median errors .24–.28 are lower than raw .58–.62 and PCA .45–.53. It would be wrong to summarize the result as “volume updates have no learnable coordinates.” The decision is narrower: this fixed dictionary has not earned the proposed stable, nearly sparse inverse-and-half-wave feature under its preregistered gate. Dense projections or a different sparse coder would be new proposals with separate budgets, not a threshold relaxed after these results.

There is a material algorithmic limitation: fitting optimizes a **one-line** assignment objective while evaluation permits two atoms. If simultaneous latent updates dominate, learned lines can represent frequent combinations rather than original coordinates. Hard initialization and a single optimization run are further limits. Therefore the real-data failure is not a mathematical refutation of any 11-dimensional mixing or sparse coding.

After the real-data run, a separately recorded deterministic positive control (`quant_toy.py`, `quant-toy-result.json`, seed20261002) planted a known orthogonal 11-by-11 mixing with one or two independent Gaussian latent updates per row, 6000 fit and 6000 evaluation samples each. The true dictionary has 100% best-two coverage, as it must; this implementation also achieved 100% coverage and recovered all 11 atoms at cosine≥.95 in both cases. Exact pair, degenerate-pair, raw-axis and weighted-summary unit checks pass. This rules out a gross reconstruction or fit failure on those two easy cases; it does not settle correlated, unequal-frequency, nonlinear or ill-conditioned cases. The toy was requested after the diagnostic and cannot retroactively strengthen its preregistration.

**Revised quant ranking:** (1) keep conditional trade-mark innovation as a separately specified reserve, with its train-only predictability gate first; (2) retain learned volume coordinates as an observed geometric structure, but allocate no immediate inverse-queue feature training; (3) defer physical queue-intensity reconstruction until zeros, orientation, relabeling and event/time identification have evidence. No alternate mark predictor was fitted in this round. Existing data remain exposed, and none of these measurements establishes competition improvement or an independent confirmation.
