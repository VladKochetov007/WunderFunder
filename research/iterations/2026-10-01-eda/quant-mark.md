# Trade-code innovation: premise, fixed diagnostic and limits

**Completed outcome:** the registered nuisance-study gate passed. First-order adaptation reduces mean per-chain mark log loss from 1.92528 to 1.13552 nats, a gain of 0.78976 [conditional chain 95% interval 0.78148–0.79799]. This is strong evidence of detected-code transition structure beyond the specified adaptive marginal-frequency control. Detector topology, prior mismatch on shuffled paths and existing GRU information remain unresolved. No price model, WP comparison or GPU training is authorized by this result. The final section gives the complete interpretation; the pre-outcome reasoning below is retained.

**Pre-outcome decision:** test whether past coded marks carry useful next-mark information beyond adaptation to each sequence's own frequencies. This does not require physical identification of orders. It also does not establish incremental price-prediction value: the existing GRU sees the causal raw codes and may already represent their dependence. The fixed study is [quant-mark-plan.json](quant-mark-plan.json), implemented by [quant_mark_eda.py](quant_mark_eda.py). No real-data outcome was read when this memo was written.

## What survived the previous round

The previous volume dictionary failed its fixed gate: only 6.91–11.93% of increments had sufficiently accurate two-atom reconstruction. That rejects the particular 11-ray fitting recipe and its proposed sparse inverse-coordinate feature; it does not prove that anonymized volumes have no useful coordinates. No alternate dictionary is fitted here. See [the completed quant dossier](../2026-10-01-deep/quant.md).

Trade-code innovation is a different premise. It asks about the conditional distribution of four observed trade-volume codes, without claiming to reconstruct aggressor side, execution size, physical queues, metaorders or seconds. A deterministic causal summary need not add information to the full input history to help a finite forecasting model. It must nevertheless earn its complexity against an adequate history control and then, in a separate authorized experiment, improve the actual prediction objective.

The relevant microstructure motivation concerns predictable order flow and the market's conditional response. Taranto–Bormetti–Lillo define a prior-trade sign predictor and study penetration, quote revisions and impact conditional on predictability; their longer-horizon discussion does not equate sign-prediction accuracy with profitable price forecasts. Their actual signed market orders are better identified than these anonymous codes. The present statistic transfers the conditional-prediction idea, not the paper's economic interpretation. [Full text, §§3.1–3.5 and Appendix A](https://arxiv.org/html/1403.0842v2)

## The detector is part of the estimand

The organizer identifies trade-related feature groups but gives no inverse transform that turns their coordinates into signed physical volume. [Official data overview](https://wundernn.io/connectome/docs/data_overview)

Let `v_t` be the observed four-vector. An emitted update requires two consecutive finite vectors whose absolute components are below 5.19, an exact float32 code change, and current cross-component range greater than `1e-6`. Row zero initializes state; it emits no mark. Every row advances the previous raw vector, including invalid and neutral rows. Consequently the first valid row after an invalid row cannot emit, and return from a neutral vector may emit. There is no full-sequence median, inferred future rest code or future event endpoint.

This detects a **non-neutral code update**, not a verified execution. Two trades with the same code can merge; one execution followed by representation changes can split; sentinel filtering may remove genuine extremes; quantization can create repeated categories; and the non-neutral condition selects a subset of states. Even correct prefix causality does not remove those selection effects. That is why the result includes invalid/extreme pairs, same-code holds, neutral changes, event gaps, exact code returns after one intervening event and cap retention. A strong dependence result with many alternating/reset patterns would still be observable dependence, not evidence of order splitting.

We score only the first 4,096 detected updates of each selected sequence. Full-row detector counts reveal how much this truncates. No thinning changes adjacency. The object of study is therefore an early event-prefix distribution, with later row coverage varying by activity; it is neither a random sample of all rows nor all exchange events. Empty sequences remain in the detector accounting but do not acquire an invented zero log loss. One-event sequences contribute their real cold-start loss.

## Fixed encoding and a causal prediction target

Each fitting half supplies four static weighted medians. A nonempty fit sequence contributes total weight one through at most 512 evenly spaced events from its retained prefix. For column `j`, the first sorted value whose cumulative weight reaches one half is the center `c_j`. Encode

`M_n = sum_j 2^j * 1[v_(t_n),j > c_j]`, with exact ties mapped to zero.

The resulting alphabet has at most 16 categories, including unseen categories. Its binary labels have no buy/sell meaning. The fit center uses complete fitting sequences, which is permissible nuisance fitting; no evaluation sequence supplies its own future center. The opposite fitting half's alternative encoding is reported only as a stability diagnostic. Tie counts and codebook frequencies expose unstable or collapsed cuts; they cannot trigger a better cut after outcomes.

At emitted event `n`, the probability of `M_n` is fixed before reading its mark. It depends on previous emitted marks only, not the current raw vector, current gap, class or sequence identity. Detection identifies when to evaluate an already available next-mark distribution. After observing the mark, its loss is scored and online counts advance. State resets at every competition sequence boundary, even if two sequences belong to a recovered chain. If a later feature uses innovation, it becomes available only after the current code is observed; it must never enter prediction of that same code.

The vectorized occurrence-rank implementation is a computational shortcut for strictly previous counts, not a fit using future frequencies. A literal online implementation checks identical probabilities, positive normalized categorical distributions, sequence reset and independence from current/future marks. Raw-row and event-prefix tests check the detector and scoring implementation separately.

## Why the control is adaptive

A stationary global frequency baseline is too weak. Sequences can have different category mixes, so a predictor that learns only which sequence it is in can appear to model temporal order. The fixed global prior `pi` averages each nonempty fit sequence's frequency vector with equal weight, plus uniform total smoothing mass one. The transition prior `R` similarly averages normalized within-sequence transition counts and smooths each context toward `pi` with mass one.

The three predictions are:

`G_n(j) = pi_j`

`F_n(j) = (N_j(<n) + 16*pi_j) / (n-1+16)`

`T_n(j) = (N_(M_(n-1),j)(<n) + 16*R_(M_(n-1),j)) / (N_(M_(n-1),·)(<n)+16)`.

At the first event all three equal `pi`. All probabilities are positive, so no outcome-dependent clipping of log loss is needed. The primary comparator is `F`, which adapts to a sequence's own marginal frequencies. `T` adds dependence on the immediately previous category and adaptive transition counts. The fixed concentration 16 is per active transition context for `T` and total frequency count for `F`; they have different capacity and adaptation rates. This is a comparison of two specified online estimators, not a perfectly capacity-matched test of a physical mechanism. No κ sweep or alternate logistic model is allocated.

## What the two order-destroying diagnostics answer

Full within-sequence permutation preserves exact mark counts and length while destroying temporal order. A difference between original and fully permuted gain can reflect short memory, changing regimes, bursts or detector artifacts. It cannot isolate economic causation. Fixed blocks of 128 consecutive marks preserve coarse local frequencies and their ordering while scrambling within-block adjacency. An original-minus-block advantage is a stronger check against slow frequency drift, though drift inside blocks and boundary effects remain possible. Block size is fixed before outcomes rather than chosen where a result becomes favorable.

Both diagnostics run the same frozen priors and causal online updates on the permuted sequence. Their construction sees future marks and is **offline**: neither permutation is an inference feature or a deployable baseline. Three fixed seeds quantify sensitivity to those particular reorderings; averaging them is not a calibrated permutation test. We report every seed and both null types, without treating an adverse null as a reason to invent a third.

The changed/non-neutral detector itself can induce transition constraints. Arbitrarily shuffled mark sequences need not be realizable by applying that detector to a plausible raw stream. Consequently even a large original-minus-block advantage may describe learned representation-state transitions or detector topology, rather than market-flow innovation. If this study passes, a separately registered detector-respecting surrogate or topology-aware control would be a useful next identification check; it is not silently added to the current recipe. This distinction does not forbid a useful causal feature, but it restricts the name and mechanism assigned to it.

## Sampling, uncertainty and the decision boundary

The design selects 256 distinct train-only chains, one sequence per chain, 64 from each historical class. It excludes all chains containing validation rows, the entire chain containing corrupt train row group 528 and chains used in the previous dictionary sample. Each eligible chain's smallest train row group is its representative; stable hashes then select chains within class and alternate them into two fitting/evaluation halves. The representative is not randomized, so any relation between row-group position and sequence regime can bias this representative-prefix sample. Class metadata is historically target-derived and is used only for sampling and descriptive tables, never for a predictor. The class-balanced design is intentional and is not a natural-market-share estimate.

Each half is evaluated with centers and priors fitted on the opposite half. These exposed, disjoint diagnostic samples are not a pristine holdout. Whole-chain paired bootstrap draws within half/class strata keep every predictor and permutation coupled; centers and priors remain fixed. Its intervals describe conditional chain variation and omit nuisance-fit variation, prior adaptive selection and future-test shift. Mutual information is descriptive; dependent events do not create an IID chi-square test or millions of independent observations.

The primary estimand is the equal-chain average of `LL(F)-LL(T)`. The event-weighted ratio of total loss difference to total retained events is separately reported: it can disagree when activity and dependence differ. First 16 marks and the remaining prefix are fixed cold-start slices. Every half/class and all null seeds are reported, with no class selected as an inference rule.

A later nuisance-feature study is nominated only if the original per-chain gain is at least 1% of `F` loss, both halves are positive, the conditional pooled gain interval is above zero, and the original-minus-block128 gain interval is above zero. This is an exploratory spending gate, not a discovery threshold. Passing would support short-context prediction of this detector's categories beyond the chosen adaptive control. It would still not establish aggressor-flow identification, causal market dynamics, target usefulness or improvement over the current GRU. Failing rejects this one cheap first-order proposal for immediate allocation, not every possible innovation representation.

If a subsequent, separately authorized price-feature study is justified, the measurable feature is `one_hot(M_n)-T_n` or realized surprise after event `n`. Its comparison must preserve the raw mark, marginal frequency and context information, and match extra dimension/compute. A linear innovation without nonlinear interactions can merely reparameterize already available inputs. The relevant outcome would be full eligible-row pooled competition score and incremental contribution to fixed C, under matched training seeds. No such training or target comparison is part of this round.

## Pre-outcome execution status

Plan and code are ready for root's freeze and single-slot scheduling. Synthetic checks passed for online/vectorized parity, smoothing, reset, current/future independence, prefix equality, permutation counts, detector transitions and tie handling; compilation, JSON parsing and whitespace checks passed. Resource limit: two threads, 160% CPU quota, 10 GiB, 900 seconds, zero GPU jobs and zero WP queries. Exclusive input/result files prevent overwriting earlier observations. The manifest records source identity and footer hash, not a claimed hash of the entire Parquet file. The run subsequently completed successfully; its real-data results and gate decision are reported below.

## Completed fixed study

The single real-data run used frozen commit `ad018c78fd5ebdf37f6baccfd30bcc80201c2375`. Script time was 8.09 seconds; the resource log reports 8.78 seconds wall time and 1,039,764 KiB maximum RSS, approximately 0.99 GiB. It completed successfully with all prefix checks and fixed surrogate checks passing, zero targets read, zero WP queries and zero GPU jobs. Inputs and selected IDs are in [quant-mark-inputs.json](quant-mark-inputs.json), complete per-chain/slice/seed results in [quant-mark-result.json](quant-mark-result.json), and resources in [quant-mark-run.log](quant-mark-run.log). Source size/stat identity was unchanged. These files were not overwritten and no second real-data recipe was run.

### Prediction and uncertainty

All 256 selected chains supplied at least 178 retained marks. There were 681,335 retained marks; min/median/max per chain were 178/2,587/4,096. The main estimates are:

| Quantity | Global G | Adaptive frequencies F | Adaptive transitions T |
|---|---:|---:|---:|
| Mean per-chain log loss, nats | 2.24982 | 1.92528 | 1.13552 |
| Perplexity from that mean loss | 9.48605 | 6.85704 | 3.11279 |
| Event-weighted log loss, nats | 2.24907 | 1.94448 | 1.15812 |
| First 16 marks, mean per-chain loss | 2.23662 | 2.09750 | 1.30264 |
| After 16 marks, mean per-chain loss | 2.24983 | 1.92350 | 1.13414 |

The primary gain is **0.78976 nats**, a 41.02% reduction relative to F, with fixed-nuisance conditional chain interval **[0.78148, 0.79799]**. The event-weighted gain is 0.78637 [0.77538, 0.79833]. The two evaluation halves gain 0.79236 and 0.78715. Thus the effect does not arise solely from the deliberately balanced classes or from one fitting direction. The first-16 gain is already 0.79486, similar to the later-prefix gain 0.78937: substantial structure transfers through the fitted transition prior before much online context-specific adaptation. This is consistent with shared code/detector structure; it does not identify market order-flow persistence.

| Historical class | Chains / retained marks | Mean-chain F loss | T loss | F−T gain | Event-weighted gain |
|---|---:|---:|---:|---:|---:|
| 0 | 64 / 144,449 | 1.90263 | 1.17742 | 0.72522 | 0.72420 |
| 1 | 64 / 253,151 | 2.08915 | 1.26109 | 0.82806 | 0.82572 |
| 2 | 64 / 134,731 | 2.06825 | 1.12767 | 0.94058 | 0.93133 |
| 3 | 64 / 149,004 | 1.64108 | 0.97591 | 0.66517 | 0.64868 |

These classes describe the balanced sample; none becomes a serving input or a selected hypothesis. Mean empirical within-chain entropy is 1.90475 nats and plug-in lag1 mutual information is 0.83887. The latter is descriptive and upward-biased in finite samples; the prequential prediction gain is the more useful result.

### All fixed order diagnostics

| Reordering | Seed | Mean-chain F−T gain | Event-weighted F−T gain |
|---|---:|---:|---:|
| Full shuffle | 20261011 | −0.103177 | −0.087846 |
| Full shuffle | 20261012 | −0.103565 | −0.087966 |
| Full shuffle | 20261013 | −0.103187 | −0.087823 |
| Within block128 | 20261011 | −0.085348 | −0.067510 |
| Within block128 | 20261012 | −0.084956 | −0.067395 |
| Within block128 | 20261013 | −0.085921 | −0.067972 |

Original minus mean full-shuffle gain is 0.89307 [0.88271, 0.90460]; original minus mean block128 gain is 0.87516 [0.86434, 0.88684]. The full-sequence F log loss is numerically identical across all orderings. That is expected: with fixed Dirichlet prior, the product of sequential marginal-frequency probabilities is exchangeable and depends only on category counts. It is a useful mathematical consistency check, not evidence that F used future counts. Its first-16 loss can and does vary with order.

T is worse than F on every shuffle. This does **not** make the original-minus-shuffle difference a pure market-order effect. In addition to different online adaptation/capacity, `R` was learned on original-order fitting data and then held fixed on reordered paths. Its prior is therefore mismatched to the null. The very large first-16 null losses—T approximately 2.79–2.83 for full shuffle and 2.71–2.74 for block shuffle, versus 1.30 originally—make this limitation material. The registered gate remains passed as written, but the ~0.88–0.89 null contrast must not be renamed an estimate of causal flow information.

The original mean same-category adjacency is 5.79%, compared with approximately 18.11% under full shuffle and 20.12% under block shuffle. That pattern is not simple sign persistence. It is compatible with the changed-code detector, alternating representation states or other transition topology. A shuffled category stream need not be realizable under the raw detector, even though counts match exactly. Both nulls support the descriptive statement that order matters for this frozen predictor; neither separates economic events from encoding mechanics.

### Detector, truncation and encoding audit

Across 5,120,000 raw rows there are 5,119,744 possible adjacent pairs. Fourteen pairs fail the finite/extreme guard. Of the valid pairs, 4,029,080 hold exactly the same four-vector and 264,139 change to a neutral vector. The detector emits 826,511 updates before the cap, 16.14% of all adjacent pairs. Retaining 681,335 events keeps 82.44%; 80 of 256 sequences hit the cap. Losses apply only to these retained prefixes, whereas detector gap/return counts cover the full raw sequences.

| Class | Full detected updates | Retention after cap | Capped sequences / 64 |
|---|---:|---:|---:|
| 0 | 146,728 | 98.45% | 4 |
| 1 | 376,423 | 67.25% | 57 |
| 2 | 135,356 | 99.54% | 3 |
| 3 | 168,004 | 88.69% | 16 |

Class1 is especially affected by prefix truncation; the study cannot infer its late-sequence behavior. Full-sequence event-count min/max are 178/9,190. Among successive detected events, raw-row gaps are one row in 67.09%, two to eight in 19.32%, and nine or more in 13.59%. An identical raw four-vector returns after one intervening detected event in 8.01% of eligible triples, ranging from 4.90% in class1 to 17.10% in class0. These are detector properties and possible reset/alternation clues, not verified paired executions. First detected rows have min/median/max 1/10/1,384; last detected rows 14,435/19,986/19,999. Actual retained-prefix endpoints are not reconstructed from the aggregates.

Fit-half centers are `[-1.071347, 1.403542, -1.404296, -1.041349]` and `[-1.018313, 1.425838, -1.406531, -1.041349]`. Applying the other center set would change 4.043% of evaluation marks: 9.675% in class0, 3.255% in class1, 3.955% in class2 and 0.002% in class3. These differences are reported without replacing the fixed center. Exact per-bit tie fractions over all retained marks are 1.872%, 0.260%, 0.805% and 1.759%. The first and fourth ties are concentrated in class0 (8.831% and 8.296% of that class's marks); the third includes a 2.830% tie rate in class3. Thus median ties and regime-specific code geometry are real, rather than negligible numerical details.

Category 1 is absent from both evaluation halves and both original fitting sets; categories 10 and 15 are rare. Smoothing keeps all 16 probabilities positive. Mean-chain empirical entropy differs by class (1.878/2.079/2.050/1.613 nats). The complete per-chain category counts, tie counts and centers are preserved in JSON; there is no posthoc removal of rare categories, class-specific center or inferred buy/sell assignment.

### Revised priority and the next evidence requirement

The result promotes **causal detected-code transition summaries** above the failed inverse-queue feature within the quant branch. It establishes a cheap, transferable nuisance-prediction regularity under the chosen detector and encoding. It does not establish price benefit or make a new forecasting fit the automatic next action. The premise requiring care is now less “does any dependence exist?” and more “how much is learned representation/detector structure, and can an explicit summary add anything to the existing raw-input GRU?”

A concrete next CPU control, if separately authorized and registered, is to match the prior to the surrogate distribution: retain the same selected chains, centers, κ=16, one-step recipe and fixed block128/three-seed orderings, but fit each null's `R` on the correspondingly permuted **fitting** half before evaluating that seed's permuted opposite half. The original predictor remains frozen. Recompute only the prespecified paired mark-loss contrasts. No source target or WP is needed. This would test how much of the adverse null performance is an original-prior mismatch; it still would not make shuffled streams detector-realizable or identify a physical market process. This control is a proposed new experiment, not a hidden continuation of the completed one.

A detector-respecting or topology-aware control would require a separately justified raw-stream generative rule; arbitrary mark shuffles are insufficient. It should not be improvised to preserve whichever observed transition looks favorable. Physical identification remains optional for competition utility: even a coding-state innovation could be useful, but should be named accordingly. Any later forecasting allocation must compare it with a matched causal history/control representation, use paired training seeds and the fixed full-row objective, and show incremental contribution over current C. No such allocation, new readout on raw source arrays, or GitHub upload occurred in this round.
