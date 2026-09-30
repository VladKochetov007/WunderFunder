# Trade-code innovation: premise, fixed diagnostic and limits

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

## Execution status

Plan and code are ready for root's freeze and single-slot scheduling. Synthetic checks passed for online/vectorized parity, smoothing, reset, current/future independence, prefix equality, permutation counts, detector transitions and tie handling; compilation, JSON parsing and whitespace checks passed. Resource limit: two threads, 160% CPU quota, 10 GiB, 900 seconds, zero GPU jobs and zero WP queries. Exclusive input/result files prevent overwriting earlier observations. The manifest records source identity and footer hash, not a claimed hash of the entire Parquet file. Real-data results and the gate decision are pending.
