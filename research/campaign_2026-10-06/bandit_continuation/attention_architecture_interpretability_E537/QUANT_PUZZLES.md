# Quant puzzle lenses: what information survives the measurement pipeline?

These are small mathematical questions that constrain an experiment. They are neither established economic symmetries nor reasons to discard raw information. They were added after independent commitment and duplicate reconciliation, without reading other current agents' proposals.

## 1. Which transformations preserve an event but change its apparent size?

If an anonymous channel is transformed by a strictly increasing injective function, exact equality and the sign of a change are preserved, while change magnitude and cross-channel ratios generally are not. Within-feature ranks survive, but their calibration must be fitted on the training membership. Finite precision, quantization, clipping and sentinels can destroy injectivity. Thus an event mask is potentially more measurement-invariant than a purported volume imbalance, yet discarding size may remove precisely the predictive information.

Concrete test: compare raw amplitude, train-fitted rank/PLE, and both paths on identical fitting membership and backbone. Use synthetic monotone re-encodings to check which proposed feature identities survive; this tests an invariance of the feature construction, not unchanged real-world labels under a fictitious market transformation. A win by raw+rank supports representation complementarity. It does not recover a physical unit or prove the host's encoding is monotone.

This is the reason for the raw bypass in E537-C and the refusal to treat an arbitrary cross-channel ratio as order-flow imbalance.

## 2. Does a duplicated observation represent zero elapsed information?

Two consecutive equal values can mean no instrument update, a latent event hidden by quantization, or an observation of continued inactivity. A deterministic model receiving identical *entire* observed prefixes must output identically; prefixes with a different count of repeated rows are different inputs. Enforcing global invariance to repeated rows would erase potentially informative survival time.

Concrete test: split representation into an event-history branch and a row-age branch. In a synthetic fixture, inserting a zero-change observation must preserve geometric/event content while advancing the explicit row clock. Then perform a feature-equivalent trained comparison of row-distance and row+event-distance attention. This prevents a clock adapter from getting credit simply because its control lacks the existing age feature. E535 makes that control particularly important.

Do not train on future label-transition distance. It can define a posthoc exposed-data diagnostic cell, but cannot be input, online gate, sampling propensity for a held-out claim, or a mechanism purportedly available to the deployed model.

## 3. Can the same final state encode opposite histories?

Consider two two-channel paths with increments `(1,0),(0,1)` and `(0,1),(1,0)`. They have identical endpoint and identical per-channel totals. Their antisymmetric second-level area is respectively +1/2 and −1/2. Thus a model restricted to endpoints cannot distinguish their order, whereas a tiny ordered statistic can. [Signature composition basis](https://arxiv.org/html/1603.03788v2).

Concrete test: displacement-only versus displacement+area versus symmetric-moment features under H227. Keep the raw current state and clocks. The signed-area fixture in ALGEBRA_EXAMPLES.json verifies this arithmetic. On real data, artificially reversing histories does not produce a known counterfactual price label; it is only a model/representation probe.

This is an unconventional sufficient-statistic route rather than a requirement for a larger memory network. Because H227 already exists, implement its declared descendant rather than create another signature family.

## 4. Which symmetries are implementation facts, and which are economic guesses?

| Candidate symmetry | Defensible scope | Discriminating check |
|---|---|---|
| Permute feature channels and simultaneously permute feature-specific parameters/schema | A coordinated relabeling should not change a correctly wired model | Reindex raw features, embeddings and grouping map together; compare output. Catches hidden positional assumptions. |
| Permute anonymous input channels alone | Generally **not** a symmetry: channel identity can matter | Preserve an identity-aware control when testing set pooling. H228/D cannot assume exchangeability from adjacent column numbers. |
| Swap bid/ask or negate all channels and target | Economic guess; anonymous transforms and target definition need not respect it | No augmentation until a train-only semantic mapping and label action are justified. Sign plots are not such proof. |
| Swap the two instruments and the two targets | Not implied by “two instruments/two outputs”; outputs may be horizons of one instrument | Read the actual schema and target semantics; never infer target permutation from count alone. |
| Positive rescale a q or k coordinate in normalized prefix channel attention | R is invariant away from the epsilon floor when history is rescaled consistently | Check the normalized moment matrix, not the entire v/readout model; save epsilon behavior. |
| Orthogonally change the key basis in a delta memory | Algebraic state-coordinate freedom with a corresponding transformation of M | For orthogonal Q: k'=Qk, q'=Qq, M'=M Qᵀ, then M'q'=Mq and the delta update transforms consistently. Test a known rotation. |

The last symmetry explains an interpretability limit: an individual memory coordinate can change meaning under an equivalent basis. Interpreting a single key axis as “buy pressure” is weaker than showing that a basis-independent memory subspace or a causal block intervention carries relevant information.

## 5. Is uncertainty about sign different from uncertainty about magnitude?

Under the official weighted distribution Q, the predictive scalar is `E_Q[Y|X]`. Zero-target cases carry no metric weight; a rare large outcome can dominate a frequent small opposite-sign outcome. For the toy conditional law Y=−2 with probability0.2 and Y=0.5 with probability0.8, the ordinary mean is0 but the weighted conditional mean is−0.75. This does not make a classifier inherently superior: direct weighted MSE already targets the latter functional.

Concrete test: given an own causal hidden state, compare the same finite-capacity scalar WMSE head and a properly declared distributional approximation, keeping losses/parameter counts interpretable. H111 already made this observation and has relevant adverse historical evidence, so E537-F is lower priority unless the new comparison changes an actual approximation bottleneck. A ratio with a noisy denominator can hurt; the Bayes formula is not a guarantee for finite learned heads.

## 6. Can “interpretation” change the next engineering choice?

A useful explanation should nominate an intervention: retain a longer event-history state, separate channel groups, change a clock gate, supervise an observable innovation, or compress a teacher block. A picture of high attention weight is insufficient. [Attention caution](https://arxiv.org/abs/1902.10186), [patching methodology](https://arxiv.org/abs/2309.16042).

For example, match two discovery prefixes by current inputs and row age but with different historical group order. Replacing only an own-model memory block with a matched donor tests whether that block influences prediction under that constructed pairing. Repeat with a zero donor and random donor, track activation distance, and verify the same method recovers the correct block in a synthetic known-memory model. Even a robust result identifies model computation, not market causality or the counterfactual label. Its payoff is a concrete next architecture/auxiliary/compression test.
