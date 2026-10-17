# Deep reflection: rare exceptions, real bottlenecks, and the 25 mechanisms

Date: 2026-10-17. This is an independent synthesis after reading the final R1 portfolio and its R2–R4 reviews, the current model catalog, the R1 EDA and the local historical audit. The pre-card framework and its post-review correction are preserved in [`ROUND1_FRAMEWORK.md`](ROUND1_FRAMEWORK.md). No training, optimizer step, calibrator, model inference, external-data access, or new data scan was run for this reflection. The R1 proposals remain proposals; no local score or mechanism is upgraded here.

## The exceptions are sparse counterexamples, not a new law

The user’s rarity intuition is numerically right. The census found 941 train and 312 valid eligible rows where at least one target changed without the selected `i0_p0 AND i0_p11` price-code event; 92 valid exception rows are scored. The 1,253 exceptions are about 5 per million of all 249.6 million scanned rows. Relative to the census’s eligible target-change counts, the per-split rates are about 35 per million in train and 48 per million in valid. They are sparse.

“Sparse” does not settle either their predictive value or their score influence. The full-census script did not measure every exception’s target-change distribution: it retained per-sequence maxima and first-in-order examples, not a representative sample. The earlier 52 listed exceptions were selected first examples, not a random draw. All 52 were at least tick-sized, and one scored example changed the targets by roughly −1.689 and +2.172 while both selected prices stayed fixed. This rejects “all exceptions are floating-point jitter,” but cannot characterize the other 1,201. The exceptions may cluster within sequences, so their row count is not an independent sample size.

The condition is also narrower than “no new market information.” It tests two price-like columns, not all 112 current features or the permitted history. In the separate 256-sequence sample, some other feature changed at every adjacent row. That sample-level fact makes “selected clock quiet” different from “entire observable stream quiet”; it does not prove the other movement predicts the target. The four-column full census does not tell which other input changed on each exception. The target definition is intentionally anonymous beyond “future price movement of i0,” so neither its horizon nor an exact label-update process is documented.

The correct conclusion is narrow: the selected two-price event is not a necessary event for every target transition. This falsifies an **exact target-clock implication**. It does not show that the approximate event feature has no value, that the model stopped updating its hidden state, or that any particular output-holding policy hurts WP. A model can use a useful approximate clock while updating hidden state on every row; a separate emission rule can hold or smooth predictions while the state still consumes every row. The separate output-hold investigation must answer that empirical question.

## Separate information limits from modeling limits

The task can be written as three steps:

\[
S_t=U_\theta(S_{t-1},X_t),\qquad
P_t=G_\theta(S_t,X_t),\qquad
\text{score}=\operatorname{WP}(Y,P;M).
\]

The official callback requires predictions on every eligible row, updates through warm-up and unscored rows, causal prefix-only inputs, and sequence-local resets. `M` (the scoring mask) and `Y` are unavailable to the deployed predictor. During training, allowed labels and inputs shape \(\theta\), but they do not become inference-time state.

For target k, let \(Y_k=\operatorname{clip}(t_k,-2,2)\) and \(Q_k\) be the row measure with mass proportional to \(M|Y_k|\). Under that measure, the scorer is a global Pearson correlation. Its best possible direction among predictors using causal history \(H_t\) is the weighted conditional mean

\[
m_k(H_t)=\frac{\mathbb E[M|Y_k|Y_k\mid H_t]}{\mathbb E[M|Y_k|\mid H_t]},
\]

and the corresponding population correlation ceiling is

\[
\sqrt{\operatorname{Var}_{Q_k}(m_k(H_t))/\operatorname{Var}_{Q_k}(Y_k)}.
\]

These follow from the scorer’s weighted moments; they are not estimated by the event census. A ceiling below one can come from genuinely unpredictable target variation given \(H_t\), while a fitted model below that ceiling can reflect state compression, estimation, optimization, training-target choice, or readout error. A feature-clock counterexample does not tell which case applies.

This also explains why exception count alone cannot bound score impact. For a small unsaturated change \(dP_t\) to one prediction, the first-order change in target-k correlation under the fixed score weights is proportional to

\[
\frac{M_t|Y_{k,t}|}{Z_k}dP_t\left[
\frac{Y_{k,t}-\mu_{Y,k}}{\sigma_{Y,k}\sigma_{P,k}}
-\rho_k\frac{P_{k,t}-\mu_{P,k}}{\sigma_{P,k}^{2}}
\right],
\]

where \(Z_k=\sum M|Y_k|\). Clipping makes the derivative zero outside a saturated prediction and nonsmooth at the boundary. The bracket depends on residual direction and global prediction variance, so a small weighted subset could matter or could contribute almost nothing. This influence expression is a diagnostic approximation, not a subgroup score decomposition. The reliable check is to grade the exact full prediction vector under the unchanged scorer and report the event-defined subgroup’s residual/influence summaries separately.

## Which bottlenecks the 25 cards actually address

The cards span useful mechanism families, but none has identified a Bayes ceiling or shown how existing prediction residuals fall on the 92 scored exception rows. Those measurements should precede an expensive architecture story if saved, hash-bound predictions are available. The full metric remains the decision surface.

| Bottleneck | Cards | What they test | What they cannot establish alone |
|---|---|---|---|
| **Compact endpoint / capacity** | 01 | Wide one-layer versus deeper GRU at matched exposure and parameter budget. | This is not a new representation. A previous one-seed H512×1 screen scored above H256×2 on exposed F0; it is a useful lead, not a replicated, converged or runtime result. |
| **Input state and representation** | 03, 05, 06, 09, 11, 12, 13, 22, 23 | Soft event-paced state writes; asynchronous cross-stream lags; prefix normalization; path ordering; covariance orientation; change alarms; bounded same-sequence retrieval; within-side tying; transition topology. | Descriptive input structure is not target information. No full-feature census or out-of-sample residual test has established which state summary helps. |
| **Teacher-to-student transfer** | 02, 04, 07 | Same-context output distillation; hidden-state hints; distribution/dependence summaries. | These are supervisory choices on a shared transfer path, not three independent guarantees. Teacher improvement does not imply a better compact student. |
| **Self-supervised representation / latent rollout** | 08, 17, 18, 25 | Predict input events and durations; reconstruct masked current features; predict future feature latents; teach with imagined causal feature dynamics. | Future inputs can only be training targets. These losses do not add inference information and require a matched student result to show that their representation helps target prediction. |
| **Memory operator / architecture** | 14, 15, 16, 24 | Mamba-3 MIMO/SISO state; xLSTM scalar/matrix memory; local attention plus recurrence; causal fast-weight teacher. | Language or synthetic capability is not forecasting evidence. Every path must be judged on a real student endpoint and actual CPU/runtime cost. |
| **Training dynamics / objective** | 19, 21 | Compensated long-horizon gradient estimator; bounded train-only group robustness. | ARTBP changes gradient estimation, not inference state. Group-DRO changes the training estimand and still has to win the pooled official WP. |
| **Target/output parameterization** | 10, 20 | Nonlinear common/contrast capacity constraint; cross-fitted scalar score-to-weighted-mean map. | Target correlation alone does not justify parameter sharing. Calibration matters only if the current output is not already in a positive-affine optimum, or clipping makes a nonlinear change relevant; it needs honest cross-fitting. |

The most important **uncovered measurement bottleneck** is actual model residuals. The model catalog records distinct production paths: a 211-input distilled GRU with 99 event features and a 263-input tied-lead GRU with a different teacher/input lineage. Both demonstrate that event features coexist with successful models, but the catalog is not a matched event-feature ablation. We cannot infer the 99 event inputs caused the score, nor that they encode target maturity. For the deployed model, only fixed prediction residuals can show whether any proposed event age, quiet-period state, or teacher carries complementary information. A teacher’s own WP, loss, or hidden-state novelty is not the endpoint.

## Equivalences and opposing explanations to keep live

**The soft-write, event-auxiliary and CUSUM cards are adjacent but not identical.** Card 03 changes how much the recurrent state writes or decays; card 08 supervises an auxiliary model of *observable input* events and waiting time; card 12 detects a standardized innovation shift and changes state dynamics. A common “event family” story does not make these one model. Their shared prerequisite is a stable, nontrivial event vocabulary. The catch-all `a0…a7` changes nearly every row in the sample, so it is a poor sparse trigger. If CUSUM alarms reduce to event count or activity magnitude, it collapses toward simpler event-paced traces. If event targets are predictable but the deployed state does not benefit, card 08 fails at the student endpoint.

**The two cross-stream cards compress to one ordering comparison unless proven otherwise.** Card 05’s lag basis and card 09’s low-order antisymmetric path area are both summaries of temporal ordering between selected feature groups. For two increments, the degree-two antisymmetric signature is a signed sum of pairwise cross-products, so it is close to a fixed skew-symmetric lag kernel. First compare a tiny fixed lag basis with count/scale-matched and phase-shift controls; retain the area term only if it contributes stable input structure beyond that basis. Current lag-control signs vary by retrospective stratum, and the old i1-flow/cross-impact studies lack a reproducible forecast gain. The preprocessing replay that fixed a near-zero model score established a restore bug, not an i1 alpha effect.

**The three KD cards share a target endpoint but ask different transfer questions.** Card 02 asks whether any causal teacher improves a fixed student over direct supervision. Card 04 asks whether aligned hidden states add learning signal beyond point outputs. Card 07 asks whether conditional output distributions/dependence provide useful regularization beyond output means/family targets. The weighted-WP derivation says the metric-optimal prediction direction is a target-specific conditional weighted mean; distributional summaries are not inherently better predictions, only possible training signals for estimating that mean. The local ledger’s histogram-MSE and histogram-CE results were negative, the E468 codebook delta was negative against a losing shared anchor, and earlier family/moment heads already exist. Thus #07 needs a precise “what conditional dependence is missing?” answer before any comparison. Existing distillation has positive recipe-specific history, but local records also show the teacher improved more than its distilled student. A matched student, ancestry, context and update budget are essential.

**Sequence compression is not the same as skipping duplicate rows.** No adjacent identical full 112-feature vectors occurred in the 256-sequence sample. That removes a simple run-length-compression premise in that sample; it does not rule out useful sparse updates, near-state recurrence, learned memory compression, or long-lag recall. Card 13 retrieval and card 23 topology are separate from exact row compression, but the prior volume-code ridge screen reversed from +0.0062 in-sample to −0.0078 on chain-half cross-fit. Accumulated-interval memory also failed against its instantaneous control. Those results make recurrence/topology hypotheses possible but low-priority until input dynamics or fixed-model residuals show a repeatable reason.

**A persistent latent state is not a persistent observed label.** A target can remain numerically unchanged while a hidden input state evolves, and a target can change while one observed event flag remains false. At deployment, no lagged target label is supplied. A learned state-duration prior or target-transition auxiliary can only condition on current and past inputs; it cannot treat the previous true label as an autoregressive input. The old four-event class-3 decoder reconstructed about 90% of label rows but its errors carried roughly 25% of target-weight mass, with poorer behavior on long event horizons. That is partial structure, not an exact target generator or safe label oracle.

**A stochastic ceiling and an optimization failure can look similar in one score.** If all fixed teachers and students make the same errors on the same weighted rows, that may mean little information remains in the inputs, or it may mean shared input encoding / common target treatment has hidden the same signal. Diverse architectures alone do not separate these. Conversely, uncorrelated residuals can indicate complementary estimation errors without proving a causal mechanism. Compare fixed-model residuals across matched rows, then test the complete endpoint. Do not infer a “Bayes ceiling” from a shallow model plateau.

## What the prior negatives say about this portfolio

- **Event-time and clock family:** the old exact clock looked persuasive on 120 valid sequences but is now falsified at census scale. Earlier label reconstruction was partial, and age since the selected price conjunction is left-censored before its first event. Field-age inputs differ from group clocks descriptively, but the tested field-age model was only profiled; the completed G-versus-C comparison was about +0.00003 with an interval crossing zero. An input representation can be different without improving the prediction.
- **i1 and microstructure:** sample event co-occurrence has heterogeneous direction. Historical order-flow/cross-impact analyses did not establish a reproducible gain. The corrected frozen-checkpoint replay is a strong preprocessing diagnosis, but it does not isolate input benefit. Keep #05 conditional and judge a repaired same-student no-i1 control only in a future authorized comparison.
- **Teacher/student learning:** Wave19’s improved teacher exceeded its old teacher by about +0.000805 on strict425, while its student pair improved by about +0.000214 under its recorded matched comparison. This illustrates compression loss. Hidden-state residual probes passed some input gates, but their regression-WMSE intervals crossed zero and the full hidden-distillation comparison was not run. Those are neither validation of #04 nor proof it cannot help.
- **Architecture:** the matched GDN pair scored about 0.7118 versus 0.7124 for GRU in that recipe. This is counterevidence against assuming any gated state-space model is better, not a universal veto on Mamba-3 MIMO or xLSTM. The one-seed H512×1 screen was positive against H256×2 on exposed F0 and merits confirmation; it says nothing about long-context memory or a serving speed advantage.
- **Temporal memory / gradient horizon:** accumulated-interval memory failed against instantaneous control; fixed detachment-horizon results were small or inconclusive. Card #19 ARTBP is mathematically distinct from a random chunk-size sweep, but its variance/compute burden must be justified before a long experiment. Neither result decides whether hidden state should update on quiet rows.
- **Output and loss geometry:** histogram MSE/CE were negative on recorded strict comparisons; the histogram-CE interval was fully negative. The pooled-correlation memory recipe was negative in one matched two-stage comparison and inconclusive head-only. Its only limited reopening is a source-level clipping-semantics check; do not relabel that old result as support for #20. Positive-affine maps leave Pearson unchanged when clipping is inactive, while prior ridge corrections had positive fit skill but negative chain-half skill. #20 is a small conditional opportunity, not an automatic no-cost gain.
- **Rao–Blackwell terminology:** a prior label-denoising screen was biased in target-weight mass and covariance and failed its fixed bias gates. That is evidence against that approximation. It is not a test of Rao–Blackwellized *filtering* for a tractable latent-state model. The literature’s method requires an explicit partition where one conditional component can be integrated exactly; it does not promise that such a partition exists here.

All comparisons are on exposed development panels. Historical negatives constrain the exact tested recipe, not every descendant. Keep lineage, panel, metric and uncertainty attached to each result.

## Three high-risk variants that are coherent enough to preserve

These are refinements of mechanism, not three new acronym families. None should be trained under the current no-fit instruction.

### 1. A duration-bearing latent state with a separate output head

Model a small hidden regime \(z_t\) with a feature-conditioned, non-geometric transition hazard \(h(a_t,\phi(X_{\le t}))\), where \(a_t\) is the state’s elapsed row age. Maintain a continuous within-regime summary and emit a two-target prediction on every row from the current posterior state. The regime may persist while observations and state updates continue. Do not identify the regime with the target label, the `p0 AND p11` clock, or an output hold.

This is a semi-Markov duration hypothesis, structurally different from simply appending more EWMAs. Its prediction is specific: if fixed-model weighted residuals have a reproducible nonlinear relation to *input-state* duration after matching current features and sequence position, an explicit hazard may represent that state more efficiently than a fixed-age trace. If the pattern vanishes after current features or row position, the extra duration model is redundant. The next comparison would keep the output head and update path matched, altering only the state-duration transition. The exact finite-state/sequence inference cost and student transfer remain high risks.

### 2. Dense state updates with a prediction-innovation readout

Keep a normal state update on every row, \(S_t=U(S_{t-1},X_t)\), then factor the emitted prediction into a previous-*prediction* anchor and a causal innovation, for example \(P_t=P_{t-1}+\Delta_\theta(S_t,X_t,P_{t-1})\). The previous prediction is generated model state, never the previous true label. A bounded alternative is a learned blend of a direct readout and the prior prediction. Every required row still receives an output, and the readout is free to change when an event arrives.

This isolates output smoothing/holding from hidden-state skipping. It is only justified if fixed checkpoint predictions show noisy high-frequency output changes that hurt full-panel WP while the underlying causal input state still carries information; it is disfavored if the hidden input changes precede useful target movement and smoothing creates lag. Compare the direct readout to the same dense state with a predeclared innovation/blend head; do not use target transitions to decide when to update. The separate output-hold analysis now underway is the right first evidence, but it is not a model fit or a Bayes-ceiling estimate.

### 3. Rao–Blackwellized filtering for a tractable substate, if residuals imply a mixture

If existing residuals or label geometry support distinct latent regimes with continuous within-regime dynamics, define a finite or bounded set of discrete regime/duration hypotheses and a low-dimensional conditionally linear-Gaussian substate. Filter the regime hypotheses causally; update each Gaussian substate analytically conditional on its regime; average the target-specific readouts by filtered posterior mass. Keep the full state sequence-local and emit on every row. This is a state estimator for input history, not label denoising from hidden true targets.

It is coherent because analytical marginalization can reduce the sampled/represented state only when the conditional subproblem is actually tractable. It is high-risk because markets need not satisfy that factorization, the posterior may be poorly calibrated, and a full particle system could violate serving cost. The prior label-smoothing failure is a warning about bias, not direct refutation. A suitable future no-fit gate would first validate a toy factorization and compare the candidate state decomposition against simple event-age, covariance and recurrent baselines on feature-only sequences; a real score claim still requires a fitted endpoint and the official metric.

## Observations that would change priority

1. **Exception weights and residuals:** the pending full exception characterization should report per-target clipped magnitude/weight, not only counts. Join those rows to already-existing causal prediction artifacts if their provenance and masks are sound. If exception rows have little target weight or already-small residual influence across the incumbent, keep the exact-clock correction but lower exception-specific memory work. If a small set carries substantial score-weighted residual influence across multiple models, prioritize a soft state/readout mechanism and use full-panel regrading to test it.
2. **Other input events:** if the 1,253 exceptions consistently coincide with changes in other allowed features, the main question is whether current representations compress those changes badly; if some occur with all permitted features unchanged, target noise/hidden exogenous shocks become more plausible. Neither outcome by itself proves forecastability or a score effect.
3. **Within-label-plateau predictions:** for already-generated checkpoint outputs, compare dense predictions, fixed output holding and predeclared cadence/EMA controls to exact full WP. If holding loses while an oracle plateau mean reduces prediction variance, that suggests useful information arrives within constant-label spans and smoothing may lag it; it does not show label plateaus are knowable online or establish a ceiling. If holding helps only when fixed-model innovations are noise-like and does not miss scored exceptions, an emission smoother could be useful. The oracle is retrospective and nondeployable.
4. **Long-age residuals:** class-3 weight share and 43.69% age-≥64 share motivate a fixed-model residual check. A stable long-age error pattern across sequences would raise duration-memory priority; no pattern after conditioning on current input lowers it. The age remains the selected input-clock age, not target age or a test-time class.
5. **Teacher/student residual geometry:** a teacher with higher WP but nearly identical student errors suggests limited new signal or weak transfer; state-level hints matter only if they improve student residuals, not just teacher loss. Stable complementary teacher/student residuals could justify distillation or a blend. Different seeds, architectures or agents are not independent confirmation on the exposed folds.
6. **Event-time controls:** the current i1 lag association is heterogeneous. A larger fixed input-only sequence sample that preserves the same lag direction and beats phase-shift controls would raise #05/#09. Mixed or null controls lower their priority but cannot refute nonlinear target value by themselves.

The parent’s new exception-weight, label-reconstruction and prediction-cadence analysis is pending. This reflection makes no claim about those results. In particular, output-hold effects and oracle plateau diagnostics should be interpreted separately from hidden-state update behavior and from the conditional Bayes ceiling.

## Primary literature used for the representation distinctions

The papers motivate mathematical operators; their task results are not forecast evidence for this competition. Full source records, checked claims and transfer limits are in [`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md).

- Semi-Markov models put labels or latent states on variable-duration segments and allow duration-sensitive segment features. This motivates an explicit state-age/hazard hypothesis, not target holding ([Sarawagi & Cohen, 2004](https://papers.nips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html)).
- Sticky HMMs add self-transition bias to address latent states that switch too rapidly. That is state persistence, not evidence that a numeric forecast label itself persists or that its transitions coincide with a particular input code ([Fox et al., 2011](https://people.eecs.berkeley.edu/~jordan/papers/AOAS395.pdf)).
- Neural Hawkes models let hidden recurrent state modulate continuous-time event intensities, including excitation and inhibition. The structural analogy is event-time memory; this dataset supplies row order rather than wall-clock timestamps and predicts an anonymous future movement, not an event process ([Mei & Eisner, 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/6463c88460bd63bbe256e495c63aa40b-Paper.pdf)).
- Rao–Blackwellized particle filtering samples one latent partition and analytically marginalizes a conditionally tractable remainder, reducing the dimension over which sampling is required. It is not a generic label smoother or free neural compression method ([Doucet et al., 2000](https://www.cs.ubc.ca/~murphyk/papers/rbpf_uai00.pdf)).
- The Compressive Transformer stores compressed past memory for long-range sequence tasks. It supports learned approximate memory as a representation family, not exact skipping of identical input rows; the reported language, speech and RL settings differ from this numeric forecasting task ([Rae et al., 2020](https://arxiv.org/abs/1911.05507)).

## Provenance and limits


## Round 2 addendum: prior-mechanism attack

A source-level novelty audit narrows the three variants above. The duration-state
proposal substantially overlaps the executed Belief v2 posterior over observed
input-event marks; only a latent-regime duration law that differs from that
mark-survival model remains distinct, and it stays conditional. Generic
future-run consistency is not a missing training objective: the current loss
already has bootstrap, far-distance teacher mixing, reanchoring and EMA-teacher
paths, and fixed-label weighted squared error already decomposes into run-mean
error plus within-run prediction variation. Wave38's guarded h8 teacher target
lost to the stronger family arm on its point estimate, with a CI crossing zero.
The Rao–Blackwellized filter is mathematically different from the discrete
Belief v2 filter, but no conditionally tractable continuous substate or residual
evidence for one has been shown. A second novelty check finds that semantic
future-observable queries may refine existing cards 04/08/18, but the next-event
posterior, current quote/event auxiliaries, and old privileged-moment probe
substantially constrain that novelty. See
[`ROUND2_PRIOR_ATTACK.md`](ROUND2_PRIOR_ATTACK.md) and the sibling
[state-clock/query refinement](../theory/STATE_CLOCKS_AND_PREDICTIVE_QUERIES.md).
