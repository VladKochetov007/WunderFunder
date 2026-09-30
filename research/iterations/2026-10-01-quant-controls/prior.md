# PRIOR02: separate transferred prior from online transition adaptation

**Completed result:** matching the transition prior to block-shuffled fitting data improves shuffled-path log loss by **0.05297 nats per mark, averaged equally over chains** [conditional 95% interval **0.04960–0.05665**]. The matched transition predictor still loses to adaptive frequencies by 0.03243 nats per mark. Original-order performance reproduces exactly; both transferred conditional structure and online context adaptation help their declared controls. These are estimator contrasts, not an additive causal decomposition or evidence of price-prediction benefit.

**Pre-outcome specification.** The previous fixed study found a mean-chain gain of 0.78976 nats for adaptive first-order prediction over adaptive marginal frequencies. Its shuffled transition predictors lost to marginal frequencies, but used a transition prior fitted on original-order paths. This follow-up asks how much that mismatch costs; it does not retest whether the original measured gain exists. The exact experiment is [prior-plan.json](prior-plan.json).

For each frozen fitting half and each of the same three block128 seeds, fit the same normalized transition-count prior on the correspondingly permuted fitting sequences. Use that fitting half's original frozen centers on both fit and evaluation sequences. Evaluate its opposite half under the same seed's prior permutation. Comparing old and matched priors on exactly the same observed marks and online counts changes only the transferred transition table. Marginal `pi` remains unchanged because each sequence's mark counts are preserved. No full-shuffle arm, center fitting, detector change, κ tuning or extra sample is included.

The primary signed effect is `LL(old R on shuffled paths) − LL(matched R on shuffled paths)`, averaged equally over seeds and chains. A positive result quantifies the cost of this particular prior mismatch for the fixed online estimator. The remaining `LL(F) − LL(T_matched)` shows whether context conditioning helps that shuffled distribution after matching its prior. Neither quantity is a pure causal ordering effect: artificial mark paths may violate detector constraints, sequence mixtures remain, and the online estimators have different capacity.

Two fixed original-order controls separate other parts of the predictor. Replacing each transition-prior row by `pi` retains context-specific online counts while removing transferred conditional structure. Using static `R[previous_mark]` retains transferred structure while removing online transition adaptation; the first event remains `pi`. Their differences from original T are estimator contrasts, not an additive decomposition into independent information sources. Context frequency, smoothing and fitted prior interact. All-event, first16 and after16 losses are reported, along with both halves and all historical classes.

The original result and old block controls must first reproduce per-chain mark hashes, detector/category counts and all three loss slices. Own-half encodings used to fit a half's R are distinct from the opposite-half encodings used to evaluate that sequence previously. Explicitly preserving this distinction prevents the tempting error of fitting R to one codebook and evaluating a different one. New matched priors only use fitting-half sequences; probabilities use past marks and are scored before current-mark updates. Surrogate generation remains an offline diagnostic, not a deployable feature.

The 256 previously exposed chains and original class-balanced design are unchanged. Whole-chain bootstrap within half/class reuses each resampling draw across all paired conditions; fitted priors, centers and permutations stay fixed. Its intervals omit nuisance-fit variation and the adaptive decision to run this follow-up. Macro-chain and event-weighted estimates answer different questions and are both reported. The predecessor's frequency joint likelihood is exchangeable, so its complete-prefix F loss must remain unchanged under block permutation; first16 loss need not.

This is one bounded CPU nuisance study, with six predetermined prior-table fits, no price targets or WP query, and no forecasting training. Passing correctness checks is required for interpretation. The result may justify changing the explanation for the previous null contrast; no numerical threshold here automatically authorizes a forecasting model or identifies physical order flow. Implementation, synthetic checks and empirical status will be appended after review and root scheduling.

The prior run's inventory recorded remote checkout HEAD `ad018c7`, while its executed code and plan had been copied by SSH after local freeze `4782c70`. Therefore the remote base HEAD must not be described as the commit containing those executed files. Their exact recorded SHA256 hashes identify the predecessor implementation; this study verifies that code hash before importing/reusing it and records its own code/plan hashes separately from runtime checkout HEAD. Past result files are left unchanged.

Synthetic tests passed for the new original-order controls, matched-versus-identical prior parity, positive normalized online vectors, prefix and current/future independence, resets, fixed permutation invariance of pi and cumulative frequency loss, and an exact hand-computed transition-prior example. Compilation, plan JSON parsing and whitespace checks passed. The root will freeze and execute the real-data study; it has not been run by this author.

## Frozen execution and verification

Root executed the single real-data recipe from `d0f7f06bce30c71b92922dcb20f4fb419bb097d9`, with script SHA256 `16fe6e8589de3505deec83922651531061c79b70c577426277045ea75ca8b57b`. It completed in 7.53 seconds of script time, 7.83 seconds resource-log wall time and 1,041,472 KiB maximum RSS, approximately 0.99 GiB. No targets, prediction caches, WP queries, GPU jobs or forecasting fits were used. See [prior-inputs.json](prior-inputs.json), [prior-result.json](prior-result.json) and [prior-run.log](prior-run.log).

All 256 original mark hashes, category and detector counts, both original priors, and all original/old-block loss slices reproduced to the registered tolerance. Source identity and causal prefix checks passed. The same 681,335 retained events are scored; no chains or marks were selected after the outcome. Bootstrap uses the newly frozen RNG20261031, so the reproduced original point estimate is identical while its Monte Carlo interval differs slightly from the predecessor's RNG20261021 interval. That is expected and is not new independent evidence.

A synthetic-only aggregation-test edit crossed root's freeze notification. It never changed empirical run logic and was not executed on real data. The local file was restored byte-for-byte to `d0f7f06`, and the extra artificial-array test was preserved separately as [prior_synthetic_aggregation.py](prior_synthetic_aggregation.py); it passed. [freeze-crossing-receipt.json](freeze-crossing-receipt.json) records the scope, timing and hashes. This was neither an empirical retry nor a new scientific recipe. Independent checks bind to the restored frozen script.

## Prior matching changes the adverse null, not the original result

For the three fixed block128 seeds, mean-chain F loss is unchanged at 1.925276 nats per mark. Only the transition prior changes between the two T columns:

| Seed | T with old R | T with matched R | Old−matched improvement | Conditional 95% interval |
|---|---:|---:|---:|---:|
| 20261011 | 2.010624 | 1.957574 | 0.053050 | [0.049500, 0.056817] |
| 20261012 | 2.010232 | 1.957548 | 0.052684 | [0.049356, 0.056275] |
| 20261013 | 2.011197 | 1.958010 | 0.053187 | [0.049785, 0.056843] |
| Equal-seed mean | 2.010684 | 1.957710 | **0.052974** | **[0.049603, 0.056654]** |

Thus matching reduces T's null loss penalty relative to F from 0.085408 to **0.032434** nats; the remaining `F−T_matched` interval is **[−0.034928, −0.029580]**. Original-prior mismatch was a material part of the adverse-null result, but correcting it does not make first-order adaptation outperform marginal adaptation on these whole shuffled prefixes. Different estimator capacity, context sparsity, regime mixtures and local frequency drift remain possible explanations for the residual. The observation does not identify a physical market process, and no alternate κ is tried to remove it.

Event weighting gives prior-matching improvement **0.042289** [0.040681, 0.043929] and residual matched-null gain **−0.025336** [−0.028460, −0.021710]. Lower event-weighted improvement indicates that equal-chain and event-weighted questions differ; it is not grounds to select the more favorable estimate.

## Cold start, transferred structure and adaptation

Original-order mean-chain losses are:

| Slice | G | F | Original adaptive T | T with independent prior rows | Static original R |
|---|---:|---:|---:|---:|---:|
| All marks | 2.249823 | 1.925276 | 1.135520 | 1.212911 | 1.291035 |
| First 16 | 2.236618 | 2.097503 | 1.302638 | 2.056250 | 1.322963 |
| After 16 | 2.249833 | 1.923504 | 1.134138 | 1.205552 | 1.290716 |

The fixed original T beats its independent-prior variant by **0.077391** [0.072402, 0.082851] overall. That advantage is **0.753612** [0.730780, 0.777297] on the first 16 marks and **0.071414** [0.067020, 0.076159] afterward. Transferred conditional structure therefore matters especially at sequence start, before enough context-specific counts accumulate. This is consistent with a shared representation/detector grammar or recurring flow structure; the experiment does not distinguish those explanations.

Adaptive T beats static R by **0.155515** [0.147037, 0.165704] overall, only **0.020324** [0.014604, 0.026608] in the first 16, and **0.156578** [0.148083, 0.166839] afterward. Online adaptation matters once observations accumulate. Yet a transition estimator starting from independent prior rows still obtains loss 1.212911 versus F's 1.925276 over the complete retained prefix: the effect is not solely a beneficial pretrained conditional table. Conversely, static R alone already improves substantially over F. Those comparisons overlap in information and learning behavior and cannot be summed into independent “fractions of the mechanism.”

The null correction is also concentrated in cold start: old-minus-matched improvement is **0.620358** [0.595405, 0.642631] in the first 16, versus **0.048298** [0.045172, 0.051646] afterward. Matched-null T modestly beats F in the pooled first-16 slice, **+0.027672** [0.018250, 0.036923], but loses afterward, **−0.033210** [−0.035825, −0.030380]. That cold-start average hides class differences: first-16 matched-null gains are +0.102820, −0.047093, −0.044190 and +0.099151 for classes0–3. A pooled positive cold-start null is not universal market behavior. Conditional priors can exploit sequence mixtures or transferred structure before the marginal estimator adapts, even on reordered data.

## Both directions and every class

The following complete-prefix contrasts are mean-chain nats. “Transfer benefit” means independent-prior T minus original T; “adaptation benefit” means static R minus original T. Each column is its declared paired comparison.

| Scope | Matched-prior improvement [95% interval] | Matched-null F−T | Original transfer benefit | Original adaptation benefit |
|---|---:|---:|---:|---:|
| Evaluation half0 | 0.056181 [0.050430, 0.063454] | −0.032874 | 0.081175 | 0.151726 |
| Evaluation half1 | 0.049767 [0.046185, 0.053326] | −0.031994 | 0.073607 | 0.159305 |
| Class0 | 0.056350 [0.049103, 0.065710] | −0.057576 | 0.090752 | 0.197641 |
| Class1 | 0.038404 [0.035865, 0.042716] | −0.013639 | 0.044906 | 0.186616 |
| Class2 | 0.077248 [0.069669, 0.085462] | −0.033985 | 0.099717 | 0.096057 |
| Class3 | 0.039895 [0.033223, 0.046923] | −0.024536 | 0.074190 | 0.141745 |

The matching effect is positive in both directions and every class; matched-null whole-prefix gain remains negative in every class. No favorable class or seed is selected. Every first16/after16, individual-seed, class/half, macro/event estimate and conditional interval is retained in the result JSON, including all six old-R/matched-R evaluations across two halves and three seeds. Historical classes remain descriptive metadata, never inference inputs.

## What this changes next

The original-code predictive regularity survives this control, while the previous adverse-null magnitude needs the prior-mismatch qualification. The next separately frozen DETECTOR02 study targets a different ambiguity: whether shuffling joint raw-code/run-length packets, then rerunning the same detector and fitting matched surrogate priors, reproduces much of that regularity. It changes the emitted population and uses a different block clock; its contrast cannot be equated to the mark-shuffle contrast or subtracted to identify a causal component.

Before seeing that second result, the justified quant hypothesis remains a useful causal representation-state predictor, with physical order-flow interpretation unresolved. No result here shows incremental future-price information beyond the existing GRU. This follow-up is motivated by previously exposed outcomes; all intervals remain conditional on the fixed nuisance fits and permutations, with no correction for adaptive selection, fit variability or hidden-test shift. There is no automatic new training, source-array follow-up or external upload.
