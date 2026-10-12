# Review: Recommended Next Steps for Improving Model Performance

I would change the direction of the search at this point. The report reveals a specific discrepancy with the official metric, several places where training may be optimizing the wrong distribution, and an underutilized signal from the second instrument. These are much more promising areas to focus on than another large learning-rate and dropout sweep.

The gap from 0.6664 to 0.672 is +0.0056. Based on the provided material, it would be unreasonable to promise that any single modification will close this gap. However, there is also insufficient evidence to conclude that we have already reached the performance ceiling of the task.

I reviewed the report, cross-checked the competition's official documentation, and examined research on limit order book forecasting. The source code and saved predictions are not available here, so I distinguish below between confirmed discrepancies, hypotheses that need verification, and new experiments.

My main priorities are:

| Priority            | Action                                                                           | Potential source of improvement                                               |
| ------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Immediate           | Verify the metric and rescore saved models                                       | The current search may be ranking solutions incorrectly                       |
| Immediate           | Blend the main head with the already-trained family heads                        | Additional performance at virtually no inference cost                         |
| Primary experiment  | Symmetric change features, separate event clocks, cross-instrument relationships | A representation that makes predictive signals more accessible to a small GRU |
| Parallel experiment | Check loss normalization and concentration on the scoring mask                   | More robust generalization to the hidden test set                             |
| Next stage          | Train on future changes at multiple horizons                                     | More useful internal representations without additional serving layers        |
| Engineering track   | Investigate the h384 speed anomaly and measure the available ensemble budget     | Preserve diversity or additional capacity in the actual submission            |

## 1. Most Important Finding: Your Metric Description Differs from the Official Definition

The report states:

> Pearson pooled over rows and both targets

Taken literally, this means the two targets are concatenated into a single long vector.

The official documentation specifies a different operation:

\\[ S=\frac{\rho\_{w,t0}+\rho\_{w,t1}}{2}. \\]

Correlation is calculated separately for each target across all scored rows, and the two correlations are then averaged. On rows selected by the scoring mask, targets and predictions are clipped to \\([-2,2]\\), and the weight is the absolute value of the clipped target. [Official documentation](https://wundernn.io/connectome/docs/data_overview?utm_source=chatgpt.com)

This is a substantial difference. Correlation calculated on the concatenated array depends on differences in means and scales between the targets. The average of two separate correlations does not.

### A Minimal Test to Verify This

Consider the following values:

| Row | True t0 | True t1 | Predicted t0 | Predicted t1 |
| --- | ------- | ------- | ------------ | ------------ |
| 1   | −0.5    | −0.5    | 0.5          | −1.5         |
| 2   | 0.5     | 0.5     | 1.5          | −0.5         |

Here, each target is predicted perfectly up to a constant offset. All values fall within the clipping bounds.

The results are:

- Official average of the two correlations: 1.0
- Correlation after concatenating the targets: 0.4472136

The first action should be to run this example through `gmetric.py` and the official scorer.

If the local code returns 1.0, the problem is only in the report's wording, and the issue can be closed. If it returns 0.4472, the saved prediction bank needs to be rescored immediately: checkpoints, teachers, students, ensembles, and HPO results.

This could reveal a very inexpensive improvement: some already-trained models may perform better under the actual metric but were rejected by the current evaluation criterion. Correcting the evaluator will not improve predictions directly; it will enable proper model selection and submission assembly.

During rescoring, I would explicitly report three numbers: `WP_t0`, `WP_t1`, and their average. I would also compare the official scores of GPU predictions and native predictions. High correlation between the outputs of the two implementations does not, by itself, verify that the score is preserved after clipping.

## 2. The Cheapest Modeling Opportunity: Use the Heads H230 Has Already Trained

H230 has a primary pair of outputs and additional heads trained to predict the raw outputs of four teacher families. Only the first two outputs are used during serving.

I would extract predictions from all these heads before starting any new training runs.

For each target, I would test:

\\[ p'\_j=(1-\lambda_j)p\_{\text{main},j} +\lambda_j p\_{\text{family-average},j}. \\]

Start with a few predefined values of \\(\lambda\\), independently for `t0` and `t1`. If the effect is consistent, move to a small regularized blend of individual family heads.

This is particularly interesting because, if all outputs are produced by a single final `Linear` layer from the shared hidden representation,

\\[ p\_{k,j}=W\_{k,j}h+b\_{k,j}, \\]

then a fixed linear combination can be folded exactly into a new final layer:

\\[ W'\_j=\sum_k\beta\_{k,j}W\_{k,j}, \qquad b'\_j=\sum_k\beta\_{k,j}b\_{k,j}. \\]

As a result, inference still requires only the same two output rows. No additional GRU execution is needed.

The head architecture must be checked to ensure there are no separate nonlinearities that would prevent this folding. Blending should be performed at the same point that can be reproduced during export, followed by official clipping.

There is no guarantee of improvement: the heads share a representation, and the main head has already been trained on a teacher mixture. But the cost of testing this is very low.

### Another Inexpensive Test Using the Same Outputs

Decompose the predictions into:

\\[ c=\frac{t_0+t_1}{2}, \qquad d=\frac{t_0-t_1}{2}. \\]

Check whether one head predicts the common component \\(c\\) better, while another predicts the difference component \\(d\\) better. Such a combination can also be folded into the final linear layer.

Simply renaming the two outputs as \\(c,d\\) does not expand the model class. The point is to select different already-trained estimates for different components. I would also avoid substituting the current spread for \\(c\\): the report relates the targets to the future order book, and clipping further complicates the financial interpretation of their sum.

## 3. Primary Opportunity for New Performance Gains: Changes in the Second Instrument and Event Clocks

According to the description of the 99 engineered features, you have:

- Price changes for the first instrument.
- Volume changes for the first instrument.
- Price changes for the second instrument.
- Eight event flags.
- Only three time-since-event counters.

Symmetric volume-change features for the second instrument are missing from this list. Raw volumes are present in the input, but the GRU must reconstruct their changes itself. When different blocks update at different frequencies, this places an additional burden on the model's memory.

I see a stronger hypothesis here than simply increasing the hidden size again: a small model should be given an explicit representation of changes and delays between related data streams.

### First Feature Package: Keep GRU-256 and Extend the Input

| Addition                                                     | What it helps distinguish                                                    |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Missing 22 volume-like change features for `i1`              | Liquidity dynamics of the second instrument                                  |
| Changes in trade price and trade volume for both instruments | Trade-stream events that may currently be represented only by a general flag |
| Time since the last update of each actual block              | Fresh information versus values held for a long time                         |
| Number of updates over several fixed windows                 | Event intensity                                                              |
| Short- and long-window accumulated changes                   | Momentum, persistent pressure, and decay                                     |
| Recent `i1` events, their age, and causal lags               | Delayed responses of the first instrument                                    |

Windows should be represented on two scales:

1. Number of rows.
2. Number of updates to the corresponding block.

For example, the last 16 price updates and the last 16 rows can represent very different intervals when prices update infrequently. Currently, these scales must be combined inside a single recurrent state.

I would start with a small fixed set of windows, without launching another broad search over window sizes.

### Anonymization Requires Care

The official documentation explicitly states that a column's index does not indicate its depth level. `p0` is not necessarily the best bid, and the first five indices cannot be assumed to represent the first five order book levels. [Official documentation](https://wundernn.io/connectome/docs/data_overview?utm_source=chatgpt.com)

According to your report, prices and volumes have also undergone a quantile transformation. Therefore, conventional microprice or imbalance formulas applied mechanically to these values may not retain their intended meaning.

The first feature implementation should use what is actually observable: changes in persistent components, their signs, events, time since updates, and interactions. Physical order flow imbalance (OFI) should be introduced only after confirming the column mappings and the suitability of their scales.

### Second Feature Package: Small Learnable Interactions

After the first package, I would test a compact block of the form:

\\[ u_t=(A\\,s_t)\odot(B\\,\Delta v_t), \\]

where \\(s_t\\) represents changes in price-like components and \\(\Delta v_t\\) represents changes in volume-like components.

The goal is to distinguish volume changes based on which price components have updated, without imposing unknown relationships between order book levels. A few output channels per instrument/side should be sufficient initially.

This is a concrete change to the data representation, distinct from the previously tested general token-attention encoder.

### Why I Favor This Direction

In their study, Kolm, Turiel, and Westray reported that models based on order flow features substantially outperformed most models using raw order book states. Cont, Cucuringu, and Zhang separately demonstrated the value of lagged order flows from other instruments for predicting future returns.

This does not guarantee improvements on our anonymized data, but it provides a strong rationale for investigating this specific direction. [Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413?utm_source=chatgpt.com)

The computational cost appears manageable: adding, for example, 60 input features would introduce \\(60\times256=15{,}360\\) additional multiply-accumulate operations in the encoder per row. Actual latency must be measured, but this experiment preserves the core recurrent architecture.

### How to Quickly Test Whether the Additional Signal Exists

Using predictions from the baseline model on components it has not seen, train a small regressor to correct its residual errors using the new event features.

Evaluate the improvement in the official metric after adding this correction, using different components.

If even a small regressor produces a consistent improvement, there is a good reason to run a full GRU retraining experiment with the new inputs. If the linear version does not help, that does not prove the signal is absent: it may reside in interactions. Nevertheless, this test can help prioritize expensive training runs.

## 4. Another Strong Idea: Train the Model to Predict Future Changes at Multiple Horizons

You already have `structure` and `next-change` auxiliary heads. Therefore, the first step is to examine their exact definitions. The new task should add information to the existing training process rather than duplicate an existing head.

I would test training the model to predict changes in several price components of both instruments at the following horizons:

\\[ H\in\\{2,8,32,128\\}. \\]

For example:

\\[ z\_{t,H}=P\left(x^{\text{price}}\_{t+H}-x^{\text{price}}\_t\right), \\]

where \\(P\\) reduces dimensionality to a few components and is fitted only on the training data. Selected individual columns could also be used initially.

Additional heads predict \\(z\_{t,H}\\) from the current causal hidden state. These heads are removed during export.

Why predict changes specifically? Predicting a raw future snapshot can often be accomplished by copying the current value, especially for columns whose values are held between updates. Predicting changes puts more pressure on the part of the representation responsible for future movements.

In practice, I would:

- Normalize auxiliary losses according to the scale of their respective horizons.
- Average these losses so that increasing the number of heads does not arbitrarily increase regularization strength.
- Start with two small auxiliary loss coefficients.
- Mask the last \\(H\\) rows of each sequence for the corresponding auxiliary task.
- Ensure the main head receives only values available at inference time.

Research on multi-horizon order book forecasting supports this direction, particularly for longer horizons. The lightweight training-only heads proposed here are an adaptation to our computational budget, not a reproduction of the heavier architecture used in that research. [arXiv](https://arxiv.org/abs/2105.10430?utm_source=chatgpt.com)

### Why Your Lookahead Ladder Does Not Establish a Performance Ceiling

The result “causal 0.666, with future rows 0.680 / 0.711 / …” demonstrates that future observations contain information about the target.

It does not measure how much information available in the past the current GRU has failed to extract. The table compares trained models, not theoretically optimal functions of their respective input sets.

For the same reason, a teacher with lookahead 512 and a score of 0.96 does not guarantee a strong causal student: much of the teacher's advantage may come from unpredictable future events.

Research on privileged distillation has indeed found a non-monotonic relationship between teacher strength and student performance. This is a useful warning when designing the experiment, although the research addresses a different task. [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2022/file/aa31dc84098add7dd2ffdd20646f2043-Paper-Conference.pdf?utm_source=chatgpt.com)

My preferred approach here is to improve the learning of predictive representations from past observations, measuring the outcome using the primary metric.

## 5. Two Aspects of the Loss That May Explain Weak Generalization

### 5.1. Normalizing Each Chunk Changes the Effective Row Weights

If the implementation uses:

\\[ L=\frac1C\sum_c \frac{\sum\_{i\in c}w_i e_i^2}{\sum\_{i\in c}w_i}, \\]

then every chunk receives the same total weight.

A row is effectively weighted in proportion to:

\\[ \frac{w_i}{W_c}, \qquad W_c=\sum\_{i\in c}w_i. \\]

Consequently, a chunk with a small total mass of official weights influences training just as much as a chunk with a large mass. This introduces additional reweighting on top of `|clip(y,2)|` and propensity.

The first check should be the distribution of \\(W_c\\). If the masses are similar, this hypothesis is relatively unimportant. If they differ by large factors, a controlled retraining run should use a fixed normalization factor that preserves the relative row weights.

It is important to:

- Match the overall gradient scale to the baseline.
- Keep the same sampler, optimizer, and training budget.
- Account for the equal contributions of the two targets to the official metric.

I do not consider weighted MSE fundamentally inappropriate for Pearson correlation. With correct weights, the conditional mean is well aligned with this objective. The suspicious aspects are the additional normalization steps and the training distribution.

### 5.2. Squared Scoring Propensity May Concentrate Training Too Narrowly

You use estimated propensity raised to power 2, along with the actual scoring mask for validation sequences included in the training data.

If \\(p(x)\\) is the probability that a row is included in scoring, then under certain assumptions, selection corresponds to weighting by \\(p(x)\\). Squaring it to \\(p(x)^2\\) is an additional concentration heuristic, not a consequence of the probability definition.

It may help. However, repeated model selection using the same visible mask creates a specific hypothesis: the model may become overly specialized to this subset of time points.

The official FAQ explicitly warns that strong performance on the provided validation subset does not guarantee strong performance on the hidden test set and recommends additional holdout evaluations. [Official FAQ](https://wundernn.io/connectome/docs/faq?utm_source=chatgpt.com)

I would compare the current recipe against one broader alternative: linear propensity weighting or a mixture of broad training and mask-weighted training. The choice should follow a diagnosis of propensity calibration.

For saved models, we can already examine:

- Official score.
- Performance across all required moments.
- Performance outside the visible mask.
- Performance changes across propensity ranges.
- Separate results for each target.

These are diagnostic slices. The primary selection criterion should remain the official metric, not whichever post-hoc slice looks best.

If the mask depends on future events even after conditioning on available history, causal propensity alone cannot fully reconstruct the selection mechanism. Therefore, changing the exponent from 2 to 1 also requires empirical validation.

## 6. Distillation: First Determine What Is Actually Being Lost

I would investigate three specific questions.

### Teacher Weights Sum to 0.8125

The report lists coefficients that sum to 0.8125. The builder may normalize them automatically, in which case there is no issue.

If normalization is absent, and \\(T\\) denotes the normalized ensemble, the effective target becomes:

\\[ 0.3y+0.7\cdot0.8125T = 0.86875\left(0.34532y+0.65468T\right). \\]

This simultaneously changes:

- The effective teacher contribution to approximately 0.655.
- The overall target scale to 0.86875.

This is not a reason to promise a large performance gain. However, it is a reason to verify the actual processing chain:

`weights → cache → blend → clipping → loss`

If the discrepancy is confirmed, scale and alpha should be tested separately. Simply normalizing the weights would change both factors.

Also, the main loss weights must correspond to the chosen objective on the true labels. Silently recalculating `|y|` from the pseudo-target would create yet another training objective.

### Does the Ensemble Improve Performance on Components Unseen by Its Teachers?

We need one table using the correct metric:

Each teacher → teacher ensemble → student → blended student heads.

If the ensemble itself does not improve performance on unseen components, the argument that we need to transfer its advantage more effectively becomes weak.

If the ensemble is substantially stronger but the student loses that advantage, we have identified a concrete compression problem involving the targets, training procedure, or student capacity.

### Are the Teachers Transferring Generalization or Memorization?

The report indicates noticeable overfitting when training is continued. Therefore, a small out-of-fold (OOF) distillation experiment would be worthwhile: the student receives predictions from a teacher that was not trained on the corresponding component.

For a fair comparison, divide the training components into A and B, train two teachers, and obtain:

- In-sample targets: A from teacher-A, B from teacher-B.
- OOF targets: A from teacher-B, B from teacher-A.

Train two students with the same parent and computational budget, then evaluate them on a third, external set.

This uses the same teachers trained on equally sized datasets. It isolates the effect of in-sample predictions without confounding it with the advantage of a teacher trained on more data.

I would run a full, expensive cross-fitting procedure only after a positive pilot experiment.

## 7. Serving May Be a Performance Bottleneck Worth Removing

According to the report:

- h256: approximately 74 µs/row.
- h384: approximately 733 µs/row.
- h256 on the competition platform: 33–37 minutes.

The official limits are one vCPU and 60 minutes; the current test contains 39.4 million rows. [Submission guide](https://wundernn.io/connectome/docs/submission_guide?utm_source=chatgpt.com)

This gives an overall budget of:

\\[ \frac{3600}{39\\,400\\,000}\approx91.4\ \text{µs/row}. \\]

However, 74 µs across 39.4 million rows corresponds to approximately 48.6 minutes, not 33–37 minutes. This means the reported measurements come from different conditions or measurement boundaries.

Furthermore:

\\[ 733/74\approx9.9, \\]

whereas the quadratic portion of computation increases by approximately 2.25 times when moving from 256 to 384.

This warrants kernel profiling. Possible explanations include a different execution path, fallback behavior, data packing, memory access patterns, or exceeding cache capacity. The cause cannot be established from a single measurement.

On one runner, I would benchmark several widths and separately measure the encoder, both recurrent layers, the head, feature processing, and the callback.

Then I would evaluate two possibilities:

1. Serve a stronger, wider model directly.
2. Serve a compact ensemble whose members have genuinely different errors.

### High Correlation Between Models Does Not Necessarily Make Ensembling Useless

For illustration, consider two equally strong, appropriately normalized predictions for one target, without clipping effects. The correlation of their average is:

\\[ r\_{\text{avg}}=r\sqrt{\frac{2}{1+c}}, \\]

where \\(c\\) is the correlation between the predictions.

For a hypothetical \\(r=0.6664\\):

| Correlation between predictions | Correlation of their average in this idealized setting |
| ------------------------------- | ------------------------------------------------------ |
| 0.99                            | 0.66807                                                |
| 0.97                            | 0.67145                                                |
| 0.95                            | 0.67489                                                |

This is not a prediction of our ensemble's score. The assumptions may not hold, and the report does not clearly specify how teacher correlations were measured. However, the calculation illustrates why we should measure each model's actual incremental contribution to the champion.

Even an individually weaker model can be a valuable addition if it corrects the champion's errors.

If the full ensemble does not fit within the inference budget, I would next consider a small correction model using the new flow features, or a shared network backbone with different upper branches. The choice should be based on measured improvement per unit of additional inference time.

## 8. Validation Must Evaluate the Actual Future Submission

There are two questions that cannot be resolved from the report.

First: which exact models correspond to the reported numbers?

- 0.6664: the current leaderboard score from your message.
- 0.66922: local H230 score on fold 1.
- 0.6639: the public result of the model family identified in the report.

Without an exact mapping of `ZIP → checkpoint → parent → cache → training set`, we cannot estimate how local improvements transfer to the public leaderboard.

Nor can we simply add the +0.003 improvement from distillation to the +0.0012 improvement from width, because these gains were measured against different baselines.

Second: the component split across the combined training and validation data.

The report describes component-disjoint validation folds and training on `train + other valid folds`. We need to confirm that all segments belonging to the evaluation component are excluded from training, including those found in `train.parquet`.

This applies to the entire pipeline: parent, teachers, caches, normalization, and propensity estimator. Excluding a component only during the final student-training stage does not undo its use by an ancestor model.

These are verification steps, not claims that leakage exists.

For new comparisons, I would use identical folds and matched seeds, and calculate confidence intervals from paired changes in the official metric, using a component-level bootstrap.

For each bootstrap sample, the global correlation must be recalculated. Averaging correlations calculated separately for individual components would produce a different statistic.

Repeatedly evaluating new finalists on a reserved fold also gradually turns that fold into part of the development process. This should be taken into account when interpreting the strength of the evidence.

## 9. What I Would Put in the Next Experiment Queue

### First: Extract Maximum Value from Existing Models

Without any new training:

1. Verify the official scorer using the minimal example.
2. Rescore the main saved predictions and identify the champion again.
3. Test blends of main and family heads separately for each target.
4. Compare GPU and native implementations using the official score.
5. Verify teacher-weight normalization and the distribution of chunk weight mass.
6. Map the current 0.6664 score to the exact artifact and training configuration.

The result should be a reproducible baseline against which it makes sense to spend additional GPU hours.

### Then: A Limited Set of New Training Runs

| Experiment                      | Single primary change                                               | Question to answer                                                          |
| ------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| A: Control                      | Exact current recipe                                                | Establish baseline performance and variability                              |
| B: Normalization                | Fixed normalization of the main loss                                | Does hidden chunk reweighting hurt performance?                             |
| C: Broader training             | Reduced concentration on the scoring mask                           | Does generalization across sets of moments improve?                         |
| D: Events from both instruments | Symmetric changes, event ages, intensities, and accumulated changes | Is there underutilized predictive signal in the current data?               |
| E: Change forecasting           | Multi-horizon auxiliary targets                                     | Does the predictive representation improve without increasing serving cost? |

I would first compare these experiments on two development folds, using matched seeds and identical computational budgets.

Then I would advance only the two most promising directions to additional seeds and less frequently used components.

Changes should be combined only after individual comparisons. Otherwise, it will be difficult to determine what helped and whether the improvement generalizes.

If experiment D produces a consistent improvement, the next architectural step would be a small memory mechanism updated by events from a selected stream, while preserving row-by-row processing and time-since-event counters.

The principle of separating memory according to update frequencies has research support. The specific event-driven adaptation would still need to be compared against a baseline with equivalent capacity. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v32/koutnik14.html?utm_source=chatgpt.com)

I would allocate most new GPU runs to cross-instrument event features and multi-horizon change prediction. I would immediately verify the metric, test blends of existing heads, and profile the native inference engine.

This combination offers a substantive path toward the required +0.0056 improvement: select the best predictors already available, provide the model with a more useful representation of predictive signals, and preserve any improvements in the actual submission.

To move from hypotheses to precise code patches, the first files needed are `gmetric.py`, the official scorer/`utils.py`, `features.py`, the loss computation code, `build_distill_cache.py`, and the head definitions in `nn.py`.

These will make it possible to quickly distinguish inaccuracies in the report from genuine implementation problems.