**The campaign has useful ideas left. Its main problem is that several experiments cannot distinguish “the idea failed” from “the test failed to preserve a competent model.”** I would prioritize stronger supervision and a few complete-model comparisons over another architecture sweep.

I traced the reviews into training code, provenance, logs and results, with two independent checks. The **0.6660 public result is real**. Nothing reviewed establishes a ceiling below 0.7000, but nothing currently supports expecting 0.7000 either.

**1. Some negative experiments have severely degraded controls.**

This is more serious than the existing review’s discussion of the joint-head experiment:

- **E415:** the capacity control loses approximately **0.038 WP**. Target-clock, event-age and cycle features recover roughly **0.007** against it, while remaining below the preserved model.
- **E421:** the unchanged predictor scores **0.6821**, but the newly fitted WMSE control scores **0.4926**.
- **E468:** even the “regression” control receives joint cross-entropy supervision and loses against its parent.

These results reject the fitted candidates. They provide weak evidence against their underlying mechanisms. A screening procedure that destroys an existing predictor is poorly suited to detecting small improvements. [E415 results](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-04/E415_conditional_proxy_report.md), [E421 results](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-04/E421_BRANCH_REPORT.md), [E468 loss implementation](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-05/joint_target_model/model.py:151).

**Change:** before screening more features, establish an anchored fitting procedure that preserves the parent and can recover a deliberately withheld, predictable correction. Then test nominated mechanisms individually. This is a small experimental repair, not a reason to rebuild the research infrastructure.

**2. The incumbent’s training targets deserve more attention than its activation functions.**

On rows where the label remains unchanged through eight future steps, the current main regression target is:

\[
\widetilde y_t=0.5\,T_{t+8}+0.5\,T_t.
\]

The realized label disappears from that regression target. It still determines weighting, the survival decision and other supervision, so this is not complete removal of ground truth. But it makes teacher error particularly relevant on long quiet spans. This follows directly from the [training implementation](/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/train_rnn.py:2021).

The recipe works publicly; I would preserve it as the control. Nevertheless, **a small amount of original-label supervision restored specifically on these rows** is a more direct experiment than another teacher ensemble or optimizer variant.

Use one fixed reanchoring coefficient, unchanged teacher caches and training schedule, and the complete final-head procedure. Include bootstrap-off as a control. Setting distillation alpha to zero while keeping bootstrap enabled does **not** produce a raw-label control.

This cannot establish that teacher bias is the bottleneck until tested. It is, however, a cheap intervention on the exact mechanism the champion uses.

**3. One promising idea was rejected using the wrong metric.**

E103 rejected label-aware scoring-selection weights because adding labels improved mask-prediction AUC by only **0.0022**, below a **0.005** gate. That does not establish that the improvement is irrelevant to weighted Pearson.

The distinction matters. Let \(S\) indicate a scored row, \(X\) be causal information, and \(Y\) the clipped target. For a loss \(\ell\),

\[
E[S\,\ell(X,Y)]
=
E[P(S=1\mid X,Y)\,\ell(X,Y)].
\]

Replacing that probability with \(P(S=1\mid X)\) requires an additional condition. A small AUC difference does not verify it, especially when metric influence varies sharply with target magnitude.

**Reopen the diagnostic, not a large training programme:** compare feature-only and label-aware selectors on component-excluded validation data using calibration and metric-weighted residual moments. Only then test training weights. Labels would be used to construct training weights; inference remains causal.

There are meaningful counterarguments: sharper propensity already failed to help an EMA recipe, and a coarse conditional-mask correction lost **0.001296 WP**. Also, transferring selection weights from validation to training requires a stable selection mechanism across those populations. These are reasons for a cheap falsifier, rather than an AUC-based veto. [Ledger E103/E170/E186](/home/vlad/development/wunder/wnn_connectome_starterpack/research/experiments.jsonl:103).

**4. Joint target structure is being overinterpreted.**

The metric evaluates the targets separately, with separate absolute-target weights. Therefore, optimal point predictions need not obey relationships satisfied by the observed label pairs.

For example, suppose a conditional distribution gives equal probability to **(1, −2)** and **(−2, 1)**. Every label pair sums to −1. Yet each target’s weighted conditional mean is −1, so the prediction pair sums to −2.

Consequently:

- Enforcing a label relationship directly on point outputs can hurt.
- A joint distribution can help through shared representation or statistical efficiency.
- Joint dependence alone does not add anything to a readout determined by the two marginal weighted moments.

E468 implements separate tilted readouts correctly. The issue is the rationale for expecting another codebook to help. I would rank that below improving temporal information or supervision. [Readout implementation](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-05/joint_target_model/model.py:12).

**5. An apparent “abandoned winner” is a false lead, and the calibration evidence is stale.**

The uncommitted `INDEPENDENT_ELEPHANTS_2026-10-05.md` recommends reviving `ab85_*_gclpall` using its superior “clean holdout” score. That conclusion is wrong.

Its launch script uses `finalall`, which explicitly enables training on **all 1,873 validation sequences—including the 368 being evaluated**. The attractive blend gains inherit the same contamination. This does not prove the models are poor; it invalidates the comparison used to prioritize them. [Launch command](/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/logs/q56.sh:5), [split implementation](/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/ft_gclp.sh:25).

Two related corrections matter:

- `PUB143` is an **inferred class-mixture scenario**, not an observed public composition.
- The cited transfer RMSE comes from **12 older submissions**; `lb_probe.py fit` reads a cached statistics file that excludes the current combo outcomes.

Use several mixture scenarios and update calibration by model family and chronology. Do not interpret the local–public score gap as a measured amount of recoverable distribution-shift loss. [Calibration implementation](/home/vlad/development/wunder/wnn_connectome_starterpack/research/lb/lb_probe.py:93), [mixture evidence, E301](/home/vlad/development/wunder/wnn_connectome_starterpack/research/experiments.jsonl:301).

I would run the next work in this order:

| Priority | Direction | Decisive comparison |
|---|---|---|
| **P0** | Repair evidence eligibility and the feature-screen control | Exclude directly fitted comparisons; demonstrate that the fitting procedure preserves the strong parent before interpreting feature negatives. |
| **P1** | Transfer the fixed GELU/LeakyReLU mixture | Current GRU256 recipe, unchanged control, complete final-head procedure. Confirm across another seed and component fold. |
| **P2** | Restore some original-label supervision on surviving bootstrap spans | One fixed reanchor versus unchanged bootstrap and bootstrap-off. Stop if the benefit disappears after the final head or reverses on confirmation. |
| **P3** | Revisit label-aware selection weights | First test held-out metric moments and calibration; proceed to a student only if there is a repeatable discrepancy worth correcting. |
| **P4** | Add a small learned memory branch | Learned versus fixed decay and a matched static branch, alongside the existing GRU. Preserve quiet-row updates. |
| **P5** | Original-origin privileged supervision | A future-informed teacher predicts the **origin row’s** target moments; compare with a causal teacher before fitting students. |

The fixed activation mixture is first because it has replicated prototype evidence and modest implementation cost—not because its **+0.0043 to +0.0089** small-panel gains should transfer at that magnitude. The learned contextual activation gate has not earned priority over the fixed mixture. [E488 results](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-05/context_mixture_duration/REPORT.md).

For memory, the most defensible lead is **learning the decay**, which helped in all three prototype seeds. Start with a compact residual memory of causal innovations and event age. Adding another event flag is insufficient: the incumbent already receives event flags, ages and last-event deltas. The new hypothesis must change how information is retained and updated. [Sequence results](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-05/sequence_components/THREE_SEED_REPORT.md), [existing features](/home/vlad/development/wunder/wnn_connectome_starterpack/src/wnn/features.py:24). Stable diagonal recurrence has relevant architectural precedent, but its benefit here remains an experiment. [LRU paper](https://proceedings.mlr.press/v202/orvieto23a.html).

For the privileged teacher, a concrete extension is to predict the origin’s conditional moments \(E[W_t\mid G_t]\) and \(E[W_tY_t\mid G_t]\), then train the causal student with the corresponding quadratic loss. This preserves the population objective **if those moments are correct**; learned estimates can introduce bias. It also avoids pretending that predicting a later row’s label is equivalent to improving the origin’s prediction. Related coefficient-denoising work was already proposed locally, and training-only future information has relevant literature. [Earlier proposal](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/continuation/round2/objective/REPORT.md:201), [privileged-information paper](https://proceedings.mlr.press/v151/k-a-karlsson22a.html).

Width and compression should run as a supporting track. **W384 remains scientifically unresolved because replication was RAM-censored**, but timing work already exists: dense quantized W384 was faster than the tested FP32 factorization. Start with actual candidate latency and memory, then one bounded width comparison. Another QAT fit needs a demonstrated quantization problem to solve. [Runtime measurements](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-04/E420_factorized/RUNTIME_REPORT.md).

My suggested allocation is **60% complete-predictor comparisons and confirmation, 25% supervision/memory experiments, 15% deployment and recovery**. Keep the queue short. Preserve the genuine negatives—hard output holding, the tested teacher ensemble, coarse categorical heads—and distinguish them from untested mechanisms.

This review changed no files and launched no fits or submissions.