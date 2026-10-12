---
kind: review
status: complete
created_utc: 2026-10-12T19:08:27Z
updated_utc: 2026-10-12T19:08:27Z
hypothesis_ids: [H-STRATEGY-20261012-01, H-STRATEGY-20261012-02, H-STRATEGY-20261012-03, H-STRATEGY-20261012-04]
experiment_ids: []
evidence_paths:
  - research/unified/archive/2026-10-12/README.md
  - research/campaign_2026-10-12/strategy_review_r1/CONTRACT.json
---

# Challenge 2 research review and public and private leaderboard strategy

The campaign produced useful models, a correct official scorer, meaningful leakage controls and a working native Q8 submission path. Its largest weaknesses were incomplete comparisons between complete recipes, too much selection on a few exposed folds, repeated delivery failures, and explanations that became more certain than their evidence. The next phase should concentrate on three different sources of improvement: preserve and strengthen heterogeneous distillation, learn a representation less dependent on historical regime fingerprints, and test whether longer causal memory supplies information the current GRU misses.

The public objective and the future private objective need different final selection rules. Public selection seeks a high score on the public distribution. Taking the user's description of the private set as two weeks or a month of future data, private selection needs evidence that the whole predictor tolerates changes in regimes, scoring frequency and evaluation volume. Both objectives still use the unchanged official metric and locally justified candidate construction.

This is a completed review and a proposed research plan. No fits, prediction generation, blend search, uploads, provider mutations or paid actions were performed. The earlier experimental stop remains in force. The [review contract](../../campaign_2026-10-12/strategy_review_r1/CONTRACT.json) records the scope. Three native Luna/max roles independently reviewed validation, models and literature; their agreement is not statistical confirmation. Future-transfer claims remain verification tier C; local numerical evidence is exposed development evidence.

## What the evidence actually establishes

| Evidence | Observation | What it supports |
|---|---|---|
| Public A, B, C | A 0.6656 / 24:31; B 0.6668 / 25:56; C 0.6671 / 36:36 | C is the highest user-reported public result in the archive; all three are overlapping, different mixtures |
| Clean T3 F0 students | Seed 3: 0.68115536; seed 4: 0.68125842; mean 0.68120689 | Two student/initializer seeds on one exposed fold, sharing one teacher cohort |
| Older Z4 F0 student | 0.68314181 | Stronger than those T3 endpoints on this panel, but ancestry, recipe and forward precision differ |
| Old student plus T3 student, equal blend | 0.68325495, delta +0.00011314; component interval −0.001287 to +0.001517 | Small uncertain conditional value; insufficient basis for a recipe winner |
| Three H384 teacher seeds | Mean single 0.67901819; equal trio 0.68090199 | Averaging reduces some teacher error; adding half the trio to the old student adds only 0.00002562 |
| New/old F1 student mixture | 0.67182 versus new student 0.67103 | Useful candidate interaction on exposed F1; incomplete cross-fold pipeline comparison |
| H382 epoch ladder | More epochs help F0 and hurt F2/F3 in the cited recipes | Recipe-by-fold interaction; no universal best epoch |
| Native shared-body Q8 endpoint | F0 0.68349444; F1 0.67002463 | Exact serving-path quality for those specific clean endpoints, not A/B/C quality |
| Public-score regression | MAE 0.00097353 versus recent-five baseline 0.00078941 | The tested regressor supplies no useful current-package forecast |

Sources: [final models and receipts](../archive/2026-10-12/pages/02-models-and-training.md), [submission identities](../archive/2026-10-12/pages/03-submissions.md), [cross-fold source audit](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/H382_CROSSFOLD_GRADE_EVIDENCE_R1.json), [public-regression synthesis](../../campaign_2026-10-12/public_score_regression_r1/ROOT_SYNTHESIS_R1.md). Public values are user-transcribed observations, not independently verified portal readings in this review.

Several strong local results deserve retention. The H384 F1 trio at 0.67254162 and its fixed H256 mixture at 0.67264009 remain candidate teacher/blend sources; their same-fold exposure and unmatched final ancestry prevent ranking them directly against C. H382 F3 at 0.70865 is useful relative to its matched F3 comparison, not a public 0.70 result. Z4 at 0.68314181 remains a strong F0 reference. None is excluded because it lacks public confirmation.

## Consequential problems in the previous approach

1. **The comparison unit changed too often.** Scratch teachers, warm-started students, extra-head predictors, FP16/FP32/Q8 paths and all-data packages answered different questions. A promising scratch score did not establish that its distilled student or complete blend improved. The clean T3 versus old Z4 gap cannot identify whether teacher homogeneity, initializer, six-epoch schedule, scale or precision caused it. Establish a complete reference pipeline and change one factor at a time when the claim is causal; use full endpoint comparisons when the aim is simply a better predictor.

2. **Too many selection decisions depended on F1.** HPO could prune after F1 and used a fold-difficulty offset when combining F0/F1. A constant offset cannot remove recipe-by-fold interactions. Later F2/F3 evidence confirms such interactions. Search results remain useful for identifying disasters and promising families; small differences need complete matched comparisons. The final archive supersedes older claims that F2/F3 are untouched. [HPO protocol](../../../hpo/VALIDATION_PROTOCOL.md), [R4 correction](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/CONTRACT_R4_REPLAN_20261010.md).

3. **The public-incumbent recipe lacks an exact local bridge.** C's final weights fit supplied validation data. Its clean sibling must reproduce its complete member recipes and mixture while excluding each outer fold from every parent, teacher, propensity fit and learned combiner. An approximately similar student is a useful comparator but cannot silently stand in for that bridge. If exact reconstruction is costly, start with the strongest existing complete clean endpoint and label the mismatch; do not halt all useful research until the bridge is perfect.

4. **Target semantics were overclaimed.** Official documentation leaves t0/t1 anonymous. E-074 and E-302 restrict or contradict the broad next-four-event/visible-spread story; E-H284 fails to recover the proposed identity from visible columns. A formula inferred on one class should not control the whole architecture or justify delayed true-label feedback. Column indices also do not identify book depth, and volume codes may mix physical depth information. Ordered-book convolutions, physical order-flow formulas and exact symmetry augmentations need a validated mapping first. [Official data description](../../../docs/data_overview.md), [primary experiment ledger](../../experiments.jsonl), [early findings](../../findings-2026-09-26.md).

5. **Lookahead performance was mistaken for a causal ceiling.** A model scoring much better with future inputs establishes the value of unavailable information. It neither bounds the best causal model nor establishes that future-teacher gains can be distilled. Similarly, the late decline of one six-epoch OneCycle run does not isolate memorization: changing the total horizon also changes the learning-rate path. Both are hypotheses with discriminating experiments. [E-H330 and associated records](../../experiments.jsonl), [schedule implementation](../../../src/wnn/training_dynamics.py).

6. **Training weighting was treated too casually in summaries.** The actual student objective combines clipped-label magnitude, propensity powers/floors, possible since/class/episode weights, chunk normalization, teacher scale and clipping, and auxiliary losses. Alpha 0.9 alone does not describe it: the recent cache is 0.8125 times a teacher mean before clipping. These are legitimate experimental choices, but they change which errors receive optimization effort. In particular, downweighting old quiet rows can conflict with a future period containing more such rows. Measure this interaction rather than declaring the weighting wrong. [Actual weighting and target construction](../../../scripts/train_rnn.py:621), [independent target-equation review](../../campaign_2026-10-12/clean_t3_f0_cp5_grade_review_r25/INDEPENDENT_REVIEW_R1.md).

7. **Infrastructure effort did not consistently deliver scientific comparisons.** R7 passed earlier preparation stages and still invoked a literal `{python}`; it produced no fit. Other failures involved actual array formats, mount paths and memory accounting. The useful correction is one exercised path from final packet through actual producer artifacts, a short real-device profile, fit, export and grade. More bespoke receipts or a larger controller cannot substitute for this. Preserve successful compute and repair its consumer. [Terminal failure](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261012/r7_terminal_triage_r1/TERMINAL_TRIAGE_R1.json), [execution archive](../archive/2026-10-12/pages/08-execution-and-preservation.md).

8. **There is an execution backlog rather than an idea shortage.** Normalized innovations, multiple clocks, path signatures, conditional moments, privileged teachers and residual models already have registered parents. Their unrun implementations and narrow negatives must be distinguished. The next allocation should buy decisive comparisons for a small set, with an exploration reserve. Do not regenerate the same ideas under new names.

Keep the correct pooled scorer, connected-component grouping, independent weights, explicit ancestry, source-qualified group IDs, immutable negative results, and deterministic native Q8 implementation. Keep uncertainty reporting, but do not require every confidence interval to exclude zero before a bounded useful follow-up.

## A common evaluation foundation

Use one recipe/endpoint table. Each row binds parent and teacher checkpoints, fitting membership, features, sampler, target formula, optimizer and schedule, selected checkpoint, inference precision, mixture and runtime. Report what actually completed. A model name such as H390 is a hypothesis/recipe identifier, not a hidden width: C's three exported recurrent members are width 256, as shown by their `[768,256]` recurrent matrix shapes in the [build receipt](../../../submissions/10-12-best-r1/receipts/h390_s3_s4_q8_z4half_BUILD_R1.json). Its public runtime does not establish width-384 feasibility.

For each candidate, reuse valid predictions and controls first. Screen on two contrasting existing folds with one paired seed. Advance at most two complete finalists to four folds and two paired fitting seeds per fold, filling only missing valid cells. Count all teacher and initializer fits in the cost; eight student fits are not eight independent teacher pipelines. For the winner, rebuild a teacher cohort on two contrasting folds if the gain plausibly depends on one cached teacher draw. Shared teachers are acceptable for a controlled student-stage comparison but qualify the claim.

Pool the official metric across out-of-fold predictions using a common, predefined output convention, and show each fold's paired difference and the mean of seed-specific endpoint scores. Also grade the actual prediction ensemble: averaging member scores is not ensemble scoring. Fold-specific label-fitted scaling would make the pooled figure misleading; any calibration belongs inside the fitting/inner-selection process. Component bootstrap and fitting-seed variation remain separate. Report all tried configurations, not only winners.

Use training-only residual association and causal regime coverage to nominate a small number of combinations. Prediction disagreement by itself is not useful diversity. Keep equal or strongly shrunk blend weights as the reference; J1's fitted weights lost to equal weights on its evaluation partition. A target-specific mixture is eligible because the metric averages two separate target correlations, but the older H-393 proposal already covers that idea and it needs its own held-out complete-endpoint comparison.

Repartitioning exposed data cannot create a pristine test. This protocol improves the reliability of decisions while preserving that limitation.

## Plan for the best public leaderboard result

Maintain C as the observed public reference. Form new candidates from matched local evidence, then submit a small frozen set for transfer confirmation. Investigate a local/public disagreement through served precision, source recipe, all-data training and distribution diagnostics. Do not fit blend weights to the leaderboard or revive the failed public-score regressor.

The near-term route is a better complete heterogeneous student or a complementary representation that earns weight in its blend. Compare the old heterogeneous pipeline, the clean homogeneous pipeline, and a changed source treatment with the same initializer, student schedule and precision. Teacher family and target amplitude are separate factors; do not change both under one treatment name. Reuse the current cache to test amplitude/clipping cheaply before training more teachers when that ambiguity blocks interpretation.

Then advance one teacher scale ladder and one representation ladder. Suggested teacher widths are 256 → 384 → 512, with 768 conditional on measured resource fit and useful teacher, student or conditional-blend signal. An isolated negative H512 mixture does not close larger teachers. Compare both matched update/data budgets and a bounded compute-matched alternative; holding epochs constant alone can confound optimization and cost. A larger teacher is allowed to exceed serving limits, with a specific compression/distillation comparison.

Use Q8 serving headroom for validated signal. Additional seeds, a second model family, wider state, a shared trunk with extra heads and quantization are competing ways to spend that headroom. Compare the full compressed endpoint. Do not assume Q4 is twice as fast: the tested conversion lost about 0.00180 on a tiny panel and did not establish a native INT4 speed benefit. Recurrent precision needs complete 20,000-row trajectories and resets, since one-step agreement can hide accumulated state error.

The gap from the archived 0.6671 to the 0.700 objective is 0.0329 on the public metric. The completed small blending gains do not explain that gap. This motivates a meaningful reserve for new information representations and teacher capacity, alongside exploitation. It supplies neither a ceiling nor a score forecast.

## Plan for the most reliable future private result

**Use time-forward evaluation only with real chronology.** Within-sequence steps are ordered, but sequence IDs are shuffled. Inferred adjacency components protect train/evaluation separation; they do not establish calendar dates or authorize carrying state across sequences. If a permitted sequence-time map or additional labeled data explicitly authorized for participant training becomes available, use rolling-origin fits with separate two-week and four-week assessment horizons and keep connected groups disjoint. An embargo requires a supported upper bound on target overlap; the anonymous target horizon cannot be replaced by the disputed four-event formula. Where overlap cannot be bounded, keep whole connected components separated and report residual uncertainty. Recency weighting must be learned from these comparisons, not from sequence IDs. Without that information, call the following tests domain stress tests.

**Construct a small fixed set of plausible shifts.** Fit domain definitions using training-only observable summaries: activity/change frequency, causal variability, instrument-1 update frequency, code/feature support and train-versus-valid source similarity. Use complete components and sufficient support per environment. Historical source separation AUC near 0.99 motivates this diagnosis, but it does not identify what the next month will look like. Fit normalizers and domain definitions within each outer split; whole-sequence summaries may be used for offline evaluation grouping but never to route an early-row prediction.

Evaluate ordinary grouped CV, held-out domains, high/low activity mixtures, quiet/transition-heavy mixtures and different plausible scoring propensities. Predeclare a small scenario set; selecting hundreds of stress panels until one predicts the public order is another overfit. Preserve official-mask scores as primary measurements. Alternative masks or reweighting are sensitivity analyses. When domain proportions change, recompute pooled WP from combined moments or predictions: a weighted average of domain correlations is generally the wrong functional.

**Choose broad support with limited complexity.** Prefer a complete endpoint with strong ordinary CV, low seed instability and tolerable degradation under several plausible shifts. Inspect relative candidate/control losses on difficult domains rather than maximizing the minimum absolute domain correlation, which can concentrate all effort on irreducible noise. A small mean gain may be traded for reduced downside, but select the risk preference before seeing public outcomes. Group-DRO is one training treatment; strongly regularized ordinary training remains a required control.

**Allow causal adaptation only to information actually received.** A raw branch plus causal running-statistic/innovation branch can respond within a sequence. Initialize its statistics from fitting data, shrink early estimates toward that prior, preserve raw scale/context, process the 99 warm-up rows, and reset everything at the sequence boundary. Train with exactly the same causal update equations. The callback does not receive true targets or `is_scored`, so supervised online residual correction is unavailable unless an independently validated label reconstruction exists. Full-sequence normalization and cross-sequence memory violate the intended interface. Generic unsupervised parameter updates are an exploratory arm because feature-prediction improvement need not improve t0/t1.

**Treat runtime as part of private reliability.** C's public time is 2,196 seconds; B's is 1,556 seconds. Illustratively, twice the rows at the same marginal cost would give approximately 73.2 and 51.9 minutes respectively, before changes in setup/host behavior. Under the [documented 60-minute whole-test cap](../../../docs/submission_guide.md), the first would exceed the limit. These are sensitivity calculations, not runtime forecasts or evidence that private has twice the rows. Establish the private row-count bound and applicable time policy before final freeze. Budget `setup + rows × measured callback cost + margin` and use a fixed deterministic architecture. Do not fill all public-time headroom while final volume is unknown.

Keep a compact robust candidate alongside the higher-mean public candidate. The repository's competition rules say final choices come from previously submitted and scored solutions, so a robust private candidate also needs a successful submission before the deadline even if its public score is slightly lower. Confirm the actual number of final-selection slots and current deadline through the official portal before scheduling; do not assume two slots. If only one is allowed, prioritize the robust candidate under the user's future-data premise, provided its loss in expected score is justified by measured risk reduction.

## Ranked mechanisms to investigate

The [machine-readable cards](../ideas/STRATEGY_REFINEMENTS_20261012.json) contain parent IDs, controls, falsifiers and relative cost. These are refinements or interactions, not claims that old ideas are novel.

| Priority | Mechanism | First useful comparison | Why deepen it |
|---|---|---|---|
| 1 | Complete heterogeneous distillation and source weighting | Matched old/new source targets with fixed parent, student and precision; separately isolate target scale or since/propensity treatment | Best existing production lineage; present comparisons leave the main source of improvement unresolved |
| 2 | Raw context plus causal innovations, tested against source/activity shifts | Raw; raw plus causal moments; raw plus moments and normalized innovations, equal-capacity controls | Plausible way to retain useful level information while reducing regime dependence; direct private relevance |
| 3 | Multi-clock memory and gradient credit | Hold optimizer window fixed while changing detach horizon; separately test event/path memory with duration and novelty summaries | Forward state alone does not prove useful long-memory learning; event/row rates vary |
| 4 | Larger causal teachers with a complete student handoff | Width/state/context rung versus matched teacher; identical downstream student and fixed endpoint | Can buy information unavailable from additional same-family student seeds |
| 5 | Privileged moment teacher projected into a causal student | Cross-fitted short-lookahead teacher versus causal teacher, with original-time labels/weights and a genuine causal student | High upside, but oracle teacher quality alone is uninformative about transfer |
| 6 | Relational distillation or a small conditional residual | Output-only versus output-plus-relational hints, or a small residual against an equal-capacity control | May retain complementary teacher structure without paying for several served models |

There are four useful concrete refinements.

**Causal innovations with mask robustness.** Interact the existing H-238 representation with a bounded weighting comparison. Keep a fixed raw-context path; add slow and fast causal feature moments and normalized changes. Compare the existing mask-propensity weighting with a less concentrated, predeclared alternative using the same training membership. Diagnose whether gains follow input shift, mask concentration or their interaction. A normalizer that removes predictable regime information can lose even when it makes distributions look more similar.

**Path memory with a learned activity clock.** Refine H-227/H-E537-E/H-136 by storing low-dimensional signed path-area summaries of instrument-0 versus instrument-1 innovations over trailing event windows. Such summaries can distinguish which instrument moved first even when endpoints and simple sums match. Learn a bounded causal time increment from observed change magnitudes; retain elapsed-row duration and process every row. The learned clock must change memory decay/window membership or contribute a time channel: merely reparameterizing time leaves spatial path areas unchanged. Use increment-only, fixed-clock area and learned-decay/time-augmented controls. This differs from the already weak second event-GRU recipe. First verify complementary residual information with a bounded screen; full-model training determines predictive value. An LRU/state-space teacher is a second implementation option, not a required native-runtime rewrite.

**Metric-aware privileged moments.** For one target, let `Y = clip(y,−2,2)`, `W = |Y|`, `S` be the scoring indicator and `H` the allowed history. Under the scored, weighted distribution, the conditional-mean predictor is

`m(H) = E[S W Y | H] / E[S W | H]`.

With nonzero denominator and variance, this conditional mean maximizes weighted correlation up to a positive affine transformation before clipping; weighted squared error has the same population minimizer. This is a mathematical property, not a promise that a neural estimator achieves it. Therefore ordinary clipped-label WMSE is already a sensible baseline. A distinct experiment predicts numerator and denominator using own, cross-fitted teachers and projects those moments separately into a causal student, with denominator shrinkage and a direct-regression fallback. Averaging teacher ratios is not equivalent to a ratio of projected moments. Applying the original outcome weight again to the moment-projection loss would also change its estimand; preserve the declared moment regression measure. It must beat direct WMSE and output distillation. The existing 504-code outcome head also uses tilted moments; the refinement avoids relying on an unverified universal spread/target law. Any scoring-propensity approximation is estimated within fitting data and its assumptions remain explicit.

**Geometry-preserving teacher compression.** After a better own teacher portfolio exists, transfer distances/angles between selected teacher representations rather than forcing coordinate equality between unrelated hidden states. Select a small set of within-component pairs stratified by causal event age; avoid a quadratic all-row comparison. Compare output distillation alone, coordinate hints and relational hints with the same student. Keep hints training-only. Their value is measured by official score and compressed blend contribution, not by how closely student embeddings resemble the teacher.

## Literature that changes the experiment design

The sources motivate components; none establishes an improvement on this competition.

| Primary source | Useful component and limit |
|---|---|
| [Sirignano and Cont, Universal features of price formation, 1803.06917](https://arxiv.org/abs/1803.06917) | Supports investigating pooled training and longer price/flow history. Their conventional volatility/price/spread normalizations did not improve results, which is counterevidence to blanket normalization. Anonymous features here also limit direct physical transfer. |
| [Gulrajani and Lopez-Paz, In Search of Lost Domain Generalization, 2007.01434](https://arxiv.org/abs/2007.01434) | Treat the selection procedure as part of the method. Use a strong ordinary-training baseline before attributing gains to invariance or a domain algorithm. |
| [Sagawa et al., Distributionally Robust Neural Networks for Group Shifts, 1911.08731](https://arxiv.org/abs/1911.08731) | Robust group weighting needs regularization; fitting the worst training group does not guarantee worst-group generalization. Groups here must be defined without evaluation labels. |
| [Lopez-Paz et al., Unifying distillation and privileged information, 1511.03643](https://arxiv.org/abs/1511.03643) | A teacher can use a richer training representation. The causal student must demonstrate the improvement on excluded groups; future-aware teacher score alone is not the outcome. |
| [Heckel and Yilmaz, Early Stopping in Deep Networks, 2007.10099](https://arxiv.org/abs/2007.10099) | Motivates separating layer learning rates and training duration when diagnosing late deterioration. It does not establish double descent in our GRU. |
| [Orvieto et al., Resurrecting Recurrent Neural Networks for Long Sequences, 2303.06349](https://arxiv.org/abs/2303.06349) | A linear recurrent state is a useful alternative for a large-memory teacher or compact memory branch. Sequence benchmark efficiency is not evidence of one-row CPU latency here. |
| [Gu and Dao, Mamba, 2312.00752](https://arxiv.org/abs/2312.00752) | Input-dependent state updates motivate selective memory. Its GPU implementation and language results do not certify a deployable competition model. Try as a teacher only if a discriminating memory hypothesis survives. |
| [Liu et al., Non-stationary Transformers, 2205.14415](https://arxiv.org/abs/2205.14415) | Preserving distribution context alongside stationarization motivates the raw-plus-innovation branch. Transfer the representation idea, with causal prefix statistics, rather than assuming the paper's full model fits this interface. |
| [Jhin et al., Learnable Path in Neural Controlled Differential Equations, 2301.04333](https://arxiv.org/abs/2301.04333) | Learning a latent path motivates a bounded input-derived memory clock. The proposed decay/time-channel refinement is our extrapolation, and must avoid vacuous time reparameterization. |
| [Park et al., Relational Knowledge Distillation, 1904.05068](https://arxiv.org/abs/1904.05068) | Distance/angle matching offers a training-only way to transfer representation relationships without identical hidden coordinates. Market forecasting value remains untested. |

The arXiv skill wrapper was inspected and executed by the scout; it used its curl fallback because the Python arxiv package was absent. Four queries covered order-book forecasting, order-flow imbalance, point-process microstructure and nonstationary time-series normalization. Broad retrieval was noisy, so source claims were verified on primary arXiv pages. A random neighboring-method selection used seed 20261012 over `[1612.09328, 2005.08926, 2009.08295, 2301.04333]` and selected index 3. The sampled abstract component was the replacement of a fixed interpolated path with a learned latent path. This selection supplied the path-memory refinement; it is not empirical support. [Search receipt](../../campaign_2026-10-12/strategy_review_r1/LITERATURE_SEARCH.json).

## Dimensions worth scaling

| Dimension | Bounded ladder | Confound to hold fixed or measure |
|---|---|---|
| Independent data coverage | Complete-component 25%, 50%, 100% screens; full-data confirmation for scale-sensitive arms | Repeated rows and sampling with replacement do not add independent trajectories; report unique groups, rows and updates |
| Teacher capacity | Width 256, 384, 512, then conditional 768; alternatively one linear-state/low-rank memory family | Teacher quality, conditional value and student transfer are separate outcomes; include cache/export cost |
| Gradient credit | Example: 500 versus 2000 connected rows at a common 2000-row optimizer window; extend only if informative | Keep optimizer, LR, EMA and loss-denominator clocks fixed; profile actual graph memory |
| Memory representation | Row context plus a small event/decay state, then a larger research teacher | Duration proxies are not verified physical timestamps; reset at every competition sequence |
| Data views | Full inputs versus a declared instrument-1-masked training view or a group-aware encoder | Preserve column identities; block masking is a robustness treatment, not a proved label symmetry |
| Independent fits | Two, then four student seeds only if marginal ensemble value justifies cost | Same teacher cache does not provide independent teacher replication; do not select the best seed |
| Compression | Existing Q8, then selected sensitive tensors at higher precision or trained low-rank/QAT variants | Full-sequence score, actual callback cost and archive bytes decide value; bit width alone does not |

Avoid a Cartesian product of these ladders. A promising mechanism receives a small local sweep; a weak one preserves its evidence and gives way to a different information source. The existing next-feature forecasting auxiliary and pure chunk-correlation losses already have history; inspect their exact terminal results before spending on renamed copies.

## A bounded next phase

Once experiments are explicitly resumed, first establish the complete baseline and salvage reusable exact predictions/checkpoints. Run one final-packet smoke path against real producer formats, then reuse that path. Do not build another orchestration framework.

The first scientific cohort should contain at most three treatments: a matched distillation-source or target-scale contrast, the raw-plus-causal-innovation representation, and a memory-credit comparison. Use two contrasting folds with a paired seed and a shared matching control where valid. This means at most six candidate student fits plus necessary controls, teachers and exports; it is a selection budget, not admission against stale cloud credit. A small-data screen preserves whole histories/components and only removes clear failures when ranking fidelity is unknown.

Advance at most two endpoints to the four-fold/two-seed matrix. Reserve a larger-memory or privileged-teacher pilot as the exploration branch if the first cohort gives no useful signal. A positive uncertain result can earn another bounded test when its information value is high; a low-value result can lose allocation without being declared impossible. Measure completed valid endpoint comparisons per GPU-hour and end-to-end wall time, including exports and collection.

Suggested effort allocation for a resumed phase is 45% complete-pipeline improvement, 30% representation/teacher exploration, and 25% independent confirmation plus exact serving. Reallocate after the first cohort. These percentages are a planning preference, not a spending authorization or a claim about optimal allocation. Current cloud quota/access must be refreshed at admission; historical zero Vast credit cannot fund a new rental, and no top-up is authorized.

Scale the winning information source, then compress and freeze two purposes: a high-mean public endpoint and a compact endpoint robust across plausible future shifts. They may converge to the same predictor. The first decisive deliverable is a trustworthy table showing whether a new complete predictor improves on the strong clean reference across folds, seeds and declared shifts. More hypothesis rows alone do not deliver that result.

## Source records and review limits

- [Final archive](../archive/2026-10-12/README.md) and [completed drain](../archive/2026-10-12/pages/00-overview.md).
- [Current-state log](../../../specs/current-state.md): its newest drain entry is at the bottom; the opening 05:54 snapshot is not the final state.
- [Official metric implementation](../../../src/wnn/gmetric.py) and [submission requirements](../../../docs/submission_guide.md).
- [Historical registered hypotheses](../../hypotheses.jsonl), [experiment records](../../experiments.jsonl), and [new refinement cards](../ideas/STRATEGY_REFINEMENTS_20261012.json).
- [Review contract](../../campaign_2026-10-12/strategy_review_r1/CONTRACT.json) and [experimental stop receipt](../../campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json).

This review inspected source text, artifact manifests and recorded grades; it did not recompute prediction arrays or independently retrain the models. The final future-period premise is taken from the user. The official portal pages returned an empty application shell to the root browser tool, so current private volume, final-selection slots and any changes to deadlines/rules remain operational checks before release. No leaderboard rank or score above the observed incumbent is forecast.
