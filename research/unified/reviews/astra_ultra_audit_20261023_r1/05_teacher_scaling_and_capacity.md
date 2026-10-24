---
kind: review
status: complete
created_utc: 2026-10-23
updated_utc: 2026-10-23
hypothesis_ids: [H-ASTRA23-05-A, H-ASTRA23-05-B, H-ASTRA23-05-C, H-ASTRA23-05-D]
experiment_ids: []
evidence_paths:
  - research/experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json
  - research/unified/ideas/TEACHER_QUALITY_SCALING_20261019_R1.md
  - research/unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md
  - research/unified/sources/TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md
  - research/unified/reviews/B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md
  - research/campaign_2026-10-15/student_quality_tradeoff_r1/PARENT_UPDATE_BUDGET_AUDIT_R1.json
  - research/campaign_2026-10-13/strategy_execution_r1/gradient_credit_f0_s3_r1/README.md
  - research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/DYNAMICS_PROFILE_RESULT_R2.json
---

# Review 05: identify the bottleneck before choosing teacher scale

## Scope and judgment

The strongest next question is whether the current models lack **learnable temporal credit, optimization exposure, or useful representation capacity**. The record does not yet distinguish these. Keep large causal teachers eligible, but add a compact Transformer anchor and test training clocks before treating 0.2–1.6B parameters as the natural first comparison.

This is an independent tier-C source review under the [review contract](CONTRACT.md). The [October23 stop](../../../campaign_2026-10-24/resume4h_r1/publication_source_r1/reference_snapshots/43_USER_STOP_R1.json) remains binding. I read bounded text, historical Git objects and primary literature, and wrote this report only. No fits, imports, tests, profiles, arrays, exports, provider actions or sibling reports were used. Historical measurements below retain their original exposure and ancestry. Proposed experiments are unexecuted and require separate authorization and complete delivery budgets.

## Five findings

### 1. There is no measured Transformer frontier to extrapolate

**Recorded fact.** The [teacher-first amendment](../../ideas/TEACHER_QUALITY_SCALING_20261019_R1.md) proposes six fits at approximately 206M, 679M and 1.644B parameters. Its trainer and throughput remain unmeasured in the reviewed record. The [large-teacher design](../../ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md) carefully distinguishes analytical memory from actual admission, and retains a strong recurrent comparator. These are strengths.

The original B300 [stop/result packet at commit 967f2f3](https://github.com/VladKochetov007/WunderFunder/blob/967f2f342e919e69f4f6cee39373b3b01b25c16a/research/campaign_2026-10-18/b300_execution_r1/stop_20261019_r1/STOP_CONFIRMED.json), inspected through the local Git object, contains twelve verified recurrent width/depth pairs. Equal-fold wide/deep scores are 0.68409111/0.68331379; F3 reverses the overall ordering. These are six-epoch, labels-only, fixed-epoch-5 GRUs. They neither establish a Transformer ceiling nor justify its proposed smallest scale. I did not rerun the grade verifier or rehash payloads.

**Inference.** Beginning at 206M makes architecture discovery expensive and obscures whether attention itself helps. A roughly 19M/76M same-depth contrast can expose optimization failures and useful scaling much earlier. Those sizes are suggested engineering rungs, not a predicted optimum or evidence that 206M is excessive.

**Disposition.** Retain teacher quality as the objective and distillation as optional. Modify the first ladder; defer the flagship allocation until there is either an informative compact result or an explicit decision to fund a larger exploratory leap. Do not demand that every intermediate size win: recipe rankings can change with training duration.

### 2. Nominal epochs hide an already demonstrated optimization confound

**Recorded fact.** The [parent update audit](../../../campaign_2026-10-15/student_quality_tradeoff_r1/PARENT_UPDATE_BUDGET_AUDIT_R1.json) reports 29,828 completed updates for the current four-epoch B32 parent and 59,654 for the older four-epoch B16 parent. Their recorded scores differ, but Muon scope, target cache, normalization and inference precision also differ. This is evidence that the comparison is confounded, not evidence that smaller batches caused the gain.

The distinction matters even at identical supervised row presentations. Changing batch size changes update opportunities; per-update EMA changes its sample-space timescale; changing truncation can change gradient credit without changing forward memory. A large model with poor training loss might be insufficiently optimized, incorrectly normalized, overclipped or simply using an unsuitable learning rate. High training loss alone does not prove a capacity shortage. Low training loss and poor validation do not identify an irreducible noise ceiling either.

The existing [S14 packet](../../../campaign_2026-10-13/strategy_execution_r1/gradient_credit_f0_s3_r1/README.md) is unusually discriminating: 500-row microchunks, a common 2,000-row optimizer window, and detach at 500 versus 2,000. Its later entries report completed engineering profiles but no scientific fits. The historical FP16 packet cannot serve unchanged as a new strict-FP32 experiment.

**Disposition.** Retain separate N/D/T/I/K/C/U accounting from the teacher amendment. Add fixed fitting-probe losses and explicit forward-memory/gradient-horizon clocks. Prefer a controlled credit test to another unconstrained architecture menu. A bounded LR repair is appropriate after a concrete optimization failure; indefinite tuning until the preferred architecture wins is not.

### 3. Effective data scale is neither 249.6M independent examples nor one example per component

**Recorded fact.** The [data contract](../../../../docs/data_overview.md) describes 12,480 supplied sequences of 20,000 rows, totaling 249.6M rows. Source-qualified component grouping is important because sequences and fitting/evaluation membership can be dependent. The i1 grade, for example, evaluates 468 F0 sequences grouped into 141 components.

**Inference.** Counting every row independently overstates regime diversity. Treating every component as a single observation also throws away potentially useful repeated transition structure within it. Statistical dependence, target difficulty, representation reuse and optimization exposure are different dimensions; a single guessed “effective sample size” cannot set a parameter ceiling.

Larger models could learn a shared nonlinear transition rule or optimize a representation more effectively from fixed data. Conversely, increasing width can fit sequence-specific nuisance structure without adding useful predictions. Neither possibility follows from parameter-to-row arithmetic. The informative experiment varies fitting-component diversity while holding presentations and updates fixed, then examines train and evaluation trajectories.

An overlooked ancestry trap: subsampling a dataset after loading a parent trained on the full fitting set does **not** create a half-data model. The same applies to teachers, fitted feature normalizers and pretrained stems. A real data-scaling arm must restrict its entire learned ancestry. This makes existing warm parents useful for optimization studies but generally unsuitable controls for a fresh information-budget claim.

**Disposition.** Retain nested component subsampling as a later mechanism test. Deprioritize fitting a multi-parameter scaling law from six trajectories; correlated checkpoints do not supply extra independent fits.

### 4. Attention versus recurrence is a question about access and credit, not just size

A recurrent state can carry information across 20,000 rows while gradients are truncated much earlier. A 4,096-row Transformer has shorter explicit forward access but direct connections within its window. The proposed completed-block memory introduces a third distinction: historical information is compressed before access. Equal parameter counts do not equate these learning problems.

**Recorded fact.** The [six-fit i1 grade](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json) and [independent review](../I1_SIX_FIT_HISTORY_GRADE_REVIEW_20261022_R1.md) report weaker standalone history models but fixed 90/10 mixture gains of +0.00035953 and +0.00027145 across the two exposed cells. C also changes the initial function; A's mixture improves F2. Therefore this is evidence of conditional complementarity, not proof that longer memory is the missing ingredient.

**Inference.** A useful Transformer teacher might win through easier gradient paths or selective retrieval even when a larger GRU does not. A GRU might win because streaming state and locality fit the task better. A hybrid stem is plausible, but adding learned filters, group encoders, long attention, Muon and dynamics losses simultaneously would leave any gain uninterpretable.

**Disposition.** Keep plain causal attention and the established recurrent recipe as anchors. Separate memory visibility, compression and gradient credit. Treat inference-time history removal as a sensitivity diagnostic: it also shifts the model's input distribution and cannot alone measure the value of training with that history.

### 5. Dynamics learning is a relevant hypothesis, not an automatic source of more independent data

The [dynamics profile receipt](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-21/next_cycle_r1/DYNAMICS_PROFILE_RESULT_R2.json) records 1,100 finite updates, zero skips and a complete-panel projection above its stage ceiling. All four scientific fits were deferred. That is an engineering result; there is no negative target-quality experiment to explain away.

Observable future-change targets can enrich supervision, but they come from the same sequences and may emphasize predictable persistence or irrelevant changes. Better auxiliary likelihood need not mean better weighted correlation on t0/t1. The B300 design already asks for change-conditioned errors and supervised anchors; preserve these requirements.

**Disposition.** Retain one fixed-capacity objective contrast after a functioning supervised teacher exists. Do not use an auxiliary task to justify both a size jump and a longer fit simultaneously. Supervised train/evaluation loss, auxiliary progress and official score should be separate readouts. A tiny future-head screen can identify a useful signal but cannot certify that the same coefficient or mechanism transfers to a billion-parameter teacher.

## Observations that would change allocation

These are decision aids, not proofs of unique causes.

| Observation under a frozen comparison | Supported next decision | What remains unresolved |
|---|---|---|
| Larger improves fitting loss and evaluation WP at matched presentations; improvement survives a matched-compute comparison | Allocate one higher rung | Shape, LR and optimization contributions; unseen transfer |
| Longer small model catches larger at similar measured cost | Prefer more optimization for this budget | Whether a larger model at a later budget overtakes it |
| Larger has worse fitting loss, unstable update ratios or frequent clipping | Audit optimization before calling scale ineffective | Whether repair will improve evaluation |
| More fitting components beats more repetitions at equal updates/presentations | Prioritize diversity or regularization | Composition effects and data-quality differences |
| Longer gradient credit helps without changing forward states or update clocks | Prioritize credit/memory access | Whether attention is better than repaired recurrence |
| Auxiliary loss improves while official WP and a fixed comparator mixture do not | Reduce priority of that objective/weight | Other targets, capacities and interactions |

Measure fitting loss on a fixed fitting-only probe in evaluation mode; online minibatch loss mixes changing samples, dropout and objective terms. Use fixed checkpoint rules and full prescribed evaluation coverage for decisions. Component resampling conditions on the fitted models; two fold/seed cells do not separately identify fold and seed variance. None of these outcomes warrants a public-score extrapolation.

## Four hypothesis cards

All cards have status **not_tested**. They are alternatives or conditional follow-ups, not a funded combined sweep. GPU-hour estimates below are formulas or historical anchors, never B300 forecasts. Profiles, exports and grading require additional measured allowances.

### H-ASTRA23-05-A — Recover the actual learning curves

**Family/change:** optimization diagnosis; reconstruct successful-update, row-presentation and schedule clocks for a bounded pair of existing recurrent trajectories, adding evaluation-mode fitting-probe losses at declared saved checkpoints if those checkpoints exist.

**Assumptions/mechanism:** retained checkpoints and manifests are sufficient; apparent model plateaus may be endpoint or optimization-clock artifacts.

**Predicted observable:** continued fitting-loss improvement, divergent evaluation trajectories, or an epoch/update mismatch that changes which comparison is meaningful.

**Cheapest test/controls:** first read existing text logs and manifests; then, only under future authority, evaluate at most four checkpoints per trajectory on one fixed fitting probe and one fixed evaluation panel. Keep precision, scorer and masks identical. Nonmatching historical recipes remain contextual comparisons.

**Falsifier:** comparable clocks and flat fitting/evaluation curves weaken the specific undertraining explanation; missing checkpoints make the test unavailable, not negative.

**Future cost/reuse:** zero new fits, zero teacher/student fitting; up to sixteen panel forward passes, fewer if exact caches exist. Cost is their measured sum plus grading; no current ETA. Reuse checkpoints, logs, membership and cache manifests.

**Ancestry/exposure/serving:** preserve original teacher and initializer histories; all development panels remain exposed. Diagnostic only, no serving change. Related: [teacher scaling](../../ideas/TEACHER_QUALITY_SCALING_20261019_R1.md), [quality bottleneck](../../ideas/QUALITY_BOTTLENECK_AND_STUDENT_TRADEOFFS_20261015_R1.md).

### H-ASTRA23-05-B — Learn longer credit before buying more state

**Family/change:** temporal optimization; compare detach500 versus detach2000 with identical 500-row microchunks and a 2,000-row update window.

**Assumptions/mechanism:** useful target credit extends past 500 rows, and the longer graph remains numerically and operationally feasible. Both arms retain identical forward state values before learning; the treatment changes credit assignment.

**Predicted observable:** longer credit improves the fixed endpoint or its predeclared mixture without a parameter increase; gradients and clipping may reveal why.

**Cheapest test/controls:** four new strict-FP32 continuations, two arms × F0/seed3 and F1/seed4, using catalogued fold-clean parents and each cell's fixed targets. Freeze two epochs, CP1, sampler, normalization, whole-update denominators, LR/EMA clocks and clipping once per update. Preserve the old S14 packet and declare this a new descendant. Parent-output parity and gradient-path checks precede any future launch.

**Falsifier:** longer credit loses both paired endpoints with stable optimization, or yields no decision-useful difference for its measured cost. That lowers this recipe's priority; it does not prove that older information is irrelevant.

**Future cost/reuse:** two new controls plus two treatments; zero exact reusable control fits and zero new teacher fits. These are four student continuations with inherited teacher targets. The [parent catalog](../STRONG_PARENT_F0_F1_WARM_INIT_CATALOG_20261022_R1.md) records 4,240 seconds for one historical two-epoch FP32 continuation: four times that is 16,960 seconds, approximately 4.71 hours of fitting only. Different microchunks, thermal conditions and exports prevent treating that as an ETA or an admission.

**Ancestry/exposure/serving:** preserve different F0/F1 parent/cohort histories; report each paired delta separately. Historical FP16 ancestry remains explicit despite new FP32 updates. No model-size or recurrent serving change. Related: [S14](../../../campaign_2026-10-13/strategy_execution_r1/gradient_credit_f0_s3_r1/README.md).

### H-ASTRA23-05-C — Compare capacity with buying more training

**Family/change:** compact causal Transformer capacity/optimization; add a size-versus-duration triangle below the flagship ladder.

**Assumptions/mechanism:** a compact plain supervised teacher can learn enough to make scale comparisons informative. Larger width may improve representation, while a smaller model may instead benefit from more updates.

**Predicted observable:** at equal presentations the larger teacher wins, or a longer-trained small teacher catches it at comparable measured compute.

**Cheapest test/controls:** six new fits: small at T, large at T, small at rT, each on two matched fold/seed cells. Illustrative trunks are six layers at width512/FF1360 and width1024/FF2720, approximately 19M/75M before encoders and heads. Freeze one encoding, context/core, supervision and global row batch. Determine integer r from pre-quality throughput measurements to approximate the large arm's cost; disclose the actual mismatch. Predeclare cooled terminal endpoints and intermediate stable snapshots consistently. Width changes the recipe; do not claim pure parameter causality or silently import muP guarantees.

**Falsifier:** a reliably optimized larger model has no useful fitting/evaluation advantage, and extra training also does not help at the measured budget. A failed or underoptimized large run is inconclusive, not a capacity ceiling.

**Future cost/reuse:** two fits per arm, six total teacher fits, zero student fits and zero exact reused controls. Estimated fitting work is 2(Csmall + Clarge + Clong); receiver profiles supply values. Existing recurrent predictions are contextual quality anchors unless memberships/objectives match, not matched architecture controls.

**Ancestry/exposure/serving:** fresh weights and fitting-only preprocessing; exposed cells remain exposed. Compact teachers may still exceed recurrent runtime limits; transfer is a later optional stage. Related: [teacher-first ladder](../../ideas/TEACHER_QUALITY_SCALING_20261019_R1.md).

### H-ASTRA23-05-D — Diversity versus repetitions on a fixed architecture

**Family/change:** information supply; compare full fitting-component membership with a preselected nested half at equal supervised presentations, successful updates and batch size.

**Assumptions/mechanism:** broader transition/regime coverage may matter more than additional visits to already observed components.

**Predicted observable:** full diversity improves evaluation while the repeated half fits its own probe more closely; the trajectory distinguishes useful optimization from narrower memorization.

**Cheapest test/controls:** four fresh-weight fits, two memberships × two fold/seed cells, one modest proven architecture. Fix subset selection before outcomes, retain whole sequences, document composition and match source proportions where feasible. Repeat the half to match presentations. All learned preprocessing and any teacher ancestry must exclude omitted components; prefer labels-only to avoid extra teacher refits.

**Falsifier:** full diversity does not help over repeated half at this budget, or the apparent advantage disappears under prespecified composition checks. This does not establish a universal data ceiling.

**Future cost/reuse:** two controls plus two treatments, four model fits, zero separate teacher/student stages and zero reused trained controls. Cost approximately four same-architecture fit budgets plus setup/export; measured sampler differences remain charged. Reuse split/component identities and fixed evaluation protocol, not full-data initializers.

**Ancestry/exposure/serving:** cold ancestry is essential; evaluation remains exposed. No serving change. Related: [component-fraction proposal](../../ideas/TEACHER_QUALITY_SCALING_20261019_R1.md#how-to-estimate-scaling-without-fooling-ourselves).

## Seven questions whose answers change the plan

1. Which fixed fitting-probe and evaluation curves show that a strong recurrent teacher has exhausted useful optimization, rather than merely reached its allocated endpoint?
2. Does detach2000 improve over detach500 when update frequency, loss denominators and forward memory are identical?
3. Is the first supervised Transformer learning the two targets at all before auxiliary tasks, numerical embeddings and Muon are added?
4. At a common cost, does a larger model beat a longer-trained compact one after comparable cooldown?
5. Does additional fitting-component diversity help more than repeated presentations, with the entire initializer/teacher ancestry restricted correctly?
6. Does useful long-history information survive compression into completed-block summaries, and which rows lose when it does not?
7. What measured incremental teacher quality and complete-fit cost would justify one 0.68B or 1.65B allocation, even if a smaller exploratory rung disappoints?

## Serious counterargument

A sequence of inexpensive diagnostics can become a systematic bias toward small, fast-learning models. A compact Transformer may lack the capacity needed for the relevant representation; warm recurrent continuations may remain trapped in their inherited solution. Insisting that both succeed before permitting a large teacher would prejudge the research question. The training engine itself also has a fixed implementation cost that small parameter counts do not eliminate.

This objection limits my recommendation. The credit screen should be one bounded allocation decision, not a mandatory gate for all attention research. Keep a later large exploratory arm eligible if resources are explicitly restored. Conversely, double-descent anecdotes are not evidence that an unseen large-model rescue exists here; they justify avoiding categorical exclusions, not unlimited training.

## Primary literature: inspected claims and transfer limits

- [Hoffmann et al., Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556): abstract inspected. Model size and training exposure must be studied jointly. Its token ratios and exponents are not constants for weighted financial regression.
- [Muennighoff et al., Scaling Data-Constrained Language Models, v5](https://arxiv.org/abs/2305.16264v5): abstract inspected. Repetition can remain useful before diminishing returns. Neither its repetition threshold nor its fitted law specifies our epoch budget.
- [Aicher et al., adaptive TBPTT](https://proceedings.mlr.press/v115/aicher20a.html): primary proceedings abstract inspected. Truncation introduces gradient bias, and adaptive control relies on stated decay assumptions. This motivates a fixed credit contrast; it does not establish those assumptions for our GRUs.
- [Dai et al., Transformer-XL, §3.2](https://arxiv.org/html/1901.02860v3): segment recurrence and the stop-gradient equation inspected. Reused memory can expand forward information while gradients remain local. Its language-model gains are not a numerical prior for this task.
- [Shi et al., Scaling Law for Time Series Forecasting, §§4.2–5.1](https://arxiv.org/html/2405.15124v4): context interactions and stated limitations inspected. Their in-domain forecasting analysis motivates treating history as an axis; its assumptions and experimental tasks do not validate an optimum for our hidden target semantics.
- [Nakkiran et al., Deep Double Descent, introduction and discussion](https://arxiv.org/html/1912.02292v1): model/epoch nonmonotonicity and early-stopping qualifications inspected. This counters a universal “too many parameters for few samples” prohibition, not evidence that our models occupy a double-descent regime.

## Best next decision — proposed and unexecuted

After an explicit resume, choose **H-ASTRA23-05-B**, the four-fit FP32 gradient-credit screen, if its complete cost is affordable. It reuses meaningful existing parents and an already formulated control, tests a real missing mechanism, and changes the case for larger state versus easier historical access. Use card A's existing text evidence to freeze its clocks; do not launch new diagnostics during the stop. If affordable execution is unavailable, retain this report as the completed deliverable rather than starting another partial panel.

Keep the compact Transformer triangle as a separate next allocation, not an automatic continuation or a claim that the flagship must fail. The current evidence supports keeping scale open while making its next decision observable.
