# Final research portfolio — 25 source-qualified ideas

**Date:** 2026-10-17. **Status:** proposal only. Model fitting/training, calibration fitting, and online optimizer updates are not authorized; bounded inference of frozen checkpoints is allowed by the contract. These are 25 diverse research hypotheses, not 25 claims of novelty, feasibility, or winning effect.

**Evidence boundary.** The 17 October EDA is exploratory: its all-sequence census reads only four price-code columns plus targets/masks; its full-feature sample covers 256 sequences with equal retrospective strata. Every validation fold is exposed. Retrospective strata are diagnostics only, never model inputs or gates. The i0 conjunction is not an exact target trigger (1,253 counterexamples; 92 valid exceptions are officially scored), zero identical adjacent full rows was observed only in the sample, and i1 marginal drift is confounded by sequence composition. The official pooled clipped weighted-Pearson metric remains unchanged.

**Shared future protocol (CP30; reference only):** CP30 reference, only after its gate passes and training is explicitly authorized: 30 paired fold×seed cells per arm (60 cell-runs for a two-arm comparison): F0 and F1 each use seeds 1001–1008; F2 and F3 each use seeds 1001–1007. Fold-membership artifacts, exact recipes, update counts, model adapters, and commands remain pending. Before any run, record clean, auditable teacher/initializer ancestry. Use strict FP32 with autocast and TF32 disabled; same exposure, resets, and unchanged official scorer global_wp_clip2_v1 for each paired candidate/control cell. The primary delta D is the equally weighted mean of four fold means, each fold mean averaging its paired seed deltas over its available 8/8/7/7 cells. For the 10,000-replicate seed bootstrap, hold the four folds fixed and resample the eight seed identities with replacement, using the same seed multiplicities across folds; within each replicate, renormalize over the available seed cells in each fold and average the four fold means equally. Do not resample the four folds. Any component resampling is conditional on fixed predictions and does not replace training-variation uncertainty. These exposed folds are development data, not a new holdout; candidate-quality comparisons remain future-only until training is authorized. Any blend is a separate interaction test; gains are not additive and no numerical gain prior is asserted.

Priority tiers mean information value per cost, not predicted score change. Cards use `new`, `reopened`, and `continuation` relative to the audited project inventory, not to all literature. No model has been trained for this portfolio.

## H-RR17-01 — Converged cold wide one-layer student versus two-layer student

**Широкий однослойный студент с обучением с нуля против двухслойного** · **Status:** continuation · **Priority:** T1 — High information value for the current compact endpoint; no score-gain estimate.

**Mechanism.** Compare a wide one-layer GRU student trained from scratch to convergence with a two-layer student, matching parameter budget, training rows, updates, input path, and stopping rule. Keep full causal state and reset each 20,000-row sequence.

**Evidence and counterevidence.** The old H512 screen and PR2 already exposed this depth/cold-start direction, but do not establish a replicated converged comparison or CPU win. One layer may reduce sequential inference work; it may also underfit or need more updates.

**Prior overlap.** Continuation of the H512/PR2 screen, GRU width-depth and cold-start inventory; not a new model family.

**No-fit gate.** Audit checkpoint ancestry, exposure, logs, parameter counts, and callback latency estimates for the old pair. If the recipe is already matched or the runtime difference is negligible, do not reopen.

**Falsifier.** Defer if one layer underfits at matched exposure or fails to improve paired endpoint quality or measured CPU cost.

**Future control.** CP30: pair cold one-layer and two-layer endpoints under the same data, optimizer-update, selection, and inference budgets; grade the student, not an intermediate checkpoint.

**Compute and CPU path.** Moderate future training. The one-layer model is itself the CPU endpoint; measure callback time and package size rather than inferring speed from parameter count.

**Sources.** [1](https://wundernn.io/connectome/docs/data_overview)

**Training authorized:** no.

## H-RR17-02 — Large causal teacher with a same-context compact student

**Крупный причинный учитель и компактный студент с тем же контекстом** · **Status:** continuation · **Priority:** T1 — High information value because it tests the actual compression route; no expected lift implied.

**Mechanism.** Train a larger causal sequence teacher and a compact student on identical current-and-past inputs, prefix statistics, row alignment, and sequence resets. Distill per-row outputs while retaining supervised training labels; never let the teacher see later inputs or target history.

**Evidence and counterevidence.** This is the existing PR1/PR3 route, supported by recipe-specific local KD positives. Those panels are exposed and do not identify a universally better teacher or student transfer; the newer warm-parent result is one trajectory.

**Prior overlap.** Explicit continuation, not new: existing causal Transformer/GRU teachers, heterogeneous GRU KD, and teacher-cohort proposals.

**No-fit gate.** Audit teacher/student preprocessing hashes, context masks, free-running state, and reset parity. If contexts differ or the source cache is incomplete, resolve provenance before considering a comparison.

**Falsifier.** The student fails against direct supervised training on paired cells, regardless of teacher loss or teacher-only score.

**Future control.** CP30: same student architecture with direct supervision versus same-context output KD; preserve one fixed teacher lineage and count the student as the deployable endpoint.

**Compute and CPU path.** High teacher-training and cache cost; compact recurrent student has ordinary CPU inference and no teacher runtime dependency.

**Sources.** [1](https://arxiv.org/abs/2608.13391), [2](https://github.com/nv-tlabs/cmd), [3](https://wundernn.io/connectome/docs/data_overview)

**Training authorized:** no.

## H-RR17-03 — Soft event-paced recurrent write and decay

**Мягкая событийная скорость записи и затухания рекуррентного состояния** · **Status:** reopened · **Priority:** T1 — High information value after the falsified hard-event rule; no effect-size forecast.

**Mechanism.** Use a few past-only input-change and duration traces to modulate a recurrent state’s soft write/decay rate. Every row still updates the state; the gate never freezes outputs or uses the i0-price conjunction as an exact target trigger.

**Evidence and counterevidence.** The input-only lag audit finds short-run persistence but heterogeneous i1 alignment and no target evidence. Extras change on nearly every sampled row and no adjacent full rows repeat. The exact i0 conjunction has 1,253 census counterexamples, including scored valid exceptions.

**Prior overlap.** Overlaps event-age, CUSUM, Hawkes-trace, GDN and clock proposals. The distinct claim is state-write control, not another appended EMA bank.

**No-fit gate.** On existing feature-only streams, test predeclared traces against scalar activity and row-age baselines across sequence halves; exclude catch-all extras. Do not inspect labels to choose a gate.

**Falsifier.** Stop if traces collapse to activity/row age, are unstable across halves, or fail a later matched endpoint comparison.

**Future control.** CP30: compare soft write/decay with the same recurrent model using fixed state updates; keep all rows available and use sequence-only resets.

**Compute and CPU path.** Low state cost for a few scalar traces; make the gated recurrent model the student endpoint before considering a larger teacher.

**Sources.** [1](https://arxiv.org/abs/2412.06464), [2](https://eml.berkeley.edu/symposia/nsf98/engle_russell.pdf), [3](https://wundernn.io/connectome/docs/data_overview)

**Training authorized:** no.

## H-RR17-04 — Teacher hidden-state hints versus output-only distillation

**Подсказки скрытого состояния учителя против дистилляции только выходов** · **Status:** new · **Priority:** T1 — High information value on the teacher-to-student bottleneck; no gain claim.

**Mechanism.** With teacher and student on the same causal context, match a small set of teacher intermediate states to student states through a projection loss, alongside the same label and output losses. Remove hint projections at inference; do not add future-feature, family-label, or moment targets.

**Evidence and counterevidence.** FitNets introduced intermediate hint matching, but on vision tasks. Existing local student work mostly tests output targets; old moment/family auxiliaries are not hidden-state matching. No evidence shows hidden-state transfer helps this stream, and an incompatible state may overconstrain the student.

**Prior overlap.** Distinct from output KD in card 02 and from label-derived moment/family heads; adjacent to existing representation distillation proposals.

**No-fit gate.** Use synthetic causal sequences to verify row alignment, reset behavior, projection shape, and future-suffix invariance. Audit that teacher state at t depends only on inputs through t.

**Falsifier.** Drop if hidden hints do not improve the compact student over output-only KD under equal training exposure, or damage CPU feasibility.

**Future control.** CP30: output-only KD versus output-plus-hidden-hints using one fixed teacher and student; report student endpoint and keep hint weight selection inside training folds.

**Compute and CPU path.** Moderate extra teacher-state cache and student training cost; no added inference heads or CPU state after projection removal.

**Sources.** [1](https://arxiv.org/abs/1412.6550), [2](https://arxiv.org/abs/2608.13391)

**Training authorized:** no.

## H-RR17-05 — Asynchronous i1 lag reliability gate with a no-i1 expert

**Шлюз надёжности асинхронных лагов i1 с ветвью без i1** · **Status:** reopened · **Priority:** T1 — High information value because a preprocessing defect tainted earlier interpretation; no uplift implied.

**Mechanism.** After restoring saved preprocessing, compare an i1 expert with a small causal lag basis and feature-only reliability gate to a no-i1 fallback. The gate uses current/past inputs only, never retrospective classes or future targets.

**Evidence and counterevidence.** Frozen CPU replay on strict425 groups [3, 6, 14, 15] (11,522 scored rows) moved WP from 0.06289742 with ZERO_COLS=None to 0.72994782 after settings restoration; manual/config inputs and predictions matched bitwise. This isolates a serious preprocessing defect on that slice, not i1 benefit or general model quality. Sampled i1 drift is confounded; lag effects vary by stratum.

**Prior overlap.** Reopens i1-flow/cross-impact after a confirmed restore defect; overlaps prior i1 features and cross-stream lags. Replay validates restoration only.

**No-fit gate.** First reproduce bitwise manual/config parity. For the lag hypothesis, use predeclared input-only lag and phase-shift controls across sequence halves; never tune on labels or retrospective classes.

**Falsifier.** Replay is not an i1 result. If lag signs remain heterogeneous, controls fail, or paired endpoint adds no value, retain the no-i1 path.

**Future control.** CP30: same student and exposure with versus without the i1 lag gate; score pooled WP and keep all routing causal.

**Compute and CPU path.** Medium cost; dual experts may exceed CPU limits, so distill a surviving gate into one compact student or drop it.

**Sources.** [1](https://arxiv.org/abs/2112.13213), [2](https://www.ms.u-tokyo.ac.jp/~nakahiro/mypapers_for_personal_use/hayyos03.pdf), [3](https://wundernn.io/connectome/docs/data_overview)

**Training authorized:** no.

## H-RR17-06 — Strict prefix normalization with a raw-level escape path

**Строгая префиксная нормализация с сохранением исходного масштаба** · **Status:** continuation · **Priority:** T2 — Useful information test at low serving cost; no expected gain implied.

**Mechanism.** Normalize each feature with sequence-local location/scale computed through t−1, predict at t, then update statistics with x[t]. Preserve a bounded raw-feature bypass so normalization cannot erase absolute level; reset statistics per sequence.

**Evidence and counterevidence.** RevIN motivates shift-robust normalization, not this causal information set. Sampled i1-volume JS is descriptive on 32 sequences per stratum and may reflect sequence mix. Prior prefix-normalization work remains an audit/continuation, not evidence of forecast gain.

**Prior overlap.** Continuation of PR1/PR3 prefix-only normalization and causal smoothing; distinguish from full-window RevIN.

**No-fit gate.** Future-suffix perturbation must leave earlier transformed rows unchanged; verify finite scales, warm-up floors, reset parity, and exact raw bypass on fixed inputs.

**Falsifier.** Stop if normalization removes useful level ordering by construction, prefix scales are unstable, or the matched endpoint does not improve.

**Future control.** CP30: strict-prefix transform plus raw bypass versus raw baseline, with the same student and training exposure; teacher and student use identical statistics.

**Compute and CPU path.** Low streaming counter cost; width increases modestly, so measure native CPU latency and model size.

**Sources.** [1](https://openreview.net/forum?id=cGDAkQo1C0p), [2](https://github.com/ts-kim/RevIN), [3](https://wundernn.io/connectome/docs/data_overview)

**Training authorized:** no.

## H-RR17-07 — Distribution- and dependence-preserving distillation versus mean/family targets

**Дистилляция распределения и зависимости против средних и семейных целей** · **Status:** continuation · **Priority:** T2 — Tests a specific distillation bottleneck; no score forecast.

**Mechanism.** Distill a fixed causal teacher’s small per-target quantile summaries and cross-target dependence summary into a compact student, alongside identical supervised labels. The deployed interface remains two point predictions; this tests the information bottleneck in mean-only/family-target distillation, not a generic categorical head.

**Evidence and counterevidence.** Soft targets can preserve output structure in principle, but no project result validates this transfer. E468 joint codebook was negative against a losing anchor, and histogram MSE/CE were negative; those are strong warnings against renaming a probability head.

**Prior overlap.** Continues the R2 distribution-preserving KD synthesis. Distinct from hidden-state hints only if it carries conditional output distribution/dependence, not hidden activations.

**No-fit gate.** Check that an existing causal teacher has stable out-of-fold quantile/dependence outputs beyond its means. If no such frozen artifact exists, do not generate a teacher now.

**Falsifier.** Stop if the summaries collapse to means across folds or the student fails against output-only KD under the same teacher and compute.

**Future control.** CP30: matched compact student with output KD versus output-plus-distribution/dependence summaries; use same teacher, contexts, labels, and scoring rule.

**Compute and CPU path.** High teacher/cache cost, moderate student cost; remove distribution heads at inference so the CPU student still emits two values.

**Sources.** [1](https://arxiv.org/abs/1503.02531), [2](https://arxiv.org/abs/2608.13391)

**Training authorized:** no.

## H-RR17-08 — Next observable event family and waiting-time auxiliary

**Тип следующего наблюдаемого события и время ожидания как вспомогательная цель** · **Status:** continuation · **Priority:** T2 — Cheap representation question after event-vocabulary gate; no gain estimate.

**Mechanism.** Train a small auxiliary head to predict the next predeclared input-feature event family and a censored row waiting-time distribution from current/past context. These future input events are training targets only; they never become prediction-time features or target-event proxies.

**Evidence and counterevidence.** ACD work motivates duration modeling, and feature-only lag controls show some short dependence. Prior next-change auxiliaries exist and did not establish main-score gain; catch-all extras make an unfiltered any-change target nearly constant.

**Prior overlap.** Continuation of next-event/first-move auxiliaries; sharpened to observable input events and censored waiting time, not future target labels.

**No-fit gate.** On fixed feature-only segments, require event-family renewal prediction to beat a sequence-local constant-rate baseline. Exclude near-always-changing extras before any training proposal.

**Falsifier.** If event vocabulary is trivial, censoring dominates, or the same student gains nothing over ordinary supervision, defer.

**Future control.** CP30: same student with and without the auxiliary; head removed at serving. Keep event labels strictly future-training targets and respect sequence-end censoring.

**Compute and CPU path.** Low-to-medium training overhead; no serving overhead after removing the auxiliary output.

**Sources.** [1](https://eml.berkeley.edu/symposia/nsf98/engle_russell.pdf), [2](https://arxiv.org/abs/2412.09631)

**Training authorized:** no.

## H-RR17-09 — Low-order path signature versus a fixed cross-lag basis

**Сигнатура траектории низкого порядка против фиксированного межпоточного лагового базиса** · **Status:** continuation · **Priority:** T2 — Information priority is conditional on beating the simpler lag basis; no gain implied.

**Mechanism.** Compute one reset-local antisymmetric second-order area from a few stable feature-group innovations, then test whether it adds ordering information beyond a small fixed row-lag cross-basis. Keep increments causal and avoid claims about signed order flow or physical depth.

**Evidence and counterevidence.** Rough-path signatures encode path order, but in a small stream the degree-two term is a structured sum of lag interactions. R2 therefore treats it as a diagnostic refinement, not an independent model family; there is no measured target support.

**Prior overlap.** Continuation of path-signature and i1-to-i0 lag proposals; do not stack it with multiple lag banks.

**No-fit gate.** On existing feature inputs, compare area against the fixed lag basis across sequence halves and count/sum-preserving shuffles. Retain only stable incremental input information.

**Falsifier.** Stop if area is explained by lag counts/sums, reverses across halves, or fails a later exact-score comparison.

**Future control.** CP30: one predeclared area term versus the same model with fixed-lag terms and a no-area control; grade each separately, not additively.

**Compute and CPU path.** Low incremental recurrence for few channels; student can compute it online with sequence-local reset.

**Sources.** [1](https://arxiv.org/abs/1307.7244), [2](https://arxiv.org/abs/2112.13213)

**Training authorized:** no.

## H-RR17-10 — Nonlinear common/contrast model with asymmetric capacity

**Нелинейная модель общей и контрастной компонент с асимметричной ёмкостью** · **Status:** reopened · **Priority:** T2 — Tests a nontrivial constraint after exact rotation is closed; no expected lift.

**Mechanism.** Parameterize p0=c+d and p1=c−d, but make this a real inductive bias: a compact shared common branch plus a separately sized nonlinear contrast branch. Train/evaluate the reconstructed targets with their own official per-target metric weights.

**Evidence and counterevidence.** A linear rotation with inverse reconstruction is an exact no-op. The targets are correlated, but strata are retrospective, the exact target law has 1,253 counterexamples, and weighted Pearson uses separate target measures; correlation alone does not justify tying them.

**Prior overlap.** Reopens the rejected common/contrast coordinate proposal only as an explicit nonlinear capacity constraint; no hard target equality or class gate.

**No-fit gate.** Before fitting, verify parameter ties change the function class, preserve two free outputs, and do not use target-derived strata. Compare parameter/FLOP counts to independent heads.

**Falsifier.** If the asymmetric branch is equivalent to the baseline, or matched cells do not improve the unchanged score, drop it.

**Future control.** CP30: constrained common/contrast versus independent two-head student at matched capacity/compute; reconstruct p0,p1 before official scoring.

**Compute and CPU path.** Medium training, similar recurrent CPU path; a compact shared branch could save parameters only if the contrast residual remains adequate.

**Sources.** [1](https://arxiv.org/html/0912.0902), [2](https://wundernn.io/connectome/docs/data_overview)

**Training authorized:** no.

## H-RR17-11 — Covariance-orientation state versus scalar activity trace

**Состояние ориентации ковариации против скалярного следа активности** · **Status:** continuation · **Priority:** T2 — Low-cost redundancy question; no forecast effect asserted.

**Mechanism.** Maintain a small causal covariance matrix of robust feature-group innovations and separate its orientation from its trace/activity. Feed both only if orientation adds stable state beyond the scalar EW magnitude; reset the matrix each sequence.

**Evidence and counterevidence.** The EDA does not show covariance orientation predicts targets. R2 keeps covariance only as a redundancy diagnostic: trace, eigenvalues, and log determinant summarize the same second-order state and may collapse to activity.

**Prior overlap.** Continuation of continuous covariance-state proposals; distinct comparison is orientation versus scalar activity, not another discrete regime label.

**No-fit gate.** Using input-only sequence halves, compare orientation variability and dependence beyond scalar EW activity. Do not select a retrospective label stratum or tune against exposed validation.

**Falsifier.** Defer if orientation is unstable, redundant with activity, or fails the matched endpoint control.

**Future control.** CP30: same feature groups and width with scalar trace only versus trace-plus-orientation; use identical model and official metric.

**Compute and CPU path.** Low state memory for a few groups; modest matrix updates, with a small student projection if native CPU cost is material.

**Sources.** [1](https://www.ms.u-tokyo.ac.jp/~nakahiro/mypapers_for_personal_use/hayyos03.pdf)

**Training authorized:** no.

## H-RR17-12 — CUSUM slow-state gate without a hard main-state reset

**Медленный шлюз состояния CUSUM без жёсткого сброса основного состояния** · **Status:** continuation · **Priority:** T2 — A cheap stress-test direction, not evidence of real shift or lift.

**Mechanism.** Compute a past-standardized CUSUM over robust feature innovations and let it softly change recurrent write/decay. Keep the main state continuous and predict every row; alarms never freeze outputs or reset the model mid-sequence.

**Evidence and counterevidence.** Change-point methods are principled under stated generative assumptions, but EDA has no verified shift labels and age associations are not target maturity. CUSUM may reduce to event counts or a thresholded EWMA.

**Prior overlap.** Deduplicates CUSUM/FOCuS and BOCPD gate proposals into one fixed low-cost operator; it is not a new market-regime claim.

**No-fit gate.** Predeclare a synthetic feature-shift injection and threshold; measure false alarms on ordinary feature segments and detection delay on copied shifts. Compare to event-age and EW magnitude only.

**Falsifier.** Stop if false alarms track normal event rate, alarms duplicate simpler traces, or paired endpoints do not improve.

**Future control.** CP30: one soft CUSUM-modulated recurrent state versus identical fixed-write recurrence; thresholds are fixed from training inputs only.

**Compute and CPU path.** Very low recurrence cost; no separate detector model or inference-time gradient updates.

**Sources.** [1](https://arxiv.org/abs/2110.08205), [2](https://arxiv.org/abs/0710.3742)

**Training authorized:** no.

## H-RR17-13 — Bounded current-sequence content retrieval

**Ограниченное извлечение содержимого внутри текущей последовательности** · **Status:** new · **Priority:** T3 — Exploratory memory hypothesis with substantial CPU risk; no effect estimate.

**Mechanism.** Keep a small bounded cache of causal input/state summaries keyed by current feature context. Query only prior entries from the same sequence before writing the current row; reset the cache at every boundary. Store no targets, labels, scores, or cross-sequence state.

**Evidence and counterevidence.** This differs from fixed taps by content-addressed recall and from event traces by recalling prior episodes. The EDA rules out adjacent exact-row compression only in its sample; R3 warns that nearest-neighbor lookup can be costly and has no target evidence.

**Prior overlap.** Near dyadic taps, fixed retrieval, TTT fast memory, and suffix motifs; retain only continuous bounded same-sequence retrieval as the distinct mechanism.

**No-fit gate.** On a read-only forward input stream, test recurrence frequency and next-input-summary value versus recency and shuffled-key controls. Bound lookup work; stop if repeats are rare or add no input information.

**Falsifier.** Defer if retrieval is unstable, not better than recency, or exceeds the CPU callback envelope.

**Future control.** CP30: bounded cache student versus recurrence-only and a fixed-tap control at matched state/memory budgets; same sequence resets and official score.

**Compute and CPU path.** Potentially high per-row lookup and memory cost; use a small fixed cache or do not ship. Distill only after a teacher proves value.

**Sources.** [1](https://arxiv.org/abs/2501.00663), [2](https://wundernn.io/connectome/docs/data_overview)

**Training authorized:** no.

## H-RR17-14 — Mamba-3 MIMO versus SISO causal teacher

**Причинный учитель Mamba-3 MIMO против SISO** · **Status:** continuation · **Priority:** T2 — Architecture information value is conditional on a feasible student; no gain forecast.

**Mechanism.** Compare the authors’ multi-input/multi-output recurrent read/write state with a single-input/single-output recurrence at matched state size and compute. Use only past/current numeric rows, exact sequence resets, and stepwise inference parity.

**Evidence and counterevidence.** Mamba-3 reports long-context retrieval and state-tracking on language tasks; its authors provide code. This is not market evidence. Prior GDN had a below-GRU point estimate with an interval crossing zero, and the student may lose latent state advantages.

**Prior overlap.** Continuation of the Mamba-3 teacher proposal; isolate MIMO versus SISO, not a generic SSM sweep or an unqualified GDN retry.

**No-fit gate.** Audit author-code stepwise support, dependency/runtime path, reset parity, and synthetic future perturbation before any data fit.

**Falsifier.** Drop if MIMO is not more expressive than matched SISO on the synthetic test or its compact student fails paired endpoint comparisons.

**Future control.** CP30: matched MIMO/SISO teacher compute, with identical CPU student and distillation; teacher-only quality is not a deployment result.

**Compute and CPU path.** High teacher training and likely accelerator-oriented kernels; route only through a compact recurrent CPU student after proving transfer.

**Sources.** [1](https://arxiv.org/abs/2603.15569), [2](https://github.com/state-spaces/mamba)

**Training authorized:** no.

## H-RR17-15 — xLSTM matrix memory versus scalar memory

**Матричная память xLSTM против скалярной памяти** · **Status:** continuation · **Priority:** T2 — Conditional architecture test; no expected score change implied.

**Mechanism.** Compare xLSTM matrix-memory and scalar-memory recurrence, plus one fixed mixed block, at matched parameter/compute budgets. Require state reset, exact chunk-to-step parity, and a deterministic CPU export path.

**Evidence and counterevidence.** The paper introduces scalar and matrix memory variants with language-model evidence and author code. Local EDA shows no cross-feature memory target benefit; matrix memory overlaps fast-weight/linear-attention ideas and may not distill through point outputs.

**Prior overlap.** Continuation of the xLSTM scout; keep operator comparison distinct from generic width/depth tuning and TTT fast memory.

**No-fit gate.** Use synthetic impulses and paired feature associations to test causal state persistence, reset, and whether matrix memory retains a relation the scalar control loses.

**Falsifier.** Defer if the matrix path adds no synthetic capability, has unsafe streaming parity, or the student cannot transfer it.

**Future control.** CP30: matrix-memory teacher versus scalar-memory teacher with identical student and exposure; include the final CPU student in the comparison.

**Compute and CPU path.** High teacher memory/training cost; student is the likely submission route unless native matrix state meets runtime and ZIP constraints.

**Sources.** [1](https://arxiv.org/abs/2405.04517), [2](https://github.com/NX-AI/xlstm)

**Training authorized:** no.

## H-RR17-16 — Local causal attention plus recurrence versus recurrence alone

**Локальное причинное внимание с рекуррентностью против одной рекуррентности** · **Status:** continuation · **Priority:** T2 — Mechanism-specific hybrid test, not a ranking of expected gains.

**Mechanism.** Add one fixed-size strictly left-causal local attention window to a recurrent state model, testing whether short ordered interactions add value beyond the recurrent summary. Align cached and full-window outputs at every row.

**Evidence and counterevidence.** Griffin/RecurrentGemma motivates local-attention and gated-recurrence hybrids on language tasks. EDA does not show local neighborhoods add information conditional on the recurrent state; the branch can collapse to a short-window feature mixer.

**Prior overlap.** Continuation of the Griffin hybrid proposal and existing causal Transformer/GRU teacher work. One window only; no window sweep.

**No-fit gate.** Run synthetic two-event order tests, future-suffix perturbation, reset/chunk parity, and a conservative callback runtime estimate.

**Falsifier.** Drop if the local branch is not incrementally causal, exceeds CPU budget, or fails matched student evaluation.

**Future control.** CP30: fixed local window plus recurrence versus recurrence-only, with same hidden budget and context; score reconstructed outputs under the official metric.

**Compute and CPU path.** Medium-to-high training and student complexity; ship only a student that retains the local pathway within CPU limits.

**Sources.** [1](https://arxiv.org/abs/2402.19427), [2](https://github.com/google-deepmind/recurrentgemma)

**Training authorized:** no.

## H-RR17-17 — Masked current-feature self-supervision

**Самонаблюдаемое обучение восстановлению замаскированных текущих признаков** · **Status:** continuation · **Priority:** T2 — A controlled representation test; no performance claim from MOMENT.

**Mechanism.** Pretrain the eventual student on allowed training sequences by masking a whole predefined current-feature group before every encoder, normalization, delta, and auxiliary path, then reconstruct that group from remaining current/past inputs.

**Evidence and counterevidence.** MOMENT shows masked time-series pretraining as a method, but its external corpus/weights are ineligible. The sample has no adjacent identical full rows, which neither proves nor disproves useful cross-feature reconstructability.

**Prior overlap.** Continuation of masked-current-row reconstruction; explicitly not future-input reconstruction and not an external pretrained foundation model.

**No-fit gate.** Audit the complete derived-feature graph with synthetic copy sentinels and verify the masked group cannot leak through normalization, deltas, skips, or auxiliary labels.

**Falsifier.** Stop if the objective is a same-row copy task, any unmasked duplicate path remains, or student score does not beat identical no-pretraining control.

**Future control.** CP30: pretrain the deployment-sized student versus the same student without reconstruction; train and evaluate only on permitted sequences, no external corpus.

**Compute and CPU path.** Medium self-supervised compute; discard reconstruction head at inference and retain an ordinary compact CPU student.

**Sources.** [1](https://arxiv.org/abs/2402.03885), [2](https://github.com/moment-timeseries-foundation-model/moment)

**Training authorized:** no.

## H-RR17-18 — CPC contrastive prediction of future latents

**Контрастивное предсказание будущих латентных представлений CPC** · **Status:** new · **Priority:** T2 — Representation hypothesis only; information value is not a gain forecast.

**Mechanism.** Train a causal encoder to predict future feature latents contrastively from the current prefix. Use future inputs only as training targets; draw negatives within the same sequence and match activity/context so sequence identity or easy activity cannot solve the task.

**Evidence and counterevidence.** CPC provides a primary contrastive predictive-coding mechanism, but no evidence transfers to this stream. Future-latent tasks can reward trivial persistence, timestamp, or sequence shortcuts and do not themselves improve the two target forecasts.

**Prior overlap.** Distinct from masked current-row reconstruction and next-event classification; adjacent to future-observation auxiliary losses. No actual future input is available to the serving student.

**No-fit gate.** Audit causal alignment and negative sampling on a synthetic stream; show that future-suffix perturbations do not alter earlier representations and matched negatives defeat sequence-ID shortcuts.

**Falsifier.** Stop if the task is solved by persistence/activity shortcuts or the trained compact student does not beat supervised-only control.

**Future control.** CP30: same compact student and supervised labels with versus without CPC pretraining; preserve same-sequence negatives and remove contrastive head at inference.

**Compute and CPU path.** Medium-to-high training and memory-bank cost; no added serving cost after the encoder/student is frozen.

**Sources.** [1](https://arxiv.org/abs/1807.03748)

**Training authorized:** no.

## H-RR17-19 — Randomized compensated gradient horizons (ARTBP)

**Случайные компенсированные горизонты градиента ARTBP** · **Status:** reopened · **Priority:** T3 — Expensive optimization question with weak local support; no effect claim.

**Mechanism.** Use ARTBP-style random truncation with compensation to estimate recurrent gradients across variable horizons, while leaving the causal forward recurrence and inference state unchanged. Compare with fixed truncation; do not label ordinary mixed chunk lengths as ARTBP.

**Evidence and counterevidence.** The primary paper proposes an unbiased estimator, but compensated variance and compute are concerns. Prior fixed detach-horizon screens were small/inconclusive and accumulated-interval memory failed its instantaneous control.

**Prior overlap.** Reopens gradient-horizon research with a mathematically distinct compensated estimator; not a generic detach sweep or extra forward memory.

**No-fit gate.** Verify estimator mean against exact gradients on a tiny recurrence and quantify variance across synthetic streams. A scale/sign error or excessive variance closes it.

**Falsifier.** Defer if unbiasedness fails, variance consumes the update budget, or matched student endpoints do not improve.

**Future control.** CP30: ARTBP versus one fixed-horizon control at equal forward rows and optimizer updates; if used for a teacher, grade the same compact student route.

**Compute and CPU path.** High training overhead from compensation/variance; no inference overhead if the forward model is unchanged.

**Sources.** [1](https://arxiv.org/abs/1705.08209)

**Training authorized:** no.

## H-RR17-20 — Cross-fitted scalar calibration under the weighted metric

**Кросс-валидированная скалярная калибровка под взвешенную метрику** · **Status:** new · **Priority:** T2 — High information per compute, but no forecast of numerical lift.

**Mechanism.** For each target and fixed bins of an honest out-of-fold student score s, estimate the metric-weighted conditional mean E[M·|Y_c|·Y_c | s] / E[M·|Y_c| | s], where Y_c=clip(Y,−2,2) and M is the official scoring mask. Bin-level EDA is allowed now; fitting a calibration map is future-only on training folds, never grade components.

**Evidence and counterevidence.** The weighted-mean geometry is exact for the official score, and positive affine transforms do nothing before clipping. Histogram MSE/CE were negative; this proposal is a low-dimensional scalar post-map, not another codebook head, and has no measured gain.

**Prior overlap.** New narrow candidate from R3; overlaps histogram calibration enough that the exact old arm and held-out component protocol must be checked first.

**No-fit gate.** Inspect existing frozen-checkpoint outputs if available; bounded inference of a frozen checkpoint is allowed under the contract if needed. Plot only predeclared weighted bin means, support, and component stability; do not fit a calibration map now or tune on grade components.

**Falsifier.** Defer if the curve is affine, sparse, unstable, or no separately graded components improve under the later authorized map.

**Future control.** CP30: identity readout versus per-target cross-fitted map, with map fit only on training folds and score measured on disjoint exposed components.

**Compute and CPU path.** Low offline CPU cost; two tiny tables run statelessly after a CPU student. Calibrate the student’s own outputs, not a mismatched teacher.

**Sources.** [1](https://arxiv.org/html/0912.0902), [2](https://wundernn.io/connectome/docs/data_overview)

**Training authorized:** no.

## H-RR17-21 — Bounded group-DRO training objective with pooled WP primary

**Ограниченный group-DRO в обучении при основном pooled WP** · **Status:** reopened · **Priority:** T2 — Robustness question with no demonstrated shift; not an expected-score claim.

**Mechanism.** Test a bounded train-only group-robust auxiliary objective using predeclared groups that may be label-defined for loss bookkeeping. The predictor receives no group, class, sequence ID, or validation-derived boundary; official pooled weighted Pearson remains the primary selection metric.

**Evidence and counterevidence.** Group-DRO is a known robust-risk method, but EDA class strata are retrospective and deliberately equal-sampled; they cannot define hidden-test mixture or inference gates. Reweighting can optimize a different estimand from pooled WP.

**Prior overlap.** Reopens group robustness only as training loss bookkeeping; not a new class-conditional serving model or an assumption of population shift.

**No-fit gate.** On training rows only, count group support and target-weight mass; toy-check how the robust objective can alter pooled weighting. Never use validation labels to choose groups or weights.

**Falsifier.** Stop if groups are sparse, objective overwhelms official mass without a rationale, or exact pooled WP fails matched comparison.

**Future control.** CP30: standard pooled objective versus one bounded group-robust objective with identical model, rows, and updates; report official WP first and group scores only as diagnostics.

**Compute and CPU path.** Low-to-medium training overhead; no inference or CPU cost because groups are not inputs.

**Sources.** [1](https://arxiv.org/abs/2212.13669), [2](https://wundernn.io/connectome/docs/data_overview)

**Training authorized:** no.

## H-RR17-22 — Within-side component-equivariant encoder with identity escape

**Эквивариантный энкодер компонентов внутри стороны с тождественным обходом** · **Status:** new · **Priority:** T3 — Novelty is structural, while label benefit is unknown; no effect estimate.

**Mechanism.** Apply shared permutation-equivariant transforms separately to documented bid-side and ask-side component sets, while retaining stable component IDs through an identity-preserving residual path. Do not assume depth order or permute across sides.

**Evidence and counterevidence.** The data overview defines indices 0–10 versus 11–21 as bid/ask sides but says index order is not physical depth. Input symmetry could reduce parameters; it does not establish label invariance. Existing grouped-input/graph ideas are close prior work.

**Prior overlap.** Distinct refinement of anonymous grouped encoders: a tested within-side equivariance constraint plus identity escape, not HLOB message passing or full permutation invariance.

**No-fit gate.** On permitted training inputs only, compare predeclared within-side permutation statistics across sequence halves. Statistical symmetry is only a necessary input check, never target evidence.

**Falsifier.** Do not fit if input symmetry fails; after fitting, reject if the constrained student loses to matched ID-specific encoding.

**Future control.** CP30: within-side equivariant-plus-ID path versus equal-width ID-specific encoder, same features and official output scorer.

**Compute and CPU path.** Medium architecture cost; parameter sharing may reduce model size, but measure actual CPU latency and preserve feature identity.

**Sources.** [1](https://arxiv.org/abs/1703.06114), [2](https://wundernn.io/connectome/docs/data_overview)

**Training authorized:** no.

## H-RR17-23 — Latent-code transition topology versus occupancy and one-step transitions

**Топология переходов латентных кодов против частот и одношаговых переходов** · **Status:** new · **Priority:** T3 — Low-priority mechanism screen given direct code-probe reversal; no gain implied.

**Mechanism.** From permitted training inputs, define a small fixed feature codebook and maintain per-sequence transition/cycle summaries. Test whether returns and short cycles add information beyond code occupancy and one-step counts; all state resets per sequence and contains no labels.

**Evidence and counterevidence.** R3 identifies temporal topology as distinct from a generic graph, but the local volume-code probe was +0.0062 in-sample and −0.0078 on chain-half cross-fit. HLOB/event-stream literature does not validate this codebook or its target value.

**Prior overlap.** Near volume-code prediction, event-transition heads, HLOB graphs, and path signatures; keep only higher-order cycle topology beyond occupancy/one-step dynamics.

**No-fit gate.** With predeclared deterministic coarse bins, compare cycle/return stability across sequence halves and transition-order shuffles that preserve code frequencies.

**Falsifier.** Close if topology is unstable or adds no descriptive distinction beyond occupancy and first-order transitions.

**Future control.** CP30: topology-state student versus occupancy-plus-one-step control and base recurrence; same codebook procedure, exact official WP, no additive blending.

**Compute and CPU path.** Medium-to-high state/feature cost; only consider a compact codebook and bounded counters for CPU serving.

**Sources.** [1](https://arxiv.org/abs/2405.18938), [2](https://arxiv.org/abs/2412.09631)

**Training authorized:** no.

## H-RR17-24 — Test-time-training fast memory as a training-only teacher

**Быстрая память test-time training как учитель только для обучения** · **Status:** reopened · **Priority:** T3 — Speculative memory transfer with substantial compute; no quality prior.

**Mechanism.** A future teacher may adapt a small fast-weight memory using a causal self-supervised feature loss within each training sequence, reset per sequence, then transfer its state/output behavior to a fixed student. The eventual student uses no online gradients; none are run now.

**Evidence and counterevidence.** TTT/Titans show expressive test-time memory mechanisms on language tasks, not this forecast task. Prior inference-time adaptation is costly, student transfer is uncertain, and the current contract forbids training now.

**Prior overlap.** Reopens fast-memory ideas only as an offline teacher path; distinct from deterministic retrieval and from training a deployed online optimizer.

**No-fit gate.** On synthetic feature sequences, verify update order, no future access, deterministic reset, and worst-case update cost. No market model or target is touched.

**Falsifier.** Stop if causal self-supervised updates are unstable, too costly to produce teacher states, or their fixed student cannot beat output-only distillation.

**Future control.** CP30: direct student versus student distilled from the training-only fast-memory teacher; evaluate only the frozen no-gradient student.

**Compute and CPU path.** Very high teacher-generation/training cost; final CPU path is a fixed recurrent student with no per-row optimizer state.

**Sources.** [1](https://arxiv.org/abs/2407.04620), [2](https://github.com/test-time-training/ttt-lm-jax), [3](https://arxiv.org/abs/2501.00663)

**Training authorized:** no.

## H-RR17-25 — Causal latent world-model rollouts as a teacher

**Причинные rollout’ы латентной world-model как учитель** · **Status:** new · **Priority:** T3 — Cross-domain speculative path; highest uncertainty, no gain forecast.

**Mechanism.** Train a latent sequence model only on permitted training inputs, then roll forward using its own predicted feature latents. Use those imagined dynamics as a representation teacher for a causal student; no actions, reward, RL framing, external market corpus, or actual future inputs inside the rollout.

**Evidence and counterevidence.** DreamerV3 reports general model-based RL results (Nature 2025), but those rely on action-conditioned environments and do not establish value here. Own imagined feature paths can compound error and may teach the student model bias.

**Prior overlap.** New transfer hypothesis, adjacent to CPC, TTT, and causal teacher distillation; synthetic rollouts are model predictions, not market observations or oracle futures.

**No-fit gate.** On input-only held segments, compare multi-step latent rollout calibration with persistence and one-step baselines; require stable errors and future-suffix invariance before any student training.

**Falsifier.** Defer if rollouts diverge rapidly, add no information beyond one-step prediction, or student endpoint fails the matched control.

**Future control.** CP30: causal student with versus without rollout-derived latent hints, identical labels/context and official WP; never count teacher-only rollout quality as a result.

**Compute and CPU path.** Very high world-model and rollout cost; retain only a compact distilled CPU student, with no rollout loop in the final callback.

**Sources.** [1](https://www.nature.com/articles/s41586-025-08744-2)

**Training authorized:** no.
