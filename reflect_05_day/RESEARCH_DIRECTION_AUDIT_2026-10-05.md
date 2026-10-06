# Research priorities after the October 5 review

The campaign has useful ideas and a real public improvement. Its search has stalled because many inexpensive prototype comparisons never reach the complete predictor, while narrow failures have sometimes closed much broader questions. The next window should test one simple transfer candidate and one changed temporal representation against the current GRU256. A third branch should test whether an original-target privileged teacher can improve supervision without the assumptions behind stopped bootstrap.

The immediate candidate is the fixed GELU/LeakyReLU readout. The strongest structural hypothesis is a persistent event state with a separate quiet-row belief-update path. Neither currently supports expected public 0.7000.

This is a source and saved-result audit, with two separately tasked reviewers of mechanisms and training. They shared the same model family; their agreement is corroborating inspection, not independent empirical replication. I inspected the named handoffs, selected historical negatives, raw JSON results, incumbent provenance and implementations; I did not read all 7,176 indexed files. No new fits, prediction-array scans, numerical EDA, provider requests, timing trials or uploads were performed. Historical reports and existing untracked work remain unchanged.

## The evidence anchor

The fixed seed-4 complete combo has public **0.6660**, runtime **37m07s**, memory **626 MB**. Seed 3 scored **0.6655**. Its parent alone scored **0.6641**, its final head **0.6654**, and its complete parent/head mixture **0.6655**. Four complete seed-3 CV predictors gained **0.002414379**, with a conditional component-bootstrap interval **[0.001565628, 0.003448539]**. These observations establish useful transfer; they do not isolate Muon, chunk size and bootstrap individually. [Public evidence](campaign_2026-10-04/PUBLIC_RESULTS_REFLECTION_20261004_LATE.md), [complete CV result](campaign_2026-10-04/checks/combo_merged_all_folds_report.json).

The actual seed-4 parent is a two-layer GRU256, Muon, chunk1000, four epochs, alpha 0.5 distillation from two historical since teachers, bootstrap horizon 8, EMA consistency 1.0, structural/next auxiliaries 0.3 each, propensity power 2 plus floor 0.03, and since weighting 32. The final head freezes the representation, uses one AdamW/global-correlation epoch on the historical validation fitting pool, and is equally merged with the parent. The head uses FT seed 3 for both parent seeds. [Parent provenance](../artifacts/runs/c20261004_combo_s4_final/provenance.json), [head provenance](../artifacts/runs/c20261004_combo_head_s4_final/provenance.json).

The metric clips both targets and predictions to ±2, pools selected rows separately for each target under its absolute clipped-target weights, and averages the two correlations. All serious comparisons need this complete endpoint. Class correlations and residual-energy shares cannot be added to obtain its delta. Public and eventual private performance remain tier-C external-validity questions; paired local CV supplies conditional empirical evidence, with disclosed teacher and selection limitations. [Metric implementation](../src/wnn/gmetric.py), [validation limitations](campaign_2026-10-04/VALIDATION_SCORECARD.md).

## Findings that change the next decision

### 1. An overlooked-model claim fails its fitting-provenance check

The existing [independent draft](../reflect_05_day/INDEPENDENT_ELEPHANTS_2026-10-05.md) reports `ab85_final_ema6_s4_gclpall` at 0.70404 on the 368-sequence panel and proposes it as an overlooked clean winner. The score is consistent with its saved history, but the claimed evaluation status is wrong.

Both its head and full fine-tune logs explicitly record `train_source="valid"`, `include_heldout_valid=true`, and `valid_train_fraction=1.0`. [The launch script](../artifacts/logs/q56.sh) calls `ft_gclp.sh ... finalall`; [the fine-tuning script](../scripts/ft_gclp.sh) explicitly says that `finalall` includes the holdout and its score is in-sample. [Final fit log](../artifacts/logs/ab85_final_ema6_s4_gclpall.log), [head fit log](../artifacts/logs/ab85_final_ema6_s4_gclpallh.log), [saved history](../artifacts/runs/ab85_final_ema6_s4_gclpall/history.json).

This is direct supervised fitting of the evaluation rows, beyond the selection and chain-overlap caveats acknowledged in that draft. The proposed blends inherit it. A correctly recomputed correlation cannot repair that comparison. The model may still generalize, but the 0.70404 result gives no evidence that it was unfairly rejected. It should not become the first deployment priority without a comparably excluded lineage or different external evidence. Do not rerun its holdout scan to make the same comparison more precise.

### 2. The supposed public class mix is an inferred scenario

`PUB143=[0.183,0.160,0.198,0.459]` is useful for sensitivity analysis. It is not an observed test composition. The [original class-mix artifact](campaign_2026-10-01/data_eval/class_mix.json) derives the class-3 share by making a local reference match public 0.6622 within a one-parameter composition family. Its own note says one equation identifies only that family; its interval is approximately **0.41–0.51**, conditional on the assumptions. [October 1 explanation](campaign_2026-10-01/REPORT.md).

Class 3 deserves attention because it is weak locally and has substantial weighted error. However, neither “public has exactly 45.9% class 3” nor “the 0.036 local/public gap is a removable distribution-shift loss” is established. Composition, period difficulty, masks, fitting exposure and predictor lineage are confounded. Keep pooled and target scores primary; report several plausible class-mix scenarios, including PUB143, without choosing a candidate because of one fitted mix. Public feedback measures an external sample, but becomes selection-exposed when repeatedly used and is not an unbiased estimate of eventual private performance. [Rules](../docs/rules.md).

### 3. The diagnostic clock and production clock differ

The strong target probe defines an event when **either** `i0_p0` or `i0_p11` changes. Production `ev_i0px` requires **both** to change. Consequently, the impressive **3,761/3,763** and **3,841/3,842** scored-transition alignment counts concern the OR clock. They do not establish an equivalent law for the AND clock. [Probe source](campaign_2026-10-05/target_mechanics/strong_residual_probe.py), [production features](../src/wnn/features.py).

This is a measurement distinction, not a demonstrated prediction bug: raw inputs and both price-difference features already expose the individual changes. The incumbent also already receives event flags, age and last-event deltas. “Add event features” therefore repeats an existing input representation. The open question is which event semantics should govern learned state updates and memory. An event is nearly necessary for a target change on these panels; it is not sufficient, so a hard reset at every event is also unjustified.

### 4. An empirical bootstrap success was promoted into an unjustified theorem

The older [DTW/objective report](campaign_2026-10-02/dtw/objective/REPORT.md) calls stopped bootstrap unbiased, reports bias +0.0001, and concludes that hold-and-update, hazard/two-timescale states have no room and cross-fitted teachers are unnecessary. Its “population bias” is actually a cross-fitted per-class linear projection on **[1, parent prediction, teacher prediction]**. That narrow diagnostic cannot establish full-history conditional unbiasedness. [Analysis implementation](campaign_2026-10-02/dtw/objective/analyze_targets.py).

For surviving label pairs, the actual alpha-0.5 student target becomes **0.5·teacher(t) + 0.5·teacher(t+8)**; otherwise it includes the realized origin label. Origin-row absolute-label weights remain. The future teacher predicts its own row's target, and the survival decision uses future changes in the target pair. A martingale/optional-stopping argument needs calibration for the fixed origin outcome under the origin weighting, plus the relevant measurability conditions. Those conditions were not established by the small projection. The current trainer correctly calls this an empirical denoising hypothesis, with no unbiasedness guarantee. [Target construction](../src/wnn/training_controls.py), [trainer](../scripts/train_rnn.py).

Preserve the successful recipe. Reopen original-target teacher and temporal-state alternatives because the theoretical veto is too broad, not because the public improvement is suspect. There is no literal ceiling at the old teachers' public score: future-conditioned supervision, original labels on other rows and auxiliary learning can let a student exceed them.

### 5. The recent architecture results answer narrower questions than their names suggest

E464 completed the missing final heads and is valid scientific evidence. None of its candidates has a higher point estimate than the complete combo on the exposed panel; most corresponding intervals include zero. Within matched smaller-model families, the effects remain uncertain: specialist versus GRU **−0.000461** with CI **[−0.001956,+0.001507]**; conditional adapter **+0.000250**, CI **[−0.000947,+0.001428]**; diagonal adapter **+0.000786**, CI **[−0.000142,+0.001792]**. [Raw complete results](campaign_2026-10-04/staging/e464/results/e464_report_all_fp32_v1.json).

The specialist's route freezes after the first **99 rows**, and all expert GRUs update on every row. This tests sequence-prefix specialization, not dynamic event/quiet allocation or reduced recurrent compute. Budget controls are GRU96/128, whereas the public anchor is GRU256. Architecture initialization also changes the sampled normalization statistics; the preregistration acknowledges this. [Specialist source](../src/wnn/specialist.py), [parent controls](campaign_2026-10-04/E424_H093_H094_PREREG.md), [head protocol](campaign_2026-10-04/E464_COMPLETE_ARCHITECTURE_PREREG.md).

Do not rerun these exact routes under a new name. A materially different state-update mechanism remains open. For a new matched experiment, fix normalization membership and data-order RNG independently of model initialization. The trainer already offers a seeded batch-order path; its ordinary normalization uses two fitting batches sampled every seventh row. This improves attribution without inventing another research programme. [Trainer](../scripts/train_rnn.py).

### 6. The strongest prototype leads need transfer, not another sweep

The fixed 50/50 GELU–LeakyReLU readout gains **+0.008930/+0.004308/+0.008690** at epoch 3 across three seeds. Two conditional intervals exclude zero. Recurrent context gating adds only tiny effects over a static gate, with all three intervals crossing zero. This supports a bounded readout transfer experiment, not dynamic activation orchestration or periodic-market explanations. [E488 raw contrasts and report](campaign_2026-10-05/context_mixture_duration/REPORT.md).

Learned diagonal decay also repeatedly beats fixed decay (**+0.003630/+0.003816/+0.003043**), but those are small width-32 prototypes with scores far below the incumbent. Use learned memory as a control or small residual branch before replacing the entire GRU. [Three-seed sequence results](campaign_2026-10-05/sequence_components/THREE_SEED_REPORT.md).

The 224 completed activation trajectories do not represent 224 incumbent comparisons, and fit count is not GPU-hour cost. Their work found controls and some reproducible leads. The mistake would be to keep growing that same exposed-panel catalogue instead of testing transfer. [Retrospective](campaign_2026-10-05/ACTIVATION_RETROSPECTIVE_2026-10-05.md).

## Which old rejections deserve reopening

| Idea | Actual evidence | Appropriate disposition |
|---|---|---|
| Joint distribution heads | E468 joint tilted −0.000618; regression anchor −0.000555. Frozen historical GRU128 states, 32 joint/16 marginal atoms, one pass; the regression arm also receives CE. | Weak family negative. First repair the fitting comparison; posterior complexity comes later. |
| Target-specific histories | Older econometric clock contrasts move targets in opposite directions; E433 remains planned. E415 tested proxies; E426 altered teacher EMA timing. | Input assimilation remains open. Test both/swapped histories and age/hazard controls; do not hard-code a universal target-to-clock assignment. |
| Current-target direction auxiliary | E244 single +0.000656; replacement pair +0.000468 ±0.000712, below the +0.0005 stopping gate. | An uncertain positive, not a proven failure of direction supervision. The 0.000032 cutoff shortfall is a stopping decision. Revisit only within a stronger relational/teacher hypothesis. |
| Run-relative volume input | E413 +0.000962, CI [−0.000080,+0.002060], one exposed fold/seed. | Tested, promising but unresolved. Replication can accompany a predeclared temporal branch; no generic feature grid. |
| Original-target future teacher | A future quote-path auxiliary lost in E251; the incumbent's future prediction substitution is a different treatment. | Origin-preserving privileged supervision remains open; require a causal-teacher control and component cross-fitting. |
| Distilled loss normalizer/cadence | Actual update mass varies; E430 accumulation has drafts and arithmetic checks, no fitted predictive contrast. Cached distillation bypasses the old running-normalizer path. | One bounded identification experiment is justified. A synthetic identity is not a score gain. |
| Width384 | One old fold signal; W320 near-null; W384 replication stopped on host-RAM guards. | Width384 is unresolved, not disproved. Repair actual resident memory and establish CPU feasibility before admitting another pair. |
| Equal-target loss weighting | E204 scoring mass 0.497078/0.502922; loss mass 0.503398/0.496602; earlier metric-Jacobian coefficients differed 1–8%. | Keep low priority. The common denominator alone does not show harmful task imbalance. |

Sources: [ledger](experiments.jsonl), [E468 results](campaign_2026-10-05/joint_target_model/RESULT.json), [E468 loss source](campaign_2026-10-05/joint_target_model/model.py), [original inventory](campaign_2026-10-04/EXPERIMENT_INVENTORY.md), [E413 raw report](campaign_2026-10-04/checks/e413_paired_report.json), [target-mass diagnostic context](game_changers.md).

## Ranked experiment directions

These are proposals for a future admitted window, not launched work. Rank reflects expected information and feasible transfer, not a forecast that their local deltas will add to 0.7000.

### Priority 1A: finish the simple readout transfer question

**Change:** the fixed mixture at the readout only on the actual GRU256 recipe. Keep embedding and recurrent activations unchanged. This transfers the surviving E488 treatment rather than combining several attractive knobs.

**Smallest decisive test:** candidate/control at two fixed training seeds, with the same data order, fitting-only normalization, teacher lineage, schedule and final-head/merge procedure. Save parent and complete results. Measure candidate native cost before expansion. A cheap frozen-function swap is a diagnostic, not the trained candidate comparison.

**Falsifier:** benefit disappears at the complete endpoint, fails seed confirmation, or loses deployment headroom. The H64 magnitude supplies no expected-public gain estimate. This is the short route to another useful predictor, with limited evidence that it changes the long-term ceiling.

### Priority 1B: separate event state from quiet-row belief updates

**Change:** a small causal residual state around the strong predictor, with separately learned event writes and quiet-row innovations. For example, update a branch with `event * event_update + (1-event) * quiet_update`, and zero-initialize its output correction. Events choose an update mechanism; they do not assert that the target changed or erase all memory.

**Why:** hard output holding loses **0.081486/0.076225 WP** on fitted/row-excluded panels. Class-3 quiet-row output updates reduce weighted error in both. This is direct evidence that useful information arrives between target transitions. It motivates testing an explicit state allocation, rather than holding an already trained model's output. The row-excluded panel still links to fitted components. [Strong diagnostics](campaign_2026-10-05/target_mechanics/strong_baseline/REPORT.md).

**Smallest decisive test:** preserved strong baseline; equal-capacity unrestricted recurrent residual; event/quiet residual with the same inputs; causal stale-event control. Audit OR versus AND semantics explicitly, and include age/activity in the controls so a duration effect cannot masquerade as event alignment. Start on strong causal states; only survivors warrant end-to-end changes. Streaming must process every row, warm-up and reset correctly. Frozen-state failure does not close a hypothesis that requires a changed backbone.

**Falsifier:** alignment adds no incremental gain over the unrestricted/duration controls, benefit occurs only on fitted groups, or confirmation reverses. An additive diagonal memory is a useful cheap control; content-addressed memory should be added only when direct retrieval provides extra information.

### Priority 2: train the privileged teacher on the origin outcome

**Change:** a training-only teacher reads through `t+8` but predicts **Y_t**, using the **origin row's** clipped label, weight and mask treatment. Compare it with an otherwise matched causal teacher. A causal student learns the cross-fitted supervision; deploy no future access and no label-derived survival gate.

**Smallest decisive test:** first establish held-component teacher benefit for the original weighted task. Then one fixed student contrast against the same causal-teacher student, followed by the historical final head. Both teachers and all learned ancestors must exclude evaluation components; use inner cross-fitting for student targets. Teacher advantage alone is insufficient: future information may be impossible for a causal student to exploit.

**Falsifier:** no origin-target teacher benefit, no student transfer, component leakage, or changed target origin. Keep the current stopped-bootstrap predictor as the empirical reference. This directly tests the assumption that smoothing through an old future-origin teacher is the best way to reduce repeated-outcome noise.

### Priority 3: target-specific history assimilation inside the temporal branch

**Change:** give each target a small learned adapter over row-clock and event-clock causal histories, preserving the common strong state and quiet updates. Prefer this bounded extension to two full recurrent backbones.

**Smallest decisive test:** shared access to both clocks, asymmetric adapters, swapped assignments, and matched hazard/age histories. Earlier event-versus-row effects were **−0.007712 for t0 / +0.006624 for t1** in a weak econometric model; this motivates asymmetry, not a fixed assignment. “t1 is harder” is not a reliable premise: current fitted strong-model WP is higher for t1. [Clock results](campaign_2026-10-03/eda/econometrics/cache/continuation_results.json), [hazard controls](campaign_2026-10-03/eda/econometrics/cache/continuation_hazard_results.json).

**Falsifier:** no advantage over shared history access and equal-capacity controls through the complete endpoint. Sequence composition and smoothing half-life must not explain the result.

### Priority 4: identify cadence and the actual loss-normalization effect

**Change:** execute the prepared E430 distinction between update1000, pooled accumulation2000/detach1000, and connected2000. Match nominal EMA exposure and total data. Separately compare historical distilled normalization with a fitting-pool fixed denominator and its shuffled-multiplier sham.

**Why:** changing chunk size changes optimizer cadence, gradient credit and state freshness together. The actual distilled branch divides each update by its own label-weight mass; changing `--loss-norm running` would not treat it. The saved normalizer analysis found a **5.11%** shift in a teacher-gradient proxy, with cosine **0.99876**, not a demonstrated full-GRU improvement. [Objective analysis](campaign_2026-10-03/continuation/round2/objective/REPORT.md), [E430 protocol](campaign_2026-10-04/E430_optimizer_cadence/CRITIQUE_AND_PREREG.md).

**Falsifier:** no complete-predictor difference after matching treatment and clocks. Finish this bounded test rather than spending another round proving estimator identities. Actual main/auxiliary/EMA gradient alignment can be a limited diagnostic; it does not justify a new task-weight sweep without measured interference.

### Priority 5: deployment-enabled capacity and diversity

**Change:** repair the W384 resident-memory path and benchmark the Python-3.11-compatible INT8 engine as enabling infrastructure. A wider single trunk or genuinely diverse compatible member could change the achievable predictor under the one-CPU limit.

**Smallest decisive test:** one real production-shape loader allocation and one candidate runtime benchmark before a replicated width comparison. Two T4s share host RAM. For ensembles, first require comparable excluded lineage and measure incremental score at fixed weights; decorrelation by itself is insufficient. The incumbent's 37-minute runtime leaves about 23 minutes of hard budget, less after a safety margin. A second GRU256 cannot be assumed feasible; an older smaller member may be, but needs measurement.

**Falsifier:** actual memory/runtime makes the proposed shape infeasible, accuracy is lost, or matched replicated score is null. An ABI or RAM failure censors the scientific question. Accuracy-preserving quantization alone does not constitute a new best. [Submission/INT8 audit](campaign_2026-10-05/NEW_SUBMISSION_AUDIT_2026-10-05.md), [width incident evidence](campaign_2026-10-05/review_20261005/EXECUTION_AND_EVIDENCE_AUDIT.md).

## Literature changes mechanisms, not this project's evidence

The papers below support specific distinctions. Their benchmark results do not supply Connectome gains, valid anonymous-price decoders or CPU runtime guarantees.

- [Phased LSTM, NeurIPS 2016](https://proceedings.neurips.cc/paper/2016/hash/5bce843dd76db8c939d5323dd3e54ec9-Abstract.html) shows gated recurrent updates for asynchronous/event streams. The inference here is to test observable-event state allocation with a quiet update path; its oscillatory gate is not evidence of a periodic market signal.
- [Karlsson et al., AISTATS 2022](https://proceedings.mlr.press/v151/k-a-karlsson22a.html) studies training-only intermediate time-series information and proves sample-efficiency results under a fixed-horizon Gaussian-linear system. It motivates the origin-preserving teacher question, with no guarantee for these clipped, selected market labels.
- [Gated Delta Networks, ICLR 2025](https://arxiv.org/abs/2412.06464) separates adaptive erasure from targeted error-driven memory writes. A small event-addressed branch is distinct from the tested additive positive-kernel memory. Use it only if cheaper memory controls leave recoverable information.
- [LRU, ICML 2023](https://proceedings.mlr.press/v202/orvieto23a.html) isolates stable diagonal recurrence and training ingredients. It motivates controlled memory experiments rather than a wholesale “try Mamba” replacement.
- [MixStyle, ICLR 2021](https://arxiv.org/abs/2104.02008) provides a style-statistic augmentation mechanism. The old shift scout already proposed this family. Keep it conditional on evidence of a nuisance coordinate; arbitrary feature mixing could break code equality and valid label relationships. The local/public level gap alone is insufficient motivation.

## How to use the next window

Start two scientific questions: the readout transfer and the event/quiet representation. Keep the original-target teacher as the next structural branch rather than launching the entire deferred portfolio. Reserve confirmation compute before selecting a winner. Carry every admitted arm through its declared final head and fixed merge, even if its parent screen is less flattering.

A branch gets one simple control that can fit competitively, one mechanism-specific control, a fixed complete endpoint, and a predeclared confirmation. Report paired per-seed effects and conditional component intervals; do not count snapshots as independent fits or turn pooled small-panel effects into a narrow public forecast. Use lineage-aware component exclusions available in the repository, while acknowledging historical selection exposure. Archived replay should reproduce its declared reference within numerical tolerance; fresh fits require seed-aware comparisons. A degraded fitting control cannot close a broad mechanism family.

Keep the public milestone **0.7000** as the user's aspiration, with **0.6660** the measured anchor. Do not replace that goal without the user, but do not organize every small experiment around proving it achievable. First seek a reproducible complete-predictor gain and evidence about what limits it. A larger jump needs a changed information or estimation mechanism; the report currently establishes hypotheses, not that jump.

Current admissions remain closed under the recorded October 5 cutoff. This review does not reopen queues or allocate additional budget. The useful next output is a small set of discriminating complete-model comparisons in a newly admitted window, rather than another catalogue of ideas or another layer of instruments.
