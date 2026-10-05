# Elephants review — October 5, after public 0.6660

**The campaign has produced a real public improvement and useful mechanistic leads. Its current bottleneck is the missing transfer stage between cheap prototype experiments and the complete submitted predictor.** It also repeatedly mistakes a failed implementation, a degraded fitting control, or a narrow negative for evidence against a whole method.

This is a targeted, read-only audit, not a claim to have read every indexed file. Root inspected public/validation records, complete-predictor CV, target mechanics, joint-head source and saved component reports; two independent Luna critics separately audited mechanisms and execution. Their reports are [mechanism audit](campaign_2026-10-05/review_20261005/MECHANISM_AUDIT.md) and [execution/evidence audit](campaign_2026-10-05/review_20261005/EXECUTION_AND_EVIDENCE_AUDIT.md). Primary literature was checked for the proposed methodological distinctions. No new scientific fits, EDA runs or uploads were started. This supplements the [current review guide](SECOND_PAIR_OF_EYES_2026-10-05.md); historical reviews remain unchanged.

## What changed since the previous reviews

The strongest public predictor is now **0.6660**, complete combo seed 4, compared with 0.6655 for seed 3 and 0.6641 for the previous distilled incumbent. The old recommendation to finish and upload the combo has been executed. Seed-3 parent-only scored 0.6641, head-only 0.6654, complete parent/head 0.6655: the final procedure is central to this public result. The four complete-recipe CV models also finished: pooled gain **+0.002414379**, conditional component interval **[+0.001565628,+0.003448539]**. These are useful positive results, not an untouched test or a guarantee for every training seed. [Public reflection](campaign_2026-10-04/PUBLIC_RESULTS_REFLECTION_20261004_LATE.md), [E410](campaign_2026-10-04/E410_complete_predictor.md).

The new honest expected-public goal is **0.7000**. No model, transfer calibration or completed submission currently supports that expectation. A large weak-model gain or the incumbent's 0.70077 score on partly fitted supplied validation is not evidence that the public target is close.

## The elephants

### 1. Many experiments, little comparison with the actual champion

The activation campaign has **224 completed, independently graded fit trajectories**. Most are short small-model runs on the same 24 fitting and 16 historically exposed evaluation components. E488 extends duration to three epochs, but remains H64 on that panel. Its 24 saved snapshots belong to 12 trajectories, not 24 independent fits. E467/E469's separate 30 sequence-component fits are width-32 prototypes with absolute WP around 0.44–0.55. A repeatable improvement there can be an optimization or capacity effect that the full GRU256 already resolves.

This work produced controls and narrowed mechanisms. It has not produced 224 chances to beat the complete incumbent. The next stage should test a few leads with the incumbent's data mixture, teacher targets, optimizer cadence and final-head procedure. More minor variants on the same 16 components will add less knowledge about the requested milestone. [Activation retrospective](campaign_2026-10-05/ACTIVATION_RETROSPECTIVE_2026-10-05.md), [sequence synthesis](campaign_2026-10-05/sequence_components/THREE_SEED_REPORT.md), [wrap](campaign_2026-10-05/RESEARCH_WRAP_2026-10-05.md).

### 2. The full supplied validation score is a misleading promotion anchor

Previous D256 scores **0.700792587** over all supplied validation sequences, slightly above combo seed 4's **0.700771292**, yet loses publicly **0.6641 versus 0.6660**. Of the 1,873 supplied sequences, 1,505 entered final fitting. The excluded 368 are historically exposed; component relationships and teacher lineage also matter. The stronger target-mechanics replication explicitly finds training-component links even on its directly row-excluded panel.

There is no hidden fresh holdout available merely by renaming a row subset. We can still use component-excluded paired comparisons, new training seeds and lineage-aware cross-fitting, but must describe their limits. Public changes are external measurements; selecting repeatedly against them eventually also creates selection exposure. A two-point public gain cannot calibrate a narrow forecast to 0.7000. [Validation scorecard](campaign_2026-10-04/VALIDATION_SCORECARD.md), [strong diagnostics](campaign_2026-10-05/target_mechanics/strong_baseline/REPORT.md).

### 3. A repeated future label does not imply a repeated optimal prediction

Nearly every scored target jump coincides with a current observed `i0_p0`/`i0_p11` event: **3,761/3,763** on the fitted panel and **3,841/3,842** on the row-excluded panel. This is a strong structural clue, with rare exceptions. It does not identify a physical instrument, exact horizon or deterministic future outcome.

Hard holding the previous prediction between these events loses **0.081486** and **0.076225 WP** on those respective panels. The underlying recurrent state still updates in this experiment. Quiet-run observations can change a conditional belief about an unchanged future label; the strong model's class-3 quiet-run prediction updates reduce weighted squared error on average in both panels. Therefore a learned event-aware model must retain a path for posterior updates during quiet spans. Neither a hard output hold nor a state reset tests that mechanism. [Strong report and raw diagnostics](campaign_2026-10-05/target_mechanics/strong_baseline/REPORT.md).

### 4. Some structured-head negatives have losing fitting controls

E468's joint codebook head does not improve its historical GRU128 pair. Joint tilted delta is **−0.000618**, while the newly fitted regression anchor also loses **−0.000555**. The heads start exactly at the parent using zero residual weights and a zero metric gate; root checked [model.py](campaign_2026-10-05/joint_target_model/model.py). This is not an initialization that discards the parent. However, all arms, including `regression_anchor`, also receive a cross-entropy auxiliary term through their shared newly fitted feature trunk, and all follow the same one-pass procedure.

The negative is real. It is weak evidence against joint representation in general because the fitting protocol does not even improve its direct anchored control. It is not proof of a software bug or proof that a longer fit would work. Before spending on an elaborate posterior head, compare preserved parent, WMSE-only anchored regression, the historical correlation head procedure and the structured arm at equal fitting budget. Distinguish fitting a readout from learning a new recurrent representation. The production fine-tune already demonstrates that objective/procedure can materially change the final score. [E468 report](campaign_2026-10-05/joint_target_model/REPORT.md), [fit source](campaign_2026-10-05/joint_target_model/fit.py).

### 5. Activation gains do not establish a periodic signal

Sine-like candidates can beat a baseline on the diagnostic panel, with optimizer dependence. But E473's linear control and E481's center/slope-matched cubic control reproduce or exceed some benefits. E476's large first-seed schedule gain reverses in all eight new-seed contrasts in E480. E486 frozen-weight swaps show training adaptation matters, but changing an activation at inference also changes hidden states, output scale and clipping; it does not isolate a biological or periodic mechanism.

The clearer transferable lead is simpler: E488's fixed **50/50 GELU–LeakyReLU readout** beats GELU at epoch 3 by **+0.00893/+0.00431/+0.00869** across three seeds; two conditional intervals are positive, one crosses zero. Its learned recurrent-context gate is nearly tied with a static learned gate, with all three epoch-3 intervals crossing zero. This supports a complete-predictor test of the fixed mixture, not a claim that dynamic activation orchestration works. [Duration report](campaign_2026-10-05/context_mixture_duration/REPORT.md), [activation retrospective](campaign_2026-10-05/ACTIVATION_RETROSPECTIVE_2026-10-05.md).

### 6. Operational failures censor important questions

W384 v7 passed both actual-device one-batch probes and started both paired fits. Host-RAM guards stopped them during partial epoch 0; no completed paired checkpoints or scores exist. The width question remains unanswered. A valid retry requires changing and measuring the actual resident-memory path; increasing declared limits does not create host capacity.

Likewise E462's submitted INT8 ZIP failed Python linkage, not predictive quality. Its Python 3.11 repair has compatibility evidence, but runtime approval is separate. E419's provider error occurred after scientific exports, during output copying; recovered artifacts were verified and graded. E463's actual TPU request remains distinct from the earlier hardware smoke. None of these failures is a negative model result. [Execution audit](campaign_2026-10-05/review_20261005/EXECUTION_AND_EVIDENCE_AUDIT.md), [ZIP audit](campaign_2026-10-05/NEW_SUBMISSION_AUDIT_2026-10-05.md).

Strict parity is appropriate when testing interchangeable training implementations. It is not a universal requirement that a deliberately changed model reproduce the incumbent's optimizer update. Preserve E485's original failed gate; a separate explicit FP32/custom-recurrence experiment can ask another question. Do not retrospectively widen tolerances or silently convert deployment checks into predictive conclusions.

### 7. The error decomposition points to a problem, not a free solution

About **84–85%** of the strong model's joint residual energy lies in the target-difference mode, and about **91–92%** in class 3. Target variance itself is concentrated there. A sum/difference rotation adds no information; it cannot justify claiming that a relational head will gain that proportion of WP. The official score is nonlinear and separately target-weighted.

The unconstrained weighted-MSE optimum is already the absolute-target-tilted conditional mean, `E[|Y|Y | history] / E[|Y| | history]`, with clipped target Y. E468 implements this readout exactly for its finite atoms. “Predict the tilted mean” is not a new discovery or a replacement method. A substantive new method must improve the conditional representation, support, temporal state or finite-model optimization. [Earlier algebra](panel/brainteaser_algebra.md), [E468 readout](campaign_2026-10-05/joint_target_model/model.py).

## Rejections that should be kept narrow

| Direction | What actually happened | Question still open |
|---|---|---|
| Run-relative volume | E413 ran the earlier E321 GRU-input follow-up: +0.000962, interval [−0.000080,+0.002060], one exposed fold/seed. | Does it replicate, and help a complete model after the final-head procedure? It is no longer “never tested”. |
| Specialist/architecture heads | E464 completed all eight final heads and 16 complete-predictor contrasts; none beat seed-3 combo. | A materially different learned belief/state allocation, with a direct control, rather than rerunning these same specialists. |
| Target-specific clocks | E426 altered teacher-consistency EMA and was near-null. Earlier ridge effects disagreed by target/class. | Causal target-specific **input/history** adapters; EMA consistency is a different treatment. |
| Learned SSM and splines | Learned versus fixed diagonal decay gains +0.003630/+0.003816/+0.003043, all conditional CIs positive; spline benefits shrink across seeds. | Strong-model residual memory or a properly scaled standalone recurrence with matched nonlinear head and CPU cost. |
| Attention without a transformer | The tested positive-kernel memory is mixed; SSM attention loses in two seeds. | Targeted erase/write memory or event-keyed bounded retrieval; those are not the same recurrence as additive positive-kernel accumulation. |
| Joint/branch heads | Historical categorical fits and the new frozen coarse-codebook fits exist and lose their relevant controls. | Continuous anchored posterior dynamics, provided a clean regression-fitting control is competitive. |
| Normalization/cadence | E423 EMA-horizon screen loses; E414 calibrates mass variation; E430 accumulation remains drafts. | Proper pooled-numerator/denominator accumulation and the actual distilled-loss normalizer, each a separately identified treatment. |
| Width | Original W384 signal exists; W320 is a relevant same-recipe null; W384 replication is RAM-censored. | Replicated width effect and Muon×width interaction, after host-memory and CPU feasibility are measured. |

Raw current references: [E413 paired result](campaign_2026-10-04/checks/e413_paired_report.json), [experiment inventory](campaign_2026-10-04/EXPERIMENT_INVENTORY.md), [experiment ledger](experiments.jsonl), [component synthesis](campaign_2026-10-05/sequence_components/THREE_SEED_REPORT.md). Existing negatives stay in the record; different treatments get different identities.

## New directions and priorities

These are proposals for a future admitted research window, not started experiments. Costs below name the smallest sensible stage rather than inventing GPU-hour estimates from prototype time.

| Priority | Changed mechanism | Cheapest decisive comparison | Falsifier / promotion requirement |
|---|---|---|---|
| **1 — transfer, not another sweep** | Fixed GELU/LeakyReLU readout on the actual GRU256, separately one bounded recurrent-candidate intervention with slope-matched controls. | Current parent/head data and complete procedure, unchanged control, two named training seeds; retain frozen-function swap as a diagnostic. Start with readout transfer, not simultaneous embedding/candidate/head changes. | No complete-predictor gain on both seed comparisons; benefit disappears after final head; or native implementation cannot meet CPU budget. No direct public expectation from E488's magnitude. |
| **2 — event-aware belief updates** | Maintain an event-conditioned latent state plus a continuous quiet-run innovation path. Encode current/past events and causal age; never freeze outputs based on an oracle label. | Zero-initialized residual adapter using strong states, input innovations and event age, versus parameter-matched input-only/row-age adapter. Use causal lag controls or an explicitly nondeployable shuffle solely as a diagnostic. Then test one end-to-end state intervention if warranted. | No gain over baseline and matched controls on a second component block/seed. All inputs must be reproducible from the streaming callback; target-derived classes are diagnostic only. |
| **3 — cheap learned memory with local nonlinear readout** | Stable diagonal/LRU memory of causal innovations and a compact spline or hinge readout, initially a residual branch alongside the GRU. | Learned versus fixed decay; spline versus matched hinge/MLP; identical train-only knot statistics and final-head procedure. Measure score and CPU cycles before enlarging state. | No added information beyond the existing state/EMA at matched capacity; gain only on the tiny panel; or CPU latency removes deployment headroom. |
| **4 — address head fitting before posterior complexity** | Joint support plus continuous correction around the strong parent, with a pure regression anchor control and preserved original head. | Fit support on training components, compare pure WMSE residual, existing correlation fine-tune, joint tilted head and matched marginals; do not give the “regression” control an unexplained auxiliary CE burden. | Structured arm fails against competitive regression; poor support coverage; losses only on fitted groups. Use a continuous fallback, and preserve every arm. |
| **5 — original-target privileged teacher** | A short future-feature teacher predicts **the origin row's label** during training; causal student learns cross-fitted targets. This differs from substituting a future prediction. | One fixed future window and component-cross-fitted teacher; preserve origin label/weight and compare causal teacher/student controls. Test teacher value before a multigeneration programme. | Teacher advantage fails to transfer to student; component leakage; targets change origin; or improvement depends on oracle features at inference. More costly than head transfer; reserve a structural slice of the budget. |
| **6 — complete the low-cost open controls** | Replicate volume input, actual distilled normalizer, properly normalized accumulation and target-asymmetric histories as separate treatments. | One matched current-recipe comparison per treatment, then second-seed/component confirmation for any lead. | No gain at the complete-predictor endpoint. A negative closes the tested configuration, not every clock/loss/input family. |

### Why the proposed memory is different from the tested attention

The tested positive-kernel attention accumulates a small weighted memory. A delta-rule memory instead updates an addressed association using its prediction error; gating can separately erase obsolete memory. For example, a design could decay `S`, then add a rank-one correction proportional to `v − S k`, and read using a current causal query. That could retain “last comparable event” information without storing an unbounded sequence. It is a distinct hypothesis, not proof of retrieval value or a free CPU improvement. Start with small key/value widths and a parameter/cycle-matched additive-memory control.

This distinction is supported by **Gated Delta Networks**, which combines adaptive erasure with targeted delta updates. Its reported successes concern language/retrieval benchmarks, not these market targets. The proposed event-keyed residual is an inference from the mechanism. [Yang et al., ICLR 2025](https://arxiv.org/abs/2412.06464).

Likewise the learned-memory lead should not be reduced to “try Mamba”. **LRU** isolates linear/diagonal recurrence, stable parameterization, initialization and normalization. The local prototype tests only one small real diagonal recurrence. A scaled residual memory with controlled decay/innovation/readout tests the relevant ingredients more directly than wholesale architecture substitution. [Orvieto et al., ICML 2023](https://proceedings.mlr.press/v202/orvieto23a.html).

The original-target teacher proposal has precedent in time-series privileged-information learning, whose efficiency guarantee assumes a fixed-horizon Gaussian-linear system. Those assumptions do not establish validity for this challenge. It supplies a way to pose the training-only information experiment, not a promised gain. [Karlsson et al., AISTATS 2022](https://proceedings.mlr.press/v151/k-a-karlsson22a.html).

## How the next window should produce knowledge

Use three parallel research tracks with a shared incumbent endpoint: a small transfer track, a structural posterior/memory track, and a deployment/resource track. At least one structural method should be investigated even if low-cost transfer wins first. Reserve the majority of scientific compute for complete-predictor comparisons and their confirmation rather than another broad tiny-panel sweep. The exact allocation and ETAs must come from actual production-shape probes and an updated shared budget; the original 80-hour plan and account quota are different quantities.

Before fitting, bind the actual entrypoint, dataset/split/teacher lineage, device, controls and expected export layout. Keep checks necessary to prevent leakage, resource overflow and broken deployment. Avoid demanding update equivalence when a new algorithm intentionally changes updates. Before grading, exercise the assembled receipt/export schema once. Preserve operational failures; they are useful engineering evidence, but they must not consume the scientific negative column.

After fitting, report every registered arm through the fixed final-head procedure; keep per-seed paired differences, target metrics and component overlap explicit. A positive tiny screen nominates a transfer experiment. A complete-predictor comparison nominates a deployment artifact. A compatible ZIP still needs runtime evidence. A public submission measures external transfer. None of these steps authorizes adding local deltas to 0.6660 to manufacture an honest 0.7000 expectation.

The strongest underused idea is therefore **a model of evolving conditional beliefs, evaluated against a preserved strong predictor**. It combines event structure, quiet-run innovations and targeted memory without assuming that an anonymous physical decoder or a static branch codebook must be correct. The evidence motivates testing it; it does not yet establish that it works.
