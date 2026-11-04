# Methods, sampling, and metric

- [Fixed eight-L4 delivery](../../experiment_drafts/galam-l4-quality8-20261028/launch/README.md)
  implements bounded source-audit/index/component/split preparation, cached
  train-only112/211-feature moments, one-batch prefetch and strict profile-based
  admission. Eight complete-source profiles and their independent review passed;
  all eight full fits now run with detached ownership and node GitHub reporting.
  Source/evidence byte archives preserve the exact receiver identities.

- [GALAM live monitoring and L4 preparation](../../experiment_drafts/galam-l4-monitoring-20261028/README.md)
  separates comparable train-eval/valid curves from online fitting loss, preserves
  full chronological primary evaluation, and defines bounded model/resource
  telemetry. Demo data are not training evidence. A partial train download and
  full disks block actual L4 fits.

- [Mandatory chronological primary validation, ADR0044](../../../specs/adr/0044-chronological-primary-validation.md)
  replaces prospective random-fold primary selection with earlier-train/later-valid
  membership. [Executable producer and admission checks](../../../src/wnn/chronological_splits.py)
  pin source coverage and exclude valid-linked fit components; no real-data audit
  or new fit was performed. [GALAM packet](../../experiment_drafts/galam-reconstruction-20261027/README.md)
  demonstrates the new policy in a source-traced, CPU-checked adaptation.

[Canonical tiedlead implementation](../../campaign_2026-10-18/quality_gap_r1/tiedlead_capability_publication_r1/README.md) preserves sequence-wide causal lead gathering, raw-scale shared branch encoding, per-call delta reset and carried recurrent state. The isolated branch supports canonical checkpoint prediction/evaluation; actual-data/GPU and native ZIP/ONNX behavior are unmeasured.

[Strict propensity V3 engineering publication](../../campaign_2026-10-18/quality_gap_r1/propensity_capability_publication_r2/README.md) adds optimizer diagnostics and exact epoch-boundary state recovery to the opt-in strict path;33tinyCPUtests and independent review pass. Branch235db75a is published, not staged on B or merged into the running source. Recovery retains failures and unknown in-flight step calls, with journal-corruption and runtime/GPU limits explicit. This supplies an engineering component for the modified final-supervision experiment, not a quality result or admission.

[Canonical final-supervision capability audit](../../campaign_2026-10-18/quality_gap_r1/tl3_final_supervision_capability_audit_r1/README.md) and [raw-scale/chunk addendum](../../campaign_2026-10-18/quality_gap_r1/tl3_final_supervision_capability_audit_r1/ADDENDUM_RAW_SCALE_AND_CHUNKS_V1.md) identify the two missing implementations for the matched modified-TL3 design: tiedlead's shared raw-scale encoder/sequence-wide lead features, and identical family-head/cache topology with explicit zero-weight loss branches. All21source pins match independent static review. Delta resets per encode call; recurrent state carries. These are recovered source semantics and proposed tests, not historical runtime parity, implemented capability or a fit. Existing six-source arithmetic is reused.

- [Released F0 native submission](../../campaign_2026-10-18/f0_submission_r1/receipts/RELEASE.json): fixed75%wide/25%deep Q8 ZIP17.48MB; actual full-F0 WP0.6738263577 confirmed by independent original-label/direct grades. The [packet](../../campaign_2026-10-18/f0_submission_r1/README.md) retains selection exposure and package provenance; the subsequent user-reported portal result was0.65839 in45m47s, below the incumbent, without portal-byte SHA confirmation.

Reusable metric, causal data handling, statistical and implementation contracts.

## Current navigation

- [Prospective fast-screen policy / ADR0043](../../../specs/adr/0043-fast-screens-and-transformer-teacher-scale.md) — replaces mandatory CP30 for future drafts with4–6 total new fits per initial idea screen, explicit matched cells, cached controls and teacher/student costs. The updated [workflow](EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md), [template](../templates/EXPERIMENT_DRAFT_PR.md) and [plan helper](../../../scripts/experiment_draft_plan.py) apply it; historical explicit30-cell recipes remain unchanged. Small panels estimate direction, not significance. The [focused migration tests](../../../tests/test_experiment_draft_plan.py) check budget, identity, precision, reuse and legacy preservation;10 CPU cases pass. Adds large B300 teacher planning without resuming stopped jobs.

- [Autoresearch plateau guideline](AUTORESEARCH_PLATEAU_GUIDELINE_20261019_R1.md) — verbatim user `research.txt` with source SHA-256 and English scope. Recommends changing tested assumptions, retaining informative negatives, measuring screen fidelity and choosing work by information value. Selected primary-source checks remain in the October 15 reset note; the imported citations and numerical claims are not all independently verified.

- [Accepted October 20 local/Kaggle plan](../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md) — canonical continuation specification after the user's plateau review and compute revision. Defines 15 local GPU-hours, verified Kaggle parallelism, the scoped 30-run waiver, matched i1 masking/history arms, causal equations, scoring and conditional follow-ups. Implementation is active; no new fits or scores. Read its linked campaign frontier after compaction. Historical instructions and source proposals do not override this accepted scope.
- [User-requested F0 submission](../../campaign_2026-10-18/f0_submission_r1/README.md) — the completed epoch5 wide/deep pair is packaged under an explicit exploratory release exception. A fixed75/25 mixture scores0.673850944 across468 exposed-F0 groups and wins the five-point screen. Checkpoint-to-package tensors and preliminary native callbacks pass; full Q8 replay and release are pending. This is not CP30 or a public-score forecast.

- [R2 metric decompositions and clocks](../../campaign_2026-10-17/research_reset_r2/theory/README.md) — exact label-run variance/covariance identity, existing WMSE fluctuation penalty and within/between-sequence pooled-score decomposition. All84 target-scores reconstruct within5.56e-16; pure numerical fixtures exercise the canonical scorer. Observation/state/output/label/gradient clocks are distinguished; oracle gain is not a causal gain or Bayes ceiling. E-RR17-R2-DECOMPOSITION/OUTPUTS.
- [Fixed causal Kalman input probe](../../campaign_2026-10-18/online_features_r1/kalman_filter_probe/README.md) — a prediction-only frozen-GRU replay on 16 selected exposed-F0 groups. Local-trend replacement is +0.000338 WP versus raw, with a 12-component conditional interval [−0.000019, +0.000716]; random walk is +0.000119. This small fixed-endpoint screen does not evaluate training with learned filter features or online parameter updates. The separate execution receipt records a 1.126 GB worker-cgroup peak and zero OOM events; process RSS differs. The trained retained-raw teacher/student proposal is draft PR #17, with B300 profiling and all task fits pending.

- [No-fit research contract and future CP30 design](../../campaign_2026-10-17/research_reset_r1/README.md) — bounded EDA/frozen inference now, exact30paired cells per arm only after future fit authority, strictFP32, clean ancestry and fixed-fold shared-seed uncertainty. Also links the weighted-target derivation and explicit limitations.

- [Validation protocol map](../validation/VALIDATION_PROTOCOLS.md) — pooled official scoring, exact panel counts and interpretation limits.
- [Training component map](../../../src/wnn/training/README.md) — maintained implementation and explicit runtime environment.
- [Development checks](../../../docs/DEVELOPMENT.md) — repeatable code-quality and compatibility checks.

For active status, start with [current state](../../../specs/current-state.md).
The dated entries below retain their original descriptions and links. Their
“pending”, “active” and authorization statements reflect their recording time;
use a later result receipt or the current audit to establish final status.

<details>
<summary>Dated index preserved before the October 17 maintenance</summary>

This domain collects the reusable research protocol: data membership, causal sequence handling, target treatments, official scoring, and uncertainty interpretation.

## Bounded objective follow-up — 4 November2026

[The new objective/clipping packet](../../experiment_drafts/l4-objective-20261104/README.md)
connects the prior audit to a fitting-only derivative diagnostic and maximum-six-fit
teacher/student comparison. It preserves exact reused controls, official primary
validation, default-off behavior and auxiliary-head layout, with an explicit
selection rule and transient-cache disk policy. Preparation only until actual
receiving diagnostic/profile/fit evidence is recorded; no quality/public gain.

## Clipping and objective audit — 4 November2026

[The L4 clipping audit](../../experiment_drafts/l4-teacher-capacity-20261104/CLIPPING_AND_OBJECTIVE_AUDIT_R1.md)
traces global clipping through gatewise Muon/AdamW and the actual main/structural/
next-loss reductions. A24-step synthetic CPU replay shows near-invariance to
constant gradient scaling but changed update directions under varying clipping;
it is not a GRU quality result. The note distinguishes head tensors from loss
sources and proposes fitting-only gradient decomposition before further paired
fits. Exact source/result hashes and independent-review limitations are linked.
The [campaign packet](../../experiment_drafts/l4-teacher-capacity-20261104/README.md)
also indexes the completed six-GRU negative result and fixed-endpoint curves.

## Read these sources

- [Causal AR teacher/student and DDP training proposal](../ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md) gives free-running output recurrence, exact sequence/window/reset semantics, separately normalized weighted-DDP gradients with accumulation, official pooled evaluation, fitting ancestry and conditional eight-B300 cost arithmetic. The user's quality-first steering expands the scale/context/horizon and teacher-assistant portfolio. Four H-TRANSFORMER-* hypotheses and three proposed comparisons are recorded; no runnable trainer, hardware profile, admitted fit or quality claim. The canonical architecture/source description is in the [ideas index](../ideas/README.md).

- [October15 research-cycle reset](RESEARCH_LOOP_RESET_20261015_R1.md) implements the user's instruction not to get stuck: separates warm-parent KD attribution, short-screen fidelity, fresh mechanism generation and delivery failures; assigns concrete measurements and outcome-dependent decisions. It preserves uncertain branches and tracks information over the next four scientific comparisons. Primary AIRA/Amazon/JANUS sources are checked with transfer limitations. Records an actual double-counted-memory correction and collector/producer inventory failure; no new framework, paid spending or quality gain is claimed.

- [Project research methods](../../RESEARCH_METHODS.md)
- [Official scoring reference](COMPETITION_METRIC.md)
- [Research objective](../../../specs/research-objective.md), [pipeline](../../../specs/research-pipeline.md), and [ADR index](../../../specs/adr/README.md)
- [Data, metric, and validation archive](../archive/2026-10-12/pages/01-data-and-validation.md)
- [Model and training archive](../archive/2026-10-12/pages/02-models-and-training.md)
- [Historical validation protocol](../../../hpo/VALIDATION_PROTOCOL.md) records the October 10 scratch-bank variance estimates, 28-recipe fold comparison and five-comparison composite-domain observation. These are useful calibration evidence, with stale F1 counts and reserved-fold claims: all four folds are now exposed. Its minimum detectable effects are recipe-specific power estimates, not universal allocation thresholds. [The first-principles assessment](../ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md) explains these limits and the planned paired comparisons.

The official metric and fold membership are part of every scored claim. Training-target clipping, teacher targets, initializer ancestry, and fitting membership are separate method details; preserve them in the recipe and experiment record.

## Activation-aware recurrent QAT — 2026-10-13

[Activation-aware recurrent QAT backend](../../campaign_2026-10-13/qat_renewable_r1/ACTIVATION_AWARE_BACKEND_R1.md) adds scan-based GRU recurrence and native-style A8 fake quantization before every GRU input/recurrent matvec while carrying FP32 state. [Its CPU/native receipt](../../campaign_2026-10-13/qat_renewable_r1/ACTIVATION_AWARE_CPU_NATIVE_RECEIPT_R1.json) records four passing engineering tests and a controlled recurrent-state comparison (max error `5.96e-08`; disabling A8 changes state by `6.90e-04`) on the pre-correction source. Raw input/projection-weight/I16 differences and head hidden/projection activation blocks remain outside scope; no quality claim exists.

[Actual-recipe activation-QAT profile plan](../../campaign_2026-10-13/qat_renewable_r1/ACTIVATION_AWARE_ACTUAL_PROFILE_PLAN_R1.json) pins the 211-feature clean-T3 CP5 initializer and derives the 10% recurrent-W4 command, actual homogeneous teacher/propensity caches, and existing main/structural/next-change objective through `train_rnn.py`. It reuses one B16 batch for two first-prefix updates per compiled scan arm (four updates maximum), records cache/source/normalization identity and resource/compile evidence, and remains pending root-owned GPU admission. The earlier random-input square-loss path is explicitly synthetic-only. No fit-quality inference follows from either profile.

## Resumed strategy execution — 2026-10-13

[Strategy execution campaign](../../campaign_2026-10-13/strategy_execution_r1/README.md) implements the full October12 review under the user's subsequent explicit resume. It links the approved plan, full proposal coverage, resource authority, frozen comparisons and live frontier. All Kaggle allowance and actual local GPU capacity are authorized; CPU/RAM protections and no-paid-spend remain. Initial status is preparation, with no new fitted result. Existing folds remain exposed.

- [Evidence-based first-wave allocation](../../campaign_2026-10-13/strategy_execution_r1/ALLOCATION_AFTER_FIRST_WAVE_20261013_R1.json) and [reasoned review](../reviews/STRATEGY_PROGRESS_AND_REALLOCATION_20261013.md) — maintains all39 hypotheses while concentrating the next full fits on two mechanisms, separating cheap uncertain leads from expensive weak knobs, and requiring complete endpoint/serving confirmation. This is a current allocation, not a universal noise threshold or a permanent family ban.

- [Paired domain-grade consumer](../../../scripts/grade_strategy_domains.py) and [tests](../../../tests/test_strategy_domain_grade.py) — computes ordinary official WP and frozen component-frequency sensitivities from exactly aligned FP32 producer arrays, one target mask and bounded actual NPY/NPZ headers. Two tests match official pooled scores and complete-component replication and reject header/order errors. Engineering only; independent review and actual endpoint stress scoring remain pending.

- [Clipped mixture derivative and input-conditioned recurrence checks](../../campaign_2026-10-14/cpu_student_frontier_r1/DISTILLATION_AND_RECURRENCE_THEORY_R1.md) derives a local fixed-weight WP blend diagnostic and exact one-layer minGRU state-difference product. The accompanying script and JSON verify both on synthetic inputs, including clipping; this is mathematical implementation evidence, not grade or fit evidence.

- [Actual recurrent weight geometry](../../campaign_2026-10-14/cpu_student_frontier_r1/WEIGHT_GEOMETRY_R1.md): H256 control/H384 Muon teacher checkpoint spectra and exact W8/W4 row-quantizer distortion, tied to the4Hr versus3H² cost formula. All95%-energy ranks exceed the low-rank arithmetic break-even threshold. This is unweighted historical-matrix evidence, not a controlled optimizer comparison, score estimate or reason to terminate trained low-rank exploration. E-STUDENT-WEIGHT-GEOMETRY-20261014-R1.

- [Recurrent timescales and initializer gradients](../../campaign_2026-10-14/cpu_student_frontier_r1/RECURRENT_TIMESCALES_AND_INITIALIZER_GRADIENTS_R1.md): chrono initialization (Tallec/Ollivier2018), minLSTM initialization analysis (Feng et al.2024), truncation-bias literature and official checkpointing API inform a distinct minGRU bias arm and complete-sequence initializer training. Six scalar checks and five synthetic gradient tests pass; no fitted-model benefit. H-MINGRU-TIMESCALES-20261014 / H-STATE-INIT-20261014.

- [Complete-sequence initializer profile R2](../../campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R2_README.md) repairs three prelaunch checks in the one-sequence CUDA resource smoke. A meta-device placement and host-cap rounding test passes 10/10; the separate 400 MB standard-library preflight binds checkpoint/cache ancestry. No CUDA run, fit or quality result exists yet. H-STATE-INIT-20261014.

- [Research-team draft experiment workflow](EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md) preserves the October17 branch/evidence contract, with the October19 prospective revision to 4–6 total new fits per initial idea screen. Strict FP32, source-clean evaluation, uncertainty, progress logging, separate archives and unified result/merge trace remain. Historical frozen CP30 plans are not rewritten. Its [PR template](../templates/EXPERIMENT_DRAFT_PR.md) gives eleven required sections.

- [Draft matrix expander](../../../scripts/experiment_draft_plan.py) reads a bounded recipe, defaults to two matched cells (four fits for two arms), validates explicit screen/follow-up and reused-control accounting, and prints precision requirements and missing bindings. Unchanged historical v1 CP30 recipes require an explicit source-commit override. It launches no training and supplies no queue. The [reporting API](../../../src/wnn/research_reporting.py) records immutable run metadata and append-only finite metrics; the [archive CLI](../../../scripts/research_result_bundle.py) streams explicitly selected evidence/weights into separate tarballs with hashes. These helpers require explicit trainer integration; static checks only, no new fit evidence.

- [Modular RNN trainer refactor](MODULAR_RNN_TRAINER_REFACTOR_20261017_R1.md) records semantic components, explicit per-run state, preserved172-option CLI, source/portable-packet handling and Sol6.1/medium independent corrections. Ruff and structural checks pass; no fits, tests, numeric parity or throughput evidence. The [component map](../../../src/wnn/training/README.md) documents reusable APIs and every module.

</details>
