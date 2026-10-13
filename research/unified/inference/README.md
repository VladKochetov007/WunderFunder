# Inference, quantization, and runtime

This domain covers competition CPU execution, native kernels, quantization, packaging, and measured runtime.

- [Versioned Python QAT and packed-row format](../../campaign_2026-10-13/qat_renewable_r1/QUANTIZATION_SPEC_R1.md) defines recurrent W4 and mixed input-W4/recurrent-W8 policy selection, exact Q4 nibble/code-scale transport, cache and checkpoint semantics, and the native A8/I16 differences. It indexes implementation in `src/wnn/quantization.py`, `src/wnn/training_qat.py`, and `src/wnn/quantization_export.py`; this is code-contract evidence only, with no model-quality or speed result.

- [QAT deployment consumer review](../../campaign_2026-10-13/qat_renewable_r1/DEPLOYMENT_CONSUMER_REVIEW_R1.md) checks the fixed evaluator's archive, native import, scoring mask, full-sequence reset/replay, and source manifest against hash-identified code. It records two corrected integration dependencies, remaining metadata/provenance guards, and explicitly makes no independent quantizer approval or quality/runtime claim.

- [Same-weight paired runtime harness](../../../scripts/benchmark_team_runtimes.py), described in the [team comparison](../../campaign_2026-10-13/team_models_r1/README.md), uses the frozen exact callback loader, repeated A-B-B-A, source/ELF/ISA receipts and reset checks. Four unit checks pass; no controlled engine timing or target-host winner is claimed from implementation alone. Python3.11 is required for imported C++.

- [EPYC9224 runtime reflection](../../campaign_2026-10-13/team_models_r1/EPYC_RUNTIME_REFLECTION_R1.md) reconciles portal durations, library/member hashes, Intel profiles and AMD Zen4 documentation. Q8 VNNI remains the baseline; no EPYC engine winner is established. It prioritizes paired same-weight timing, exact reuse and ensemble distillation; quantization and size limits remain explicit. H-TEAM-NATIVE-20261013 / E-TEAM-ARCHIVE-REPLAY-20261013-R1.

- [Joint team inference comparison](../../campaign_2026-10-13/team_models_r1/README.md): xteam_h4 already uses byte-identical own native Q8 binaries; pair_ftt2 uses teammate C++ dynamic-int8. Eight short original/conversion profiles passed reset replay, and full diagnostic replays are admitted. Conversion is a numerical treatment until exact prediction/quality comparison; no speedup or new winner is claimed from the short panel.

## Read these sources

- [Inference and quantization archive](../archive/2026-10-12/pages/06-inference-and-quantization.md)
- [Competition submission guide](../../../docs/submission_guide.md) and [native Q8 notes](../../../docs/native_q8.md)
- [CPU optimization research index](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/zen4_cpu_research_20261012_r1/INDEX.md)
- [A/B/C package review](../../../artifacts/submission_review_10_12_best_r1/SUBMISSION_PACKAGE_REVIEW_R1.json)

Runtime claims must name the hardware, exact ZIP, callback protocol, complete-test scaling method, and uncertainty. Local callback timings are not an AMD EPYC full-submission guarantee. The tested Q4 conversion lost quality on its screen; Q8 is the demonstrated package path in this archive.

## Resumed strategy execution — 2026-10-13

[Strategy execution campaign](../../campaign_2026-10-13/strategy_execution_r1/README.md) implements the full October12 review under the user's subsequent explicit resume. It links the approved plan, full proposal coverage, resource authority, frozen comparisons and live frontier. All Kaggle allowance and actual local GPU capacity are authorized; CPU/RAM protections and no-paid-spend remain. Initial status is preparation, with no new fitted result. Existing folds remain exposed.

## Current QAT implementation

[Deployment evaluator](../../../scripts/evaluate_qat_deployment.py), described in the [QAT campaign](../../campaign_2026-10-13/qat_renewable_r1/README.md), freezes FP32/nativeQ8/exact packedQ8/uniformW4/mixedW4W8 artifacts, profiles complete fitting trajectories and exports fixed-full-fold native callback predictions with official scores and component statistics. The R2 manifest/source guards, actual selected-mask reconstruction, exact transport, state trace and A→B→A callback checks pass; [independent review](../reviews/QAT_RENEWABLE_R1_INDEPENDENT_REVIEW.md) records the implementation and ancestry findings, while [R2 clearance](../reviews/QAT_DEPLOYMENT_R2_FULLFOLD_CLEARANCE.md) clears the pinned 468-group F0 baseline comparison for controller-owned execution. Profile scores are fit-component diagnostics; no QAT gain, EPYC speed claim, or clean-holdout status follows.

[QAT plan and campaign](../../campaign_2026-10-13/qat_renewable_r1/README.md) records the authorized uniform recurrent W4/mixed W4-W8, exact native transport, activation-aware and learned-scale comparisons, using renewable Kaggle/local capacity. [Plan](../../campaign_2026-10-13/qat_renewable_r1/PLAN.md), [contract](../../campaign_2026-10-13/qat_renewable_r1/CONTRACT.json) and [frontier](../../campaign_2026-10-13/qat_renewable_r1/FRONTIER.md) bind clean-T3 parent ancestry, matched controls and exposed multi-fold/seed validation (H-QAT-W4/A8/LSQ-20261013; E-QAT-DEPLOYMENT-20261013-R1). Initial status is implementation; archive savings are distinct from CPU speed and no quality gain is established.

[Post-fit QAT exact-export receipt](../../campaign_2026-10-13/qat_renewable_r1/POSTFIT_QAT_EXACT_EXPORT_R1.md) documents the evaluator's checkpoint-schema guards and preservation of matching trained code/scale tensors, verified on the completed recurrent-W4 checkpoint. This closes an archive integration gap; native full-fold quality/runtime evaluation remains pending.

[Checkpoint-path correction](../../campaign_2026-10-13/qat_renewable_r1/POSTFIT_QAT_CHECKPOINT_PATH_CORRECTION_R1.md) retains the original exact-export receipt and corrects its smoke label: the recorded hash identifies `model.pt`; its normalized tensor payload is byte-identical to `model_ep0.pt`, though the smoke did not explicitly resolve `@ep0`.

[Learned row-scale QAT specification](../../campaign_2026-10-13/qat_renewable_r1/LSQ_ROW_SCALE_SPEC_R1.md) describes the opt-in LSQ-inspired scale derivative, matched initial-representation invariant, and optimizer decay semantics. The [prototype receipt](../../campaign_2026-10-13/qat_renewable_r1/LSQ_ROW_SCALE_PROTOTYPE_R1.md) records code and 23 passing bounded CPU checks, including actual trainer resume/EMA export; no fit or scale-learning benefit is claimed.

## QAT native transport — 2026-10-13

[Native dynamic Q8 transport and state diagnostics](QAT_NATIVE_TRANSPORT_R1.md) records exact row-code/scale import/export for `q8vnni*` and copied recurrent-state access for the QAT deployment plan. Rust tests and a Python 3.13 synthetic integration smoke pass, including exact transport and reset/replay. This is implementation evidence only: no actual-data quality, archive-size, runtime, or INT4-speed result is claimed. Related: H-QAT-W4-20261013, H-QAT-A8-20261013, H-QAT-LSQ-20261013, E-QAT-DEPLOYMENT-20261013-R1.
[Real-model QAT deployment checks](../../campaign_2026-10-13/qat_renewable_r1/deployment_checks_r1/README.md) bind clean-T3 H256 CP5 F0s3 and two complete fitting trajectories. Exact packedQ8 preserves native predictions/states; all formats reset/replay. Actual Q8/W4/mixed archives are1,086,824/678,883/882,966bytes, but resident weight storage is identical after expansion. H-QAT-W4-20261013; fitting-only engineering evidence, no QAT/CV gain or INT4 speed claim.

[CP5 native full-F0 baseline](../../campaign_2026-10-13/qat_renewable_r1/baseline_results_r2/README.md) completes468 groups/141 components/808,863 scored rows at WP=.681135399, with reset/replay and exact source/library/archive manifest. Unchanged-parent exposed-F0 baseline only; paired compressed/QAT results, bootstrap and seed variation remain pending. H-QAT-W4-20261013; no public forecast.

[Completed five-format CP5 baseline](../../campaign_2026-10-13/qat_renewable_r1/baseline_results_r2/README.md) now records full-F0 FP32 0.681154152, native/exact packed Q8 0.681135399, W4 0.680288607 and mixed W4/W8 0.680057216. Native and packed Q8 complete prediction hashes match. The linked per-format receipts pin scores, callbacks and resident storage; archive savings do not imply INT4 execution. Exposed F0 and one parent only.

[Uniform 10% matched grade](../reviews/QAT_10PCT_UNIFORM_W4_MATCHED_GRADE_R1.md) and [mixed 10% matched grade](../reviews/QAT_10PCT_MIXED_W4W8_MATCHED_GRADE_R1.md) independently validate ancestry, sampling and exact checkpoint transport: native QAT minus matched PTQ −0.0001149 and +0.0001390, respectively. Both component-bootstrap intervals span zero. The [30% allocation](../../campaign_2026-10-13/qat_renewable_r1/ALLOCATION_30PCT_MIXED_R1.json) retains scale-dependent follow-up without declaring a winner or failure.
