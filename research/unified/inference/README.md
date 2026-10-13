# Inference, quantization, and runtime

This domain covers competition CPU execution, native kernels, quantization, packaging, and measured runtime.

- [Versioned Python QAT and packed-row format](../../campaign_2026-10-13/qat_renewable_r1/QUANTIZATION_SPEC_R1.md) defines recurrent W4 and mixed input-W4/recurrent-W8 policy selection, exact Q4 nibble/code-scale transport, cache and checkpoint semantics, and the native A8/I16 differences. It indexes implementation in `src/wnn/quantization.py`, `src/wnn/training_qat.py`, and `src/wnn/quantization_export.py`; this is code-contract evidence only, with no model-quality or speed result.

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

[Deployment evaluator](../../../scripts/evaluate_qat_deployment.py), described in the [QAT campaign](../../campaign_2026-10-13/qat_renewable_r1/README.md), freezes FP32/nativeQ8/exact packedQ8/uniformW4/mixedW4W8 artifacts, profiles complete fitting trajectories and exports fixed-full-fold native callback predictions with official scores and component statistics. It also samples recurrent states and verifies reset/replay. Implementation awaits integrated native/archive checks; no quality or CPU gain is inferred.

[QAT plan and campaign](../../campaign_2026-10-13/qat_renewable_r1/README.md) records the authorized uniform recurrent W4/mixed W4-W8, exact native transport, activation-aware and learned-scale comparisons, using renewable Kaggle/local capacity. [Plan](../../campaign_2026-10-13/qat_renewable_r1/PLAN.md), [contract](../../campaign_2026-10-13/qat_renewable_r1/CONTRACT.json) and [frontier](../../campaign_2026-10-13/qat_renewable_r1/FRONTIER.md) bind clean-T3 parent ancestry, matched controls and exposed multi-fold/seed validation (H-QAT-W4/A8/LSQ-20261013; E-QAT-DEPLOYMENT-20261013-R1). Initial status is implementation; archive savings are distinct from CPU speed and no quality gain is established.

## QAT native transport — 2026-10-13

[Native dynamic Q8 transport and state diagnostics](QAT_NATIVE_TRANSPORT_R1.md) records exact row-code/scale import/export for `q8vnni*` and copied recurrent-state access for the QAT deployment plan. Rust tests and a Python 3.13 synthetic integration smoke pass, including exact transport and reset/replay. This is implementation evidence only: no actual-data quality, archive-size, runtime, or INT4-speed result is claimed. Related: H-QAT-W4-20261013, H-QAT-A8-20261013, H-QAT-LSQ-20261013, E-QAT-DEPLOYMENT-20261013-R1.
[Real-model QAT deployment checks](../../campaign_2026-10-13/qat_renewable_r1/deployment_checks_r1/README.md) bind clean-T3 H256 CP5 F0s3 and two complete fitting trajectories. Exact packedQ8 preserves native predictions/states; all formats reset/replay. Actual Q8/W4/mixed archives are1,086,824/678,883/882,966bytes, but resident weight storage is identical after expansion. H-QAT-W4-20261013; fitting-only engineering evidence, no QAT/CV gain or INT4 speed claim.
