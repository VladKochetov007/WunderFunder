# Methods, sampling, and metric

This domain collects the reusable research protocol: data membership, causal sequence handling, target treatments, official scoring, and uncertainty interpretation.

## Read these sources

- [Project research methods](../../RESEARCH_METHODS.md)
- [Official scoring reference](COMPETITION_METRIC.md)
- [Research objective](../../../specs/research-objective.md), [pipeline](../../../specs/research-pipeline.md), and [ADR index](../../../specs/adr/README.md)
- [Data, metric, and validation archive](../archive/2026-10-12/pages/01-data-and-validation.md)
- [Model and training archive](../archive/2026-10-12/pages/02-models-and-training.md)
- [Historical validation protocol](../../../hpo/VALIDATION_PROTOCOL.md) records the October 10 scratch-bank variance estimates, 28-recipe fold comparison and five-comparison composite-domain observation. These are useful calibration evidence, with stale F1 counts and reserved-fold claims: all four folds are now exposed. Its minimum detectable effects are recipe-specific power estimates, not universal allocation thresholds. [The first-principles assessment](../ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md) explains these limits and the planned paired comparisons.

The official metric and fold membership are part of every scored claim. Training-target clipping, teacher targets, initializer ancestry, and fitting membership are separate method details; preserve them in the recipe and experiment record.

## Activation-aware recurrent QAT — 2026-10-13

[Activation-aware recurrent QAT backend](../../campaign_2026-10-13/qat_renewable_r1/ACTIVATION_AWARE_BACKEND_R1.md) adds scan-based GRU recurrence and native-style A8 fake quantization at the input/recurrent matvec boundaries while carrying FP32 state. [Its CPU/native receipt](../../campaign_2026-10-13/qat_renewable_r1/ACTIVATION_AWARE_CPU_NATIVE_RECEIPT_R1.json) records four passing engineering tests and a controlled recurrent-state comparison (max error `5.96e-08`; disabling A8 changes state by `6.90e-04`). The compiled B16/C1000 profile is unrun; projection/head activation treatment is outside this implementation, and no quality claim exists.

## Resumed strategy execution — 2026-10-13

[Strategy execution campaign](../../campaign_2026-10-13/strategy_execution_r1/README.md) implements the full October12 review under the user's subsequent explicit resume. It links the approved plan, full proposal coverage, resource authority, frozen comparisons and live frontier. All Kaggle allowance and actual local GPU capacity are authorized; CPU/RAM protections and no-paid-spend remain. Initial status is preparation, with no new fitted result. Existing folds remain exposed.

- [Evidence-based first-wave allocation](../../campaign_2026-10-13/strategy_execution_r1/ALLOCATION_AFTER_FIRST_WAVE_20261013_R1.json) and [reasoned review](../reviews/STRATEGY_PROGRESS_AND_REALLOCATION_20261013.md) — maintains all39 hypotheses while concentrating the next full fits on two mechanisms, separating cheap uncertain leads from expensive weak knobs, and requiring complete endpoint/serving confirmation. This is a current allocation, not a universal noise threshold or a permanent family ban.

- [Paired domain-grade consumer](../../../scripts/grade_strategy_domains.py) and [tests](../../../tests/test_strategy_domain_grade.py) — computes ordinary official WP and frozen component-frequency sensitivities from exactly aligned FP32 producer arrays, one target mask and bounded actual NPY/NPZ headers. Two tests match official pooled scores and complete-component replication and reject header/order errors. Engineering only; independent review and actual endpoint stress scoring remain pending.
