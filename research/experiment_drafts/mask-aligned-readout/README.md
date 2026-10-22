# Mask-aligned readout fine-tuning

**Status: BLOCKED PRE-ADMISSION.** This is a docs-only child of the existing H-023/H-030/H-035 mask/propensity fine-tuning line. It proposes four matched, head-only fits; completed new fits: **0**. No new parent fits, run results, or execution approval are included.

The comparison tests whether weighting the head's fitting loss by the actual `is_scored` mask improves the challenge weighted-Pearson score over fitting on all valid `need_prediction` rows. Both arms share one frozen, fold-clean GRU parent per cell; the mask is a loss weight only and never an input. `train.parquet` has no `is_scored`, so fitting must use the valid-side fitting components and their real mask.

The public [MarkBrezina/WunderConnectome](https://github.com/MarkBrezina/WunderConnectome/tree/5d086fc90ea0c11a39391dde702e2a4f0a47a869) README reports a +0.000802 scored-row head-fine-tuning delta on an author-run sample. Treat this as an unverified local claim, not a competition result or reproducible estimate; the repository has no license identified. No code or weights are copied.

## Admission blockers

The source-pinned [F0 provenance record](https://github.com/VladKochetov007/WunderFunder/blob/c452e8cf4f8ef57e5c07c6801a6e10fc7616251d/artifacts/runs/oldz4_h230_parent_f0_s3_r3/provenance.json) records split-manifest SHA `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`; the independent [F2 provenance record](https://github.com/VladKochetov007/WunderFunder/blob/c452e8cf4f8ef57e5c07c6801a6e10fc7616251d/artifacts/runs/h382_muonall256prop0_f2_s3_e4/provenance.json) records `a9973e5798d3109362571c299b2d08eca71d01e92008da39b0d0ddde54c93aa9`. Their repository paths are `artifacts/runs/oldz4_h230_parent_f0_s3_r3/provenance.json` and `artifacts/runs/h382_muonall256prop0_f2_s3_e4/provenance.json`. The actual checkpoint files and full split manifests are outside this sparse checkout, so their availability, checkpoint hashes, and complete source-clean ancestry remain unbound. F2 must remain an independent fold-2 parent; F0 cannot be cloned.

Both provenance records report historical FP16 parent training. All new child forward/backward and optimizer computation is proposed in FP32, with the frozen parent loaded as FP32 tensors. This does not claim end-to-end FP32 parent history. The receiving owner resolves historical-parent eligibility under the current precision policy before admission; missing parent artifacts and ancestry bindings remain blockers.

## Packet files

- [`PR.md`](PR.md) contains the complete 11-section experiment proposal.
- [`recipe.json`](recipe.json) freezes the cells, arms, budget, metrics, decision rule, and admission conditions.

The source checkout is branch `experiment/mask-aligned-readout` at `c452e8cf4f8ef57e5c07c6801a6e10fc7616251d`. The canonical entrypoint is identified in the trainer's [CLI README](../../../src/wnn/training/README.md#L3) and [`scripts/train_rnn.py`](../../../scripts/train_rnn.py#L20); [`cli_data.py`](../../../src/wnn/training/cli_data.py#L42) exposes fixed checkpoints, valid-side source, initializer, and trunk freeze; [`cli_features.py`](../../../src/wnn/training/cli_features.py#L35) exposes scored-mask weighting; [`cli_performance.py`](../../../src/wnn/training/cli_performance.py#L33) exposes precision; [`setup_lineage.py`](../../../src/wnn/training/setup_lineage.py#L163) configures strict FP32 and rejects mask weighting with `train`; [`targets.py`](../../../src/wnn/training/targets.py#L135) applies the mask to the main loss. Metric and resampling references are [`gmetric.py`](../../../src/wnn/gmetric.py#L159) and [`evaluation.py`](../../../src/wnn/evaluation.py#L89). No adapter is proposed. Exact commands remain unbound pending parent artifacts, manifest paths, and inherited settings.
