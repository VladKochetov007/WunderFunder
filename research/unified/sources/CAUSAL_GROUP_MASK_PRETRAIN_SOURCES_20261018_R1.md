---
kind: literature-review
status: source-reviewed-proposal
created_utc: 2026-10-18T06:00:51Z
hypothesis_ids: [H-CAUSAL-GROUP-MASK-PRETRAIN-20261018-R1]
experiment_ids: [E-CAUSAL-GROUP-MASK-PRETRAIN-20261018-R1]
evidence_paths:
  - ../../experiment_drafts/causal-group-mask-pretraining-b300/README.md
---

# Causal group-masked pretraining: source review

## Primary sources

1. [Ti-MAE: Self-Supervised Masked Time Series Autoencoders](https://arxiv.org/abs/2301.08871) masks embedded series values and reconstructs them at point level. Its paper reports representation results on its forecasting/classification benchmarks. It motivates raw-input reconstruction, but does not establish gains for this dataset or authorize bidirectional inference.
2. [TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders](https://arxiv.org/abs/2303.00320), accepted at WSDM'26, elevates nonoverlapping subseries into semantic units and separates visible/masked-region processing. This motivates masking meaningful chunks and measuring reconstruction transfer; its reported datasets and task do not establish our result.
3. [Learning to Embed Time Series Patches Independently](https://arxiv.org/abs/2312.16427), accepted at ICLR'24, argues for independently reconstructing patches and adds a separate adjacent-patch contrastive objective. It is useful counterevidence to assuming every masked objective should model cross-patch dependencies. The current draft uses no contrastive loss and tests grouped cross-channel reconstruction rather than temporal-patch embedding.

## Transfer to this task

The dataset exposes 112 raw input channels with explicit feature ranges: twelve 4–11-channel groups across instruments i0 and i1, plus a0…a7. The proposed mechanism masks one whole group for a 32-row block and reconstructs only that group. This asks the shared recurrent state to encode relations among contemporaneous channel groups while retaining causal historical context. The mask bit distinguishes an intentionally hidden value from a normalized zero. The method is a project-specific hypothesis from the provided schema, not a claim that channel labels have known physical semantics.

The pretrainer rebuilds the 99 event channels from the masked raw inputs. That prevents deterministic event features from copying the hidden current group. The GRU only receives the current partially visible row, mask indicator and its prior state; no future row, competition label, or evaluation score mask is read. The teacher uses the same 211 raw-plus-event input order at supervised fine-tuning; the converter copies those projection columns and recurrent weights, discards mask-only columns and reconstruction head, and retains the same task-readout initialization across paired arms.

This differs from the existing [next-input auxiliary draft, PR #18](https://github.com/VladKochetov007/WunderFunder/pull/18): that predicts the next raw row from current state during supervised training. It also differs from [meta-TTA PR #20](https://github.com/VladKochetov007/WunderFunder/pull/20): that adapts per-sequence fast weights online. The proposed arm performs fold-clean feature-only pretraining, then ordinary causal supervised fine-tuning with fixed weights at inference. The methods can later be combined only as a separately measured interaction.

## Main risks and falsifiers

- Group reconstruction may mostly exploit same-row redundancy and learn representations unrelated to the target.
- Mask-derived event channels and full-input fine-tuning may produce a representation mismatch; the task metric, not pretraining loss, decides whether transfer helps.
- The masked pretraining compute may be inferior to simply spending the same updates on supervised training. The frozen 8-epoch cold control is included for that reason.
- Previous-competition masked-autoencoder reports supplied by the user include fitting on a dataset containing validation sequences. Those reports are inspiration, not clean evidence; this recipe excludes every outer evaluation component from statistics, pretraining and supervised fitting.
- All local component folds have been inspected previously. The proposed 30 paired cells measure training/fold robustness on exposed folds, not pristine confirmation or private transfer.

No local fit, hardware profile, score or competition gain is established by these papers or this source review.
