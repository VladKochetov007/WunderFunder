# Prospective runtime amendment — teacher transfer (PR #28)

This dated amendment defines the runnable replacement panel for future admission. It does not change the original proposal record or authorize execution. [PR_CP30_FROZEN.md](PR_CP30_FROZEN.md) preserves the prior PR text byte-for-byte; [recipe.json](recipe.json) is unchanged and remains the historic CP30 recipe, not a CLI config.

## Shared executable dependency

Use [runtime PR #32](https://github.com/VladKochetov007/WunderFunder/pull/32), executable source pin [`d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6`](https://github.com/VladKochetov007/WunderFunder/commit/d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6). The runtime source is external to this packet and must not be copied here. Fetch and check out the exact source as shown in the amended [PR.md](PR.md). It supplies raw112 causal Transformer/GRU adapters, streaming state, cache/export/scoring, strict-FP32 trainer, AdamW objective/update handling and remote acceptance procedures.

## Amended panel

| Cell | Teacher fit | Label-control fit | KD-student fit |
|---|---|---|---|
| F0 / s1001 / teacher s101001 | `configs/teacher_t768_l24_f0_s1001.json` | `configs/gru256_labels_f0_s1001.json` | `configs/gru256_kd_f0_s1001.json` |
| F2 / s1002 / teacher s101002 | `configs/teacher_t768_l24_f2_s1002.json` | `configs/gru256_labels_f2_s1002.json` | `configs/gru256_kd_f2_s1002.json` |

Six new fits total: two teachers and four students. No cache or control reuse is claimed in this initial panel. Exact same-fold teacher predictions become prerequisites for KD and later amendment panels only after source/membership/cache hashes and held-out scoring are independently checked. The historical 240-student / up-to-90-teacher CP30 grid is not the current executable panel.

## Schema and objective differences

The old proposal used raw112 inputs with anchored4096 teacher windows, eight student epochs, alpha=0.9 teacher/label mixing and historical model shapes. The executable amendment instead uses raw112, fitting-only normalization over the first 32 fitting sequences, a causal sliding KV cache of 512 rows per teacher layer; the GRU uses recurrent state reset at sequence boundaries, full 20,000-row traversal in 256-row chunks, three fixed epochs, TBPTT detach at chunk boundaries, strict FP32 and AdamW/cosine. Teacher shape is 768×24 with 12 heads and FF multiplier 4; student is canonical GRU256×2. Teacher LR is 1e-4 and student LR 3e-4; weight decay 0.01, clip 1.0 and warmup fraction 0.05. No autocast, TF32, GradScaler, quantization, or output autoregression is enabled.

The objective is additive weighted MSE: `WMSE(clipped labels) + kd_weight*WMSE(clipped teacher outputs) + hint_weight*MSE(hints)`. Label and output KD terms use absolute clipped-label weights; teacher and label-only controls have KD/hint weight zero; KD student uses weight 1.0 and hint weight zero. This replaces alpha=0.9 interpolation. Mask is `is_scored & row>=99`; all rows still advance causal state. Exact behavioral details are defined by the pinned runtime source and configs.

## Current status and gates

No remote checks, full-shape profiles, fit, cache export or grade have been run for this amendment. The receiving operator must bind and review F0/F2 canonical data/split manifests and cell ownership; pass remote unit/distributed/causality/resume checks; record actual device identity; profile each full configuration and approve a finite-headroom strict-FP32 profile; then run the six-fit panel and official scorer. No job, cloud resource, or B300 allocation is implied. No measured B300 result or expected score is claimed.
