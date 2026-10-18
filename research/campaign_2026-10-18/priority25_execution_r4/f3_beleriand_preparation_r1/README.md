# F3/seed1001 Beleriand pair — preparation only

**Status: UNADMITTED.** This packet prepares one F3/seed1001 paired continuation cell using the unchanged RANK01 trainer and seven-step guarded plan shape from F1. It contains no admission, cell claim, staging, launch, fit, export, grade, or provider mutation. Root must perform fresh ownership/resource/source checks and record a separate admission after the direct diagnostic finishes.

## Bound F3 split and expected schedule

A bounded read-only Python stdlib observation on B read only the exact `/home/grinchenkoeg/computer-vision-homework-20261018/splits/fold3.json` JSON (1,389,282 bytes). Its SHA-256 is `21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f`. The ordered membership counts are 10,539 fit-train groups, 1,368 fit-valid groups, and 505 evaluation-valid groups across 145 evaluation components; exact digests and source metadata are in [F3_SPLIT_METADATA_V1.json](F3_SPLIT_METADATA_V1.json). No parquet, target, or cache payload was read.

Using the frozen batch/chunk recipe: 11,907 fit groups / batch 8 gives 1,488 dropped-last batches per epoch; sequence length 20,000 / chunk 2,000 gives 10 optimizer updates per batch; expected 14,880 updates per epoch, 29,760 after the 2-epoch stage1, and 89,280 across six epochs. This is static arithmetic only. Continuation must stop unless its actual stage1 state records `total_steps=89280`, `next_epoch=2`, and `cumulative_updates=29760`.

## Fixed pair

- `wide`: GRU 512×1, seed 1001, run `rr18_r01_wide_f3_s1001` to fixed epoch 5 / six-epoch endpoint `rr18_r01_wide_f3_s1001_e6`.
- `deep`: GRU 384×2, seed 1001, run `rr18_r01_deep_f3_s1001` to fixed epoch 5 / six-epoch endpoint `rr18_r01_deep_f3_s1001_e6`.
- Both use identical fit membership, raw112 + canonical99 features, fit-only normalization, labels-only WMSE/chunk denominator, FP32 with autocast and TF32 off, deterministic seeded batch order, AdamW, and the exact B source/runtime from the frozen recipe.
- The seven steps and exact expected/guard launch argv are in [F3_BELERIAND_PLAN.json](F3_BELERIAND_PLAN.json). Names and output paths are unique to F3.

The F1 source/runtime preflight is historical (07:57:22 UTC); its passed status and Python 3.13.15 / torch 2.14.0+cu130 / NumPy 2.2.6 / PyArrow 19.0.1 identify the prior environment only. Admission requires a new B check. F2/seed1001 remains the separate issue16 request; current authority records no F3 cell row, which is not a live ownership check.

An [independent review request](INDEPENDENT_REVIEW_REQUEST.md) is included. Review must bind the plan and split-metadata hashes; root owns all staging and admission. This is one exposed fold/seed cell, 1 of 30 required paired cells per arm, and is not a full comparison or convergence claim.
