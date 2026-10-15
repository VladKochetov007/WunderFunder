---
kind: experiment
status: frozen recipe; no profile or fit admitted
created_utc: 2026-10-15
updated_utc: 2026-10-15
hypothesis_ids: [H-STATE-INIT-20261014]
experiment_ids: []
evidence_paths: [recipes/CONTRACT_R1.json, recipes/FIT_SUBSET_R1.json]
---

# Initializer-family screen on F0

This packet freezes a small comparison of learned initial states around the already-trained clean-T3 H256×2 model. It was prepared from source and metadata after the user asked to pause new research; it has no executed preflight, profile, fit, prediction export, or grade. The source files and commands are preserved so the work can resume without reconstructing the recipe.

## Frozen comparison

The anchor is CP5 (`model_ep5.pt`) from `clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1`, checkpoint SHA-256 `e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda`; its provenance SHA-256 is `bdf643043884b1344564b636a2972ad52b8202ee30cf22598b1afd90405dd961`. This parent is a mature six-epoch, alpha-.9 homogeneous clean-T3 continuation initialized from `c20261003_combo_s3_f0@ep3`. The screen tests the value of three learned initial-state parameterizations on top of that fitted parent; it is not a fresh-weight student or an isolated test of distillation.

The three arms are a learned constant, a projection of the first normalized row, and a 64-hidden-unit MLP over the existing 99-row warmup prefix (rows 0–98). The frozen GRU backbone weights stay byte-for-byte unchanged. All arms use the same parent mean/std buffers, and the predictor state is reset at each sequence. This is the existing 99-row prefix mode; no new prefix-length variant was added.

The target remains the frozen homogeneous T3 cache: an FP32 arithmetic mean of the three CP2 H384 teachers multiplied by `.8125`, then consumed as `0.1*clip(label,2)+0.9*clip(T3,2)`. The teacher checkpoints, cache files, split, propensity model and sidecars are pinned in the contract. Teacher fitting excludes the fixed F0 evaluation groups. The initializer subset is drawn without labels from 28 **complete fit-only components**, including every train and valid source node in each selected component: 86 train sequences plus 18 valid sequences, 104 total. All 468 groups across 141 components remain the same repeatedly exposed F0 evaluation panel.

Each arm uses two FP32 epochs, batch size 1, 20,000 rows per sequence and 2,000-row checkpointed chunks. It applies the same seeded `EpochOrderSampler` ordering. Each complete sequence contributes one optimizer update; there are 104 updates per epoch and 208 per arm. The weighted squared-error objective covers rows 99–19,999 and is normalized by the whole sequence's weight mass. The hidden-state graph is retained across chunks, with non-reentrant checkpointing; the method does not detach the initializer gradient after an early chunk. AdamW uses LR `.001`, weight decay `.01`, and global gradient norm cap 1. No evaluation is used during fitting.

The loss uses the source-pinned propensity cache (`main_valid = valid*(p^2+0.03)`), the true scored-row mask, and label magnitude `abs(clip(label,2))`; the resulting weights are normalized by the complete sequence mass. The CP5 parent was trained with `.3/.3` auxiliary weights, while this initializer-only update contract specifies the main complete-sequence weighted squared-error term and no separate auxiliary-head losses. That objective difference is explicit.

The two-epoch outputs are fixed at zero-based epoch 1. Full-valid FP32 feature-only predictions are to be graded with the unchanged `global_wp_clip2_v1` evaluator against the exact CP5 parent. Report the fixed position bins `[99,200)`, `[200,1000)`, `[1000,5000)`, `[5000,10000)`, and `[10000,20000)`. F0 has been repeatedly exposed, so this is a screening result if run, not fresh cross-validation or serving evidence.

## Commands and remaining checks

All commands below are future stage invocations only. The preflight is read-only and source-bound; profile and fit must be launched through the project's shared resource controller after resume and fresh admission, not as an unbounded direct shell job. The runner verifies source binding and producer data before model work, fixes output paths in the contract, rejects existing outputs, and requires the exact profile-receipt SHA before fit-export.

```sh
python scripts/run_initializer_family_f0_r1.py \
  --phase preflight \
  --contract research/campaign_2026-10-15/initializer_family_f0_r1/recipes/CONTRACT_R1.json \
  --receipt research/campaign_2026-10-15/initializer_family_f0_r1/profiles/STATIC_PREFLIGHT_R1.json

python scripts/run_initializer_family_f0_r1.py \
  --phase profile \
  --contract research/campaign_2026-10-15/initializer_family_f0_r1/recipes/CONTRACT_R1.json

python scripts/run_initializer_family_f0_r1.py \
  --phase fit-export \
  --contract research/campaign_2026-10-15/initializer_family_f0_r1/recipes/CONTRACT_R1.json \
  --profile-result-sha256 <sha256-of-the-successful-profile-receipt>
```

The contract bounds the resource profile to 600 seconds and the fit/export phase to 7,200 seconds, one worker CPU, 5,000,000,000 bytes of host memory, no swap, 3,500,000,000 bytes CUDA-reserved memory, and observed VRAM strictly below 5,000,000,000 bytes. These are ceilings, not evidence that the job fits; the actual profile and fresh admission are still required. After a successful fit/export and source-bound validation, grade each arm with `scripts/score_fold0_panel.py <run> --epoch 1 --reference clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1@ep5 --out <unique-grade-path>`. No grade is present in this packet.

## Files and evidence status

The [recipe index](recipes/README.md) links the source binding, contract, subset and exact implementation hashes. The contract SHA-256 is `97af50e9c815cc86de954b86ca39ee641f773758ee5e691a6a2f95ea698291a7`; the source-binding SHA-256 is `4e7f2bfa5c6525fadf7764dd4ebbc915eef5398e2f87e82a312ff54dc2a5dac3`. Ruff formatting and lint checks passed for the two new scripts. The runner's preflight and all GPU-dependent phases remain unrun; no result or resource claim is made.
