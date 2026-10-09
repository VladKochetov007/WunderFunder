# H233: frozen C1 conditional readouts

H233 asks whether causal event context and the C1 GRU's recurrent-plus-projection representation can improve its primary pair through small low-rank readouts. The seed-3 C1 model is the fixed parent. Its GRU, input projection, normalization, and full 23-channel output head are frozen; only four separately parameterized 2-output heads learn. They share one detached parent pass for each 2,000-row chunk. No full-sequence hidden-state cache is made.

The four trained controls are hidden-only, additive hidden-plus-context, multiplicative low-rank interaction, and conditional residual over hidden/context/product features. Every final output projection starts at zero, so each endpoint begins exactly at the parent pair. Each control has 22,722 allocated parameters at rank 32; the hidden-only control intentionally leaves its context branch unused. The parent auxiliary outputs remain in each checkpoint but receive no H233 loss.

## Frozen source and objective

The source is our own C1 seed-3 epoch-1 checkpoint (`7d34ee56…ec3f0`), bound to E511 provenance and the fold-0 split. The exact training membership is 10,535 train groups plus 1,405 validation-fit groups; 468 fold-0 evaluation groups are excluded from fitting. The two teacher caches are our own C1-combo seed-3 and seed-4 epoch-3 predictions, each scaled by 0.8125 and averaged. The parent C1 propensity model and its train/valid probability caches supply the original row weights.

For each 2,000-row chunk, use `label = clip(y, -2, 2)`, `teacher = clip(0.8125 * mean(teacher_s3, teacher_s4), -2, 2)`, and `target = 0.3 * label + 0.7 * teacher`. Row weight is the parent `main_row_weights` result: cached propensity squared plus 0.03 with the 0.03 valid-true-mask treatment. The primary-pair WMSE multiplies those row weights by `abs(label)` and divides by that chunk's weight sum, matching E511's main distilled loss. H233 deliberately removes the parent structural/next-change auxiliary *training losses* because the parent is frozen; its 21 auxiliary outputs are still preserved. All four head losses are backpropagated and clipped independently before one AdamW step, so one control's gradient norm cannot clip another's.

The readout fit is two epochs, seed 3, batch 8, chunk 2,000, constant AdamW learning rate `1e-3`, weight decay `0.01`, per-control gradient norm cap 1, FP16 autocast for the frozen CUDA trunk, and FP32 heads. Fixed epoch 1 is the only selected checkpoint. The constant schedule is the declared H233 treatment; it does not claim to reproduce C1's OneCycle optimizer path. Fold 0 is previously exposed development evidence. The export command creates predictions and the fixed mask but does not score, calibrate, or estimate a public result.

## Prepared worker commands

These are inner worker commands only. They are **not admitted**. Root must create the attempt, verify available device/quota and reserve the single GPU lane, then launch through the approved campaign resource/dispatch tools. The script rejects CPU fit/profile/export because those would change the declared CUDA FP16 trunk treatment.

```sh
PYTHONPATH=src .venv/bin/python scripts/train_conditional_readout.py profile \
  --warmup-chunks 2 --steps 8 \
  --output research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/profiles/H233_C1S3_B8_PROFILE_V1.json

PYTHONPATH=src .venv/bin/python scripts/train_conditional_readout.py fit \
  --run-name bd26_h233_c1s3_rank32_f0 \
  --output-dir artifacts/runs/bd26_h233_c1s3_rank32_f0

PYTHONPATH=src .venv/bin/python scripts/train_conditional_readout.py fit \
  --run-name bd26_h233_c1s3_rank32_f0 \
  --output-dir artifacts/runs/bd26_h233_c1s3_rank32_f0 \
  --resume-from artifacts/runs/bd26_h233_c1s3_rank32_f0/model_ep0.pt

PYTHONPATH=src .venv/bin/python scripts/train_conditional_readout.py export \
  --checkpoint artifacts/runs/bd26_h233_c1s3_rank32_f0/model_ep1.pt \
  --split valid \
  --output-dir research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/exports/H233_C1S3_E1_VALID_V1
```

The profile performs two warm-up and eight timed readout updates on one real batch and reports optimizer attempts/completions, finite-loss/gradient checks, CUDA/cuDNN versions, memory, and chunk throughput. It writes one immutable JSON receipt and refuses to overwrite it. The fit writes full parent-plus-head checkpoints at epochs 0 and 1, `model.pt` fixed to epoch 1, and source/group/cache-bound provenance. An epoch-0 checkpoint contains optimizer and RNG state for the explicit restart command. Export streams the frozen GRU over each sequence, writes four FP32 prediction arrays and `selected_mask.npy`, and commits its directory only after all rows and hashes validate. For validation it also compares the recomputed parent anchor to the existing E511 prediction array (maximum absolute tolerance `1e-3`); parity failure leaves a `.partial/export_failure.json` receipt.

Validation export has 1,873 rows in the supplied `valid_targets.npz` order: the 1,405 fit-dev groups plus 468 read-only evaluation groups. Each pair array is approximately 300 MB; the script writes them as memmaps one chunk at a time. The mask excludes the first 99 rows and retains the supplied `is_scored` mask. The export records row-group order and the original anchor prediction SHA for graders. Training-set export is available separately but is not part of the initial bounded decision.

## Resource and evidence limits

The future root-owned worker reservation is 3 CPU cores and 6.5 GB RSS, zero loader workers, one GPU lane, no more than 4.3 GB CUDA allocator use and 5 GB observed whole-device VRAM. Suggested bounded wall ceilings are 15 minutes for profile, 6 hours for the two-epoch fit, and 30 minutes for validation export; root must adjust only with a new recorded admission. No profile, fit, export, Vast rental, or public submission was performed while preparing this code milestone. The managed session has no usable GPU/systemd runtime; correctness tests ran on CPU under one-core affinity and an RSS monitor (5 passed, peak 867,936 KiB). These checks establish implementation invariants, not quality or throughput.

For result interpretation, compare anchor and each readout on the same full fold-0 endpoint using `global_wp_clip2_v1`; preserve the weaker heads because any later blend or teacher-usefulness test is a separate interaction node. Fold-0 is exposed, and no local result is a public-score forecast. Rust CPU inference integration, complete-path latency, and any served endpoint remain separate delivery work.
