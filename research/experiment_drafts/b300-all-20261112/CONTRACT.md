# B300 wave execution contract

This wave carries PR81's ten implemented mechanisms forward with an explicit
single-B300 placement amendment for the seven Transformer cards. It leaves the
original PR81 packet, its static receipts and its historical results unchanged.
The new placement has its own source, config, receiver and profile identities.

## 1. Data and primary validation

- Use the supplied official `train.parquet` fitting file and independent
  `valid.parquet` evaluation file, preserving whole sequences.
- Bind the split receipt SHA256
  `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
- Expected membership is 10,325 training sequences and 1,873 validation
  sequences; score all 3,528,273 eligible validation rows.
- Fit normalizers, PLE quantile knots, teachers and other preprocessing only on
  training members. Validation labels do not enter preprocessing or model ancestry.
- Use `global_wp_clip2_v1` and the fixed E4 Transformer or E6 GRU endpoint.
  Score full validation after every epoch.
- Validation is previously exposed development data. Do not call it a clean
  holdout or claim public/private transfer from it.
- `seq_ix` is shuffled and is not chronology. Additional internal time splits
  need actual timestamp/order evidence and a frozen source-qualified manifest.

## 2. Sources and precision

- Cards 01?07 use a clean detached Git worktree at PR75
  `8626000a4b965763a70011a0291a891aa6535d7e`.
- Cards 08?10 use a clean detached Git worktree at PR79
  `67c3bd818ea243177eeb1dc789cd3d145eba0e81`.
- Packet adapters remain isolated in `runtime/impact81`; canonical `src/` and
  `scripts/` are not patched.
- Transformer compute uses BF16/BF16x9 forward with FP32 master weights,
  gradients, loss and optimizer state. Preserve
  `CUBLAS_EMULATE_SINGLE_PRECISION=1`.
- GRU cells keep PR79's FP32/AdamW recipe and the explicit B300 device requirement.
- The Transformer placement changes from world size 2 / local batch 128 to
  world size 1 / batch 256. LR, weight decay, epochs, context, accumulation and
  optimizer stay fixed. Each one-GPU update receives the same 256 sequence IDs
  as the former two-rank update; floating-point reduction order differs, so the
  amended recipe uses fresh controls.

Runtime admission requires exact `git rev-parse HEAD`, clean tracked `src`,
`scripts` and `utils.py`, no untracked Python modules there, and admission before
any `wnn` import. A source tar alone cannot pass the Git identity check. The
receiver supplies sparse clean Git worktrees containing the pinned trees and
required blobs; compare their source manifests with the archived capsules.

## 3. Frozen recipes and schedule

| Family | Cards | Model | Placement | Endpoint | Objective and optimizer |
|---|---|---|---|---|---|
| Transformer | 01?07 | H768/L24/12 heads/FF4/context 1024 | 1 B300, batch 256, accumulation 4, chunk/horizon 256 | E4 | PR75 WMSE, Muon/AdamW roles, LR `3e-4`, WD `0.01`, clip 1, 5% warmup and cosine floor 0.1, BF16/BF16x9 |
| GRU | 08?10 | K16/H256/L2, chunk 1000 | 1 B300, batch 32, accumulation 1 | E6 | PR79 WMSE, AdamW, LR `3e-4`, WD `0.01`, clip 1, 5% warmup and cosine floor 0.1, strict FP32 |

Transformer global sequence batch is 256; four accumulated chunks make 1,024
sequences per optimizer update. The fixed full-data schedule is 820 updates per
epoch and 3,280 updates over E4. Preserve optimizer, scheduler, model, RNG and
data-position state at safe checkpoints.

## 4. Fit inventory and queue

The exact roster and stage order are in
[`WAVE_20261112.json`](WAVE_20261112.json):

- Cards 01?10 each have fresh control and treatment fits at seeds 204101 and
  204102: 20 controls + 20 treatments = 40 planned fits.
- No old control, teacher or pretraining fit is reused.
- First cohort: cards 01?02 at both seeds, eight cells across eight B300s. Each
  mechanism receives a complete two-seed paired screen.
- All ten distinct mechanisms remain in the queue, including card 08's
  perturbed carried-state branch and card 10's raw-label tail-class head.
- Card 10 is conditional on its frozen train-only tail-prevalence gate. If it
  fails, preserve the count receipt and omit both card 10 arms.
- Later cohorts follow the five stages in the wave manifest. Preserve every
  pending cell at the shared cutoff; do not imply that all 40 completed unless
  receipts show it.

The external campaign `CLOCK_R1.json` supplies absolute fit and wave cutoffs.
Do not reset it or extend an attempt beyond it.

## 5. Receiver gates and per-cell profiles

1. Verify exact runtime pins, packet/runtime source hashes, data hashes,
   official split membership and train-only preprocessing ancestry.
2. Run one `prepare` stage and receiver `check` per family. Preserve failures as
   source/runtime evidence, not as negative mechanism results.
3. Capture a full-shape profile with at least 50 updates for every selected cell,
   using the exact runtime, config, placement and data. Inspect update time,
   reload proof, peak VRAM, host RSS and disk headroom.
4. Record operator approval for each exact profile receipt. Do not auto-approve
   or start a fit from a profile alone.
5. Fit to E4/E6 with full-validation scoring each epoch, per-target metrics,
   primary/auxiliary losses, LR groups, preclip gradient norm, clipping and
   nonfinite/skipped counts, throughput, elapsed time and resource peaks.
6. Export, score and compare complete paired endpoints. Require source, config,
   preprocessing and split identity. Do not replace E4/E6 with best-valid.

Every failed or partial attempt retains its evidence. Resume only after a
specific interruption is diagnosed, from a safe checkpoint with identical
runtime, packet, config, data and preprocessing, and within the shared cutoff.
Keep source, config, preprocessing, checkpoint, prediction and result manifests.

## 6. Decision and interpretation

Report each seed delta and per-target result. Two seeds on one exposed official
validation split do not establish significance or estimate training variance.
Component bootstrap describes evaluation sensitivity of fixed predictions
only. Preserve mixed, negative and incomplete outcomes. The PR81 practical
gain threshold is an allocation heuristic, not an automatic promotion rule.
