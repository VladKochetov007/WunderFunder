# Parsed-cache consumer and first matched loss pair

Owner: Luna/max code-only delivery. Status: **root review required; execution is not admitted by this packet**. The implementation is isolated from the active pinned trainer and does not launch a cache build, provider job, or fit.

## Frozen comparison

This is a supervised fresh-initialization teacher screen on the already exposed fold 0. It asks whether a robust per-row loss changes the endpoint contribution when all model, data, optimizer, schedule, seed, and update-count settings are held fixed. It makes no fresh-fold or public-score claim.

The component-preserving membership artifact is [loss_pair_r1_membership_fold0_f03_s20261009.json](loss_pair_r1_membership_fold0_f03_s20261009.json), SHA-256 `dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd`. It binds split manifest SHA-256 `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d` and portable-input registry SHA-256 `90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c`. The deterministic selector used seed `20261009`, fraction `0.03`, and SHA-256 ordering over whole components: 95 of 3,148 eligible components, 350 `fit_train` sequences, 28 `fit_valid` sequences, and all 468 fixed `evaluation_valid` sequences. The cache shape is 350 train sequences and 496 valid sequences (28 fit plus 468 evaluation). Feature, target, and mask payloads total 7,732,440,000 decimal bytes before NPY headers and metadata; the cache builder must still make a fresh disk-headroom admission on its destination.

The fixed outer evaluation is exposed development evidence. All selected fitting components remain disjoint from the 468 evaluation components. Both arms train on the 350 selected `fit_train` groups only; the 28 `fit_valid` groups are retained by the generic cache format but are not silently added to this fit. Evaluation uses the same complete 468-group membership, with the official `global_wp_clip2_v1` scorer after causal predictions are exported. Do not inspect evaluation labels during fitting or select a checkpoint from them.

Both arms use a fresh two-layer GRU (hidden size 256, linear input projection, default GELU model activations), identical seed `20261009`, AdamW, OneCycle, fp16, batch size 32, seeded sequence order, 2,000-row TBPTT chunks, 2 epochs, and the same fixed epoch-1 checkpoint. The cohort path uses `drop_last=False`, so 350 sequences are all consumed: `ceil(350/32) × 10 chunks × 2 epochs = 220` optimizer steps per arm. The trainer keeps `y = clip(target, -2, 2)` and `w = abs(y)` in both losses:

- Control: weighted MSE, `sum(w * (prediction - y)^2) / sum(w)`.
- Treatment: weighted Huber, `sum(w * Huber_delta=1(prediction - y)) / sum(w)`.

`--huber-delta 1.0` is passed to both commands so the serialized recipe differs only in `--loss`. Delta is in the clipped target units. This is a supervised teacher; Huber is not inserted into the trainer's WMSE-only distillation path. The pair tests a loss mechanism, not standalone rank or automatic blend value. If either arm contributes useful error correction, root can later test a fixed, separately evaluated mixture.

The optimizer is held at AdamW for this pair. A later optimizer arm must compare AdamW to this repository's Muon implementation as its own factor: `muon_param_groups` applies 3-block orthogonalized Muon updates to `gru.weight_ih_l*` and `gru.weight_hh_l*`; input projection, output head, and biases stay on AdamW under `--muon-scope rnn`. It must not be described as all-parameter Muon or combined with the loss contrast in this first matrix.

## Cache consumer and bounded source-stage profile

`src/wnn/parsed_sequence_cache_reader.py` opens only a cohort cache whose manifest and one-time validation receipt are pinned. It verifies the cache's source receipt, membership, split/fold, exact six-array schema, hashes, and read-only NPY headers. `train_rnn.py` and `predict_run.py` use those mmap arrays by source-qualified row group. Training copies one sequence into owned writable arrays at a time. The trainer does not parse Parquet or verify the same 34.8 GB of raw Parquet bytes once per loss arm: the separately hash-bound source receipt is checked before the split loader skips only those two now-unused raw files; small split graph/index inputs retain their normal checks. The cache arrays remain pinned and unchanged between the one cohort hash pass and both arms.

The producer now calls `fdatasync` (or `fsync`) and advises `POSIX_FADV_DONTNEED` after each 64 MiB written to each NPY file. This bounds dirty/writeback accumulation by chunks rather than buffering the complete feature arrays. The advice is explicitly advisory. Producer phase receipts record cgroup `anon`, `file`, `file_dirty`, `file_writeback`, RSS/PSS and process I/O before/after source hashing, parsing, writing, and mmap validation. The Kaggle source-stage gate must inspect the actual returned telemetry; do not infer bounded memory from RSS or mmap alone.

The cache contains about 7.73 GB of payload for this membership. Its exact mount path, free-space reserves, and final manifest SHA are unknown until root chooses the Kaggle asset and performs a fresh admission. The CPU-only reader profile then streams the full cache hashes once (with blockwise `DONTNEED`), opens read-only mappings, copies two sequences from each membership role, and records `/proc/self/io`, cgroup anonymous/file/dirty/writeback memory, elapsed time, and the verified receipt hashes. This provides an actual mounted-cache profile before any GPU fit. It does not claim a Kaggle run has happened.

From the repository root, after root has staged the exact pinned source and membership on the selected CPU worker and admitted its output directory, build once with the producer command (do not run this locally):

```sh
PYTHONPATH=src python scripts/parsed_sequence_cache.py build \
  --membership research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/loss_pair_r1_membership_fold0_f03_s20261009.json \
  --expected-membership-sha256 dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd \
  --source-root "$WNN_SOURCE_ROOT" \
  --output-dir "$WNN_CACHE_ROOT" \
  --output-reserve-bytes "$WNN_OUTPUT_RESERVE_BYTES" \
  --checkpoint-reserve-bytes "$WNN_CHECKPOINT_RESERVE_BYTES" \
  --free-reserve-bytes "$WNN_FREE_RESERVE_BYTES"
```

`WNN_SOURCE_ROOT` must be the discovered mount root containing `datasets/train.parquet` and `datasets/valid.parquet`; `WNN_CACHE_ROOT` must be a new directory on the measured destination filesystem. Root supplies the numeric reserve values after live disk/quota review. The cache build verifies complete source hashes once, parses each selected source-qualified row group once, writes/hash-binds the arrays, and records actual parse/decode/write telemetry. No assumption is made about the Kaggle input slug or nested mount path.

After the cache is built and its manifest SHA is admitted and frozen, run this CPU-only profile in that same provider environment before GPU retry. It hashes the cache once for the reusable validation receipt and measures actual cache replay:

```sh
PYTHONPATH=src python scripts/profile_parsed_sequence_cache.py \
  --manifest "$WNN_CACHE_ROOT/manifest.json" \
  --expected-manifest-sha256 "$WNN_CACHE_MANIFEST_SHA256" \
  --validation-receipt "$WNN_CACHE_ROOT/reader_validation.json" \
  --profile-output "$WNN_CACHE_ROOT/reader_profile.json" \
  --per-role 2 --seed 20261009
```

The profile emits the validation receipt SHA-256 and cache key. Record those alongside the profile SHA, producer manifest/receipt hashes, source bytes and hashes, selected membership SHA, observed disk headroom, worker memory limit, and actual device inventory. Root/independent review decides whether the cache staging and source-stage resource profile support proceeding to the paired fits.

## Frozen command template for the two GPU arms

The following are the exact matched trainer arguments. Root must replace the four cache pins/paths with values from the reviewed provider cache receipt and must independently admit the two fits. The commands deliberately include no distillation, ensemble parent, or teammate artifact.

```sh
COMMON=(
  --device cuda --arch gru --cell gru --enc linear --hidden 256 --layers 2
  --train-source train --valid-fold 0 --full-valid
  --split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json
  --parsed-cache-manifest "$WNN_CACHE_ROOT/manifest.json"
  --parsed-cache-validation-receipt "$WNN_CACHE_ROOT/reader_validation.json"
  --expected-parsed-cache-manifest-sha256 "$WNN_CACHE_MANIFEST_SHA256"
  --expected-parsed-cache-validation-sha256 "$WNN_CACHE_VALIDATION_SHA256"
  --expected-parsed-cache-key "$WNN_CACHE_KEY"
  --seed 20261009 --batch-order seeded --batch 32 --chunk 2000 --epochs 2
  --optim adamw --sched onecycle --lr 0.001 --wd 0.01 --clip 1.0
  --precision fp16 --loss-norm chunk --huber-delta 1.0
  --checkpoint-policy fixed --checkpoint-epoch 1
  --no-eval-during-fit --save-training-state
)
PYTHONPATH=src python scripts/train_rnn.py --name loss_pair_r1_wmse "${COMMON[@]}" --loss wmse
PYTHONPATH=src python scripts/train_rnn.py --name loss_pair_r1_huber_d1 "${COMMON[@]}" --loss huber
```

After both terminal checkpoints pass their training-state/checkpoint SHA checks, export the entire fixed evaluation group for each arm through the mmap consumer. The resulting prediction sidecars must bind the exact row-group order and cache/split/membership/checkpoint hashes:

```sh
PYTHONPATH=src python scripts/predict_run.py loss_pair_r1_wmse valid \
  --device cuda --parsed-cache-manifest "$WNN_CACHE_ROOT/manifest.json" \
  --parsed-cache-validation-receipt "$WNN_CACHE_ROOT/reader_validation.json" \
  --expected-parsed-cache-manifest-sha256 "$WNN_CACHE_MANIFEST_SHA256" \
  --expected-parsed-cache-validation-sha256 "$WNN_CACHE_VALIDATION_SHA256" \
  --expected-parsed-cache-key "$WNN_CACHE_KEY"
PYTHONPATH=src python scripts/predict_run.py loss_pair_r1_huber_d1 valid \
  --device cuda --parsed-cache-manifest "$WNN_CACHE_ROOT/manifest.json" \
  --parsed-cache-validation-receipt "$WNN_CACHE_ROOT/reader_validation.json" \
  --expected-parsed-cache-manifest-sha256 "$WNN_CACHE_MANIFEST_SHA256" \
  --expected-parsed-cache-validation-sha256 "$WNN_CACHE_VALIDATION_SHA256" \
  --expected-parsed-cache-key "$WNN_CACHE_KEY"
```

Store both checkpoints, per-epoch update/loss/AMP-skip/resource logs, exact trainer source hash, recipe argv, resumable epoch state, prediction bytes/sidecars, and the root grader's per-target and pooled `global_wp_clip2_v1` results on SSD. Report actual parse time, cache verification/replay time, fit wall time, peak CUDA allocation, RSS/PSS and cgroup counters, checkpoint/prediction bytes, and all-in provider cost separately. This packet provides no training-time or cost estimate beyond the exact 220 optimizer steps per arm; hardware timing and serving value remain unknown.

## Admission and known gaps

This is an exposed-fold low-fidelity mechanism screen, not a claim that either arm improves public performance. The root owns the actual provider quota check, disk-reserve choice, mounted asset identity, source stage, fit admission, result collection, and independent grading. The reviewer should verify the actual Kaggle CLI bootstrap includes this frozen trainer/consumer source and can write the specified outputs before any GPU dispatch.

Code tests use tiny synthetic Parquet producer fixtures only; no real cache build, full Parquet read, Kaggle CPU profile, GPU fit, prediction export, or score was run here. The producer's `fdatasync`/`DONTNEED` cadence is a bounded-write mitigation, not a measured guarantee; accept only actual `file_dirty`, `file_writeback`, RSS/PSS, I/O and free-disk receipts from the Kaggle source stage. An optimizer comparison, TCN/history change, fresh fold, teacher blend, and student/distillation experiment are deliberately outside this first pair.
