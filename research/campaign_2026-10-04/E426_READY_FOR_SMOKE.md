# E426 local smoke and three-arm fit specification

**Status: prepared, not launched.** This document fixes the proposed inputs,
resource envelope, and run contrasts. Passing the bounded smoke is a gate for a
separate launch decision; this document does not authorize the three fits.
Root sources remain unchanged. Use the staged entrypoint
`research/campaign_2026-10-04/staging/e426/scripts/run_event_ema_trial.py`.

## Question and evidence tier

Does aligning the consistency-teacher EMA to observed endpoint-event counts
change training relative to a fresh, same-source constant-clock control, with
the inference/evaluation EMA, batches, objective, and all other settings held
fixed? The third arm permutes the same epoch-specific update-count multiset to
test whether any effect depends on when the event density occurs.

The claim is empirical on the campaign-exposed fold-0 evaluation components
(tier B with a known exposure/model-to-deployment gap). A fold-0 result alone
cannot establish generalization or a public-score gain. The historical combo
run is descriptive only: the **primary control is the new constant-clock fit**
under the same seeded sampler and separated EMA implementation.

## Bounded 20,000-row GPU resource smoke

The trainer rejects `--max-batches` whenever `--teacher-ema-clock` is
`constant`, `aligned`, or `shuffled`, because those arms require a complete
seeded epoch to normalize and align update counts. Do not bypass that guard.
The bounded smoke therefore uses `--teacher-ema-clock shared`, which exercises
the real fold-0 data loader, 20k sequence, batch-32, chunk-1000 recurrent
forward/backward, combo objective, Muon, propensity cache, CUDA prefetch, and
checkpoint path, but **does not test the E426 event clock or separate EMA
roles**. CPU v3 covered those staged paths on synthetic 200-row sequences.

`fit_normalization` reads two batches before training. With
`--max-batches 2 --epochs 1`, the smoke therefore reads two normalization
batches and processes two training batches (40 optimizer updates), plus any
bounded dataloader prefetch. It saves only a noncandidate epoch-0 resource
checkpoint, performs no validation, and does not write to the experiment
ledger.

Common smoke arguments:

```text
--name e426_smoke_shared_f0_seed3_2b_v1
--device cuda --train-source train+valid --train-seqs 10607 --valid-repeat 1
--valid-train-fraction 1 --full-valid --split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json
--no-eval-during-fit --loss wmse --aux-struct 0.3 --aux-next 0.3
--propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 --prop-cache
--valid-true-mask 0.03 --hidden 256 --layers 2 --dropout 0
--batch 32 --chunk 1000 --epochs 1 --max-batches 2
--lr 0.001 --wd 0.01 --seed 3 --optim muon --muon-scope rnn
--ema-teacher 1 --ema-decay 0.999 --eval-ema
--since-weight 32 --since-floor 0.25
--distill c20260929_since_s3_f0,c20260929_since_s4_f0
--distill-alpha 0.5 --distill-boot-h 8 --fast
--precision fp16 --batch-order seeded
--teacher-ema-clock shared
--checkpoint-policy fixed --checkpoint-epoch 0
```

Run this only through the local resource runner with one GPU, 8 CPU cores,
24 GiB host memory, no swap, a PyTorch allocator cap of 4,200,000,000 bytes,
and a total-VRAM hard stop at 4,800,000,000 bytes. Set the fit timeout to
600 seconds. Environment thread controls: `WNN_THREADS=8`,
`WNN_LOADER_WORKERS=2`, `WNN_LOADER_RAM_GB=3`, and one thread each for
OpenMP/MKL/OpenBLAS/Numba/Arrow. Keep the GPU exclusive for this smoke.

Accept the smoke only if it exits successfully with teardown verified, writes
the named noncandidate checkpoint, processes exactly two training batches / 40
optimizer updates, stays below both VRAM limits and 24 GiB host memory, records
no cgroup OOM, and finishes within 600 seconds. The checkpoint must not enter
selection, prediction, or scoring. A pass establishes only bounded
20k-sequence resource viability. It does not cover the event-count pre-read;
that path reads the raw `a6` column for the fit pool (about 955 MB of
uncompressed float32 values before Parquet overhead) and is exercised by the
full event-clock arms. The production fit timeout must retain enough headroom
for this scan and all four epochs.

## Common production recipe

Run three fresh fits sequentially through the same staged entrypoint and
resource runner, one worker at a time. Use the fixed combination objective and
model settings above, replacing the smoke name, `--epochs 1`,
`--max-batches 2`, `--checkpoint-epoch 0`, and `--teacher-ema-clock shared`
with the arm values below. Set `--epochs 4`, omit `--max-batches`, select fixed
epoch 3, add `--save-training-state`, and use `--no-eval-during-fit`. Leave
`--update-rows` and `--detach-rows` unset: both default to the 1,000-row chunk,
preserving the original combo training objective/path. All three arms use
`--batch-order seeded --seed 3`; the E426 sampler requirement is an amendment
shared by the constant control and both interventions.

| Run name | `--teacher-ema-clock` | Per-update consistency-teacher decay |
| --- | --- | --- |
| `e426_f0_constant_s3_4ep_v1` | `constant` | 0.999 on every update |
| `e426_f0_aligned_s3_4ep_v1` | `aligned` | `0.999 ** (events_t * 7460 / total_events_epoch)` |
| `e426_f0_shuffled_s3_4ep_v1` | `shuffled` | Same aligned formula after seeded within-epoch permutation of that epoch's update counts |

Set `--teacher-ema-shuffle-seed 86173` for all arms. The constant arm ignores
the permutation but shares every other option. Student initialization,
normalization order, batch order for each epoch, epoch count, objective,
teacher predictions, propensity weighting, and fixed evaluation EMA must
match. The consistency teacher and evaluation EMA are separate checkpointed
states in all three arms; evaluation EMA uses decay 0.999 after each optimizer
update and alone supplies epoch-3 inference weights. No checkpoint selection
uses validation or holdout. The constant arm is not the old shared-EMA/random
sampler run.

There are 11,940 fit sequences (10,535 TRAIN and 1,405 fit-VALID), batch size
32, and `drop_last=True`: 373 complete batches / 11,936 sequences per epoch.
At 20 update windows per sequence, each arm has 7,460 optimizer updates per
epoch and 29,840 over four epochs. The four dropped sequences may differ by
epoch under the seeded sampler; event counts and their hashes must reflect the
actual update order and dropped tail for each epoch. One full four-epoch arm
gets a 12,000-second timeout under the same 8-CPU / 24-GiB / 4.2-GB allocator /
4.8-GB stop limits. Run arms sequentially; do not restart an interrupted arm
under the same run identity.

## Frozen data and lineage identities

All paths are relative to the repository root. These are the current object
identities to pin again when sealing each runner specification:

| Input | Membership / role | SHA-256 or identity |
| --- | --- | --- |
| `research/campaign_2026-09-29/artifacts/splits/fold0.json` | Fold 0 manifest; 10,535 fit TRAIN, 1,405 fit-VALID, 468 evaluation VALID; corrupt TRAIN group 528 excluded | `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d` |
| `.fit.train` group array | Exact TRAIN fit membership from that manifest | `140171ba3f3fb6b29b5ef32ae9f570140f986055e6abe23ef0367ec9a53248c4` |
| `.fit.valid` group array | Exact fit-VALID membership | `d6f49677ecebf4f27d8ba0a4b402023ee064ca7930bc2d31537dd776b673b9f8` |
| `.evaluation.valid` group array | Scoring-only evaluation membership | `3e4f174cb8dac8a922bdd1fe138c2c281605878115c3289cb554d28c7b206385` |
| `datasets/train.parquet` | 10,607 absolute row-group slots, with corrupt group 528 excluded from fitting | Portable input manifest SHA `4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81`; 29,240,219,011 bytes |
| `datasets/valid.parquet` | 1,873 absolute row groups; fit/evaluation membership from fold manifest | Portable input manifest SHA `2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`; 5,615,509,128 bytes |
| `cache/train_targets.npz` / `cache/valid_targets.npz` | Target arrays and canonical row-group maps used by `Sequences` | `ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9` / `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955` |
| `artifacts/runs/c20260929_since_s3_f0/model_ep3.pt` | Teacher 1 fixed epoch-3 checkpoint | `e3da529a7afe8d78a8cadf4940acacd7ee08c5582e803da7359a7cd9677d3a38` |
| `artifacts/runs/c20260929_since_s4_f0/model_ep3.pt` | Teacher 2 fixed epoch-3 checkpoint | `a1d53aecdef183c94098ffe5fe2422a377a775b022dcc101c48efef5778417e8` |
| `cache/pred_c20260929_since_s3_f0_{train,valid}.npy` | Teacher 1 cached predictions; bytes checked against sidecars | train `2e74e44773d3d6d3e7c48c245a71f281b1659e6f6592ea2a9e4f96d37d19dc`; valid `7a01c040849d9455016dee51a1f9c82c1079be085a0e6be23c99c23bddb3fba9` |
| `cache/pred_c20260929_since_s4_f0_{train,valid}.npy` | Teacher 2 cached predictions; bytes checked against sidecars | train `819ab8ef0b1303b289af3475e173da8e9e7ff404ca7a8d8e7e10a3d443648241`; valid `6aa7b150097ebdf43ddbea8c3a52f8d705ebd2dbd01470e3c45b7eb2dbcfc430` |
| `cache/pred_c20260929_since_s3_f0_{train,valid}.json` | Teacher 1 sidecars: fold hash, epoch, parent SHA, row-group list, prediction SHA | train `da6b0b23fb0f6a063e7ab59d2bd59e4a045f3ba764dd23246c7386a42683e81f`; valid `9ba5a0105e0d5265415cc621e50d90148211c85427a540f0ade06e01bb1665ca` |
| `cache/pred_c20260929_since_s4_f0_{train,valid}.json` | Teacher 2 equivalent sidecars | train `978d949b9d7464c2706051bd351db6c1bbabe4ca3d32256afca3212b5439f5fe`; valid `5f1aae3212e4316d94752b171692a8b8c5fb7d8bd9e141924aa8ddd03e708a07` |
| `artifacts/runs/c20260929_prop_f0/model.pt` | Propensity checkpoint | `b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7` |
| `cache/propcache_c20260929_prop_f0_{train,valid}.npy` | Absolute Parquet row-group-indexed propensity cache | train `811ce4328be739b039bc62ba1f696386e59b168fbe3733be81783a41417da174`; valid `9f3e09f04b9fadfd142d4ecbaba8b4d17d6d1ae04ea40e2cb8d4b64ecd0a718c` |
| propensity cache metadata `{train,valid}.json` | Model/checkpoint identity, source split, row slots and fill counts | train `b90e502506c4c93903057f5e8f15485f8e3af9148024055ecf77110f1f0155f9`; valid `775ee227b15d0b5510c9eeee5012b23d674fed227a343e4581344568eacfb82a` |

For both teachers, sidecars currently name the same fold-0 manifest hash. Their
ordered `row_groups` arrays have these compact-JSON hashes:
`693467c84e6064a80c53f3285fa0e55f4705fa4310911ed516a18d944f8c8a51`
(TRAIN, 10,606 rows) and
`1b21625973fbcf2924b874699f5e265c0526ad69d1aebbd0b43c02cbd547766e` (VALID,
1,873 rows). At launch, the staged trainer must verify each ordered sidecar
array equals `cache/{split}_targets.npz.row_group` and revalidate prediction
bytes; a digest or row count alone does not substitute for that equality check.
The train propensity cache has 10,607 absolute slots and
10,606 filled rows; index 528 remains a hole, so consumers must keep absolute
Parquet indexing. Do not compact or shift this cache. The event source identity
is a per-run seed-3 permutation of the manifest TRAIN list followed by the
fit-VALID list; provenance hashes the ordered `(split,row_group)` source list
and per-update count matrix. Evaluation groups and future labels never enter
that source.

The 12/12 E426 CPU v3 result covers the staged clock and resume loop only on the
200-row synthetic fixture. Its split SHA is synthetic, and its group IDs 0/1 do
not exercise the real TRAIN-528 guard; the fixture's excluded-hole provenance
assertion is metadata-only. Do not cite it as proof of real fold membership.
Production must load the actual manifest above, verify fit counts/membership,
and reject group 528 before reading event counts. This remaining real-data
check is part of the prelaunch review.

## Analysis and stop criteria

Before each candidate arm, seal the exact source closure from the entrypoint,
all input hashes above, argv, package/runtime identity, cgroup limits, timeout,
and unique run name. Capture stdout/stderr, monitor and teardown evidence,
per-epoch event-count schedule hashes, actual batch/update counts, EMA state
identity, checkpoint hash, and input hashes in immutable run artifacts. If a
source/cache hash, sidecar membership, fold count, event-count total, optimizer
update count, EMA identity, memory cap, or timeout check fails, stop the sequence
and preserve the failed run. Do not relaunch it under the same name.

Report all three arm scores with paired component-bootstrap intervals and the
predeclared class-harm guard; intervals crossing zero remain inconclusive. A
lead does not promote E426: require a new seed/component replication before a
generalization claim. No arm, score, or resource smoke was launched while
preparing this document.
