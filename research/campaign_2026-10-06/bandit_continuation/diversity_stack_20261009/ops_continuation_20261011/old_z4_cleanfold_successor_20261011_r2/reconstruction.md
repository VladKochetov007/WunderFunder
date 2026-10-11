# Z4-like clean-fold queue successor (R2)

Prepared 2026-10-11 in isolated writer worktree `agent/old-pipeline-queue-successor-20261011-r1`, based on `e8ed2daba3973379a2fecec6d20bc146d228cef5`. This is a runnable packet for root review; it is not an admission. No fit, actual-device profile, prediction/export, queue or provider mutation, upload, score, budget/grader edit, or main frozen-source edit was performed here. The previously integrated R1 recipe and inventory at `old_pipeline_reconstruction_20261011_r1/` remain unchanged.

## First useful local cohort

The minimal same-method comparison is fold 0 and fold 1 with seed 3 carried through each chain: six scientific fits, two deterministic core-crop transforms, and six fixed-epoch exports. It is diagnostic (`n=1` per fold); this does not estimate seed variance or authorize seed-averaged endpoint ranking. The next seed confirmation is a separate root allocation after this screen.

For each fold, the sequence is:

```text
fold-matched own H230 teacher family cache + propensity
  → fresh H230 GRU-256×2 hidden-scope Muon parent, 4 epochs, use @ep3
  → 2-epoch four-family student initialized from that parent, use @ep1
  → audited 31-head to 23-head core crop at @ep1
  → 1-epoch masked GCorr HFT initialized from the cropped student, use @ep0
```

The parent is bound to B16/C1000, WMSE/chunk normalization, Adam-independent hidden Muon, LR `0.001`, weight decay `0.01`, OneCycle schedule with warmup fraction `0.05`, fp16, Muon multiplier `3` and momentum `0.95`, fold propensity power `2` / epsilon `0.03`, main-cache KD alpha `0.5` / bootstrap horizon `8`, EMA teacher (`0.999`, evaluate EMA), and since auxiliary weight `32` / floor `0.25`. Parent and student recompute normalizers on the same fold-eligible `train+valid` members. HFT inherits the exact cropped student normalizer. Full arguments, parents, fixed epochs, checkpoints, source hashes and outputs are in the per-stage contracts and sequence plans.

The fold-specific caches already contain eight own independently trained teachers per fold and the matching family targets; the fold-specific propensity checkpoints are reusable. The outer evaluation members are excluded from every parent/student/teacher/propensity fit. The cache and split identities are bound in [`INPUT_BINDINGS_R2.json`](INPUT_BINDINGS_R2.json). The public all-data Z1c/Z4 weights are not initializers. Recipe comparison does not require deterministic recovery of historical seed-99 weights.

## Stage state and release gates

[`QUEUE_STAGE_INVENTORY_R2.json`](QUEUE_STAGE_INVENTORY_R2.json) records 3/3 eligible reusable inputs per fold (teacher cache, propensity and split binding), no teacher/cache rebuilds or propensity refits, and the exact missing stage counts. Existing fold-0 `bd26_h230_hetero_s3_f0@ep1` → audited crop → `f0_h230_hetero_s3_hft_v1@ep0` and fold-1 `bd26_h230_hetero_s3_f1@ep1` are useful H230 analogues, but they start from combo parents and are not the Z4-method chain.

The H230 B16/C1000 F0 profile contract uses the same parent recipe and fold-specific production inputs, one warmup plus one timed batch, bounded 600-second timeout, two no-gradient normalization batches, no checkpoint, no training-state save, 2 worker CPU threads, and a 4.3 GB allocator cap under the 5 GB observed-VRAM ceiling. Its pass is a profile only; it does not admit a fit. The four already-admitted local H382 F3 e3/e4 fit/export attempts are ordered ahead of the profile. Root should wait for their terminal collection, recheck the empty GPU lane/quota/output paths and data bindings, run this profile, inspect its actual cgroup/VRAM evidence, and only then decide whether to append this cohort. The packet includes the exact profile command and R2 attempt ID in [`PROFILE_CONTRACT_R2.json`](PROFILE_CONTRACT_R2.json) and [`PROFILE_LAUNCH_PACKET_R2.json`](PROFILE_LAUNCH_PACKET_R2.json).

Loader workers are explicitly set to zero in every frozen profile, fit and export environment. The exact imported `wnn.resources.dataloader_budget` was exercised under each intended PYTHONPATH for v3 student, v4 parent and c337 HFT fit/export; all returned `(0, 0)` with the default maximum of four, and AST wiring confirms the caller passes that value to `DataLoader.num_workers`. Hashes and outputs are frozen in [`ZERO_LOADER_VERIFICATION_R2.json`](ZERO_LOADER_VERIFICATION_R2.json) and included in source bindings. No unnecessary source patch was made.

Fit and export workers request 2 CPU cores, 6.5 GB RAM, one GPU slot and at most 4.3 GB allocator VRAM; hard timeouts are 10,800 seconds for fits and 1,800 seconds for exports. CPU crop and cache-handoff helpers are bounded to 50% CPU / 700 MB / 120 seconds. Parent limits recorded in the packet are 5 CPU cores, 18 GB memory-high, 20 GB max, no swap, one GPU worker. No fit ETA is asserted: the profile precedent and nearby historical fit durations are recorded as uncertain and recipe/load-specific in `QUEUE_PLAN_R2.json`.

The existing cache-reclaim helper runs only after successful terminal `predict_run` exports and only when the runtime GPU lock can be acquired nonblocking; a non-ready receipt or lock refusal gates the next GPU stage. This uses the existing handoff and does not alter the runner. There is no Kaggle family-cache upload in this local first cohort.

The exact same-fold/seed strong-new-T3 control ID and full ancestry are still root-bound. As root confirmed, this blocks comparative grading, not admission of the candidate fits after profile PASS. Grade only matched fold evaluation members with `global_wp_clip2_v1`; treat both folds as previously exposed robustness evidence. Do not choose a seed or endpoint from this diagnostic screen alone.

## Frozen payload map

- [`QUEUE_PLAN_R2.json`](QUEUE_PLAN_R2.json): six fit/export sequences, two crop transforms, order, resource limits and post-export reclaim gates.
- [`contracts/`](contracts/): unique R2 attempt contracts. Parent checkpoints are fixed at epoch 3, family students at epoch 1, and HFT at epoch 0. HFT argv explicitly spells out `--sched onecycle`.
- [`sequences/`](sequences/): parent/export, student/export and HFT/export two-step plans for each fold.
- [`source_snapshots/`](source_snapshots/): c337 HFT source archived from its reviewed commit and the crop producer archived from its reviewed commit; manifests bind bytes and provenance.
- [`SOURCE_SNAPSHOT_BINDINGS_R2.json`](SOURCE_SNAPSHOT_BINDINGS_R2.json): source commits, file hashes, R1 recipe link and zero-worker verification.

The exact T3 control, live quota, current queue/cgroup/device state, output absence, actual cache/member hashes and profile evidence remain root-owned admission checks. This R2 packet does not mutate their ledgers or authorize its own launch.
