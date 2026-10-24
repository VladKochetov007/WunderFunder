---
kind: incident
status: inconclusive
created_utc: 2026-10-24T12:39:37Z
updated_utc: 2026-10-24T12:41:33Z
hypothesis_ids: [H-AGE-READOUT-20261023-R1]
experiment_ids: [E-AGE-READOUT-20261023-R1]
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/kaggle_age_r4/TERMINAL_R4.json
  - research/campaign_2026-10-24/resume4h_r1/kaggle_age_r4/KERNEL_PUSH_R4.json
  - research/experiment_drafts/b300-fast-local-kaggle-20261023/LOCAL_CACHE_VERIFICATION_R1.json
  - research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json
---

# R4 cache reconstruction and six-fit feasibility

Written 2026-10-24 12:39 UTC from the version-4 terminal receipts and source-only inspection. No cache payloads, labels, fitted results, Kaggle inputs or provider state were read for this note. The six-fit panel below is an estimate from the frozen recipe and source, not an execution result.

## What version 4 established

Kaggle accepted private kernel version 4 at 12:21:59 UTC, reported it running at 12:22:20, and reported `KernelWorkerStatus.ERROR` at 12:36:39. The driver ran 855.521 seconds against its 900-second reservation. Archive hash, safe extraction, overlay application, the 66-module source map and raw-source identity checks passed. The driver then timed out while waiting on the profile child.

Recorded pre-child phase times were 24.298 seconds to hash the bundle, 8.258 seconds to extract and apply the overlay, and 8.546 seconds for source and raw-source validation. The 814.419-second difference between total elapsed time and those three phases includes unmeasured driver setup and the child interval; it is not a cache-rebuild measurement. `run_profile()` only records `profile_seconds` after the child returns. The timeout interrupts that assignment, so the final driver receipt has no child duration or exit code.

The profile child emits its profile JSON only after cache loading, all three microfit arms, and evaluation finish. The final profile JSON is absent, stdout and stderr are empty, and the driver has no incremental child-phase receipt. Source inspection also shows that the driver's `target_cache_opened` output is set to the constant `false`, not sampled from the child. Therefore this run cannot tell whether cache reconstruction completed, whether the fit loop began, or how many of the admitted 360 engineering updates ran. It did not produce a complete profile, scientific fit endpoint, or quality grade. The timeout is execution evidence, not a negative scientific result.

The exact files, hashes, bounds, chronology, and this telemetry limitation are in [TERMINAL_R4.json](TERMINAL_R4.json). Raw small terminal outputs remain under [terminal_r4/](terminal_r4/); [KERNEL_COLLECTION_R4.json](KERNEL_COLLECTION_R4.json) records the strict output filter and confirms no cache or weight payload was requested. No retry is authorized by [KERNEL_PUSH_R4.json](KERNEL_PUSH_R4.json).

## Cache size and producer path

The existing local cache verification receipt reports exact full-file hashes and NPY headers for fold 0. Its three files total **16,024,320,384 bytes**: 11,568,960,128 bytes of fit sequences (1,236 × 20,000 × 117 float32), 4,380,480,128 bytes of evaluation sequences (468 × 20,000 × 117), and 74,880,128 bytes of cached parent predictions. Fold 1's corresponding three files total **15,985,120,384 bytes** (1,243 fit sequences, 457 evaluation sequences). Both folds together require 32,009,440,768 bytes of full cache and parent arrays. The local files still exist at the receipt's paths with matching recorded sizes, inode numbers, and modification times; this task used metadata-only `stat` checks.

The local receipt [LOCAL_CACHE_VERIFICATION_R1.json](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/b300-fast-local-kaggle-20261023/LOCAL_CACHE_VERIFICATION_R1.json) records full payload hashes and headers against the frozen source-cache recipe. Its limits say that labels were not decoded or scored and that semantic membership/ancestry checks remain the canonical loader and independent grader's responsibility. The private 1.14-GB transport bundle deliberately excludes these full cache arrays and raw Parquet; [PROFILE_BUNDLE_PREPARE_R2.json](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/b300-fast-local-kaggle-20261023/PROFILE_BUNDLE_PREPARE_R2.json) says `actual_full_cache_arrays_included=false`, `raw_parquet_included=false`, and `target_cache_included=false`. Raw train/valid data are separate Kaggle inputs.

The portable loader in [age_readout_portable.py](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/age_readout_portable.py) reconstructs the complete F0 arrays from the frozen membership and transported suffix columns. [age_readout_portable_builder.py](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/age_readout_portable_builder.py) reads each referenced raw Parquet row group, streams its 112 raw columns together with the transported fit/evaluation suffix, writes every full 117-column array, and checks the resulting full-file SHA. Evaluation rows get the declared zero columns and parent predictions; sidecars and membership are then revalidated. For F0 that path logically materializes 1,704 raw sequences (about 15.27 GB of uncompressed 112-wide float32 values), writes about 15.95 GB of fit/evaluation arrays, reads the suffix arrays, and hashes the outputs. Those byte volumes follow from source shapes; no remote throughput is measured.

## Six-fit panel cost and route options

The frozen recipe fixes six new readout fits: A/G/F on fold 0 seed 1001 and fold 1 seed 1002, with three epochs per fit, 8,192-row batches, retained final tails, strict FP32, and unchanged objective and evaluation. It reuses the two existing parent endpoints and performs no new GRU or teacher fits. `iter_fit_batches()` in [age_readout_cohort.py](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/age_readout_cohort.py) streams batches over the complete ordered sequence membership. Using the pinned sequence counts, this is 9,054 optimizer updates per fold-0 arm plus 9,105 per fold-1 arm, **54,477 updates total**, before evaluating all 925 evaluation sequences for each of three arms (55.5 million prediction rows). The admitted profile sampled only 16 fit and 8 evaluation sequences per head and capped engineering work at 360 updates; it cannot supply full-panel timing.

There is no current source evidence that all six fits can finish within roughly two hours. The only current remote timing is a timeout during the one-fold CPU reconstruction/profile path; it gives neither a reconstruction duration nor a completed head timing. The scientific CUDA path is also not ready: the R4 portable loader and CLI reject CUDA, while the GPU R2 packet permits CUDA only for an engineering profile and remains unadmitted. Available GPU quota is not a substitute for an executable, source-reviewed route or wall-time measurements.

Reusing full local caches is the shortest credible route only if root can attach or transfer exact cache payloads and sidecars, validate their actual remote bytes, confirm provider disk and upload limits, and admit a source path that consumes them through the canonical membership/metadata checks. The existing local receipt is strong byte provenance, but the receipts reviewed here show no Kaggle dataset containing the full cache arrays. Transfer time, remote disk capacity, cache-mapping adaptation cost, full-fit throughput, and grade/collection cost remain unknown.

A selected-row cache cannot stand in for the full scientific panel: the frozen fits train over all 1,236/1,243 fit sequences and evaluate all 468/457 evaluation sequences. It is appropriate only for the already-authorized engineering microprofile. A streamed full-cache reader could preserve the objective only if it yields every original row in the same ordered batches, preserves per-sequence weight semantics and normalization, and passes exact parity against the current full-array iterator. That implementation and parity evidence do not exist here; adding it now would be a new code path, not a demonstrated shortcut.

## Minimal future instrumentation

For any separately authorized diagnostic run, the smallest useful source change is to atomically write a bounded phase receipt immediately after: (1) `load_portable_caches()` returns with actual fit/evaluation payload hashes and membership checks, and (2) the profile loop returns. Record elapsed time, phase name and completed optimizer updates; have the driver preserve that receipt on timeout. Do not infer phase progress from the current constant `target_cache_opened=false` or the driver's default `profile_wall_seconds=0.0`. This instrumentation would diagnose the timeout but would not reduce reconstruction work or authorize another run.

## Feasibility conclusion

The full six-head panel is **not currently cost-admissible from the evidence on hand**. The data and recipe are available, but a complete remote execution route within the remaining time is unmeasured; the only admitted CPU attempt timed out before emitting a phase result, and the available CUDA candidate does not support scientific fits. Root should consider another execution only after a byte-verified full-cache transfer or a bounded end-to-end reconstruction timing, an executable device path that retains the frozen data/fit semantics, and a realistic budget for all six fits, prediction collection, and grading. No fit or provider action is proposed by this note.
