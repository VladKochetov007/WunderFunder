---
kind: review
status: complete
created_utc: 2026-10-15T08:30:00Z
updated_utc: 2026-10-15T08:30:00Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/TERMINAL_R6.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/collected/attempt_001_http_get_r6/collection_manifest.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/ROOT_CHECKPOINT_VERIFICATION_R1.json
---

# Clean F3 teacher R5 terminal fit review

## Question or claim

Does the collected R5 artifact match the frozen clean F3 seed-3 teacher recipe and fold membership, and what does it establish? This review covers the fit and checkpoint collection only; it does not grade the teacher or treat it as a completed teacher cache.

## Evidence and interpretation

The R6 collection manifest lists 18 objects: 17 result payloads and the kernel log. I streamed each file through SHA-256 and checked its length against the terminal manifest; every entry matched. The collected payload is 35,215,690 bytes and the terminal receipt reports 35,225,185 bytes including collection metadata, below the one-GB bound. The provider status is terminal `COMPLETE`, return code is zero, and no automatic retry was used. R6 changes the failed R5 `HEAD` inventory request to a streaming `GET` that reads headers before downloading; the bounded result transfer still checks lengths and hashes. The R6 collector and helper hashes match their receipts. The helper rejects nonterminal kernels and unsafe result paths, streams under the aggregate byte limit, removes partial files on failure, and uses a physical filesystem with no-overwrite output paths.

The collected trainer spec matches the R4 teacher fit's training arguments, environment, inputs, limits, and artifact contract after normalizing the run-name revision suffix. The H390 source snapshot is unchanged. R5's driver makes an explicit operational change: it admits only the fit and leaves fit-train, fit-valid, and full-valid exports pending because the quota window does not cover them. This is not a change to the fit targets or optimizer, but it means the checkpoint alone is not a usable teacher cache.

The trainer provenance binds split-manifest SHA-256 `78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343`, matching `fold3.json`. Its fit groups equal the manifest memberships: 10,539 `train` groups and 1,368 `valid` groups. The 505 evaluation groups equal the manifest's held-out `valid` groups and are disjoint from fit-valid. There are 501 identical bare integer IDs between fit-train and evaluation, but those IDs refer to different sources (`train.parquet` and `valid.parquet`); the provenance's `train`/`valid` keys preserve that distinction. The worker receipt records complete portable SHA-256 reads of both parquet inputs, matching the spec. Downstream consumers must retain source-qualified identities rather than compare bare group numbers.

The model starts without a parent checkpoint (`init` is empty), uses seed 3, H384×2 GRU, three epochs, fixed checkpoint epoch 2, WMSE with chunk normalization, Muon over all parameters, FP16, and no evaluation during fitting. The propensity input is the fixed `c20260929_prop_f3` model; its provenance binds the same fold3 manifest, trains on the 1,368 fit-valid groups, and withholds the 505 evaluation groups. Thus it is a recorded same-fit-fold weighting ancestor, not an untracked initialization. History records 7,440 optimizer attempts per epoch, 22,320 total attempts, 22,310 completed updates, and 10 AMP-skipped updates. All three fit evaluations are explicitly withheld. The selected checkpoint is `model_ep2.pt`, SHA-256 `d67bed97eaa16f73338ea275c316bf76c785b37f8451aa5cee8fe5daadf2f1b5`; result and history hashes match the collection manifest.

The worker-reported sampled GPU high-water is 5,308,940,288 bytes; its reported worker-process RSS high-water is 5,503,868,928 bytes. These are below the R5 per-worker spec's 12-GiB GPU allocation and 6-GB RSS limits, but the GPU observation is 308,940,288 bytes above a 5,000,000,000-byte threshold. R5 does not declare that 5-GB threshold as its cloud hard cap. The measurement is sampled `nvidia-smi` usage, not allocator-reserved memory or an instantaneous absolute maximum. Do not describe this worker as meeting a 5-GB observed-VRAM target. Root's separate checkpoint receipt reports 16 finite terminal model tensors; I did not load checkpoint binaries in this review.

## Limits and validity

This is a recipe, provenance, split-membership, and artifact-transport review of one F3 seed-3 teacher. It provides no teacher score, student benefit, independent seed replication, or public-transfer evidence. The three required prediction exports and the teacher grade remain pending. R6 validates the provider URL's initial HTTPS scheme, while the shared downloader follows redirects without checking the final scheme; the output URLs are provider-generated signed URLs and no unsafe redirect was observed, but final-scheme checking would be a useful future hardening.

## Decision and next step

The collected fit passes the bounded identity and split audit for downstream export under the declared R5 worker limits. Preserve the above-5-GB observation in resource accounting. Do not call this artifact a completed teacher cache or infer quality until all required exports are validated and graded.

## Source records

- Frozen campaign objective: [CPU student-frontier plan](../../campaign_2026-10-14/cpu_student_frontier_r1/PLAN.md) and [R5 teacher packet](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/README.md).
- Terminal fit and transfer records: [R6 terminal receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/TERMINAL_R6.json), [collection manifest](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/collected/attempt_001_http_get_r6/collection_manifest.json), [trainer spec](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/collected/attempt_001_http_get_r6/results/fit/runtime_specs/fit_clean_t3_f3_muonall_h384_e3_s3_r5.json), and [root checkpoint verification](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/ROOT_CHECKPOINT_VERIFICATION_R1.json).
- Split and weighting ancestry: [fold 3 manifest](../../campaign_2026-09-29/artifacts/splits/fold3.json), [propensity job](../../campaign_2026-09-29/jobs/c20260929_prop_f3.json), and [propensity provenance](../../../artifacts/runs/c20260929_prop_f3/provenance.json).
- Collector repair: [R6 collector](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/collect_terminal_r6.py) and its hash-pinned [R4 streaming helper](../../campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r4/collect_terminal_r4.py).
