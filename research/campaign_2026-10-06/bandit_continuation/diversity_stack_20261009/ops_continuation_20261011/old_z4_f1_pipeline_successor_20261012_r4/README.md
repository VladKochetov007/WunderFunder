# F1 old-Z4 cleanfold continuation (R4)

Prepared 2026-10-11 in isolated worktree `agent/oldz4-f1-continuation-20261012-r1` at base `f6a5db756c9b34ef0c3a200309b4a883aeeebfa5`. This is a frozen code/recipe handoff, not a launch authorization. No GPU fit, export, provider action, queue/ledger mutation, public upload, grading, or frozen-source edit was performed.

## First two runnable stages

The exact R3 F1 seed-3 H230 Muon parent recipe is carried forward unchanged under fresh R4 run/attempt IDs. It trains four epochs on fold-1 eligible train+valid members, saves fixed `@ep3`, and uses the own F1 eight-teacher cache plus the existing fold-1 propensity. The first export is full-valid inference for all 1,873 valid sequences; the fixed score mask selects the 457 outer evaluation groups. The parent has no exact eligible completed F1 seed-3 counterpart, so reuse is not possible. F1 `parent-fit` and `parent-export` are runnable after root's current resource/profile and quota admission.

The next two contracts preserve the R3 same-parent, two-epoch family-head student recipe and full-valid export. The student starts only from this exact F1 parent `@ep3`, and saves fixed `@ep1`. Its inputs and outer mask are the same fold-specific source-qualified split.

See [`F1_STAGE_PLAN_R4.json`](F1_STAGE_PLAN_R4.json) for the four stage order and per-stage gates; executable argv is in [`contracts/`](contracts/).

## Membership and ancestry

The audited F1 split has 10,533 `train.parquet` fit groups and 1,416 fold-eligible `valid.parquet` fit groups, 11,949 source-qualified fit members total. It reserves 457 outer evaluation groups, which are excluded from fitting. R3's old prose said “10,606 train.parquet groups”; that was stale and is corrected here. Canonical source-qualified fingerprints and exact split digest are in [`INPUT_BINDINGS_R4.json`](INPUT_BINDINGS_R4.json).

All eight own independently trained F1 teacher sources and their checkpoint hashes are recorded in the input binding, along with the exact F1 propensity model and provenance hashes. The immutable cache metadata file is 842,417 bytes and SHA256 `8bd3854d867c47eacbeed56528ea9712f7598cc7eb21a46f5c85e7a0f2fba703`. Its estimated payload is 13,382,880,000 bytes; payload members were not rehashed during packet preparation. Root's admission must validate the actual cache consumer/member bindings before the parent fit. No teacher/cache rebuild or propensity refit is needed.

No all-data public Z1c/Z4 checkpoint or F0 model initializes this chain. The parent/student normalizers are recomputed only on the fold-eligible fit members; later HFT inherits the exact cropped student normalizer in the separate downstream packet.

## Recipe identity and sources

The parent and student argv are copied from the R3 F1 contracts with only run IDs, attempt IDs and binding/output references changed. Their scientific args, fixed epochs, targets, optimizer/scheduler, propensity settings, and source implementations are unchanged. The R3 recipe fingerprint is `bc7bb99844804def1dacd0c6c2f1afe6f56d008cb1a0fb9d550c4f1a313e58c2`. Immutable source snapshots remain under `/home/vlad/development/wunder/wnn_old_pipeline_reconstruct_20261011_r1/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/old_z4_cleanfold_successor_20261011_r3` with source-map SHA256 `f0181c3715812eef2582ed1d62879089e35db9c403605a157c1466c9d7beabbf`; no source files are copied into or modified by R4.

All loaders remain explicitly zero-worker. The fixed full-valid exports are float32 `[1873, 20000, 2]` (299,680,000 payload bytes). Export contracts bind the exact v3/c337 runner path and source hash as in the audited R3 source map.

## Resource evidence and gates

Historical fold-0 same-recipe analogues completed parent fit in 5,634.15 seconds (peak RAM 5,852,327,936 bytes; peak observed VRAM 1,752,170,496 bytes) and student fit in 2,098.50 seconds (peak RAM 6,122,377,216 bytes; peak observed VRAM 2,666,528,768 bytes). These are measurements from fold 0, not F1 duration promises or admission evidence; exact runtime receipt hashes and event-scope qualification are in [`MEASURED_F0_ANALOGUES_R4.json`](MEASURED_F0_ANALOGUES_R4.json). The parent+student analogues total 7,732.65 seconds of fitting, excluding exports, setup, and cleanup. Both attempt-specific memory-event deltas had zero OOM/kill/max events. The parent-hierarchy counters changed during the parent interval, but include other children and cannot be attributed to that run. Actual F1 launch still requires current root profile/quota admission and live aggregate-resource checks.

All four contracts are `prepared_code_only_not_admitted`. Parent fit requires profile/admission and final source/membership checks. Parent export and student fit require successful CP3 terminal validation. Student export requires successful CP1 validation. No fit may start solely because a predecessor process exited zero: root must validate checkpoint/provenance/finite state and source-qualified membership first. Fold 1 is exposed robustness evidence and must not be presented as a new holdout.
