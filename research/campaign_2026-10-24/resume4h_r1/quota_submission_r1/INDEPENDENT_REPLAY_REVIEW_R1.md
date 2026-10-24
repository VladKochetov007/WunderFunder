# Independent static review — quota package replay R1

## Finding

**PASS for the bounded feature-only replays, conditional on root admission and the stated cross-replay gate.** I found no remaining static mismatch in the exact selection, plan, ZIP identities, rosters, or current runner. This review does not certify native import, callback behavior, runtime, memory peak, or predictive quality; no replay or grade was run. The plan currently says `prepared_for_independent_review`, while the runner requires `status="admitted"`; root must issue the admitted plan and its new hash before execution.

## Exact identities checked

- Runner: `research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/replay_package.py`, SHA-256 `2a491353e2523f82c398ffcfca32168de0fcbce30a035b60524afd100981a43a`.
- Frozen selection: [SELECTION_R1.json](SELECTION_R1.json), SHA-256 `48c017d13884d8294cdb21f19cef6be1df4663fa4fc54eb592e651a21bcaff07`.
- Draft plan: [REPLAY_PLAN_DRAFT_R1.json](REPLAY_PLAN_DRAFT_R1.json), SHA-256 `8414418a9875d069a356b829d3be0e473ac5d2c3656fbb43b02bfe8548bd0a57`.
- Package build recipe: `submissions/quota_20261024/BUILD_RECIPE_R1.json`, SHA-256 `6838151cad4dea2127a84e00dfe17c6932860164c03d94df7074c7b79ca22954`; the two package receipts bind this recipe.
- The four runner/helper source pins match: runner, `native_runtime.py`, `check_submission_abi.py`, and `utils.py`. Both fold split-manifest hashes match the current files.
- I independently recomputed the deterministic panel from the split JSON metadata. Recomputed members match both selection and plan exactly: 64 entries, 32 each from F0 and F2, 32 distinct components within each fold, 64 unique row groups, and 64 unique `seq_ix` values.

## ZIP pins and member rosters

Each plan ZIP hash and byte count matches the local archive. `members.json` rosters and member counts match the plan. All three packaged native-library hashes for each ZIP match the plan; no library was imported.

| Candidate | Bytes | SHA-256 | Roster |
|---|---:|---|---|
| Baseline TL3 | 12,538,394 | `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569` | S5 0.34, TL5 0.33, TL6 0.33 |
| H390 S3 replacement | 12,537,972 | `56ebf18d26187541e539a31078a50dcbeb698e395ca0ecde5e1174aa2e929ded` | S3 0.34, TL5 0.33, TL6 0.33 |
| H390 S3/S4 replacement | 16,161,940 | `8bf614168d97d50b58557db3a1f5befa16f4912b5b8b1fbf609b087f62f96be2` | S3 0.17, S4 0.17, TL5 0.33, TL6 0.33 |

The two built-package receipts report exact S3/S4 source payload hashes, preserved TL3 native/source entries, no native execution, zero target reads, and zero new fits. They also state that H390 S3/S4 used train+valid and that original TL5/TL6 provenance/exposure remains unknown. No exact composite grade or public-gain estimate is claimed.

## Replay contract

- `load_plan` pins the plan, runner/helper sources, both split manifests, each ZIP's byte count/hash, the selected panel, and Python 3.11. It recomputes the sample from the pinned split metadata and rejects any difference. The panel selection uses a fixed SHA-256 rank within each fold and chooses one complete sequence per chosen component.
- The callback reads `seq_ix`, `step_in_seq`, `need_prediction`, and the 112 `FEATURE_COLUMNS`; it does not request target values or score masks. It validates sequence identity, step order, warmup flags, feature shape and finiteness. `run_sequence` checks finite required outputs and suppresses warmup predictions.
- Static ABI inspection precedes package loading. The runner then CRC-checks and safely extracts the exact ZIP, loads the embedded callback, checks member count and selected native-library hash, and performs an A–B–A bitwise reset check. It writes a fixed-shape prediction cache and per-sequence feature/need hashes. No score or contrast is computed.
- Plan resources are 400,000,000 bytes, one CPU core, 300 seconds per replay, three serial replays/900 seconds reserved, zero swap, and scratch at `/home/vlad/development/wunder/research_tmp`; the interpreter field points to `.venv-py311-profile/bin/python`. The runner enforces Python 3.11 but does not compare `sys.executable` to that exact path. Resource caps and physical scratch also depend on root using `scripts/limited` and choosing output paths under that scratch root.

## Remaining limits and gate before grading

The raw validation file is 5,615,509,128 bytes. The plan records its historically verified full SHA-256, size, mtime and row-group count; the runner checks current size, mtime and row-group count but does not rehash all raw bytes. It also records selected feature and `need_prediction` hashes in each replay receipt. Before any quality grade, root must compare those 64 ordered hashes across the three fresh replay receipts and require exact equality. The runner is intentionally one candidate per process and does not itself perform that cross-receipt comparison.

All validation has been repeatedly exposed, H390 S3/S4 were fit on train+valid, and TL3 ancestry is unknown. F0/F2 are sampling strata, not clean holdout claims. Later scores must be reported as exposed diagnostics; the local callback panel is not a competition-runtime forecast. No validation target/mask values or prediction arrays were read. ZIP and native-library bytes were hashed for identity; no model or native library was imported or executed.
