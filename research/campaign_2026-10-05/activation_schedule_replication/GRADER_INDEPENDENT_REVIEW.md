# E480 scorer independent source review

**Decision: GO for the explicit sealed-spec invocation; the default filename caveat below remains.** I reviewed the scorer source and tests, validated the completed fit's exact sealed spec, runner receipt, LR traces, 24 checkpoint receipts, and main-path fit-only normalization without reading test parquet rows, targets, or making predictions. The scoring path is ready when called with the present sealed spec explicitly. A plain invocation relying on the scorer's default sealed-spec path will fail before validation.

## Current identity and fit receipts

- Scorer SHA-256: `9a83cf5e10c0019bc49106eeded55f79d2f6c49107f3aeb0ec4c9e1374d06862`.
- Synthetic tests SHA-256: `6bce83c0aea32ee8cc0991bfcd74fe1e44cf94097f243cc58d7c7299f58df652`.
- Fit seal file actually present: `e480_twentyfour_fit_v1.sealed.json`, SHA-256 `d0279d27f9103f04d29dc99d1e727bb76cd0923341a5ede48bd61cb4f4de4651`; its spec name is `e480_activation_schedule_replication_twentyfour_fit_v1`.
- Attempt receipt: `research/campaign_2026-10-04/runner/attempts/e480-activation-schedule-replication-twentyfour-fit-v1.json`, SHA-256 `8588a2147bce30a8fd54b13a4e017a9c78bab5acbeee5a2599ecdc4736bfb44f`.
- Bound spec digest: `323d01c7b6e1d967ddd6ea31c42113a11f84659a4ec726b82aa510cc316f05ab`.
- The attempt is finished with exit 0, successful unit result, clean worker and parent OOM counters, and verified teardown. Its artifact hashes match the current fit manifest (`911fd6471e9c7cd5a73eb44ef52d7db54a628cd1c235cfc83e05ff918289fa03`) and frozen LR trace receipt (`046f38f67792af73450bbbb958403834d948495e4ba64b263a2a9a2ff935e4ff`).
- I ran the scorer's read-only receipt path with the existing seal supplied explicitly. It verified the runner/spec/runtime/artifact identities and all 24 checkpoint hashes and state-dict identities; each checkpoint has 61,058 parameters. Manifest status is `all_24_complete_test_unread`, with fit-only true and exposed test groups loaded false. No parquet rows were read in that validation.

## Default-path caveat

`SEALED_FIT_SPEC` defaults to `e480_activation_schedule_replication_twentyfour_fit_v1.sealed.json`, but the repository contains `e480_twentyfour_fit_v1.sealed.json`. The expected spec name inside the existing file is correct, but its filename differs from the default. Therefore a command without `--sealed-spec` raises `FileNotFoundError` before validation. The controller can use the exact extant seal through `--sealed-spec research/campaign_2026-10-05/activation_schedule_replication/e480_twentyfour_fit_v1.sealed.json`; I validated the scorer's full main provenance path using that explicit path and intercepted it immediately before its first test-row read. The current scorer source identity is unchanged, and the caveat does not block that explicitly bound run.

## Source checks that passed

The independent implementation recreates the GELU-plus-sine embedding while preserving the checkpoint key layout and 61,058-parameter topology. It verifies the E476 schedule traces and their frozen receipt, exact 24-cell identities/order/update counts/telemetry/checkpoint hashes, runner/runtime/OOM/teardown receipt, external fit-only normalization against an independently recomputed statistic, identity model buffers, and raw target/prediction finiteness before official metric clipping. Its paired bootstrap uses a common component multiplicity matrix across the 24 cells and all registered contrasts; results remain separate by seed. The design's preregistered primary count is eight (four matched warm-cosine-minus-constant-mean effects at each of two seeds); no arm pruning is coded.

The six synthetic scorer tests pass under the bounded helper. Using the explicit seal path, the scorer's actual `main()` path validated the exact spec, runner/runtime/artifact receipt, all 24 checkpoint receipts, fit-only normalization, and cohort order, then stopped at an intercepted `include_test=True` call before any test rows were loaded; it created no output directory. `GRADER_SOURCE_NOTE.md` predates the completed E480 full fits and its statement that no fit ran is stale, although this review itself did not score or access test labels.
