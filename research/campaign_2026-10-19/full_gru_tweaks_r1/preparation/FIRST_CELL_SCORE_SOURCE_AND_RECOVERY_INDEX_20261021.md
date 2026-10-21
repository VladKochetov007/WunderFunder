**21 October 12:04 UTC current state:** V12 passed independent source review and launched the first actual score-v1 worker once. A single bounded status capture found that worker failed at 11:59:31 UTC in `observe_registered_source` on the cache-parent alias check, before split-input/index preflight, prediction opening, or scoring. The captured 103,906-byte stdout is SHA-256 `818c905ddcf6277120be5ea7f5d6eb58f6a704a5c1f2c1354afe84a8838ccac2`; no result was read. Keep failed unit/D1 forever: no query, stop, retry, or restart. The bounded parent/leaf diagnostic showed `data/cache` resolving to itself and `data/datasets` resolving to `/home/grinchenkoeg/WunderFunder/datasets`; the six fresh leaf identities still match the registration. [V6 source packet](first_cell_evaluation_score_v6/README.md) removes only the unsupported static parent target while retaining exact leaf checks. V6 independent scorer source review is pending; no quality result exists. The accepted V4 export and receipts remain unchanged.

**Historical suffix note:** the text below is retained byte-for-byte from the earlier pre-capture status. Any older statement that V12 was pending or had not been submitted is superseded by this current outcome above.

# First-cell score admission: source and recovery index

This page tracks the source-only admission revisions and the one actual score-admission preflight failure. It is a navigation and status record; it does not authorize a new submission or report a score. The current candidate is V12, which is still awaiting its independent exact-body review. No V12 admission has been submitted.

The score worker consumes the previously accepted fixed-five prediction export and its accepted stop receipts. The V11 attempt failed while validating the cache receipt metadata, before creating its worker directory or submitting a systemd unit. The preserved records support a narrow V12 correction to the cache metadata join. No scientific score or quality result exists in these records.

## Version path

| Version | Source change and evidence | State |
| --- | --- | --- |
| [V9](first_cell_evaluation_score_admission_v9/README.md) | Added a `SCORE_D` directory fsync after successful submission. The focused proof shows an exact inverse to V8 and unchanged scorer payload pins. | Not a usable candidate: the next review found that unresolved and failed receipt branches skipped this fsync. Preserved as a source negative. |
| [V10](first_cell_evaluation_score_admission_v10/README.md) | Moved the launch fsync into the successful receipt branch and added it after the unresolved receipt write; retained the pre-submit fsync. | Rejected source-only because source-file hashing could read past the aggregate cap before rejecting. No V10 unit was submitted. |
| [V11](first_cell_evaluation_score_admission_v11/README.md) | Added a pre-bounded manifest total, pre-open file-size checks, and an incrementally capped hash read. Independent source review passed. | Submitted once. Actual preflight failed on the cache receipt field mismatch before worker-directory creation and before `systemd-run`; the failed body was not retried. |
| [V12](first_cell_evaluation_score_admission_v12/README.md) | Changes only the cache acceptance join: the metadata carries `split_manifest_sha256`; the acceptance carries `metadata_sha256`. The six embedded scorer payload values remain unchanged. | Focused proof/fixture passed locally. Independent exact-body source review is pending; no admission submission has occurred. |

## Current candidate

The V12 body is [REMOTE_SUBMISSION_V12.py](first_cell_evaluation_score_admission_v12/REMOTE_SUBMISSION_V12.py), bound by [SCORE_ADMISSION_SOURCE_BINDING_V12.json](first_cell_evaluation_score_admission_v12/SCORE_ADMISSION_SOURCE_BINDING_V12.json). The body is 188,838 bytes, SHA-256 `133bb309b5b3ebba238ce4f40a9e9efa60c6813a0fcd208ab90ff4bd7bba799e`; the binding is 8,761 bytes, SHA-256 `f177f935578639e8ee1933750d85e9d610e27def44842dc7a3685c986e743241`.

[The V12 cache-schema proof](first_cell_evaluation_score_admission_v12/ACTUAL_SCORE_ADMISSION_V12_CACHE_SCHEMA_PROOF.json) records the actual metadata-only receipt join, the one-predicate change, exact inverse to V11, and unchanged six payload values. [The focused fixture](first_cell_evaluation_score_admission_v12/test_score_admission_v12_cache_schema.py) exercises the extracted `check_cache` helper with the actual receipt JSON and synthetic stat observations; it does not read or hash array payloads. [Local check notes](first_cell_evaluation_score_admission_v12/LOCAL_CHECKS.md) record the three fixture cases and inherited Ruff findings. These checks do not replace the pending independent source review.

## Actual V11 preflight record

The [score-v1 record README](../records/first_cell_evaluation_v1/score_v1/README.md) summarizes the single attempt and its bounded read-only outcome. The exact V11 candidate and source review are pinned in the [V11 cache-schema negative](first_cell_evaluation_score_admission_v12/V11_CACHE_SCHEMA_NEGATIVE_V1.json). Its raw stderr is 305 bytes, SHA-256 `710c9b7646bce81483174b540dff743907a5d2746e8b443a7434f8523caad51d`; stdout is empty, and the transport record reports SSH return code 1. The failure was `check_cache` asking the accepted cache terminal receipt for `split_manifest_sha256`, a field present in the metadata file but absent from that receipt. The acceptance instead links to metadata through `metadata_sha256`.

The negative and proof also pin the cache metadata, terminal acceptance, exact-stop record, and original AllThree fit admission used for the local metadata-only fixture. That fixture uses synthetic filesystem-stat observations; it does not load model weights or cache arrays. The failed V11 body, binding, review, stdout, stderr, and transport remain preserved. No retry was made, and the V12 candidate has not been submitted.

## Earlier source findings

- [V9’s V8 durability finding](first_cell_evaluation_score_admission_v9/ACTUAL_SCORE_ADMISSION_V8_DURABILITY_NEGATIVE_V1.json) records why the first successful-path fsync was insufficient. [The V9 inverse proof](first_cell_evaluation_score_admission_v9/ACTUAL_SCORE_ADMISSION_V9_FSYNC_INSERT_PROOF.json) preserves the exact one-block change and payload identity.
- [V10’s V9 unknown-branch finding](first_cell_evaluation_score_admission_v10/ACTUAL_SCORE_ADMISSION_V9_UNKNOWN_BRANCH_NEGATIVE_V1.json) records the unresolved/failed receipt paths that required a separate fsync. [The V10 branch proof](first_cell_evaluation_score_admission_v10/ACTUAL_SCORE_ADMISSION_V10_FSYNC_BRANCHES_PROOF.json) records the branch coverage and inverse.
- [The V10 source review](first_cell_evaluation_score_admission_v10/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V10.json) identifies the source hash-cap issue fixed in V11. [The V11 source review](first_cell_evaluation_score_admission_v11/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V11.json) records the independent pass for that source correction, before the later actual cache-schema failure.
- V11’s first local draft is deliberately retained but marked invalid: [draft note](first_cell_evaluation_score_admission_v11/V11_INITIAL_BAD_DRAFT_NOTE.json), [draft body](first_cell_evaluation_score_admission_v11/REMOTE_SUBMISSION_V11_INITIAL_BAD_DRAFT.py), and [draft binding](first_cell_evaluation_score_admission_v11/SCORE_ADMISSION_SOURCE_BINDING_V11_INITIAL_BAD_DRAFT.json). Use the reviewed [final V11 body](first_cell_evaluation_score_admission_v11/REMOTE_SUBMISSION_V11.py) and [final binding](first_cell_evaluation_score_admission_v11/SCORE_ADMISSION_SOURCE_BINDING_V11.json) when following the actual attempt.

## Review boundary

Source proofs and local fixtures establish narrow source or metadata properties only. They do not establish a score, model quality, worker launch, or successful result. The next gate is the independent exact-body review of V12; any later admission must use the root-owned raw-first capture process and a fresh, separately authorized attempt.
