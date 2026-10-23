# First-cell score source and recovery status (updated 2026-10-21 12:58 UTC)

This update supersedes any older “pending V12” wording in the preserved historical suffix below. It is a navigation record, not a score claim or authorization for another worker action.

## Current V7 scorer source and V14 admission

The final scorer source is [V7](first_cell_evaluation_score_v7/README.md). Its core adds a bounded preflight for only the two selected target-index members (`row_group.npy`, `seq_ix.npy`): unique exact names, size bounds, and a bounded NPY-header check for strict one-dimensional integer indices. The canonical source loader remains responsible for registered source hashing and split validation; the preflight does not read `y.npy` or `mask.npy`. The V7 source pair has independent status `PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR_V7` in [SCORE_SOURCE_REVIEW_V7.json](first_cell_evaluation_score_v7/SCORE_SOURCE_REVIEW_V7.json), 14,150 bytes / SHA-256 `7dc58f88435b33655f4ae72ce3c87a270491c49f33b808b8ae568164997deadc`.

The V7 focused fixture exercises valid selected headers, malformed/object/oversized cases, and selected-member policy. Reviewer qualification: it does **not** contain a distinct truncated-header fixture case, although the frozen core has a truncation check. Do not describe truncated-header behavior as fixture-tested. The [V7 proof](first_cell_evaluation_score_v7/SCORE_V7_NPZ_INDEX_PREFLIGHT_PROOF.json) and [fixture](first_cell_evaluation_score_v7/test_score_v7_npz_index_preflight.py) are source-only/synthetic; no real NPZ was opened in this review.

The current one-shot admission is [V14](first_cell_evaluation_score_admission_v14/README.md). Its exact body is 193,262 bytes / `fea0734c5485fa984e0bb961db58dafc01950c1308a3893dd400d15d7c332670`, with independent status `PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V14`. The current binding is [SCORE_ADMISSION_SOURCE_BINDING_V14_V2.json](first_cell_evaluation_score_admission_v14/SCORE_ADMISSION_SOURCE_BINDING_V14_V2.json), 4,835 bytes / `41d4a76da41a0613458d76b16f6d3321808acf5dd6627639276f20438382e80b`. It binds the V7 scorer review separately from the unchanged V4 export review. The root capture driver is [ROOT_CAPTURE_SCORE_ADMISSION_V4.py](first_cell_evaluation_score_admission_v14/ROOT_CAPTURE_SCORE_ADMISSION_V4.py), 6,728 bytes / `f8f3d9501c88498bf465f744be9b79812f5f8080f396c704938be58e70785df3`, with an exact unit-only inverse proof.

The first V14 binding was preserved as a source-only negative: [SCORE_ADMISSION_SOURCE_BINDING_V14.json](first_cell_evaluation_score_admission_v14/SCORE_ADMISSION_SOURCE_BINDING_V14.json), 4,830 bytes / `d22e382df5ab7eefc100bbed8a956db2f338b2ec08d73f2eaf496448e1f50e31`, incorrectly pointed `score_result_path` under the score-v3 private directory. The corrected V2 binding changes only that JSON field to the frozen body’s actual `EXPORT_D/RESULT.json` path; [the correction proof](first_cell_evaluation_score_admission_v14/V14_BINDING_CORRECTION_PROOF_V2.json) gives the exact inverse. The body itself did not change.

ROOT submitted V14 once at 2026-10-21 12:57:39 UTC for unit `computer-vision-homework-fg19-first-cell-score-v3.service`, invocation `9bd64d6759e8462bb40985b06a9a0ad6`, in [the root-owned score-v4 record](../records/first_cell_evaluation_v1/score_v4/). The worker’s later state and result belong to ROOT’s durable capture and acceptance records; this index does not infer them. No score or quality claim is made here. The prior score-v1 and score-v2 worker failures are preserved; never query, stop, restart, retry, or reuse those units/directories.

## Reader V6 review snapshot

The [preservation note](first_cell_evaluation_score_capture_v6/READER_V6_REVIEW_SNAPSHOT_PRESERVATION.md) distinguishes the separately preserved announced review snapshot (9,673 bytes / `5e067c1d9972be3f00f462a075084b5ea315e3708bdb18669a0cf52be1a2d8ed`) from the current canonical review (9,525 bytes / `844fb16d236a6778179f55edfe37a553809097691b19060869309c0c07bbeeec`). The separate snapshot is described as reconstructed from the recorded exact row replacements; it is not claimed restored at the canonical path. `INDEPENDENT_SOURCE_REVIEW_V6_V2.json` is an identical copy of the current canonical review. Consumers must bind the explicit reader review path/SHA named in their own versioned binding.

## Historical records

The prior V11/V12 source history and first score-v1 failure documentation remain below as an exact preserved suffix. Earlier statements that V12 was pending and no V12 admission had been submitted are historical and superseded by this update. The second actual worker failure (score-v2) is preserved in [score-v3 status capture](../records/first_cell_evaluation_v1/score_v3/status_capture_v1/); its failure occurred in NPZ selected-index preflight, before prediction-array opening or score calculation. The V7 source fix was reviewed before the V14 admission.

---

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
