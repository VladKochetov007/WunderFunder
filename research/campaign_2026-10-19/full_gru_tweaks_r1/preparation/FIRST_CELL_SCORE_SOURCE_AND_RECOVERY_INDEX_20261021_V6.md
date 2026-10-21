# First-cell score source and recovery index — V6 lifecycle closure

This addendum supersedes prior pending-acceptance/stop status in the preserved V5 history below. Root acceptance and the one successful score-v3 exact stop are now durably recorded. Root owns the final comparison report, canonical navigation, and ledger; this index records evidence lineage and limitations only.

## Closed score lifecycle

The captured fixed-model score report is [RESULT.json](../records/first_cell_evaluation_v1/score_v4/status_capture_v2/RESULT.json), 33,852 B, SHA-256 `1646d8656a042ce49b355f22521134206d601c1dbac3697991fcd308aa1a94d5`. The root acceptance is [SCORE_ACCEPTANCE.json](../records/first_cell_evaluation_v1/score_v4/SCORE_ACCEPTANCE.json), 21,229 B, `6e9a8162321b2490d5131039e4fe9cf9aded13e58b789d5789491558861c0ed8`; its root binding is 8,932 B, `829f0e90ee5d99600f8db85d8c37d024c7235012f2c792c38fd8f5d7f6c65dd3`.

The only successful score unit was `computer-vision-homework-fg19-first-cell-score-v3.service`, invocation `9bd64d6759e8462bb40985b06a9a0ad6`. Its exact stop was captured at `2026-10-21T13:29:44.567799+00:00`: before state `active/exited`, PID 0, code/status `1/0`, matching invocation; after state `not-found/inactive/dead`, PID 0, empty invocation; empty populated snapshots; stop return code 0. [EXACT_STOP.json](../records/first_cell_evaluation_v1/score_v4/exact_stop/EXACT_STOP.json) is 1,211 B, SHA-256 `7ae50e926fb66948a7a3c95ce227c4fc11c88d68e55f2483012275c662bb953e`. The preserved stdout is byte-identical to that exact-stop file; transport records SSH return code 0, zero stderr, persistence before parse, and no retry. Failed score units 1 and 2 were excluded and never stopped.

## Source and transport authority

The final acceptance helper is [V4 source](first_cell_evaluation_score_acceptance_v4/score_result_acceptance_v4.py), 46,350 B / `55b4814921e522d9fa24f6c9e4d830218989d42a834ebf9a30af056d617741dc`, with independent source review 15,718 B / `3597f9be9196a5ad0fa9aed7c208e26f29e4c10876d8ecfb0ef546e3e4e2370b`, fixed-model delta proof 1,101 B / `fc14916e4602ea6df4517ac9b91dce5c6da947f6443a2853694accb9cf735bc8`, and a focused actual-metadata fixture (7,383 B / `cdb073116034f655133f2d3f5859d00e1cf4f6ac142f1989cc90c811c990fa55`). The final correction matches the actual `fixed_models` checkpoint object `{bytes, relative_path, sha256}` and leaves export-side loaded-checkpoint validation intact. See the [V4 helper README](first_cell_evaluation_score_acceptance_v4/README.md).

The ROOT acceptance caller is V9, 32,464 B / `1c3b5f91dbb0cd72938ea3d62f3a0f9b85c9badb1e3042e50eb2d3b004e1ec1b`, with source review 8,142 B / `7b28323a5d4e38bd8ad71c65ffb96a9f8ab9f742d057427721c57d6228ce7eb1` and helper-V4 rebase proof 1,914 B / `30d7e2eaf5099a2cec7cc9a4c9d7475aa8e5f7cdda52f8cc249de32723bfad87`. Its reviewed ancestry retains the exact final V7 namespace candidate at 32,281 B / `811e44d27f1e6c798d11f26d450b7aed76c6c6a6cbd888de047e91cbb25a4c2e` (V7 namespace proof V2); older preliminary V7 bytes are separately classified in [the refreshed classification record](FINAL_PUBLICATION_PIN_CLASSIFICATIONS_20261021_V2.json). Any previous 31,723-byte note is stale; the final V7 candidate and proof V2 both pin 32,281 bytes and the SHA above. See the [acceptance V3](first_cell_evaluation_score_acceptance_v3/README.md), [acceptance V4](first_cell_evaluation_score_acceptance_v4/README.md), and [result capture V6](first_cell_evaluation_score_capture_v6/README.md) README files for packet boundaries.

The actual score-stop body review is [SOURCE_REVIEW.json](../records/first_cell_evaluation_v1/score_v4/exact_stop/SOURCE_REVIEW.json), 6,046 B / `27324ec08a9d170847b9d20015ed848ec91eb253e15430c63c838b00aaad9955`. Transport review V1 (4,156 B / `0fadd507ee176ee314535f4ed1acb73222ab9d9a969a70477ac045cbadb64ea0`) rejected the driver’s uncapped local SSH output capture and remains preserved as a historical finding. Transport review V2 (6,784 B / `9b8f79af8672f55e97af7d295aea75ff375e63f9b838a8f8071236de35cb61b2`) grants a conditional source-only pass under the exact trusted ROOT scope and explicitly retains that local `capture_output=True` is not a general pre-capture cap. The actual transport and exact-stop pins are recorded in the score-v4 record directory; see its [README](../records/first_cell_evaluation_v1/score_v4/exact_stop/README.md) and the [V4 stop-packet README](first_cell_evaluation_score_stop_v4/README.md). The historical score-v1 reader/helper leaf compatibility check is described in its [README](first_cell_evaluation_score_leaf_compatibility_v1/README.md).

## Preparation README coverage

New README files summarize the bounded purpose and lineage for [acceptance helper V3](first_cell_evaluation_score_acceptance_v3/README.md), [acceptance helper V4](first_cell_evaluation_score_acceptance_v4/README.md), [result capture V6](first_cell_evaluation_score_capture_v6/README.md), [leaf compatibility V1](first_cell_evaluation_score_leaf_compatibility_v1/README.md), [stop V3](first_cell_evaluation_score_stop_v3/README.md), and [stop V4](first_cell_evaluation_score_stop_v4/README.md). They are navigation aids and do not modify review-pinned source, reviews, results, or receipts.

The V8 ROOT local acceptance rejection note, source/archive classifications, and all prior versioned snapshots remain separately preserved. This closure does not add a generalization, public-transfer, or forecast claim; ROOT's separate final comparison report owns the scientific summary.

# First-cell score source and recovery index — V5 status addendum

This addendum supersedes the status statements in the V4 snapshot below. It preserves that full V4 index byte-for-byte as its suffix; it does not rewrite or reinterpret prior reviews, result evidence, or runtime records.

The captured score result remains the same 33,852-byte artifact (`1646d8656a042ce49b355f22521134206d601c1dbac3697991fcd308aa1a94d5`). A single root-local metadata acceptance application rejected before writing either output because helper V3 expected a `loaded_sha256` checkpoint field that the actual result does not contain. The actual `fixed_models` checkpoint object has only `bytes`, `relative_path`, and `sha256`; accepted export authority already binds the loaded checkpoint identity. This was a local metadata validation failure only: no B action, new worker unit, score replay, or scientific fit occurred. `SCORE_ACCEPTANCE.json` and `ROOT_ACCEPTANCE_BINDING.json` were absent after the rejection.

The durable rejection record is [ROOT_LOCAL_ACCEPTANCE_V8_REJECTION.json](../records/first_cell_evaluation_v1/score_v4/ROOT_LOCAL_ACCEPTANCE_V8_REJECTION.json) (1,595 bytes, SHA-256 `c768ac838296e4e09ba46a65806297fc5b434a187643dea8a3ce42f82a249afe`). It pins the exact caller and unchanged captured result. A source-only helper V4 proof records the narrow expected-shape correction; this metadata note does not claim acceptance, exact stop, or further execution. Root owns any later acceptance/stop and publication.

# First-cell score source and recovery status (updated 2026-10-21 after score-v4 capture)

This is a new current-status prefix. The complete V3 index follows byte-for-byte as a preserved suffix. ROOT owns final score acceptance/exact stop and publication; this index and the separate comparison draft do not substitute for those records.

## Captured score report

The exact captured [`RESULT.json`](../records/first_cell_evaluation_v1/score_v4/status_capture_v2/RESULT.json) is 33,852 bytes / SHA-256 `1646d8656a042ce49b355f22521134206d601c1dbac3697991fcd308aa1a94d5`, status `FIXED_FIRST_CELL_DEVELOPMENT_EVIDENCE_SCORED`. The score unit `computer-vision-homework-fg19-first-cell-score-v3.service` exited successfully at 13:05:12 UTC; ROOT captured and digest-joined the original result once at 13:08:51 UTC. See the [`status observation`](../records/first_cell_evaluation_v1/score_v4/status_capture_v2/OBSERVATION.json), [`raw transport`](../records/first_cell_evaluation_v1/score_v4/status_capture_v2/TRANSPORT.json), and [score-v4 record README](../records/first_cell_evaluation_v1/score_v4/README.md). The launch receipt remains labeled `ACCEPTED_SINGLE_CPU_SCORE_LAUNCH_NOT_RESULT`. Score-result ROOT acceptance and exact stop are not yet in the record directory, so the captured report is not yet final publication authority.

The five fixed prediction exports were accepted and exact-stopped separately; their receipts are [export acceptance](../records/first_cell_evaluation_v1/export_v5/EXPORT_ACCEPTANCE.json) and [export exact stop](../records/first_cell_evaluation_v1/export_v5/exact_stop/EXACT_STOP.json). Those records do not accept or stop the score unit. A standalone comparison-report draft exists at `/tmp/wunder_first_cell_comparison_final_draft.md`; ROOT owns its final copy and publication after score acceptance/stop.

## Source-only corrections, no result replay

[The V7 caller review correction](first_cell_evaluation_score_acceptance_v1/ROOT_CALLER_V7_REVIEW_CORRECTION_V1.json), 2,157 bytes / SHA-256 `8680591845a81da58771380ef617842bce99254100e1ea8d6e5b634cfc282750`, has status `CORRECTS_V7_SOURCE_FINDING_MISATTRIBUTION`. It establishes that the rejected finding misread `not raw` in `json_object()` as a `read()` check: the exact caller's `read()` accepts a zero-byte regular file after its size and stability checks. The note corrects the source-location finding only; it did not edit or execute the caller, run tests, or access a result or payload. Preserve the original rejection and correction as separate records.

The local expected checkpoint-shape correction is pinned in [helper V4's inverse proof](first_cell_evaluation_score_acceptance_v4/HELPER_V4_FIXED_MODEL_DELTA_PROOF.json), 1,101 bytes / SHA-256 `fc14916e4602ea6df4517ac9b91dce5c6da947f6443a2853694accb9cf735bc8`. The captured `RESULT.json` represents each `fixed_models[].checkpoint` as the scorer's exact model dictionary (`bytes`, `relative_path`, `sha256`); it has no `loaded_sha256` member there. The prior expected checkpoint object invented that extra field, so a local acceptance helper would reject the actual report shape. Helper V4 changes the expected fixed-model checkpoint to `dict(arm["model"])`; the existing export-sidecar loaded-checkpoint hash checks remain. The proof verifies the exact inverse to helper V3 and records that neither the helper nor its metadata fixture was run for that proof. This is a source/metadata contract correction only, not a score replay, result acceptance, or quality finding. See [the helper V4 source](first_cell_evaluation_score_acceptance_v4/score_result_acceptance_v4.py) and [its actual-metadata fixture source](first_cell_evaluation_score_acceptance_v4/test_score_result_acceptance_v4_actual_metadata.py); this update does not claim that fixture was executed.

The comparison result remains limited to one F0/seed-1001, fixed-epoch-5 five-model panel on 468 fit-excluded but exposed development groups. The score and bootstrap intervals are not a public-holdout, seed-robustness, or general forecasting claim. ROOT's final comparison page and evidence-state updates remain pending its acceptance and exact-stop records.

---

## Preserved previous index (exact V3 suffix)

# First-cell score source and recovery status (updated 2026-10-21 12:59 UTC)

This update supersedes the 12:58 entry in the preserved V2 index suffix. It records only local/source and root-reported lifecycle facts; ROOT owns all later status/result interpretation.

## V14 admission and latest root records

The final V14 exact-body review is [ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V14.json](first_cell_evaluation_score_admission_v14/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V14.json), 18,342 bytes / SHA-256 `4cd8c964d4bfb59032e9e6b79cfd63777417d0f96d0031ae9f2cd01f21227a82`, status `PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V14`. It reviewed the 193,262-byte body (`fea0734c5485fa984e0bb961db58dafc01950c1308a3893dd400d15d7c332670`), corrected V2 binding (`41d4a76da41a0613458d76b16f6d3321808acf5dd6627639276f20438382e80b`), six fixed payloads, separate V7 scorer review and unchanged V4 export review, and root driver V4.

The preserved original binding (`SCORE_ADMISSION_SOURCE_BINDING_V14.json`, 4,830 bytes / `d22e382df5ab7eefc100bbed8a956db2f338b2ec08d73f2eaf496448e1f50e31`) is a source-only negative: it placed `score_result_path` under the fresh score-v3 private directory. The exact correction is [`V14_BINDING_CORRECTION_PROOF_V2.json`](first_cell_evaluation_score_admission_v14/V14_BINDING_CORRECTION_PROOF_V2.json), 1,453 bytes / `282b198aab0b8e20775363ce1c4c7e45e823b1d31c7cfe0936574c5ba5e6d7a1`; it changes only that field in [`SCORE_ADMISSION_SOURCE_BINDING_V14_V2.json`](first_cell_evaluation_score_admission_v14/SCORE_ADMISSION_SOURCE_BINDING_V14_V2.json). The worker body remains unchanged.

ROOT also preserved a local review-pin preflight note: [`ROOT_LOCAL_PROVISIONAL_REVIEW_PIN_GATE_NOTE.json`](first_cell_evaluation_score_admission_v14/ROOT_LOCAL_PROVISIONAL_REVIEW_PIN_GATE_NOTE.json), 1,844 bytes / `36e73f1ce4ac61098feb2993bc69d8413c7f2d9d5f738e32a65ee994efeb92e3`. It distinguishes an unannounced 17,758-byte provisional review from the final 18,342-byte review. Two local guards rejected the provisional pin before record-directory creation and SSH; they created no CPU unit. The note does not claim those provisional bytes were archived.

ROOT submitted the final V14 body once at 12:57:39–12:57:40 UTC, invocation `9bd64d6759e8462bb40985b06a9a0ad6`, for `computer-vision-homework-fg19-first-cell-score-v3.service`. ROOT's append-only launch row and record documentation are under [score-v4](../records/first_cell_evaluation_v1/score_v4/). This index does not infer worker terminal status or read/claim a score; use ROOT's durable capture and acceptance documents for the authoritative outcome. Failed score-v1 and score-v2 workers remain immutable and must not be queried, stopped, restarted, retried, or reused.

## V7 reviewer qualification and reader-review history

The V7 source review is PASS as linked in the previous update. Its focused fixture checks oversized-header behavior but has no distinct truncated-header test case; the implementation has a truncation check, so do not characterize that branch as fixture-tested. This qualification is included to keep the source review's scope precise; no real target NPZ members were opened locally.

The reader V6 preservation note is [`READER_V6_REVIEW_SNAPSHOT_PRESERVATION.md`](first_cell_evaluation_score_capture_v6/READER_V6_REVIEW_SNAPSHOT_PRESERVATION.md). It separates the announced snapshot at 9,673 bytes / `5e067c1d9972be3f00f462a075084b5ea315e3708bdb18669a0cf52be1a2d8ed` from the current canonical review at 9,525 bytes / `844fb16d236a6778179f55edfe37a553809097691b19060869309c0c07bbeeec`; `INDEPENDENT_SOURCE_REVIEW_V6_V2.json` is an identical current-review copy. The announced snapshot is retained at its own path, not substituted for the current canonical file. Each consumer must bind an exact review path and digest.

---

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
