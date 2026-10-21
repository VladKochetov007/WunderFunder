[Final V10 independent source review](ACCEPT_EXPORT_CAPTURE_SOURCE_REVIEW_V10.json),8400B/10cae43b, passes the exact recovered54513B/00e962 dual-transport acceptor only. It is unexecuted and writes no score; prior V8 negative and V9 announcement-race history remain indexed below.

# Current review and capture navigation — 2026-10-21

This section supplements the preserved navigation below. Source reviews and
static proofs are conditional preparation evidence; they do not establish a
terminal export acceptance, exact stop, score, or quality result.

## Current export record

- [Export V5 launch and latest startup observation](../../records/first_cell_evaluation_v1/export_v5/README.md) records the actual launch and 10:32 UTC active-unit observation. The unit was still running; there was no terminal manifest, acceptance, or quality result in that observation.

## Acceptor source history

- [V10 acceptor candidate](ACCEPT_EXPORT_CAPTURE_V10.py) is an exact copy of the recovered [announced V9 snapshot](ACCEPT_EXPORT_CAPTURE_V9_ANNOUNCED_SNAPSHOT.py), both 54,513 bytes with SHA-256 `00e96230c85398f5b9d15f1c2aea6e50e9dc685766e7a507a3b36c25329bbe75`. Its [exact-copy proof](ACCEPT_EXPORT_CAPTURE_V10_EXACT_COPY_PROOF.json) verifies the recovery and the fixture's filename-only fork. V10 was sent for independent source review; until that review is recorded, treat it as a candidate only.
- [V9 announcement-race note](FROZEN_ANNOUNCEMENT_RACE_NOTE_V1.json) preserves the limits of that recovery: the current [V9 path](ACCEPT_EXPORT_CAPTURE_V9.py) is the 54,963-byte / `4a10af92357d4a470aeb0cabf6033bbe21f591f78de4015b664fba8dd1bb4adf` incomplete post-freeze edit and must not be used or tested. The earlier 54,138-byte announcement was not archived or reconstructed. The five-fixture V9 result recorded there was reported against the exact 54,513-byte snapshot before the race, not the current V9 path.
- [V8 acceptor](ACCEPT_EXPORT_CAPTURE_V8.py) and its [independent rejection](ACCEPT_EXPORT_CAPTURE_SOURCE_REVIEW_V8.json) are retained as a negative source review: V8 conflates the launch and later status-capture roots/transports.

## Status and score capture readers

- [Status driver V5](ROOT_CAPTURE_EXPORT_STATUS_V5.py), its [source review](ROOT_CAPTURE_EXPORT_STATUS_SOURCE_REVIEW_V5.json), and [review-pin correction](ROOT_CAPTURE_EXPORT_STATUS_V5_REVIEW_PIN_CORRECTION_V1.json) preserve the reviewed bounded status-capture path. The correction addresses a fixture-hash transcription only; it does not replace the original review or revalidate an actual capture.
- [Reader template V4](READ_EXPORT_TEMPLATE_V4.py) has a [reader source review](READ_EXPORT_SOURCE_REVIEW_V3.json); the [third-unit root-tools review](THIRD_UNIT_ROOT_TOOLS_SOURCE_REVIEW_V1.json) records source-only PASS for the reader/stop/root-driver set. Their actual literal bindings were unbound in those reviews, so consult the dated records for actual state.
- [Conditional score status capture V2](../first_cell_evaluation_score_capture_v2/README.md) links the [V2 root-driver source review](../first_cell_evaluation_score_capture_v2/ROOT_SCORE_STATUS_DRIVER_SOURCE_REVIEW_V2.json), SHA-256 `ae7d68b74b61423d4ef15f93ca9e1c30581287229fd03d9028c2a80ee54b264e`. The reader is source-reviewed, but the five actual score-receipt/invocation literals remain unbound and no score is established by this packet.

# Root-owned export capture artifacts

This index links the root-authored capture and receipt-consumer artifacts kept beside the local admission packet. The templates are preparation materials; a template, focused proof, or source review does not itself establish an export acceptance, exact stop, score, or quality result. For the one actual failed export-admission preflight, use [the canonical records folder](../../records/first_cell_evaluation_v1/export_v1/README.md).

## Capture plan and drivers

- [Root orchestration plan](ROOT_ORCHESTRATION_PLAN.json) records the intended ordering and ownership boundaries.
- [Reader binding plan](READ_EXPORT_BINDING_PLAN.json) records the root reader's expected evidence and capture bindings.
- [Admission capture driver](ROOT_CAPTURE_EXPORT_ADMISSION_V1.py) and [status-capture driver](ROOT_CAPTURE_EXPORT_STATUS_V1.py) are the root-side capture entrypoints saved with this packet.

## Root reader

- [Reader template V1](READ_EXPORT_TEMPLATE.py) and [reader source review V1](READ_EXPORT_SOURCE_REVIEW_V1.json) preserve the initial root-side metadata reader.
- [Reader template V2](READ_EXPORT_TEMPLATE_V2.py), [reader source review V2](READ_EXPORT_SOURCE_REVIEW_V2.json), [V2 focused proof](READ_EXPORT_V2_FOCUSED_PROOF.json), [reader local checks](READ_EXPORT_LOCAL_CHECKS.json), [contract fixture](test_read_export_contract.py), and [boundary fixture](test_read_export_v2_boundaries.py) retain the later reader version and its narrow local evidence.

## Acceptance capture

- [Acceptance capture V1](ACCEPT_EXPORT_CAPTURE.py), [source review V1](ACCEPT_EXPORT_CAPTURE_SOURCE_REVIEW_V1.json), and [source review V2](ACCEPT_EXPORT_CAPTURE_SOURCE_REVIEW_V2.json) preserve the initial acceptor and review history.
- [Acceptance capture V2](ACCEPT_EXPORT_CAPTURE_V2.py), [focused checks V2](ROOT_EXPORT_ACCEPTANCE_V2_FOCUSED_CHECKS.json), and [metadata fixture V2](test_root_export_acceptance_v2.py) record the next acceptance-capture version.
- [Acceptance capture V3](ACCEPT_EXPORT_CAPTURE_V3.py), [focused proof V3](ROOT_EXPORT_ACCEPTANCE_V3_FOCUSED_PROOF.json), and [native-identity fixture V3](test_root_export_acceptance_v3_native_id.py) retain its explicit identity checks.
- [Acceptance capture V4](ACCEPT_EXPORT_CAPTURE_V4.py) and [V4 dependency proof](ROOT_EXPORT_ACCEPTANCE_V4_DEPENDENCY_PROOF.json) link the later root-owned capture variant. Review status must be taken from its corresponding exact review artifact, not inferred from the version number.

## Exact-stop and source helpers

- [Export exact-stop template](STOP_EXPORT_TEMPLATE.py) and its [source review](STOP_EXPORT_SOURCE_REVIEW_V1.json) preserve the separate stop consumer.
- [Admission source binding](ADMISSION_SOURCE_BINDING_V1.json), [body template](ADMISSION_BODY_TEMPLATE_V1.py), [remote submission template](REMOTE_SUBMISSION_TEMPLATE.py), and [remote source review](REMOTE_SUBMISSION_SOURCE_REVIEW_V1.json) tie together the initial admission-side handoff.
- [V5 stdlib helper extraction](V5_STDLIB_HELPERS_EXTRACTED.py), its [extraction record](V5_STDLIB_HELPER_EXTRACTION_V1.json), and [extraction script](extract_v5_stdlib_helpers.py) document the source used for bounded standard-library helpers.
- [Admission renderer](render_admission_remote_v1.py), [admission fixture](test_admission_packet_v1.py), and [local checks](LOCAL_CHECKS.md) cover local packet preparation; they do not execute any remote body.

For the current actual state and preserved raw streams, follow [records/first_cell_evaluation_v1](../../records/first_cell_evaluation_v1/README.md). The audit of the one failed V2 preflight is at [ACTUAL_PREFLIGHT_FAILURE_METADATA_REVIEW.json](../../records/first_cell_evaluation_v1/failedexport_v1/ACTUAL_PREFLIGHT_FAILURE_METADATA_REVIEW.json).
