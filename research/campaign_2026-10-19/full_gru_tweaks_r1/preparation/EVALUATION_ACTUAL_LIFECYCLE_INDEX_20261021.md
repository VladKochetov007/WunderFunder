# First-cell evaluation lifecycle — 21 October 2026

This index separates the two actual zero-unit preflights, the first export worker launch and its failure, the reviewed producer/consumer source pair, and the still-pending next launch. The `export_v3` directory under `records/` is the record folder for the first launched attempt; that attempt used export unit V1. It is distinct from the reviewed V3 exporter source and from the pending V5 admission for export unit V2.

## Current state

The last saved runtime evidence is the 21 October 09:18 UTC startup capture. The first export worker (`computer-vision-homework-fg19-first-cell-export-v1.service`, invocation `da6c9e1f5897478f9f3f686de58e39cc`) failed at a metadata-shape check before model loading or prediction. Its unit and private directory remain preserved; do not poll, stop, restart, or retry that failed unit. No export completion, scorer result, quality metric, or public-transfer claim exists.

The next candidate is the local [export admission V5](first_cell_evaluation_admission_v5/README.md), which targets the reviewed exporter/scorer V3 pair, unit `computer-vision-homework-fg19-first-cell-export-v2.service`, and private evidence root `fg19_first_cell_f0_s1001_evaluation_v2` (the new D2 root). Its [V5 body](first_cell_evaluation_admission_v5/REMOTE_SUBMISSION_V5.py) and [source binding](first_cell_evaluation_admission_v5/ADMISSION_SOURCE_BINDING_V5.json) have been rendered locally for independent review; V5 has not been launched. The [conditional CPU score admission V6](first_cell_evaluation_score_admission_v6/README.md) also remains local-only and unbound: the export acceptance/exact-stop paths and pins are null. The [score result reader V2](first_cell_evaluation_score_capture_v2/README.md) has no actual score ADMISSION/LAUNCH pins or result to capture.

## Actual preflights and first launched export

- [First actual export preflight](../records/first_cell_evaluation_v1/export_v1/README.md) stopped on the incorrectly nested helper path before creating the worker unit or evidence directory. Its [read-only no-unit outcome](../records/first_cell_evaluation_v1/export_v1/preflight_outcome_v1/README.md) and [independent failure metadata audit](../records/first_cell_evaluation_v1/failedexport_v1/ACTUAL_PREFLIGHT_FAILURE_METADATA_REVIEW.json) preserve that boundary.
- [Second actual export preflight](../records/first_cell_evaluation_v1/export_v2/README.md) stopped before unit creation because the admission expected a nonexistent top-level invocation field in the historical AllThree launch receipt. Its [read-only no-unit outcome](../records/first_cell_evaluation_v1/export_v2/preflight_outcome_v1/README.md) and [independent audit](../records/first_cell_evaluation_v1/export_v2/ACTUAL_PREFLIGHT_FAILURE_METADATA_REVIEW_V2.json) preserve the actual receipt-shape finding.
- The [first actual export launch record](../records/first_cell_evaluation_v1/export_v3/README.md) binds the one V1-unit invocation, original ADMISSION/LAUNCH, raw reply, and root launch acceptance. Its [launch metadata review](../records/first_cell_evaluation_v1/export_v3/ACTUAL_LAUNCH_METADATA_REVIEW.json) covers the admission/launch boundary.
- The following [startup failure record](../records/first_cell_evaluation_v1/export_v3/startup_health_v1/README.md) and [failure metadata review](../records/first_cell_evaluation_v1/export_v3/startup_health_v1/ACTUAL_FAILURE_METADATA_REVIEW.json) confirm the worker failed before model loading: the canonical source manifest entries contained `path` fields that the exporter’s metadata comparison did not expect. That failure is preserved; it is not a model-quality result.

## Reviewed exporter/scorer source pair

The [exporter V3 packet](first_cell_evaluation_export_v3/README.md), its [source review](first_cell_evaluation_export_v3/EXPORT_SOURCE_REVIEW_V3.json), and the [scorer V3 packet](first_cell_evaluation_score_v3/README.md) define the fixed five-arm feature-only prediction export followed by bounded streaming scoring. The source review status is `PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR`; it approves only the local producer/consumer source relationship, not a remote admission, export, score, or quality result. The source review and exact producer/scorer pins are consumed by the pending admission V5 and score admission V6.

## Pending export V5, unit V2, D2

The [admission V5 README](first_cell_evaluation_admission_v5/README.md), [body template](first_cell_evaluation_admission_v5/ADMISSION_BODY_TEMPLATE_V5.py), [renderer](first_cell_evaluation_admission_v5/render_admission_remote_v5.py), [rendered local body](first_cell_evaluation_admission_v5/REMOTE_SUBMISSION_V5.py), [binding](first_cell_evaluation_admission_v5/ADMISSION_SOURCE_BINDING_V5.json), [inverse proof](first_cell_evaluation_admission_v5/ADMISSION_V5_INVERSE_PROOF.json), and [focused review-shape fixture](first_cell_evaluation_admission_v5/test_admission_v5_export_review_contract.py) bind the V3 export review, the two zero-unit preflight histories, and the failed V1-unit attempt. The rendered body is 248,948 bytes / `e24a6fc87e72a43056d30e1b4edf2cb12e410a37f549a858e89420e54cdb38c0`; its binding is 17,800 bytes / `76359e9a889ed9f0647d4aeb901c01e1b7cfa9b44bf885481bf1cd7c87b7fd1c`. V5 selects the fresh V2 export unit and D2 evidence root. The rendered body awaits independent source review; it has not been submitted or launched. Do not infer an actual admission or launch from these files.

## Conditional CPU score and result capture

- [Score admission V6](first_cell_evaluation_score_admission_v6/README.md) binds the source-reviewed exporter/scorer V3 pair and checks the exact V3 `reviewed_files` array and join metadata. Its [binding plan](first_cell_evaluation_score_admission_v6/SCORE_ADMISSION_BINDING_PLAN_V6.json) keeps the future export acceptance and exact-stop pins null. The [focused fixture](first_cell_evaluation_score_admission_v6/test_score_admission_v6_review_metadata.py) and [local checks](first_cell_evaluation_score_admission_v6/LOCAL_CHECKS.md) are local metadata/source evidence only.
- [Score result capture V2](first_cell_evaluation_score_capture_v2/SCORE_RESULT_CAPTURE_TEMPLATE_V2.py), its [inverse proof](first_cell_evaluation_score_capture_v2/SCORE_RESULT_CAPTURE_V2_INVERSE_PROOF.json), [source proof](first_cell_evaluation_score_capture_v2/SOURCE_PROOF_V2.json), and [independent review](first_cell_evaluation_score_capture_v2/INDEPENDENT_SOURCE_REVIEW_V2.json) bind the future score unit identity guard. Actual score ADMISSION/LAUNCH literals remain unbound, and no score result is present.

## Root-owned admission, status, acceptance, reader, and stop helpers

The [root capture index](first_cell_evaluation_admission_v1/ROOT_CAPTURE_INDEX.md) is the complete navigation map for the versioned root-side consumers and focused proofs. Current versioned references include:

- Root admission capture [V2](first_cell_evaluation_admission_v1/ROOT_CAPTURE_EXPORT_ADMISSION_V2.py), its [source review](first_cell_evaluation_admission_v1/ROOT_CAPTURE_EXPORT_ADMISSION_SOURCE_REVIEW_V2.json), and [proof](first_cell_evaluation_admission_v1/ROOT_CAPTURE_EXPORT_ADMISSION_V2_PROOF.json).
- Export status capture [V2](first_cell_evaluation_admission_v1/ROOT_CAPTURE_EXPORT_STATUS_V2.py), its [source review](first_cell_evaluation_admission_v1/ROOT_CAPTURE_EXPORT_STATUS_SOURCE_REVIEW_V2.json), and [proof](first_cell_evaluation_admission_v1/ROOT_CAPTURE_EXPORT_STATUS_V2_PROOF.json).
- Root acceptance capture [V5](first_cell_evaluation_admission_v1/ACCEPT_EXPORT_CAPTURE_V5.py), its [source review](first_cell_evaluation_admission_v1/ACCEPT_EXPORT_CAPTURE_SOURCE_REVIEW_V5.json), and [focused proof](first_cell_evaluation_admission_v1/ACCEPT_EXPORT_CAPTURE_V5_PROOF.json).
- Bounded export reader [V3](first_cell_evaluation_admission_v1/READ_EXPORT_TEMPLATE_V3.py), its [source review](first_cell_evaluation_admission_v1/READ_EXPORT_SOURCE_REVIEW_V3.json), and [local checks](first_cell_evaluation_admission_v1/READ_EXPORT_LOCAL_CHECKS.json).
- Exact-stop template [V2](first_cell_evaluation_admission_v1/STOP_EXPORT_TEMPLATE_V2.py) and its [source review](first_cell_evaluation_admission_v1/STOP_EXPORT_SOURCE_REVIEW_V2.json).

These helpers and reviews define future root-owned capture gates; none creates an export acceptance or exact stop by itself. Follow the versioned review status, not the file version alone.

## Scope

This is a navigation index only. It does not run tests, query B or systemd, read prediction/target/model/data payloads, stage files, launch workers, score outputs, or make a quality claim. Root owns the current continuation, actual admission and capture, exact stop, publication, and higher-level navigation/ledger.
