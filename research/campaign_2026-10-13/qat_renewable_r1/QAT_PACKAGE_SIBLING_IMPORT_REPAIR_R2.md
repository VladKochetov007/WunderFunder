---
kind: incident
status: complete
created_utc: 2026-10-13T20:46:00Z
updated_utc: 2026-10-13T20:59:00Z
hypothesis_ids: [H-QAT-LSQ-20261013]
experiment_ids: [E-QAT-FINAL-SELF-DISTILL-20261013-R2]
evidence_paths: [tests/test_qat_submission_archive.py, scripts/templates/qat_solution.py]
---

# QAT package sibling import repair

## Question or claim

The submission template imported `qat_archive` before making its own directory available on `sys.path`. The existing offline benchmark/import consumers use `spec.loader.exec_module`, which does not establish the package directory automatically. Root static inspection found this before final ZIP dispatch.

## Mechanism and method

The template now derives `HERE` and inserts it before importing its shipped sibling. This changes only import resolution. Actor qat_python independently reviewed the three-line change and returned GO for bounded actual-ZIP verification.

## Evidence and interpretation

Validation: 14 generic tests passed with three optional fixtures skipped, then all17 passed in3.41s with the actual final S4 learned-W4 archive, exact native library, original xteam weights and portable native manifest. Root ran the fixture under1CPU/1GB/180s; no GPU fit or provider action. Template SHA256: `2c3fbba2b1c805480fc1c321242f3be74572f2a9d1320aaf3171ec7aaf4b62b2`. Final ZIP callbacks and benchmark remain separate checks.

## Limits and validity

This is import-resolution evidence, not a quality or public-transfer result. Final ZIP callback/reset/parity and runtime checks are separate.

## Decision and next step

Use the corrected template for the one selected public-test ZIP.

## Source records

[Template](../../../scripts/templates/qat_solution.py), [actual archive fixture tests](../../../tests/test_qat_submission_archive.py), [fixed final recipe](final_self_distill_r2/RECIPE_R2.json).
