# Prepared verifier V2 — final packet used

V2 is the final restricted-loader preparation. It adds only the canonical `TorchVersion` type to the scoped safe-global set while keeping `weights_only=True`; source review and the final local checks passed. This packet was used for the successful bounded S3 integrity verification.

- [Final prepared packet](PREPARATION.json), [remote submission](REMOTE_SUBMISSION.py), [V2 checker](../teacher_terminal_check_v2.py), [synthetic receipt](../TEACHER_TERMINAL_PREPARATION_V2.json), and [source review](../TEACHER_TERMINAL_REVIEW_V2.json).
- [Root-final checks](../teacher_terminal_verifier_v2_checks_root_final/README.md) capture two passing tests, Ruff check/format passes, and generator validation. Earlier formatting failure is preserved in [the historical check round](../teacher_terminal_verifier_v2_checks_final/README.md).
- Actual bounded result: [V2 verification](../../records/teacher_s3_v1/terminal_verification_v2/README.md), [terminal acceptance](../../records/teacher_s3_v1/TERMINAL_ACCEPTANCE.json), and [independent actual review](../ACTUAL_TERMINAL_REVIEW.json). No inference or quality grading occurred.
