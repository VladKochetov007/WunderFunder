# Prepared verifier V1 — first bounded attempt

This was the first prepared restricted terminal checker and was used once. Its consumer load failed on the exact `TorchVersion` metadata global; the native S3 run was not changed or replayed.

- [Prepared packet](PREPARATION.json) and [remote submission](REMOTE_SUBMISSION.py).
- [Checker source](../teacher_terminal_check.py), [synthetic preparation receipt](../TEACHER_TERMINAL_PREPARATION.json), and [source review](../TEACHER_TERMINAL_REVIEW.json).
- Actual result: [V1 failure record](../../records/teacher_s3_v1/terminal_verification_v1/FAILURE.json). The scoped fix and successful bounded consumer are [V2](../teacher_terminal_verifier_v2/README.md).
