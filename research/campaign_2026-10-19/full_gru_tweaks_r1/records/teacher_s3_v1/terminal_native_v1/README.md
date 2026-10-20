# S3 native terminal capture

The seed-3 native unit `computer-vision-homework-fg19-teacher-s3-v1.service` (invocation `02a8c00b433b49a185c2700624397769`) exited with integer status 0. This capture initially established only native process completion; it explicitly did not yet accept checkpoint integrity.

- [Normalized terminal evidence](TERMINAL_EVIDENCE.json), [observation](OBSERVATION.json), [transport receipt](TRANSPORT.json), and preserved [stdout](STDOUT.raw)/[stderr](STDERR.raw).
- Later checkpoint integrity is recorded in the parent [terminal acceptance](../TERMINAL_ACCEPTANCE.json) and the successful [restricted V2 verification](../terminal_verification_v2/README.md).
- The subsequent [exact stop](../terminal_exact_stop/README.md) records stopping the already-exited native unit and successful verifier. This is integrity evidence only; no inference, grading, or quality result is claimed.
