# Resource admission review R1

This folder contains an independent code review of commit `b185e7048785e1c6db60303dbd228c8ce803bd3a`, which replaces the local durable launcher's fixed GPU-memory thresholds with a free-memory policy. The review passes the GPU-limit removal, allocator wiring, reserved-environment handling, historical-record parsing, and watchdog-path checks. It also finds a confirmed RAM-admission gap: simultaneous reservations can exceed the campaign's 17 GB committed target even when the two existing checks pass. That gap does not block the current zero-active single-worker first launch, but must be corrected before concurrent worker/support admission.

- [Independent review](INDEPENDENT_RESOURCE_REVIEW_20261013.md) gives findings, evidence, scope, and the conditional first-launch disposition.
- [Machine-readable review receipt](INDEPENDENT_RESOURCE_REVIEW_20261013.json) records commit identities, checked paths, tests, and findings.

The candidate was inspected in its isolated worktree against parent `fda03869747eb1a0938ebf9676c20d8db1dd1310`. Targeted local tests used the main checkout's interpreter with the candidate worktree on `PYTHONPATH`; no GPU, Kaggle, Vast, or billing action was taken. The canonical strategy handoff should link this folder from the campaign README/frontier when the review is integrated.
