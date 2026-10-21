# Conditional score-stop ROOT driver V2

`ROOT_CAPTURE_SCORE_STOP_V2.py` is a source-only path rebase of the V1 ROOT
capture driver. It writes under the separate local
`records/first_cell_evaluation_v1/score_v2/exact_stop` record while retaining
the exact `computer-vision-homework-fg19-first-cell-score-v1.service` unit;
the service name remains V1 even though this is a later local attempt. The
inverse AST proof demonstrates that only the output record path changes.

The existing [`SOURCE_REVIEW_V1.json`](../first_cell_evaluation_score_stop_v1/SOURCE_REVIEW_V1.json)
reviews the remote `STOP_SCORE_TEMPLATE_V1.py` and its focused fixture, not the
ROOT capture driver. A separate source-only review of the ROOT driver is
available in [`ROOT_CAPTURE_SCORE_STOP_V2_SOURCE_REVIEW.json`](ROOT_CAPTURE_SCORE_STOP_V2_SOURCE_REVIEW.json),
5,266 bytes / SHA-256
`75852e1a312953344028e8d22a9cc1ce2d2ecd98e7568ee7652de8e9a2523c31`; its
status is `PASS_CONDITIONAL_SOURCE_ONLY_ROOT_CAPTURE_DRIVER_V2_UNBOUND`. The
actual stop body, review, score acceptance, and invocation pins remain
unbound. This packet has not invoked the driver or queried B/systemd.

The outer driver uses `subprocess.run(capture_output=True, timeout=60)` and
checks stdout/stderr limits after the process returns. The call is time-bounded,
but its local pipe capture is not hard-bounded before capture. The remote
template has its own bounded commands; this distinction remains visible for
review and limits this conditional approval: no ROOT-side precapture memory
claim is made.
