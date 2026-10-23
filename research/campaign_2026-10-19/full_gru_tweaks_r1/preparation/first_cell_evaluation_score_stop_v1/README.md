# Conditional exact stop for the first-cell score unit

This packet is a local, unbound template for one possible stop of the future
`computer-vision-homework-fg19-first-cell-score-v1.service` invocation. It stops
only after ROOT has written and pinned a successful score-result acceptance. It
does not run, inspect, or stop any unit by itself. The invocation ID and ROOT
acceptance byte count, SHA-256, and base64 remain unbound, so this template is not
ready to render or use.

The proposed ROOT acceptance contract is
`fg19.first-cell-score-root-acceptance.v1` /
`ACCEPTED_FIXED_FIRST_CELL_DEVELOPMENT_SCORE_REPORT`. The stop gate joins its
exact bytes to the score unit and invocation, requires strict native terminal
integers for exit `0-1-0`, and checks the result pin against
`fg19.first-cell-five-arm-streaming-score.v1` /
`FIXED_FIRST_CELL_DEVELOPMENT_EVIDENCE_SCORED`. The result pin must have a
nonempty path, positive strict integer byte count, and lowercase SHA-256. The
acceptance must still say `PENDING_EXACT_STOP`.

Before stopping, the live unit must match the same invocation and successful
exited state, with `MainPID=0`, `ExecMainCode=1`, `ExecMainStatus=0`, and
`Result=success`. The campaign parent slice is scanned with fixed directory,
entry, and byte caps; an absent, redirected, root-level, or populated cgroup
fails closed. The sole stop command targets the score unit. Afterward, the unit
must be `not-found` / `inactive` / `dead`, PID 0, with an empty invocation, and
the campaign cgroup must remain empty. Bounded systemd command capture enforces
deadlines and output caps while reading. Any uncertain stop outcome is retained
for inspection and is never retried.

The stop output schema is `fg19-first-cell-score-exact-owned-stop.v1`. Its
fourteen fields record the exact score and result digests, unit states, cgroup
observations, stop command status, and a scope statement. The stop status itself
makes no claim about score quality or public transfer.

The frozen scorer CLI writes a create-only `RESULT.json` with top-level score
schema/status, source and split pins, fixed model receipts, export and registered
data pins, five pooled arm scores, four paired component-bootstrap summaries,
and limitations. Each arm result has `t0`, `t1`, pooled `wp`, `selected_rows`,
and `metric_id`. Each contrast uses the canonical bootstrap fields
`candidate_wp`, `reference_wp`, `mean_diff`, `bootstrap_sd`, `ci95`, `p_le_0`,
`n_sequences`, `n_components`, `n_boot`, `seed`, `metric_id`, and
`centered_moments`. A compact ROOT acceptance can retain those scalar tables
with the file pin; it must not copy predictions or moment arrays.

The only checks run here are the new receipt/state/cgroup boundary fixtures and
Ruff on the two files in this folder. No score, systemd query, stop, B action,
prediction, data, cache, checkpoint, or model payload was accessed. Independent
source review is still required before any future binding or use. See
[LOCAL_CHECKS.md](LOCAL_CHECKS.md) for exact test results and source pins.

## Later score-v2 ROOT stop record

The next local score attempt uses a separately reviewed ROOT capture driver at
[`first_cell_evaluation_score_stop_v2/README.md`](../first_cell_evaluation_score_stop_v2/README.md).
That version changes only the local exact-stop record path to `score_v2`; the
systemd unit remains the accepted `computer-vision-homework-fg19-first-cell-score-v1.service`.
The V1 driver and its history remain unchanged.
