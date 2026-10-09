# Fold1 GRU384 seed3 recovery gate R1

This is a separate, inert recovery path for the already admitted seed3 attempt
`fold1-gru384-s3-fit-r1`. It does not alter the original queue, source, output,
durable R1 sequence, or seed4 waiter. It does not launch during preparation.

If the original fit exits zero, preparation returns `recovery_not_needed`; the
existing R1 postfit gate remains the owner of seed4. Only an exact 7200-second
timeout exit (`124`) with a terminal unit, zero OOM deltas, in-budget RAM/VRAM,
unchanged source pins, and a valid epoch-boundary state can produce a request.
Other failures remain `needs_review` and do not trigger a resume.

The request binds the post-terminal `training_state.pt` SHA, contract SHA,
`next_epoch`, and a command cloned from the original seed3 command. It changes
only the timeout (capped at 14,400 seconds), output name, and `--resume-state`
path. `--epochs 4`, optimizer/scheduler horizon, data membership, source and
all scientific flags remain fixed. The new output path is
`artifacts/runs/bd26_w384raw_s3_f1_resume_r1`; the original directory remains
untouched. A new root admission must bind the exact request SHA, state SHA,
contract SHA, command hash, source pins, and resource envelope before `dispatch`
can use `scripts/bandit_resources.py`.

After a successful recovery, `audit` reuses the frozen R1 producer verifier,
then proves that `provenance.json` names the exact original state and that the
final training-state contract is unchanged with `next_epoch == 4`. Seed4 still
needs a separate root admission and output/provenance gate. No scores are
computed here.

Static source packet: `RECOVERY_GATE_PACKET_R1.json`. The current checkpoint
state SHA is intentionally absent until the original attempt is terminal.
Review/installation belongs to root; this package is not an admission or an
active watcher.
