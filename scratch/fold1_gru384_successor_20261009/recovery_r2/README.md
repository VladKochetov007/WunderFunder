# Fold1 GRU384 seed3 recovery gate R2

This is an inert, separate recovery path for admitted attempt
`fold1-gru384-s3-fit-r1`. It does not alter the original queue, source, output,
durable R1 sequence, waiter, or seed4 fit. Preparation and dispatch both require
the original attempt to be terminal; dispatch additionally requires a separate
root authority bound to the exact request, original terminal attempt receipt,
state checkpoint, command, data-identity receipt, and source pins.

Recovery is eligible only for the exact 7200-second timeout exit `124`, a
terminal systemd unit, explicit positive RAM and VRAM measurements, complete
zero-delta local/parent OOM receipts, and an epoch-boundary training state.
Missing telemetry fails closed. Other failures do not trigger a resume.

The gate loads the existing fold1 split validator and the pinned
`scripts/build_campaign_splits.py::source_identity` helper. It compares current
Parquet footer identity and `(device, inode, size, mtime, ctime)` against the
frozen fold manifest and saved contract without rehashing the 34.856 GB datasets
during prepare or dispatch. The trainer's existing resume-contract check still
computes and verifies the full dataset SHA before restoring RNG or beginning
optimizer updates. A portable-data-manifest environment is rejected because
the split loader would otherwise initiate another full-file verification.

The request freezes the terminal `training_state.pt` SHA, contract SHA,
`next_epoch`, original attempt-receipt SHA, and a command cloned from the
original seed3 command. It changes only the timeout (up to 14,400 seconds),
output name, and `--resume-state` path. Epoch count, optimizer/scheduler
horizon, data membership, source and scientific flags remain fixed. The new
output is `artifacts/runs/bd26_w384raw_s3_f1_resume_r2`; the original directory
remains untouched. R2 does not launch seed4.

After a successful recovery, `audit` reuses the frozen R1 producer verifier and
checks that both `provenance.json` and the embedded epoch-3 checkpoint cite the
exact original state path/SHA/epoch. It also checks the unchanged training
contract and `next_epoch == 4`. Seed4 still needs a separate root admission and
its own output/provenance gate. No scores are computed here.

Static packet: `RECOVERY_GATE_PACKET_R2.json`. The terminal checkpoint SHA is
intentionally absent until the original attempt finishes. This package is not
an admission, watcher, or authorization to launch.
