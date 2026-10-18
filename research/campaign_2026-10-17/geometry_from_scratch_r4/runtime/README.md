# R4 durable resource ledger

This folder is the dedicated `bandit_resources.py` state for the explicitly
authorized local R4 matrix. It records launch admission, the worker's systemd
unit, resource reservation, heartbeats, terminal status and any reconciliation
receipt. It shares the campaign parent slice and the existing global GPU lock.

The dedicated state keeps the older October 6 campaign's closed admission
barrier intact. It does not bypass aggregate CPU, RAM, swap or GPU checks, and
it does not admit any unrelated experiment. The frozen scientific recipe and
current source/data identities are in sibling `admission_r3.json` and `plan.json`.

The first attempt, `geometry-r4-matrix-20261018-r1`, records a pre-run
`PermissionError` because the script was invoked as a non-executable file. It
ended before runner startup, with no optimizer updates or GPU work. Attempt R2
was admitted with a 5,000,000,000-byte RAM reservation, one CPU core, no swap
and the profiled RTX 3060, then correctly stopped because the R2 admission
receipt's training-file hash did not match the actual file. It made no fit and
recorded no OOM. All three data hashes were recomputed from the actual files and
are bound in sibling `admission_r3.json`. Attempt R3 was rejected before service
creation during the periodic monitor's full-core CPU reservation. The monitor's
quota was lowered to 20% while retaining its five-minute timer. Attempt R4 then
passed fresh admission and started the fold-0 / seed-42000 control fit at
2026-10-17 21:27:24 UTC. The ledger preserves each admission attempt and its
resource checks.

A successful resource reservation means the worker was admitted; it does not
establish that training completed or that the feature improved the score. The
R4 progress, resource and score ledgers in the parent folder determine
scientific progress.

## October 18 user scope supersedes local execution

The user cancelled the multi-day local matrix in favor of one short local probe and a submission ZIP within two hours. The running first control was stopped before completing its first epoch; no quality score exists. Controller reconciliation and drain closed this matrix admission. Preserve all prior attempts and logs; cancellation is operational scope evidence. See [new local probe](../../../campaign_2026-10-18/quick_geometry_probe_r1/README.md). Deep paired confirmation is assigned to the 8×B300 draft handoff.
