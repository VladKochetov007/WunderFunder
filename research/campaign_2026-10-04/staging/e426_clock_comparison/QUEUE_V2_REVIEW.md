# E426 fixed-export queue v2 source packet

This packet is staged only. It has not been sealed, run, or approved for
scientific use. It preserves the earlier v1 export/score drafts and their
source review. The new v2 identities have fresh runner attempt names and a
fresh score-report path; the prediction cache paths match the previously
reviewed E426 naming contract and are currently absent.

The CPU provenance preflight reads only the three finished constant/aligned/
shuffled fits. It requires each exact sealed fit spec and successful runner
record, verifies hashes for the epoch-3 checkpoint, provenance, history, and
training state, checks all four epochs' 373 completed batches and 7,460 update
attempts, and confirms that each fixed checkpoint is tensor-equal to the saved
`evaluation_ema.state`. Its report explicitly contains no predictions or
scores. The draft limit is one CPU, 2 GiB RAM, no GPU, 600 seconds.

The queue waits for the exact E426 fit-service success and terminal timing-v2
service. A normal timing result must be `timing_complete` or the preregistered
`timing_deferred` status and the oneshot service may be either inactive/exit 0
or active/exited/exit 0. The one observed pre-callback relative-path failure is
accepted only through the separately root-approved, SHA-pinned incident
release at `checks/ROOT_TIMING_INCIDENT_RELEASE.json`; the gate checks the exact
result, failed attempt and log bytes, both failed service handles, zero PIDs,
no OOM, and the science-only permission scope. That branch forbids timing
readiness, retry, or runtime claims. All unknown timing failures stop the
queue. The controller runs the sealed provenance preflight only after this
timing gate and requires its output to be bound by its successful runner
record. Before every export reservation,
it conservatively sums all E426 GPU charged time (or reserved timeout for any
unfinished E426 GPU attempt) and refuses a job if its full sealed timeout
would exceed the 24-hour E426-local reservation. The campaign runner's global
80-device-GPU-hour ceiling remains authoritative.

The controller runs the three unchanged `predict_run.py` fixed-epoch-3 FP32
exports one at a time. For each, it checks the exact worker service reached
`inactive` with exit 0, that the runner's artifact hashes match current bytes,
that the sidecar binds the fixed checkpoint and fold manifest, and that the
NumPy header is FP32 C-order `[1873, 20000, 2]` with exactly 1,873 typed
canonical VALID IDs. Only after all three exports pass does it run the
existing E426 paired-screen validator/scorer as one CPU job. Neither source
checkpoint selection nor scoring math changed; the score wrapper keeps the
previously reviewed fold-0, fixed-epoch, class and bootstrap gates.

The three export specs are sealed only after the successful preflight report
exists. The score spec remains a PATH-only draft until all three export
attempts and output hashes pass; the controller then seals that draft once and
checks its exact path fields, every input/source hash, and code-closure digest
before scoring.

Any failed attempt is terminal for this controller. It records `stopped` and
does not retry. An interrupted controller requires an explicit `--resume`,
and existing attempt identity, sealed specs, and output hashes must still
match. Root must review these drafts, source closure, and service ordering,
then admit the exact queue unit after the timing-v2 service and E426 fit gate.

The source controller performs only artifact/header checks itself; the paired
scorer remains responsible for the full finite-value, mask, prediction, and
score validation before metric computation. No runtime, score, or scientific
result is claimed by this staging packet.
