# Independent review: E488 resource-only amendment v3

Reviewed `RESOURCE_ONLY_ADMISSION_AMENDMENT_V3.md` at SHA-256
`9309269da0c263053f8fd767142e29dc6a127ac30477c35074e6589535ef9072`.
The amendment is separate from the original E488 protocol and the original
900-second gate remains failed. It proposes an 1,800-second total operational
allowance before any full trajectory or new evaluation scoring, while retaining
the same 12 trajectories, arms, epochs, optimizer, data, source, thresholds,
memory caps and telemetry. It consumes only 0.5 GPU-hours from the existing
authorized campaign budget; it does not increase the 2-GiB allocator limit,
3-GB device stop, 2-CPU/6-GiB worker envelope or parent RAM/CPU limits. The
actual probe record reports exit 0, verified teardown, no failure/signals,
11.659668447 charged seconds, peak observed VRAM 385,875,968 bytes and peak
RAM 1,385,603,072 bytes. Its probe artifact reports 12 complete cells and
exact native parity. No full fit or test scoring had been admitted at review.

One arithmetic clarification is required before sealing the new operational
spec. The reported `1075.263010451` projection uses the Python wrapper's
`probe_seconds` of 5.645549 seconds; the sealed runner attempt charged
11.659668447 seconds. Recomputing with the same measured full-fit update
projection (633.078307739 seconds), the registered 1.5 multiplier and 120-
second overhead gives:

`633.078307739 × 1.5 + 120 + 11.659668447 = 1081.277130055 seconds`.

This remains comfortably below 1,800 seconds. Bind this charged-time value in
the amendment's pre-admission predicate and set the full-worker timeout to
`floor(1800 - 11.659668447) = 1788` seconds. The 120-second margin remains
unchanged. Do not describe 1,075.263 seconds as including the entire charged
attempt.

**Conditional GO for the resource-only amendment after that projection
correction is recorded and the v3 spec binds the unchanged source/probe/data
identities.** Preserve the 900-second failure receipt as-is; do not rewrite
the prior result as a pass. The amended fit may start only if the controller's
current no-OOM/teardown and available-budget checks pass, before the registered
admission cutoff, and all 12 trajectories remain one indivisible complete
matrix under the amended timeout. No scientific threshold or evaluation
selection may change. This reviewer inspected only the operational amendment
and fit-free probe receipts; no evaluation labels or full-fit outputs were
read.
