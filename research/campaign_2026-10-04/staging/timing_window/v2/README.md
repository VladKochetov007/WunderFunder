# CPU timing window v2 (draft)

This new identity replaces no v1 evidence. It preserves the failed v1 attempt,
controller output, unit, and worker artifacts. The v2 sequence uses fresh
attempt names and fresh output directories so no timing result can be
overwritten or silently reused.

After the exact E426 fit service is terminal-successful and its four attempt
records are verified, the controller waits up to 900 seconds for load1 at or
below 0.3, no active CUDA compute processes, and zero CUDA utilization. If
that admission window does not appear, it records `timing_deferred` and exits
zero so E425 science can proceed. After admission, each unchanged worker keeps
its own `--max-load1 1.0`, CPU15 affinity, one-CPU/2-GiB limits, and 1,800-second
timeout. The controller rechecks the 0.3 admission threshold before starting
the next worker. A worker failure, lost handle, or identity mismatch fails
closed; the controller never retries a stopped attempt.

The order is seed4 i16 versus the fixed seed3 i16 control, then seed3 i16
versus q8r/q8x. Both use the original worker sources, actual v3 native
callbacks, three complete 20,000-row E457 streams per trial, five alternating
paired repeats, and CPU15. The E450 first worker writes to the v2 directory;
the Q8 worker's numerical/parity gates and the E450 measurement loop are
unchanged. The Q8 source module hash remains the exact value pinned by the
existing E450 wrapper.

The staged unit uses a 25%-CPU / 512-MiB cap and six-hour startup timeout. Root
must update the E425 unit dependency to require and order after
`wnn-timing-window-v2.service`; the live v1 unit and E425 drop-in remain
untouched. Root seals the top-level path-only drafts only after the sources
and prerequisites pass review. This packet has not been sealed or launched.

Run the separate stdlib-only consistency fixture from its draft spec before
sealing the timing workers. It checks identities, order, outputs, limits,
load thresholds, and that both v2 workers point to the unchanged timing
sources.
