# E462 / E450 quiet CPU timing window

This staging tree contains a fixed one-CPU timing pair before E425's science
service. Root seals the two worker drafts only after reviewing their exact
input and source closures. The timing-window systemd unit is Type=oneshot,
limited to 25% CPU and 512 MiB, and its staged E425 drop-in has both Requires
and After on that unit. A recorded quiet-window deferral exits successfully,
so E425 can continue without timing evidence.

The controller first requires E426's exact fits_complete state, inactive
successful service, and all four successful teardown-verified attempt records.
It then requires successful E462 build/parity/score and actual E450 seed-4
20-stream parity/reset artifacts. It waits at most 900 seconds for host load1
at or below 1.0, no active CUDA compute PIDs, and zero GPU utilization. If that
quiet state never arrives, it writes timing_deferred and exits zero. A failed
timing worker or mismatched attempt never triggers a retry.

The first worker uses frozen staging/e462/time_public_q8.py unchanged:
original seed-3 i16 versus q8r and q8x, v3 native level, three complete E457
20k callback streams, five alternating paired repeats, and the module's
_load_streams / _trial loop. The second worker imports that exact loop and
compares the original seed-3 i16 ZIP to the actual seed-4 i16 ZIP on the same
three streams, in five alternating pairs. Both workers require CPU15 affinity,
load1 <= 1.0, one CPU, 2 GiB RAM, no GPU allocation, and 1,800-second limits.
The seed-4 journal records the 99-row warmup count, 59,703 finite prediction
rows per trial, elapsed time and optional hardware cycle counters. Neither
worker scores predictions or changes a package.

Only after both timing reports finish does the controller derive the
39,000,000-row linear projections and compare them descriptively with 52
minutes. That extrapolation is not a measured competition runtime or upload
readiness decision. No timing worker has been launched from this staging tree.

Run python research/campaign_2026-10-04/staging/timing_window/build_timing_specs.py
to refresh the two unsealed drafts after prerequisites and source bytes are
final. The builder uses only standard-library reads and hashing.
