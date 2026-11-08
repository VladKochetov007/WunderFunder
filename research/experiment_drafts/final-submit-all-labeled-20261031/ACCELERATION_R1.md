# Pipelined final-fit delivery, 1 November 2026

## Recovery applied, 02:06UTC

The user confirmed restored SSH; persistent session59354 is now connected.
The reviewed repair capsule was downloaded and its exact files verified before
use. Old waiting R3 group3643146 and W18 watcher3612009 were checked against their
launch receipts and stopped; no successor fit output/receipt existed, neither
process owned a child training stage, and the same healthy FBC identity was
verified afterward. No scientific fit or foreign process was signaled.

Replacement wrappers are R3 handoff3661172, TLead5 profiler3661173, TLead6
profiler3661176 and W18 profiler3661178. The new R3 source is67ab6056…c4bd67;
its fresh immutable intent is present in final-submit-r3-r2. Both TLead-r4
profiles have started; W18 waits for its actual FBC/cache prerequisites. The
early W18 publisher remains healthy, waiting on11/13 ancestry receipts. The
14 completed scientific fits and current FBC training are retained.

The interruption and earlier deployment snapshots below are historical.

Both TLead-r4 profiles subsequently passed the unchanged independent checker
(exit0). Each performed80/80 optimizer updates including warmup, with finite
loss/gradients, zero OOM and measured6.56sequences/s. Peak allocated GPU memory
was2,482,946,048bytes and host RSS about7.72GB. Both canonical four-epoch
scientific fits then started through R3 on GPU3/4; their owned stage wrappers
are3667647 and3667654. Together with the original FBC this is17/18 scientific
fits started,14 complete, three concurrent fits and only W18 not yet started.
The old controller's own counters remain15started until the final lock-bound
R3 adoption; do not confuse that bookkeeping delay with missing successors.

At the measured rate, a TLead fit costs12,480×4/6.56 ≈7,610seconds
(2hours7minutes), plus startup/checkpoint overhead. W18's earlier artifact
path remains independent of those completion times. Its07:00–08:00MSK estimate
still includes the unfinished FBC/export/cache, W18 fit and actual ZIP checks.
The [received recovery proofs](receiving/ssh-recovery-r1/receipts/profile-env-repair-live-r1.json)
bind these real launches and preserve the failed attempts.

Latest qualification:14/18 fits were complete at the last live observation.
Both TLead engineering startups then failed before CUDA with
`ModuleNotFoundError: scripts`: the watcher's in-process sys.path did not become
the child process's PYTHONPATH. No successor scientific fit started. The reviewed
repair explicitly supplies S:S/src, preserves both failed-r3 attempts, and uses
fresh TLead-r4 profile paths. All W18 request paths also point directly to inputs;
replacement R3 proofs use final-submit-r3-r2. The global profile budget remains
24 attempts:15 existing approvals +4 historical startups +2 failed TLead
startups +3 remaining successful profiles. No scientific budget changes.

The repair capsule `final-control-profile-env-repair-67ab6056.tar.gz` is staged
on the build-input release but has not been applied: the sole SSH connection
reset, and four bounded recovery connections timed out/closed before
authentication, including one60-second banner wait and one noninteractive
no-QoS transport probe. No password was transmitted in these recovery probes.
The GitHub reporter remained live at01:57:47UTC and showed FBC around3.05/4
epochs, so an SSH observation failure is not evidence that training stopped.
Stop only the exact old waiting R3 wrapper3643146 and W18 watcher3612009 before
launching replacements, after checking their recorded identities and absence of
successor fit children. Preserve R2/FBC and early publisher3647718. The latter
passed its actual Linux preflight and was healthy in waiting_for_w18_closure
with11/13 ancestors verified. The07:00–08:00MSK delivery estimate is provisional
while remote access prevents completing this profile repair.

The user requests maximum delivery speed. This amendment changes scheduling and
artifact dependencies only. Numerical source, all 18 scientific roles, epochs,
batch sizes, precision, fitting membership and the original deadline stay fixed.
No healthy fit was stopped, restarted or repeated.

## Actual deployment

One persistent SSH session42104 is used. Only the exact owned R8 Python supervisor
PID3509652 was paused; its GNU timeout, R2 director and independent fit/export
process groups continue. The pause proof records PID/PGID/birth, source/argv and
before/after process identities. The R3 sidecar has passed this live preflight and
written its immutable handoff intent. Its active source SHA256 is
`db19ddac4953a466a46a5184b7de8521373c25c99e8310fca9f37c4ca3627770`.

The three disjoint successors wait only for their own approved device profile and
parents. TLead seeds5/6 use GPU3/4 and no longer wait for the unrelated FBC/cache
branch; W18 uses GPU5 after its six-teacher target cache completes. Each fit uses
the unchanged R2 execution helper, canonical GPU lock, logs and child receipts.
No shared controller-state write occurs while R2 owns its director lock. After
all three successors finish, R3 adopts their qualified receipts under the released
lock and resumes that exact R8 Python process. GPU stages end before08:35UTC,
supervisor resume is bounded before08:45UTC, and the original09:05UTC hard
deadline remains in force. No foreign process is signaled.

Four failed R3 startup guards were retained before handoff acceptance. They
incorrectly assumed explicit R8 path flags/no `-u`, a live lane counter equal to
running workers, and boolean `timed_out` in a still-running receipt. The corrected
checks parse the frozen R8 command/default resolver, retain exact ownership and
command identities, bound the telemetry counter, and require running timeout
results to remain null. These were operational preflight attempts, not fits.

## Earlier W18 delivery

An additional detached CPU-only worker builds a separately identified W18 package
after its own exact13-role training ancestry and successful final fit receipt.
It never waits for TLead/LSTM or global18-fit adoption. Its source SHA256 is
`1726028f29a4b6fec0a06f2fb1382489df35daa7a2586488c6207d09c55313dd`.
The actual Linux/Python3.11 import and CLI preflight passed for all five pinned
helpers. Auth helper pins were corrected against exact bytes in the unchanged
scientific source capsule; the import probe runs outside the older source tree
to prevent current-directory namespace shadowing.

R12 serves one W18 directly. All69 Rust tests passed (one ignored); v3, AVX2 and
v2 old-reference20,000-row forecasts are bitwise equal to frozen R11 W18, including
reset/replay and future-prefix checks. These checks do not qualify new weights.
The actual new ZIP must separately pass Python3.11 ABI, numeric ceilings,
reset/causality, size, memory and CPU timing gates before the worker publishes it
as SemyonEpanov, verifies the actual download, and comments on PR54. The original
full-artifact worker continues unchanged. Platform upload is a human step.

All12,480 official train+valid sequences fit these submission-only models;
independent evaluation is zero. No new valid/public score or positive gain is
claimed. The first W18 ZIP is an early fallback, not the full four-model mixture.
