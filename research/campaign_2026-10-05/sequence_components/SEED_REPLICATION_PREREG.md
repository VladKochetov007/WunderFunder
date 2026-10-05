# E469: fixed replication of all ten component arms

Run all ten E467 arms, including negative pilots and their controls, on seeds
271829 and 271830. Same frozen 24-fit/16-test exposed component cohort, complete
20,000-row sequences, one epoch, B4/chunk256, 474 optimizer updates, unchanged
loss, normalized features, parameter counts and official scoring. No arm is
pruned, tuned or selected; no fresh-cohort/public-success claim.

A separate source adds an explicit seed argument and records its own identity.
The original E467 source/checkpoints/results remain immutable. Pair each arm
against its declared control within seed. Report all three seed effects and
variation; conditional component intervals do not replace seed replication.

20 scientific fits, two sequential ten-arm jobs. At most one local GPU worker,
2 CPUs / 6 GiB worker RAM / 2 GiB allocator / 3 GB observed GPU stop. Parent
now enforces 12 CPUs / 30 decimal GB RAM (27 GB soft) / no swap; use at most
10 CPUs and 20 GB for a future worker plus support headroom. Each job capped
3600 seconds, projected about eight minutes from E467, charged to the existing
80-device-GPU-hour budget and current local block. Abort on the existing
scientific/resource guard and retain failure; no automatic retry.

The persistent five-minute checker monitors the controlled attempts and this
block's systemd service. A completed job is scored before the next starts.
