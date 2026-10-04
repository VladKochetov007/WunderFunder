# E-413 private paired volume experiment

Private kernel `vladkochetov07/wnn-e413-volume-20261004-v1-runner`, version1,
was pushed with explicit `NvidiaTeslaT4` and a7800-second provider timeout.
Provider status was RUNNING at2026-10-04 02:21:24 UTC. No scientific result or
preflight success is inferred from that provider status; live logs were empty
at the first check. The final complete output bundle will establish worker
assignments, preflight results, resource peaks, fit/export completion and scores.

The paired arms are `c20261004_volume28_s3_f0_t4v1` and
`c20261004_volume28_current_s3_f0_t4v1`. Both use fold0, seed3, the registered
combo recipe and fixed epoch3 after four epochs. Each reserves2 CPUs and10GiB
host RAM, with11GiB allocator,12GiB observed operational limit,13GiB GPU stop,
and7200-second worker timeout. The runner admits at most one worker per physical
T4. Each wrapper stays alive through preflight, fitting, prediction and scoring;
its physical device remains reserved for that entire sequence.

Each worker first runs the current actual-trainer CUDA replay and training-state
tests. A nonzero preflight prevents that worker's fit and causes the runner to
stop the affected batch. On success it trains, exports full-validation
predictions from fixed epoch3 and computes the existing frozen scorer's fold0
single-arm report. All checkpoints, training state, history, provenance, JUnit,
step report, predictions, prediction hashes and score report are declared outputs.
The paired scientific comparison belongs to the controller after both exports.

Final sealed code SHA-256:
`bbdba0e2810339765ce9b655fcf1dfaa292fb6642734c23eb894509b2972c9ee`.
All41 Python files in the root frozen snapshot
`96ca4b1e61823bec3953158bea12124be1efa3ec29493f7902d3eeea5be885a3`
were checked byte-identical. The transport bundle is136,380 bytes. Raw data and
23 reusable assets are mounted from their existing private datasets; each file
is SHA-verified before any worker launches. Input symlinks are unlinked after
execution so provider output capture cannot copy the mounted inputs. The saved
`bundle_before_input_cleanup` and `.pre_cleanup.json` drafts were never pushed.

Quota before launch:29.97 GPU hours and20.00 TPU hours remaining. Even maximum
7800-second charging for both devices consumes less than the available quota
above its20% reserve. The TPU hardware probe remains queued and uncharged.
No competition upload was performed.
