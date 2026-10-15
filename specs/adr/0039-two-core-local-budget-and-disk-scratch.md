# ADR0039: two local CPU cores and physical-disk scratch

Status: accepted, 2026-10-15. Supersedes earlier five-core local allocations.
Scientific recipes, GPU authority and memory ceilings remain unchanged.

The user reports laptop freezes and full temporary storage, requiring at most
two aggregate CPU cores. GPU research continues. The live campaign slice is now
200%; the dated controller guard records the new boundary at14:23:35UTC and
rejects a five-core parent. Keep one one-core GPU worker, zero loader workers,
and keep lightweight CPU support at20% quota. When the GPU worker is idle, one
existing scorer/state check may use one core, with fresh total reservations below
two; do not overlap it with the model worker. Use the updated operational controller
with unchanged frozen trainer commands; do not raise caps for old controllers.

Both scripts/limited and the controller route temporary files and TorchInductor
caches to /home/vlad/development/wunder/research_tmp on physical Btrfs. The
wrapper defaults to one core, one thread and the campaign slice. The controller
must set temp variables in both service and model-child environments: independent
review caught that its fresh child environment initially omitted them. Root fixed
that before any new worker launch.

At inspection /tmp used297MiB of32GiB; the reported earlier full condition was not
reproduced. After a bounded open-file check, roughly210MiB of closed research
TorchInductor/pytest/review scratch was moved to physical storage with compatibility
symlinks. /tmp then used87MiB. No checkpoint, prediction, user application or
model process was deleted. Physical storage had44GiB free. This is not a diagnosis
of all desktop freezes or authority for large staging.

Stopped eight redundant timers (all wnn- prefixed): research-monitor,
local-queue-r1, campaign-kaggle-terminal-collector, vast-tick-lane-b,
campaign-kaggle-poller, collect-native-tpu-v4, collect-fold-pins and
collect-capacity-v5r2. Low-quota resource/watchdog guards remain. No fit/export was
running during maintenance. Alpha0 P1 and its valid next_epoch2 state are preserved.

Actual bounded checks passed: shell syntax; current two-core and historical
pre-boundary five-core behavior; rejection of a five-core parent; campaign
membership; actual temporary-file creation on the /home device; generated service
environment; and a clean subprocess with the same physical temp/cache environment.
The latter used20% CPU/300MB with no Torch/GPU work. Independent Luna/max review
identified the child-environment omission and confirmed its correction, issuing
GO for one one-core worker subject to the fresh live reservation gate. Ruff and
the staged-diff whitespace check also passed.

Resume the already-authorized GPU sequence only after the fresh two-core
reservation gate. The fixed six-epoch primary endpoint remains unchanged.

Sources: latest user instruction; [controller](../../scripts/bandit_resources.py),
[wrapper](../../scripts/limited), [P1 terminal state](../../research/campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_P1_TERMINAL_STATE_R1.json).
