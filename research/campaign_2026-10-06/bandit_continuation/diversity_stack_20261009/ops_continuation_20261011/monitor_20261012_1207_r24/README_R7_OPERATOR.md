# R7 provider and terminal handoff

R7 version 1 was accepted at 2026-10-12 13:03:52 UTC. The exact read-only
poll at 13:13:54 UTC reported `KernelWorkerStatus.QUEUED`; this is not evidence
that the profile or any fit has started. The poll also recorded the account
quota snapshot. Kaggle omitted a session ID, so the usage values are not
attributed to R7.

`provider_targets_active_20261009_r2.json` adds R7 after the seven existing
targets. The existing poller service already reads this file. The target binds
the accepted kernel version, R7 root release, version-2 source dataset,
version-1 asset dataset, and unique terminal receipt/output paths.

`collect_r7_terminal.py` checks the exact root-release and accepted-push
receipts before polling. While status is nonterminal, it only records the
read-only status/quota snapshot. At terminal status it calls the existing
bounded `collect_terminal_outputs.collect_terminal_target` implementation to
download logs and outputs, with an 8 GiB cumulative byte ceiling, 4 GiB free
disk reserve, and 800-second runtime bound. The generic collector receipt
proves download completeness and file hashes; a later artifact audit must
separately classify whether the pipeline reached CP5 and produced its complete
prediction export. No fit retry or provider write is performed by this
operator wrapper.
