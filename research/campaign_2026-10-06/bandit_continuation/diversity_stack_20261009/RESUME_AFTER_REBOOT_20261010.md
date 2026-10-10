# Resume checklist after the laptop restart (2026-10-10)

State before restart: all weights, caches, predictions and run directories live on disk (`artifacts/runs`, `cache`, `artifacts`), not in /tmp.
/tmp held only rebuildable dependency trees (python3.11 venv, jax deps), pytest scratch, and copies of files that are identical in the repo
(three h230 all-data submission ZIPs, hash-verified; scratchpad model copies, byte-identical to artifacts/runs).

1. Local queue was paused with `local_queue_r1/STOP` so no fit is killed mid-epoch. After boot: `rm research/campaign_2026-10-06/bandit_continuation/local_queue_r1/STOP`.
   Pending (in order): fold-0 ep3 default seeds 4 and 5 (fit + export), fold-1 h230 student with bootstrap h8 (fit + export).
2. Persistent units created for things that were transient (they would not survive a reboot):
   `wnn-vast-tick-lane-b.timer` (Vast lane B controller tick, cap 36) and `wnn-slice-filecache-guard.service` (RAM guard). Check
   `systemctl --user list-timers | rg wnn` and `systemctl --user is-active wnn-slice-filecache-guard.service`.
   Already persistent: wnn-hpo@teacher_v2, local-queue, watchdog, kaggle-rolling, kaggle poller/collector, grade-bank timers.
   Lane A (Vast account 1) is destroyed; its transient tick timer is intentionally not recreated.
3. Vast lane B (instance 55080658, account `vastai2`) keeps running during the restart; HPO ticks pause and resume by themselves.
4. The in-session /loop cron job (`60b0db18`) dies with the session: re-create it with `/loop 30m status? utilization? reflect on research results landed, write to research notes, continue and make robusltly better and research promising directions or what i have made`.
5. Verify after boot: `python -m hpo.orchestrator status --study teacher_v2`, `scripts/watchdog.py --once`, `nvidia-smi`, `df -h /`.
