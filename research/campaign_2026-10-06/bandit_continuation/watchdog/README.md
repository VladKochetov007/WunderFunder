# Lane watchdog

`scripts/watchdog.py --once --notify` runs every 3 minutes (user timer `wnn-watchdog.timer`, unit files in `units/`; install with
`cp units/* ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now wnn-watchdog.timer`).
It checks: the receipt-owned Vast instance (status, ssh, GPU idle with a queue, refill daemon, stalled or failed jobs, spend),
the local GPU queue (closed admission barrier, items moved to failed/launch_failed/blocked, idle GPU with a queue, expected timers),
the Kaggle capacity kernel (ERROR/CANCEL, completion), failed wnn systemd units, disk and RAM.
State changes append to `ALERTS.jsonl` (ALERT / RECOVERED rows) and raise a desktop notification; a broken check is itself an alert.
The Claude session follows `ALERTS.jsonl` with a persistent monitor and checks `.runtime/watchdog_heartbeat` for staleness.
