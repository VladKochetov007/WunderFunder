# Crash / alert postmortem 2026-10-10..11 (user: react to every watchdog alert; prevent repeats)
Why I missed alerts: the watchdog (scripts/watchdog.py --once) only prints NEW alerts per pass and writes them to watchdog/ALERTS.jsonl + desktop notification; my loop invoked it and saw only recoveries. Fix: scripts/alerts_unread.sh (cursor-based reader of ALERTS.jsonl) is now the first command of every loop pass.
| alert | root cause | fix / prevention |
|---|---|---|
| vast_job_failed long_stable_swa_e14, muonall_e3, muonhidden | CUDA OOM at start: feed admitted 9 jobs on 2x24GB | remote_refill.py: launch only on GPU with >=7000 MiB free; auto-requeue OOM/CUDA failures up to 3x (.tryN); MAX_JOBS 7 |
| local_failed/blocked 881 hpo52 s4 + barrier closed + oom_kill | CPU agent (GBM probe) 11 GB RSS pushed wnn-campaign.slice over 18 GB aggregate RAM | agent budgets: <=1 core, <=3 GB RAM, explicit in every agent prompt; STOP_ADMISSION barrier reviewed + reopened with REVIEW.md |
| local_failed 885 prop0lwf s6 + blocked cascade | runner treated phase `admission_recheck_failed` (parent RAM 9 GB + 9 GB reserve > 18 GB) as permanent failure | scripts/local_queue_runner.py now retries both admission phases (<=40); items requeued |
| vast_job_failed d_pre1/d_pre2x/d_prop0_pre1 | v7 pretrain flag incompatible with --save-training-state in base job argv | jobs rebuilt without it and requeued |
| vast_job_failed d_warp | race: scp of train_rnn.py before arch_v8.py -> ImportError ClockBankModel | requeued; rule: deploy to stage dir then atomic mv, never mid-run partial copies |
| vast_job_failed w13_dsq_prop0 (+ older w12 dsq) | scripts/distill_pipeline.py + build_distill_cache.py missing on host | shipped to /root/wnn/scripts; job requeued |
| unit_failed kaggle-poller | transient, recovered by itself (poller Finished every 5 min since) | none |
| hpo_* no_progress/ticks_stopped | deliberate: user stopped HPO | .runtime/hpo_disabled mutes check |
| vast_b_spend_high | informational: tick cap is 36 on lane b; CLAUDE.md says $20 | pending user decision |
