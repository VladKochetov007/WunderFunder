# Logout incident and bounded recovery

At 2026-10-22 19:23:45 UTC the user systemd manager exited when the SSH session ended with Linger=no. Its journal shows all original experiment services stopping. This was an operator configuration error. The archive preserves the raw incident journal and the failed first recovery setup. Linger=yes is now verified. No evidence here attributes this shutdown to OOM.

The first recovery restored six numeric Transformer fits but passed --save-training-state to NLL profiles, which the canonical trainer rejects. Those NLL profiles did not enter scientific fitting. The corrected separate operator passes that flag only during fit; profile/evaluate command checks passed before launch. The healthy six numeric fits and their sealed runtime were left unchanged. Independent source review and observed host RAM (1.8 TiB available) are recorded.

RECOVERY_LIVE.json records six numeric plus sixteen H256 plus sixteen H512 jobs in fitting phase and 38 CUDA clients. Both GRU width controllers admitted all eight paired NLL/WMSE cells after all sixteen real-shape profiles passed per width. Early GPU utilization is startup observation, not a steady-state throughput claim. Every archived corrected operator file matches a SHA256 captured on the node.

These are fresh retries of the same folds/seeds, not extra independent replications. Original logs and partial inference checkpoints are preserved in their old namespaces. No complete optimizer-state checkpoint was available to resume the original GRU attempt. Four saved epoch-0 inference checkpoints are not treated as resumed training. No paired quality gain or public-score improvement is claimed.

The original node wall remains 18:30:55 to 22:30:55 UTC; fitting stops by 21:35:55 UTC. Evaluation, grading, own-cgroup termination and final PR publication retain their original absolute cutoffs. Each corrected service uses only the remaining allowance, no restart, control-group kill and bounded final publication. Hourly and per-completed-pair reporting update PR37 under SemyonEpanov.

[Incident comment](https://github.com/VladKochetov007/WunderFunder/pull/37#issuecomment-6349648665), [H256 recovery](https://github.com/VladKochetov007/WunderFunder/pull/37#issuecomment-6349650897), [H512 recovery](https://github.com/VladKochetov007/WunderFunder/pull/37#issuecomment-6349651002).
