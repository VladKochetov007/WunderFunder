# Independent seed4 terminal audit

**Verdict: pass for fit and checkpoint integrity, with one disclosed reporting-inventory drift.** The admitted seed4 command finished successfully, saved the fixed epoch-3 model and an epoch-boundary training state, and completed its full four-epoch schedule. This review makes no validation-score or performance claim because evaluation was withheld.

The command hash matches both the seed4 intent and prelaunch receipt. Its allocation hash matches the recorded root allocation. The fit used a fresh seed4 initialization, with no initial checkpoint, teacher predictions, or consumed caches. The split manifest hash matches the exact run receipt. The training-state contract records 53 trainer/model source files; all 53 still match their hashes.

`model.pt` and `model_ep3.pt` both identify epoch 3, and their 16 model tensors are exactly equal and finite. The 25,868,847-byte training state says `next_epoch=4`, records all 29,840 planned optimizer attempts and scheduler steps, and has finite model and optimizer tensors. Training diagnostics show 15 FP16 scaler-skipped updates and 29,825 completed updates over 2,984 batches. Skipped attempts are not counted as completed updates.

The terminal attempt records `exit_status=0`. The now-unloaded transient systemd unit reports inactive/dead, `Result=success`, and `ExecMainStatus=0`. Peak worker RAM was 8,002,596,864 bytes, peak parent RAM 15,968,665,600 bytes, and observed VRAM 3,982,491,648 bytes. Worker, parent-local, and parent-hierarchical `max`, `oom`, `oom_kill`, and `oom_group_kill` deltas were zero. The worker `high` counter increased by 1,027,600 bytes; the parent hierarchical counter reflects that same child event.

One of the attempt's 332 broad controller inventory pins no longer matches: `scripts/build_research_matrix.py` changed after the run started, with current hash `d56f8f…` versus pinned `86d5aa…`. Its file timestamp and later commit bind the change to the post-start reporting-matrix refresh. It is not in the executed training closure and is not imported by the trainer or `src/wnn`; the actual 53-file closure remains exact. This discrepancy is disclosed and does not invalidate the fit.

Raw parquet bytes and SHA values were read from the model's embedded training-state contract; this bounded review did not hash the 34.9 GB source datasets again. The split file itself was re-hashed. Exact artifact hashes, source-drift qualification, resource receipts, and all check results are in [the machine-readable review](SEED4_TERMINAL_AUDIT_LUNA_R1.json).
