# GRU recovery R3: preserve the actual state_dict container

Native R2 canonical CPU load_resume succeeded at8E/51,680, then the probe's
verification falsely failed: its CPU tensor projection created a plain dict,
while the saved model state was an OrderedDict. The strict same_payload helper
correctly rejects different container types. R3 preserves the real state_dict
container and explicitly checks its type and ordered keys before applying the
unchanged strict dtype/shape/stride/tensor-byte comparison. Migration's helper
and every S4 source/config/profile/optimizer/RNG setting remain unchanged.

The corrected probe has a real torch model.state_dict OrderedDict fixture:
equal projection passes, a changed tensor fails, and a plain-dict projection
still fails. All6 focused CPU tests pass. Independent reviewer
`/root/monitor_execution_20261107` verified the exact R2 diff, all five hashes,
6 tests (3.18seconds) and Ruff; no remaining mechanical blocker was found.

R3 creates only fresh `R/control/gru-monitor-recovery-r3` control artifacts and
`R/runs/gru_sSEED_monitor_opt_r3/fit` sibling outputs from the same original S1
E8 checkpoints. It preserves all R2 failures and additionally pins/checks their
owned process receipts before admission. Original S1 state hashes, source S4,
ROOT-approved S4 profiles, train/valid membership, original deadline and
12E/77,520 endpoint are unchanged. No new fit, seed or profile is created.

Use the same five runtime/builder filenames as R2, staged in Q3, alongside the
unchanged actor `_s4.py` helpers. Build using the R2 command in
[the prior packet](../l4_continuation_recovery_r2/README.md), changing only
`--tools` and `--output` to Q3. Keep `--legacy-control` pointing to Q1 and
`--approved-profile-*` pointing to the existing Q2 ROOT-approved S4 copies.
The builder derives the exact R3 control/output/schema and pins all ten runtime
files, original Q1 requests, original/S3/R2 owned receipts and ROOT cleanup proof.
Launch `Q3/recovery_supervisor.py --plan Q3/recovery-plan.json --plan-sha256 SHA --detach`
with the same S1/src:Q1 PYTHONPATH and original venv Python.

| File | Frozen SHA256 |
|---|---|
|build_recovery_plan.py|1ddfcdbfb0f2d2818b22eada9e64a8c0537b3cadbe1c4e58c200eaa0f23dbd6f|
|canonical_resume_probe.py|4780f5c0075a0b7d8f9df93fcd12b7234a169b9ebb3503339e35a84ba4acc472|
|recovery_checks.py|3ee5bf8d5218adef00fbed03d1fb9d9116ddc541390ccdd7dbdf475f5e060431|
|recovery_supervisor.py|7726ae53a6ab3d5ca9b46c76c21237e74b5838cea511f78721931d0d181e92f5|
|run_probe.py|18911c7ecf7088b5257adc27b0276982ca7b66218bb6a266978fe0f12e89dae0|

Native canonical CPU and separate real CUDA zero-update restoration probes
remain mandatory gates before archiver-ready production continuation. CPU mode
still stubs only the CUDA RNG setter and never claims real CUDA restoration.
The fixed bounded callback/reaping and archive-death guards are retained. The
preparation agent performed no SSH, GPU, process or native recovery action.
