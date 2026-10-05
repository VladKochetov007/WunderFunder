# Kaggle read-only status observation — 2026-10-05 00:48:37 UTC

The exact W384 v7 job `vladkochetov07/wnn-w384-fold1-dualt4-20261005-v7-runner` reports `RUNNING`. The exact existing E463 TPU preflight `vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1-runner` reports `QUEUED`. The CLI and installed SDK agree; both SDK responses have `failureMessage: null`.

Filtered output retrieval requested only JSON, log, and text reports whose names indicate phase, admission, inventory, summary, stderr/stdout, error, status, result, or manifest. Both calls returned exit code 0 but surfaced no matching files. Therefore the W384 stage cannot currently be distinguished among bootstrap, probe, host admission, or fitting from available evidence. `RUNNING` alone makes no scientific claim. Neither job was restarted, cancelled, duplicated, or modified.

E463's existing private version 1 metadata requests `TpuV5E8` with TPU enabled, GPU and internet disabled. The provider status surface returned only status and failure message; no queue position/reason or current accelerator supply inventory was exposed. Given submission at 2026-10-04 17:23:26 UTC, the observed queue duration is 26,711 seconds (7 h 25 m 11 s). No new queue explanation is supported.

Raw status command/SDK evidence is in `provider_status_raw.txt`; structured evidence is in `observation.json`. No matching output artifacts were downloaded. This receipt is an operational status snapshot only; it contains no completed fit result or score.
