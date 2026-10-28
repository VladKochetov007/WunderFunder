## First complete GRU epoch — intermediate result

The fresh `g0-plain-gru256x2` completed epoch1 and continues training. Full
official-valid **WP = 0.668133** (t0 =0.665978, t1 =0.670287), covering all1873
valid sequences and3,528,273 official scored rows. Maximum20epochs and the
declared min5/patience4/delta0.0005 stop policy remain unchanged.

| Report cohort | Row mask | WP | Weighted MSE | Sequences | Selected rows |
|---|---|---:|---:|---:|---:|
| Fixed train diagnostic | all_after_warmup |0.449869|1.341924|32|636832|
| Full official valid | is_scored |0.668133|1.026292|1873|3528273|

These cohorts and masks differ; their score gap does not establish overfit or
better generalization. Online full-train loss1.449732 is a separate optimizer
objective. This is one exposed-development epoch/seed, with no final model ranking
or public-score claim. The other seven fits are running.

The epoch took768.43s for fitting and75.16s for paired train/valid reporting.
The best and last pointers both reference the saved epoch1 checkpoint, verified
SHA256 `94db7c641883d5de298e2dbc54d3671b894acac452f8ef03547b3b5e85798536`,
4,539,443bytes. Model weights remain on the owned node.

Launch `e635250b62fa489d94c987d4898ff417`; R9 source commit50053909,
training-source SHA `f9685716d64ecb5ceef2ec821781fa4078ff21661201edfcdac8292867990230`.
First-epoch raw metrics and independent metadata review are in the launch packet;
full source and engineering evidence archives are linked there. The node publisher
will continue hourly and per-experiment updates independently of SSH.

Follow-up at 2026-10-28T06:47:04.673788Z: the same GRU has completed epoch4 with full-valid WP **0.6760966529** (t0 **0.6688052605**, t1 **0.6833880453**), weighted MSE **0.9808772327**. Last/best checkpoint SHA `008f5370056ba33c2566663a483dd79fe6b84743501ec907bc6f1915d8649b1a`. All8 continue; the seven GALAM arms have not completed their first epoch yet. This remains an intermediate single-seed result.
