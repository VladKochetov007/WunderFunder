# E450: fixed seed-4 replication and equal seed bag

The seed-4 parent completed all four epochs, selected fixed epoch 3, then
completed the same seed-3 head fine-tune at fixed epoch 0. Both full-validation
FP32 exports finished. All four local jobs returned zero with verified teardown
and no new worker OOM events. Parent peak observed VRAM was 4,484,759,552 bytes;
peak cgroup RAM, including charged file cache, was 22,672,068,608 bytes.

The frozen paired screen completed in 30.77 seconds, with 1,222,365,184-byte peak
RAM. Evaluation uses the same 368 historically exposed excluded sequences and
249 components. These results are not an untouched validation estimate.

| Fixed predictor | Global WP |
| --- | ---: |
| Seed-3 equal parent/head, public-confirmed recipe | 0.699822159 |
| Seed-4 equal parent/head | 0.701694557 |
| Equal seed-3/seed-4 bag | 0.701420130 |
| Seed-4 parent alone | 0.700363274 |
| Seed-4 fine-tuned head alone | 0.701577995 |

| Preregistered comparison | Difference | 95% paired component interval |
| --- | ---: | --- |
| Seed 4 minus seed 3 | +0.001872398 | [-0.000975354, +0.006931454] |
| Equal seed bag minus seed 3 | +0.001597972 | [-0.000039884, +0.004622049] |
| Equal seed bag minus seed 4 | -0.000274427 | [-0.002340905, +0.000929490] |
| Seed-4 head minus parent | +0.001214721 | [-0.000153328, +0.002491309] |

All intervals include zero. The seed-4 versus seed-3 point difference is mostly
associated with class 0 (+0.006151173); class 3 is slightly lower (-0.000250381).
The bag raises class-3 WP by about +0.000405672 versus seed 3. Class correlations
do not add linearly to global WP. These intervals condition on fitted models;
they do not establish a training-seed distribution or selection correction.

The seed-4 package build is a separate bounded CPU attempt. Manual-upload
readiness still requires actual callback/parity/reset/causality and runtime
evidence. Its public score is unmeasured. The public-confirmed seed-3 score
0.6655 is an empirical anchor, not a calibrated narrow forecast for seed 4.
The fixed two-seed bag must meet CPU runtime limits or serve as a teacher for a
separately registered student; its descriptive score alone does not authorize
deployment.

Raw evidence: `checks/e450_report_seed4_bag_fp32_v1.json`, frozen design
`checks/e450_screen_seed4_bag_fp32_v1.json`, sealed scorer specification
`specs/e450_score_seed4_bag_fp32_v1.sealed.json`, all four E450 fit/export worker
records, and `E450_SEED4_REPLICATION_PREREG.md`. Marimo-generated figures and
source hashes are under `eda/seed4_replication/`.
