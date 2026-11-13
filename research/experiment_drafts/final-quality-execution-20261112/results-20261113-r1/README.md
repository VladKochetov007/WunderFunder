# Final-quality wave: complete results,13 November2026

All16 admitted scientific fits completed their fixed endpoints: four PR91 Transformers at E4 and twelve GRUs at E6. No fit was restarted by this readout. Primary validation remains10325 train-only fitting sequences,1873 official valid sequences /3528273 scored rows, `global_wp_clip2_v1`; validation is development-exposed.

| Contrast | Control mean WP | Treatment mean WP | Mean paired delta |
|---|---:|---:|---:|
| Zero-since GRU vs since32 |0.682533974|0.685438194|+0.002904220|
| Since32 vs since8 |0.685438194|0.685425715|-0.000012480|
| Since32 vs since128 |0.685438194|0.685226866|-0.000211328|
| Same-cache lambda0 vs KD0.2 |0.682103185|0.683948980|+0.001845795|
| Linear vs bilinear Transformer |0.679392141|0.679156263|-0.000235878|

Each row compares two matched seeds. Since32 paired deltas are+0.002225896/+0.003582545; KD paired deltas are+0.001944372/+0.001747217. Positive changes on both seeds support follow-up, not significance or a public-score forecast. KD uses different seeds from the since panel; their absolute scores are not a matched comparison, and their combination has not been tested. Since8/128 did not improve on since32. Bilinear input did not improve the scheduled Transformer endpoint; all four Transformers deteriorated from E3 to E4. Retained E3 values do not replace the predeclared E4 comparison.

GRU/KD values come from complete fixed-endpoint canonical monitor journals. The preceding since32/zero endpoints were independently exported/scored and agreed with the monitor within6e-10WP. Transformer values above come from fresh reads of independent canonical exports/scorer on B300, each with complete1873-sequence coverage. Small BF16 monitor/export differences are retained explicitly in the JSON report.

[RESULTS_R1.json](RESULTS_R1.json) records every seed, endpoint, score, checkpoint/config/source identity, prior-epoch score and producer receipt hash. [CANONICAL_B300_R1.json](CANONICAL_B300_R1.json) binds the freshly read Transformer grades and checkpoint/grade hashes. [Raw retained evidence](fixed-endpoint-monitor-evidence-r1.tar.gz) contains64 canonical metadata/journal files for all16 fits, without dataset payloads or weights; its digest is in RESULTS_R1.json. [Collector](collect_results_r1.py) checks endpoint completion, training-only membership and complete official metric coverage before computing comparisons. The canonical native grading files were read over one restored SSH session; no new training was launched. B300 GPU memory/utilization was0 on all eight cards at that fresh read; no current L4 resource claim is made from retained logs.

## Actual platform status

The signed-in official submission page and public leaderboard were visually checked on13 November2026. Team kotiki has108 total submissions,0 today, best public0.66832, rank61. Latest upload remains `tabdata-w18-final-20261103-r1.zip`, ID1H7KP3OQ, dated4 November, score0.66453, duration14m32s,627MB. No new fullmix upload or new public gain is observed.

The prepared14.8MB all-data TL3/LSTM ZIP remains unsubmitted. Its optimized native callback has verified bitwise parity against its baseline on the engineering panel and a measured2.6629s per20000-row sequence on one owned-host CPU core. The documented1970-sequence linear extrapolation is87.4min against the platform60min envelope. This is a conservative host projection, not an observed platform timeout or platform runtime. Final fitting used all12480 supplied labeled sequences, so those final weights have no independent validation score.

Interactive browser access now works, superseding the older desktop-unavailable observation in PR94. Runtime admission remains unresolved. This report does not silently promote a failed runtime check or duplicate the already submitted W18 singleton. See [PR94](https://github.com/VladKochetov007/WunderFunder/pull/94), [PR95](https://github.com/VladKochetov007/WunderFunder/pull/95) and [PR96](https://github.com/VladKochetov007/WunderFunder/pull/96).
