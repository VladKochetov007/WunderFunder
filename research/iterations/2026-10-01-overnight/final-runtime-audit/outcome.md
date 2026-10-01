# Full-valid baseline3 runtime audit

PASS for the prospectively fixed conjunction: functional completion, callback projection≤50min, both SMT58 quiet measures<5%, full-process projection≤60min, and actual completion within65min. The single authorized run ended2026-10-01 04:39:57.664UTC with exit0. WorkerPID4127744 was absent and the owned systemd scope was inactive/dead at04:40:40UTC; CPU18/58 was explicitly released. No retry occurred.

All1,873 complete20,000-row sequences were processed through the freshly extracted original PredictionModel.predict, totaling37,460,000 callbacks and37,274,573 emitted float32 prediction pairs. All callbacks before step99 updated state and returnedNone. All inputs and emitted predictions passed finiteness checks. Sequence identities were unique, reset boundaries were observed, and the raw-only reader performed exactly1,873 row-group reads. Targets and scoring masks were not decoded; no WP query or forecast-quality change was made.

| Measurement | Observed37.46M | Projected39.4M |
|---|---:|---:|
| Complete callback loop |41.2239min|43.3588min|
| Full process, including startup, input reading, checks and closing hash |43.1610min|45.3963min|

The complete process includes38.00s input decode/conversion,45.85s validation/checksum work,13.60s opening input hash and13.48s closing input hash. Do not subtract these costs from the process deadline. The opening hash warmed the page cache; this is not a cold-cache guarantee. The projection is a fixed linear39.4M reference-size extrapolation from this run, not another measured dataset.

SMT58 busy fractions were0.0295355% during summed callback regions and0.0293326% across the complete trace, both well below5%. The process stayed on CPU18 with100% cgroup quota,16GiB memory and zero swap. Maximum observed RSS was347,136KiB. Two OS threads were observed; the helper accumulated zero user/system CPU ticks and every observed thread retained affinity[18]. The claim is one computeCPU, not one literal OS thread.

The original baseline3 ZIP remains SHA9471821e81e7c3f80a088147512183b72e0ab52bc593a1136d2becc229d0d098. The full input hash2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782 matched before and after, as did the frozen source/package hashes. This run covers the delivered nativev3 path on the measured machine. Other-ISA timing remains a separately proposed, unexecuted supplement.

The metadata-only outcome_summary.py independently reconstructs row counts, time sums, projections, SMT tick sums, unique sequences, bound receipts and all qualification booleans from the preserved JSONL/JSON evidence. It reads no original arrays, model or forecast cache. See outcome-summary.json and the immutable execution receipts for exact values and hashes; independent critic review is separate.
