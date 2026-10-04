# E422 mature-episode gain probe

The corrected GPU cache retries retained209 and219 sequences (2466 and2030 later evaluation rows).17/23 selected sequences were excluded only by prespecified insufficient-episode-sample rules. Backbone ancestry excludes evaluated components; the gain heads use reciprocal component-disjoint directions. Earlier label episode1024:4096, anchor after48 endpoint events, later scoring gap250 rows.48event maturity is an archived estimate, not newly established label maturity.

| Direction | State gain minus unchanged WP |95% component interval | State R² t0/t1 | Input-summary R² t0/t1 |
|---|---:|---|---|---|
| A→B |−0.01283145|[−0.01932117,−0.00582063]|−0.4553/−0.3755|−0.3323/−0.1804|
| B→A |−0.01414106|[−0.02040514,−0.00893912]|−0.3281/−0.2451|−0.1056/−0.1784|

State probe does not beat input summaries; both have negative R² against training-mean gain. State minus input-summary WP intervals include zero. The earlier-label oracle also loses later-episode WP (−0.032165 and−0.024970). This suggests this measured episode-specific gain is not stable across the fixed windows; it does not establish representation exhaustion, a forecasting ceiling or the futility of every auxiliary objective. No memory architecture was fitted here.

The first CPU4GiB cache helper OOM remains recorded. Selected-state views were replaced by copies and lifetime regression tests passed. Successful production retry used GPU8GiB, so it does not show that the original CPU4GiB configuration now fits.

Evidence: checks/e422_reciprocal_report.json, feature_gain_cache/f0_{a,b}_v2.json, feature_gain_results/e422_gain_screen_f0_d{0,1}_cpu_v2/report.json, sealed specs, attempts and E422_MEMORY_INCIDENT.md.
