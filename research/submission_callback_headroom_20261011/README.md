# Submission callback headroom profiles — 2026-10-11

These three fixed Q8VNNI packages were replayed through the exact shipped
Python callback and native predictor on ten complete causal sequences, five
reset-aware passes each. Prediction arrays were bitwise identical across
replays. The benchmark times callback calls; it excludes Parquet decoding,
state reset, and competition-side orchestration. These are runtime-capacity
measurements, not scores or public-score forecasts.

| Package | Members | ZIP bytes | ZIP SHA-256 | Mean callback µs/row | Median pass µs/row |
|---|---:|---:|---|---:|---:|
| `headroom_h256_s3_s4_equal_q8vnni` | 3 × H256 | 12,181,995 | `9c00bc0c3eb58145900c47e02d099821b53e2fbdd3ecd64202e661075fe4360a` | 102.6653 | 104.1843 |
| `headroom_student256_h384s3_equal_q8vnni` | H256 + H384 | 12,913,680 | `95ea869cc8965b1f5df9b65263c78895549eac34cd7f4117e2076f30acf29e4f` | 113.1419 | 115.7689 |
| `capacity_h384_s3_s4_q8vnni` | 2 × H384 | 17,271,543 | `3eb7581369c05c036c00c1ba2b6ee78da976fa708d66cd805ed1c03ae3791861` | 158.9771 | 158.6115 |

The measurement host was an Intel i9-11900H, one CPU core, 1.2 GB cgroup
memory maximum, with peak RSS between 634 and 668 MiB. Exact raw receipts are
in [`profiles/`](profiles/). The ZIPs and build receipts remain in the ignored
main-workspace directory `submissions/10-11-headroom/`.

Root's exposed fold-1 FP32 grades were 0.67326 for the H256 trio, 0.67177 for
the H256+H384 pair, and 0.67161 for the two-H384 pair. Those grades belong to
the fixed FP32 prediction blends and do not establish the score of these Q8
packages. The exact grade receipt is
`research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/HEADROOM_BLEND_SCORES_R1.txt`.

Portal uploads measured 14:37 and 14:52 for the singles, and 24:56 and 27:01
for the pair. Those times came from different hardware and include portal-side
execution; do not project them from the local per-row callback ratios.
