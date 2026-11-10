# Fixed-E6 causal-stem GRU results,10 November2026

All6 fresh6E/38,760 fits and all6 independent full-valid re-exports completed. Train10325 only -> official valid1873 /3,528,273 scored rows, unchanged global_wp_clip2_v1. Source336b1c436cf9f7a7d2196fb29eb0f55ca50e5da2 / b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188; ROOT admission48589244a653cc76a55abd3ffcc5113eccc81488722ac96599f4a8befd8cf5b7.

| Input | Seed203301 WP | Seed203302 WP | Mean WP |
|---|---:|---:|---:|
| plain | 0.681875908 | 0.681102876 | 0.681489392 |
| k1 | 0.681868074 | 0.681596179 | 0.681732126 |
| k16 | 0.683019498 | 0.682629372 | 0.682824435 |

Primary K16 minus K1: +0.001151424 and +0.001033192; descriptive mean +0.001092308. K16 minus plain is positive at both seeds; mean +0.001335042. All four per-target K16 minus K1 deltas are positive. K1 adds112 and K16 adds1,792 parameters; this is not a parameter-matched panel.

The predeclared follow-up allocation screen passes (mean K16 minus K1>=0.0005, both paired deltas>0, K16 minus plain>=0 at both seeds). This supports another reviewed study; it launches no new fit. Two seeds do not establish significance or private/public transfer. The same development validation has been repeatedly inspected.

Canonical scores agreed with all three training-monitor target/mean WP metrics to less than8e-10. This verifies prediction export and metric computation on fixed existing weights; it is not an independent training replication or a fresh holdout.

The first readout attempts falsely rejected valid producer metadata: raw-vs-normalized config hashes, omitted normalized K0, and an incorrect contiguous train-ID roster. R3 uses the source-qualified loader and exact sparse official manifest, retaining source/data/ancestry/UUID/model/time/scoring gates. First failures remain preserved. A first R3 launcher pointed to the wrong receiver directory and did no GPU work; its explicit-path replacement passed full native preflight and completed. Scientific fits and source336 remained unchanged.

[Verified summary](READOUT_R3.json), [45-file evidence archive](final-readout-evidence-r3.tar.gz), [file hashes](FINAL_EVIDENCE_R3.json), [reader checks](READOUT_REPAIR_CHECK_R3.json), [published canonical result](https://github.com/VladKochetov007/WunderFunder/pull/79#issuecomment-6778521007). Weight hashes and exact native checkpoint/prediction paths are in the archive; the large weight/prediction arrays remain on the L4 host.

The companion Transformer input-stem screen is running on8B300 in [PR74](https://github.com/VladKochetov007/WunderFunder/pull/74#issuecomment-6778321915). No cross-architecture or overall TL3 improvement is inferred from this GRU-only comparison.
