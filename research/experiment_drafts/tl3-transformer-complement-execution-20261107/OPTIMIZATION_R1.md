# Measured B300 throughput optimization,7 November2026

Current execution: [PR69](https://github.com/VladKochetov007/WunderFunder/pull/69),
four fresh matched B128/A4/two-B300 fits, fixed30E/24600 updates, sourceS3
6ab2f679e70928f913b97d5a29f0d1e09ceacaa3. The user explicitly requested
optimization after observing poor utilization. Old B4 attempts were retired
at1291/1269/1264/1263updates and remain partial historical attempts. No old
stopped fit was resumed. New configurations preserve every field except batch;
the canonical scheduler recomputes its update/warmup counts from30epochs.

| Real-data engineering profile | Warm rows/s | Mean warm window | Peak reserved GB |
|---|---:|---:|---:|
|B16/world2, labels |44157 |.7305s |16.57 |
|B32/world2, labels |85082 |.7582s |30.07 |
|B64/world2, labels |157782 |.8177s |59.14 |
|B128/world2, labels |216900 |1.1897s |118.09 |
|B128/world2, complement |215064 |1.1999s |117.80 |
|B128/world2, peak-LR labels |219933 |1.1733s |117.80 |
|B256/world2, labels |235378 |2.1926s |235.24 |
|B256/world1, labels |123821 |2.0840s |246.41 |
|B256/world1, complement |124786 |2.0679s |245.92 |

Each completed50updates, errors[], no scientific checkpoint. Warm columns use
native updates21–50, including recurring shorter tail windows; cold compilation,
loading and total worker/profile time are separate. R1 total profile times
84.35/86.82/92.11/116.15s. R2 total times: B128 complement89.21s,
peak87.69s, B256world2 175.92s; single-GPU labels166.29s/complement162.20s.
The first B128 provisional single-window.91s estimate was superseded by its
complete1.19s mean. These are engineering speed/resource observations, not
convergence or quality measurements. Concurrent cohorts share host/cache load.

B128 versus old B4~11krows/s is about19times the useful row throughput. B256
adds only~8.6% versus B128 while doubling batch and reserved memory. B128 keeps
twice as many optimizer updates as B256; both labels/complement and both seeds
use the same selected recipe. One GPU at the same global batch is substantially
slower, supporting four two-GPU pairs. This does not prove B128's model quality.
Quantization, precision, contexts, masks and scientific objectives are unchanged.
Optional batched-Muon source21cdf3fe77 is prepared/CPU-tested but is not used:
its remaining theoretical window-time opportunity is only~4–8% after batching,
and native numerical/gradient/state parity and performance are still unmeasured.

Exactly four additional50-update production profiles then ran with the actual
four physical pairs, exact seeds/configs and source. All passed finite/resource
checks before ROOT-approved exact native copies. Production panel SHA:
78ac20e1fb651c314ae38ded8728fb0826db65823742d47fff2e6b80e9b438f0.
ROOT admission SHA:
b3e53eab3ebd0526f1a5687627ecd959bbb808ce2e7aff4b27b1780d47de6fb0.
Model source identity remains
bdad606dcff66781b904ae4774e7435b8ee21c21de2911cd0320eb2636c725d7.
All original native acceptance, qualified reference cache and baseline replay
are reused unchanged. Official train10325 only/zero valid fitting; full valid
1873/3528273 scored rows; split0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.

New detached controller1516965/birth148863077, directory
/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/control/b300-optimized-r1.
Only exact ownership-proven descendants can be stopped. The24h allocation from
first production profile includes a1h final collection reserve; the initial
insufficient B4/24h and amended9day receipts are historical. Complete30E work is
estimated at~8h core fitting before data/validation/checkpoint overhead; actual
first-epoch wall and full-valid times will refine it. Every epoch evaluates full
valid, saves safe optimizer/model state, and supplies native plot journals.

Run root: runtime/throughput-engineering-r1/runs. Online dashboard uses distinct
b300-opt run IDs alongside old historical b300 attempts. Reporter has an explicit
frozen source/canonical-config/split/update manifest and posts under SemyonEpanov
every30minutes or terminal change to PR69; profiles cannot count as completed
fits. Fit completion, full endpoint grade and artifact upload are separate.
No public-score forecast, final quality gain or significance is asserted.
