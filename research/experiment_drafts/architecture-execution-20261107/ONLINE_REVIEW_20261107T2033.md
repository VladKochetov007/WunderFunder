# Native online review, 7 November 2026 20:33:18 UTC

Actual native RPC observations114/L4 and115/B300, collected through the existing
single persistent stream per host. Four scientific fits completed; eight remain
active, including two B300 cells evaluating epoch3. All active heartbeat ages
were below5seconds. The last100 native logged loss/preclip-gradient samples per
cell were finite; this does not claim an independent lifetime nonfinite counter.

| Cell family | Latest full-valid epoch | Seed101001 WP | Seed101002 WP | State |
|---|---:|---:|---:|---|
|B300 labels, fixed90%TL3+10%Transformer |3/30 |0.6926189192863913 |0.6924820206329949 |active |
|B300 complement, same fixed mixture |2/30 |0.6916931534466201 |0.6919028043383280 |evaluating epoch3 |
|plain GRU256x2 |8/12 |0.6763340399338484 |0.6775980008454134 |active, about8.68E |
|GRU256x2 plus local attention |8/12 |0.6759607411170973 |0.6764172544544621 |active, about8.5E |
|Mamba2 |12/12 |0.6514818864145935 |0.6604850495099179 |completed |
|EMA/custom TabM-style |12/12 |0.6726947477701329 |0.6712658573059384 |completed |

Clean TL3 reference is0.6923774248169172. The two labels mixtures have small
positive epoch3 deltas0.000241494/0.000104596; these are intermediate exposed
development observations, not a fixed30E result, ensemble score, significance,
public transfer or a new independently validated winner. The complement cells'
epoch3 evaluation was not complete at collection. L4 scores are standalone;
B300 columns are explicitly fixed mixtures, so their raw rows are not equivalent
architecture-only comparisons.

Mamba2's observed epoch6 WP0.680358/0.682011 deteriorated substantially by final
epoch12 while fitting diagnostics improved. Plain/local GRUs also declined from
their earlier observed peaks; continuing the declared fixed endpoint preserves
the comparison. All historical checkpoints remain available. A checkpoint's
descriptive best-observed valid is distinct from a qualified final endpoint.
Train diagnostics and valid have different cohorts/masks, so their absolute
vertical separation alone is not the evidence for overfitting.

Both corrected plain-GRU native producers were past the original51680 frontier
at56050/56090 updates. Their external full-state archivers were live, exact
PID/birth2238610/2148031192 and2238608/2148031187. The actual CPU and real CUDA
resume gates, state/source hashes and preserved failed attempts are in
[the active continuation record](l4/throughput_probe_r1/ACTIVE_CONTINUATION_S4_R3.md).
These are the original two fits, with unchanged12E/77520 endpoints and clock.

Every reported evaluation covers official valid1873 sequences and3528273 scored
rows. Fitting remains10325 official train members, zero valid fitting/ancestry;
the split SHA is0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.
Validation remains development-exposed. The unchanged official pooled clipped
weighted Pearson is reported; no random or seq_ix-ordered primary folds.

[Dashboard](http://127.0.0.1:8769/) and actual native journals/PNG/SVG snapshots
remain active. The frozen reporting panel577e129c4acd27c5ddf8faee5fee2631d0f63f3e1295528d105b08d201d2e044
selects exact S4/R3 producers for the two original GRU rows and the unchanged
B128 B300 panel. Verified SemyonEpanov reports:
[PR68](https://github.com/VladKochetov007/WunderFunder/pull/68#issuecomment-6712620742),
[PR69](https://github.com/VladKochetov007/WunderFunder/pull/69#issuecomment-6712621433).
Native completion, final archive completion and artifact upload remain distinct.
