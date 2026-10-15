# Matched-matrix GRU and minGRU native cost profile R1

## Result

On the local i9-11900H, the H384×2 minGRU Q8 VNNI callback took 0.93937 s per complete 20,000-row sequence, versus 0.30860 s for H256×2 GRU, a 3.044× runtime cost at nearly matched serving matrix counts. This is a measured cost disadvantage for this deterministic synthetic-weight proxy. It is not a quality result or a rejection of minGRU. With FP32, minGRU took 3.58502 s and GRU 4.15645 s. The four workers, build and bounded reference check are recorded in [attempts](ATTEMPTS_R1.json), [worker receipts](runs_r1/), [build receipt](BUILD_R1.json), and [reference receipt](REFERENCE_R1.json).

## Frozen comparison and callback

The serving shapes are two-layer, two-output H256 GRU and H384 minGRU. Their matrix counts, excluding biases and normalization, are 972,032 and 966,528 (0.57% apart). The run uses the sealed native source commit `7364a452e0c72c2600a27e21cb66f6deb9e714ce`; it supports independent minGRU encoder/recurrent/readout widths, while the ordinary GRU remains shared-width in this source revision. The later independent-width GRU merge is not included in this binary.

Each callback consumes the same source-qualified fitting sequence: train row-group 37, sequence 7,598,721, 20,000 raw 112-feature rows, SHA-256 `fe24c974e6d6e3e720d36937be06d222d00f743efd65e4a0a3ec178db891f57a`. The existing causal feature builder maps each raw row to 211 model features and handles the first 99 warm-up rows. Both families use the same clean-F0 normalization arrays, derived from checkpoint SHA-256 `e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda`. Targets and masks were not loaded.

Weights are deterministic NumPy PCG64 synthetic float32 tensors, seeded by `[20261014, family_id]`, uniform per matrix in `±1/sqrt(fan_in)` with zero biases. They are runtime proxies, not trained checkpoints. All callback workers ran sequentially with a one-core CPU quota, one thread, no swap, and explicit predictor reset before each of three full-sequence replays. They assert the requested precision was selected, check finite outputs/state, and find bitwise-identical predictions and final state across each worker's resets. Every callback timing includes the Python-to-native row calls, raw-row sanitation, streaming feature construction and all 99 warm-up rows.

## Measurements

| Family and precision | Mean seconds / 20,000 rows | µs per raw row | Constructor plus weight import/pack (s) | Native weight bytes | Native scratch bytes |
|---|---:|---:|---:|---:|---:|
| GRU FP32 | 4.15645 | 207.82 | 0.00881 | 3,902,528 | 11,084 |
| minGRU FP32 | 3.58502 | 179.25 | 0.00145 | 3,877,024 | 10,068 |
| GRU Q8 VNNI | 0.30860 | 15.43 | 0.01450 | 1,071,552 | 11,864 |
| minGRU Q8 VNNI | 0.93937 | 46.97 | 0.01362 | 998,560 | 11,816 |

The constructor column excludes importing the native Python extension; it records construction, tensor import/copy and any Q8 weight packing. The native bytes are post-import storage. For the GRU Q8 path, the encoder remains I16; recurrent and readout matrices use Q8 VNNI, with the final head in FP32. The minGRU Q8 path stores its encoder, cells and first head layer as Q8, while its final output layer stays FP32. Thus both callbacks selected the requested `q8vnni` policy, but their layerwise quantization schemes are not identical.

Reported cgroup `memory.peak` values were 51,687,424 B (GRU FP32), 35,799,040 B (minGRU FP32), 36,794,368 B (GRU Q8), and 35,782,656 B (minGRU Q8); every worker recorded zero OOM, max and swap events. The receipts separately report process `ru_maxrss` of 621,481,984 B for these Python 3.11 workers. Those two accounting series do not reconcile, so the report preserves them as distinct measurements and makes no peak-working-set claim from combining them. The profile's bounded Python reference process used 595,333,120 B cgroup peak and 763,465,728 B `ru_maxrss` under its 1.5 GB scope.

The same rows and weight hashes repeat bitwise after reset in all four worker receipts. The FP32 native callback was also compared with the Python reference on the first 137 rows per family using exact activations: GRU prediction max-absolute error `1.4901e-7` (RMSE `4.1422e-8`) and state max-absolute error `5.5134e-7` (RMSE `1.4638e-7`); minGRU prediction max-absolute error `1.6391e-7` (RMSE `4.4158e-8`) and state max-absolute error `3.5763e-7` (RMSE `6.9027e-8`). This is close numerical agreement, not bitwise native/Python parity. The exact-activation reference is separate from the fast-activation callback timing workers.

## Source-level explanation of the Q8 gap

The code gives a plausible mechanism for minGRU's Q8 cost, but this profile did not isolate per-phase time, so it does not establish which operation caused the measured gap. Both timed workers request fast activations and assert `q8vnni`; in the GRU path, that selects `step_dyn_vnni` and the AVX-512 VNNI implementation in `native/src/rnn_fast.rs`. It uses AVX2 activation quantization, vectorized fast GELU/GRU updates, and carries the quantized projection and layer-state codes into recurrent GEMVs and the skip readout. Its input projection remains the I16/half path.

In `native/src/min_gru.rs`, each Q8 `Linear.apply` invokes scalar `quantize` on its input and then calls the shared dynamic Q8 GEMV. The callback repeats those conversions for normalized input, encoder projection, layer inputs and the concatenated skip-head input; its candidate/write-state update loops also run elementwise in scalar Rust. The positive minGRU candidate branch and GRU reset/update/tanh equations are different by design. These visible schedule and conversion differences are candidates for the gap, not proof of its cause. A bounded phase/matrix/activation/quantization profile is the appropriate next diagnostic before changing this implementation.

## Provenance and limits

The optimized release binary is 1,224,720 bytes, SHA-256 `5e29328c940aeaada3f38c86a75a824585fedae78bf68d6a1dd5d78b98996f13`, built from the sealed source revision with Cargo release optimization. The build receipt pins the compiler, source hashes, command and Build ID. It was built under a one-core, 2 GB, no-swap scope; the scope disappeared before its peak was read, so the compile's actual peak is unknown. The native source was not changed for this profile.

This is one Intel host, one fitting sequence and synthetic deterministic weights. A source-qualified fitting sequence is not a held-out quality set. The workers used one-core quotas but were not pinned to an exclusive physical core; the H192L2 fitting job remained active during the profile, so host contention may add timing noise. The results do not rank prediction quality, training capacity or competition performance and do not predict an AMD or competition-host runtime. The earlier debug GRU FP32 run is preserved in [its own receipt](runs_r1/gru_f32_worker.json) and is excluded from the optimized release table. The pre-existing [eight-shape cost screen](../architecture_profile_r1/REPORT.md) is context only because it used different sequence rows, normalization and repetitions.
