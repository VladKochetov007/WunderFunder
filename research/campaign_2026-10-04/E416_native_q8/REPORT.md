# E416 — native byte-weight inference

Disposition: both byte presets pass the fixed trained-model numerical gates and bitwise checks. Recurrent-plus-head q8x compresses streamed weight storage by 42–44%, but shows no wall-time speedup in the clean confirmation. Keep q8 experimental; this is not package or public-score verification.

The D256 payload is the actual equal parent/head predictor from E411. W384 has its existing single trained head from E409; no duplicate head or pairbag was measured. Each case uses the same three complete 20,000-row streams. Native q8 uses int8 weights and transient int16 activation codes, with float32 persistent recurrent state, input projection and final output layer. This adds an activation approximation beyond the weight-only E411/QAT reference.

## Numerical gates

| Payload / mode | Max absolute error | Pooled WP delta | Streamed bytes | Byte panels + scales |
|---|---:|---:|---:|---:|
| d256_q16x | 0.000187285244 | -2.75380386e-07 | 2,241,984 | 0 |
| d256_q8r | 0.000109508634 | -8.83578177e-08 | 2,082,880 | 798,720 |
| d256_q8x | 0.000146955252 | -1.71734362e-07 | 1,300,544 | 1,064,960 |
| w384_q16x | 0.000137597322 | +2.68967386e-07 | 4,337,856 | 0 |
| w384_q8r | 9.44137573e-05 | +2.98571338e-07 | 3,316,288 | 1,787,904 |
| w384_q8x | 0.000155717134 | +3.5166065e-07 | 2,434,624 | 2,085,888 |

Every case passed maxabs ≤3e-4 and absolute pooled WP delta ≤5e-6 against its own represented-weight independent Torch reference. Both CPU builds passed portable/native and reset repeat checks; all six full v2/v3 prediction arrays are bitwise identical. Preserved codes and float32 scales were checked byte-for-byte on native import/export. Separate audits recomputed the errors, checked expected six-case completeness and all prepared/source/library hashes. These three-stream checks do not establish universal accuracy for future checkpoints.

## Clean confirmation timing

Five forward/reverse alternating trials per case, actual scorer callback including feature updates, pinned CPU15 on the Intel i9-11900H. Model construction is outside the timing. Each stage is contained under wnn-campaign.slice at one CPU, 2 GiB, swap zero; timing cap 1800 seconds.

| Payload / mode | Median us/row (min–max) | Median kcycles/row (min–max) |
|---|---:|---:|
| d256_q16x | 59.535 (57.729–60.591) | 234.956 (229.866–237.287) |
| d256_q8r | 110.931 (104.786–112.909) | 437.388 (420.288–445.280) |
| d256_q8x | 63.046 (62.699–64.186) | 240.312 (238.326–242.998) |
| w384_q16x | 107.765 (104.044–107.844) | 421.893 (412.860–424.719) |
| w384_q8r | 162.719 (161.700–163.047) | 628.173 (624.595–637.579) |
| w384_q8x | 111.247 (110.089–112.416) | 416.669 (416.296–422.013) |

For D256, q8x wall time is 5.9% above q16x; for W384 it is 3.2% above. W384 q8x cycles are about 1.2% lower while elapsed time remains higher, so these measurements do not support an elapsed-time improvement. Recurrent-only q8r is markedly slower. No conversion to public/scorer runtime is made, and the host-built libraries still require a scorer-image build and package verification.

## Preserved execution incidents

The original 30 timing trials remain in E416_native_q8/result.json and trials.jsonl. A 10.4-second pytest run (03:16:27.252–03:16:37.678 UTC), plus less than 0.3 seconds of Ruff, overlapped that run. It is retained as loaded-run evidence. After all actors acknowledged a hold, the controller authorized exactly one identical confirmation, saved separately under E416_native_q8_confirmation. That run lasted 03:20:28–03:23:33 UTC, 184.97 seconds, peak RSS166108 KiB; all30 trials completed with exit0. Each trial retains active campaign unit and load-average snapshots; notebook and watcher remained present.

The v3 scientific parity subprocess finished exit0, but its outer shell later exited2 because the shared command file was edited while bash awaited the subprocess. This is preserved in control/WRAPPER_INCIDENT.md with reconstructed original command text, snapshots and an independent completeness/content audit. It is not reported as an entirely successful wrapper attempt. V2 and both timing wrappers exited0. Future stages used frozen separate command files.

The initial unscaled-random stress check also remains a recorded failure (0.0024118423 versus its original1e-3 threshold), with the original fixture retained as a stability/determinism test and a separate fan-in-scaled accuracy test. No original failed check is relabeled as unchanged-and-passing. Actual trained-model gates remained fixed before inference.

Own preparation/parity/timing cgroup OOM counters remained zero. Source commit b19996f, exact native source/build flags and host libraries, all commands, resource snapshots, arrays, exports and audits are retained. See execution.json for content hashes. E412’s later package numeric failure concerns different weights and is not superseded by E416.
