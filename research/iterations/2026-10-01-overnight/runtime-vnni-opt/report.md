# Dynamic INT8 feasibility: not promoted

The fixed four-member ensemble became substantially cheaper, but no tested version met the frozen 50-minute runtime target. On the controlled CPU18 repeats, the best variant averaged 86.15 µs per complete row callback, projecting to 56.57 minutes for 39.4 million rows. The required threshold was 76.14 µs. No weighted-Pearson evaluation, target read, training, deployment, or public submission was performed. The bounded optimization stage is closed.

The single numerical operator was frozen in `../runtime-vnni/proposal.md`: symmetric per-output-row weight INT8, per-input-vector dynamic INT8, uint8-by-int8 VNNI dot products with explicit zero-point correction, exact int32 accumulation, and float32 rescaling. Input projection, nonlinearities and final output head remain float arithmetic. The initial quantized outputs differed from fp16 by relative RMSE 0.00418/0.00433, p99 absolute error 0.00871/0.00922, and maximum absolute error 0.02796/0.04507 on the five raw sequences. These are engineering diagnostics, not evidence that the competition metric is preserved.

Every subsequent implementation produced the same complete 100,000-row prediction hash as that initial quantized operator. Scalar integer and float-rescale oracles passed 160 matrix cases and 26,200 output rows, including signed extremes, zero inputs, nearest-even ties, arbitrary output tails, and invalid-width/nonfinite-input rejection. Projection variants additionally matched the original float projection bitwise on 100 synthetic cases. Sequence reset, future-prefix independence, streaming and independent-member sum checks passed for the initial quantized mode.

Controlled timing used the same CPU18, one thread, scorer-image Python3.11, five raw sequences, and forward/reverse condition order. Per-run sibling CPU58 utilization stayed below the preregistered 5% bound. Complete receipts retain all CPU utilization counters:

| Implementation | Complete callback, µs/row |
| --- | ---: |
| Initial row-major VNNI | 96.40 / 98.40 |
| Packed 64-output tile | 93.85 / 93.12 |
| Packed tile plus register-held projection | 88.94 / 86.81 |
| Packed 128-output tile, eight independent chains | 87.08 / 85.22 |
| Explicitly aligned packed weights, single follow-up | 88.82 |
| YMM-only dot/quantization/projection, single follow-up | 90.58 |

Earlier CPU22 results are preserved separately, including its attractive 81.07 µs register-projection result and the 119.72 µs block128 result. A subsequent `/proc/stat` sample found its SMT sibling62 56% busy. CPUtime approximately equalling walltime does not detect that contention. These earlier measurements are not substituted for the controlled comparison. The YMM-only kernel contained no ZMM instructions in the inspected GEMV disassembly; its result did not support the proposed frequency-related benefit.

Component profiling identified projection and recurrent matrices as the remaining work. Event feature construction across all four members was only about 0.77 µs, so sharing that state was not implemented. Dynamic input quantization already used SIMD and cost roughly 0.06–0.16 µs per input vector. System performance counters were unavailable (`perf_event_paranoid=4`); permissions and system settings were left unchanged.

All source variants, failed attempts, proposals and JSON receipts remain available. `stage-summary.json` contains the structured conclusion; `stage-freeze.json` hashes the current local source and receipts. CP311 native binaries and raw prediction arrays remain at the corresponding remote paths on beleriand. No portable integer fallback was built because the runtime gate failed. A future candidate would still require explicit ISA dispatch, an exact integer fallback without saturating `maddubs`, independent validation and a separately registered held-fold metric evaluation.

A separate successful deliverable is the independently reviewed exact-order engine in `../critic/native_order_v2/`. The original order-only source had a subtle coefficient-dependent compiler contraction difference. The critic's derivative explicitly uses FMA on FMA targets and a rounded product on v2. This agent independently verified all three ISAs on six member/coefficient compositions, including nonbinary coefficients, duplicate h256 models, heterogeneous order, reset and prefix checks. Those receipts are `independent-order-v2-{v2,avx2,v3}.json`. This correction is separate from the unpromoted quantization study.
