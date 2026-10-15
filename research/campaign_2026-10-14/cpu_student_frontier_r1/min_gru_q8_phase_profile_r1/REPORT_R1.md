# minGRU Q8 phase profile results

The current merged native source builds successfully with the profile feature, and all three profiled complete callbacks match the sealed uninstrumented minGRU Q8 output and final state bit for bit. Each run reset state first, asserted `q8vnni` and fast activations, processed the full 20k-row raw112 sequence through the causal 211-feature builder, and produced finite predictions and state.

For the instrumented build, the three complete callback times were 1.03895s, 0.92752s and 0.96537s (mean 0.97728s). The uninstrumented baseline run in the same process came first at 1.06991s; the earlier uninstrumented release profile averaged 0.93937s. The instrumented mean is 4.04% above that earlier average and 8.66% below the same-process first run. Because the baseline was first, later runs had warmer caches, and other admitted jobs shared the host, these values do not isolate instrumentation overhead. No positive overhead estimate is claimed; the full instrumented wall-time range is reported for context.

The measured RDTSC phase shares of the inclusive raw callback body were:

| Phase | Ticks per row | Share |
|---|---:|---:|
| Candidate, write gate and state update | 43,799 | 36.84% |
| Q8 activation quantization | 28,181 | 23.70% |
| Q8 matvec | 27,440 | 23.08% |
| Encoder and head fast GELU combined | 16,712 | 14.06% |
| Raw sanitation and causal feature builder | 756 | 0.64% |
| FP32 two-output matvec | 359 | 0.30% |
| Normalization | 228 | 0.19% |
| Skip concatenation | 166 | 0.14% |

The two largest phases and Q8 matvec account for most callback work. This narrows the earlier roughly 3.04× equal-matrix-count Q8 gap to three candidate areas: scalar candidate/write/state loops, scalar activation quantization, and matrix work. The source comparison shows the GRU Q8 path uses AVX2 quantization and vector GELU/update helpers, while minGRU's current loop uses scalar implementations. The input quantization spans are 211+384+384+768=1,747 values per minGRU row, compared with three 256-wide hidden vectors in the matched two-layer GRU path. This is consistent with the phase measurements, but the profile does not causally assign the full gap to any one implementation choice.

The source uses the same `fast_activations=true` flag and VNNI512 Q8 ISA for both model families. GRU's encoder is I16, while the minGRU encoder is among its Q8-weight linears; these are different precision allocations despite both workers being labeled Q8. The minGRU recurrence and candidate/write equations were not changed by profiling.

The release extension was built from source commit `8521382a1bba4242bc64e5b603f75a3743a66c8d`, which includes current MAIN's independent encoder/recurrent/readout width implementation and native minGRU source. The build establishes compile compatibility; the profile callback specifically exercises native minGRU, not the independent-width GRU constructor. Python 3.11.16 with NumPy 2.2.6 loaded the abi3 extension. The Python process maximum RSS was 621,481,984 bytes; the worker's two-gigabyte systemd cap and aggregate snapshots are recorded in the gate receipt. No OOM event increased.

This is one synthetic model and one fitting-membership sequence used only as a deterministic serving input; labels and masks were not loaded. It establishes no fit, quality effect, CPU capacity, training benefit, public transfer, or competition-host speed. A follow-up may reuse the existing bit-exact AVX2 quantizer/GELU/sigmoid helpers for a limited performance test; any such change remains separate from this measured baseline and must pass full-sequence bitwise parity before speed is interpreted.
