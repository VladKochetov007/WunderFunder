# Scorer CPU spec (organizer-supplied lscpu, 2026-10-10)

- AMD EPYC 9224, Zen 4 (family 25 model 17), 24C/48T, 1 socket, 1 NUMA node. Max 2.5 GHz (min 1.5), boost enabled.
- ISA: avx2, fma, avx512f/dq/cd/bw/vl, avx512ifma, avx512vbmi/vbmi2, **avx512_vnni**, avx512_bf16, avx512_bitalg, vpopcntdq, gfni, vaes, bmi1/2.
- Caches: L1d 32 KiB/core, L1i 32 KiB/core, L2 1 MiB/core (24 MiB total), L3 64 MiB in 4 instances (16 MiB per CCX). Line 64 B.
- Organizer Dockerfile: python:3.11-slim-bookworm, libgomp1, build-essential present; offline; HOME=/app.
- Organizer (sggpls 2026-09-25): AVX2 definitely supported; custom C extensions welcome if source included and open.

## Consequences
- q8vnni v3 (AVX-512 VNNI) path is valid on scorer; our i9-11900H (Tiger Lake, true 512-bit) is faster per clock than Zen 4 (double-pumped 256-bit datapath for 512-bit ops) and boosts higher (4.9 GHz vs 2.5-3.7 GHz). Local->portal calibration (0.616x) was measured on portal, so it already includes this; but VNNI speedup may differ from local.
- Whole Z5 model weights (5 MB int8-ish) fit in 1 MiB L2 only partly; per-layer GRU 256x768 int8 = ~200 KB x ~3 matrices fits L2. L1d 32 KiB is small: tile vectors.
- Probe ZIP (0_cpu_probe) should still be uploaded first to confirm actual runtime CPU (lscpu flags on the scorer VM may be masked).

## Haiku microarchitecture digest (sourced; inference marked)
- Zen 4 datapath 256-bit; 512-bit ops = 2 cycles through 256-bit pipes (Hot Chips 2023). VPDPBUSD zmm: latency 4, recip. throughput 1.0, FP0/FP1 (uops.info). No AVX-512 downclock (Phoronix). Peak ~256 int8 MAC/cycle/core (inference).
- L1D 32 KB / 4 cyc; L2 1 MB / 14 cyc; L3 victim cache ~8-9 ns, ~27 B/cyc/core; L1 DTLB 72, L2 DTLB 3072 (Chips and Cheese). Cacheline-crossing stores cost 2 cycles; store-forward failure ~19 cyc.
- docs/submission_guide.md: 1 vCPU, 16 GB, Python 3.11, 60 min full test set. L3 shared with host neighbours.
- Recs: 4-8 independent accumulators per output block; int8 weights resident in L2 (~0.8 MB vs 1.6 MB int16 spills to L3); 64B-aligned tiled packing; check asm for zmm vpdpbusd; vzeroupper at dispatch boundaries; LUT/poly nonlinearities, no gathers; overlap next-step input GEMV with recurrent GEMV; measure on scorer with probe.
- Unverified: SMT sharing of AVX-512 units, gather throughput, FTZ behaviour, 64B line (standard for x86).
