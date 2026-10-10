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
