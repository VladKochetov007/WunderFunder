# E485 GPU probe admission checks

The v3 probe is a single fit-only diagnostic. Its source keeps the registered
early-stop checks at 10% relative update-delta RMS and 0.5 times reference RMS
for the per-tensor maximum error. Passing those source thresholds is not
sufficient for full-trajectory admission. Before any E485 full fit is sealed,
root must inspect `gpu_probe_v3/result.json` and independently require, for
each of the four GRU matrices and each emitted AdamW family, both:

- update-delta error RMS / reference update RMS <= 0.02;
- maximum absolute update-delta error <= 2e-5.

Also require the output/state/input-gradient/parameter-gradient native parity
checks to pass, exact shared-gradient local-vs-frozen-E474 update and optimizer
state bytes, active FP32 backend flags captured inside the full-IEEE context,
six finite parameter-changing actual-fit-only updates, allocator peak <=2 GiB,
observed device memory <=3,000,000,000 bytes, and a measured full-matrix
projection <=7,200 seconds. Check the resource receipt and all artifact/source
hashes before deciding. Any failed or missing condition is a no-go; do not
relax a tolerance based on the observed result.

This document records the external strict root gate. The probe executable's
own looser delta bounds remain an early-stop diagnostic and must not be
misreported as sufficient admission evidence. No full trajectory or test-role
data read is authorized by this document.
