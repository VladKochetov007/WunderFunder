# E487 source packet v2

Version 1 source hashes are preserved under `versions/v1/`; the v1 diagnostic was not launched. Independent source review found three pre-launch proof gaps. Version 2 closes them without changing the registered numerical recipe or gates:

1. `capture_actual_fit_batch` now reads only the 24 manifest-declared fit row groups, checks each raw 20,000×20 feature matrix for finiteness before calling the frozen E470 loader (whose `causal_features` uses `nan_to_num`), and records ordered row groups and raw-feature digests. Synthetic tests inject NaN and both infinities and demonstrate that E470 would sanitize them while the new check rejects them.
2. The GPU probe sets `WNN_GPU_MEMORY_BYTES=2147483648` before CUDA availability/configuration or tensor allocation, so the allocator limit is active during work rather than checked only afterward. The device-wide 3,000,000,000-byte stop remains separately enforced.
3. Every replay explicitly checks all populated optimizer-state tensors and numeric scalars for finiteness after the step. The crossed-finite receipt is derived from the four actual replay states/updates, not hard-coded.

The CPU-only v6 preflight passed on the same four fit groups and confirms all 24 raw fit groups were checked; it opened no test-role rows. CPU uses the frozen FP32 orthogonalizer branch, so its identical BF16-source/FP32-diagnostic results do not resolve the CUDA precision question. No CUDA probe or fit was launched in this packet.

The strict E485 native/custom update gates remain 2% family/matrix RMS and 2e-5 maximum absolute delta error. Expected finite gate misses remain recorded so a BF16 miss cannot suppress the FP32 diagnostic cells. CPU/CUDA elementwise orthogonalizer parity remains `1e-5 + 1e-4*abs(CPU_reference)`. Same-gradient helper controls remain exact.
