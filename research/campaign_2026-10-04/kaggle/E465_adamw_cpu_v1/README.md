# E465 AdamW FP32 CPU parity transport draft

This packet is a private, CPU-only transport for the staged actual-FP32 JAX/Torch AdamW parity fixture. It contains only `worker.py`, the immutable staged `rnn.py` and `onecycle.py`, and `check_e465_adamw_fp32_parity.py`. It attaches no model checkpoint, labels, cache, or dataset. The worker compares the compiled JAX update scan with scalar Torch AdamW for 7,460 dynamic-beta schedule steps; it does not load a model or run a fit.

The source identity is pinned to the staged E465 files:

| File | SHA-256 |
| --- | --- |
| `src/wnn_jax/rnn.py` | `294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc` |
| `src/wnn_jax/onecycle.py` | `f857fe857a361249c2742e3486cd32b846e0f47413a5c4db2ae1fb6c65509ded` |
| `scripts/check_e465_adamw_fp32_parity.py` | `612435a3ffe899e8c69ea3cb69aec009bcbbe03faa16c0344e1450c427982861` |

The environment target is Python 3.13, JAX/JAXlib 0.11.1, NumPy 2.1.3, and Torch 2.11.0+cpu. Python/JAX/JAXlib/NumPy were observed in the E465 CPU v3 runtime. That runtime did not record Torch; the Torch version is taken from the separate E458 CPU parity v3 runtime, which recorded the same Python/JAX/JAXlib/NumPy versions and Torch 2.11.0+cpu. The worker records package versions and fails closed on mismatch, CUDA visibility, a non-CPU JAX backend/device, or CUDA-enabled Torch.

The private kernel metadata disables GPU, TPU, and Internet. A 600-second alarm begins at wrapper entry, before output setup, manifest hashing, archive checks, or source extraction checks. The external loop independently limits the child process group plus wrapper RSS to 4 GiB, the worker to 300 seconds, and CPU affinity to two cores. Its host-headroom check reserves 20% of host RAM plus 512 MiB beyond the RSS cap. It sets BLAS/OpenMP/XLA threads to one. These limits are copied from the previously verified CPU transport pattern and narrowed to the requested budget.

`prepare_packet.py` builds a deterministic source archive and source manifest with exact file hashes, renders those hashes into the wrapper, and emits private kernel metadata. It refuses to overwrite an existing archive. Five stdlib-only worker-contract tests exercise exact success, failed status/exit, missing output, and checker/packet source mutation rejection. This is a source packet only: no upload, push, or remote invocation has been made. The next review should validate the packet and actual private runtime inventory before any launch.

The E465 formula-only FP64 reference check remains a separate local result and is not part of this packet. The staged FP32 numerical fixture is still unrun in this transport; its known comparison tolerances are parameter `5e-6`, first moment `1e-6`, and second moment `1e-5`.
