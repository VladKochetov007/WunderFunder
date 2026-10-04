# Private accelerator transport

Authorized by the root `EXECUTION_CONTRACT.md`; no competition upload commands
are implemented here. The user-owned root `kaggle_kernel/` is untouched.

## TPU inventory milestone

Private kernel: `vladkochetov07/wnn-tpu-probe-20261004`, version 1.
Requested accelerator `TpuV5E8`; provider timeout 900 seconds. Each framework
child has a 300-second timeout. Internet is disabled and no packages are installed.
The explicit tensors are far below 128 MiB; framework/compiler memory is separately
reported and is not claimed bounded by that tensor allowance.

`tpu_probe_v1/` preserves submitted bytes, source binding, push log and pre-run
quota. JAX records device topology/HBM and fp32/bf16 GRU-equation gradient/Adam
checks. PyTorch/XLA, when installed, runs the exact source-bound repository
RNNModel at hidden16, batch1, length8. Unsupported frameworks are recorded as
failures. Neither check establishes full training parity or a scientific result.
At handoff the provider still reports QUEUED; no TPU hardware result is claimed.

The accelerator selector and private metadata follow the official CLI documents:
[kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
and [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).

## Portable dual-T4 runner

`scripts/kaggle_bundle.py` accepts sealed version1 campaign specs. Code descriptors
are sorted by path and hashed as sorted-key compact JSON. Every declared code/input
file is hashed before packaging. Include `scripts/kaggle_common.py`,
`scripts/kaggle_gpu_runner.py` and `scripts/kaggle_kernel.py` in sealed code.
All required trainer modules, caches, checkpoint parents and split/portability
manifests must be explicitly listed. The builder does not infer dependencies.

Use `--external datasets/train.parquet=train.parquet` and corresponding entries
for valid/mask to mount the private raw dataset instead of copying its 35 GB.
The raw files must still have complete content identities in the spec. Other
declared inputs enter `bundle.wnn` (gzip tar bytes with an opaque extension because
Kaggle expands recognized archive extensions). The dataset metadata is private; create
without the CLI `--public` flag. Generated kernel metadata also remains private,
attaches only the bundle and `vladkochetov07/wunder-funder`, and disables internet.
No provider creation or push happens inside the builder.

The generated bootstrap binds the expected archive SHA, rejects archive links,
duplicate/undeclared entries and traversal, then creates logical raw-data symlinks.
The runner verifies every dependency again before launching any job. Checkpoint
and result artifacts are exported with byte hashes. Failed jobs retain outputs,
logs and an incident record; they do not count as valid results.

One worker runs per physical T4. CPU affinities do not overlap. Host RAM/CPU
reservations determine whether two workers fit; the second starts at least30s
after the first. Defaults come from the frozen spec, with maxima12GiB operational
GPU memory and13GiB stop. The repository allocator may impose a lower cap.
An external watchdog samples every0.25s and terminates admitted process groups
on resource/timeout/monitor failures. This is a sampled stop mechanism, **not a
provider-enforced RSS or GPU hard cap**. Provider quota charging remains empirical;
capture `kaggle quota --format json` before and after each run.

Local checks run through `scripts/limited` in `wnn-campaign.slice`, one CPU,
2GiB, no GPU,120s timeout. `test_kaggle_transport.py` covers privacy/source binding,
tampered input rejection, aggregate admission, output symlink escape, successful
and failed worker lifecycle, and external-data bundle construction. Real remote
dual-T4 training compatibility is still pending controller review/run.
