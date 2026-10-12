# Python 3.11 ABI correction and callback comparison R1

## Result

The shared-F0 prototype builds and imports through the existing CPython 3.11
abi3 route. The reviewed ELF checker passes: the extension needs no `libpython`,
and its highest GLIBC symbol requirement is 2.34, within the competition's
bookworm GLIBC 2.36 envelope. One full Python 3.11 callback comparison also
passes exact output-hash parity and reproduces a large local runtime reduction.
This remains an isolated, benchmark-only prototype. It has not passed target
AMD runtime, full-stream quality, or final packaging/selection gates.

## Frozen source and corrected binding

This packet supplements and leaves untouched the original Python 3.13 receipts
and the implementation commit. The actual source commit is
`70749a118e1333fd04f1e09596de37b1d9126c80` (tree
`9287db9fb059aac4fbfe027c8209f69204c058b2`), based on
`759c1450cd4852899ac518c2db1fdbe997af6ba1`, in branch
`codex/shared-hft-backbone-20261012`. The source files at that commit hash to:

| File | SHA-256 |
|---|---|
| `native/src/python.rs` | `d9673f45b4e411177435f44bdf87447c0bdeec94a844160b8eb799f3b19f363a` |
| `native/src/rnn_fast.rs` | `6f6d1868a3d2b92b5efc1829013e2b6b9387cede3096a64548343b38727d43ec` |
| `native/Cargo.toml` | `fa085c1000de9b8e398f2c99a4ed7804594542976007f69ab8b8d6cee8639974` |
| `native/pyproject.toml` | `ce3fef53184a0fecae9a0928196f8a7e5359e64ed699470ae6419cd6c1fc907d` |
| `benchmark_shared_f0.py` | `0333be5b058298b8dab2227a2581a1540d0748aee5ae1aa50d2a59a9deab09d3` |
| `benchmark_shared_f0_py311_bridge_r1.py` | `12f5305ff74d28a03ff7859a0320937f84d7fb4f34b2fc3a8183f71804ba2d59` |

The first-generation R1 summary and pair receipts contain a malformed
40-character `source_commit` transcription (`70749a11b6e71209...`). Their
recorded eight-character prefix resolves to the actual commit above. Those
frozen receipts were not edited; this additive receipt pins the valid full
object ID and source-file hashes directly. The frozen benchmark program hash
matches the file at that commit.

The own F0 checkpoints and grade ancestry are unchanged: student
`oldz4_h230_student_f0_s3_r3/model_ep1.pt`, SHA-256
`664643b2653a85d192f74a976d76afa933ae56c02b2621387d2f576fd61695eb`; HFT
`oldz4_h230_hft_f0_s3_r5/model_ep0.pt`, SHA-256
`2bde556119ededc3693733280444a42f0f64138a2e3759bed6b068cb72f18ec8`.
The benchmark used their actual float32 tensors, retained distinct head
nonlinearities, and fixed the output blend at 0.5/0.5. No labels were loaded.

## Python 3.11 wheel and ABI gate

The build used the already-installed reviewed interpreter
`/tmp/wnn-projection-precompute-r1/py311/bin/python`, CPython 3.11.16,
SOABI `cpython-311-x86_64-linux-gnu`, with NumPy 2.2.6. Maturin was 1.15.0,
Cargo and rustc were 1.94.0 (LLVM 21.1.8). `native/pyproject.toml` selects
`extension-module`; Cargo enables `pyo3/extension-module`; PyO3 0.29.2 is
configured with `abi3-py311`. Release settings are opt-level 3, fat LTO,
one codegen unit, and panic-abort. The build log is preserved at
`BUILD_PY311_R1.log`.

The actual artifact is
`/tmp/wnn-shared-f0-py311-wheel-r1/wnn_native-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl`,
SHA-256 `b4d71f3e06b51027d46bb74a7b2317405a69221f883e4c9c0efe2c8ea68f979f`.
The extension inside the wheel is `wnn_native/wnn_native.abi3.so`, SHA-256
`f8aceef7322ee4deb3bf49992afcd0fec01ce5934a27d5269112a2052b2f5326`.
`scripts/check_submission_abi.py` (SHA-256
`a00bfca90d444c338eb7aff5ac4c99d6693aca552475ba1e6a782c94c9d76f13`)
passed on that exact wheel. Its `DT_NEEDED` set is only `libgcc_s.so.1`,
`libm.so.6`, `libc.so.6`, and `ld-linux-x86-64.so.2`; no `libpython` is
required. Maximum GLIBC requirement is 2.34. Full machine output and checker
log are `ELF_ABI_PY311_R1.json` and `ELF_ABI_PY311_R1.log`.

The wheel was unpacked and imported using this CPython 3.11 environment. The
Python 3.11 self-test verifies default-disabled equals explicit
`shared_backbone=False`, shared Q8 outputs equal the default path, and an
incompatible trunk is rejected (`API_SELF_TEST_PY311_R1.json`). The import and
callback run did not use a Python-library `LD_LIBRARY_PATH` workaround. The
prior Python 3.13 test's loader issue is therefore resolved by the supported
abi3 package path rather than by an environment workaround.

The first Python 3.11 pair receipt's `runtime.native_module_path` is the
package `__init__.py`, and its same-named hash is that initializer's hash; it
does not itself record the ELF path. The frozen benchmark's
`load_native_module()` imports `wnn_native`, and that exact package initializer
contains `from .wnn_native import *`. A post-pair import probe with the same
Python 3.11 interpreter and the pair-recorded package directory supplied as
`PYTHONPATH` resolves the extension to
`/tmp/wnn-shared-f0-py311-package-r1/wnn_native/wnn_native.abi3.so`, SHA-256
`f8aceef7322ee4deb3bf49992afcd0fec01ce5934a27d5269112a2052b2f5326`; it
matches the extension inside the ABI-checked wheel. The pair's recorded
initializer path/hash, frozen relative-import code, and package path therefore
bind its callback run to that ELF. The original pair receipt still
does not contain a direct `sys.modules` ELF hash; that gap is preserved and
made explicit. Probe output is `ELF_IMPORT_PY311_R1.json` (stdout is also
preserved in `.log`).

## Fair-control Python 3.11 callback result

The control and treatment execute the same frozen `benchmark_shared_f0.py`
benchmark and `benchmark_submission_callbacks.run_callbacks` feature/sequence
path. Both instantiate `native.EnsemblePredictor` with the same pinned weight
arrays, weights `[0.5, 0.5]`, `fast_activations=True`, and `precision="q8vnni"`;
the sole method difference is `shared_backbone=False` versus `True`. Thus the
comparison is not two Python predictors against one native predictor, or two
different feature paths.

Python 3.11's existing environment does not contain Torch, so the committed
benchmark-only bridge first exported actual tensors from the two pinned
checkpoints with the frozen loader under the project's Torch-enabled Python
3.13 environment. Archive SHA-256 is
`774a33de04b8ad330fab24d469bd70217f1341dcaa3361f8b195f8d7015f9120`;
export receipt SHA-256 is
`1ece1d9a1891befe95ee9f3be1d6c6e01eb2fadcf9da616f1f0f8589aba13ee0`.
The bridge verifies the archive hash and exact tensor-key set, then changes
only the frozen loader's source of those arrays. The bridge source hash and
all tensor producer metadata are bound in `PY311_PAIR_R1.json` and
`PY311_WEIGHT_EXPORT_R1.json`.

The frozen panel was the features-only file
`/tmp/wnn_submission_exploit_runtime_r1/valid_first10_complete_groups.parquet`,
SHA-256 `e5e6472cedfcb20ecf01195777ab986b3560379c5e81caea71973632dde0d6a3`.
Each method ran ten complete 20,000-row sequences over two reset passes, or
400,000 callbacks. Results:

| Method | Mean callback time | Prediction hashes |
|---|---:|---|
| Separate backbones (`shared_backbone=False`) | 58.50839 µs/row | Exact match |
| Shared backbone (`shared_backbone=True`) | 28.72919 µs/row | Exact match |

This is 50.8973% lower callback time in this single bounded comparison.
All ten sequence output hashes match between methods and each method reproduces
identical predictions across resets. The full receipt is
`PY311_PAIR_R1.json` (SHA-256
`1c3b89bb4e2e79d4dc224fe57111f097a557076a83c2478492447fc987a1c249`).

The measured host was an Intel i9-11900H, pinned to CPU 0, not the target AMD
EPYC 9224. The benchmark child had `cpu.max=100000 100000`, memory max
1,199,996,928 bytes, memory peak 209,920,000 bytes, and zero `high`, `max`,
`oom`, or `oom_kill` events. The parent campaign slice was about 13.92 GB at
the ABI-correction check, with 18 GB high, 20 GB max, no swap, and 500% CPU
quota. Build, checker, and callback execution used the absolute main-repository
`scripts/limited`, one CPU/one thread, 1.2 GB, 120 seconds per command, and one
Cargo job. No additional benchmark pair was run.

The previous pair receipts' `runtime.native_module_sha256` field hashes
`wnn_native/__init__.py` (0cbf62fddaa57ad9aae99f62e03cd21af51050a02ec44f55e1027df320e05eae),
not the ELF extension. `PY311_PAIR_R1.json`'s Python 3.11 initializer path is
`/tmp/wnn-shared-f0-py311-package-r1/wnn_native/__init__.py`; the direct
import probe binds the sibling ELF. This naming correction is additive; the
original receipts remain frozen.

## Command and attempt record

The bounded release build used:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s CARGO_BUILD_JOBS=1 \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/maturin build \
--release --manifest-path native/Cargo.toml \
--out /tmp/wnn-shared-f0-py311-wheel-r1 \
--interpreter /tmp/wnn-projection-precompute-r1/py311/bin/python
```

The ABI checker invocation used that wheel as its positional artifact and
wrote `ELF_ABI_PY311_R1.json`; its log includes the complete dependency report.
The callback invocation used the committed bridge with `--run-py311`, the
hash-pinned weights/export receipt, `--order separate-shared`, `--cpu 0`, and
the output path `PY311_PAIR_R1.json`, under the same limited wrapper and
`PYTHONPATH=/tmp/wnn-shared-f0-py311-package-r1`. The bridge script hash,
arguments that affect method/control, artifact pins, runtime environment,
panel identity, and outcome are in the JSON receipts. The original shell
command line was not copied into the first pair log; the above describes the
executed argument set without modifying that frozen log.

The system `/usr/bin/python3.11` was found without the required NumPy/PyArrow
environment, so it was not used and no packages were installed. A read-only
source hash probe also referenced nonexistent `native/src/ensemble.rs`; the
actual modified recurrent implementation is `native/src/rnn_fast.rs`, which
is pinned above. Neither attempt changed source or executed a benchmark.
