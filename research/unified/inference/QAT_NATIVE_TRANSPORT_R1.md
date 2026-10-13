# Native dynamic Q8 transport and state diagnostics — R1

This is the native engineering milestone for the QAT deployment plan at
[`campaign_2026-10-13/qat_renewable_r1/PLAN.md`](../../campaign_2026-10-13/qat_renewable_r1/PLAN.md).
It adds exact row-code/scale import and export for the existing `q8vnni`,
`q8vnni_avx2`, and `q8vnni_portable` kernels, plus a read-only recurrent-state
diagnostic. It does not run training or evaluate model quality.

The Python transport is a mapping with two entries per native dynamic matrix:
`<matrix-name>.codes` is dense row-major `int8`, and `<matrix-name>.scales` is
one `float32` per row. Matrix names and shapes come from the initialized native
model; recurrent names include `gru.weight_ih_l{k}.0` and
`gru.weight_hh_l{k}.0`, while the first head is split into
`head.0.weight.0` and `head.0.weight.1`. Import requires every expected code
and scale key with no extras and checks each exact shape before allocation.
Accepted signed-byte values are `[-127, 127]` (`-128` is reserved); scales must
be finite and nonnegative, and a zero-scale row must have all-zero codes.
Import calls `DynQ8::from_rows` and does not requantize. The Q4 archive path
expands packed nibbles once in Python and passes signed `int8` codes in
`[-7, 7]` with their original scales through this same mapping. Bit-width and
packed-archive metadata remain the Python archive's responsibility.

`FastRnn::dyn8_matrices` exposes the native matrix identities for exact
export, and `load_dyn8_matrices` validates identity, order, count, and matrix
dimensions before replacing constructor-quantized matrices. Existing `q8r`
and `q8x` constructor/import behavior is unchanged. This work leaves the
existing native projection path intact; it does not change projection
precision, recurrent-state precision, biases, or the final head. Existing
`FastRnn` output behavior is checked with identical imported bytes across
portable, AVX2, and normal runtime-dispatch precision selections.

`FastRnn::hidden_state_shape()` reports `(layers, hidden)` and
`hidden_state()` gives a read-only flattened borrow, one hidden-width row per
layer. Python `Predictor.hidden_state` and `ModelStepper.hidden_state` return a
detached, read-only array with shape `[layers, hidden]`; taking a snapshot
copies state only when requested, with no per-step work added. Reset restores
the all-zero state.

## Provenance and validation

The isolated implementation worktree started at source revision
`df93b75b6b7098f6dc8fb1b736538aa388c05191`. This milestone changes only
`native/src/vnni.rs`, `native/src/rnn_fast.rs`, and `native/src/python.rs`,
their in-file Rust tests, and this note/index entry. The exact native changes
cover invalid dimensions, missing/excess keys, invalid signed bytes, invalid
or zero-scale rows, exact code/scale export, matrix identity, portable/ISA
consistency, reset/replay, and Python snapshot behavior.

Under the campaign's one-CPU / 1.5-GB / 300-second wrapper, the focused Rust
suite passed: 60 tests passed, 0 failed, and 1 pre-existing test was ignored.
A Python 3.13 extension smoke also passed using a synthetic two-layer model:
it imported an expanded Q4 mapping into both `Predictor` and
`EnsemblePredictor`, verified exact code/scale export, then checked hidden
state shape/read-only status, reset-to-zero and deterministic replay. This is
interface evidence only, not an actual-data, quality, package-size, or runtime
result.

Run the Rust and Python-bridge unit tests from the isolated worktree with the
same physical target directory:

```sh
LD_LIBRARY_PATH=/home/vlad/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib \
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=300s CARGO_BUILD_JOBS=1 \
CARGO_TARGET_DIR=/home/vlad/development/wunder/wnn_qat_native_r1/native/target \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
cargo test --manifest-path native/Cargo.toml --features python
```

Build the Python extension with:

```sh
LD_LIBRARY_PATH=/home/vlad/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib \
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=300s CARGO_BUILD_JOBS=1 \
CARGO_TARGET_DIR=/home/vlad/development/wunder/wnn_qat_native_r1/native/target \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
cargo build --manifest-path native/Cargo.toml --release --features extension-module
```

The built Python 3.13 module is at
`/home/vlad/development/wunder/wnn_qat_native_r1/native/target/release/wnn_native.so`
(a symlink to `libwnn_native.so`). The library SHA-256 is
`a4390ecf7bcd54bb6826b43f5dcf6d7a40064c64ed1450bdc1780c34c2a6e108`.
The target directory is a local ignored build artifact, not committed source.

No speed or INT4-execution claim follows from this transport: kernels still
consume signed bytes. No quantized quality or full-trajectory result is
claimed; those actual-data runs and grading remain owned by the root actor.
