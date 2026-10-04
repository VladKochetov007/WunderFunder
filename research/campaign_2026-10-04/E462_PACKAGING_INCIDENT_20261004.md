# E462 submission packaging incident — q8 ZIPs withdrawn

The user reported `CHECK:ERROR` / `PRERUN ERROR` for submission **1yv88h4s**
(`E462_public_q8_build_v2q8r.zip`), portal time **2026-10-04 23:50**, with:

> ImportError: libpython3.13.so.1.0: cannot open shared object file

There is **no public score**. This was an avoidable packaging failure, not an
expected experiment outcome. The research goal remains paused. The work below
repairs packaging and verifies compatibility; it starts no training or kernels.

## Withdrawn artifacts

Both original E462 ZIPs use the same affected native libraries:

| Variant | Original ZIP SHA-256 | Status |
| --- | --- | --- |
| q8r | `7edb26d8903bafa2f3c38bf31b1a5d0dad1fd6f79e1f4d5a0ea13eaf3ca044fd` | Public prerun failure; withdrawn |
| q8x | `b435c04c0b047c72aeaca93a0be5aa2436a885d3abce5da483bd7fa43bfa87e7` | Same dependency defect; withdrawn before submission |

The renamed q8r ZIP's bytes match the recorded original SHA-256. Keep the failed
archives and raw historical reports for provenance; do not upload either.
This notice supersedes their entries in the October 4 submission shortlist.

## Cause and missing check

`readelf -d` finds `DT_NEEDED: libpython3.13.so.1.0` in both v2 and v3 binaries.
E416 built these on the research host with `cargo --features python`, without
PyO3's `extension-module` feature. The `abi3-py311` feature and `.abi3.so` name
did not remove the version-specific linker dependency. E462 reused these host
libraries. Local parity/scoring on Python 3.13 did not test the scorer's Python
3.11 environment. E416's build note already said deployment required a
scorer-image build; packaging failed to enforce that requirement.

The public-confirmed 0.6655 i16 ZIP, fixed parent/head endpoints and seed-4 i16
ZIP do not have this direct libpython dependency. They are being checked from
their actual ZIP bytes as well. This does not waive any outstanding runtime gate.

## Repair and prevention

Rebuild the exact frozen E416 numerical source in the Python 3.11 / Debian
bookworm build container with explicit `python,pyo3/extension-module` features.
Preserve weights, quantization codes, callback and member bytes. Build distinct
repair ZIPs without overwriting the failed originals. Require static ELF checks
and actual import, constructor and streaming callbacks in an offline Python 3.11
container before compatibility approval. Record each tested ZIP's hash.

## Completed compatibility verification

The original q8r ZIP reproduced the exact missing-library failure in a clean
Python **3.11.16** / Debian bookworm container. The rebuilt q8r/q8x ZIPs pass
static ELF checks and have **no libpython dependency**. Their callback, weights,
member metadata and quantization-code bytes are unchanged; the numerical Rust
source is byte-identical to the E416 snapshot. Only binaries and their build
provenance changed.

All **16 positive checks** passed in fresh, network-disabled Python 3.11
containers: v2/v3 for both repaired q8 variants, and v2/AVX2/v3 for the incumbent,
head, parent and seed-4 i16 ZIPs. Each check streamed two complete sequences
(40,000 rows), then tested a sequence reset. Across **640,000 streamed rows**,
every output was **bitwise identical** to that package's archived native output.
All positive exits were zero; Docker reported no OOM kills. Each check had one
CPU and 1 GiB RAM, no GPU and no swap allowance above its memory cap.

The native build configuration now exposes an explicit `extension-module`
feature. Native/submission builders invoke it and reject invalid ELF dependencies.
Two regression tests passed, including a real ELF that falsely bears an
`.abi3.so` filename while linking to a mock `libpython3.13.so.1.0`.

Receipts, frozen-source and ZIP hashes, original failure logs, reproducible build
commands and per-container resource/exit evidence are in
[E462_python311_repair_v1](E462_python311_repair_v1/compatibility_report.json).

**No repaired q8 replacement is recommended for upload yet.** Compatibility is
fixed and verified; the pre-existing competition runtime requirement remains
outstanding and research is still paused. Head/parent endpoints retain their
earlier runtime approval and now have actual offline Python 3.11 checks too.
