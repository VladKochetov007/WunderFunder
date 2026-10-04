# Native build provenance

Source commit: b19996f (native files unchanged between successful builds and commit).
Exact Rust source and Cargo manifests are archived in the sibling build directory.
Libraries are host builds; deployment still requires a scorer-image build.

Both builds used the root `.venv/bin/python` as `PYO3_PYTHON`, one Cargo job,
`--release --features python --manifest-path native/Cargo.toml`, and respectively
`RUSTFLAGS='-C target-cpu=x86-64-v2'` / `x86-64-v3`.
They ran through `scripts/limited` under `wnn-campaign.slice`, CPUQuota=100%,
MemoryMax=2G, MemorySwapMax=0, RuntimeMaxSec=900, CUDA hidden. Target directories:
`/tmp/wnn-q8-cargo-v2` and `/tmp/wnn-q8-cargo-v3`.

Captured tool results: v2 build exit 0, 30.59 seconds, peak RSS 386008 KiB;
v3 exit 0, 23.17 seconds, peak RSS 388864 KiB. The full console logs were not
persisted; these are transcribed execution summaries. The archived libraries
have content hashes bound by the E416 prepared manifest.

Pre-probe checks: native release tests 45 passed, 1 preexisting ignored;
Python export/quantization/native import tests 15 passed; separate v2/v3 test
1 passed. Original unscaled random stress error and changed fixture are retained
in `native_q8_implementation/initial_stress_failure.txt` at the source commit.
