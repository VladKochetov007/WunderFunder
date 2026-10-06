# Independent source review: E489 clipping diagnostic

Disposition: **conditional GO for one non-scoring fit-only diagnostic**, after
the root seals an exact attempt with the current source and input identities,
one CPU, a 3,000,000,000-byte host cap, a 2-GiB PyTorch allocator cap, a
3,000,000,000-byte observed-device stop, and a 180-second timeout. The
diagnostic does not pass the frozen E489 gate or authorize E489 full fits.
No actual-data or CUDA execution was performed for this review.

The reviewed source identities are:

| File | SHA256 |
|---|---|
| `DESIGN.md` | `be338571e8031818ef85647a789fa8bdef28ec6e47fbbfcf5647e4804c608277` |
| `diagnose.py` | `4315ce41714550f2cc1f2500c767f2b3ef6797a78b3dd21d21e5804936e14249` |
| `diagnostic_math.py` | `ef7dc90425aafa20433278e03e8d1dc8156af00daf91208c1b37b0568fbeecb5` |
| `frozen_runtime.py` | `a172b96f7781ee458ae56ab67a400c04b246521996bb309884c8ec870cd7860d` |
| `seed_probe.py` | `3d63c7cb089f10397b7069a179e8779cebbe7fa9adab9970261284a76dfa1146` |
| `test_diagnostic_math.py` | `95cefbaa76fb665fa9c7a99b1622889e85f9fc5bb81d172d6dd14bf2fa08ba29` |
| Frozen E489 model `fit_fp32_duration.py` | `af2ad34d8bac617fe5f53e28a2120c5d59a303888b54937e35e27edd1b925cdc` |
| Frozen E489 probe `gpu_probe_fp32_duration_v2.py` | `7ee1bf79dac158d7aaa6530f55173126ad5e8f8df1c4e50832d83e9b0ca19b20` |

The first-batch selection matches the frozen probe: `include_test=False` must
return exactly 24 `fit` groups; each seed uses the frozen epoch-zero order's
first four IDs and the first 256 rows. Parent validation checks the frozen
model, optimizer, proof, cohort and dataset dependencies before CUDA setup.
The script records batch identities, initial states, parent source identities,
the saved E487 proof, data/cohort hashes, and per-seed raw-gradient archives.

The diagnostic is decomposed into four independently hash-bound modules;
the root attempt must include all four, the unit tests, and the design.
The diagnostic compares native and alpha-zero custom outputs, recurrent
states, input gradients and every parameter gradient. It saves raw gradients,
then performs four one-step branches from the same initial bytes with fresh
optimizer states: native/custom and raw/clip-1. These branches use the frozen
FP32 E487 optimizer and compare each of the four recurrent Muon matrices plus
the remaining AdamW family. Clipping activity and the original v2 update
thresholds are descriptive fields; finite numerical or coverage misses are
preserved rather than treated as E489 admission. Nonfinite values, changed
provenance and memory-stop violations abort. Each completed seed's arrays and
JSON are written before processing the next seed. The report states that the
diagnostic grants no E489 pass or full-fit admission.

The five CPU arithmetic tests passed under one CPU and two GiB RAM in 0.86 s;
`py_compile` and Ruff passed on the four modules and unit test. The Python source checks device memory before
and after collection; continuous observed-device and timeout enforcement must
come from the separately sealed runner. Root must bind the exact diagnostic
source and all frozen inputs in the attempt record. The failed frozen E489 v2
probe remains unchanged and is not repaired by this diagnostic.
