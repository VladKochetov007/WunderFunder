# E489 source and fit-only preflight receipt

This packet implements the registered E489 H128 four-epoch candidate as a new
study. It does not modify E485, E487, shared training code, or any grader. The
E487 saved-delta admission check binds the v4 verifier and review receipt, the
root proof SHA, strict FP32 pass, finite delta arrays, source identities, and
the verified teardown/unchanged OOM counters. BF16 failure remains expected.

The callable CUDA probe checks native alpha-zero output, state, input and
parameter gradients, and clipped optimizer updates at all three seeds. It also
times two continuous B4×256 chunks for all six seed/alpha cells. Timers include
the input copies, and the report records body wall time separately from the
provider-charged time, which must be used for the 7200-second admission gate.
The probe records actual initialization hashes and active IEEE FP32 precision
metadata. No CUDA probe or full trajectory has been run for E489.

The CPU-only synthetic suite passed 4 tests under one CPU/2 GiB. The real-data
fit-only preflight passed all six seed/alpha cells, each with one changed,
finite B4×256 update, under one CPU/2 GiB; it loaded only the 24 fit-role
groups and records `test_rows_loaded: false`. Its manifest SHA-256 is
`3bcd0909c800f9c6f133ad55a23b9490cf85ea64ed8b4deec8151e73702bed6a`.

Frozen source identities for independent review:

- `fit_fp32_duration.py`: `af2ad34d8bac617fe5f53e28a2120c5d59a303888b54937e35e27edd1b925cdc`
- `gpu_probe_fp32_duration.py`: `2f0bd08aa0e760956a233ec144cb41c82613cc5ce98aec1cc5ed7b4400effd6f`
- `tests_fp32_duration.py`: `fa5cd3da2afd9860b0a34d2a929ae1526747650d1f9dcd71cfbb34e553d66ec6`
- E487 root saved-delta gate: `4fced259c38beae9a74652cf486ccd952d62c830e5927e5316be161aba6c6cac`
- E487 verifier v4: `f717d0d30d1c6e1456b80bdd31d76896633a588ea22e5b9243652af5f92391e6`
- E487 verifier review: `0685ce65335c956a94003700827798c8d3080d746bb876362074cc19545f9132`
- E487 verification receipt: `8820f1eb3edf954f1d2279843be85d418a8883fad39fff3f9a9689396eeae6a9`

Validation commands:

```bash
WNN_CPU_PCT=4 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_THREADS=1 \
  .venv/bin/python -m unittest discover \
  -s research/campaign_2026-10-05/fp32_activation_duration \
  -p 'tests_fp32_duration.py' -v
WNN_CPU_PCT=4 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_THREADS=1 \
  WNN_RUNTIME_MAX=300 .venv/bin/python \
  research/campaign_2026-10-05/fp32_activation_duration/fit_fp32_duration.py \
  --preflight-real --device cpu \
  --output research/campaign_2026-10-05/fp32_activation_duration/cpu_preflight_v1
```
