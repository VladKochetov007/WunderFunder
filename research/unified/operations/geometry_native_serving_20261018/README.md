# Native geometry callback serving check — 2026-10-18

Status: implementation, synthetic engineering checks and a separate exploratory trained q16x endpoint check passed their stated scopes. The exact adapter training recipe is negative versus its parent; the FP32 native/Torch tolerance failed. Complete target-host competition runtime is unmeasured.

## Implementation

Source is in commits `bdf996f9d`, `ce5f8f63c`, correction `03e6db635977d9514f46ec990738112075c588de`, indexed evidence `04252f6a35bc99711edf57397b775b66f6f11fe3`, and callback reset-compatibility fix `52e7ed6bb6f5861846bb11eac29ceeaf3669dc0b`. `submission/solution_geometry.py` exposes the competition `PredictionModel` callback and uses native `wnn_native.GeometryPredictor` for raw, event, geometry, reset, recurrent-state, and masked-row handling. The callback exposes the native predictor as `.model` so standard validators can reset it between source-qualified sequences. The required native API is present in the library below. `scripts/geometry_serving_reference.py` and `GeometryFeatureStepper` are checker/reference support, not production callback fallbacks.

The root packager can stage this endpoint with:

```sh
uv run python scripts/package_submission.py CANDIDATE RUN:COEF \
  --solution submission/solution_geometry.py \
  --native-from /home/vlad/development/wunder/research_tmp/geometry-serving-20261018/native-built-pass2 \
  --extra-file /path/to/geometry_prior.json \
  --precision q8vnni
```

The prior must be exported from the fitted child's `research_feature_prior` as root-level `geometry_prior.json` with schema `causal_geometry_discrete_v1`. The packager creates `solution.py`, `members.json`, `precision.txt`, and `weights_0.npz`; the required auxiliary file is `geometry_prior.json`. It copies native libraries under `v2/`, `avx2/`, and `v3/`. The model member must have exactly 271 inputs (112 raw + 99 causal event + 60 causal geometry). Root owns selecting and validating actual fitted weights and coefficient.

## ABI and runtime receipts

- The generic runtime-dispatch library is at `/home/vlad/development/wunder/research_tmp/geometry-serving-20261018/native-built-pass2/{v2,avx2,v3}/wnn_native.abi3.so`, SHA-256 `ebba8fe295175bf1c2ab57dbe76dd764c5c893941020af774be95cfe8a83043d`. All three copies are identical. Static ABI scan passed with max GLIBC 2.34 and no libpython dependency; an offline Python 3.11 abi3 import passed. See [ABI scan](abi_receipt.json).
- Streaming events and geometry are bitwise equal to the canonical Python references on the bounded synthetic 20,000-row fixture. FP32 predictions, sequence reset, and masked-row updates also match bitwise. Fixed-precision q8vnni, i16, and q16x paths passed the same synthetic callback parity. See [Q8](profile_q8vnni.json), [i16](profile_i16.json), and [q16x](profile_q16x.json).
- At 30% of one CPU core, measured callback row costs were 85.6 µs for q8vnni, 249.6 µs for i16, and 165.5 µs for q16x. Multiplying by 39.4 million rows and scaling to one full core gives row-cost proxies of 16.9, 49.2, and 32.6 minutes, respectively. These are synthetic proxies; full competition runtime is not measured. FP32's earlier proxy was 91.5 minutes, so fixed q8vnni is the only tested option under a 20-minute row-cost window, pending root's actual-weight quality/runtime check.
- The first native parity run failed before correction: row 0, geometry channel 2 emitted `0.0` rather than reference `-1.0` because the category field-stride was incorrectly applied to scalar stat blocks. This is preserved in [first failure](first_parity_failure.json); corrected parity passed after the channel-order fix and f64 dominant-event division.

All initial profile and parity fixtures above are synthetic engineering checks, not scientific comparisons or quality claims. Root's separate exploratory q16x endpoint is fully F0-scored, but the FP32 native/Torch tolerance failed and complete competition runtime is unmeasured.

## Root R2 selected q16x endpoint

The root-owned [R2 packet bundle](../../../experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_native_q16x/README.md) records the selected `geometry_adapter_r2_q16x_20261018.zip` identity (5,323,200 bytes; SHA-256 `955bff715871ff8a977d96a43f4edb89b1f5ecec72ae8ee49c0c2b53a3b6688b`) without copying the archive, predictions, weights or data. Its exact native q16x callback scored 0.6804519697 over all 468 exposed F0 groups, versus Torch FP32 at 0.6804499040. The adapter fit itself scored 0.6804499040 versus the fixed parent at 0.6812265132, so this is not a training gain. The offline exact-ZIP Python 3.11 smoke passed. The corrected uint32 cache comparison found 0 of 318,416 different output bit patterns, including signed zero, on the first eight source-qualified groups.

The separately predeclared FP32 native/Torch panel failed its `rtol=1e-5, atol=1e-6` threshold: 287,989 of 318,416 compared values were outside tolerance, with maximum absolute difference 0.0007533133. The native path uses `Activations::Fast`; the numerical cause is not isolated. The q16 endpoint was scored directly and no Torch-equivalence claim is made. Local callback row-cost projects to 40.43 minutes over 39.4M rows; full-replay-cost projection is 44.41 minutes. Neither is complete competition runtime or a target-host guarantee. The archive was not uploaded.
