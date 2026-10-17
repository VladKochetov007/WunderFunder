# Native geometry callback serving check — 2026-10-18

Status: implementation and synthetic engineering parity/profile checks passed. No model-quality result or complete competition runtime is claimed.

## Implementation

Source is in commits `bdf996f9d`, `ce5f8f63c`, and correction `03e6db635977d9514f46ec990738112075c588de`. `submission/solution_geometry.py` exposes the competition `PredictionModel` callback and uses native `wnn_native.GeometryPredictor` for raw, event, geometry, reset, recurrent-state, and masked-row handling. The required native API is present in the library below. `scripts/geometry_serving_reference.py` and `GeometryFeatureStepper` are checker/reference support, not production callback fallbacks.

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

All profile and parity fixtures here are synthetic engineering checks, not scientific comparisons, validation scores, or full-package acceptance. Root must run actual-checkpoint parity and the competition package gate.

## Actual F0 package check

[`scripts/score_geometry_served_f0.py`](../../../../scripts/score_geometry_served_f0.py) replays an exact ZIP against the F0 split's declared evaluation row groups only. It validates each row-group/sequence identity, uses the official scored-row mask and `global_wp_clip2_v1`, writes a row-group-indexed prediction array and sidecar, and records callback-only plus end-to-end timing. An optional FP32 cache is accepted only with a matching split-hash and row-group sidecar. Run one fixed precision per fresh process, with the package ZIP, build receipt, root `datasets/valid.parquet`, and fold-0 split pinned. The script does not read the full validation table or require Torch.

The feature adapter callback now exposes the native stepper at `.model`, so existing reset-aware ZIP validators can invoke `model.model.reset()` between sequence groups without rebuilding the extension.
