# Native geometry callback serving check — 2026-10-18

Status: implementation, synthetic parity checks, and actual exposed-F0 package replay are complete. The candidate did not improve its matched warm parent; complete competition runtime is unmeasured.

## Implementation

Source is in commits `bdf996f9d`, `ce5f8f63c`, `03e6db635977d9514f46ec990738112075c588de`, `52e7ed6bb`, `60e0b1e02`, `001e01307`, and `2c47d6083`. `submission/solution_geometry.py` exposes the competition `PredictionModel` callback and uses native `wnn_native.GeometryPredictor` for raw, event, geometry, reset, recurrent-state, and masked-row handling. The required native API is present in the library below. `scripts/geometry_serving_reference.py` and `GeometryFeatureStepper` are checker/reference support, not production callback fallbacks.

The root packager can stage this endpoint with:

```sh
uv run python scripts/package_submission.py CANDIDATE RUN:COEF \
  --solution submission/solution_geometry.py \
  --native-from /home/vlad/development/wunder/research_tmp/geometry-serving-20261018/native-built-pass2 \
  --extra-file /path/to/geometry_prior.json \
  --precision q16x
```

The prior must be exported from the fitted child's `research_feature_prior` as root-level `geometry_prior.json` with schema `causal_geometry_discrete_v1`. The packager creates `solution.py`, `members.json`, `precision.txt`, and `weights_0.npz`; the required auxiliary file is `geometry_prior.json`. It copies native libraries under `v2/`, `avx2/`, and `v3/`. The served R2 child has exactly 271 inputs (112 raw + 99 causal event + 60 causal geometry) and nonzero new geometry weights; its frozen-parent proof is in the linked campaign packet.

## ABI and runtime receipts

- The generic runtime-dispatch library is at `/home/vlad/development/wunder/research_tmp/geometry-serving-20261018/native-built-pass2/{v2,avx2,v3}/wnn_native.abi3.so`, SHA-256 `ebba8fe295175bf1c2ab57dbe76dd764c5c893941020af774be95cfe8a83043d`. All three copies are identical. Static ABI scan passed with max GLIBC 2.34 and no libpython dependency; an offline Python 3.11 abi3 import passed. See [ABI scan](abi_receipt.json).
- Streaming events and geometry are bitwise equal to the canonical Python references on the bounded synthetic 20,000-row fixture. FP32 predictions, sequence reset, and masked-row updates also match bitwise. Fixed-precision q8vnni, i16, and q16x paths passed the same synthetic callback parity. See [Q8](profile_q8vnni.json), [i16](profile_i16.json), and [q16x](profile_q16x.json).
- The 30%-core row costs and the synthetic FP32 proxy below are engineering profiles, not package-level measurements. The actual trained checkpoint and selected q16x ZIP were subsequently replayed on all 468 exposed F0 groups; results and measured callback-only projection are recorded below. Full competition runtime remains unmeasured.
- The first native parity run failed before correction: row 0, geometry channel 2 emitted `0.0` rather than reference `-1.0` because the category field-stride was incorrectly applied to scalar stat blocks. This is preserved in [first failure](first_parity_failure.json); corrected parity passed after the channel-order fix and f64 dominant-event division.

All profile and parity fixtures above are synthetic engineering checks, not scientific comparisons or validation scores.

## Actual F0 package check

[`scripts/score_geometry_served_f0.py`](../../../../scripts/score_geometry_served_f0.py) replays an exact ZIP against the F0 split's declared evaluation row groups only. It validates each row-group/sequence identity, uses the official scored-row mask and `global_wp_clip2_v1`, writes a row-group-indexed prediction array and sidecar, and records callback-only plus end-to-end timing. An optional FP32 cache is accepted only with a matching split-hash and row-group sidecar. Run one fixed precision per fresh process, with the package ZIP, build receipt, root `datasets/valid.parquet`, and fold-0 split pinned. The script does not read the full validation table or require Torch.

[`scripts/export_geometry_prior.py`](../../../../scripts/export_geometry_prior.py) copies and validates `research_feature_prior` from the selected checkpoint without refitting it; it writes an output hash receipt for the package's `geometry_prior.json` auxiliary file.

[`scripts/create_geometry_alpha_aliases.py`](../../../../scripts/create_geometry_alpha_aliases.py) is a source-only helper for the bounded R2 geometry contribution screen. After fit workers drain, it writes `predict_run.py`-compatible inference aliases under `artifacts/runs/`, scaling only `inp.weight[:, 211:271]`; the default scales are `−0.5`, `−0.25`, `+0.25`, and `+0.5`. Each run gets a `provenance.json` sidecar that labels the alias inference-only and binds its alpha to the source checkpoint hash, historical fit/initializer, and source provenance hash. The required `--expected-source-sha256` pins alias creation to the checkpoint approved in the plan. The manifest hashes each alias and sidecar. Alpha zero is opt-in because the parent-function control cache already exists. No aliases or scores are produced by preparing the helper.

[`scripts/check_geometry_adapter_weights.py`](../../../../scripts/check_geometry_adapter_weights.py) is a post-fit, local-checkpoint proof helper. `--mode extension --variant moments_only|dictionary_only` checks the inherited 211-input model weights bitwise and the selected 20-channel mask for exact zeros; `--mode scale-alias --alpha VALUE` checks that an inference alias changes only geometry input columns by the selected scale. It reads only the two explicitly named checkpoint files, applies a 64 MiB bound to each, and writes a create-only hash-bound receipt. It is not run during fitting.

The feature adapter callback now exposes the native stepper at `.model`, so existing reset-aware ZIP validators can invoke `model.model.reset()` between sequence groups without rebuilding the extension.

### Actual trained checkpoint and package results

The final exploratory package is [`geometry_adapter_r2_q16x_20261018.zip`](../../../../submissions/geometry_adapter_r2_q16x_20261018.zip), 5,323,200 bytes, SHA-256 `955bff715871ff8a977d96a43f4edb89b1f5ecec72ae8ee49c0c2b53a3b6688b`. The alias is byte-identical to the exact ZIP used by the offline Python 3.11 smoke and full F0 replay. The package uses the R2 trained checkpoint with nonzero geometry weights and fixed q16x precision.

- Full exposed F0: q16x scored `0.6804519697`, versus the same checkpoint's FP32 producer cache at `0.6804499040` (difference `+0.0000020656`). q8vnni scored `0.6803847537` (difference `−0.0000651504`). These are comparisons on exposed F0 data, not a gain claim; the paired R2 candidate comparison against its warm parent was negative. See the [q16x full-F0 receipt](../../../campaign_2026-10-18/geometry_adapter_probe_r2/geometry_adapter_r2_q16x_f0_serving.json), [q8vnni full-F0 receipt](../../../campaign_2026-10-18/geometry_adapter_probe_r2/geometry_adapter_r2_q8vnni_f0_serving.json), and [parent comparison](../../../campaign_2026-10-18/geometry_adapter_probe_r2/comparison.json).
- On the first eight source-qualified F0 groups, the selected q16x ZIP exactly matched its q16x native full-F0 cache at all 318,416 prediction values by `uint32` view (including signed-zero distinctions). This checks exact-ZIP callback/cache consistency, not Torch equivalence. The Python 3.11 exact-ZIP smoke separately passed offline ABI/import, finite outputs, warmup suppression, and synthetic A-B-A reset checks. See the [uint32 parity receipt](../../../campaign_2026-10-18/geometry_adapter_probe_r2/geometry_adapter_r2_q16x_native_cache_uint32_parity.json), [label correction](../../../campaign_2026-10-18/geometry_adapter_probe_r2/geometry_adapter_r2_q16x_native_cache_parity_label_correction.json), and [Python 3.11 smoke receipt](../../../campaign_2026-10-18/geometry_adapter_probe_r2/geometry_adapter_r2_q16x_python311_smoke.json).
- The preregistered FP32 native-versus-Torch panel did not meet its `rtol=1e-5`, `atol=1e-6` tolerance: 287,989 of 318,416 prediction values were outside the threshold; max absolute difference was `7.533133e-4`, mean `5.551173e-5`, RMS `7.635945e-5`. This is an implementation-level comparison only. The cause was not isolated, and no Torch/native equivalence claim is made. See the [predeclared protocol](../../../campaign_2026-10-18/geometry_adapter_probe_r2/geometry_adapter_r2_f32_native_torch_panel_protocol.json), [panel result](../../../campaign_2026-10-18/geometry_adapter_probe_r2/geometry_adapter_r2_f32_native_torch_panel.json), and [tolerance assessment](../../../campaign_2026-10-18/geometry_adapter_probe_r2/geometry_adapter_r2_f32_native_torch_tolerance_assessment.json).
- At 100% of one CPU core, actual q16x callback cost was `61.5748 µs/row`; multiplying by 39.4 million rows gives a `40.43 min` callback-only row-cost projection. q8vnni measured `27.0093 µs/row` (`17.74 min` proxy). The full F0 replay elapsed `633.02 s` for q16x and included only the 9.36 million declared F0 rows. These measured row-cost projections are not complete competition runtime measurements; they exclude unmeasured runtime overhead and do not establish competition acceptance by themselves. See the [q16x serving receipt](../../../campaign-2026-10-18/geometry_adapter_probe_r2/geometry_adapter_r2_q16x_f0_serving.json) and [q8vnni serving receipt](../../../campaign-2026-10-18/geometry_adapter_probe_r2/geometry_adapter_r2_q8vnni_f0_serving.json).
