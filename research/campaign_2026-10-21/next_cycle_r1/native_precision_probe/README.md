# Native H512 precision probe

This bounded diagnostic isolates native precision effects in the existing H512 CP5 member. It compares the same exported member under dynamic int8, fast float32, and non-fast float32 execution, then checks the original TL3 student and its two retained members against the frozen full-endpoint F0 caches. It reads features only during inference. The eight selected groups are fixed by a SHA-256 rank over `valid:<row_group>`, with one group per feature-only connected component; the selection seed is 271828. The plan is generated without labels or prediction reads and should be reviewed before the runner starts.

The five stored tracks are H512 dynamic int8, H512 fast float32, H512 exact float32, the original TL3 student, and the retained-member pair. The endpoint reconstructions use the frozen coefficients: `0.34 * original_student + retained_pair` and `0.34 * H512_dynamic + retained_pair`. The latter and former must reproduce their saved full-valid native predictions within maximum absolute error `1e-5`. The counterfactual is fixed as `saved_native_CP5 + 0.34 * (H512_fast_float32 - H512_dynamic_int8)`; it is diagnostic only and is not a coefficient search or promotion rule.

All five feature-only prediction arrays and their manifest are written before the frozen target cache is mapped. If construction parity fails, the run records pre-label evidence and stops without accessing targets. After parity passes, the script uses the canonical `campaign_score` moments and global `global_wp_clip2_v1` scorer on the eight selected components. It reports each fixed score, paired deltas, raw prediction RMS, raw clipping fraction, target-magnitude-weighted prediction and residual standard deviations, and native-versus-Torch prediction RMS differences. There is no bootstrap on this eight-component diagnostic.

Inputs are pinned to the original TL3 ZIP SHA-256 `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`, CP5 ZIP `ff5d5bdade2d024e0c2bf98237024cef4569664bad0fa3c99d4686e9621f54e2`, and `NATIVE_GRADE_R3.json` `872ae52f0c83611d390e778d9666f910773e8d24e6494c12cb56fd8384788e35`. Both packages must contain identical wrapper source and byte-identical v3 extensions; the runner requires `LEVEL == "v3"` and records the active quantization backend. Frozen native cache hashes are TL3 `3a7456f0a2e294a852618535b9e3f2551fc15c5b70fd033824bddb10be83bbe3` and CP5 `2a4d95a332f12f7787a75fbe0acc66811d938d614c203fee28dcc7735611cfe2`. The comparison Torch cache is `root/cache/pred_r01_h512_f0_s1003_local_cp5_r1_valid.npy`, SHA-256 `14cf538ee27dd44420f3b682dc449766bda1fe37bcc377f21ff81d10aa59f240`, checkpoint SHA-256 `4311fbb71d1ba121a285fefe06b431f9fa2c16f7fabd23f18b72135180bffe2f`. The native TL3 and CP5 caches each cover all 468 F0 evaluation sequences with shape `(468, 20000, 2)` and unique row-group sidecars. The Torch cache covers all 1873 validation sequences with shape `(1873, 20000, 2)`.

The first stage creates a fresh output directory on physical scratch and writes `selection_plan.json` only:

```bash
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
OUT=/home/vlad/development/wunder/research_tmp/native-h512-precision-probe-output-20261021
"$ROOT/.venv-py311-profile/bin/python" \
  "$ROOT/research/campaign_2026-10-21/next_cycle_r1/native_precision_probe/prepare_selection.py" \
  --repo-root "$ROOT" --output-dir "$OUT"
```

After reviewing the exact eight membership rows and `selected_membership_sha256`, run the bounded feature replay and grader:

```bash
"$ROOT/.venv-py311-profile/bin/python" \
  "$ROOT/research/campaign_2026-10-21/next_cycle_r1/native_precision_probe/run_precision_probe.py" \
  --repo-root "$ROOT" --selection-plan "$OUT/selection_plan.json" --output-dir "$OUT"
```

Expected files are `member_predictions.npz`, `outputs_manifest.json`, `prelabel_validation.json`, and `precision_probe_result.json`. The runner accepts only an output directory containing the reviewed plan and creates all later files exclusively. It is designed for the supplied Python 3.11 binary with NumPy and PyArrow; it uses no Torch, GPU, provider, or new fit. Source/format checks are the only builder verification. Runtime native parity and score execution remain pending root’s bounded run, and the result is an exposed F0 diagnostic rather than clean CV evidence.
