# Native H512 precision probe

This bounded diagnostic isolates native precision effects in the existing H512 CP5 member. It compares the same exported member under dynamic int8, fast float32, and non-fast float32 execution. Its first run attempted to reconstruct the original TL3 endpoint from separately constructed member groups; that original reconstruction failed the pre-label check, so no score was computed. It reads features only during inference. The eight selected groups are fixed by a SHA-256 rank over `valid:<row_group>`, with one group per feature-only connected component; the selection seed is 271828. The plan is generated without labels or prediction reads and should be reviewed before the runner starts.

The first replay saved five feature-only tracks: H512 dynamic int8, H512 fast float32, H512 exact float32, the original TL3 student at unit coefficient, and the retained-member pair. Its pre-label check passed the H512 reconstruction (`max abs 2.10e-7`) but failed the original TL3 reconstruction (`max abs 0.00209`): the first probe used a unit-coefficient student plus retained pair, while the pinned wrapper constructs one `Ensemble` over all three members with coefficients `[0.34, 0.33, 0.33]`. The subsequent exact-wrapper replay also failed at the same magnitude, so topology is not an established cause. Both failures are preserved. R2 below reuses the five tracks only if the fixed `0.34 * original_student + retained_pair` reconstruction passes a new fresh-host full-reference gate.

The corrected replay calls the exact pinned ZIP's `PredictionModel.predict` on the same eight feature-only sequences and writes `original_endpoint_exact_callback.npy` with its create-only pre-label receipt. It verifies the existing five-track NPZ hash and member hashes without modifying that NPZ. Both exact original endpoint and saved H512 CP5 checks against legacy full-valid caches fail at the original-endpoint gate, so these attempts remain unscored. The fixed counterfactual is still `saved_native_CP5 + 0.34 * (H512_fast_float32 - H512_dynamic_int8)`; this is diagnostic only, not a coefficient search or promotion rule.

The initial five feature-only arrays and manifest were saved before any target access. Both legacy-reference pre-label attempts failed and stopped before targets. The fresh-host adapter below tests the fixed reconstruction against a complete same-runtime endpoint cache before any score call. This eight-component diagnostic has no bootstrap and cannot support a clean-CV claim.

Inputs are pinned to the original TL3 ZIP SHA-256 `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`, CP5 ZIP `ff5d5bdade2d024e0c2bf98237024cef4569664bad0fa3c99d4686e9621f54e2`, and `NATIVE_GRADE_R3.json` `872ae52f0c83611d390e778d9666f910773e8d24e6494c12cb56fd8384788e35`. Both packages must contain identical wrapper source and byte-identical v3 extensions; the runner requires `LEVEL == "v3"` and records the active quantization backend. Frozen native cache hashes are TL3 `3a7456f0a2e294a852618535b9e3f2551fc15c5b70fd033824bddb10be83bbe3` and CP5 `2a4d95a332f12f7787a75fbe0acc66811d938d614c203fee28dcc7735611cfe2`. The comparison Torch cache is `root/cache/pred_r01_h512_f0_s1003_local_cp5_r1_valid.npy`, SHA-256 `14cf538ee27dd44420f3b682dc449766bda1fe37bcc377f21ff81d10aa59f240`, checkpoint SHA-256 `4311fbb71d1ba121a285fefe06b431f9fa2c16f7fabd23f18b72135180bffe2f`. The native TL3 and CP5 caches each cover all 468 F0 evaluation sequences with shape `(468, 20000, 2)` and unique row-group sidecars. The Torch cache covers all 1873 validation sequences with shape `(1873, 20000, 2)`.

The first stage creates a fresh output directory on physical scratch and writes `selection_plan.json` only:

```bash
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
OUT=/home/vlad/development/wunder/research_tmp/native-h512-precision-probe-output-20261021
"$ROOT/.venv-py311-profile/bin/python" \
  "$ROOT/research/campaign_2026-10-21/next_cycle_r1/native_precision_probe/prepare_selection.py" \
  --repo-root "$ROOT" --output-dir "$OUT"
```

After reviewing the exact eight membership rows and `selected_membership_sha256`, the first replay used this command and saved the five feature-only arrays. Its original endpoint check failed; do not rerun it into the existing output directory:

```bash
"$ROOT/.venv-py311-profile/bin/python" \
  "$ROOT/research/campaign_2026-10-21/next_cycle_r1/native_precision_probe/run_precision_probe.py" \
  --repo-root "$ROOT" --selection-plan "$OUT/selection_plan.json" --output-dir "$OUT"
```

## Source navigation and actual admission

- [Selection builder](prepare_selection.py) freezes eight distinct components without reading labels or predictions.
- [Runner](run_precision_probe.py) owns inference, output preservation, construction-parity gating and subsequent scoring.
- [Native helpers](probe_native.py) load the exact binary, materialize FP32 weight dictionaries and stream complete sequences with resets.
- [Preparation helpers](probe_prepare.py) bind archives/caches, preserve shared v3 library identity and construct the five fixed tracks.
- [Scoring helpers](probe_grade.py) reuse official clipped weighted moments and report scale diagnostics.
- [Frozen configuration](probe_config.py) names source archives, caches, hashes and approved disk scratch.
- [Root admission](../NATIVE_PRECISION_ADMISSION_R1.json) records the exact selected membership and source hashes, one CPU/1.2 GB/no-swap/600-second bound, and zero new fits/GPU time. [Runtime log](../native_precision_r1.log) belongs to this one attempt. Inference completed and the prelabel gate failed; no quality result or promotion is implied.

[Execution receipt](../NATIVE_PRECISION_EXECUTION_R1.json), [prelabel validation](prelabel_validation.json), [output manifest](outputs_manifest.json) and [log](../native_precision_r1.log) preserve completed feature-only inference. The original TL3 reconstruction failed the unchanged1e-5 gate (max0.002090), while CP5 reconstruction passed(2.1e-7). No targets were accessed and no precision score was computed. Native FP32 H512 matches Torch to1.10e-6RMS; dynamic-int8 differs by0.002655RMS on need-prediction rows. Root assigned source-only diagnosis; saved outputs remain immutable in physical scratch.
The initial runner created `member_predictions.npz`, `outputs_manifest.json`, and `prelabel_validation.json`, then stopped at the original-endpoint parity gate before target access. Do not rerun it into that existing output directory. After reviewing the corrected exact-wrapper replay source, use:

```bash
"$ROOT/.venv-py311-profile/bin/python" \
  "$ROOT/research/campaign_2026-10-21/next_cycle_r1/native_precision_probe/rerun_original_endpoint.py" \
  --repo-root "$ROOT" --selection-plan "$OUT/selection_plan.json" --output-dir "$OUT"
```

This create-only rerun adds `original_endpoint_exact_callback.npy` and `original_endpoint_prelabel_validation.json`; it leaves the original five-track NPZ, manifest, selection plan, and failed first-attempt pre-label receipt untouched. Root's full current-host replay and same-runtime endpoint grade have now completed all 468 sequences with reset ABA passing. The host replay matched the saved exact-callback sample before labels; its endpoint scores remain exposed F0 diagnostics rather than clean CV evidence.

- [Exact-wrapper correction](rerun_original_endpoint.py) replays the immutable original callback and preserves separate construction-parity evidence before labels.

The [exact-wrapper R2 receipt](original_endpoint_prelabel_validation.json) preserves the repeated0.002089739max mismatch with targets unopened. The old cache producer ran in Docker; host/container numerical runtime closure remains unresolved. Those legacy-cache precision attempts yielded no precision score; the separate fresh-host precision gate below remains pending.

The [version-matched container R3](container_original_endpoint_prelabel_validation.json) reproduces the original cache exactly(max0.0). Its [admission](../NATIVE_PRECISION_CONTAINER_ADMISSION_R3.json) pins the cached image; the historical imageID is not recorded. This demonstrates an environment effect without identifying one dependency. [A separate complete host-reference replay](../NATIVE_HOST_REFERENCE_AMENDMENT_R1.json) now closes the actual comparison on the host; no old cache or tolerance is changed.
- [Host-reference grader](grade_host_reference.py) checked the admitted 468-sequence current-host replay against the saved exact-callback sample before labels, then graded four fixed F0 endpoints and paired contrasts. The [completed host grade](../NATIVE_HOST_GRADE_R1.json) preserves full coverage, reset ABA, source pins and exposed-F0 limits; it does not close the separate eight-group precision question.

## Current-host precision R2 adapter

[The R2 adapter](grade_host_precision_r2.py) reuses the immutable five-track archive and the fixed eight-group selection. It verifies the old R1/R2 failures remain unchanged, checks the new host grade/cache/sidecars/archive/source binding, reads only `seq_ix`, `step_in_seq`, `need_prediction` and the 112 input features for those eight groups, and verifies 99 warmup plus19,901 prediction rows per sequence. Before touching the target cache it requires both `0.34 * original_student + retained_pair` versus the current full-host TL3 cache and the unchanged CP5 reconstruction versus the full CP5 cache to pass the original `1e-5` maximum-error gate. A failure records only a pre-label receipt and exits. On a pass it calls the unchanged scorer for the five saved tracks and the preregistered `saved CP5 + 0.34 * (fast FP32 - dynamic int8)` counterfactual.

The exact root-owned command is:

```bash
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
OUT2=/home/vlad/development/wunder/research_tmp/native-h512-precision-host-reference-r2-20261022
"$ROOT/.venv-py311-profile/bin/python" \
  "$ROOT/research/campaign_2026-10-21/next_cycle_r1/native_precision_probe/grade_host_precision_r2.py" \
  --root "$ROOT" --output-dir "$OUT2"
```

Root review and its admitted one-CPU/1.4-GB run are pending. The adapter was not executed during preparation; it produces no new predictions and uses no GPU or fit. Any resulting score covers eight already-exposed F0 groups from distinct components and is a fitted diagnostic only, with no CI or promotion claim.

The [fresh-host precision grade now passes](../NATIVE_HOST_PRECISION_PROBE_R2.json). Reconstruction errors are1.55e-7(original) and2.10e-7(CP5). FrozenFP32counterfactual endpoint+0.00004035 leaves−0.00328280versusTL3 on10,988scoredrows/eightcomponents. [Execution](../NATIVE_HOST_PRECISION_EXECUTION_R2.json) used5.39seconds/382380KiB/zeroGPU/fits. The failedR2importlauncher ispreserved; R3onlyadds repositoryroottoPYTHONPATH. No extrapolation, new coefficient orreleasecandidate.
