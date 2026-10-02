# Conditional ordinary warm delivery

Source preparation only. No allocation, model/checkpoint tensor load, data pass, forecast fit, native export, parity check or runtime execution has occurred here. Root must first decide from the W screen and its prospective fresh-seed confirmation. Native OOF quality is a separately owned step; it is not implemented or scored here.

`commands.json` registers exactly two final fits. The parent starts strictly from `cont_sh256_final_s3/model_ep3.pt` (SHA `28f539eca86b5885f4250f1c3721f1c41b217aa40a66eeceae396ee714c91a89`), performs one additional epoch at lr0.0005, and saves fixed EMA ep0. H2 starts from that full own-parent state, trains two epochs and saves ep1. Relative to the original final recipe, all other103 arguments remain unchanged: full train+valid parent, all-valid H2, fold−1, original final teachers and weights. `train_warm.py` is byte-identical to W, retaining the discarded Linear256→32 then GRU33→64 constructor draws for matching RNG. Mean/std are copied exactly and never refit. H2 freezes inp/gru and normalization; no added inference parameters survive.

The full teacher caches `pred_fin256_{s3,s4}_{train,valid}.npy` previously passed complete hash/shape/finite checks in the bound original final input receipt. Their exact producer epochs are unknown. This is the same explicit accepted legacy limitation as the incumbent, not proof that they came from ep3. Root must bind their current identity, original final parent, propensity inputs and dataset identities before training. `legacy-bindings.json` preserves the19-source delivery inventory; `commands.json` separately binds the training dependencies.

Root training CLI after allocation:

```sh
.venv/bin/python research/iterations/2026-10-02-ordinary-warm-delivery/run_queue.py --allocation research/iterations/2026-10-02-ordinary-warm-delivery/training-allocation.json
```

Use the existing queue schema: `execution_authorized:true`, `sources` including commands.json plus its source map, current `input_bindings`, the two commands jobs, `execution_directory` as this directory's `execution-01`, and success status `COMPLETED_2_ORDINARY_WARM_FINAL_FITS_NO_METRICS`. Jobs already carry exact argv, required outputs and postcheck CLI. Parent cap1800s, H2 cap900s are proposed maxima; root assigns actual GPU/CPU/deadline. The queue accepts owned-GPU cap≤7168MiB, allocator≤.375/6144MiB, host≤20GiB, CPU≤600%; fresh UUID scopes, no retries. No final OOF or metric job belongs in this allocation.

After successful fits/checkpoint checks and separate native-quality decision, the packager exports only the final warm member:

```sh
.venv/bin/python research/iterations/2026-10-02-ordinary-warm-delivery/package_student.py \
  --base-zip research/iterations/2026-10-01-continuation/student-runtime/package/student_fixed4.zip \
  --base-manifest research/iterations/2026-10-01-continuation/student-runtime/package/student_fixed4.manifest.json \
  --checkpoint artifacts/runs/oct02_ow_final_s3h2/model_ep1.pt \
  --checkpoint-receipt research/iterations/2026-10-02-ordinary-warm-delivery/execution-01/oct02_ow_final_s3h2.check.json \
  --completion research/iterations/2026-10-02-ordinary-warm-delivery/execution-01/result.json \
  --output research/iterations/2026-10-02-ordinary-warm-delivery/package/warm_student_fixed4.zip
```

This checks the completed queue and its hashed worker outputs, strict103args/fixed endpoint and H2 receipt. It retains the original qualified wrapper, member list, all three ISA binaries, and weights1..3 byte-for-byte. Only weights0.npz changes. Export preserves float32 normalization and truncates only the last head output to the first two forecast channels. Existing quarter weights and member order remain unchanged. It never selects a composition or rebuilds native code. The ZIP+manifest remain unqualified until fresh checks.

Within a root-owned1CPU/4GiB/swap0 scope (CUDA hidden), fresh CP311 package self-consistency:

```sh
/home/grinchenkoeg/scorer_image/run.sh research/iterations/2026-10-02-ordinary-warm-delivery/runtime/check_package.py \
  --package research/iterations/2026-10-02-ordinary-warm-delivery/package/warm_student_fixed4.zip --cpu CPU
```

`CPU` is an explicit newly allocated physical CPU. The check verifies native repeat/reset/prefix/stream/wrapper masks and ordered-member FMA on256 raw-only rows. It does not establish GPU/native quality parity; that separate twofold check must use current student-native caches as reference, not the old Q4/d256 comparison hardcoded in the historical native-quality scripts.

After package-check PASS, create an unallocated runtime plan (small metadata only):

```sh
.venv/bin/python research/iterations/2026-10-02-ordinary-warm-delivery/runtime/prepare_runtime.py \
  --package research/iterations/2026-10-02-ordinary-warm-delivery/package/warm_student_fixed4.zip --cpu CPU --smt SIBLING --hard-seconds 3900
```

This creates no authorization. Root creates a separate runtime allocation with `warm_runtime_audit_authorized:true`, `attempts:1`, and exact `plan_sha256`. Then:

```sh
.venv/bin/python research/iterations/2026-10-02-ordinary-warm-delivery/runtime/start_runtime.py \
  --allocation ROOT_RUNTIME_ALLOCATION --allocation-sha256 SHA --execute
```

Without `--execute` it prints the exact command. The new UUID scope enforces1CPU/16GiB/swap0, no Torch/GPU, new paths, no old scope touching. Workload is exactly39.4M real `PredictionModel.predict` callbacks: all1873 valid sequences plus first97 repeated with fresh IDs, warmup99,39,204,970 emitted forecasts. Full startup/extraction/input hashing/IO/checksum work is retained in the actual≤60min test; hardcap defaults65min, source permits at most70min if prospectively allocated. All OS thread affinities, CPU times and sibling occupancy are reported; no cost subtraction/retry. Same architecture suggests similar cost to incumbent44.54min but does not grant the new ZIP a runtime PASS. Naturalv3/VNNI only; no other-hardware runtime claim.

All copied historical sources are immutable. Root should freeze these new sources before their respective stages; new `runtime-plan.json` and root allocations are produced only after the conditional decisions. No public upload or GitHub action is included.
