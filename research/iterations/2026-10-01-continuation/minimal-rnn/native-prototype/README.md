# Generated FP32 native minGRU prototype

The isolated C++ implementation passed **80 generated-only numerical/state comparisons**, on the first compile and check. It ports the current `MinimalRNN.step` recurrence without a model-math change. This establishes narrow FP32 streaming feasibility; it does not qualify AMP, forecasting quality, training readiness, a submitted package, or the competition runtime limit.

## Implemented boundary

The model accepts **211 already engineered float32 features**. It applies the generated model's `(x−mean)/std`, input linear projection and exact-erf GELU, two input-gated positive-state minGRUs of width256, then the `[last_state,input_projection]` head with exact-erf GELU. Both full23-output and sliced2-output heads are checked. Outputs are raw model values; this library does not add competition clipping, blending, raw112→211 feature construction, or a submission callback/reset adapter.

The recurrence is `h = (1−sigmoid(k))*h + sigmoid(k)*g(v)`, where `g(v)=v+.5` for `v>=0` and sigmoid otherwise. Incoming nonnegative finite state is copied literally; zero reset does not pass through `g`. Gate and candidate projections depend on the current layer input. The second layer consumes the first layer's newly updated state. All stored weights, dot-product accumulation and state are FP32; no INT8 or FP16 kernels are called.

The C ABI exposes `min_create`, `min_set_state`, `min_get_state`, `min_run`, and `min_destroy`. `min_run(..., rows=1, ...)` is the one-step interface; its multirow form repeats the identical causal operator without Python overhead. Handles own separate state and scratch. Nonfinite model/input values, nonpositive normalization standard deviations, and negative/nonfinite initial states are rejected at the relevant boundary. Generated finite inputs and extreme logits were tested; this is not a general overflow-proof API for arbitrary finite magnitudes.

`kernels.hpp` and `activations.hpp` are byte-identical private copies of `research/iterations/2026-10-01-overnight/quant3/native/include/wnn/`. Only FP32 GEMV and exact libm activations are used. The correct relative source path from this directory is `../../../2026-10-01-overnight/quant3/native/include/wnn/`; the plan's provenance location omitted one parent traversal, but its source hashes bind the correct copied files. The actual Torch source is copied as `torch_reference.py`, SHA256 `7ce33c53b53c2be91a7726a9d36e3bba027dbabf81b3f1ea01b51b030dd2e65f`. Closed/active native engines were never modified.

## Fixed generated checks and results

Seed2043 creates a random `211→256×2→23` model, nontrivial normalization statistics, a length2000 input, zero/positive state, its exact two-output export, and a generated extreme-logit model. These arrays and a hashed inventory are saved before comparisons. No trained checkpoint, dataset, target, OOF cache or forecast score is read.

- Ordinary outputs and final/prefix states match actual Torch FP32 `step` and blocked `forward`, maximum absolute discrepancy **3.5763e−7**, against preregistered `atol=2e−5, rtol=2e−5`.
- Gate/candidate logits include −80, −20, 0, 20 and80. Full-model and direct-cell checks include the literal FP32 recurrence, blocked scan and FP64 oracle. Maximum absolute discrepancy **1.5259e−5**, against `atol=2e−4, rtol=2e−5`.
- Native irregular splits, reset/replay, changed-future prefix and two-output export agree **bit for bit**. Actual initial-state copying and rejected-state/input nonmutation also pass.

Authoritative detailed results: `generated-01/result.json`; immutable generated inputs/configuration: `generated-01/generated-*.npz`; pre-comparison identity: `generated-01/inputs-before-check.json`. The run recipe is `plan.json`. There were no failed attempts or model-math retries.

## Descriptive single-CPU cost

On beleriand CPU34, with one Torch/BLAS thread, five fixed warm-cache repetitions of the generated two-output model gave:

| Boundary | Median per2,000 rows | Median per row |
|---|---:|---:|
| C++ internal streaming loop |0.16031s|80.15µs|
| Python ctypes call for every row |0.18814s|94.07µs|

The respective observed ranges were0.15975–0.16136s and0.18743–0.19058s. These include input-finiteness checking and the Python wrapper's allocation where applicable. Torch's generated two-output step loop took0.76057/0.77734s for zero/positive state, but that is a different invocation boundary, not a serving benchmark against the production engine.

The native port uses host-built x86-64-v3 instructions and a C ABI loaded through ctypes. **Raw112→211 features, submission ABI/loader, competition callback/reset handling, other ensemble members and end-to-end scoring are absent.** No scorer-image compatibility or deadline extrapolation is claimed. A different generated weight distribution, CPU, compiler or activation implementation could change cost.

Compilation took1.59s; the full check process13.78s (internal11.996s), with peak RSS566,972KiB, about554MiB. Stages were bounded by CPUQuota200%, MemoryMax4GiB, swap0 and300s each. Compile used CPUs34/36; checks used CPU34 only; sibling preflight was idle. `CUDA_VISIBLE_DEVICES=''` throughout. Both systemd scopes report inactive/dead. Total retained artifacts are about8MB, copied locally and under the same isolated SSH path. No GPU or forecasting allocation follows from this result.

The controller uses exclusive output creation. A successful receipt is not silently overwritten by rerunning `run_cpu.py`.
