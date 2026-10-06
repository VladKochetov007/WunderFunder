# Prospective generated native cost check

Source only; no compilation, benchmark, real data, trained checkpoint or quality readout has been run for this probe. Root must separately allocate one CPU/quiet sibling and an owned scope with a60s total deadline (plus termination grace), no CUDA, swap0 and a suggested4GiB RAM cap. The source is an early cost falsifier, not a submission package.

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python research/iterations/2026-10-02-interval-memory/native-probe/prepare.py \
  --output-dir research/iterations/2026-10-02-interval-memory/native-probe/generated-01 --cpu <allocated-CPU>
```

`prepare.py` requires affinity exactly that CPU. It compiles one private shared library, then runs the single fixed check/benchmark. The remaining60s budget covers both stages. Root's external whole-scope timeout is authoritative, including cleanup of compiler subprocesses. There is no retry, alternate kernel, precision choice or outcome-driven workload selection.

The private headers come from the already tested `2026-10-02-mingru-readiness/native-v2/include/wnn`, whose ordinary predictor is the qualified current dynamic INT8/VNNI path. `source-origins.json` records the source bytes. Only the private `model.hpp` adds an interval hook/reset/member; the ordinary dense kernel body remains unchanged. The new `interval_memory.hpp` uses the same FP32 GEMV/activation implementations for projection32, event GRU64 and extra readout96→256. Added weights/state remainFP32, fixed before timing. The ordinary GRU/head keep their existing dynamicINT8 precision. FP32 parity mode disables trunk quantization and activation approximation; operational timing uses the existing fast activation family throughout. No change touches a candidate or common native source.

Generated checks precede timing: nonzero extra weights, exact-FP32 native versus Torch accumulated/control model, qualified baseline versus zero-extra output equality, current-event closure in every-row and long-quiet examples, arbitrary/empty chunks, reset and future-prefix causality. Timing uses nonzero extra weights; no zero-weight shortcut. The native streaming accumulator performs FP32 row additions, so trained Torch/AMP/native drift still requires later validation.

Then compare ordinary256, accumulated256 and instantaneous256, each alone and inside the same three-generated-h128x2 quarter ensemble. Traces have5000 raw112 rows and fixed i0 event periods16 and2; other coordinates update sparsely. All member raw event211 construction, native recurrence, interval work and fixed quarter accumulation are inside the timed native loop. Three repetitions use fixed A/B/C, C/B/A, A/B/C ordering, resetting every trace. Wall time, process CPU time, checksums and every repetition are retained. No Python batch-one model timing is used.

The report provides ratios and descriptive extrapolations from the previous44.5369-minute current4 audit. A large overhead could justify stopping this implementation early. A small overhead does not qualify delivery: generated weights/events, cache behavior, missing Python callbacks/loader/startup, native precision and shared-host contention limit transfer. Both workloads must be reported; neither is claimed to reconstruct the actual evaluation distribution. No model selection or quality calculation occurs.

Outputs: `sources.json`, compile/check stdout+stderr, `probe.so`, generated weight/input NPZs, numerical/cost `result.json`, and outer `execution.json`, all in the fresh output directory. The benchmark loads only those newly generated NPZs/state, not any existing model or dataset.
