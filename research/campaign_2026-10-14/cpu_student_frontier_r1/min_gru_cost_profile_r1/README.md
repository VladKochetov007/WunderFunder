# Matched-matrix GRU and minGRU native cost profile R1

This isolated CPU profile compares the production-shape H256×2 GRU and H384×2 minGRU callbacks, whose serving matrix counts are 972,032 and 966,528 for two outputs. The optimized release workers used the same source-qualified 20,000-row fitting sequence, clean-F0 normalization, deterministic synthetic FP32 weights, and fresh resets. FP32 and Q8 VNNI timings are measured separately. The result is cost-only; it does not fit a model or rank quality.

- [Completed report](REPORT.md): release callback latency, constructor/packing time, storage, resource limits, numerical reference errors, source-level Q8 hypothesis, and claim limits. At these shapes minGRU Q8 took 3.044× the GRU Q8 time; phase attribution is not measured.
- [Attempt manifest](ATTEMPTS_R1.json): exact successful build, worker, reference and static-check commands with command hashes, caps and output receipts. Wrapper elapsed time is recorded only where available; full callback timings are in the worker receipts.
- [Release build receipt](BUILD_R1.json): optimized build command, source hashes, toolchain, binary identity and width-support scope. The prior debug-FP32 run is kept separate.
- [Source and input manifest](SOURCE_MANIFEST_R1.json): source/runtime identities, fitting-sequence and normalization provenance, private payload handling, and width-support scope.
- [Input manifest](INPUT_MANIFEST_R1.json): immutable identity receipt only. The raw 112-feature payload and checkpoint remain in ignored local cache; no payload is tracked here.
- [Resource gates](RESOURCE_GATE_R1.json): initial projected-overage refusal, release-build admission, sequential release-worker admissions and bounded Python-reference admission.
- [Failed attempts and preflight correction](FAILED_ATTEMPTS_R1.json): preserves exact failed commands/hashes, diagnoses and successful corrections; distinguishes no-launch failures from the matrix-accounting preflight.
- [Release GRU FP32 worker](runs_r1/gru_f32_release_worker.json), [release minGRU FP32 worker](runs_r1/min_gru_f32_release_worker.json), [release GRU Q8 worker](runs_r1/gru_q8vnni_release_worker.json), and [release minGRU Q8 worker](runs_r1/min_gru_q8vnni_release_worker.json): each contains three complete reset-separated callback measurements, determinism and storage/resource receipts.
- [Bounded exact Python reference](REFERENCE_R1.json): native FP32 callback versus the Python GRU and minGRU models on 137 rows per family; numerical error is reported without claiming bitwise parity.
- [Earlier debug GRU FP32 worker](runs_r1/gru_f32_worker.json): first complete-sequence callback measurement, tied to a separate debug library and excluded from release comparisons.

The callback consumes raw 112-feature rows through the existing causal feature builder to 211 model inputs. Runtime weights are seeded synthetic proxies; no target or mask was loaded. The source-qualified sequence is fitting data, not a held-out quality set. See the report for full limitations and the precise distinction between the GRU I16 encoder and minGRU Q8 encoder.
