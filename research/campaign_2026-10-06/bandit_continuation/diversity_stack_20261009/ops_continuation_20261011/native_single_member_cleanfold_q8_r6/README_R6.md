# Native Q8 clean-fold singleton controls — R6 adapter

R6 is a pending-root execution adapter for the already frozen R4 single-member Q8 reader/scorer. It preserves R3 source-qualified folds, membership lists, checkpoints/provenance, native ELF and official metric; each prediction remains one independent Q8VNNI student or HFT member. It does no fitting, parameter averaging or blend search. Outputs use the new create-only root `quality_output_single_r6/`; the historical R3 output path is untouched.

The adapter verifies the R4 and R5 byte pins, runs the R5 native package loader before R4 predictor construction, and uses R5's stable Parquet schema-name guard. R4's footer-plus-trailer identity is checked on the same open descriptor after PyArrow metadata construction and again before the first group read in each fold. Descriptor/path stat identity and the resource envelope are rechecked at every row-group boundary.

## Pending root release

The code-only contract is `CONTRACT_SINGLE_R6.json`; `PREPARATION_VALIDATION_R6.json` records preparation only. Runtime fails closed unless a root-owned `ROOT_ADMISSION_R6.json` appears at the exact path and is admitted, authorized, and matches the full expected binding map, output path, exact command, callback receipt, source pins and budget. Its required schema is `wnn.root.native-single-member-cleanfold-q8-admission.r6`, decision `PASS_FOR_ONE_FROZEN_EXPOSED_CROSSFOLD_SINGLE_MEMBER_Q8_R6_RUN`. Preparation does not create that release.

The fixed envelope is one CPU, 1,000,000,000 bytes maximum RAM, zero swap, no GPU, one thread, no provider action, no spend and a hard 1,800-second outer runtime limit. The bound `scripts/limited` invocation sets the transient scope's systemd `RuntimeMaxSec=1800s`. That timer begins at scope activation and bounds the Python process and interpreter/module startup inside the scope; setup before scope activation is outside the property. The cooperative monotonic timer begins in `main()` after Python has initialized and imported the module's standard-library dependencies, so it is secondary and cannot bound interpreter startup. R6 verifies the actual process cgroup belongs to both `wnn-campaign.slice` and the named R6 scope, and checks the effective CPU, memory and swap caps immediately before loading the bound native package, before loading checkpoint weights and constructing each predictor, before opening the dataset, and before each row-group read. No dependency installation is part of this packet; the required runtime packages are only existing NumPy and PyArrow under the recorded Python 3.11 environment. Torch is not a required package.

## Exact launch command

Run from the exact MAIN repository root, and only after the root release exists:

```sh
env CUDA_VISIBLE_DEVICES= WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% \
  WNN_THREADS=1 WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=1800s \
  WNN_SCOPE_UNIT=wnn-native-q8-single-controls-r6.scope \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
  ARROW_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  scripts/limited /tmp/wnn-projection-precompute-r1/py311/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r6/run_single_controls_r6.py
```

The externally enforced scope, not the cooperative Python check, is the process-wide timeout including interpreter startup. The script additionally refuses a different working directory, entry point, interpreter, environment, scope name, cgroup path or root-release binding.

## Bounded preparation evidence and limits

The static/focused suite stops the actual R6 fold consumer at its first row-group read with a synthetic Parquet reader, before any callback can run. A separate existing R5 callback receipt is inspected, never rerun. Footer/schema mutation and admission/resource gate tests use synthetic fixtures. Checks do not load the native model, read real data pages, predict, fit, score, contact a provider or install packages. The preparation receipt must continue to say zero full-fold rows and no score claim.

`PY311_DEPENDENCY_PROBE_R6.json` uses only `find_spec`, and `RUNTIME_LIMIT_SMOKE_R6.json` records a 2-second scope-timeout smoke test; neither loads a model or real data pages. `FOCUSED_TESTS_R6.log` records the synthetic mutation and first-row-group-intercept suite. The recorded R5 fixture covers four complete single sequences with reset/replay equality; it is not a full-fold runtime or memory profile. Root admission must assess the actual 1 GB/1 CPU/30-minute limit against its current workload context. Even after a root release, successful local execution would be an exposed-fold control result only, not a fresh holdout or public-score claim.
