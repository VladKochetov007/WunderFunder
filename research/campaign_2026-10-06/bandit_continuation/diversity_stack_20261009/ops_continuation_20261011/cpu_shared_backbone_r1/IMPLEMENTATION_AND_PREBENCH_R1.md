# Exact shared F0 student/HFT backbone, R1

## Scope and endpoint

This isolated prototype tests H-CPU-SHARED-01 using the own fold-0 seed-3 student
CP1 and HFT CP0. The fixed 50/50 float grade is `.6834706264664301` versus
`.6831418133898598` for the student, with a paired conditional 95% interval
`[-.0007169960064610543, .001326790542246119]` over 468 previously inspected
groups and 141 components. Those are exposed FP16-grade results, not a Q8 grade,
public forecast, or all-data final selection.

The inference comparator is a new benchmark-only two-member Q8VNNI endpoint.
The baseline runs both actual checkpoint heads and recurrent models through the
existing `EnsemblePredictor` with its default `shared_backbone=False`. The
candidate passes `shared_backbone=True`, evaluates one recurrent trunk, then
evaluates both independent nonlinear heads and averages their two outputs at
fixed `0.5/0.5` coefficients. It does not average head parameters. The existing
`solution_native` and `EnsemblePredictor` precision path remain the serving
implementation basis.

The verified source checkpoints are:

| Member | Checkpoint | SHA-256 | Raw output rows | Served outputs |
|---|---|---|---:|---|
| Student | `oldz4_h230_student_f0_s3_r3@ep1` | `664643b2653a85d192f74a976d76afa933ae56c02b2621387d2f576fd61695eb` | 31 | first 2 |
| HFT | `oldz4_h230_hft_f0_s3_r5@ep0` | `2bde556119ededc3693733280444a42f0f64138a2e3759bed6b068cb72f18ec8` | 23 | first 2 |

The source-qualified grader and `predict_run.py` both retain the first two output
rows. The producer comparison found byte-equal `mean`, `std`, input projection,
and two-layer GRU tensors; exactly four head tensors differ. Both checkpoints
have 211 inputs, GRU hidden width 256 and two layers. Their canonical inference
settings agree: no online normalization, raw-only override, heuristic/research
or privileged input, numeric map, propensity input, or saved readout. The
checkpoint grade used FP16. The runtime screen uses dynamic Q8VNNI for both
methods, so only baseline-to-candidate serving parity is claimed.

## Implementation invariants

- `shared_backbone` defaults off; the false path continues to step every member
  independently in the original member order and blend code.
- Enabling it requires exactly two dynamic Q8VNNI members, exact fixed `0.5/0.5`
  coefficients, no governor/imported packed codes/factors, and bit-identical
  normalization, input projection, and recurrent tensors. All heads still need
  two served outputs. A different trunk is rejected at construction.
- One model advances the input projection and recurrent stack. Its post-step
  hidden/projection byte codes and their whole-vector scales feed the second
  model's separate Q8 head matrices. The second head applies its own GELU and
  final output matrix. The existing recurrent update order, activation kernels,
  feature clock/reset, Q8 representation, and output accumulation remain intact.
- The second `FastRnn` object still owns duplicate weights and recurrent state;
  it is not stepped in shared mode. The benchmark reports resident storage,
  recurrent state, scratch, and logical per-step shared-trunk/head payloads so
  this prototype does not claim a memory reduction.
- No source, model, ZIP, or live trainer outside this worktree is changed.
  There is no fit, provider call, public submission, score selection, or
  promotion in this arm.

## Frozen validation and runtime protocol

The native test `q8vnni_shared_backbone_matches_two_independent_heads_across_resets`
compares both outputs bitwise against two separately stepped portable Q8VNNI
models over two 256-row resets. The Python API self-test also requires bitwise
equality between omitted and explicit `shared_backbone=False`, parity for the
shared endpoint, and constructor rejection after changing one recurrent value.

The callback panel is
`/tmp/wnn_submission_exploit_runtime_r1/valid_first10_complete_groups.parquet`,
SHA-256 `e5e6472cedfcb20ecf01195777ab986b3560379c5e81caea71973632dde0d6a3`.
It contains ten complete 20,000-row feature-only sequences. Each method receives
two reset passes (400,000 callbacks) with the same exact loaded weights,
precision, native module and output coefficients. The callback benchmark hashes
each sequence's predicted rows and checks reset replay determinism. Parquet
decode and explicit reset are outside callback timing; labels are never read.

`pair1` is ordered separate then shared. It is the only comparison unless output
hashes match exactly and the shared method is at least 5% faster. Only then may
`pair2` run once in shared-then-separate order. Any parity failure or smaller
runtime gain ends this treatment. The Intel i9-11900H benchmark results from
other work do not stand in for the actual CPU recorded by this experiment.

Each build, test, preflight or callback command uses the absolute project
`scripts/limited` wrapper, `WNN_SLICE=wnn-campaign.slice`, one thread,
`WNN_MEM_MAX=1200000000`, and `WNN_RUNTIME_MAX=120s`; checkpoint-only preflight
uses 50% CPU and compile/test/benchmark use 100%. `CARGO_BUILD_JOBS=1`. Live
campaign memory and swap limits are checked before every bounded support run.

## Frozen commands

The source worktree is `/home/vlad/development/wunder/wnn-agent-worktrees/shared-hft-backbone-20261012`,
branch `codex/shared-hft-backbone-20261012`, based on
`759c1450cd4852899ac518c2db1fdbe997af6ba1`. Build to an isolated wheel, then
unpack it to a temporary import path; do not install it into the shared venv:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s CARGO_BUILD_JOBS=1 \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/maturin build \
  --release --manifest-path native/Cargo.toml \
  --out /tmp/wnn-shared-f0-wheel-r1 \
  --interpreter /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python
```

The callback command uses `PYTHONPATH=/tmp/wnn-shared-f0-package-r1`, the isolated
wheel extraction, and the committed `benchmark_shared_f0.py`. For `pair1`, use
`--pair pair1 --order separate-shared --output <new-receipt-path>`; an eligible
single repeat uses `--pair pair2 --order shared-separate --output <new-path>`.
The runner and utilities are pinned by SHA-256 in that script and rechecked at
runtime. The test and preflight modes use `--self-test` and `--preflight-only`.

The commit containing this contract and implementation freezes the source for
the bounded callback comparison. Results, runtime/ABI/cgroup snapshots and all
attempted outcomes will be committed separately. Root review and later target
quality/package gates remain required before any deployment use.
