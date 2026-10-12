# J1 bounded clean-F0 blend panel

R2 freezes a small, source-qualified component panel and a two-phase runner for
an exposed-fold fixed-prediction blend screen. The worktree contains a runnable
recipe, but **no root execution release has been issued and no J1 target or
prediction payload has been read**. The runner fails closed before opening
those payloads unless a separate root release binds this exact manifest and
runner hash.

R2 supersedes the preliminary static-only R1 inventory in this directory. Keep
R1 as historical preparation evidence; use only the R2 manifest for review or
future admission.

## Frozen inputs and source qualification

The candidates are the three distinct clean F0 Muon students
`clean_t3_f0_muonall_h384_e3_s{3,4,5}_r5`, each at its selected `model_ep2.pt`
checkpoint. They are independently seeded, share the frozen fold-0 fit sets,
have no model initializer or distillation teacher, and use the shared
`c20260929_prop_f0` mask-propensity source. Training used FP16; the bound full
valid predictions are FP32 and unquantized. The propensity model is recorded
as a common weighting ancestor, not a blend member or teacher.

The R2 builder joins each epoch-2 checkpoint hash, provenance hash, FP32
sidecar hash, and full prediction NPY hash across these existing receipts:

- `clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json`,
  which binds the actual official grade, all three teacher sources and outputs,
  the fixed fold-0 metric and the actual producer preflight;
- `clean_t3_f0_matched_student_r5_cache_r1/cache_adapter_r1/ACTUAL_INPUT_VALIDATION_R2.json`,
  which records recomputed full-file hashes, NPY layout and source/checkpoint
  bindings for each teacher prediction; and
- `clean_t3_f0_matched_student_r5_cache_r1/cache_adapter_r1/CACHE_MATERIALIZATION_VALIDATION_R1.json`,
  which binds the actual cache consumer, teacher ancestry, producer-input
  validation, and its own full-array finite scans.

The actual grade's finite guarantee applies to every official scored row in
the complete 468-group F0 evaluation panel. It does not claim that unscored
rows or the 99-row prefix in each NPY are finite. At an admitted J1 read, the
runner rechecks each full-file SHA through the same open descriptor used for
mapping, then rejects any nonfinite target or prediction on the frozen metric
mask. The producer's R6 provider terminal status and collector exit code are
recorded only as diagnostics; neither is used as standalone evidence of a
successful fit or grade.

The source-qualified evaluation universe is 468 valid row groups across 141
connected components. A deterministic hash order selects 48 complete
components: 32 calibration components (110 groups) and 16 disjoint evaluation
components (52 groups). The current split manifest and graph caches are pinned;
their component partition agrees. The F0 fit set is disjoint from all 468
evaluation groups and their components. Related training groups are excluded
by the original fold split. This partition is internal development data, not a
fresh holdout.

The historical official grade uses the same cached target source and metric
contract as the runner. Its mask is `is_scored AND row_index >= 99`, equivalent
to `need_prediction AND is_scored`; it clips both targets and predictions to
`[-2, 2]` and weights by the absolute clipped target. The runner binds
`src/wnn/gmetric.py` and the unchanged shared blend primitive. It reads target
values and the scored mask only from the pinned, uncompressed members of
`cache/valid_targets.npz`; row-group and `seq_ix` indices were checked against
the split manifest without loading label or mask members during preparation.

## Frozen comparison policy

Calibration reports seven fixed vectors: the three singletons, three equal
pairs and the equal trio. One additional policy is a bounded shared-weight
two-target SLSQP search with three deterministic starts, ridge `0.1`, 50
iterations per start, and simplex constraints. Its analytic initializer
averages two per-target regularized covariance solves. The objective follows a
continuous float64 clip-2 surrogate gradient; this is not the derivative of
the official float32 computation and is not a closed-form or global optimum.
Every candidate is ranked by the unchanged official scorer on float32-synthesized
predictions.

At most 11 unique weight vectors can be scored on calibration: seven fixed
vectors, one distinct analytic start, and three distinct SLSQP endpoints;
duplicate equal/current starts are deduplicated. Evaluation scores only the
seven fixed vectors and the calibration-selected solver vector. It does not
run the optimizer or change weights. The selected weights are written to a
create-only calibration receipt; the separate evaluation call requires that
receipt and checks its manifest and root-admission hashes.

## Resource plan and execution gate

Each phase first rechecks the full SHA of all three 299,680,128-byte prediction
arrays and the 337,186,198-byte target archive (1,236,226,582 bytes total),
using bounded buffers. It then maps the same verified file descriptors and
stages complete selected sequences. The known staged arrays are 72,600,000
bytes for calibration and 34,320,000 bytes for evaluation. Calibration's
known stage plus prepared-problem arrays total 213,400,000 bytes; SLSQP and
correlation workspaces remain unmeasured. Therefore the proposed cap is 1.5
GB, one CPU, no swap, and 120 seconds per phase, with an actual profile and
fresh aggregate-memory check still required at root admission.

The root release format enforced by the runner is
`wnn.j1-root-admission.v1`, status
`admitted_j1_exposed_f0_component_blend`. It must bind the R2 manifest and
runner hashes, authorize blend-weight optimization only, deny model fitting,
prediction generation, GPU and provider access, and repeat the exact resource
limits. No such release is included here.

After root review, run calibration and evaluation as separate create-only
commands from the project root. Replace every angle-bracketed path with a new
unique output path and the exact root release:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python research/campaign_2026-10-12/joint_endpoint_optimization_r1/j1_panel_runner_r1.py \
  --phase calibration \
  --manifest research/campaign_2026-10-12/joint_endpoint_optimization_r1/J1_COMPONENT_PANEL_MANIFEST_R2.json \
  --root-admission <ROOT_RELEASE> --output <NEW_CALIBRATION_RECEIPT>
```

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python research/campaign_2026-10-12/joint_endpoint_optimization_r1/j1_panel_runner_r1.py \
  --phase evaluation \
  --manifest research/campaign_2026-10-12/joint_endpoint_optimization_r1/J1_COMPONENT_PANEL_MANIFEST_R2.json \
  --root-admission <ROOT_RELEASE> --calibration-receipt <FROZEN_CALIBRATION_RECEIPT> \
  --output <NEW_EVALUATION_RECEIPT>
```

## Static preparation checks and limits

The bounded R2 preparation passed with `PASS_STATIC_METADATA_ONLY`. It checked
the split and graph-source hashes, actual producer/grade bindings, full NPY
hash receipt and current file stats, target-cache row-group/`seq_ix` identity,
all candidate fit/evaluation disjointness, component selection, and bounded
NPY headers. It did not load model checkpoints, prediction values, target or
mask values, Parquet data pages, or run the optimizer. The prior malformed NPY
header OOM journal is preserved; the builder checks the six-byte magic,
supported version and a 65,536-byte header ceiling before reading a header.

The runner's four focused synthetic safety tests pass. They cover malformed and
oversized headers, same-descriptor NPY hash/layout checks, stored NPZ member
mapping, and the fail-closed root-release schema. `ruff check` passes for the
builder, runner and test. No real-data optimization, public-score forecast or
submission claim is made by this packet.
