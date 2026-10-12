# Native standalone student/HFT controls on exposed F0/F1

Packet ID: `native-q8-single-model-controls-packet-r2`.

This packet prepares two single-member Q8 controls per fold from the same
independently trained student CP1 and HFT CP0 already used by the completed
`native_shared_cleanfold_q8_r2` evaluation. It reuses the exact R2 callback
runner, checkpoint-derived weight archive, and official `global_wp_clip2_v1`
accumulator. Each predictor contains one member with coefficient `1.0`,
`shared_backbone=False`, and that member's own nonlinear head. No weights or
parameters are averaged.

The reference is the completed R2 fixed `0.5/0.5` output endpoint. Its fold
scores are 0.6834944365 on F0 and 0.6700246323 on F1. This packet does not tune
mixture weights, make a public forecast, train models, or claim a fresh holdout.
The 468 F0 and 457 F1 evaluation groups have both been exposed. Student fits
used source-qualified `train+valid`; HFT fits used source-qualified `valid`
only. Existing R2 receipts bind the ancestry, membership, split, and zero
fit/evaluation component overlap for each fold.

## Runtime gate

The final-serving validation contract pins `/usr/bin/python3.11` (CPython
3.11). `PY311_RUNTIME_DEPENDENCY_PROBE_R1.json` records that the existing
system interpreter lacks `numpy`, `pyarrow`, `torch`, and `ruff`. No packages
were installed and no environment was created. Full scoring and a qualifying
standalone callback fixture remain blocked until the needed dependencies are
available in an already approved Python 3.11 runtime.

`CALLBACK_FIXTURE_SINGLE_R2_SUPPLEMENTARY_PY313.json` is preserved evidence
that the exact standalone native predictors completed one 20,000-row sequence
per fold/member, produced finite output, and replayed bitwise after reset under
the MAIN CPython 3.13 `abi3` runtime. It loaded no labels and did no quality
evaluation. It is explicitly supplementary and cannot satisfy the Python 3.11
serving-runtime gate. The required Python 3.11 fixture path is
`CALLBACK_FIXTURE_SINGLE_R2_PY311.json`; that file has not been generated.

The earlier bounded environment attempt
`native-q8-controls-builder-env-setup-r1` is preserved in
`PREPARATION_INCIDENT_UV_R1.json`. Its `uv` process exited before research
execution. The observed ignored worktree environment and cache remain
untouched. `PRESEAL_PACKET_ERRATUM_R1.json` preserves the initial output-path
mistake; `PRESEAL_PACKET_ERRATUM_R2.json` records the later runtime-pin
correction. The superseded contract and 3.13 receipt remain available under
their explicit historical/supplementary names.

## Commands after runtime dependencies are available

Run from the MAIN repository root through its existing bounded wrapper. The
following callback command is not currently runnable because the recorded
Python 3.11 dependencies are absent. Do not substitute CPython 3.13 for this
gate.

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s scripts/limited \
  /usr/bin/python3.11 \
  /path/to/native_single_member_cleanfold_q8_r2/run_single_controls_r2.py \
  --mode callback-fixture \
  --output /path/to/native_single_member_cleanfold_q8_r2/CALLBACK_FIXTURE_SINGLE_R2_PY311.json
```

Full scoring additionally requires a root admission receipt bound to the final
contract, both R2 folds and producer receipts, the Py3.11 standalone callback
fixture, exact weights/ELF/data identities, fold memberships and absent
output directory. The packet contains no root authority.

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=1800s scripts/limited \
  /usr/bin/python3.11 \
  /path/to/native_single_member_cleanfold_q8_r2/run_single_controls_r2.py \
  --mode full-quality \
  --output /path/to/native_single_member_cleanfold_q8_r2/quality_output_single_r2 \
  --root-admission /path/to/ROOT_ADMISSION_R2.json \
  --callback-fixture /path/to/native_single_member_cleanfold_q8_r2/CALLBACK_FIXTURE_SINGLE_R2_PY311.json
```

The four create-only prediction files are float32 arrays in canonical R2 group
order: F0 has shape `(468, 20000, 2)` and 74,880,128 bytes per member; F1 has
shape `(457, 20000, 2)` and 73,120,128 bytes per member. Total expected NPY
file bytes are 296,000,512 plus the result JSON. The 1,800-second full-run
ceiling is a proposed root-admission bound, not a measured standalone-model
runtime. R2 measured 509.89 seconds for its shared-backbone pair across both
folds; that timing is not transferred as a guarantee for these standalone
passes. No full-fold standalone evaluation has run.
