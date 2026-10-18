# Strong-control F0 feature export

This directory prepares, launches, and captures one guarded inference export
for the restored Main F0 T3 control. It does not train, grade, or edit the
frozen H390 predictor, trainer, checkpoint, split, registry, or R2 overlay.
Only the separate `export_execution_r1` utilities and create-only evidence
files are written.

The exact R2 CPU loader preflight passed before this recipe was prepared. Its
raw receipt is captured at
`preflight_execution_r2/records/preflight-r2-20261018t1520z/COMPATIBILITY_RECEIPT_R2.json`;
the contract checks its semantic bindings and records the actual receipt SHA.
The capture does not assume that another runtime will produce identical
prediction bytes. It records the observed prediction and H390 sidecar hashes.

## Bound export

The predictor is the frozen file
`frozen_source_snapshot/scripts/predict_run.py`, SHA-256
`30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0`.
The control checkpoint is `model_ep5.pt`, SHA-256
`e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda`; its
restored provenance and propensity checkpoint/provenance are linked and
verified against the R2 manifest. The unchanged Main F0 split is passed at
`research/campaign_2026-09-29/artifacts/splits/fold0.json`, SHA-256
`09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`.
The R2 compatibility overlay precedes the frozen `src` directory in
`PYTHONPATH`; the predictor itself remains unchanged.

The exact predictor arguments are:

```text
clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1 valid
--epoch 5
--output-name clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1
--campaign-split-manifest <project>/research/campaign_2026-09-29/artifacts/splits/fold0.json
--device cuda
--inference-precision fp32
--campaign-full-valid-feature-only
```

This writes the 1,873 valid sequences in `valid_targets.npz` order, with shape
`(1873, 20000, 2)` and float32 output. It reads features only and does not
compute metrics. The contract rejects any existing final prediction, sidecar,
`.partial`, `allout_<output>_valid.npy`, or stale `seqsums_<output>.npz` path;
the project `cache/` output directory must itself be a real directory.
Capture checks the output NPY header, hashes the bytes, compares the sidecar's
ordered row groups with the bounded `row_group.npy` member of
`valid_targets.npz`, checks the restored checkpoint and old split hashes, and
requires the frozen H390 metadata spelling `campaign_fit_scope` equal to
`full-valid-feature-only`. In that frozen schema, both evaluation-membership
hash fields are null; no `feature_only` or `campaign_export_scope` field is
expected.

## Resource and numerical controls

Each attempt uses the existing frozen guard at
`source-a266b036/scripts/bandit_resources.py` with the matching frozen
`scripts/limited` and B virtual environment
`/home/grinchenkoeg/WunderFunder/.venv/bin/python`. The guard reserves one CPU
and 8,000,000,000 RAM bytes, requests the exact root-admitted full GPU UUID,
and has a 300-second admission timeout. It applies its own one-GPU free-memory
reserve. The child has a hard 7,200-second timeout, one numerical thread,
`WNN_GPU_MEMORY_BYTES=4000000000`, zero loader workers, a 1-GB loader budget,
and B's physical scratch directory. `NVIDIA_TF32_OVERRIDE=0` is set; before the
unchanged predictor begins, the worker sets and records TF32 disabled, cuDNN
benchmark disabled, cuDNN deterministic enabled, and float32 matmul precision
`highest`.

Root supplies a separate admission record no older than five minutes. It binds
this attempt to the successful R2 receipt and one full GPU UUID, and records a
fresh selected-device free-memory observation of at least 5,000,000,000 bytes
plus at least 2,000,000,000 bytes free on physical scratch. Its
`campaign_active_gpu_claims` and `campaign_active_gpu_workers` fields describe
only this campaign's ledger. They do not assert that other users are absent
from the physical GPU. Root separately confirms that the aggregate parent is
within two CPU cores, 18-GB memory-high, 20-GB memory-max, and zero swap. The
4-GB PyTorch allocator cap is set in the child; the guard controller's own
allocator-admission field may remain null because it does not receive that
child-only setting.

The 1-CPU/8-GB reservation and 4-GB allocator limit are unmeasured for this
exact export. The two-hour child timeout is a hard stop, not a runtime forecast.
The frozen predictor deserializes the checkpoint onto CPU before reaching its
split-loader receipt gate. The export's frozen loader then repeats its full
portable-source validation before inference; the CPU preflight's earlier
full-source hash pass is not treated as a substitute for that run-time check.
R2's CPU preflight did not deserialize checkpoint tensors, so any failure in
the export's checkpoint deserialization or subsequent source validation must
be captured as an execution failure, not inferred from that earlier pass.

## Stage and run on B

Root must stage the listed frozen snapshot, reviewed R2 overlay and receipt,
portable links, restored artifact links, this utility directory, and the
separate R2 review into the exact project root
`/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root`.
The R2 review is staged at
`research/campaign_2026-10-18/quality_gap_r1/export_preparation/preflight_execution_r2/INDEPENDENT_REVIEW_R2.json`,
outside the runtime overlay. The six portable-source links must be created
before contract preparation. The canonical chain-fold link keeps its logical
path under `data/cache/chain_folds.npz` and resolves to
`/home/grinchenkoeg/computer-vision-homework-20261018/receipts/chain_folds_derived.npz`;
R2 has already verified its 462,719-byte, SHA-256
`a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567`
identity. Do not substitute another cache variant.

Before importing or launching any utility on B, root's standard-library stage
check must compare the staged tree against the exact allowlist in
`PREPARATION_MANIFEST_R3.json`, verify each byte count/SHA, and reject symlinks,
extra files, `__pycache__`, `.pyc`, and `.pyo` files. The remotely staged files
are exactly `bindings.py`, `prepare.py`, `launch.py`, `worker.py`,
`capture.py`, `README.md`, `PREPARATION_MANIFEST_R3.json`,
`TEST_RECEIPT_R3.json`, and the independently produced
`INDEPENDENT_REVIEW_R3.json`. The superseded manifests and test receipts are
preserved as `PREPARATION_MANIFEST_R1_DRAFT_V1.json`,
`TEST_RECEIPT_R1_DRAFT_V1.json`, `PREPARATION_MANIFEST_R2_DRAFT_V1.json`, and
`TEST_RECEIPT_R2_DRAFT_V1.json`; none are part of the stage. The local test
source `tests/test_export_execution.py` is not needed on B. At import,
`bindings.py` independently requires exactly the five reviewed top-level
runtime modules and rejects Python bytecode across the utility tree; `-B` and
`PYTHONDONTWRITEBYTECODE=1` prevent new bytecode.

Once root has recorded its fresh admission JSON at
`export_preparation/export_execution_r1/records/<attempt>/ROOT_ADMISSION.json`,
run these commands on B with its project Python. Use a new, unique attempt ID;
all contracts and captures are create-only.

```bash
cd /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root
export PYTHONDONTWRITEBYTECODE=1
PY=/home/grinchenkoeg/WunderFunder/.venv/bin/python
EXEC=research/campaign_2026-10-18/quality_gap_r1/export_preparation/export_execution_r1
ATTEMPT=control-export-20261018t160000z-a1b2c3d4  # replace with a fresh unique ID
$PY -B "$EXEC/prepare.py" --attempt "$ATTEMPT" --root-admission "$EXEC/records/$ATTEMPT/ROOT_ADMISSION.json"
$PY -B "$EXEC/launch.py" --contract "$EXEC/contracts/$ATTEMPT.json"
$PY -B "$EXEC/capture.py" --contract "$EXEC/contracts/$ATTEMPT.json"
```

Review `contracts/<attempt>.json`, `records/<attempt>/LAUNCH_STARTED.json`,
`GUARD_LAUNCH.json`, `WORKER_PRE_INFERENCE.json`, and
`EXPORT_RECEIPT.json`. The receipt binds the exact contract hash, preflight
receipt SHA, guard attempt ID/status/exit status, raw guard record SHA, exact
command and `PYTHONPATH`, prediction/sidecar paths and observed hashes, and
ordered row digest. The raw guard record, unmodified sidecar, and strict-FP32
pre-inference record are copied into the attempt directory with byte counts
and SHA-256 hashes. The worker's original `WORKER_PRE_INFERENCE.json` remains
the receipt's runtime source; capture writes a separate, create-only
`CAPTURED_WORKER_PRE_INFERENCE.json` archive so it never collides with or
overwrites worker evidence. If launch is uncertain or capture stops after
writing raw evidence, preserve it and reconcile the guard state; do not
overwrite or delete evidence. A failed or incomplete attempt requires a new
attempt ID.

## Local bounded checks

These checks read only small JSON metadata and a synthetic NPY/NPZ fixture;
they make no SSH calls and load no dataset, checkpoint tensor, prediction, or
GPU. Run them from this directory:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python3.11 -B -m unittest discover -s tests -v
uv tool run ruff format --check bindings.py prepare.py launch.py worker.py capture.py tests/test_export_execution.py
uv tool run ruff check bindings.py prepare.py launch.py worker.py capture.py tests/test_export_execution.py
```
