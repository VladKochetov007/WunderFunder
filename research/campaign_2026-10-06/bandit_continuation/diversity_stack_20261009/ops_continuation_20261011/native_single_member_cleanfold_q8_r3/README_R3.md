# Native standalone student/HFT controls: R3 input gate

Packet ID: `native-q8-single-model-controls-packet-r3`.

R3 supersedes the R2 preparation for repository/input binding only. The R2
packet, its errata, supplementary CPython 3.13 fixture, and stopped `uv`
incident remain unchanged under `native_single_member_cleanfold_q8_r2`. The
single-student and single-HFT recipes, source-qualified fold memberships,
checkpoints, fixed Q8 precision, and official metric are inherited unchanged.
R3 adds a root-side input admission gate; it does not run inference or score a
fold.

The declared competition repository root is exactly
`/home/vlad/development/wunder/wnn_connectome_starterpack`. R3's code and
receipt references are repository-relative to the reviewed main snapshot. During
preparation the code lives in an isolated worktree based on that snapshot; the
validator still resolves the data only at `datasets/valid.parquet` beneath the
exact declared MAIN root. Before any model
import/construction or prediction-data load, it verifies the frozen R2
contract, terminal handoff, root input receipt, and the current file's size,
Parquet footer length/trailer, and SHA256 over the footer plus its eight-byte
trailer. It then opens Parquet metadata only and checks row count, group count,
20,000-row complete groups, and required callback/metric columns. It never
reads row-group data pages or labels. A successful `INPUT_PREFLIGHT_R3.json`
must be included in any later root execution admission; the old R2 command
alone is not an R3 execution authorization.

The expected data identity remains the R2-frozen receipt: 5,615,509,128 bytes,
10,686,926 footer bytes, footer-plus-trailer SHA256
`4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b`,
37,460,000 rows, and 1,873 groups. R3 recomputes and compares these values
against all four frozen R2 input receipts before opening metadata.

## Validation and execution status

`PREPARATION_VALIDATION_R3.json` records static source/hash checks and the
metadata-only input preflight. The required serving interpreter remains
`/usr/bin/python3.11` (CPython 3.11). The existing frozen R2 dependency probe
reports missing `numpy`, `pyarrow`, `torch`, and `ruff` in that interpreter.
No environment or package was created, and no dependency was installed. The
CPython 3.13 fixture remains supplementary only. Full callback validation and
scoring remain blocked pending an already-approved Python 3.11 runtime with
those dependencies and a separate root admission.

The earlier bounded environment attempt remains identified as
`native-q8-controls-builder-env-setup-r1`; its process exited and its ignored
worktree environment/cache were left untouched. This R3 packet does not alter
that receipt or any R2 file.

## Metadata-only preflight command

Run from the MAIN repository root. The existing `scripts/limited` wrapper
bounds this support check; it does not import the native model or load
prediction rows.

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=180s scripts/limited \
  .venv/bin/python \
  /path/to/isolated-r3-worktree/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r3/validate_input_identity_r3.py \
  --output /path/to/isolated-r3-worktree/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r3/INPUT_PREFLIGHT_R3.json
```

The preflight process is not a fit, callback test, or full-fold evaluation. A
preflight pass does not establish that Python 3.11 serving dependencies are
available and does not authorize execution.
