# Linux/B300 receiver commands

Prepared commands, not executed results. Use the team's provisioned PR75 Python,
PyTorch/CUDA and dependencies with BF16, efficient SDPA and compiled blocks.
GNU timeout, flock, tar and Git are required. Record actual environment and GPU
topology; do not upgrade libraries or touch another worker to make a check pass.

## Qualify source, data and allocation

On a clean receiver checkout of this published experiment branch:

```bash
export WF_REPO=/absolute/path/to/experiment-checkout
export PACKET="$WF_REPO/research/experiment_drafts/query-geometry-b300-20261114"
export WF_BASE=/absolute/path/to/repository/containing/pr75
export WF_PYTHON=/absolute/path/to/provisioned/environment/bin/python
export WF_DATA=/absolute/path/to/original/data
export WF_SPLIT=/absolute/path/to/chronological-primary-split.json
export WF_RUNS=/absolute/persistent/disk/query-geometry-unique-allocation
export WF_OWNER=receiver-name
export WF_BUDGET_ID=fresh-approved-allocation-reference
mkdir -p "$WF_RUNS"
git -C "$WF_REPO" rev-parse HEAD
git -C "$WF_REPO" status --short
nvidia-smi
bash "$PACKET/run.sh" prepare
bash "$PACKET/run.sh" check
```

The allocation reference must name server, GPU UUIDs, owner, time window,
aggregate GPU-hour limit and host/GPU/disk budgets. The six-fit inventory is not
execution permission. Reserve space for native checkpoints, resume states and
prediction arrays; use disk-backed output and preserve other workers' artifacts.

`WF_BASE` supplies immutable Git objects only. The preparer extracts PR75 into
`WF_RUNS/runtime`, applies the three-file experiment patch and adds the query
component. The complete source inventory is checked before any model import.
The existing repository pipeline remains unchanged. Source changes invalidate
receipts. `check` requires original splitSHA, fit10325, fitted-valid0 and
evaluation-valid1873, and accepts only the six frozen recipes.

## Numerical admission and profiles

Choose currently free, explicitly allocated devices. The examples use indices0/1;
replace them with your allocation. Every arm at a given seed must use the **same
ordered pair of GPU identities**. Different seeds may use separate pairs with
the same software/device class.

```bash
CUDA_VISIBLE_DEVICES=0 bash "$PACKET/run.sh" probe
export CUDA_VISIBLE_DEVICES=0,1
CELL=plain_s209401
bash "$PACKET/run.sh" profile "$CELL"
```

The one-B300 probe checks explicit-sign geometry identities, BF16 conversion,
causality, gradients, initialization, active compiled treatment, direct one-layer
K/V preservation, clean evaluation and warmed RNG replay. It is not a quality
fit or memory certificate for24 layers. Then each cell gets its own native
**50-update real-shape profile** and model/optimizer/scheduler/RNG reload check.

Review profile receipt and rank logs against the allocated limits before approval:

```bash
export WF_MAX_HOST_GIB=REPLACE_WITH_ALLOCATED_TOTAL_HOST_LIMIT
export WF_MAX_VRAM_GIB=REPLACE_WITH_ALLOCATED_PER_GPU_LIMIT_WITH_HEADROOM
export WF_REVIEW_NOTE='review of this exact profile and allocation'
bash "$PACKET/run.sh" approve "$CELL"
bash "$PACKET/run.sh" fit "$CELL"
bash "$PACKET/run.sh" evaluate "$CELL"
```

Repeat explicitly, within the budget, for `angular_s209401`, `radial_s209401`,
`plain_s209402`, `angular_s209402`, `radial_s209402`. No other cells or controls
are admitted. Do not replace an OOM cell with a smaller model/batch/chunk under
the same ID; retain the engineering failure and amend a new recipe if needed.

Caps: prepare/check/approve10min, probe30min, profile1h, fit8h, export/scorer/readout
2h each. `WF_TIMEOUT_SECONDS` can reduce, not increase, a stage cap. Non-prepare
stages also have source verification capped300s; timeout kill grace is60s.
`evaluate` invokes separate export and score stages. Owner/allocation/start/end,
exit code and complete console output are retained in `stages/`; per-stage locks
reject concurrent writes and successful stages cannot be overwritten. Failure
attempts remain available. No blind retry or queue manager is provided.

The wrapper has no resume stage. Native serialization roundtrip and a warm-graph
RNG replay do not prove fresh-process stochastic resume. An interrupted fit
requires a separately reviewed recovery decision; keep partial outputs.

## Fixed endpoint, readout and MIDAS

Each fit ends at configured E4/3280 updates. Evaluate uncorrupted full valid;
`model.pt`, terminal status, epoch index and independent prediction manifest must
agree. No best-epoch substitution or tuning to public feedback.

```bash
bash "$PACKET/run.sh" readout
```

`comparison.json` requires six distinct completed endpoints with matched data,
source, config and same-seed hardware. It contains angular−plain, radial−plain
and angular−radial on each seed, conditional component intervals and descriptive
two-seed means. Individual `score.json` files are visible before completion;
partial panels do not yield a complete-panel claim.

Use [the MIDAS exporter/importer](midas/README.md) after independent scoring.
The existing metadata preview is not an execution receipt. Return preparation,
qualification, probe, profiles, rank logs, epoch/terminal metadata, predictions
manifests, grades and readout to the evidence store, including negatives. Store
weights and prediction arrays separately with hashes, sizes and actual storage
locations. Archive delivery must add archiveSHA256; do not copy supplied datasets
or weights into Git. No portal upload, spending or resource acquisition is automatic.
