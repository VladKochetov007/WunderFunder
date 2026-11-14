# Receiver commands — Linux and allocated B300 only

No commands below have been executed against models during preparation. Use the
team's provisioned PR75 environment: Python3.11+, BF16-capable B300 CUDA/PyTorch
with the inherited efficient SDPA/torch.compile support, native dependencies,
GNU timeout, flock and tar. Record actual versions; do not install a new nightly
or alter another running worker. No download, rental or paid resource is automatic.

## Source, allocation and data

In a new receiver checkout containing the published branch:

```bash
export WF_REPO=/absolute/path/to/clean/experiment-checkout
export PACKET="$WF_REPO/research/experiment_drafts/structured-perturbations-b300-20261114"
export WF_BASE=/absolute/path/to/repository/containing/pr75
export WF_PYTHON=/absolute/path/to/provisioned/environment/bin/python
export WF_DATA=/absolute/path/to/original/data
export WF_SPLIT=/absolute/path/to/chronological-primary-split.json
export WF_RUNS=/absolute/persistent/disk/structured-perturbations-unique-allocation
export WF_OWNER=receiver-name
export WF_BUDGET_ID=fresh-approved-allocation-reference
mkdir -p "$WF_RUNS"
git -C "$WF_REPO" rev-parse HEAD
git -C "$WF_REPO" status --short
nvidia-smi
bash "$PACKET/run.sh" prepare
bash "$PACKET/run.sh" check
```

The fresh allocation reference must specify the actual server, GPU UUIDs, owner,
time window and aggregate GPU-hour/disk/RAM ceilings. Ten fits are an inventory,
not authorization to consume160GPU-hours. Admit only the funded subset; if funding
stops, preserve partial evidence. Do not reuse a teammate's old allocation.

`WF_BASE` only supplies immutable Git objects. Preparation extracts PR75 into
`$WF_RUNS/runtime`, applies the packet patch and adds the focused component.
Existing source checkouts stay unchanged. `check` validates the published packet,
all effective source files, ten recipes and original source-qualified split.
Data must resolve to train10325, fitted-valid0 and evaluation-valid1873. A mismatch
blocks the stage; do not replace the split or silently turn off checks.

## Numerical admission and per-cell profile

Choose devices explicitly from the assigned free set. Use the **same two GPU
identities for every arm of a given seed**, in the same order, because the paired
readout requires them. Different seeds may use different pairs on the same software.

```bash
CUDA_VISIBLE_DEVICES=0 bash "$PACKET/run.sh" probe
export CUDA_VISIBLE_DEVICES=0,1
CELL=clean_s208401
bash "$PACKET/run.sh" profile "$CELL"
```

Probe cap30min: source-bound input/mask/causality checks plus one-layer compiled
models. Each actual config needs its own **50-update profile**, capped1h, with
full real batch/layers and engineering checkpoint/RNG reload. GPU0 above is only
an example; substitute the allocated UUID/index. A small probe never certifies
full-model memory or quality. Do not silently reduce batch/chunk/layers after OOM.

Review `profile/profile_receipt.json`, rank logs, peak host/GPU memory and runtime.
Then fill limits from the real allocation (these variables have no defaults):

```bash
export WF_MAX_HOST_GIB=REPLACE_WITH_ALLOCATED_TOTAL_HOST_LIMIT
export WF_MAX_VRAM_GIB=REPLACE_WITH_PER_GPU_LIMIT_WITH_HEADROOM
export WF_REVIEW_NOTE='receiver review of this exact profile and allocation'
bash "$PACKET/run.sh" approve "$CELL"
bash "$PACKET/run.sh" fit "$CELL"
bash "$PACKET/run.sh" evaluate "$CELL"
```

`fit` is capped8h; export and scorer are each capped2h. Set a smaller positive
`WF_TIMEOUT_SECONDS` per stage when required; the wrapper rejects larger values.
Every stage also has a separate source-verification cap300s and timeout kill
grace60s. `evaluate` consists of separate export and score stages. Stage lock,
owner, allocation, start/end, exit code and stdout/stderr are retained under
`stages/`. Completed stages cannot be overwritten; failures retain attempt logs.
An exit is not a completed epoch. Diagnose once rather than blindly retrying.
The wrapper exposes no resume action; cold-process stochastic resume needs its
own admission. The probe proves warm compiled RNG replay only.

Run the same explicit profile→review→fit→evaluate sequence for these fixed cells,
within the allocation. This is a checklist, not an automatically launched queue:

| Seed | Six-fit input panel | Four-new-fit history extension |
| --- | --- | --- |
|208401|`clean_s208401`, `stale_iid_s208401`, `stale_burst_s208401`|`history_iid_s208401`, `history_burst_s208401`|
|208402|`clean_s208402`, `stale_iid_s208402`, `stale_burst_s208402`|`history_iid_s208402`, `history_burst_s208402`|

No second clean fit is required for the history panel if its frozen lineage and
same-seed hardware contract match. Stage output directories are cell-specific.
Warmup, monitoring, optimizer, loss, gradient accumulation and checkpoint policy
are native; input corruption is one train-only hook and history corruption is
one attention read hook. Evaluation always uses the uncorrupted path.

## Independent grades and collection

```bash
bash "$PACKET/run.sh" readout
```

This admits only all ten distinct fixed-E4/3280-update checkpoints with matching
profiles, source/data identities, complete prediction manifests and canonical
scores. It writes immutable `comparison.json`. Individual `score.json` files
remain available before panel completion, but incomplete panels are labeled as
partial and do not yield an overall decision. Send negative and failed evidence
alongside positives. No validation or leaderboard-driven best-epoch selection.

For portable collection, retain `runtime_preparation.json`, `qualification.json`,
`probe.json`, stage logs, all native rank logs/metadata/profiles/epoch indices,
prediction manifests, grade JSONs and comparison in the evidence store. Store
`*.pt` and prediction arrays separately with their existing producer hashes and
size manifests on approved storage. Do not put datasets or weights into Git and
do not delete shared checkpoints. The native output carries checkpoint/source
hashes; archive delivery must add archiveSHA256 and the actual destination.

[MIDAS export/import](midas/README.md) packages independently graded endpoint
metadata after the complete readout. It neither schedules the B300 jobs nor
provides CPU serving acceptance or a portal submission.
