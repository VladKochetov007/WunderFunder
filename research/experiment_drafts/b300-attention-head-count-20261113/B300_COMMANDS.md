# B300 receiver commands

Use the team's provisioned Linux environment with the PR75 dependencies, Git, Bash, GNU timeout and flock. Reserve two free B300s per active cell. Assign an owner, wall time, memory and persistent disk before starting a cell. The preparation made no reservation. A stage keeps its attempt log on success, failure and timeout.

## 1. Source and data

In the following Bash commands, set `WF_REPO` and `WF_RUNTIME` to new sibling paths. Use the full published packet commit from the PR. Clone once and create a clean second worktree for the native trainer.

```bash
export WF_REPO=/absolute/path/to/head-count-packet
export WF_RUNTIME=/absolute/path/to/head-count-pr75-runtime
export WF_PACKET_SHA='replace-with-the-reviewed-full-commit'
```

Replace the placeholder with the actual 40-character SHA before executing the export. Set all other absolute paths to the provisioned receiver locations.

```bash
git clone --single-branch --branch experiment/b300-attention-head-count-20261113 \
  git@github.com:VladKochetov007/WunderFunder.git "$WF_REPO"
git -C "$WF_REPO" switch --detach "$WF_PACKET_SHA"
test "$(git -C "$WF_REPO" rev-parse HEAD)" = "$WF_PACKET_SHA"
git -C "$WF_REPO" fetch origin 8626000a4b965763a70011a0291a891aa6535d7e
git -C "$WF_REPO" worktree add --detach "$WF_RUNTIME" 8626000a4b965763a70011a0291a891aa6535d7e
export PACKET="$WF_REPO/research/experiment_drafts/b300-attention-head-count-20261113"
export WF_DATA=/absolute/path/to/original-pr49-data-root
export WF_SPLIT=/absolute/path/to/original-chronological-split.json
export WF_RUNS=/absolute/path/on/persistent-disk/head-count-panel
export WF_PYTHON=/absolute/path/to/provisioned/bin/python
export WF_OWNER='assigned-receiver-name'
bash "$PACKET/run.sh" check
```

The data root must contain the original train/valid parquet files **and** the canonical index, identity and chronological component cache artifacts referenced by the split. `check` compares membership, source, split SHA and the four native configs. Keep `$WF_RUNS/qualification.json`. Use a single persistent run root for all four cells, outside either checkout.

## 2. Causality probe and real-shape profiles

After the source check, run both head geometries on one allocated B300. This short synthetic probe checks causal prefixes, sequence resets and streaming across the context boundary. Its receipt binds to this source and qualification. The full-size profile still runs for every cell.

```bash
CUDA_VISIBLE_DEVICES=0 bash "$PACKET/run.sh" probe
```

Set `CUDA_VISIBLE_DEVICES` to the allocated pair in its assigned order. The example profiles the candidate for the first seed. Repeat for the three remaining IDs after assigning resources. The four IDs are `heads12_s207301`, `heads6_s207301`, `heads12_s207302`, `heads6_s207302`.

```bash
export CUDA_VISIBLE_DEVICES=0,1
CELL=heads6_s207301
bash "$PACKET/run.sh" profile "$CELL"
```

Read `$WF_RUNS/$CELL/profile/profile_receipt.json`. Inspect per-rank reserved VRAM, host RSS, elapsed time, finite updates and both ranks' model/optimizer/scheduler/RNG reload. Inspect free disk for four retained epoch checkpoints, resume state and predictions. Review the allocation before issuing approval. Use the same GPU pair in the same order for both arms of each seed. The paired readout enforces the recorded runtime identity. The second seed may use a different equivalent pair.

```bash
# Set measured allocation ceilings and describe the review in this shell.
: "${WF_MAX_HOST_GIB:?set available combined host memory in GiB}"
: "${WF_MAX_VRAM_GIB:?set available VRAM per GPU in GiB}"
: "${WF_REVIEW_NOTE:?describe the real profile and approved allocation}"
export WF_MAX_HOST_GIB WF_MAX_VRAM_GIB WF_REVIEW_NOTE
bash "$PACKET/run.sh" approve "$CELL"
```

## 3. Fixed endpoint, independent export and score

```bash
bash "$PACKET/run.sh" fit "$CELL"
bash "$PACKET/run.sh" evaluate "$CELL"
```

`evaluate` runs `export` and then `score`. If an export already completed and only grading is outstanding, run `bash "$PACKET/run.sh" score "$CELL"`. Keep `fit/model.pt`, `fit/training_state.pt`, epoch index, rank journals, `predictions/manifest.json` and `score.json`. A partial fit or monitor value stays out of the E4 comparison.

Limits per attempt are 10 minutes for check/approve, 30 minutes for the shared probe, 1 hour for profile, 8 hours for fit and 2 hours **each** for export and score. `WF_TIMEOUT_SECONDS` can shorten a stage to the remaining allocation. The owner and exit status are recorded under `$WF_RUNS/stages`. Preserve failed-stage artifacts before deciding whether a new attempt is worthwhile. No automatic retry or subsequent fit is scheduled.

## 4. Readout and evidence

After all four grades exist, run on the receiver:

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 1800 "$WF_PYTHON" "$PACKET/readout.py" \
  --runtime-root "$WF_RUNTIME" --run-root "$WF_RUNS" \
  --data-root "$WF_DATA" --split-manifest "$WF_SPLIT"
```

`$WF_RUNS/comparison.json` binds the four scored checkpoints to their configs, exports, original split and native source. It reports both paired deltas and within-seed component-bootstrap intervals. Transfer endpoint evidence using the [MIDAS commands](midas/README.md). Keep the separate weights in receiver storage with their SHA-256 digests and locations.
