# F3 checkpoint support — preparation only

**Status: UNADMITTED.** This isolated packet adapts the reviewed F1 checkpoint V2 support to the pinned F3/seed1001 plan. It contains a bounded collector for the two-epoch stage1 boundary and six-epoch terminal boundary, plus a CPU-only verifier. No F3 checkpoint exists yet; no staging, training, prediction, grading, admission, or remote operation was performed here. Root owns fresh checks and any later execution.

The immutable plan is `../f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json` (SHA-256 `b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8`). The observed B split receipt binds `/home/grinchenkoeg/computer-vision-homework-20261018/splits/fold3.json` (SHA-256 `21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f`) and has 10,539 fit-train groups, 1,368 fit-valid groups, 505 evaluation groups, and 145 evaluation components. The local legacy `fold3.json` has SHA-256 `78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343` and must not be substituted; the verifier rejects it because the pinned B manifest SHA is different. Root must freshly rehash the exact B manifest before admission and provide that exact file to verification.

The unchanged F1 recipe and canonical `a266b036` source/runtime contract give 11,907 selected fit groups. At batch 8 with `drop_last=True`, this is 1,488 batches per epoch. Ten 2,000-row chunk updates per 20,000-row sequence batch yield 14,880 optimizer updates per epoch, 29,760 at stage1 (`next_epoch=2`), and 89,280 at the six-epoch endpoint (`next_epoch=6`, selected epoch 5). These are static expectations, not observed F3 counters. Both arms must start cold, retain labels-only WMSE/chunk normalization, strict FP32 with TF32 off, identical ordered membership, and the exact frozen command. Before either continuation, root must accept stage1 `total_steps=89280`, `next_epoch=2`, and `cumulative_updates=29760` from the actual state.

The helper preserves the reviewed F1 checks after substituting F3's plan/fold/count identity: exact attempt ID, full command, F3 plan path and SHA; the captured source inventory and its canonical sorted-JSON digest; captured producer-option contract (including the known exclusion of operational `save_training_state` from persisted options); all 138 checkpoint source hashes; sealed registry/archive bindings; ordered fit and evaluation membership; no fit/evaluation component overlap; FP32 runtime and labels-only ancestry; history and optimizer/scheduler counters; finite FP32 model/Adam tensors; same-arm stage1 resume state, history, and provenance; peer stage1 recipe, membership and normalization equality; and terminal epoch-5/model endpoint equality. It emits F3-specific `fold3_manifest_sha256` and `fold3_membership` fields. One shared helper version must be used for both stage1 receipts and both terminal receipts. F1 and F3 receipts cannot be mixed.

## B commands for a separately admitted operation

These are exact helper interfaces, not authorization to launch. Run them only as separate root-admitted support units with hidden CUDA, no swap, physical scratch, the specified CPU/memory/runtime limits, live parent headroom, and create-only output paths. Collector admission requires 0.25 CPU, at most 512 MB, no swap, a two-minute service limit, and parent headroom for the 8 GB fit worker plus 512 MB support reserve. Verifier admission requires 0.5 CPU, at most 1.5 GB, no swap, 120 seconds, hidden CUDA, and parent headroom. The collector copies at most 100 MB, hashes source and copy, checks stable source stats, attempt/provenance/history, and terminal resume ancestry. A failed/interrupted collection leaves its reserved archive directory; preserve it and do not overwrite or retry into it.

Stage1 wide collection:

```sh
B_ROOT=/home/grinchenkoeg/computer-vision-homework-20261018
SUPPORT=$B_ROOT/research/campaign_2026-10-18/priority25_execution_r4/f3_checkpoint_support_r1
PLAN=$B_ROOT/research/campaign_2026-10-18/priority25_execution_r4/f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json
PYTHON=$B_ROOT/../WunderFunder/.venv/bin/python
"$PYTHON" "$SUPPORT/collect_f3_checkpoint.py" --arm wide --phase stage1 --campaign-root "$B_ROOT" \
  --plan "$PLAN" \
  --output "$B_ROOT/receipts/RANK01_WIDE_F3_S1001_STAGE1_COLLECTION.json"
```

Stage1 verification:

```sh
"$PYTHON" "$SUPPORT/verify_f3_checkpoint.py" --arm wide --phase stage1 \
  --snapshot-dir "$B_ROOT/checkpoint_snapshots/rr18_r01_wide_f3_s1001/stage1_epoch1" \
  --source-root "$B_ROOT/source-a266b036" \
  --collection-receipt "$B_ROOT/receipts/RANK01_WIDE_F3_S1001_STAGE1_COLLECTION.json" \
  --plan "$PLAN" --fold-manifest "$B_ROOT/splits/fold3.json" \
  --data-registry "$B_ROOT/support-source/terminal-v1/PORTABLE_SOURCES.json" \
  --archive-manifest "$B_ROOT/support-source/terminal-v1/PORTABLE_ARCHIVES.json" \
  --output "$B_ROOT/receipts/RANK01_WIDE_F3_S1001_STAGE1_VERIFICATION.json"
```

For deep stage1, use the `DEEP` receipt names and `rr18_r01_deep_f3_s1001` snapshot/run paths. Collect both arms' stage1 checkpoints before continuation. Verify each stage1 checkpoint using this same helper packet and preserve both receipts for terminal ancestry.

Terminal collection, after that arm's six-epoch run finishes:

```sh
"$PYTHON" "$SUPPORT/collect_f3_checkpoint.py" --arm wide --phase terminal --campaign-root "$B_ROOT" \
  --plan "$PLAN" \
  --stage1-snapshot "$B_ROOT/checkpoint_snapshots/rr18_r01_wide_f3_s1001/stage1_epoch1" \
  --output "$B_ROOT/receipts/RANK01_WIDE_F3_S1001_TERMINAL_COLLECTION.json"
```

Terminal verification requires both same-arm and peer-arm stage1 snapshots and verification receipts. For wide terminal:

```sh
"$PYTHON" "$SUPPORT/verify_f3_checkpoint.py" --arm wide --phase terminal \
  --snapshot-dir "$B_ROOT/checkpoint_snapshots/rr18_r01_wide_f3_s1001_e6/epoch5_terminal" \
  --source-root "$B_ROOT/source-a266b036" \
  --collection-receipt "$B_ROOT/receipts/RANK01_WIDE_F3_S1001_TERMINAL_COLLECTION.json" \
  --plan "$PLAN" --fold-manifest "$B_ROOT/splits/fold3.json" \
  --data-registry "$B_ROOT/support-source/terminal-v1/PORTABLE_SOURCES.json" \
  --archive-manifest "$B_ROOT/support-source/terminal-v1/PORTABLE_ARCHIVES.json" \
  --stage1-snapshot-dir "$B_ROOT/checkpoint_snapshots/rr18_r01_wide_f3_s1001/stage1_epoch1" \
  --stage1-verification "$B_ROOT/receipts/RANK01_WIDE_F3_S1001_STAGE1_VERIFICATION.json" \
  --peer-stage1-snapshot-dir "$B_ROOT/checkpoint_snapshots/rr18_r01_deep_f3_s1001/stage1_epoch1" \
  --peer-stage1-verification "$B_ROOT/receipts/RANK01_DEEP_F3_S1001_STAGE1_VERIFICATION.json" \
  --output "$B_ROOT/receipts/RANK01_WIDE_F3_S1001_TERMINAL_VERIFICATION.json"
```

For deep terminal, swap wide/deep stage1 snapshot and verification inputs, and use `DEEP` terminal collection, verification and endpoint paths. The result-collection R2 combined acceptance/grade collector is not included; root must prepare a separate F3 result collector before exports, with checks adapted to the F3 schemas emitted here.

## Local checks and limits

The bounded tests use an F1 captured producer-shaped contract fixture with only fold-specific F3 plan values and schedule counters substituted; it is explicitly a derived fixture, not observed F3 evidence. The attempt fixture uses the native source-map fields and canonical digest algorithm with a tiny synthetic inventory plus the exact F3 plan key/hash; no actual F3 attempt exists. Tiny CPU Torch tensors exercise FP32 Adam moment shape, finiteness and malformed-state guards. The test suite also runs real collector and copied-receipt verifier functions against tiny synthetic stage1 and terminal artifacts, checks same-arm parent SHA binding and rejects malformed F3 commands. The exact pinned B split is not present locally; tests confirm a nonmatching local/legacy split is rejected, while the verifier's positive membership replay remains to run against the fresh exact B `fold3.json` during later root-admitted verification.

Commands used from the repository root:

```text
PYTHONPATH=research/campaign_2026-10-18/priority25_execution_r4/f3_checkpoint_support_r1 uv run --no-sync python -m unittest discover -s research/campaign_2026-10-18/priority25_execution_r4/f3_checkpoint_support_r1 -p 'test_*.py' -v
uv run --no-sync ruff check research/campaign_2026-10-18/priority25_execution_r4/f3_checkpoint_support_r1
uv run --no-sync ruff format --check research/campaign_2026-10-18/priority25_execution_r4/f3_checkpoint_support_r1
```

No F3 optimizer update, checkpoint, prediction, official score, convergence result, or CP30 claim follows from this support packet.
