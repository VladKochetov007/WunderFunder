# F1 checkpoint support (preparation only)

This packet prepares bounded collection and CPU-only structural verification for the two fold-1, seed-1001 rank-1 arms. The plan remains **UNADMITTED**. No F1 fit, collection, checkpoint verification, prediction, grade, remote staging, or cell claim has occurred. Static counts below come from the frozen F1 manifest and `a266b036` loader/scheduler logic; they are not observed training results.

The frozen plan is `../F1_BELERIAND_PLAN.json` (SHA256 `96cfb67b3447866651ae04638166f17dccb9729b3d4a9340f1b72b8d7b089f96`). The sealed fold-1 manifest is `splits/fold1.json` (SHA256 `d8e9748d770bb3af525943b36b1801e6fd72d78259cc6b823d4661ffd78367d9`). The verified portable registry path on Beleriand is `/home/grinchenkoeg/computer-vision-homework-20261018/support-source/terminal-v1/PORTABLE_SOURCES.json` (SHA256 `1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378`); its archive inventory is `/home/grinchenkoeg/computer-vision-homework-20261018/support-source/terminal-v1/PORTABLE_ARCHIVES.json` (SHA256 `9a90e46958eaed2784f1f12e6a501292e4c865b1bc6d07e3f2bd233271da7341`). Rehash all inputs at any later admission boundary.

The fold-1 membership has 10,533 train groups, 1,416 fit-validation groups, and 457 evaluation groups (149 evaluation components). With batch 8 and `drop_last=True`, the loader has 1,493 batches per epoch. The frozen loop makes 10 updates per batch, so expected static totals are 14,930 updates/epoch, 29,860 updates/2,986 batches at the stage-1 boundary (`next_epoch=2`), and 89,580 updates/8,958 batches at the fixed six-epoch boundary (`next_epoch=6`, selected epoch 5). These values are asserted in the support code, but actual F1 training has not run.

The pair differs only in architecture: wide uses hidden 512/layers 1; deep uses hidden 384/layers 2. Both must be cold initialized, strict FP32 with TF32 disabled, fit-only during training, and use the exact fold-1 manifest. The checks bind dataset and target-cache identities to the portable registry, recompute source-qualified fold membership and order, validate all 138 canonical source hashes, inspect normalization and update history, require finite FP32 weights and complete Adam moment coverage, bind optimizer steps to the phase's update count, and verify resume lineage. Stage 1 archives both `model_ep0.pt` and `model_ep1.pt`; only `model_ep1.pt` must equal the stage-1 training-state model. The epoch-0 collection receipt is optional because a heartbeat may miss that intermediate checkpoint. Terminal verification requires verified same-arm and peer-arm stage-1 receipts, matching continuation lineage, preserved two-epoch history, `model_ep5.pt`/`model.pt` equality, and selected epoch 5.

## Future interfaces

These are interfaces for a separately admitted operation, not launch commands. Do not run them from an interactive shell: collection requires an externally constrained systemd service, and verification must run as a separate hidden-CUDA systemd support unit. The collector derives artifact names and paths from the plan and accepts no fit command.

```text
python collect_f1_checkpoint.py --arm wide --phase stage1 \
  --campaign-root /home/grinchenkoeg/computer-vision-homework-20261018 \
  --plan <B>/research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/F1_BELERIAND_PLAN.json \
  --output <new-create-only-receipt.json> [--epoch0-receipt <optional-F1-wide-epoch0-receipt.json>]
```

The collector archives the exact phase artifact set plus `attempt.json` and `training.log`, hashes source and copy bytes, checks the producer checkpoint hashes and stable before/after source stats, refuses existing archive/receipt paths, and records the plan, attempt, source generation, collector code hash, imported common-helper hash, and live resource envelope. Terminal collection additionally requires `--stage1-snapshot` for the same arm. Its built-in admission check requires a systemd service beneath `wnn-campaign.slice`, 25% CPU, at most 512 MB memory, no swap, a two-minute runtime, and live parent-slice headroom reserving the 8 GB training worker plus the collector under a 17 GB operating target. An interrupted copy leaves the reserved partial directory without a receipt; review it manually and do not reuse or overwrite it.

```text
python verify_f1_checkpoint.py --arm wide --phase stage1 \
  --snapshot-dir <collected-stage1-snapshot> \
  --source-root /home/grinchenkoeg/computer-vision-homework-20261018/source-a266b036 \
  --collection-receipt <collection-receipt.json> \
  --plan <B>/research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/F1_BELERIAND_PLAN.json \
  --fold-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold1.json \
  --data-registry /home/grinchenkoeg/computer-vision-homework-20261018/support-source/terminal-v1/PORTABLE_SOURCES.json \
  --archive-manifest /home/grinchenkoeg/computer-vision-homework-20261018/support-source/terminal-v1/PORTABLE_ARCHIVES.json \
  --output <new-create-only-verification.json>
```

For terminal verification, set `--phase terminal` and supply both `--stage1-snapshot-dir` / `--stage1-verification` and `--peer-stage1-snapshot-dir` / `--peer-stage1-verification`. A previously collected epoch-0 receipt may be supplied to stage-1 verification with `--epoch0-receipt`; it is checked if supplied but is not required. The verifier imports the canonical state loader from the selected source root, validates the actual imported path, hashes all local support dependencies plus the verifier itself, and writes a receipt only after all checks pass. Its caller must impose 50% CPU, 1.5 GB memory, no swap, a 120-second runtime, hidden CUDA, and parent headroom before execution. The verifier uses one Torch CPU thread and checks that CUDA stayed uninitialized; it does not launch a fit or inference.

The endpoint prediction export and `grade_endpoint.py` remain separate plan steps. This packet does not grade quality, establish convergence, measure resume parity, produce a CP30 result, or make a recommendation from an unrun checkpoint.

## Local preparation checks

From the repository root, the lightweight local checks used were:

```text
uv run --no-sync python -m unittest discover -s research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/checkpoints -p 'test_*.py' -v
uv run --no-sync python -m py_compile research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/checkpoints/*.py
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/ruff check research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/checkpoints
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/ruff format --check research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/checkpoints
```

The local run reported 12 tests: 11 passed and the real-Torch tensor fixture was skipped because this lightweight Mac environment has no Torch installation. No package was installed. Run that fixture later in the already bounded CUDA-hidden CPU support environment before relying on real tensor guards. See `PREPARATION_MANIFEST.json` for frozen input and helper hashes.
