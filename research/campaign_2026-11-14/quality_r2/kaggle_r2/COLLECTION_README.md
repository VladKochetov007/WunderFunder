# Kaggle R2 result collection

`collect_results.py` is a receiver for the six sealed R2 worker outputs. It does not fit models, fetch Kaggle files, export checkpoints, or promote a quality claim. Preparation reads the runner and producer layout only; it does not load any `training_state.pt`.

## Download only runner outputs

The kernel writes to `/kaggle/working/results`; the input archive is mounted under `/kaggle/input`; expanded inputs live under `/kaggle/working/experiment`, outside the filtered results directory. From the repository root, after the root owner confirms the run has finished, download only the results directory with the Kaggle CLI's filename regex:

```bash
.venv/bin/kaggle kernels output \
  vladkochetov07/wnn-q14-strict-q14s-20261114-r2-runner \
  --path /home/vlad/development/wunder/research_tmp/q14_kaggle_r2/downloaded_results \
  --file-pattern '^results/.*$'
```

The expected layout is `downloaded_results/results/summary.json` and `downloaded_results/results/<run>/result.json`, `worker.log`, plus `artifacts/artifacts/runs/<run>/...` and `artifacts/artifacts/runtime/<run>.json`. The duplicated `artifacts/` is from the existing runner's output layout: it prefixes each declared `artifacts/...` path with the per-run export directory.

## Validate and write the collection JSON

```bash
.venv/bin/python research/campaign_2026-11-14/quality_r2/kaggle_r2/collect_results.py \
  /home/vlad/development/wunder/research_tmp/q14_kaggle_r2/downloaded_results \
  --report-json research/campaign_2026-11-14/quality_r2/kaggle_r2/COLLECTION_R2.json
```

The script also prints the same JSON to stdout. Exit code 0 means all six outputs passed validation; 2 means collection is incomplete; 1 means a present result failed integrity or recipe checks. Missing result or artifact files are incomplete unless the runner records a failure/incident. A reported worker failure, hash mismatch, wrong recipe, invalid runtime receipt, or state mismatch is failed. Neither incomplete nor failed reports contain scores or a gate result.

Before touching any optimizer state, the collector requires all six complete result records, the complete runner summary, all declared files and their runner-recorded sizes/SHA-256 values, `SELECTED_FIXED 3` in each worker log, four metric-identified history rows, matching provenance, and T4 strict-FP32 runtime receipts. Only then it imports PyTorch and loads each CPU `training_state.pt` to require the final four-epoch boundary, the sealed source hashes, and `strict_fp32 is True` in every optimizer parameter group, including at least one Muon group.

The saved argument dictionaries must agree across runs except for `name`, `seed`, `distill_alpha`, and `distill_boot_h`; each run is also checked against its sealed spec. Teacher checkpoint/prediction hashes, train-only fit membership, split SHA, fixed epoch-3 selection, and checkpoint hashes are verified. Scores use epoch 3 `weighted_pearson` under `global_wp_clip2_v1`.

For a complete collection JSON, `paired_deltas` reports per-seed and mean alpha .9 minus alpha .5 and boot8 minus alpha .9. `alpha90_gate.status` is `met` only when the alpha .9 minus .5 paired mean is at least `0.0005` and both seed deltas are positive. This is the declared screen rule on exposed full-valid development evidence, not a promotion or leaderboard claim.

Preparation checks were limited to Ruff and a missing-output-root preflight. No matching completed local fixture was available, so no score path or CPU checkpoint load was exercised. Root review is required before using this receiver on downloaded outputs.

Root review corrected source qualification and exact frozen membership (order is allowed to differ), runtime normalization of eval_source and split_manifest, summary initialization and withholding per-run scores on failed state checks. [Fixture checks](COLLECTION_PREFLIGHT_R2.json) pass with real local producer metadata plus explicit invalid fixtures; the full six-worker completion path remains pending.
