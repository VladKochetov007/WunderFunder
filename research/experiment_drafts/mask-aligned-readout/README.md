# Mask-aligned readout R2: B300 receiver

Pre-outcome three-epoch amendment for [draft PR #36](https://github.com/VladKochetov007/WunderFunder/pull/36), targeting `research-team`. Status: **implemented locally, unvalidated on B300**. The R2 recipe records user authorization; receiver assets, profiles, fits and evaluations remain unverified/zero. No new public PR or remote write has been made. R1 is preserved byte-for-byte as [recipe-r1.json](recipe-r1.json); the changes are recorded in [AMENDMENT_20261022_R2.md](AMENDMENT_20261022_R2.md).

The fixed panel contains four full head fits, using existing fold-matched GRU parents. It tests whether weighting fitting losses by the real `is_scored` mask improves scored-row weighted Pearson while freezing the recurrent representation. The mask is a loss weight, never a model input. In the canonical trainer it weights the main loss **and the configured auxiliary losses**; this is not a main-loss-only ablation.

| Cell | Frozen parent, fixed checkpoint | Seed | Arms |
|---|---|---:|---|
| F0 | `oldz4_h230_parent_f0_s3_r3@ep3` | 1001 | `all_valid`, `is_scored_weighted` |
| F2 | `h382_muonall256prop0_f2_s3_e4@ep3` | 1002 | `all_valid`, `is_scored_weighted` |

Both parents are H256×2. Each child inherits the parent's input projection, recurrent trunk and preprocessing. Only readout parameters update. Three full epochs, fixed checkpoint epoch 2, batch 32, chunk 2000, AdamW LR 0.0005, weight decay 0.01, one-cycle schedule spanning all three epochs, WMSE, auxiliary coefficients 0.3/0.3, since-weight 32/floor 0.25. Both arms clear propensity, class and episode weighting; only the scored-mask flag differs. Save `model_ep2.pt`; no held-out evaluation or checkpoint selection during fitting. Four child fits remain the total budget, so each fit now has three times the R1 optimizer-update budget.

New fits and prediction exports use FP32 without autocast or TF32. The fixed parents have historical FP16 training: this is a controlled head-only diagnostic, not evidence of end-to-end FP32 training. Their checkpoint/provenance/split hashes are pinned in [recipe.json](recipe.json). Legacy parents are checked against those exact bytes and their recorded lineage; newer provenance fields are required for new runs, not retroactively invented for old ones.

## Receiver preparation

Use a Linux B300 host, this PR checkout, the Python version in `.python-version`, and the locked environment. The commands below are **receiver commands**; none were executed locally.

```bash
uv sync --locked --group research
export WNN_MASK_READOUT_ASSETS=/absolute/path/to/receiver-assets.json
export WNN_PROJECT_ROOT=/absolute/path/to/receiver/project
```

Copy [receiver-assets.example.json](receiver-assets.example.json) to the named manifest and fill in existing absolute paths. `project_root` points to the team's data/artifact root; code comes from this checkout. If `WNN_PROJECT_ROOT` is already set, it must match `project_root`.

Required assets under that root are the files named by both pinned campaign split manifests, `datasets/valid.parquet`, `cache/valid_targets.npz`, `cache/chains.npz`, `cache/chain_folds.npz`, and both parents' `model_ep3.pt` plus `provenance.json` in their canonical `artifacts/runs/<parent>/` directories. The split loader validates every declared source identity; the receiver separately checks the target cache against the valid parquet before scoring. These assets are not included in this code-only delivery. Do not substitute a different epoch or use F0 weights for F2. Missing or mismatched assets stop execution.

```bash
uv run --no-sync python research/experiment_drafts/mask-aligned-readout/receiver.py preflight
```

After `preflight`, use two cell workers on separate assigned B300s. Each worker keeps its selected device visible for both stages and fits the cell's two arms sequentially on the same device:

```bash
# Worker F0; set CUDA_VISIBLE_DEVICES to its assigned physical GPU.
CUDA_VISIBLE_DEVICES=<F0_GPU> uv run --no-sync python research/experiment_drafts/mask-aligned-readout/receiver.py profile --fold F0
CUDA_VISIBLE_DEVICES=<F0_GPU> uv run --no-sync python research/experiment_drafts/mask-aligned-readout/receiver.py fit --fold F0

# Worker F2; set CUDA_VISIBLE_DEVICES to a different assigned physical GPU.
CUDA_VISIBLE_DEVICES=<F2_GPU> uv run --no-sync python research/experiment_drafts/mask-aligned-readout/receiver.py profile --fold F2
CUDA_VISIBLE_DEVICES=<F2_GPU> uv run --no-sync python research/experiment_drafts/mask-aligned-readout/receiver.py fit --fold F2

# Run once after both fit workers finish.
uv run --no-sync python research/experiment_drafts/mask-aligned-readout/receiver.py predict
uv run --no-sync python research/experiment_drafts/mask-aligned-readout/score.py
```

`profile` executes one bounded treatment-shaped canonical GPU profile per cell, checking finite loss/gradients/updates, throughput and at least 10% memory headroom. It creates no scientific checkpoint. `fit --fold` checks that cell's receipt and actual device/driver/source identity, then launches its control and treatment sequentially. Exclusive per-fold stage locks reject duplicate workers; any lock left by a killed worker requires operator inspection. CUDA UUID is bound from the trainer process. After both fit workers finish, run the single `predict` and `score.py` commands above. The owning operator retains the four-hour node hard limit, stops only the owned GPU when that limit is reached, and collects/publishes the resulting evidence; no B300 runtime estimate is claimed here.

Before/after audits verify head-only trainables, unchanged frozen parameters and inherited preprocessing, correct fitting membership and complete held-out exclusion. `predict` exports both unchanged parents and four children using feature-only inference on their complete held-out folds. `score` requires the audit, checkpoint, prediction, source and split bindings to agree. Existing run/prediction/grade outputs are not silently overwritten; inspect any failed stage and preserve its artifacts before a fresh attempt. There is no automatic resume or budget expansion.

## Outputs and interpretation

Child checkpoints/provenance go to `artifacts/runs/mask_aligned_readout_r2_<fold>_s<seed>_e3_ep2_<arm>/`. Predictions and metadata use `cache/pred_mask_readout_r2_e3_ep2_<fold>_<role>_valid.*`. Profiles, locks, before/after audits and final `grade.json` go to `artifacts/experiment_evidence/E-MASK-ALIGNED-READOUT-20261022-R2/`. These names isolate amended outputs from R1.

The primary endpoint is treatment minus control under canonical `global_wp_clip2_v1` on `need_prediction & is_scored`. Report both fold/target scores and the equal-cell mean delta. The paired bootstrap uses 10,000 whole-component resamples per fold, independent fold streams derived from seed 221022, and replicate-wise equal-cell averaging. Its interval is conditional on the fixed fitted models: two exposed folds do not estimate training/seed variance or establish fresh-holdout significance.

Secondary, prespecified endpoints are the unchanged parent and `0.75 × parent + 0.25 × child` for each arm, including their deltas against the parent and against each other. No blend-weight search. Mean primary delta at least +0.0003 with both cells nonnegative supports a separately budgeted follow-up. Mixed signs remain uncertain; crossing zero alone does not eliminate the direction. There is no leaderboard or submission claim.

See [PR.md](PR.md) for the full scientific contract, [PROPOSAL_20261022.md](PROPOSAL_20261022.md) for the archived proposal, [recipe-r1.json](recipe-r1.json) and [AMENDMENT_20261022_R2.md](AMENDMENT_20261022_R2.md) for recipe history, and [VALIDATION.json](VALIDATION.json) for the checks completed here versus receiver checks still pending. Hardware memory, throughput and quality remain unmeasured until the receiver runs this packet.
