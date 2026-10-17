# Causal EWMA inputs versus an equal-width GRU control

[Draft PR #9](https://github.com/VladKochetov007/WunderFunder/pull/9) prepares one input-representation experiment. No GPU is connected, and no local tests, model execution, training or profiles were run. Code preparation is not numerical validation: `launch_ready=false` until the receiving host checks the actual paths and hardware.

`H-DRAFT-CAUSAL-SMOOTHING-20261017` / `E-DRAFT-CAUSAL-SMOOTHING-20261017-R1`; lineage: student inputs, potentially reusable by a future teacher. [PR.md](PR.md) gives the eleven-part scientific protocol; [recipe.json](recipe.json) binds the settings and decision rule. [Team inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) compares the other drafts and their dependencies.

## Comparison

Both arms train the same scratch GRU H256×2 with raw112 + existing99 event channels +224 additional channels:

| Arm | Additional channels | What changes |
|---|---|---|
| `raw-plus-control` | raw112 + zero112 | Same allocated model size, no extra temporal summary |
| `raw-plus-causal-ewma` | EWMA32(raw) + raw−EWMA32(raw) | Explicit persistent state and residual |

The EWMA decay is `2**(-1/32)` per observed row. It sees current/past inputs, initializes from the first row and resets at every sequence. 32 rows is a chosen scale, not physical seconds or an outer-fold-tuned optimum. Raw inputs remain available. The [standard recurrence](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm) motivates the implementation but does not establish a predictive benefit. Equal width controls allocated capacity; effective feature rank and optimization conditioning still differ.

Full eligible fitting membership from the existing four source-qualified manifests; 12 epochs; batch 8; 2000-row update/detach chunks; fixed epoch11; labels-only WMSE; AdamW and WSD settings in the recipe. Strict FP32 throughout, no autocast/scaler/TF32/quantization. Both arms explicitly use `--batch-order seeded`; normalization uses sampler epoch -1 and training uses the recorded seed/epoch order. Normalizers use the canonical fitting-only two-batch sample. The fit receipt checks every epoch’s sampling identity. F0/F1 seeds 1001–1008, F2/F3 seeds 1001–1007: 30 cells per arm, 60 fits. No teacher cache or pretrained checkpoint is needed.

## Commands

Run from the committed repository root. `run_plan.py` needs only Python's standard library and prints the whole matrix plus commands. It neither reads training data nor starts jobs:

```bash
python research/experiment_drafts/causal-smoothing/run_plan.py
```

On the receiving GPU host, synchronize the existing locked environment (`uv sync --locked`) and activate `.venv`. Set `WNN_PROJECT_ROOT` to the physical directory containing `datasets/train.parquet`, `datasets/valid.parquet`, the original `cache/` inputs and `research/campaign_2026-09-29/artifacts/splits/fold0.json` through `fold3.json`. Preserve the source manifest's file identities; mismatches need investigation, not replacement hashes to bypass validation. Record actual CUDA/driver/Torch and resource settings. Set `CUDA_VISIBLE_DEVICES` to the assigned GPU and `WNN_THREADS` / loader limits to the host allocation.

The following is a **plan command**, with no model execution. It resolves source/recipe/split identities and prints the exact trainer CLI:

```bash
python research/experiment_drafts/causal-smoothing/run_cell.py fit \
  --arm raw-plus-causal-ewma --fold 0 --seed 1001
```

Before scientific fits, run the authored tests on that host, inspect numerical differences, then profile both arms on the actual card. These commands have **not** been executed during preparation:

```bash
python -m pytest tests/test_causal_ewma_features.py
python research/experiment_drafts/causal-smoothing/run_cell.py profile \
  --arm raw-plus-control --fold 0 --seed 1001 --execute
python research/experiment_drafts/causal-smoothing/run_cell.py profile \
  --arm raw-plus-causal-ewma --fold 0 --seed 1001 --execute
```

Profiles use 5 warmup + 20 measured full-sequence batches and do not produce scientific fits. Review peak memory, finite updates and throughput; retain headroom for CUDA and other processes. A successful profile on one hardware/resource configuration is not a universal memory guarantee.

After that, execute each stage for the chosen cell, changing only the arm/fold/seed from the printed matrix:

```bash
python research/experiment_drafts/causal-smoothing/run_cell.py fit \
  --arm raw-plus-causal-ewma --fold 0 --seed 1001 --execute
python research/experiment_drafts/causal-smoothing/run_cell.py predict \
  --arm raw-plus-causal-ewma --fold 0 --seed 1001 --execute
python research/experiment_drafts/causal-smoothing/run_cell.py score \
  --arm raw-plus-causal-ewma --fold 0 --seed 1001 --execute
```

Prediction uses the same batch 8 / chunk 2000 settings as fitting and records them in its sidecar. This removes a known configuration mismatch; actual export memory still needs target-host verification.

Run the control with `--arm raw-plus-control` for the same cell. Assign distinct cells to separate GPUs; no queue manager is supplied. Default `--attempt 1` is immutable: after a diagnosed failure, preserve its receipt and retry with a new attempt number. Profiles have independent attempt numbers selected by `--profile-attempt`. Retry a failed postprocessing stage under a separately reviewed recovery procedure rather than deleting prior evidence or claiming exact resume. Saved training state is retained for diagnosis; this packet does not automatically resume a partial fit.

## Outputs and implementation

- [run_plan.py](run_plan.py): complete 60-fit matrix and nonexecuting commands via the shared plan expander.
- [run_cell.py](run_cell.py): one explicit `profile`, `fit`, `predict` or `score` stage; planning is the default, `--execute` enables work. Refuses stale predecessor bindings and existing attempts.
- [cell_config.py](cell_config.py) and [cell_plan.py](cell_plan.py): frozen recipe/command/source bindings and per-cell plans.
- [cell_support.py](cell_support.py): runtime records, exclusive attempt creation and complete-fit evidence validation.
- [grade_cell.py](grade_cell.py): official metric on exact outer evaluation groups and fixed masks, with input/checkpoint/prediction identities.
- [train.py](train.py): canonical modular trainer / existing prediction exporter with explicit strict-FP32 CUDA configuration.
- [training_report.py](training_report.py): `RunRecorder` integration and finite-update checks. Logs main WMSE, all-group LR, preclip gradient norm, attempted/completed/clipped/nonfinite updates, row throughput, elapsed time and memory peaks. Scalar recording adds synchronization; timing includes it. RSS covers only the training process, not loader children.
- [research_features.py](../../../src/wnn/research_features.py) and [causal_research_features.py](../../../src/wnn/causal_research_features.py): shared batched and streaming input implementations; [test source](../../../tests/test_causal_ewma_features.py) covers formula, prefix invariance, reset, streaming agreement and checkpoint reconstruction. Tests are authored, not executed here.
- [Modular trainer map](../../../src/wnn/training/README.md): reused inputs, normalization, data membership, optimization and checkpoints. `TrainingEnvironment.observer` is optional and absent by default; no existing CLI recipe enables the experiment reporter implicitly.

Stage receipts/logs live under `$WNN_PROJECT_ROOT/artifacts/causal-smoothing-runner/{profiles,cells}/`. Weights, trainer provenance/history and step records live under `artifacts/runs/csm_*`; prediction arrays and sidecars live under `cache/pred_csm_*_valid.{npy,json}`. `grade_result.json` is a per-cell result (`score.json` records its stage status), not the final comparison or a significance claim. Predecessor receipts are checksum-verified. When both same-attempt fits exist, scoring compares their seeded epoch identities; the first arm explicitly records `awaiting_counterpart`. Final aggregation must require a matched pair and revalidate both receipts. Different-attempt recovery needs explicit reconciliation. Do not discard failed cells or select the best seed.

Use [research_result_bundle.py](../../../scripts/research_result_bundle.py) to return explicit evidence and weights files in separate archives with SHA256 manifests. Include stage receipts/logs, trainer metadata/history/provenance, metrics/update events, prediction+sidecar and score as evidence; fixed checkpoint and training state as weights. Paths must be relative to the selected data root; no datasets are auto-collected.

## Decision and limits

Primary endpoint is the equal-quarter mean of per-fold mean paired per-fit WP deltas. Predeclared useful effect is +0.0005 WP to justify the added representation path; this is an allocation choice, not a measured gain or universal significance threshold. Use the [shared crossed-seed/component uncertainty protocol](../../unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md) after all 60 fits. Existing folds are exposed, and the two uncertainty sources remain separate. Final matrix aggregation, source/regime sensitivity and downstream student/blend/serving validation are post-fit work. An interval crossing zero does not close the idea.

Static code review, Ruff and the trainer AST structure checker are preparation evidence only. Runtime causality/reload parity, complete fit→predict→score, actual memory/time, 30-pair quality and the one-CPU submission envelope remain unmeasured. No model is claimed to fit the 20 MB/60-minute serving limits from this experiment.
