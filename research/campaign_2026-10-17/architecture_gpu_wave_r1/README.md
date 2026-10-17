# Architecture GPU wave R1

**Status: recipe and host-side runner prepared; target-host preflight and all fits are pending.** No GPU host, GPU test, fit, export, grade, rental, billing action, or submission was started while preparing this packet.

This packet compares four scratch, labels-only models on one common seed and frozen fold. It asks whether causal attention or selective state-space memory improves the same label objective over a GRU control, and whether a longer attention window helps. It does not assume any architecture wins. The Transformer and Mamba-2 use four layers while the reference GRU uses two; layer count is a declared architecture difference, so this is not a parameter-count match.

| Arm | Model | Contrast |
|---|---|---|
| A | GRU 256 × 2 | Shared scratch reference |
| B | Causal Transformer, 256 × 4, 8 heads, window 256, FFN × 4 | Attention against A |
| C | Same Transformer as B, window 2048 | Longer context against B |
| D | Mamba-2, 256 × 4, state 64, head dim 64, expand 2, conv width 4, chunk 256 | Selective state-space alternative against A |

The frozen recipe uses seed 3, one full epoch, batch 8, chunk 2000, FP32, `wmse` with chunk normalization, and `--batch-order seeded` for every arm. AdamW, learning rate 0.001, weight decay 0.01, beta2 0.999, OneCycle, and global gradient clipping at 1.0 are explicit shared settings. All four fit the same component-disjoint train and valid groups from fold 0 and use the same normalization pass. The shared input width is 211. Evaluation is withheld during fitting and checkpoints are fixed at epoch 0. No teacher or initializer contributes to targets or weights. With `drop_last`, each full epoch has 1,492 batches covering 11,936 group sequences; the final four groups are omitted consistently. Each batch has 10 optimizer updates, so the expected full-fit total is 14,920 attempted and completed updates. The frozen split SHA256 is `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`.

The preregistered grades compare B and D with A, and C with both A and B. For B, C, and D, a second fixed mixture grade uses 90% A plus 10% of the candidate predictions; its weights are declared before fitting and add no training runs.

F0 has been inspected repeatedly. Its 468 evaluation groups are development evidence, not a fresh holdout. Each model must export predictions for the full 1,873-group validation panel; the scorer then selects only the 468 frozen F0 evaluation groups. The output remains a research teacher. No native CPU support or submission readiness is claimed.

## GPU-host setup

Use a Linux host with the intended CUDA devices visible, a checkout of this source revision, and the supplied data under a single data root. The preparation command accepts a separate `--data-root`; it also accepts the existing portable data registry through `--portable-data-manifest` and exports that path as `WNN_PORTABLE_DATA_MANIFEST` to fitting and prediction processes.

Use the pinned CUDA 13.0 PyTorch wheel and install Mamba-2 in two stages. Its pinned source setup imports Torch while preparing package metadata, so Torch and its build prerequisites must already be present. The exact commit is `95d8aba8a8c75aedcaa6143713b11e745e7cd0d9` and must also match the VCS requirement entry. CUDA 13.0, Torch 2.14.0 and the Mamba source build remain unverified until this command completes on the target host. PyPI is an extra index so the build prerequisites remain available alongside the PyTorch-only wheel index:

```bash
uv sync --group research --group dev
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple \
  'torch==2.14.0+cu130' setuptools wheel packaging ninja attrs einops huggingface-hub
uv pip install --python .venv/bin/python --no-build-isolation \
  -r requirements/mamba2-gpu.txt
```

Choose an output directory on persistent storage. The CPU and loader budgets below are **per GPU worker**; the packet records their aggregate for the selected GPU count. Four GPUs with these values reserve up to eight Torch CPU threads and 24 GiB of loader budget across the workers. Adjust them to the actual host before preparing.

```bash
GPU_PYTHON="$PWD/.venv/bin/python"
GPU_DATA_ROOT="/data/wunder"
GPU_DATA_MANIFEST="/data/wunder/portable_data_manifest.json"
GPU_OUTPUT_ROOT="/data/wunder_gpu_runs"

"$GPU_PYTHON" scripts/prepare_architecture_gpu_wave.py \
  --project-root "$PWD" \
  --data-root "$GPU_DATA_ROOT" \
  --portable-data-manifest "$GPU_DATA_MANIFEST" \
  --gpu-ids 0,1,2,3 \
  --cpu-threads-per-worker 2 \
  --loader-ram-gb-per-worker 6 \
  --output-dir "$GPU_OUTPUT_ROOT/architecture_gpu_wave_r1"
```

Preparation is a target-host validation step. It requires Linux, validates that each named physical GPU can initialize CUDA, validates the pinned frozen split and portable input identities, checks all required source files, hashes the full runner/trainer/source/test closure, and writes immutable arm specs plus exact per-run SHA256 allowlists. It does not create a queue outside the selected output directory or start a model process. If preparation stops partway through writing a nonempty output directory, keep it for diagnosis and choose a fresh output directory for a new preparation.

Run the stages in order. `preflight` runs the three CUDA architecture test files and a separate one-real-batch training check for every architecture. Every test must run without a skip, failure, or error; each architecture probe must complete 10 finite optimizer updates. A preflight with peak GPU memory above 80% of that device's reported capacity is held for review. Only then can `fit` start. The runner starts at most one independent worker per selected GPU and records per-step stdout/stderr, timeouts, expected output paths, and checkpoint-resume state. It enforces the declared architecture dimensions, the full 1,492-batch/14,920-update budget, and equal fit membership, normalization identity, and actual seeded sample-order digest across arms. The `grade` coordinator first writes the standalone GRU score, then grades the preregistered comparisons and fixed mixtures and records validated receipts.

```bash
WAVE="$GPU_OUTPUT_ROOT/architecture_gpu_wave_r1"

"$GPU_PYTHON" scripts/run_architecture_gpu_wave.py \
  --manifest "$WAVE/manifest.json" --stage preflight

"$GPU_PYTHON" scripts/run_architecture_gpu_wave.py \
  --manifest "$WAVE/manifest.json" --stage fit

"$GPU_PYTHON" scripts/run_architecture_gpu_wave.py \
  --manifest "$WAVE/manifest.json" --stage grade
```

`--stage all` runs those stages in order and stops at the first failed gate. A worker interruption can resume only the exact admitted spec with its saved training state; the runner checks the queue heartbeat and delegates stale-item recovery to the existing worker, which fails closed when resumable state is absent. Failed or missing outputs never count as completed. Logs and worker status live under `$WAVE/queues/`, reports under `$WAVE/reports/`, and score/receipt/ledger artifacts under `$WAVE/grades/`. A failure is retained in its report and queue result for diagnosis; no failed arm is silently omitted from the comparison.

## Source and evidence pins

[`recipe.json`](recipe.json) is the frozen arm contract. At preparation time, the helper hashes `scripts/train_rnn.py`, `scripts/predict_run.py`, the fold scorer, comparison-receipt and queue-worker code, both packet scripts, all Python sources under `src/wnn/`, the three GPU test files, the Mamba-2 dependency file, `pyproject.toml`, `requirements.txt`, and this recipe. It writes the actual closure into `source_closure.json` and binds it into every admitted spec. The run stage recomputes those hashes and refuses changed source, recipe, split, or portable-data identities.

The target-host preflight is the first point at which CUDA causality, reset behavior, streaming/chunk parity, finite gradients, real-data normalization, or an optimizer update can be claimed. This Mac preparation used source review, static Ruff checks/formatting and Git whitespace checks; no tests, model imports, training, prediction, scoring, or profiling were run here. The final submission limits remain 1 vCPU, 16 GB RAM, 60 minutes, 20 MB, offline, without GPU; this GPU packet does not establish that any model meets them. Mamba's common AdamW recipe applies weight decay to all parameters, including `A_log`, `dt_bias`, and `D`, unlike the official Mamba no-decay tags; that is a deliberate shared-optimizer comparison choice. After this teacher comparison, evaluate a separate quantization path: the existing GRU through Q8/Q4, and only a Transformer that shows useful held-out-fold signal through FP32 CPU export, INT8 and then INT4 quality/size/runtime checks. None of that work is part of this four-arm recipe.

The GPU memory report measures the PyTorch allocator peak and requires it to stay below 80% of device capacity; the CUDA context and other processes are additional. Its RAM figure is the trainer process's lifetime peak RSS, excluding loader children and page cache. These observations are training diagnostics and do not establish the final CPU serving budget.

## Final ZIP check on one CPU

[`check_submission_budget.py`](../../../scripts/check_submission_budget.py) checks an actual packaged `solution.py` ZIP on a **Linux host with Docker**. It is separate from GPU training and does not export a Transformer or Mamba checkpoint. Those new backends still need a CPU serving implementation or distillation into an existing GRU student before this gate can qualify their endpoint. The script and image were statically reviewed only; neither Docker build nor inference was run on this Mac.

Build the [Python 3.11 CPU image](Dockerfile.cpu-budget) on that host, using only this campaign folder as build context. For the supplied complete validation panel of 1,873 sequences × 20,000 rows:

```bash
docker build \
  -f research/campaign_2026-10-17/architecture_gpu_wave_r1/Dockerfile.cpu-budget \
  -t wnn-cpu-budget:20261017 \
  research/campaign_2026-10-17/architecture_gpu_wave_r1

python3 scripts/check_submission_budget.py \
  --submission /data/wunder/submissions/candidate.zip \
  --data /data/wunder/datasets/valid.parquet \
  --expected-rows 37460000 \
  --image wnn-cpu-budget:20261017 \
  --output-dir /data/wunder_cpu_checks/candidate_r1
```

Use the actual panel path and exact row count if testing another dataset. The output directory must be new and on disk. The gate checks ZIP size ≤20,000,000 bytes, pins one allowed CPU core plus a one-CPU quota, limits RAM to 16,000,000,000 bytes with swap disabled, and runs the real streaming verifier with no network, no requested GPU, read-only inputs/root filesystem and single-thread library settings. Python 3.11 is checked inside the container. The 3,600-second deadline includes container creation, loading, all supplied rows, scoring and prediction-file output; input hashing and image preparation are outside that interval.

`budget_receipt.json` records the ZIP, dataset and verifier hashes, image ID, CPU model/core, wall time, exit/OOM state and expected/observed row coverage. A clean pass means **`passed_for_supplied_dataset`**. It does not establish hidden-test coverage or identical speed on the competition CPU. The legacy verifier's `eta_min_39M_rows` field is an estimate and is not the gate's timing evidence. Quantization must additionally preserve acceptable official WP against the corresponding FP32 endpoint; fitting the limits alone does not establish useful quality.

The checker also requires enough free disk for extracted files and both scratch/output prediction arrays. Its 16GB expanded-ZIP guard is an operational extraction safeguard, not an additional official competition limit.

## Source records

- [Architecture proposal and prior attention evidence](../../unified/ideas/TRANSFORMER_GPU_WAVE_20261017_R1.md) explains why H218 is not a matched sliding-window Transformer test.
- The same proposal records the team's later circle-feature, matched GRU-student KD and Chronos/TimesFM ideas, their causal-data constraints and implementation gaps. They are follow-up hypotheses, not extra runnable arms in this packet.
- [Original fast GRU queue](../fast_gpu_queue_r1/README.md) is a separate set of recipes and is not a dependency of this architecture wave.
