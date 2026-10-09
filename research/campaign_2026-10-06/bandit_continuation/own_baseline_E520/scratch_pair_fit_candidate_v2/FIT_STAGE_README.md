# E520 scratch-pair fit candidate v2 (code only)

This candidate implements exactly two independently scratch-initialized raw-only
GRU fits from `FIT_STAGE_CONFIG.json`: seed 3, then seed 4, serially on one GPU.
Each fit runs four epochs and retains the fixed zero-based epoch-3 checkpoint.
The worker exports raw train and valid predictions by memmap in 4,000-time-step
chunks, hashes them incrementally, and records input, split, checkpoint,
provenance, command, resource, and sidecar identities. It never invokes the
grader or selects an epoch, seed, or pair weight.

The training objective is pinned to `source_archive.tar` at
`src/wnn/nn.py::weighted_mse_loss` (the function SHA-256 is recorded in both
the config and `E520_FIT_STAGE_PLAN.json`). For each trainer chunk it converts
the two targets to float32 and clips them to `[-2, 2]` using
`METRIC_CLIP = 2.0`; weights each target channel by
`abs(clipped_target) * valid[..., None]`; then minimizes weighted squared error
between float32 predictions and those clipped targets, divided by that
chunk's weight sum clamped to `1e-8`. The training loss does not clip
predictions. This pinned training-target treatment is distinct from the
official metric, which clips both predictions and labels and applies the
fixed official scoring mask. The fit worker and packet builder reject missing
or changed semantics or fit-stage-plan hashes before any fit can start.

The root-frozen endpoint is implemented separately in `evaluate_e520_pair.py`.
It compares the equal raw seed-3/seed-4 epoch-3 candidate pair against the
frozen equal raw E511 C1-v2 seed-3/seed-4 epoch-1 anchor. It uses the exact
valid-target row order and digest, `need_prediction AND is_scored`, excludes
the first 99 steps, and applies the complete two-target `global_wp_clip2_v1`
metric. Connected-component bootstrap uses seed 271828 and 2,000 replicates.
Individual seed scores are descriptive context only. Fold 0 is exposed
development evidence and does not support a public forecast.

## Frozen handoff commands

After the separate recipe/grader reviewer and root controller publish their
exact hash receipts, the builder admits only those bytes. The first command is
safe before admission and creates no packet:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/scratch_pair_fit_candidate_v2/prepare_fit_packet.py \
  --review-targets-only
```

The later packet build requires both exact externally published hashes and
cannot proceed while either receipt, the successful profile receipt, fresh
quota/resource evidence, or fit admission is absent:

```sh
.venv/bin/python research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/scratch_pair_fit_candidate_v2/prepare_fit_packet.py \
  --build \
  --output-dir /kaggle/working/e520-scratch-pair-fit-v2 \
  --review-receipt /path/to/external_review.json \
  --review-sha256 <controller-published-review-sha256> \
  --controller-admission /path/to/controller_admission.json \
  --controller-admission-sha256 <root-published-admission-sha256>
```

The eventual Kaggle script command is `python fit_kernel_bundle.py`; the
builder emits a self-contained wrapper and Kaggle metadata with the exact
private raw-data and fold-assets datasets. The worker itself is runnable from
the extracted packet as `python fit_kernel.py`. Neither command has been run.
No packet was assembled, uploaded, or launched during this code-only work.

The separate grader command, after a complete output download and external
review, is:

```sh
.venv/bin/python research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/scratch_pair_fit_candidate_v2/evaluate_e520_pair.py \
  --repository-root /path/to/wnn_connectome_starterpack \
  --candidate-root /path/to/downloaded/e520_fit_output \
  --output-dir /path/to/new/e520_pair_grade
```

The grader command is documented only; no candidate scoring has run. The
reviewed evaluator and frozen reference sources must still pass independent
review before anyone uses it.

## Execution order and hard stops

Input and content hashes are checked before the portable registry is bound to
`WNN_PORTABLE_DATA_MANIFEST`; fold-0 train, valid, evaluation, target-index,
and corrupt-group-528 exclusion checks complete before the first fit. A live
resource guard then requires one visible GPU, RSS below 8 GB, allocated VRAM
below 2 GB, reserved VRAM below 2.5 GB, this process below 3 GB GPU memory,
at least 6 GiB free GPU memory, and host plus cgroup available memory above
both 4 GiB and 20%. Disk must hold the entire four-array output pair while
preserving the 8 GB floor and 512 MiB margin. Guards poll continuously and at
every trainer/export batch; any missing measurement or threshold breach aborts
without retry. Checkpoints, predictions, and receipts use staged writes and
atomic replacement; prediction payloads are never materialized as wide arrays.

The cumulative quota, campaign cap, provider runtime, live input receipt,
profile terminal receipt, independent review hash, and controller fit
admission remain unknown or absent here. This source tree is not admission to
fit or score.
