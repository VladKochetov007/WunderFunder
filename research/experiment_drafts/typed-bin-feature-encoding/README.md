# Fixed numerical bases for a causal Transformer

[draft PR #35](https://github.com/VladKochetov007/WunderFunder/pull/35) → `research-team`.
Status: **implemented; B300 execution and acceptance pending**. Zero fits, zero measured gains.
The pinned PR32 runtime (`d7559e94008c0d658898b59290785ae457b1b7bb`) is merged into
this branch, so a checkout contains the trainer. Its previous distillation panels are not part of this launch.

The six fixed configs compare `raw_events`, `uniform16`, `quantile16` on F0/seed1001
and F2/seed1002. One GPU per fit; parallelize independent fits across GPUs. Increasing
DDP ranks is rejected because it changes the frozen batch. Configs require a B300 device;
the actual NVIDIA driver is recorded in profile/fit identity. No queue manager is added.

## Implementation and scientific amendment, 2026-10-22

- All arms consume 323 values: raw112, canonical causal event99, and 112 indicators
  equal to one for nonfinite raw inputs. Nonfinite raw values become zero before
  event construction. Other derived overflow is an error. Events span the complete
  sequence and never reset at training chunks; inference can compute them row by row.
- Shared fitting-only FP64 moment accumulation produces FP32 mean/std; continuous
  scales have a 0.001 floor, missing indicators retain identity normalization.
  Every eligible whole fitting component is scanned; labels, scored masks and
  held-out features are not used to estimate preprocessing.
- **Prospective clarification:** quantiles are deterministic estimates from a
  uniform priority reservoir of 131072 fitting rows, seed221022. The 17 knots use
  probabilities k/16; both binned arms share full-fitting min/max endpoints. This
  bounded-memory approximation replaces the proposal's unspecified exact-quantile
  implementation before any outcomes. Repeated knots contribute zero without a
  zero division; out-of-range basis values clamp, and the raw skip stays available.
- Each binned arm adds a zero-initialized 1792→256 map, preserving common initial
  parameters and RNG. The common-parameter digest is recorded and checked by the
  panel grader. Raw versus binned comparisons remain capacity-confounded.
- RoPE causal Transformer: 4 layers, width256, heads4, FF1024, dropout0.05,
  per-layer context256, chunk256, one token per row. The model's effective receptive
  field can span 1021 rows; context256 is not a global network window.
- Full fitting uses WMSE on clipped targets ±2, absolute-target weights, no target
  standardization, no KD/hint/auxiliary losses. All rows after the 99-row warmup
  contribute: train has no `is_scored`. Evaluation always uses the actual valid
  `is_scored AND need_prediction` mask. Warmup is forwarded in both paths.
- AdamW, LR2e-4, weight decay0.01, gradient clip1, batch32, one chunk/update,
  12 epochs. WSD has 2% warmup, 20% linear cooldown, final LR ratio0.1.
  Only epoch11 is graded; no outer-score checkpoint selection. Training and
  inference use FP32, no autocast/TF32, math SDPA, bounded KV state.

## Receiver commands (Linux B300 only; not run locally)

Use the Python version in `.python-version` and the repository's locked environment: `uv sync --locked --group research`.
Check the actual device/driver with `nvidia-smi`. Transfer the complete canonical
source data, chains/folds caches and frozen split manifests. Do not edit split
membership or fake original mtimes. For changed transfer mtimes, build a content
registry on the original source host with `scripts/build_portable_manifest.py`
and set `WNN_PORTABLE_DATA_MANIFEST` to that fold's registry on the receiver.

Before expensive work run the receiver checks:

```bash
uv run --locked pytest tests/test_sequence_numeric.py tests/test_numeric_sequence_integration.py tests/test_b300_sequence_models.py
```

Set real receiver paths (example placeholders below must be replaced):

```bash
export DATA_ROOT=/srv/wunder/data
export SPLIT=/srv/wunder/splits/fold0.json
export OUT=/srv/wunder/experiments/typed-bin-r1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Prepare **once per fold**, then reuse the same artifact for all three arms:

```bash
uv run --locked python scripts/run_numeric_experiment.py prepare --fold 0 --arm raw_events --data-root "$DATA_ROOT" --split-manifest "$SPLIT" --output-root "$OUT"
uv run --locked python scripts/run_numeric_experiment.py profile --fold 0 --arm raw_events --data-root "$DATA_ROOT" --split-manifest "$SPLIT" --output-root "$OUT"
```

The four-update engineering profile writes
`$OUT/profiles/f0_raw_events/profile_receipt.json`; it is not a scientific fit or
checkpoint. Inspect all losses, runtime identity, errors, throughput and peak
allocated/reserved/free memory. Require the complete batch32 geometry to fit with
operational headroom. Each of the six configs needs its own successful profile.
A failed profile is not permission to silently change batch/precision/context.

The inherited trainer requires an explicitly approved copy of the genuine profile.
After inspection, copy it with `approved: true` (preserve every other field):

```bash
python - "$OUT/profiles/f0_raw_events/profile_receipt.json" "$OUT/profiles/f0_raw_events/approved.json" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
assert receipt['status'] == 'profile_complete' and not receipt['errors']
receipt['approved'] = True
with open(sys.argv[2], 'x') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
PY
uv run --locked python scripts/run_numeric_experiment.py fit --fold 0 --arm raw_events --data-root "$DATA_ROOT" --split-manifest "$SPLIT" --output-root "$OUT" --profile-receipt "$OUT/profiles/f0_raw_events/approved.json"
uv run --locked python scripts/run_numeric_experiment.py evaluate --fold 0 --arm raw_events --data-root "$DATA_ROOT" --split-manifest "$SPLIT" --output-root "$OUT"
```

Repeat profile/fit/evaluate for `uniform16`, `quantile16`; then prepare F2 once and
run its three arms with `--fold 2` and the F2 split/registry. Each fit writes
`runs/f{fold}_{arm}/model.pt`, fixed-final metadata and resumable epoch-boundary
state. An interrupted fit can explicitly pass `--resume <saved training_state.pt>`;
existing runs are never silently replaced. Evaluation produces immutable per-row
predictions and a canonical `score.json`.

After all six endpoints:

```bash
uv run --locked python scripts/score_numeric_panel.py --runs "$OUT/runs" --configs research/experiment_drafts/typed-bin-feature-encoding/configs --output "$OUT/panel.json"
```

The grader verifies configs, source/runtime, common initialization, shared
preprocessing and scored coverage. It reports every cell/target and all three
contrasts, recomputing pooled `global_wp_clip2_v1` under 10000 paired whole-component
resamples. Independent fold substreams are averaged replicate by replicate.
Raw-moment precision ambiguity fails closed. CIs describe these fixed fitted
predictions, excluding seed and unseen-fold uncertainty; all folds are exposed.
The primary contrast is quantile-minus-uniform, practical threshold +0.0003.
An interval crossing zero is inconclusive, not evidence to abandon the family.

CPU submission speed, packaging, distillation and quantization are separate later
experiments. This teacher panel does not claim compliance with the 1-vCPU/20-MB limits.
See [PR.md](PR.md), [recipe.json](recipe.json) and [VALIDATION.json](VALIDATION.json).
