# Serving-projected WMSE trajectory mean amendment

This is a zero-fit amendment to the existing [WMSE weight-mean hypothesis](../../../unified/ideas/README.md) and the October 20 NLL/WMSE continuation. It derives one candidate from the frozen F0 parent CP5 and WMSE-child CP0/CP1 after applying the exact two-output serving projection. The original full-state mean remains blocked by [the compatibility receipt](../WEIGHT_MEAN_COMPATIBILITY_R1.json): its head has 23 output rows, while the WMSE child has 25. No source checkpoint, original recipe, or prior finding is rewritten.

The projection is semantically exact for each source. The pinned training setup appends the two log-variance rows after all existing outputs; the WMSE path computes its main loss from `out[..., :2]`, while the NLL path reads the first two rows as means and the last two as log variance. The no-grad inference path returns `out[..., :2]`. The current native exporter and package exporter both strip the optional dynamics-only head and retain `head.2.weight[:2]` and `head.2.bias[:2]`. Because the final head is row-wise linear after a shared hidden representation, those auxiliary and variance rows cannot feed back into either scored output at inference. This justifies averaging *served projections*; it does not turn the operation into a full-state trajectory average.

The [frozen plan](plan.json) binds the exact CP5/CP0/CP1 bytes, provenance sidecars, F0 split and fitting-member identities, teacher-cache files, propensity, shared normalization identity, input-layout arguments, and producer/serving source hashes. At execution, [derive_projected_mean.py](derive_projected_mean.py) uses the canonical three-state `mean_state` and the pinned canonical native exporter, checks equal keys/shapes/dtypes and finite tensors, verifies normalization buffers bitwise, and runs a CPU FP32 check before averaging. That check feeds deterministic normalized 211-wide model-space sequences through the full and projected checkpoints with state reset per sequence and 2,000-row recurrent chunks; it requires first-two-output max error at most `1e-5`. It writes a single two-output checkpoint and provenance/derivation receipt under a new physical-disk scratch root. The code is prepared but has not loaded a checkpoint or executed the parity check; root owns admission and execution.

After derivation, the existing feature-only predictor can export the complete 1,873-row-group valid fold without reading labels or masks. Its unique run name needs a temporary artifact-path symlink because the frozen predictor resolves checkpoints under `WNN_PROJECT_ROOT/artifacts/runs`; predictions remain in the canonical cache so the unchanged [screen scorer](../../../../scripts/screen_score.py) can validate row order and sidecars. The [fixed score screen](score_screen_projected_mean_f0_r1.json) compares the projected mean with WMSE CP1, then compares fixed `0.75 oldZ4 + 0.25 projected-mean` against oldZ4 CP1 and the already frozen `0.75 oldZ4 + 0.25 WMSE` endpoint. It uses the official `global_wp_clip2_v1` scorer and 2,000 paired component resamples (seed 271828). There is no coefficient search or new training. F0 has 468 exposed evaluation groups and 141 components; intervals condition on these fixed predictions and do not estimate fit variation or public transfer.

The derivation command is prepared for the next serialized CPU slot; it has not been run:

```bash
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
PKT=$ROOT/research/campaign_2026-10-21/next_cycle_r1/weight_mean_projected_r1
SCRATCH=/home/vlad/development/wunder/research_tmp/overnight-nll-projected-mean-f0-r1
"$ROOT/.venv/bin/python" "$PKT/derive_projected_mean.py" \
  --project-root "$ROOT" --scratch-root "$SCRATCH"
```

When running that command from the isolated branch, use the committed packet path in that worktree instead of `$ROOT/...`. Root should run it under the admitted one-CPU/1.4-GB CPU cap. After it passes, the exact export and scoring calls are:

```bash
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
TRAINING_ROOT=/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020
PKT=$ROOT/research/campaign_2026-10-21/next_cycle_r1/weight_mean_projected_r1
SCRATCH=/home/vlad/development/wunder/research_tmp/overnight-nll-projected-mean-f0-r1
RUN=overnight_nll_f0_s3_wmse_projected_mean_r1
FOLD=$ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json

test ! -e "$ROOT/artifacts/runs/$RUN"
test ! -e "$ROOT/cache/pred_${RUN}_valid.npy"
test ! -e "$ROOT/cache/pred_${RUN}_valid.json"
ln -s "$SCRATCH/artifacts/runs/$RUN" "$ROOT/artifacts/runs/$RUN"
cd "$TRAINING_ROOT"
WNN_PROJECT_ROOT="$ROOT" uv run --no-sync python scripts/predict_run.py "$RUN" valid \
  --campaign-split-manifest "$FOLD" --campaign-full-valid-feature-only \
  --device cuda --inference-precision fp32 --loader-workers 0 --epoch 1
cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/screen_score.py "$PKT/score_screen_projected_mean_f0_r1.json" \
  --output "$SCRATCH/receipts/SCORE_PROJECTED_MEAN_F0_R1.json"
rm "$ROOT/artifacts/runs/$RUN"
```

The alias lets the pinned predictor load the scratch checkpoint while it reads the main repository's source data and writes to its canonical prediction cache. The command exports all 1,873 valid groups; the scorer uses the frozen 468-group F0 evaluation membership. Remove the symlink after scoring. No numerical score, package, or production readiness is claimed by this preparation note.
