# Prediction and cache safety fixes

This packet contains exact patched copies of four frozen server scripts, a reusable local helper, CPU fixtures, and a bounded replay record. Frozen audit evidence, preexisting caches, models, and logs were preserved. The four live scripts were deployed after review, with originals retained under the `.before-bugfix-20261017` suffix. The source hashes in [PATCH_MANIFEST.json](PATCH_MANIFEST.json) match the audit server-source manifest; each copied file is bound to its original server path and SHA-256.

The predictor now restores the checkpoint's supported inference settings before reading features: `zero_groups`, `a_mode`, `domain_flag`, `lookahead`, `state_reset`, `sentinel_mode`, `raw_only`, `heuristics`, `quant_feats`, `quant_i1`, `online_norm`, `prop_input`, and checkpoint readout tables. These are the same saved arguments and globals wired in the frozen trainer's `main` and `build_inputs`; the patch does not introduce new feature settings. The previous no-argument behavior remains valid fold-0 research coverage. Passing the literal `full` predicts every row group; non-valid splits require this explicit full request. Per-batch prediction checks reject malformed or nonfinite outputs without a second full-cache scan.

Every sidecar records the relative checkpoint path and SHA-256, a canonical serialized preprocessing configuration and its digest, predictor/trainer source paths and SHA-256 values, exact split membership, shape, dtype, and completion status. The consumers verify these identity fields against files in the research root. Checkpoints with future `lookahead` features are rejected by default; `RUN [SPLIT] full privileged` explicitly opts into a noncausal export and marks it `causal: false, privileged: true` in the sidecar. Do not use a privileged export for causal replay.

Both `train_rnn_plew19.py` and `train_rnn_rs.py` require that sidecar to describe a complete cache in canonical target-row order. They check the exact `(groups, 20000, 2)` shape before any teacher or family-teacher concatenation, then reject partial/misaligned caches, non-float32 arrays, stale checkpoint/source identities, nonfinite requested teacher rows, and nonfinite teacher blends. The checks are self-contained in the deploy copies so they do not require the newer main-repository `wnn` package on the creative host. A prior cache without a sidecar is intentionally not accepted as a training teacher; it needs a verified full export with a sidecar.

The xteam evaluator rejects duplicate or missing evaluation groups, target/mask/prediction shape mismatches, and selected nonfinite targets or predictions. It zeros unscored values before forming moments, so unscored NaNs cannot contaminate the result. The finite-array moment equations and `global_wp_clip2_v1` convention remain unchanged.

The reusable local implementation is [prediction_safety.py](../../../../src/wnn/prediction_safety.py). The tests load the actual changed functions and dataset classes from the deploy copies using AST extraction with mocked trainer imports, then exercise exact sidecar membership, checkpoint feature restoration, selected and unscored NaNs, malformed rows, and finite metric parity. Run them with:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_prediction_safety.py
uv run --no-project --with ruff==0.16.7 ruff check src/wnn/prediction_safety.py tests/test_prediction_safety.py
.venv/bin/python -m py_compile research/campaign_2026-10-17/research_reset_r1/bugfix/deploy/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/predict_any_env.py research/campaign_2026-10-17/research_reset_r1/bugfix/deploy/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/train_rnn_plew19.py research/campaign_2026-10-17/research_reset_r1/bugfix/deploy/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/train_rnn_rs.py research/campaign_2026-10-17/research_reset_r1/bugfix/deploy/xteam/xeval.py
```

All seven focused CPU tests pass. They use synthetic fixtures and AST-extract the changed functions/classes from the deploy copies with mocked trainer imports; they are not a checkpoint replay. A separate bounded CPU replay below exercised the deployed predictor and existing checkpoint on four canonical validation groups. No model fit or optimizer update ran. GPU output parity remains unmeasured.

## Deployment

All four files were deployed to `beleriand` at `2026-10-17T19:25:37.118201+00:00`. [DEPLOYMENT.json](DEPLOYMENT.json) records each installed SHA-256, original SHA-256, retained backup path, backup SHA-256, and `training_steps: 0`. Installed hashes equal the `after_sha256` values in [PATCH_MANIFEST.json](PATCH_MANIFEST.json). The retained originals are named `predict_any_env.py.before-bugfix-20261017`, `train_rnn_plew19.py.before-bugfix-20261017`, `train_rnn_rs.py.before-bugfix-20261017`, and `xeval.py.before-bugfix-20261017`.

To roll back the deployed scripts, restore those exact copies:

```sh
ssh beleriand 'set -eu
cp -p /home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/predict_any_env.py.before-bugfix-20261017 /home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/predict_any_env.py
cp -p /home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/train_rnn_plew19.py.before-bugfix-20261017 /home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/train_rnn_plew19.py
cp -p /home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/train_rnn_rs.py.before-bugfix-20261017 /home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/train_rnn_rs.py
cp -p /home/grinchenkoeg/xteam/xeval.py.before-bugfix-20261017 /home/grinchenkoeg/xteam/xeval.py'
```

## Bounded zero-i1 checkpoint replay

The replay loaded the existing `dftt_zi1_s3/model_ep1.pt` checkpoint and ran CPU-only inference in `eval()` plus `inference_mode()` with one intra-op and inter-op thread. Checkpoint arguments showed `zero_groups: i1` and empty `lookahead`. The trainer's manual `zero_group_cols('i1')` tensor and the deployed restore result both contained columns 52–103; inputs and predictions from those paths were bitwise identical for every selected group. The checkpoint SHA-256 before and after was `82b7c93aa9cdfbad03cf9c62b419727f0ff537ddc42ad1a88c765d01bd648c68`; optimizer steps: 0.

The corrected replay used the canonical pooled centered-moment metric `global_wp_clip2_v1` from [REPLAY_gmetric.py](REPLAY_gmetric.py), SHA-256 `2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa`. The server's older `wnn.metric.score` is a per-block mean, so it was not used. On the first four groups in canonical valid-target order from the strict425 panel, selected without model scores (`[3, 6, 14, 15]`; n=4 groups), the unmasked `ZERO_COLS=None` case scored WP 0.06289742 and restored checkpoint preprocessing scored WP 0.72994782 over the same 11,522 selected rows after the 99-row warm-up mask. This is a bounded diagnostic, not a full strict425 ranking or evidence of general model-quality gain.

The first harness attempt is preserved in [REPLAY_zero_i1.log](REPLAY_zero_i1.log). It failed before inference because `Sequences.__getitem__` returns an unbatched `(T,F)` tensor while `build_inputs` expects `(B,T,F)`. The v2 runner adds the batch dimension; the failed attempt did not alter a checkpoint, cache, or model. The corrected result, report, and log are [REPLAY_zero_i1_v2.json](REPLAY_zero_i1_v2.json), [REPLAY_zero_i1_v2.md](REPLAY_zero_i1_v2.md), and [REPLAY_zero_i1_v2.log](REPLAY_zero_i1_v2.log). [REPLAY_MANIFEST.json](REPLAY_MANIFEST.json) records exact remote/local artifact hashes, source/data identities, and replay and fetch commands.

## Patch record and limits

[diffs/](diffs/) contains short unified diffs against the exact frozen copies. The cache sidecar uses schema `wunder.full-prediction-cache.v1`; only `predict_any_env.py RUN [SPLIT] full [privileged]` with the literal `full` emits a complete-membership sidecar. Partial fold-0 exports remain usable for evaluation and diagnostics but are rejected by the training consumers. The optional `privileged` flag is required only when a checkpoint's saved `lookahead` setting uses future rows.

The old bad-cache training run remains invalid evidence. This patch does not repair or reinterpret its logs, and the four-group replay does not establish a model-quality effect. Full historical arrays without an authentic coverage sidecar are unresolved for reuse until independently verified or re-exported. No competition-data fit, calibration fit, test-time adaptation, cache regeneration, submission, or GPU inference was performed. The required code suite does include tiny synthetic optimizer/resume fixtures; those verify software and do not produce a competition model or quality result. The frozen-checkpoint replay itself performs zero optimizer steps.

## Integration verification and scope

[FINAL_CHECKS.log](FINAL_CHECKS.log) records `make quality quality-tests training-tests`: Ruff/structure checks pass, 6 stdlib checker tests pass, and 66 CPU tests pass with 2 dataset-dependent skips. Twelve preexisting NumPy warnings arise in synthetic causal-moment fixtures. [MAIN_CHECKS.log](MAIN_CHECKS.log) is the earlier run before including the seven new safety tests in Makefile. [EVENT_ASSERTION_CHECK.log](EVENT_ASSERTION_CHECK.log) rechecks the feature suite after removing the falsified H-001 target-clock assertion; streaming parity remains. No numerical feature transform changed.

[deploy_reviewed.py](deploy_reviewed.py) is the reviewed atomic deployment operation; [replay_zero_i1.py](replay_zero_i1.py) is the exact successful CPU replay harness. [deploy/README.md](deploy/README.md) indexes the four final server-source copies; [diffs/README.md](diffs/README.md) indexes their diffs. The standalone main helper is reusable and tested but is not automatically wired into every canonical trainer; deployed protection here is scoped to the four legacy server files. Identity checks intentionally reject caches if their recorded producer source has changed. Preserve that producer or regenerate a verified export; do not invent a sidecar.
