# R2 E3: three-epoch full-membership short screen

This is a pre-outcome amendment to the twelve-epoch R1 experiment. R1 remains
frozen in the parent `recipe.json`, `configs/`, and `README.md`; this directory
contains the separate three-epoch R2 recipe and configs. The only scientific
budget change is three complete epochs, ending at configured epoch 2. The model,
input representation, full fitting/evaluation membership, preprocessing,
seeds, batch, precision, loss, masks, metric, and WSD fractions remain fixed.
The result is a reduced-optimization-budget screen, not an R1 result.

The user authorized a strict four-hour node wall with several training epochs.
Receiver runtime is still unmeasured, so this amendment does not claim that
three epochs will fit the available wall. Bind the actual source and split
hashes, complete the six exact-config B300 profiles, and fit only if the
measured schedule leaves time for all six fits, complete evaluation, scoring,
artifact capture, cleanup, and publication. Preserve the deadline; do not change
the recipe again after outcomes are visible.

There are six new full-membership fits: `raw_events`, `uniform16`, and
`quantile16` for F0/seed1001 and F2/seed1002. Each config requires its own
four-update strict-FP32 profile receipt on one B300. `required_world_size=1`
keeps global sequence batch 32 fixed. The runner routes R2 profile and fit
commands to the same `r2_e3/configs/` files and uses separate
`preprocessing_r2_e3/`, `profiles_r2_e3/`, and `runs_r2_e3/` output directories.
The existing R1 outputs cannot satisfy these R2 paths.

## Receiver setup and checks

Use a Linux B300 checkout of this amendment, its locked environment, the exact
`datasets/train.parquet` and `datasets/valid.parquet`, and the canonical F0/F2
split manifests. If source mtimes change during transfer, use the existing
portable data manifest workflow; do not alter mtimes or split membership.

```bash
uv sync --locked --group research
uv run --locked pytest tests/test_sequence_numeric.py tests/test_numeric_sequence_integration.py tests/test_numeric_panel_receiver.py tests/test_b300_sequence_models.py
nvidia-smi
export DATA_ROOT=/srv/wunder/data
export SPLIT_F0=/srv/wunder/splits/fold0.json
export SPLIT_F2=/srv/wunder/splits/fold2.json
export OUT=/srv/wunder/experiments/typed-bin-r2-e3
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The test suite is receiver acceptance work and must finish before model fitting.
Run all six profiles as distinct config/data identities; assign one visible B300
to each worker when profiling concurrently. Inspect each receipt, including
finite FP32 updates and optimizer state, driver/device identity, peak memory,
free-memory headroom, and elapsed time. Do not approve a failed or incomplete
profile. After human review, create an approved copy that changes only its
`approved` field.

## Prepare, profile, fit, and score

Prepare the fitting-only numeric artifact once for each fold:

```bash
uv run --locked python scripts/run_numeric_experiment.py prepare --revision r2_e3 --fold 0 --arm raw_events --data-root "$DATA_ROOT" --split-manifest "$SPLIT_F0" --output-root "$OUT"
uv run --locked python scripts/run_numeric_experiment.py prepare --revision r2_e3 --fold 2 --arm raw_events --data-root "$DATA_ROOT" --split-manifest "$SPLIT_F2" --output-root "$OUT"
```

For every `(fold, arm)` pair in `{0,2} × {raw_events,uniform16,quantile16}`,
run the following profile, review, fit, and evaluation sequence. Use the same
`--revision r2_e3`, data root, split, and output root for every stage. For
example, the F0 raw arm is:

```bash
uv run --locked python scripts/run_numeric_experiment.py profile --revision r2_e3 --fold 0 --arm raw_events --data-root "$DATA_ROOT" --split-manifest "$SPLIT_F0" --output-root "$OUT"
PROFILE="$OUT/profiles_r2_e3/f0_raw_events/profile_receipt.json"
APPROVED="$OUT/profiles_r2_e3/f0_raw_events/approved.json"
uv run --locked python - "$PROFILE" "$APPROVED" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    receipt = json.load(stream)
assert receipt["status"] == "profile_complete" and not receipt["errors"]
receipt["approved"] = True
with open(sys.argv[2], "x", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write("\n")
PY
uv run --locked python scripts/run_numeric_experiment.py fit --revision r2_e3 --fold 0 --arm raw_events --data-root "$DATA_ROOT" --split-manifest "$SPLIT_F0" --output-root "$OUT" --profile-receipt "$APPROVED"
uv run --locked python scripts/run_numeric_experiment.py evaluate --revision r2_e3 --fold 0 --arm raw_events --data-root "$DATA_ROOT" --split-manifest "$SPLIT_F0" --output-root "$OUT"
```

Repeat for all six cells, inspecting each distinct profile before approving it.
After all six endpoints, grade only the R2 output and config directories:

```bash
uv run --locked python scripts/score_numeric_panel.py --runs "$OUT/runs_r2_e3" --configs research/experiment_drafts/typed-bin-feature-encoding/r2_e3/configs --output "$OUT/panel_r2_e3.json"
```

The panel grader requires each receipt's completed epoch count to equal the
exact config used to produce it and retains the configured-final-epoch policy.
It still reports the fixed metric, both targets, all contrasts, and paired
whole-component intervals. An interval crossing zero remains inconclusive.
Preserve failed cells and all source, split, profile, run, prediction, score,
panel, and output hashes. Runtime and quality remain unknown until the receiver
returns complete evidence.
