import marimo

__generated_with = "0.25.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    RUN_E514_COMPARISON = mo.ui.run_button(label="Run the frozen comparison after the audit passes")
    mo.vstack([mo.md(
        r"""
        # E-514 — conditional signal in a weak-model portfolio

        This notebook follows the frozen protocol in `PRE_REGISTRATION.md`. It compares
        a calibration-selected least-correlated weak member, a conditional-signal screen,
        and a fixed weak-member mean. Every candidate set and mixture weight is selected
        on other connected chains; each fold-0 chain is scored once out of fold. The fold
        and candidate pool have prior exposure, so the result is exploratory, not fresh
        confirmation or a public-score forecast.

        The final endpoint is the official `global_wp_clip2_v1` scorer. Correlations are
        diagnostics used by the preregistered screen, never a substitute for that endpoint.

        The pre-score cell validates ancestry, checkpoint identity, prediction hashes, and
        row order first. The comparison stays paused until its audit passes and this button
        is activated.
        """
    ), RUN_E514_COMPARISON])
    return RUN_E514_COMPARISON, mo


@app.cell(hide_code=True)
def _():
    from datetime import UTC, datetime
    import hashlib
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import sys

    ROOT = Path.cwd().resolve()
    _SRC_PATH = str(ROOT / "src")
    if _SRC_PATH not in sys.path:
        sys.path.insert(0, _SRC_PATH)

    from wnn.gmetric import (
        GlobalAccumulator,
        _merge_moments,
        _wp_from_moments,
        global_wp,
    )
    from wnn.splits import dev_halves, file_sha256, valid_row_groups

    LINEAGE = ROOT / "research/campaign_2026-10-06/bandit_continuation"
    OUT = LINEAGE / "conditional_weak_portfolio_E514"
    PRE_REG = OUT / "PRE_REGISTRATION.md"
    NOTEBOOK_PATH = OUT / "weak_portfolio_eda.py"
    MANIFEST_PATH = OUT / "SOURCE_MANIFEST_E514_V3.json"
    PARTITION_PATH = OUT / "E514_CHAIN_PARTITION.json"
    RESULT_PATH = OUT / "E514_RESULT.json"
    REPORT_PATH = OUT / "REPORT.md"
    PLOT_DIR = OUT / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        GlobalAccumulator,
        LINEAGE,
        MANIFEST_PATH,
        NOTEBOOK_PATH,
        OUT,
        PARTITION_PATH,
        PLOT_DIR,
        PRE_REG,
        REPORT_PATH,
        RESULT_PATH,
        ROOT,
        UTC,
        datetime,
        file_sha256,
        hashlib,
        json,
        np,
        plt,
        valid_row_groups,
    )


@app.cell
def _(
    LINEAGE,
    MANIFEST_PATH,
    NOTEBOOK_PATH,
    OUT,
    PARTITION_PATH,
    PRE_REG,
    ROOT,
    UTC,
    datetime,
    file_sha256,
    hashlib,
    json,
    mo,
    np,
    valid_row_groups,
):
    # Read source arrays in target-cache order. Large predictions remain memory-mapped.
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as _targets:
        Y = _targets["y"]
        IS_SCORED = _targets["mask"]
        ROW_GROUPS = _targets["row_group"]
        MARKET_CLASS = _targets["market_class"]
    SEQUENCE_COUNT, ROW_COUNT, TARGET_COUNT = Y.shape
    assert TARGET_COUNT == 2 and ROW_COUNT == 20_000
    OFFICIAL_MASK = IS_SCORED & (np.arange(ROW_COUNT)[None, :] >= 99)

    chain_cache = np.load(ROOT / "cache/chain_folds.npz", allow_pickle=False)
    chain_rows = chain_cache["row_group"]
    split_labels = chain_cache["split"]
    valid_chain_mask = split_labels == "valid"
    valid_positions = {
        int(group): index for index, group in enumerate(chain_rows[valid_chain_mask])
    }
    assert set(map(int, ROW_GROUPS)) == set(valid_positions)
    target_positions = np.asarray([valid_positions[int(group)] for group in ROW_GROUPS])
    CHAIN_BY_SEQUENCE = chain_cache["chain"][valid_chain_mask][target_positions]
    FOLD_BY_SEQUENCE = chain_cache["fold"][valid_chain_mask][target_positions]
    TRAIN_CHAIN_IDS = np.unique(chain_cache["chain"][split_labels == "train"])
    PURE_FOLD0 = np.flatnonzero(
        (FOLD_BY_SEQUENCE == 0) & ~np.isin(CHAIN_BY_SEQUENCE, TRAIN_CHAIN_IDS)
    )
    PURE_CHAIN_IDS = np.unique(CHAIN_BY_SEQUENCE[PURE_FOLD0])
    assert len(PURE_FOLD0) == 432 and len(PURE_CHAIN_IDS) == 123
    assert all(len(np.unique(FOLD_BY_SEQUENCE[CHAIN_BY_SEQUENCE == chain])) == 1
               for chain in np.unique(CHAIN_BY_SEQUENCE))

    PREDICTION_PATHS = {
        "c1_s3": ROOT / "cache/pred_bd26_c1v2_s3_f0_valid.npy",
        "c1_s4": ROOT / "cache/pred_bd26_c1v2_s4_f0_valid.npy",
        "ab12": ROOT / "cache/pred_e514_ab12_h128_plain_fallback_valid.npy",
        "ab48": ROOT / "cache/pred_e514_ab48_basehead_f0_valid.npy",
        "baseline": ROOT / "cache/pred_baseline_valid.npy",
    }
    PRED = {
        name: np.load(path, mmap_mode="r", allow_pickle=False)
        for name, path in PREDICTION_PATHS.items()
    }
    EXPECTED_SHAPE = (SEQUENCE_COUNT, ROW_COUNT, TARGET_COUNT)
    assert all(array.shape == EXPECTED_SHAPE and array.dtype == np.float32
               for array in PRED.values())

    def _json(path):
        return json.loads(path.read_text())

    def _write_frozen_json(path, document, *, ignore_keys=()):
        if path.exists():
            existing = _json(path)
            expected = json.loads(json.dumps(document))
            for key in ignore_keys:
                existing.pop(key, None)
                expected.pop(key, None)
            if existing != expected:
                raise RuntimeError(f"frozen artifact differs from the current audit: {path}")
            return
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split_manifest = _json(split_path)
    split_sha = file_sha256(split_path)
    assert split_sha == "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
    assert set(map(int, split_manifest["evaluation"]["valid"])) == set(
        map(int, ROW_GROUPS[FOLD_BY_SEQUENCE == 0])
    )

    C1_PROVENANCE = {}
    for seed, _prediction_key in ((3, "c1_s3"), (4, "c1_s4")):
        run = f"bd26_c1v2_s{seed}_f0"
        run_dir = ROOT / "artifacts/runs" / run
        provenance = _json(run_dir / "provenance.json")
        sidecar = _json(PREDICTION_PATHS[_prediction_key].with_suffix(".json"))
        args = provenance["args"]
        assert args["train_source"] == "train+valid"
        assert args["valid_fold"] == 0
        assert args["split_manifest_sha256"] == split_sha
        assert args["checkpoint_policy"] == "fixed"
        assert args["checkpoint_epoch"] == 1 and args["no_eval_during_fit"] is True
        assert args["seed"] == seed and args["propensity"] == "c20260929_prop_f0"
        assert provenance["selected_epoch"] == 1
        assert set(map(int, provenance["fit_row_groups"]["train"])) == set(
            map(int, split_manifest["fit"]["train"])
        )
        assert set(map(int, provenance["fit_row_groups"]["valid"])) == set(
            map(int, split_manifest["fit"]["valid"])
        )
        assert set(map(int, provenance["evaluation_row_groups"])) == set(
            map(int, split_manifest["evaluation"]["valid"])
        )
        prop = _json(ROOT / "artifacts/runs/c20260929_prop_f0/provenance.json")
        assert provenance["propensity_checkpoint_sha256"] == prop["checkpoint_sha256"]["model.pt"]
        assert prop["args"]["split_manifest_sha256"] == split_sha
        assert prop["args"]["checkpoint_policy"] == "fixed"
        assert prop["args"]["checkpoint_epoch"] == 3
        assert prop["args"]["no_eval_during_fit"] is True
        assert set(map(int, prop["fit_row_groups"]["valid"])) == set(
            map(int, split_manifest["fit"]["valid"])
        )
        assert set(map(int, prop["evaluation_row_groups"])) == set(
            map(int, split_manifest["evaluation"]["valid"])
        )
        checkpoint_name = f"model_ep{args['checkpoint_epoch']}.pt"
        checkpoint = run_dir / checkpoint_name
        assert file_sha256(checkpoint) == provenance["checkpoint_sha256"][checkpoint_name]
        assert sidecar["run"] == run and sidecar["split"] == "valid"
        assert sidecar["epoch"] == args["checkpoint_epoch"]
        assert sidecar["checkpoint_sha256"] == provenance["checkpoint_sha256"][checkpoint_name]
        assert sidecar["prediction_sha256"] == file_sha256(PREDICTION_PATHS[_prediction_key])
        assert list(map(int, sidecar["row_groups"])) == list(map(int, ROW_GROUPS))
        C1_PROVENANCE[run] = provenance

    # Legacy candidate ancestry and train/evaluation membership are reconstructed
    # from immutable run inputs and the fold map; no propensity source is allowed.
    _ab12_first_line = (ROOT / "artifacts/logs/ab12_h128_plain_fallback.log").read_text().splitlines()[0]
    ab12_args = json.loads(_ab12_first_line)
    assert ab12_args["train_source"] == "train"
    assert ab12_args["propensity"] == "" and not ab12_args["prop_input"]
    assert ab12_args["epochs"] == 1 and ab12_args["seed"] == 3

    ab48_first = (ROOT / "artifacts/logs/ab48_basehead_f0.log").read_text().splitlines()[0]
    ab48_args = json.loads(ab48_first)
    assert ab48_args["train_source"] == "valid" and ab48_args["valid_fold"] == 0
    assert ab48_args["propensity"] == "" and not ab48_args["prop_input"]
    assert ab48_args["epochs"] == 1 and ab48_args["seed"] == 3
    dev_groups, hold_groups = valid_row_groups(ROOT / "datasets/valid.parquet")
    all_valid_groups = np.concatenate([dev_groups, hold_groups])
    group_to_fold = {
        int(group): int(fold)
        for group, fold in zip(
            chain_rows[valid_chain_mask], chain_cache["fold"][valid_chain_mask], strict=True
        )
    }
    ab48_eval_groups = {g for g in map(int, all_valid_groups) if group_to_fold[g] == 0}
    ab48_fit_groups = {g for g in map(int, all_valid_groups) if group_to_fold[g] != 0}
    assert ab48_eval_groups == set(map(int, split_manifest["evaluation"]["valid"]))
    assert ab48_fit_groups.isdisjoint(ab48_eval_groups)
    assert set(map(int, ROW_GROUPS[PURE_FOLD0])).isdisjoint(ab48_fit_groups)

    # The score gate is source-only: fresh prediction sidecars bind each weak
    # model's exact checkpoint, bytes, and full target-cache row order. No
    # standalone candidate metric is calculated here.
    WEAK_SOURCE_METADATA = {}
    for _weak_name, _weak_run_name in (("ab12", "ab12_h128_plain_fallback"),
                           ("ab48", "ab48_basehead_f0")):
        prediction_path = PREDICTION_PATHS[_weak_name]
        sidecar = _json(prediction_path.with_suffix(".json"))
        checkpoint_path = ROOT / "artifacts/runs" / _weak_run_name / "model.pt"
        assert sidecar["run"] == _weak_run_name and sidecar["split"] == "valid"
        assert sidecar["checkpoint_sha256"] == file_sha256(checkpoint_path)
        assert sidecar["prediction_sha256"] == file_sha256(prediction_path)
        assert list(map(int, sidecar["row_groups"])) == list(map(int, ROW_GROUPS))
        WEAK_SOURCE_METADATA[_weak_name] = sidecar

    baseline_sidecar_path = PREDICTION_PATHS["baseline"].with_suffix(".json")
    baseline_sidecar = _json(baseline_sidecar_path)
    target_cache_path = ROOT / "cache/valid_targets.npz"
    baseline_model_path = ROOT / "baseline/baseline.onnx"
    baseline_predictor_path = ROOT / "scripts/predict_baseline.py"
    assert baseline_sidecar["source"] == "organizer_baseline_onnx_valid"
    assert baseline_sidecar["split"] == "valid"
    assert baseline_sidecar["baseline_onnx_sha256"] == file_sha256(baseline_model_path)
    assert baseline_sidecar["valid_targets_sha256"] == file_sha256(target_cache_path)
    assert baseline_sidecar["predictor_sha256"] == file_sha256(baseline_predictor_path)
    assert baseline_sidecar["receipt_writer_sha256"] == file_sha256(OUT / "make_baseline_receipt.py")
    assert baseline_sidecar["prediction_sha256"] == file_sha256(PREDICTION_PATHS["baseline"])
    assert list(map(int, baseline_sidecar["row_groups"])) == list(map(int, ROW_GROUPS))
    assert baseline_sidecar["shape"] == list(EXPECTED_SHAPE)
    assert baseline_sidecar["dtype"] == "float32"

    def _sha(path):
        return file_sha256(path)

    SOURCE_PATHS = [
        PRE_REG,
        OUT / "PRE_SCORE_AUDIT_AMENDMENT_20261007.md",
        OUT / "PRE_SCORE_AUDIT_RETRY_20261007.md",
        OUT / "PRE_SCORE_AUDIT_RETRY_2_20261007.md",
        OUT / "PRE_SCORE_GRAPH_FAILURE_20261007.json",
        OUT / "PRE_SCORE_PARTITION_REPLAY_FAILURE_20261007.json",
        OUT / "E514_EXECUTION_FAILURE_20261007.json",
        OUT / "PRE_SCORE_EXECUTION_REPAIR_20261007.md",
        OUT / "SOURCE_MANIFEST_E514.json",
        OUT / "SOURCE_MANIFEST_E514_V2.json",
        LINEAGE / "runtime/attempts/e514-ab12-prediction-v1.json",
        LINEAGE / "runtime/attempts/e514-ab48-prediction-v1.json",
        LINEAGE / "runtime/attempts/e514-baseline-prediction-v1.json",
        LINEAGE / "runtime/attempts/e514-baseline-receipt-v1.json",
        OUT / "make_baseline_receipt.py",
        NOTEBOOK_PATH,
        ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        ROOT / "cache/valid_targets.npz",
        ROOT / "cache/chain_folds.npz",
        ROOT / "src/wnn/gmetric.py",
        ROOT / "src/wnn/splits.py",
        ROOT / "scripts/train_rnn.py",
        ROOT / "scripts/predict_run.py",
        ROOT / "scripts/predict_baseline.py",
        ROOT / "scripts/decide.py",
        ROOT / "scripts/ensemble_scan.py",
        ROOT / "artifacts/logs/ab12.sh",
        ROOT / "artifacts/logs/ab12_h128_plain_fallback.log",
        ROOT / "artifacts/runs/ab12_h128_plain_fallback/model.pt",
        ROOT / "research/runs/ab12_h128_plain_fallback.md",
        ROOT / "artifacts/logs/q14.sh",
        ROOT / "artifacts/logs/ab48_basehead_f0.log",
        ROOT / "artifacts/runs/ab48_basehead_f0/model.pt",
        ROOT / "research/runs/ab48_basehead_f0.md",
        ROOT / "artifacts/runs/c20260929_prop_f0/provenance.json",
        ROOT / "artifacts/runs/c20260929_prop_f0/model.pt",
        ROOT / "research/campaign_2026-10-06/bandit_continuation/local/E511_C1_MINIMAL_PAIRED.json",
        ROOT / "research/campaign_2026-10-06/bandit_continuation/local/C1V2_C0_PAIRED_GRADER_SPEC.json",
        ROOT / "submissions/_parts/baseline_rnnmodel.npz",
        ROOT / "baseline/baseline.onnx",
        *[ROOT / "artifacts/runs" / f"bd26_c1v2_s{seed}_f0/provenance.json" for seed in (3, 4)],
        *[ROOT / "artifacts/runs" / f"bd26_c1v2_s{seed}_f0/model.pt" for seed in (3, 4)],
        *[ROOT / "artifacts/runs" / f"bd26_c1v2_s{seed}_f0/model_ep1.pt" for seed in (3, 4)],
        *[PREDICTION_PATHS[name] for name in PREDICTION_PATHS],
        *[PREDICTION_PATHS[name].with_suffix(".json") for name in PREDICTION_PATHS],
    ]
    assert all(path.is_file() for path in SOURCE_PATHS)
    source_hashes = {str(path.relative_to(ROOT)): _sha(path) for path in SOURCE_PATHS}
    assert all(np.isfinite(PRED[name][PURE_FOLD0]).all() for name in PRED)

    # This mapping uses chain IDs only, is persisted before comparative scoring,
    # and is bound to the preregistered sensors-derived seed.
    PARTITION_SEED = 474419703
    SENSOR_OUTPUT_SHA256 = "1c4711f79523231ba740478d4922bd893afbe980441de5be7c3b27c109ec2b15"
    rng_partition = np.random.default_rng(PARTITION_SEED)
    permuted_chains = rng_partition.permutation(PURE_CHAIN_IDS)
    chain_partitions = np.array_split(permuted_chains, 3)
    chain_to_outer = {
        int(chain): index
        for index, partition in enumerate(chain_partitions)
        for chain in partition
    }
    OUTER_BY_SEQUENCE = np.asarray(
        [chain_to_outer[int(chain)] for chain in CHAIN_BY_SEQUENCE[PURE_FOLD0]], dtype=np.int8
    )
    outer_counts = [
        {
            "outer_fold": index,
            "chain_count": int(len(partition)),
            "group_count": int(np.sum(OUTER_BY_SEQUENCE == index)),
        }
        for index, partition in enumerate(chain_partitions)
    ]
    partition_document = {
        "experiment_id": "E-514",
        "seed": PARTITION_SEED,
        "seed_source": "first eight hex digits of sha256(sensors stdout)",
        "sensors_stdout_sha256": SENSOR_OUTPUT_SHA256,
        "pure_fold0_chain_count": int(len(PURE_CHAIN_IDS)),
        "pure_fold0_group_count": int(len(PURE_FOLD0)),
        "outer_folds": outer_counts,
        "chain_to_outer_fold": chain_to_outer,
        "data_exposure": "Previously inspected fold 0; this is exploratory chain cross-fitting, not a fresh holdout.",
    }
    _write_frozen_json(PARTITION_PATH, partition_document)
    CHAIN_PARTITION_SHA256 = file_sha256(PARTITION_PATH)

    GATE = {
        "passed": True,
        "c1_split_and_propensity_ancestry": "both seeds: fixed epoch 1, no evaluation during fit, exact fold-0 fit/eval membership and clean fixed epoch-3 propensity source",
        "weak_candidate_ancestry": {
            "ab12": "train-only, no propensity/mask input, one fixed epoch",
            "ab48": "legacy fold-0 fit excludes fold-0 groups, no propensity input, one fixed epoch; prior score exposure recorded",
            "baseline": "supplied organizer artifact; train-only source",
        },
        "prediction_layout": "all arrays float32 with exact target-cache shape; fresh C1 and weak-model sidecars bind checkpoints and complete row-group order; organizer baseline receipt binds ONNX, target cache, predictor, predictions and rows",
        "weak_prediction_sidecars": {
            name: {"run": metadata["run"], "checkpoint_sha256": metadata["checkpoint_sha256"],
                   "prediction_sha256": metadata["prediction_sha256"]}
            for name, metadata in WEAK_SOURCE_METADATA.items()
        },
        "baseline_prediction_receipt": {
            "baseline_onnx_sha256": baseline_sidecar["baseline_onnx_sha256"],
            "prediction_sha256": baseline_sidecar["prediction_sha256"],
            "valid_targets_sha256": baseline_sidecar["valid_targets_sha256"],
        },
        "fold0_exposure": "previously inspected; no independent-confirmation claim",
    }
    source_manifest = {
        "experiment_id": "E-514",
        "created_utc": datetime.now(UTC).isoformat(),
        "stage": "pre_comparison_score_source_and_ancestry_gate_passed",
        "metric_id": "global_wp_clip2_v1",
        "pre_registration_sha256": file_sha256(PRE_REG),
        "notebook_sha256": file_sha256(NOTEBOOK_PATH),
        "source_hashes": source_hashes,
        "target_row_groups_sha256_little_endian_int64": hashlib.sha256(
            ROW_GROUPS.astype("<i8").tobytes()
        ).hexdigest(),
        "fold0_split_manifest_sha256": split_sha,
        "prediction_shapes": {name: list(map(int, array.shape)) for name, array in PRED.items()},
        "prediction_hashes": {
            name: source_hashes[str(path.relative_to(ROOT))]
            for name, path in PREDICTION_PATHS.items()
        },
        "ancestry_and_row_order_gate": GATE,
        "chain_partition": "E514_CHAIN_PARTITION.json",
        "chain_partition_sha256": CHAIN_PARTITION_SHA256,
        "pre_score_retry_amendment_sha256": file_sha256(OUT / "PRE_SCORE_AUDIT_RETRY_20261007.md"),
        "pre_score_retry_2_amendment_sha256": file_sha256(OUT / "PRE_SCORE_AUDIT_RETRY_2_20261007.md"),
        "pre_score_execution_repair_sha256": file_sha256(OUT / "PRE_SCORE_EXECUTION_REPAIR_20261007.md"),
        "previous_source_manifests": {
            "v1_sha256": file_sha256(OUT / "SOURCE_MANIFEST_E514.json"),
            "v2_sha256": file_sha256(OUT / "SOURCE_MANIFEST_E514_V2.json"),
        },
        "data_exposure": "Exploratory; all fold-0 chains and candidate families have prior exposure.",
    }
    _write_frozen_json(MANIFEST_PATH, source_manifest, ignore_keys=("created_utc",))
    MANIFEST_SHA256 = file_sha256(MANIFEST_PATH)
    AUDIT_SUMMARY = {
        "gate": "PASSED before E-514 comparative scoring",
        "pure_fold0_groups": int(len(PURE_FOLD0)),
        "pure_fold0_chains": int(len(PURE_CHAIN_IDS)),
        "outer_partition_group_counts": [item["group_count"] for item in outer_counts],
        "fresh_weak_prediction_sidecars_verified": sorted(WEAK_SOURCE_METADATA),
        "baseline_prediction_receipt_verified": True,
        "source_manifest_sha256": MANIFEST_SHA256,
        "data_exposure": "Previously inspected fold-0 development data; exploratory only.",
    }
    mo.md("**E-514 pre-score audit passed.** Fresh checkpoint-bound exports and baseline receipt are verified; no candidate score was computed in this gate. The comparison remains exposed-fold exploration.")
    return (
        CHAIN_BY_SEQUENCE,
        MANIFEST_SHA256,
        OFFICIAL_MASK,
        OUTER_BY_SEQUENCE,
        PARTITION_SEED,
        PRED,
        PURE_CHAIN_IDS,
        PURE_FOLD0,
        Y,
    )


@app.cell
def _(
    CHAIN_BY_SEQUENCE,
    GlobalAccumulator,
    MANIFEST_SHA256,
    OFFICIAL_MASK,
    OUTER_BY_SEQUENCE,
    PARTITION_SEED,
    PLOT_DIR,
    PRED,
    PURE_CHAIN_IDS,
    PURE_FOLD0,
    REPORT_PATH,
    RESULT_PATH,
    RUN_E514_COMPARISON,
    Y,
    datetime,
    json,
    mo,
    np,
    plt,
):
    from wnn.gmetric import _merge_moments, _wp_from_moments
    mo.stop(not bool(RUN_E514_COMPARISON.value), mo.md("E-514 is paused until the pre-score ancestry and row-order audit has been reviewed."))
    mo.stop(RESULT_PATH.exists() or REPORT_PATH.exists(), mo.md("E-514 outputs are already frozen; use a new experiment ID for any rerun."))
    CANDIDATE_NAMES = ("ab12", "ab48", "baseline")
    CANDIDATE_LABELS = {
        "ab12": "train-only GRU",
        "ab48": "baseline head fine-tune",
        "baseline": "organizer baseline × 0.25",
    }
    CANDIDATE_SCALE = {"ab12": 1.0, "ab48": 1.0, "baseline": 0.25}
    LAMBDAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)

    def _anchor(rows):
        return 0.5 * (
            np.asarray(PRED["c1_s3"][rows], dtype=np.float32)
            + np.asarray(PRED["c1_s4"][rows], dtype=np.float32)
        )

    def _cohort(rows, names):
        if not names:
            return _anchor(rows)
        value = np.zeros_like(_anchor(rows), dtype=np.float32)
        for name in names:
            value += np.float32(CANDIDATE_SCALE[name]) * np.asarray(
                PRED[name][rows], dtype=np.float32
            )
        return value / np.float32(len(names))

    def _blend(rows, names, lam):
        anchor = _anchor(rows)
        if not names or lam == 0:
            return anchor
        return np.float32(1.0 - lam) * anchor + np.float32(lam) * _cohort(rows, names)

    def _score(rows, names=(), lam=0.0):
        accumulator = GlobalAccumulator()
        for start in range(0, len(rows), 4):
            part = rows[start : start + 4]
            accumulator.add(Y[part], _blend(part, names, lam), OFFICIAL_MASK[part])
        return accumulator.result()

    # Stable weighted covariance summaries used only for candidate screening.
    # The weights and clipping match the official metric; final decisions use WP itself.
    def _weighted_covariance(rows, target_index, candidate, include_target):
        state_w = 0.0
        state_mean = None
        state_m2 = None
        for start in range(0, len(rows), 4):
            part = rows[start : start + 4]
            y = np.clip(Y[part, :, target_index].astype(np.float64), -2.0, 2.0)
            anchor = np.clip(_anchor(part)[:, :, target_index].astype(np.float64), -2.0, 2.0)
            other = np.clip(
                (np.float32(CANDIDATE_SCALE[candidate]) * np.asarray(PRED[candidate][part], np.float32))
                [:, :, target_index].astype(np.float64), -2.0, 2.0
            )
            selected = OFFICIAL_MASK[part] & (np.abs(y) > 0)
            w = np.abs(y[selected])
            cols = [anchor[selected], other[selected]]
            if include_target:
                cols.insert(0, y[selected])
            x = np.column_stack(cols)
            if not len(w):
                continue
            local_w = float(w.sum())
            local_mean = np.sum(x * w[:, None], axis=0) / local_w
            centered = x - local_mean
            local_m2 = (centered * w[:, None]).T @ centered
            if state_w == 0:
                state_w, state_mean, state_m2 = local_w, local_mean, local_m2
            else:
                total = state_w + local_w
                delta = local_mean - state_mean
                state_m2 = state_m2 + local_m2 + np.outer(delta, delta) * (state_w * local_w / total)
                state_mean = state_mean + delta * (local_w / total)
                state_w = total
        if state_w <= 0:
            return np.zeros((3 if include_target else 2, 3 if include_target else 2))
        return state_m2 / state_w

    def _weighted_prediction_corr(rows, candidate):
        values = []
        for target in range(2):
            cov = _weighted_covariance(rows, target, candidate, include_target=False)
            denom = np.sqrt(max(cov[0, 0] * cov[1, 1], 0.0))
            values.append(float(cov[0, 1] / denom) if denom > 1e-12 else 0.0)
        return values

    def _partial_target_corr(rows, candidate):
        values = []
        for target in range(2):
            cov = _weighted_covariance(rows, target, candidate, include_target=True)
            var_y, var_a, var_b = cov[0, 0], cov[1, 1], cov[2, 2]
            if min(var_y, var_a, var_b) <= 1e-12:
                values.append(0.0)
                continue
            cov_ya, cov_yb, cov_ab = cov[0, 1], cov[0, 2], cov[1, 2]
            partial_cov = cov_yb - cov_ya * cov_ab / var_a
            partial_var_y = max(var_y - cov_ya * cov_ya / var_a, 0.0)
            partial_var_b = max(var_b - cov_ab * cov_ab / var_a, 0.0)
            denom = np.sqrt(partial_var_y * partial_var_b)
            values.append(float(partial_cov / denom) if denom > 1e-12 else 0.0)
        return values

    def _choose_lambda(calibration_rows, names):
        scored = [(_score(calibration_rows, names, lam)["wp"], lam) for lam in LAMBDAS]
        best = max(value for value, _ in scored)
        # Preregistered deterministic tie: among values within 1e-8, take the smaller lambda.
        return min(lam for value, lam in scored if best - value <= 1e-8)

    def _best_single(calibration_rows):
        options = []
        for order, name in enumerate(CANDIDATE_NAMES):
            for lam in LAMBDAS:
                value = _score(calibration_rows, (name,), lam)["wp"]
                options.append((value, -order, -lam, name, lam))
        value, _, _, name, lam = max(options)
        return (name,), lam, value

    # One row in each map is a whole connected validation chain; no chain is split.
    pure_rows = PURE_FOLD0
    pure_chains_in_row_order = CHAIN_BY_SEQUENCE[pure_rows]
    outer_in_row_order = OUTER_BY_SEQUENCE
    strategy_names = ("anchor", "weak_equal", "lowest_correlation", "conditional_signal", "best_single")
    selected_by_fold = {}
    params_by_fold = {}
    fold_scores = {name: [] for name in strategy_names}
    chain_moments = {name: {} for name in strategy_names}

    for outer_fold in range(3):
        test_rows = pure_rows[outer_in_row_order == outer_fold]
        calibration_rows = pure_rows[outer_in_row_order != outer_fold]
        correlations = {
            name: _weighted_prediction_corr(calibration_rows, name)
            for name in CANDIDATE_NAMES
        }
        lowest = min(
            CANDIDATE_NAMES,
            key=lambda name: (float(np.mean(np.abs(correlations[name]))), CANDIDATE_NAMES.index(name)),
        )
        partials = {
            name: _partial_target_corr(calibration_rows, name)
            for name in CANDIDATE_NAMES
        }
        signal_set = tuple(
            name for name in CANDIDATE_NAMES
            if float(np.mean(partials[name])) > 0.0
        )
        best_single_names, best_single_lambda, best_single_calibration_wp = _best_single(calibration_rows)
        if signal_set:
            conditional_lambda = _choose_lambda(calibration_rows, signal_set)
        else:
            conditional_lambda = 0.0
        params = {
            "anchor": ((), 0.0),
            "weak_equal": (CANDIDATE_NAMES, _choose_lambda(calibration_rows, CANDIDATE_NAMES)),
            "lowest_correlation": ((lowest,), _choose_lambda(calibration_rows, (lowest,))),
            "conditional_signal": (signal_set, conditional_lambda),
            "best_single": (best_single_names, best_single_lambda),
        }
        selected_by_fold[str(outer_fold)] = {
            "lowest_correlation_member": lowest,
            "prediction_correlations_by_target": correlations,
            "conditional_signal_members": list(signal_set),
            "partial_target_correlations_by_target": partials,
            "best_single_member": best_single_names[0],
            "best_single_lambda": best_single_lambda,
            "best_single_calibration_wp": best_single_calibration_wp,
            "calibration_groups": int(len(calibration_rows)),
            "evaluation_groups": int(len(test_rows)),
        }
        params_by_fold[str(outer_fold)] = {
            name: {"members": list(members), "lambda": lam}
            for name, (members, lam) in params.items()
        }

        for strategy, (members, lam) in params.items():
            # Metrics for each chain are accumulated independently with the
            # official centered-moment implementation, then merged for OOF WP.
            heldout_chains = np.unique(pure_chains_in_row_order[outer_in_row_order == outer_fold])
            for chain in heldout_chains:
                rows = test_rows[pure_chains_in_row_order[outer_in_row_order == outer_fold] == chain]
                acc = GlobalAccumulator()
                for start in range(0, len(rows), 4):
                    part = rows[start : start + 4]
                    acc.add(Y[part], _blend(part, members, lam), OFFICIAL_MASK[part])
                chain_moments[strategy][int(chain)] = acc._moments.copy()
            fold_state = np.zeros((2, 6), dtype=np.float64)
            for chain in heldout_chains:
                moments = chain_moments[strategy][int(chain)]
                for target in range(2):
                    fold_state[target] = _merge_moments(fold_state[target], moments[target])
            fold_wp = float(np.mean([_wp_from_moments(fold_state[k]) for k in range(2)]))
            fold_scores[strategy].append(fold_wp)

    def _score_from_chain_set(moment_map, selected_chains):
        merged = np.zeros((2, 6), dtype=np.float64)
        for chain in selected_chains:
            for target in range(2):
                merged[target] = _merge_moments(merged[target], moment_map[int(chain)][target])
        return float(np.mean([_wp_from_moments(merged[k]) for k in range(2)]))

    all_oof_chains = sorted(map(int, PURE_CHAIN_IDS))
    oof_scores = {
        name: _score_from_chain_set(chain_moments[name], all_oof_chains)
        for name in strategy_names
    }
    anchor_score = oof_scores["anchor"]

    # Paired bootstrap over held-out chains. Each chain contributes official
    # centered moments from its once-held-out prediction; calibration is not refit.
    rng_boot = np.random.default_rng(PARTITION_SEED + 1)
    bootstrap_replicates = 2000
    bootstrap = {}
    for candidate in ("weak_equal", "lowest_correlation", "conditional_signal", "best_single"):
        delta_vs_anchor = []
        delta_vs_single = []
        for _ in range(bootstrap_replicates):
            sampled = rng_boot.choice(all_oof_chains, size=len(all_oof_chains), replace=True)
            candidate_wp = _score_from_chain_set(chain_moments[candidate], sampled)
            anchor_wp = _score_from_chain_set(chain_moments["anchor"], sampled)
            single_wp = _score_from_chain_set(chain_moments["best_single"], sampled)
            delta_vs_anchor.append(candidate_wp - anchor_wp)
            delta_vs_single.append(candidate_wp - single_wp)
        bootstrap[candidate] = {
            "delta_vs_anchor_95pct": [float(v) for v in np.quantile(delta_vs_anchor, [0.025, 0.975])],
            "delta_vs_best_single_95pct": [float(v) for v in np.quantile(delta_vs_single, [0.025, 0.975])],
            "conditional_on_outer_calibration_choices": True,
            "replicates": bootstrap_replicates,
        }

    # Standalone scores and pairwise covariance are descriptive diagnostics on
    # exposed pure fold-0 chains; they do not choose the strategies above.
    standalone = {"c1_pair": _score(pure_rows, (), 0.0)["wp"]}
    for name in CANDIDATE_NAMES:
        standalone[name] = _score(pure_rows, (name,), 1.0)["wp"]
    correlation_labels = ("C1 pair", "ab12", "ab48", "baseline × 0.25")
    correlation_names = (None, *CANDIDATE_NAMES)
    correlation_matrix = np.eye(len(correlation_names), dtype=np.float64)
    for i in range(len(correlation_names)):
        for j in range(i + 1, len(correlation_names)):
            left_name, right_name = correlation_names[i], correlation_names[j]
            target_corrs = []
            for target in range(2):
                state_w = 0.0
                mean = None
                m2 = None
                for start in range(0, len(pure_rows), 4):
                    part = pure_rows[start : start + 4]
                    y = np.clip(Y[part, :, target].astype(np.float64), -2, 2)
                    selected = OFFICIAL_MASK[part] & (np.abs(y) > 0)
                    a = _anchor(part)[:, :, target].astype(np.float64) if left_name is None else (
                        np.float32(CANDIDATE_SCALE[left_name]) * np.asarray(PRED[left_name][part], np.float32)
                    )[:, :, target].astype(np.float64)
                    b = _anchor(part)[:, :, target].astype(np.float64) if right_name is None else (
                        np.float32(CANDIDATE_SCALE[right_name]) * np.asarray(PRED[right_name][part], np.float32)
                    )[:, :, target].astype(np.float64)
                    x = np.column_stack([np.clip(a[selected], -2, 2), np.clip(b[selected], -2, 2)])
                    w = np.abs(y[selected])
                    if not len(w):
                        continue
                    local_w = float(w.sum())
                    local_mean = np.sum(x * w[:, None], axis=0) / local_w
                    centered = x - local_mean
                    local_m2 = (centered * w[:, None]).T @ centered
                    if state_w == 0:
                        state_w, mean, m2 = local_w, local_mean, local_m2
                    else:
                        total = state_w + local_w
                        delta = local_mean - mean
                        m2 = m2 + local_m2 + np.outer(delta, delta) * (state_w * local_w / total)
                        mean = mean + delta * (local_w / total)
                        state_w = total
                if state_w <= 0 or min(m2[0, 0], m2[1, 1]) <= 1e-12:
                    target_corrs.append(0.0)
                else:
                    target_corrs.append(float(m2[0, 1] / np.sqrt(m2[0, 0] * m2[1, 1])))
            correlation_matrix[i, j] = correlation_matrix[j, i] = float(np.mean(target_corrs))

    result = {
        "experiment_id": "E-514",
        "status": "completed_exploratory_crossfit",
        "created_utc": datetime.now().astimezone().isoformat(),
        "metric_id": "global_wp_clip2_v1",
        "source_manifest": "SOURCE_MANIFEST_E514.json",
        "source_manifest_sha256": MANIFEST_SHA256,
        "data_exposure": "Previously inspected fold-0 groups; exploratory, not a fresh confirmation or public forecast.",
        "pure_fold0_groups": int(len(PURE_FOLD0)),
        "pure_fold0_chains": int(len(PURE_CHAIN_IDS)),
        "outer_fold_group_counts": [int(np.sum(OUTER_BY_SEQUENCE == i)) for i in range(3)],
        "anchor": "equal raw mean of bd26_c1v2_s3_f0 and bd26_c1v2_s4_f0",
        "weak_candidate_pool": list(CANDIDATE_NAMES),
        "lambda_grid": list(LAMBDAS),
        "selection_by_outer_fold": selected_by_fold,
        "selected_endpoint_by_outer_fold": params_by_fold,
        "outer_fold_oof_wp": fold_scores,
        "concatenated_once_held_out_oof_wp": oof_scores,
        "delta_vs_anchor": {
            name: float(oof_scores[name] - anchor_score) for name in strategy_names if name != "anchor"
        },
        "paired_chain_bootstrap": bootstrap,
        "standalone_full_pure_fold0_wp_descriptive_only": standalone,
        "mean_official_weighted_prediction_correlation_descriptive_only": {
            "labels": list(correlation_labels),
            "matrix": correlation_matrix.tolist(),
            "targets_averaged_equally": True,
        },
        "bootstrap_note": "Conditional on fixed outer-fold choices; does not include member/weight selection, model training, or public-transfer uncertainty.",
        "new_public_gain_verified": False,
        "submission_ready": False,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    # Human-readable figures live beside this lineage's research result.
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    image = ax.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlation_labels)), correlation_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(correlation_labels)), correlation_labels)
    for i in range(len(correlation_labels)):
        for j in range(len(correlation_labels)):
            ax.text(j, i, f"{correlation_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Prediction correlation (descriptive, exposed pure fold 0)")
    fig.colorbar(image, ax=ax, label="Official-weighted Pearson")
    fig.savefig(PLOT_DIR / "prediction_correlation.png", dpi=180)
    plt.close(fig)

    labels = ["weak equal", "least correlated", "conditional signal", "best single"]
    keys = ["weak_equal", "lowest_correlation", "conditional_signal", "best_single"]
    deltas = [oof_scores[_strategy_key] - anchor_score for _strategy_key in keys]
    lower = [bootstrap[_strategy_key]["delta_vs_anchor_95pct"][0] for _strategy_key in keys]
    upper = [bootstrap[_strategy_key]["delta_vs_anchor_95pct"][1] for _strategy_key in keys]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    yloc = np.arange(len(keys))
    ax.errorbar(
        deltas,
        yloc,
        xerr=[np.asarray(deltas) - np.asarray(lower), np.asarray(upper) - np.asarray(deltas)],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(yloc, labels)
    ax.set_xlabel("OOF global WP delta vs C1 pair (paired chain bootstrap 95% interval)")
    ax.set_title("E-514 once-held-out chain comparison")
    fig.savefig(PLOT_DIR / "oof_delta_vs_anchor.png", dpi=180)
    plt.close(fig)

    frequency = {name: {candidate: 0 for candidate in CANDIDATE_NAMES} for name in (
        "lowest_correlation", "conditional_signal", "best_single"
    )}
    lambda_counts = {name: {} for name in strategy_names}
    for outer, selection in selected_by_fold.items():
        frequency["lowest_correlation"][selection["lowest_correlation_member"]] += 1
        for candidate in selection["conditional_signal_members"]:
            frequency["conditional_signal"][candidate] += 1
        frequency["best_single"][selection["best_single_member"]] += 1
        for strategy, params in params_by_fold[outer].items():
            value = f"{params['lambda']:.2f}"
            lambda_counts[strategy][value] = lambda_counts[strategy].get(value, 0) + 1
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    strategies = list(frequency)
    matrix = np.asarray([[frequency[strategy][candidate] for candidate in CANDIDATE_NAMES] for strategy in strategies])
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=3)
    ax.set_xticks(range(len(CANDIDATE_NAMES)), [CANDIDATE_LABELS[n] for n in CANDIDATE_NAMES], rotation=25, ha="right")
    ax.set_yticks(range(len(strategies)), strategies)
    for i in range(len(strategies)):
        for j in range(len(CANDIDATE_NAMES)):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="white")
    ax.set_title("Member-selection frequency across outer chain folds")
    fig.colorbar(im, ax=ax, label="Number of folds")
    fig.savefig(PLOT_DIR / "member_selection_frequency.png", dpi=180)
    plt.close(fig)

    lines = [
        "# E-514 result — weak portfolio and conditional-signal screen",
        "",
        "Status: completed exploratory chain cross-fit on previously inspected fold-0 groups.",
        "This is not fresh confirmation, a public-score forecast, or a submission decision.",
        "",
        f"Source manifest SHA-256: `{MANIFEST_SHA256}`.",
        f"Pure fold-0 sample: {len(PURE_FOLD0)} groups across {len(PURE_CHAIN_IDS)} connected chains; each chain was held out once.",
        "",
        "| Strategy | OOF global WP | Delta vs C1 pair | Chain-bootstrap 95% interval vs anchor |",
        "|---|---:|---:|---:|",
    ]
    for _strategy_key, label in zip(keys, labels, strict=True):
        interval = bootstrap[_strategy_key]["delta_vs_anchor_95pct"]
        lines.append(
            f"| {label} | {oof_scores[_strategy_key]:.6f} | {oof_scores[_strategy_key] - anchor_score:+.6f} | [{interval[0]:+.6f}, {interval[1]:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "Member selection and lambda were fit on the other two outer chain folds. The equal weak mean is `ab12`, `ab48`, and `baseline×0.25` averaged at the prediction level. The official metric is used for every calibration and evaluation comparison.",
            "",
            "The paired bootstrap is conditional on fold-specific member/weight choices. It does not account for candidate selection, model training, or public-transfer uncertainty. The calibration groups and the candidate pool have prior exposure.",
            "",
            "The correlation and standalone-score plots are descriptive summaries of the exposed fold. They do not choose a post-hoc winner.",
            "",
            "Figures: [prediction correlation](plots/prediction_correlation.png), [OOF deltas](plots/oof_delta_vs_anchor.png), [member selection](plots/member_selection_frequency.png).",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    mo.md(f"E-514 complete. Source manifest: `{MANIFEST_SHA256}`. No public score or submission is inferred.")
    return


if __name__ == "__main__":
    app.run()
