from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn.prediction_safety import (  # noqa: E402
    PREDICTION_CACHE_SCHEMA,
    prepare_scored_prediction_arrays,
    validate_full_teacher_cache,
    validate_teacher_rows,
)

BUGFIX = (
    Path(__file__).resolve().parents[1]
    / "research/campaign_2026-10-17/research_reset_r1/bugfix/deploy"
)
PREDICTOR = (
    BUGFIX / "WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/predict_any_env.py"
)
PLEW_TRAINER = (
    BUGFIX / "WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/train_rnn_plew19.py"
)
RS_TRAINER = (
    BUGFIX / "WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/train_rnn_rs.py"
)
XEVAL = BUGFIX / "xteam/xeval.py"


def load_ast_object(path: Path, name: str, namespace: dict[str, object]) -> object:
    """Load one real function or class from a frozen deployable source file."""
    tree = ast.parse(path.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def make_sidecar(
    name: str,
    split: str,
    row_groups: list[int],
    shape: tuple[int, ...],
    complete: bool,
    identity: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata = {
        "schema": PREDICTION_CACHE_SCHEMA,
        "split": split,
        "prediction_name": name,
        "complete": complete,
        "row_groups": row_groups,
        "shape": list(shape),
        "dtype": "float32",
    }
    if identity is not None:
        metadata.update(identity)
    return metadata


def write_cache_identity(root: Path) -> dict[str, object]:
    """Write fixture checkpoint/source files and return matching ancestry fields."""
    checkpoint_path = root / "artifacts/runs/teacher/model_ep1.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"fixture checkpoint")
    source_paths = {
        "predictor": root / "research/iterations/2026-10-08-shift/plew19/predict_any_env.py",
        "trainer": root / "research/iterations/2026-10-08-shift/plew19/train_rnn_plew19.py",
    }
    sources = {}
    for name, path in source_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {name} source\n")
        sources[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    preprocessing = {"lookahead": [], "zero_groups": "i1"}
    serialized = json.dumps(preprocessing, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "preprocessing": preprocessing,
        "preprocessing_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "source_sha256": sources,
        "lookahead": [],
        "causal": True,
        "privileged": False,
    }


def test_full_cache_helper_requires_exact_complete_membership() -> None:
    groups = [3, 8]
    predictions = np.zeros((2, 20_000, 2), dtype=np.float32)
    sidecar = make_sidecar("pred_teacher_valid.npy", "valid", groups, predictions.shape, True)

    assert (
        validate_full_teacher_cache(
            predictions,
            sidecar,
            split="valid",
            row_groups=groups,
            prediction_name="pred_teacher_valid.npy",
        )
        is predictions
    )
    with pytest.raises(ValueError, match="complete"):
        validate_full_teacher_cache(
            predictions,
            {**sidecar, "complete": False},
            split="valid",
            row_groups=groups,
            prediction_name="pred_teacher_valid.npy",
        )
    with pytest.raises(ValueError, match="row_groups"):
        validate_full_teacher_cache(
            predictions,
            {**sidecar, "row_groups": [3]},
            split="valid",
            row_groups=groups,
            prediction_name="pred_teacher_valid.npy",
        )


def test_teacher_row_helper_rejects_only_requested_nonfinite_rows() -> None:
    predictions = np.zeros((2, 20_000, 2), dtype=np.float32)
    predictions[1, 0, 0] = np.nan
    rows = validate_teacher_rows(predictions, [0])
    assert rows.shape == (1, 20_000, 2)
    with pytest.raises(ValueError, match="nonfinite"):
        validate_teacher_rows(predictions, [1])


def test_scored_array_helper_ignores_unscored_nans_and_rejects_selected_nans() -> None:
    y = np.full((120, 2), np.nan)
    p = np.full((120, 2), np.nan)
    mask = np.zeros((120, 2), dtype=bool)
    mask[100:, :] = True
    y[mask] = np.linspace(-1.0, 1.0, mask.sum())
    p[mask] = np.linspace(1.0, -1.0, mask.sum())

    safe_y, safe_p, weights, selected = prepare_scored_prediction_arrays(y, p, mask)
    assert np.isfinite(safe_y).all() and np.isfinite(safe_p).all()
    assert np.array_equal(weights, np.abs(safe_y) * selected)
    assert np.all(safe_y[~selected] == 0) and np.all(safe_p[~selected] == 0)

    p[103, 0] = np.nan
    with pytest.raises(ValueError, match="selected"):
        prepare_scored_prediction_arrays(y, p, mask)


def test_predictor_restores_saved_feature_globals_and_cache_coverage() -> None:
    restore = load_ast_object(
        PREDICTOR, "restore_checkpoint_settings", {"torch": torch, "Path": Path}
    )
    trainer = SimpleNamespace(
        zero_group_cols=lambda value: [52, 53] if value == "i1" else [],
        load_propensity=lambda path, device: (path, device),
    )
    restore(
        trainer,
        {
            "lookahead": "8,16",
            "state_reset": 1_000,
            "sentinel_mode": "indicator",
            "a_mode": "demean",
            "domain_flag": True,
            "zero_groups": "i1",
            "raw_only": True,
            "heuristics": True,
            "quant_feats": True,
            "quant_i1": True,
            "online_norm": True,
            "prop_input": "prop-test",
        },
        {"readout": None, "readout_tilted": False},
        Path("/tmp/root"),
        torch.device("cpu"),
    )
    assert trainer.LOOKAHEAD == (8, 16)
    assert trainer.STATE_RESET == 1_000
    assert trainer.SENTINEL_MODE == "indicator"
    assert trainer.A_MODE == "demean"
    assert trainer.DOMAIN_FLAG is True
    assert trainer.ZERO_COLS.tolist() == [52, 53]
    assert trainer.RAW_ONLY is True
    assert trainer.HEURISTICS is True
    assert trainer.QUANT is True and trainer.QUANT_I1 is True
    assert trainer.ONLINE_NORM is True
    assert trainer.INPUT_PROPENSITY[0] == Path("/tmp/root/artifacts/runs/prop-test/model.pt")

    prep = load_ast_object(PREDICTOR, "preprocessing_settings", {})
    settings = prep(trainer, {"zero_groups": "i1", "prop_input": "prop-test"})
    assert settings["lookahead"] == [8, 16]
    causality = load_ast_object(PREDICTOR, "validate_export_causality", {})
    assert causality((8, 16), True) is False
    with pytest.raises(ValueError, match="privileged"):
        causality((8,), False)
    with pytest.raises(ValueError, match="requires checkpoint lookahead"):
        causality((), True)
    validate_batch = load_ast_object(PREDICTOR, "validate_prediction_batch", {"np": np})
    assert validate_batch(np.zeros((1, 20_000, 2))).shape == (1, 20_000, 2)
    malformed_batch = np.zeros((1, 20_000, 3))
    with pytest.raises(ValueError, match="shape"):
        validate_batch(malformed_batch)
    nonfinite_batch = np.zeros((1, 20_000, 2))
    nonfinite_batch[0, 10, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_batch(nonfinite_batch)

    positions = load_ast_object(PREDICTOR, "prediction_positions", {"np": np})
    folds = {
        "split": np.array(["valid", "valid", "train"]),
        "row_group": np.array([10, 11, 0]),
        "fold": np.array([0, 1, 0]),
    }
    assert positions(np.array([10, 11]), "valid", folds, False).tolist() == [0]
    assert positions(np.array([10, 11]), "valid", folds, True).tolist() == [0, 1]
    with pytest.raises(ValueError, match="explicit full"):
        positions(np.array([0]), "train", folds, False)

    metadata = load_ast_object(
        PREDICTOR, "prediction_metadata", {"np": np, "Path": Path, "hashlib": hashlib, "json": json}
    )
    kwargs = {
        "checkpoint_path": Path("artifacts/runs/run/model_ep1.pt"),
        "checkpoint_sha256": "a" * 64,
        "preprocessing": {"lookahead": [], "zero_groups": "i1"},
        "source_sha256": {"predictor": {"sha256": "b" * 64}, "trainer": {"sha256": "c" * 64}},
        "privileged": False,
    }
    partial = metadata(
        Path("pred_x_valid.npy"), "valid", np.array([10, 11]), np.array([0]), False, **kwargs
    )
    full = metadata(
        Path("pred_x_valid.npy"), "valid", np.array([10, 11]), np.array([0, 1]), True, **kwargs
    )
    assert partial["complete"] is False and partial["row_groups"] == [10]
    assert full["complete"] is True and full["row_groups"] == [10, 11]
    assert full["causal"] is True and full["checkpoint_sha256"] == "a" * 64
    assert (
        full["preprocessing_sha256"]
        == hashlib.sha256(
            json.dumps(
                kwargs["preprocessing"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )


def _load_xeval() -> tuple[object, object]:
    namespace: dict[str, object] = {"np": np, "Callable": Callable, "Iterable": Iterable}
    mom = load_ast_object(XEVAL, "mom", namespace)
    wp = load_ast_object(XEVAL, "wp", namespace)
    return mom, wp


def _expected_wp(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> float:
    y = np.clip(targets.astype(np.float64), -2, 2)
    p = np.clip(predictions.astype(np.float64), -2, 2)
    selected = np.broadcast_to(mask[:, :, None], y.shape).copy()
    selected[:, :99] = False
    weights = np.abs(y) * selected
    total = weights.sum(axis=(0, 1))
    sy = (weights * y).sum(axis=(0, 1))
    sp = (weights * p).sum(axis=(0, 1))
    syy = (weights * y * y).sum(axis=(0, 1))
    spp = (weights * p * p).sum(axis=(0, 1))
    syp = (weights * y * p).sum(axis=(0, 1))
    corr = (syp - sy * sp / total) / np.sqrt((syy - sy * sy / total) * (spp - sp * sp / total))
    return float(corr.mean())


def test_actual_xeval_moments_preserve_finite_score_and_ignore_unscored_nans() -> None:
    rng = np.random.default_rng(12)
    targets = rng.normal(size=(2, 120, 2)).astype(np.float64)
    predictions = (0.25 * targets + rng.normal(scale=0.5, size=targets.shape)).astype(np.float64)
    mask = np.zeros((2, 120), dtype=bool)
    mask[:, 99:] = True
    namespace: dict[str, object] = {
        "np": np,
        "Callable": Callable,
        "Iterable": Iterable,
        "Y": targets,
        "MASK": mask,
        "pos": {20: 0, 10: 1},
    }
    mom = load_ast_object(XEVAL, "mom", namespace)
    wp = load_ast_object(XEVAL, "wp", namespace)

    predictions_by_group = {20: predictions[0].copy(), 10: predictions[1].copy()}
    finite_moments = mom([20, 10], {"m": lambda group: predictions_by_group[group]})["m"]
    score = wp(finite_moments.sum(axis=0))
    assert score == pytest.approx(_expected_wp(targets, predictions, mask), abs=1e-14)

    selected = np.broadcast_to(mask[:, :, None], targets.shape).copy()
    selected[:, :99] = False
    targets_with_holes = targets.copy()
    predictions_with_holes = predictions.copy()
    targets_with_holes[~selected] = np.nan
    predictions_with_holes[~selected] = np.nan
    namespace["Y"] = targets_with_holes
    namespace["MASK"] = mask
    nan_moments = mom(
        [20, 10], {"m": lambda group: predictions_with_holes[0 if group == 20 else 1]}
    )["m"]
    assert np.array_equal(nan_moments, finite_moments)

    predictions_by_group[20][101, 0] = np.nan
    with pytest.raises(ValueError, match="selected predictions"):
        mom([20, 10], {"m": lambda group: predictions_by_group[group]})
    predictions_by_group[20][101, 0] = 0.0
    namespace["Y"] = targets.copy()
    namespace["Y"][0, 101, 0] = np.nan
    with pytest.raises(ValueError, match="selected targets"):
        mom([20, 10], {"m": lambda group: predictions_by_group[group]})
    namespace["Y"] = targets
    with pytest.raises(ValueError, match="shape"):
        mom([20], {"m": lambda group: np.zeros((120, 1))})
    with pytest.raises(ValueError, match="duplicates"):
        mom([20, 20], {"m": lambda group: predictions_by_group[group]})
    with pytest.raises(ValueError, match="missing"):
        mom([99], {"m": lambda group: predictions_by_group[20]})


class _FakeSequenceFile:
    def __init__(self, path: Path):
        self.path = path

    def read(self, group: int) -> SimpleNamespace:
        del group
        steps = 20_000
        return SimpleNamespace(
            features=np.zeros((steps, 1), dtype=np.float32),
            targets=np.zeros((steps, 2), dtype=np.float32),
            scored=np.zeros(steps, dtype=bool),
        )


@pytest.mark.parametrize("trainer_path", [PLEW_TRAINER, RS_TRAINER])
def test_actual_trainer_consumers_require_complete_sidecars_and_finite_rows(
    tmp_path: Path, trainer_path: Path
) -> None:
    namespace: dict[str, object] = {
        "Dataset": object,
        "ROOT": tmp_path,
        "np": np,
        "torch": torch,
        "json": json,
        "hashlib": hashlib,
        "Path": Path,
        "SequenceFile": _FakeSequenceFile,
        "HEURISTICS": False,
        "DOMAIN_FLAG": False,
        "FAMILY_TEACHERS": (),
        "RS_RB": False,
    }
    for helper_name in ("_sha256_file", "_root_file", "_validate_prediction_identity"):
        load_ast_object(trainer_path, helper_name, namespace)
    load_ast_object(trainer_path, "validate_full_prediction_cache", namespace)
    sequences_type = load_ast_object(trainer_path, "Sequences", namespace)
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "train_targets.npz", row_group=np.array([7], dtype=np.int64))
    prediction_path = cache / "pred_teacher_train.npy"
    predictions = np.zeros((1, 20_000, 2), dtype=np.float32)
    np.save(prediction_path, predictions)
    identity = write_cache_identity(tmp_path)
    metadata = make_sidecar(prediction_path.name, "train", [7], predictions.shape, True, identity)
    (cache / "pred_teacher_train.json").write_text(json.dumps(metadata))

    dataset = sequences_type("train", [7], teachers=[("teacher", 1.0)])
    _, targets, _ = dataset[0]
    assert tuple(targets.shape) == (20_000, 4)

    family_path = cache / "pred_family_train.npy"
    family_bad = np.zeros((1, 20_000, 3), dtype=np.float32)
    np.save(family_path, family_bad)
    family_metadata = make_sidecar(family_path.name, "train", [7], family_bad.shape, True, identity)
    (cache / "pred_family_train.json").write_text(json.dumps(family_metadata))
    namespace["FAMILY_TEACHERS"] = ("family",)
    dataset = sequences_type("train", [7], teachers=[("teacher", 1.0)])
    with pytest.raises(ValueError, match="shape"):
        dataset[0]
    namespace["FAMILY_TEACHERS"] = ()

    checkpoint = tmp_path / str(identity["checkpoint_path"])
    checkpoint.write_bytes(b"changed checkpoint")
    dataset = sequences_type("train", [7], teachers=[("teacher", 1.0)])
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        dataset[0]
    checkpoint.write_bytes(b"fixture checkpoint")

    sources = identity["source_sha256"]
    assert isinstance(sources, dict)
    predictor_source = tmp_path / str(sources["predictor"]["path"])
    predictor_source.write_text("changed predictor source\n")
    dataset = sequences_type("train", [7], teachers=[("teacher", 1.0)])
    with pytest.raises(ValueError, match="predictor source SHA-256"):
        dataset[0]
    predictor_source.write_text("fixture predictor source\n")

    changed_metadata = {**metadata, "preprocessing": {"lookahead": [], "zero_groups": "book"}}
    (cache / "pred_teacher_train.json").write_text(json.dumps(changed_metadata))
    dataset = sequences_type("train", [7], teachers=[("teacher", 1.0)])
    with pytest.raises(ValueError, match="preprocessing SHA-256"):
        dataset[0]
    (cache / "pred_teacher_train.json").write_text(json.dumps(metadata))

    (cache / "pred_teacher_train.json").write_text(json.dumps({**metadata, "complete": False}))
    dataset = sequences_type("train", [7], teachers=[("teacher", 1.0)])
    with pytest.raises(ValueError, match="complete"):
        dataset[0]

    malformed = np.zeros((1, 20_000, 3), dtype=np.float32)
    np.save(prediction_path, malformed)
    (cache / "pred_teacher_train.json").write_text(
        json.dumps({**metadata, "shape": list(malformed.shape)})
    )
    dataset = sequences_type("train", [7], teachers=[("teacher", 1.0)])
    with pytest.raises(ValueError, match="shape"):
        dataset[0]

    np.save(prediction_path, predictions)
    (cache / "pred_teacher_train.json").write_text(json.dumps(metadata))
    predictions[0, 100, 1] = np.nan
    np.save(prediction_path, predictions)
    (cache / "pred_teacher_train.json").write_text(json.dumps(metadata))
    dataset = sequences_type("train", [7], teachers=[("teacher", 1.0)])
    with pytest.raises(ValueError, match="nonfinite"):
        dataset[0]
