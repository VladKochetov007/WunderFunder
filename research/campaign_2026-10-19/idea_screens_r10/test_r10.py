"""Focused synthetic checks for the fixed R10 metadata and geometry contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from idea_screens_r10.geometry import (
    classify_at,
    classify_window,
    query_rows,
)
from idea_screens_r10.panel_io import InputBinding, map_fit_arrays, validate_metadata_objects
from idea_screens_r10.run_probe import _scoped_cgroup_memory_path, smoke_imports

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _metadata() -> tuple[dict[str, object], dict[str, object]]:
    sample = json.loads((FIXTURES / "R3_SAMPLE.json").read_bytes())
    membership = json.loads((FIXTURES / "R3_MEMBERSHIP.json").read_bytes())
    return sample, membership


def test_fixed_metadata_fixture_and_component_relations() -> None:
    sample, membership = _metadata()
    panel = validate_metadata_objects(sample, membership)
    assert len(panel.fit_records) == 64
    assert len(panel.eval_component_order) == 32
    assert not set(panel.fit_component_order) & set(panel.eval_component_order)
    assert panel.fit_component_order[:32] == tuple(sample["selection"]["fit_low32_components"])

    broken_sample, broken_membership = _metadata()
    broken_sample["records"][0]["role"] = "eval"
    broken_membership["records"][0]["role"] = "eval"
    with pytest.raises(ValueError, match="fit records"):
        validate_metadata_objects(broken_sample, broken_membership)


def test_three_exhaustive_categories_and_degenerate_hulls() -> None:
    triangle = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), (86, 1))[:256]
    assert classify_window(triangle, np.array([0.1, 0.1])).category == "hull_inside"
    assert classify_window(triangle, np.array([0.9, 0.9])).category == "box_inside_hull_outside"
    assert classify_window(triangle, np.array([1.1, 0.1])).category == "box_outside"

    point = np.repeat([[2.0, 4.0]], 256, axis=0)
    assert classify_window(point, np.array([2.0, 4.0])).hull_kind == "point"
    assert classify_window(point, np.array([2.0, 4.0])).category == "hull_inside"

    line = np.column_stack((np.linspace(0.0, 1.0, 256), np.linspace(0.0, 1.0, 256)))
    assert classify_window(line, np.array([0.5, 0.5])).hull_kind == "line"
    assert classify_window(line, np.array([0.5, 0.5000001])).category == "box_inside_hull_outside"


def test_fixed_tolerance_affine_invariance_and_past_only_query() -> None:
    history = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), (86, 1))[:256]
    query = np.array([0.9, 0.9])
    original = classify_window(history, query)
    transform = np.array([[-4.0, 0.0], [0.0, 2.0]])
    shifted = np.array([7.0, -1.0])
    assert classify_window(history @ transform + shifted, query @ transform + shifted) == original

    rows = np.zeros((600, 2), dtype=np.float64)
    rows[143:399] = history
    rows[399] = query
    before = classify_at(rows, 399)
    assert before.category == "box_inside_hull_outside"
    rows[399] = np.array([0.1, 0.1])
    assert classify_at(rows, 399).category == "hull_inside"
    rows[399] = query
    rows[399 + 1 :] = 1000.0
    assert classify_at(rows, 399) == before

    scored = np.zeros(20_000, dtype=np.bool_)
    scored[99:] = True
    selected = query_rows(scored)
    assert selected.size == 493
    assert selected[0] == 280 and selected[-1] == 19_960


def test_scoped_cgroup_resolution_rejects_root_and_wrong_unit() -> None:
    valid = "0::/wnn-campaign.slice/computer-vision-homework-eda19-r10-v1.service/job\n"
    assert str(_scoped_cgroup_memory_path(valid)).endswith(
        "/wnn-campaign.slice/computer-vision-homework-eda19-r10-v1.service/job/memory.current"
    )
    for invalid in (
        "0::/\n",
        "0::/other.slice/computer-vision-homework-eda19-r10-v1.service\n",
        "0::/wnn-campaign.slice/other.service\n",
        "0::/wnn-campaign.slice/../other.service\n",
    ):
        with pytest.raises(RuntimeError):
            _scoped_cgroup_memory_path(invalid)


def test_array_reader_maps_only_one_fit_x_and_scored_mask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    x_path, scored_path = tmp_path / "train_1_x.npy", tmp_path / "train_1_scored.npy"
    np.save(x_path, np.zeros((20_000, 112), dtype=np.float32))
    scored = np.zeros(20_000, dtype=np.bool_)
    scored[99:] = True
    np.save(scored_path, scored)
    opened: list[str] = []
    original_load = np.load

    def guarded_load(file: object, *args: object, **kwargs: object) -> np.ndarray:
        name = Path(file).name
        if name not in {x_path.name, scored_path.name}:
            raise AssertionError(f"reader attempted an unapproved array load: {name}")
        opened.append(name)
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    binding = InputBinding(
        {"scored_rows": 19_901},
        x_path,
        "0" * 64,
        x_path.stat().st_size,
        scored_path,
        "1" * 64,
        scored_path.stat().st_size,
    )
    arrays = map_fit_arrays(binding)
    assert opened == [x_path.name, scored_path.name]
    assert isinstance(arrays.x, np.memmap) and isinstance(arrays.scored, np.memmap)
    assert not arrays.x.flags.writeable and not arrays.scored.flags.writeable


def test_import_smoke_has_no_array_or_model_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("smoke import attempted an array read")

    monkeypatch.setattr(np, "load", no_load)
    receipt = smoke_imports(ROOT, tmp_path)
    assert receipt["array_reads"] == 0
    assert receipt["new_forbidden_imports"] == []
