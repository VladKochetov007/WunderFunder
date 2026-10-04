from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from feature_cache import extract, frozen_state, selected_rows, validate_lineage  # noqa: E402
from feature_cues import CUE_NAMES, causal_cues, lagged  # noqa: E402
from feature_recipes import head_partition, recipe  # noqa: E402
from feature_screen import assert_outer_separation, digest_file, load_cache, run  # noqa: E402

from wnn.features import event_features_torch  # noqa: E402
from wnn.gmetric import reference_global_wp  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402


def fixture_cache(path: Path, first_component: int) -> None:
    rng = np.random.default_rng(first_component + 21)
    rows = 128
    cue = rng.uniform(-1, 1, (rows, 2)).astype(np.float32)
    state = rng.normal(size=(rows, 8)).astype(np.float32)
    prediction = rng.normal(scale=0.1, size=(rows, 2)).astype(np.float32)
    values = {
        "state": state,
        "pred": prediction,
        "y": cue,
        "fit_weight": np.ones(rows),
        "component": np.repeat(np.arange(4) + first_component, 32),
        "sequence": np.repeat(np.arange(4) + first_component, 32),
        "row": np.tile(np.arange(227, 227 + 32), 4),
        "model_id": np.full(rows, "a" * 64),
    }
    for name in CUE_NAMES:
        values[f"cue__{name}"] = cue
        values[f"placebo__{name}"] = rng.normal(size=(rows, 2)).astype(np.float32)
    np.savez(path, **values)
    metadata = {
        "sha256": digest_file(path),
        "rows": rows,
        "state_width": 8,
        "outer_fold": 0,
        "component_graph_sha256": "b" * 64,
        "role": "fit" if first_component == 0 else "evaluation",
        "members": {"a" * 64: {"fit_components": [100], "checkpoint_sha256": "a" * 64}},
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))


def test_all_cues_are_causal_and_placebo_never_wraps() -> None:
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(400, 112)).astype(np.float32)
    expected = causal_cues(raw)
    raw[200:] *= 100
    actual = causal_cues(raw)
    assert set(actual) == set(CUE_NAMES)
    for name in CUE_NAMES:
        np.testing.assert_array_equal(actual[name][:200], expected[name][:200])
        placebo = lagged(expected[name], 128)
        assert not placebo[:128].any()
        np.testing.assert_array_equal(placebo[227], expected[name][99])


def test_frozen_actual_head_input_reproduces_full_model() -> None:
    torch.manual_seed(7)
    model = RNNModel(211, hidden=4, layers=1).eval()
    raw = np.random.default_rng(3).normal(size=(1200, 112)).astype(np.float32)
    state, prediction = frozen_state(model, raw, "none", "cpu")
    inputs = torch.as_tensor(raw[None])
    with torch.no_grad():
        expected, _ = model(torch.cat((inputs, event_features_torch(inputs)), -1))
        restored = model.head(torch.as_tensor(state))
    np.testing.assert_allclose(prediction, expected[0].numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(prediction, restored.numpy(), rtol=2e-5, atol=2e-6)


def test_separation_and_cache_identity_fail_closed(tmp_path: Path) -> None:
    fit_path, eval_path = tmp_path / "fit.npz", tmp_path / "eval.npz"
    fixture_cache(fit_path, 0)
    fixture_cache(eval_path, 4)
    fit, fm = load_cache(fit_path)
    evaluation, em = load_cache(eval_path)
    assert_outer_separation(fit, evaluation, fm, em)
    with pytest.raises(ValueError, match="overlap"):
        assert_outer_separation(fit, fit, fm, em)
    em["members"]["a" * 64]["fit_components"] = [4]
    with pytest.raises(ValueError, match="protected"):
        assert_outer_separation(fit, evaluation, fm, em)
    fit_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_cache(fit_path)


def test_fixed_screen_reports_all_controls_with_official_metric(tmp_path: Path) -> None:
    fit_path, eval_path = tmp_path / "fit.npz", tmp_path / "eval.npz"
    fixture_cache(fit_path, 0)
    fixture_cache(eval_path, 4)
    output = tmp_path / "screen"
    run(fit_path, eval_path, "target_clocks", "numpy", 0.01, output, bootstrap=100)
    report = json.loads((output / "report.json").read_text())
    with np.load(output / "predictions.npz") as predictions, np.load(eval_path) as data:
        for arm in ("unchanged", "candidate", "placebo", "capacity"):
            expected = reference_global_wp(data["y"], predictions[arm], np.ones(128, dtype=bool))
            assert report["reports"][arm]["wp"] == expected
    assert report["reports"]["candidate_minus_capacity"]["mean_diff"] > 0.5
    assert report["selection"].startswith("none")


def test_ancestry_requires_fixed_epoch_and_complete_bindings() -> None:
    components = {("valid", 1): 10, ("train", 2): 20}
    checkpoint = {
        "epoch": 3,
        "args": {"no_eval_during_fit": True, "checkpoint_policy": "fixed", "checkpoint_epoch": 3},
        "provenance": {"fit_row_groups": {"valid": [1]}},
    }
    assert validate_lineage(checkpoint, {}, components) == {10}
    checkpoint["provenance"]["initial_checkpoint_sha256"] = "x"
    with pytest.raises(ValueError, match="Missing learned ancestor"):
        validate_lineage(checkpoint, {}, components)
    ancestor = {
        "epoch": 3,
        "args": checkpoint["args"],
        "provenance": {"fit_row_groups": {"train": [2]}},
    }
    assert validate_lineage(checkpoint, {"x": ancestor}, components) == {10, 20}


def test_sealed_cpu_gpu_recipe_and_component_partition(tmp_path: Path) -> None:
    # Real source closure is sealed; fixture caches are intentionally synthetic.
    for folder in ("scripts", "src"):
        (tmp_path / folder).symlink_to(ROOT / folder, target_is_directory=True)
    fixture_cache(tmp_path / "fit.npz", 0)
    fixture_cache(tmp_path / "evaluation.npz", 4)
    for backend in ("cpu", "gpu"):
        spec = recipe(tmp_path, "fit.npz", "evaluation.npz", "event_age", backend, "fixture")
        assert spec["kind"] == backend and spec["code_sha256"]
        assert all(len(item["sha256"]) == 64 for item in spec["inputs"])
        assert spec["limits"]["gpu_memory_bytes"] == (0 if backend == "cpu" else 1024**3)
    assert {head_partition(component) for component in range(100)} == {0, 1}


def test_actual_parquet_cache_uses_dense_lags_and_bound_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import feature_cache
    import pyarrow as pa
    import pyarrow.parquet as pq

    from wnn import data

    monkeypatch.setattr(data, "SEQUENCE_LENGTH", 300)
    monkeypatch.setattr(
        feature_cache,
        "load_campaign_split",
        lambda path, root: SimpleNamespace(
            components=(("valid", 0, 0), ("valid", 1, 100)), evaluation_groups=(0,), fold=0
        ),
    )
    raw = np.random.default_rng(41).normal(size=(300, 112)).astype(np.float32)
    columns = {name: raw[:, index] for index, name in enumerate(data.FEATURE_COLUMNS)}
    columns.update(seq_ix=np.full(300, 7), t0=raw[:, 0], t1=raw[:, 1])
    (tmp_path / "datasets").mkdir()
    pq.write_table(pa.table(columns), tmp_path / "datasets/valid.parquet")
    mask = pa.table(
        {
            "seq_ix": np.full(300, 7),
            "step_in_seq": np.arange(300),
            "is_scored": np.ones(300, dtype=bool),
        }
    )
    pq.write_table(mask, tmp_path / "datasets/valid_mask.parquet")
    (tmp_path / "split.json").write_text("{}")
    model = RNNModel(211, hidden=4, layers=1)
    checkpoint = {
        "model": model.state_dict(),
        "n_in": 211,
        "epoch": 3,
        "args": {
            "hidden": 4,
            "layers": 1,
            "no_eval_during_fit": True,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 3,
        },
        "provenance": {"fit_row_groups": {"valid": [1]}},
    }
    torch.save(checkpoint, tmp_path / "model.pt")

    def binding(name: str) -> dict:
        return {"path": name, "sha256": digest_file(tmp_path / name)}

    manifest = {
        "version": 1,
        "role": "fit",
        "source": "valid",
        "row_groups": [0],
        "checkpoint": binding("model.pt"),
        "ancestors": [],
        "campaign_split": binding("split.json"),
        "mask": binding("datasets/valid_mask.parquet"),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    extract(tmp_path, tmp_path / "manifest.json", tmp_path / "cache.npz", "cpu")
    cached, _ = load_cache(tmp_path / "cache.npz")
    selected = int(selected_rows(7, 300)[0])
    assert cached["row"].tolist() == [selected]
    np.testing.assert_array_equal(
        cached["placebo__event_age"][0], causal_cues(raw)["event_age"][selected - 128]
    )
    (tmp_path / "datasets/valid_mask.parquet").write_bytes(b"changed mask")
    with pytest.raises(ValueError, match="hash"):
        extract(tmp_path, tmp_path / "manifest.json", tmp_path / "bad.npz", "cpu")


def test_row_sampling_is_reproducible_uniform_jitter_without_phase_alias() -> None:
    rows = selected_rows(7, 20_000)
    np.testing.assert_array_equal(rows, selected_rows(7, 20_000))
    assert set(rows % 8) == set(range(8))
    assert len(np.unique((rows - 227) // 128)) == len(rows)
    assert rows.min() >= 227 and rows.max() < 20_000
    assert not np.array_equal(rows, selected_rows(8, 20_000))
