from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import cache_propensity as cache_propensity_script
from scripts import train_propensity as train_propensity_script
from wnn.propensity import (
    LEGACY_PROFILE,
    PROPENSITY_FIT_SOURCE_FILES,
    STRICT_FP32_PROFILE,
    PropensityNet,
    fit_normalization,
    ordered_groups_sha256,
    propensity,
)
from wnn.splits import CampaignSplit
from wnn.training import cache_identity
from wnn.training.cache_identity import prop_cache_path
from wnn.training.environment import TrainingEnvironment


def test_streaming_normalization_uses_all_ordered_fit_batches() -> None:
    model = PropensityNet(n_in=2, hidden=3)
    batches = [
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
    ]

    provenance = fit_normalization(model, iter(batches))

    expected = torch.cat([batch.reshape(-1, 2) for batch in batches]).to(torch.float64)
    torch.testing.assert_close(model.mean, expected.mean(0).float())
    torch.testing.assert_close(model.std, expected.std(0, unbiased=False).float())
    assert provenance["count"] == 4
    assert provenance["accumulator_dtype"] == "float64"
    assert provenance["buffer_dtype"] == "float32"
    assert provenance["method"] == "ordered_fit_population_welford_float64_v1"


def test_streaming_normalization_rejects_nonfinite_values() -> None:
    model = PropensityNet(n_in=2, hidden=3)
    batch = torch.tensor([[[1.0, float("nan")]]])
    with pytest.raises(FloatingPointError, match="non-finite"):
        fit_normalization(model, [batch])


def test_propensity_precision_profile_is_explicit_and_legacy_default_is_preserved() -> None:
    model = PropensityNet(n_in=2, hidden=3).eval()
    inputs = torch.randn(2, 7, 2)
    legacy = propensity(model, inputs, chunk=3)
    explicit_legacy = propensity(model, inputs, chunk=3, profile=LEGACY_PROFILE)
    strict = propensity(model, inputs, chunk=3, profile=STRICT_FP32_PROFILE)

    torch.testing.assert_close(legacy, explicit_legacy)
    torch.testing.assert_close(legacy, strict)
    with pytest.raises(ValueError, match="float32 inputs"):
        propensity(model, inputs.to(torch.float64), profile=STRICT_FP32_PROFILE)


def test_strict_propensity_rejects_nonfinite_inputs_and_model_outputs() -> None:
    model = PropensityNet(n_in=2, hidden=3).eval()
    bad_inputs = torch.full((1, 2, 2), float("nan"))
    with pytest.raises(FloatingPointError, match="inputs"):
        propensity(model, bad_inputs, profile=STRICT_FP32_PROFILE)

    class NonfiniteModel(torch.nn.Module):
        def forward(self, inputs, hidden=None):
            logits = torch.full(inputs.shape[:2], float("inf"), device=inputs.device)
            return logits, hidden

    with pytest.raises(FloatingPointError, match="logits"):
        propensity(NonfiniteModel(), torch.ones((1, 2, 2)), profile=STRICT_FP32_PROFILE)


def test_strict_run_directory_is_create_only_but_legacy_reuses(tmp_path: Path) -> None:
    run = tmp_path / "artifacts/runs/clean"
    train_propensity_script._prepare_run_directory(run, strict=True)
    with pytest.raises(FileExistsError):
        train_propensity_script._prepare_run_directory(run, strict=True)
    train_propensity_script._prepare_run_directory(run, strict=False)

    dangling = tmp_path / "artifacts/runs/dangling"
    dangling.parent.mkdir(parents=True, exist_ok=True)
    dangling.symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileExistsError):
        train_propensity_script._prepare_run_directory(dangling, strict=True)
    with pytest.raises(FloatingPointError, match="gradient"):
        train_propensity_script._require_finite(torch.tensor([float("inf")]), "gradient test")


def test_strict_cache_dataset_reads_feature_columns_without_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSequenceFile:
        def __init__(self, _path: Path) -> None:
            pass

        def read(self, index: int, *, targets: bool, mask: bool):
            assert index == 7
            assert not targets and not mask
            return type("Sequence", (), {"features": np.ones((2, 3), dtype=np.float32)})()

    monkeypatch.setattr(cache_propensity_script, "SequenceFile", FakeSequenceFile)
    dataset = cache_propensity_script.FeatureOnlySequences("valid", np.asarray([7]))

    assert torch.equal(dataset[0], torch.ones((2, 3)))


def _write_source(root: Path, source: Path, target: str) -> str:
    dest = root / target
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    dest.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _strict_cache_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], CampaignSplit]:
    root = Path(__file__).resolve().parents[1]
    fake_root = tmp_path / "project"
    for path in (*PROPENSITY_FIT_SOURCE_FILES, "scripts/cache_propensity.py"):
        _write_source(fake_root, root / path, path)
    (fake_root / "datasets").mkdir(parents=True)
    (fake_root / "datasets/valid.parquet").touch()
    cache_path = fake_root / "cache/propcache_test_valid.npy"
    cache_path.parent.mkdir(parents=True)
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(2, 20000))
    cache[:] = 0.25
    cache.flush()
    del cache

    mean = torch.zeros(1, dtype=torch.float32)
    std = torch.ones(1, dtype=torch.float32)
    normalization = {
        "method": "ordered_fit_population_welford_float64_v1",
        "accumulator_dtype": "float64",
        "buffer_dtype": "float32",
        "count": 20000,
        "std_floor": 0.001,
        "mean_sha256": hashlib.sha256(mean.numpy().astype("<f4").tobytes()).hexdigest(),
        "std_sha256": hashlib.sha256(std.numpy().astype("<f4").tobytes()).hexdigest(),
    }
    split = CampaignSplit(
        path=tmp_path / "split.json",
        sha256="c" * 64,
        fold=0,
        n_folds=2,
        train_groups=(0,),
        valid_groups=(1,),
        evaluation_groups=(0,),
        components=(("valid", 0, 0), ("valid", 1, 1)),
    )
    model_path = fake_root / "artifacts/runs/test/model.pt"
    model_path.parent.mkdir(parents=True)
    torch.save(
        {
            "model": {"mean": mean, "std": std},
            "hidden": 1,
            "n_in": 1,
            "args": {},
            "provenance": {
                "precision_profile": STRICT_FP32_PROFILE,
                "split_manifest_sha256": split.sha256,
                "fit_row_groups": {"valid": [1]},
                "fit_row_groups_sha256": {"valid": ordered_groups_sha256([1])},
                "evaluation_row_groups": [0],
                "normalization": normalization,
                "source_sha256": {
                    relative: hashlib.sha256((fake_root / relative).read_bytes()).hexdigest()
                    for relative in PROPENSITY_FIT_SOURCE_FILES
                },
            },
        },
        model_path,
    )
    source_hashes = {
        relative: hashlib.sha256((fake_root / relative).read_bytes()).hexdigest()
        for relative in (*PROPENSITY_FIT_SOURCE_FILES, "scripts/cache_propensity.py")
    }
    metadata: dict[str, object] = {
        "schema": "wnn.propensity_cache.strict_fp32.v1",
        "model": "test",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "split": "valid",
        "rows": 2,
        "n_filled": 2,
        "precision_profile": STRICT_FP32_PROFILE,
        "inference_precision": "float32_no_autocast_tf32_disabled",
        "split_manifest_sha256": split.sha256,
        "fit_row_groups_sha256": ordered_groups_sha256([1]),
        "fit_row_groups_count": 1,
        "ordered_output_row_groups_sha256": ordered_groups_sha256([0, 1]),
        "populated_row_groups_sha256": ordered_groups_sha256([0, 1]),
        "checkpoint_normalization": normalization,
        "cache_shape": [2, 20000],
        "cache_dtype": "float32",
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
    }
    cache_path.with_suffix(".json").write_text(json.dumps(metadata))
    return cache_path, metadata, split


def test_strict_cache_validates_fit_membership_separately_from_output_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path, metadata, split = _strict_cache_fixture(tmp_path)
    monkeypatch.setattr(cache_identity, "SequenceFile", lambda _path: [None, None])
    environment = TrainingEnvironment(root=tmp_path / "project", source_root=tmp_path / "project")

    resolved = prop_cache_path("test", "valid", environment=environment, campaign_split=split)

    assert resolved == cache_path
    assert metadata["fit_row_groups_sha256"] != metadata["ordered_output_row_groups_sha256"]

    changed = dict(metadata)
    changed["ordered_output_row_groups_sha256"] = ordered_groups_sha256([1, 0])
    cache_path.with_suffix(".json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="row-group order"):
        prop_cache_path("test", "valid", environment=environment, campaign_split=split)


def test_strict_cache_rejects_nonfinite_and_out_of_range_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path, metadata, split = _strict_cache_fixture(tmp_path)
    monkeypatch.setattr(cache_identity, "SequenceFile", lambda _path: [None, None])
    environment = TrainingEnvironment(root=tmp_path / "project", source_root=tmp_path / "project")

    cache = np.load(cache_path, mmap_mode="r+")
    cache[0, 0] = np.nan
    cache.flush()
    del cache
    metadata["cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    cache_path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="invalid probability"):
        prop_cache_path("test", "valid", environment=environment, campaign_split=split)


def test_strict_cache_preflights_both_splits_and_publishes_create_only(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    paths = cache_propensity_script._strict_output_paths(tmp_path, "new")
    assert {split for split, _, _ in paths} == {"train", "valid"}

    dangling_sidecar = paths[1][2]
    dangling_sidecar.symlink_to(tmp_path / "missing-sidecar")
    with pytest.raises(FileExistsError, match="already exists"):
        cache_propensity_script._strict_output_paths(tmp_path, "new")

    temp_array = cache_propensity_script._reserve_temp(paths[0][1])
    temp_sidecar = cache_propensity_script._reserve_temp(paths[0][2])
    temp_array.write_bytes(b"array")
    temp_sidecar.write_bytes(b"metadata")
    paths[0][1].write_bytes(b"racer")
    with pytest.raises(FileExistsError):
        cache_propensity_script._publish_pair(temp_array, temp_sidecar, paths[0][1], paths[0][2])
    assert temp_array.exists() and temp_sidecar.exists()


def test_reserved_cache_memmap_is_valid_npy_and_rejects_symlink_swap(tmp_path: Path) -> None:
    reserved = cache_propensity_script._reserve_temp(tmp_path / "cache.npy")
    mapped = cache_propensity_script._open_reserved_memmap(reserved, (2, 3))
    mapped[:] = 0.5
    mapped.flush()
    del mapped
    assert np.all(np.load(reserved, allow_pickle=False) == 0.5)

    replaced = cache_propensity_script._reserve_temp(tmp_path / "replace.npy")
    replaced.unlink()
    replaced.symlink_to(tmp_path / "target")
    with pytest.raises(OSError):
        cache_propensity_script._open_reserved_memmap(replaced, (2, 3))
