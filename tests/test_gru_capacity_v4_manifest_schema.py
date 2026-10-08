"""Test the V4 raw-input schema fix against the actual frozen V3 packet manifests."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
PACKET = CAMPAIGN / "gru512_768_muon_capacity_v4_manifest_profile_20261008/packet"
V3_PACKET = (
    CAMPAIGN
    / "gru512_768_capacity_v3_terminal_20261008/downloaded_outputs"
    / "gru512_768_muon_capacity_v3_profile_packet"
)
EXPECTED_RAW = {
    "datasets/train.parquet": (
        29_240_219_011,
        "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
    ),
    "datasets/valid.parquet": (
        5_615_509_128,
        "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    ),
}

sys.path.insert(0, str(PACKET))
_SPEC = importlib.util.spec_from_file_location("capacity_worker_v4_manifest_test", PACKET / "capacity_worker.py")
assert _SPEC is not None and _SPEC.loader is not None
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)


def _actual_inputs() -> tuple[dict[str, object], dict[str, object]]:
    portable = json.loads((V3_PACKET / "portable_inputs.json").read_text(encoding="utf-8"))
    prior = json.loads((V3_PACKET / "prior_input_identity_v2.json").read_text(encoding="utf-8"))
    return portable, prior


def test_v4_preserves_frozen_v3_scientific_recipe_and_mount_identity() -> None:
    v3_recipe = json.loads((V3_PACKET / "GRU512_768_MUON_PROFILE_RECIPE_V3.json").read_text())
    v4_recipe = json.loads((PACKET / "GRU512_768_MUON_PROFILE_RECIPE_V4.json").read_text())
    identity_fields = {"packet_id", "status", "source", "runtime", "authorization"}
    assert {key: value for key, value in v4_recipe.items() if key not in identity_fields} == {
        key: value for key, value in v3_recipe.items() if key not in identity_fields
    }
    v3_runtime = {key: value for key, value in v3_recipe["runtime"].items() if key not in {
        "requested_kernel",
        "requested_source_dataset",
    }}
    v4_runtime = {key: value for key, value in v4_recipe["runtime"].items() if key not in {
        "requested_kernel",
        "requested_source_dataset",
    }}
    assert v4_runtime == v3_runtime
    for key in (
        "independent_source_commit",
        "omitted_members",
        "optimizer_sha256",
        "predictor_sha256",
        "runtime_dependency_additions",
        "sampling_sha256",
        "source_archive",
        "source_manifest",
        "source_restrictions",
        "trainer_sha256",
    ):
        assert v4_recipe["source"][key] == v3_recipe["source"][key]


@pytest.mark.parametrize("logical_name", sorted(EXPECTED_RAW))
def test_real_producer_raw_manifest_schema_matches_prior_fingerprint_and_size(
    logical_name: str,
) -> None:
    portable, prior = _actual_inputs()
    expected_bytes, expected_sha = EXPECTED_RAW[logical_name]
    entry = portable["sources"][logical_name]
    observed = prior["observed"][logical_name]

    assert set(entry) == {"original_identity", "sha256"}
    assert set(entry["original_identity"]) == {
        "bytes",
        "footer_bytes",
        "footer_sha256",
        "mtime_ns",
    }
    assert entry["sha256"] == expected_sha
    assert entry["original_identity"]["bytes"] == expected_bytes
    assert observed["sha256"] == expected_sha
    assert observed["bytes"] == expected_bytes
    assert worker._raw_source_expected_binding(entry, observed, logical_name) == (
        expected_sha,
        expected_bytes,
    )


def test_old_flat_bytes_shape_is_rejected_instead_of_silently_falling_back() -> None:
    portable, prior = _actual_inputs()
    logical_name = "datasets/train.parquet"
    expected_bytes, _ = EXPECTED_RAW[logical_name]
    old_shape = copy.deepcopy(portable["sources"][logical_name])
    old_shape.pop("original_identity")
    old_shape["bytes"] = expected_bytes

    with pytest.raises(ValueError, match="portable raw entry schema changed"):
        worker._raw_source_expected_binding(old_shape, prior["observed"][logical_name], logical_name)


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("bytes", 1)])
def test_prior_raw_fingerprint_or_size_mismatch_fails_before_any_file_hash(
    field: str, value: object
) -> None:
    portable, prior = _actual_inputs()
    logical_name = "datasets/valid.parquet"
    expected = portable["sources"][logical_name]
    observed = copy.deepcopy(prior["observed"][logical_name])
    observed[field] = value

    with pytest.raises(ValueError, match="prior raw fingerprint or size differs"):
        worker._raw_source_expected_binding(expected, observed, logical_name)


def test_invalid_nested_size_is_rejected() -> None:
    portable, prior = _actual_inputs()
    logical_name = "datasets/train.parquet"
    expected = copy.deepcopy(portable["sources"][logical_name])
    expected["original_identity"]["bytes"] = True

    with pytest.raises(ValueError, match="portable raw fingerprint or size is malformed"):
        worker._raw_source_expected_binding(
            expected, prior["observed"][logical_name], logical_name
        )


def test_exact_private_transport_materializes_and_verifies_pending_profile_packet(
    tmp_path: Path,
) -> None:
    driver_spec = importlib.util.spec_from_file_location(
        "gru_capacity_v4_kaggle_driver_test", PACKET / "kernel_upload/kernel.py"
    )
    assert driver_spec is not None and driver_spec.loader is not None
    driver = importlib.util.module_from_spec(driver_spec)
    driver_spec.loader.exec_module(driver)

    input_root = tmp_path / "input"
    source_root = input_root / "datasets/vladkochetov07/e529-gru512-768-muon-capacity-v4-source-20261008"
    source_root.parent.mkdir(parents=True)
    shutil.copytree(PACKET / "source_dataset_upload", source_root)
    unrelated = input_root / "datasets/another-owner/unrelated-dataset"
    unrelated.mkdir(parents=True)
    (unrelated / "assets-manifest.json").write_text(
        json.dumps({"dataset": "another-owner/unrelated-dataset", "files": []}),
        encoding="utf-8",
    )

    assert driver.source_dataset_root(input_root) == source_root
    materialized = tmp_path / "working/packet"
    driver.materialize_packet(input_root, materialized)
    recipe, authorization = worker._verify_packet(materialized)
    assert recipe["packet_id"] == worker.PACKET_ID
    assert authorization["profile_authorized"] is False
    assert authorization["conditional_fit_authorized"] is False
