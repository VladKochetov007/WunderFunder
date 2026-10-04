from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.probe_factorized_runtime as runtime
from scripts.probe_factorized_runtime import (
    finite_pair_max_abs,
    slice_deployment_head,
    timing_order,
)


def parity_fixture(tmp_path, monkeypatch):
    source, control, data, manifest = [
        tmp_path / name for name in ("source", "control", "data", "manifest")
    ]
    for path in (source, control, data, manifest):
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(runtime, "EXPECTED_SOURCE_SHA256", runtime.file_sha256(source))
    monkeypatch.setattr(runtime, "EXPECTED_CONTROL_SHA256", runtime.file_sha256(control))
    monkeypatch.setattr(runtime, "EXPECTED_DATA_SHA256", runtime.file_sha256(data))
    monkeypatch.setattr(runtime, "EXPECTED_E409_MANIFEST_SHA256", runtime.file_sha256(manifest))
    monkeypatch.setattr(runtime, "EXPECTED_CLOSURE_SHA256", "closure-test")

    library_root = tmp_path / "libs"
    library_hashes = {}
    for level in ("v2", "v3"):
        package = library_root / level / "wnn_native"
        package.mkdir(parents=True)
        library = package / "wnn_native.abi3.so"
        library.write_bytes(level.encode())
        library_hashes[level] = runtime.file_sha256(library)
    report = {
        "experiment": "E420",
        "purpose": "native full-stream representation parity; no timing or model selection",
        "source_checkpoint_sha256": runtime.EXPECTED_SOURCE_SHA256,
        "zero_update_control_sha256": runtime.EXPECTED_CONTROL_SHA256,
        "e409_manifest_sha256": runtime.EXPECTED_E409_MANIFEST_SHA256,
        "source_closure_sha256": runtime.EXPECTED_CLOSURE_SHA256,
        "e409_data_sha256": runtime.EXPECTED_DATA_SHA256,
        "gate": {"max_abs": 3e-4, "abs_global_wp_delta": 5e-6},
        "row_groups": [0, 1, 2],
        "v2_v3_bitwise_equal": True,
        "cases": {
            level: {
                "passed": True,
                "max_abs_first_two_all_rows": 1e-6,
                "abs_global_wp_delta": 1e-7,
                "portable_bitwise_equal": True,
                "native_library_sha256": library_hashes[level],
            }
            for level in ("v2", "v3")
        },
    }
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps(report))
    monkeypatch.setattr(runtime, "EXPECTED_PARITY_SHA256", runtime.file_sha256(parity))
    return parity, source, control, data, manifest, library_root, library_hashes, report


def test_runtime_slices_only_the_two_scored_head_rows() -> None:
    weights = {
        "head.2.weight": np.arange(23 * 384, dtype=np.float32).reshape(23, 384),
        "head.2.bias": np.arange(23, dtype=np.float32),
        "gru.bias_hh_l0": np.zeros(1152, dtype=np.float32),
    }
    sliced = slice_deployment_head(weights)
    assert sliced["head.2.weight"].shape == (2, 384)
    assert sliced["head.2.bias"].shape == (2,)
    np.testing.assert_array_equal(sliced["head.2.weight"], weights["head.2.weight"][:2])
    np.testing.assert_array_equal(sliced["head.2.bias"], weights["head.2.bias"][:2])
    assert sliced["gru.bias_hh_l0"] is weights["gru.bias_hh_l0"]
    assert weights["head.2.weight"].shape == (23, 384)


def test_runtime_rejects_a_control_without_the_auxiliary_output_layout() -> None:
    with pytest.raises(ValueError, match="23-output head"):
        slice_deployment_head(
            {
                "head.2.weight": np.zeros((2, 384), dtype=np.float32),
                "head.2.bias": np.zeros(2, dtype=np.float32),
            }
        )


def test_timing_cases_alternate_forward_and_reverse() -> None:
    forward = timing_order(0)
    reverse = timing_order(1)
    assert forward == ("factor_f32", "effective_dense_f32", "original_w384_q16x")
    assert reverse == tuple(reversed(forward))
    assert timing_order(2) == forward
    assert len(forward) == len(set(forward)) == 3


def test_callback_comparison_ignores_matching_warmup_nans() -> None:
    left = np.array([[np.nan, np.nan], [1.0, 2.0]], dtype=np.float32)
    right = np.array([[np.nan, np.nan], [1.0, 2.00001]], dtype=np.float32)
    assert finite_pair_max_abs(left, right) == pytest.approx(1e-5, rel=2e-3)


def test_callback_comparison_rejects_different_finite_rows() -> None:
    left = np.array([[np.nan], [1.0]], dtype=np.float32)
    right = np.array([[0.0], [1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="different valid-row masks"):
        finite_pair_max_abs(left, right)


def test_runtime_parity_gate_binds_both_native_libraries(tmp_path, monkeypatch) -> None:
    parity, source, control, data, manifest, library_root, library_hashes, _report = parity_fixture(
        tmp_path, monkeypatch
    )
    actual, actual_hashes = runtime.validate_parity(
        parity, source, control, data, manifest, library_root
    )
    assert actual["experiment"] == "E420"
    assert actual_hashes == library_hashes


def test_runtime_parity_gate_rejects_failed_v2(tmp_path, monkeypatch) -> None:
    parity, source, control, data, manifest, library_root, _hashes, report = parity_fixture(
        tmp_path, monkeypatch
    )
    report["cases"]["v2"]["passed"] = False
    parity.write_text(json.dumps(report))
    monkeypatch.setattr(runtime, "EXPECTED_PARITY_SHA256", runtime.file_sha256(parity))
    with pytest.raises(ValueError, match="both levels"):
        runtime.validate_parity(parity, source, control, data, manifest, library_root)
