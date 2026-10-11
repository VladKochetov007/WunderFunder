from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "build_export_recovery.py"
SPEC = importlib.util.spec_from_file_location("h382_export_recovery_r4_builder", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import the frozen R4 packet builder")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
HERE = BUILDER.HERE


def generated_source() -> str:
    manifest = json.loads((HERE / "checkpoint_asset_manifest.json").read_text())
    source = BUILDER.patch_kernel(manifest).decode()
    assert source == (HERE / "kernel/kernel.py").read_text()
    ast.parse(source)
    return source


def load_collection_helper() -> object:
    source = (HERE / "kernel/kernel.py").read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_collected_prediction_pair"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, object] = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "sha256_file": BUILDER.sha256_file,
    }
    exec(compile(isolated, "kernel.py", "exec"), namespace)
    return namespace["validate_collected_prediction_pair"]


def fixture_pair(directory: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    run = "kg_h382r3_f0_muonall_prop2_e3_s3"
    prediction = directory / f"pred_{run}_valid.npy"
    sidecar = directory / f"pred_{run}_valid.json"
    prediction.write_bytes(b"tiny-npy-header-and-payload")
    sidecar.write_text(
        json.dumps(
            {
                "run": run,
                "split": "valid",
                "epoch": 2,
                "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    receipt: dict[str, object] = {
        "run": run,
        "source_packet_id": "vladkochetov07/wnn-kaggle-crossfold-h382-20261011-r3",
        "source_bundle_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "checkpoint_bytes": 1234,
        "epoch": 2,
        "prediction_shape": [1873, 20_000, 2],
        "prediction_dtype": "<f4",
        "prediction_finite": True,
        "prediction_bytes": prediction.stat().st_size,
        "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        "sidecar_bytes": sidecar.stat().st_size,
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }
    expected = {
        key: receipt[key]
        for key in (
            "run",
            "source_packet_id",
            "source_bundle_sha256",
            "split_manifest_sha256",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "epoch",
            "prediction_shape",
            "prediction_dtype",
            "prediction_finite",
        )
    }
    return prediction, sidecar, receipt, expected


def test_r4_is_export_only_and_binds_same_run_fresh_receipt() -> None:
    source = generated_source()
    assert 'SOURCE_PACKET_ID = "vladkochetov07/wnn-kaggle-crossfold-h382-20261011-r3"' in source
    assert 'PACKET_ID = "vladkochetov07/wnn-h382-export-recovery-20261011-r4"' in source
    assert "job[\"steps\"] = export_steps" in source
    assert "validate_collected_prediction_pair(" in source
    assert "fresh_export_receipt_sha256" in source
    assert "all(check[\"valid\"] for check in collection_checks)" in source
    assert '"sidecar_bytes": sidecar.stat().st_size' in source
    assert '"source_bundle_sha256": BUNDLE_SHA256' in source
    assert '"prediction_finite": True' in source
    assert "train_rnn.py" in source
    assert source.count('"label": "export"') == 0  # Export commands are inherited from the sealed source job JSON.
    manifest = json.loads((HERE / "checkpoint_asset_manifest.json").read_text())
    assert manifest["checkpoint_archive"]["bytes"] == 12_441_600
    assert manifest["checkpoint_archive"]["sha256"] == "a57df09373de67618c7a5826a11a3756fbdb1640b38781af1467a6cbbc468285"
    assert BUILDER.CHECKPOINT_DATASET_ID.endswith("-r4")


def test_collection_pair_accepts_tiny_receipt_bound_files(tmp_path: Path) -> None:
    prediction, sidecar, receipt, expected = fixture_pair(tmp_path)
    validate = load_collection_helper()
    result = validate(
        str(receipt["run"]), prediction, sidecar, receipt, expected
    )  # type: ignore[operator]
    assert result["valid"] is True  # type: ignore[index]
    assert result["errors"] == []  # type: ignore[index]
    assert set(result["files"]) == {"npy", "sidecar"}  # type: ignore[index]


def test_collection_pair_rejects_corrupt_copies_and_wrong_identity(tmp_path: Path) -> None:
    prediction, sidecar, receipt, expected = fixture_pair(tmp_path)
    validate = load_collection_helper()
    prediction.write_bytes(b"truncated")
    wrong_identity = dict(expected, source_packet_id="wrong-source")
    result = validate(
        str(receipt["run"]), prediction, sidecar, receipt, wrong_identity
    )  # type: ignore[operator]
    assert result["valid"] is False  # type: ignore[index]
    assert any("source_packet_id_mismatch" in error for error in result["errors"])  # type: ignore[index]
    assert any("collected_npy_bytes_mismatch" in error for error in result["errors"])  # type: ignore[index]
    assert any("collected_npy_sha256_mismatch" in error for error in result["errors"])  # type: ignore[index]


def test_collection_pair_rejects_missing_copy(tmp_path: Path) -> None:
    prediction, sidecar, receipt, expected = fixture_pair(tmp_path)
    sidecar.unlink()
    validate = load_collection_helper()
    result = validate(
        str(receipt["run"]), prediction, sidecar, receipt, expected
    )  # type: ignore[operator]
    assert result["valid"] is False  # type: ignore[index]
    assert f"{receipt['run']}:collected_sidecar_missing" in result["errors"]  # type: ignore[index]
