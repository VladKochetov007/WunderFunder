from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("collect_r4_terminal.py")
SPEC = importlib.util.spec_from_file_location("collect_r4_terminal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def test_frozen_bindings_membership_and_root_download_admission() -> None:
    static = COLLECTOR.validate_frozen_r4_bindings()
    membership = COLLECTOR.load_expected_membership()
    assert static["archive_sha256"] == COLLECTOR.R4_ARCHIVE_SHA256
    assert membership["counts"] == {
        "train": 10_535,
        "valid": 1_405,
        "excluded_valid_eval": 468,
    }
    release = COLLECTOR.ROOT_RELEASE
    status = release / "ROOT_PROVIDER_POLL_R2.json"
    admission = release / "ROOT_TERMINAL_DOWNLOAD_ADMISSION_R1.json"
    inventory = (
        SCRIPT.parent / "read_only_inventory_attempt_001/R4_READONLY_INVENTORY_RECEIPT_R1.json"
    )
    payload = json.loads(admission.read_text(encoding="utf-8"))
    result = COLLECTOR.validate_download_admission(
        admission, status, inventory, Path(payload["collection_dir"])
    )
    assert result["provider_status"] == "KernelWorkerStatus.ERROR"
    assert result["download_budget_bytes"] == 1024**3


def test_actual_previous_producer_npy_header_and_sidecar() -> None:
    array_receipt = COLLECTOR.validate_npy(
        COLLECTOR.H382_ACTUAL_NPY,
        (1_873, 20_000, 2),
        "float32",
        finite_rows=16,
    )
    sidecar = json.loads(COLLECTOR.H382_ACTUAL_SIDECAR.read_text(encoding="utf-8"))
    assert array_receipt["npy_bytes"] == 299_680_128
    assert array_receipt["rows_scanned"] == 16
    assert array_receipt["all_values_finite"] is False
    assert sidecar["run"] == "kg_h382r3_f0_muonall_prop2_e3_s3"
    assert sidecar["split"] == "valid"
    assert sidecar["epoch"] == 2
    assert sidecar["checkpoint_sha256"] == (
        "69f069dca20d71f0f5d9ea1d08bcd6884cb7a3d29699a5606e8828bd363467fa"
    )
    assert sidecar["prediction_sha256"] == COLLECTOR.H382_ACTUAL_NPY_SHA256
    assert sidecar["split_manifest_sha256"] == COLLECTOR.SPLIT_SHA256
    assert sidecar["row_groups"] == list(range(1_873))


def test_full_finite_scan_rejects_nonfinite_values(tmp_path: Path) -> None:
    array = np.zeros((2, 3, 2), dtype=np.float32)
    array[1, 1, 0] = np.nan
    path = tmp_path / "nonfinite.npy"
    np.save(path, array, allow_pickle=False)
    with pytest.raises(ValueError, match="NaN or infinity"):
        COLLECTOR.validate_npy(path, (2, 3, 2), "float32")
