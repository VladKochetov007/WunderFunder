from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from packet_contract import REPO, file_hash, read, require, write_new

# JSON metadata includes heterogeneous test-report and artifact identity fields.
TEST_MODULES = (
    "test_sequence_source_splits",
    "test_sequence_kd_normalization",
    "test_source_kd_packet",
    "test_b300_sequence_models",
    "test_b300_sequence_contracts",
    "test_sequence_precision",
    "test_sequence_precision_cuda",
    "test_sequence_recovery_cuda",
)


def seal_acceptance(root: Path) -> None:
    """Require all named receiver modules to pass without skipped correctness cases."""
    report = root / "acceptance/pytest.xml"
    xml = ET.parse(report).getroot()
    cases = list(xml.iter("testcase"))
    require(bool(cases), "No receiving-host correctness cases ran")
    require(
        not any(list(xml.iter(tag)) for tag in ("failure", "error", "skipped")),
        "Receiver correctness checks failed or skipped",
    )
    for name in TEST_MODULES:
        require(
            any(name in item.get("classname", "").split(".") for item in cases),
            f"Missing receiver module: {name}",
        )
    value: dict[str, Any] = {
        "schema": "wnn.source-kd-acceptance.v1",
        "status": "passed",
        "packet_sha256": file_hash(root / "packet.json"),
        "command_sha256": file_hash(root / "acceptance/command.json"),
        "report_sha256": file_hash(report),
        "tests": {name: file_hash(REPO / "tests" / f"{name}.py") for name in TEST_MODULES},
        "test_cases": len(cases),
        "limits": "Receiver correctness only; fresh BF16/BF16x9 real-data profiles remain mandatory per cell.",
    }
    write_new(root / "acceptance/receipt.json", value)


def require_acceptance(root: Path) -> None:
    """Reject stale, edited or incomplete receiver correctness evidence."""
    value = read(root / "acceptance/receipt.json")
    require(
        value.get("schema") == "wnn.source-kd-acceptance.v1" and value.get("status") == "passed",
        "Receiver acceptance required",
    )
    require(
        value.get("packet_sha256") == file_hash(root / "packet.json"),
        "Acceptance belongs to another packet",
    )
    require(
        value.get("command_sha256") == file_hash(root / "acceptance/command.json"),
        "Acceptance command changed",
    )
    require(
        value.get("report_sha256") == file_hash(root / "acceptance/pytest.xml"),
        "Acceptance report changed",
    )
    require(
        value.get("tests")
        == {name: file_hash(REPO / "tests" / f"{name}.py") for name in TEST_MODULES},
        "Receiver test source changed",
    )
