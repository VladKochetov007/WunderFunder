"""Collect terminal metadata for the frozen online relationship panel."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import struct
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

PACKET = Path("research/experiment_drafts/online-relationships-20261022")
PANEL_RECEIPT = "PANEL_EXECUTION_R1.json"
ADMISSION_RECEIPT = "PANEL_ADMISSION_R1.json"
OBSERVER_RECEIPT = "CPU_OBSERVER_EXECUTION_R1.json"
ARTIFACT_NAMES = {
    "checkpoint": "model_ep2.pt",
    "provenance": "provenance.json",
    "prediction_sidecar": "prediction.json",
    "prediction_npy": "prediction.npy",
}
SYSTEMD_PROPERTIES = (
    "InvocationID",
    "ActiveState",
    "SubState",
    "Result",
    "ExecMainCode",
    "ExecMainStatus",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
    "ActiveEnterTimestamp",
    "InactiveEnterTimestamp",
    "RuntimeMaxUSec",
    "CPUUsageNSec",
    "CPUQuotaPerSecUSec",
    "MemoryCurrent",
    "MemoryPeak",
    "MemorySwapCurrent",
    "OOMKilled",
    "FreezerState",
)


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object and reject other top-level values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, object], value)


def sha256_file(path: Path) -> str:
    """Hash a small metadata or checkpoint file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_npy_header(path: Path) -> dict[str, object]:
    """Read only an NPY header and report the remaining byte count."""
    with path.open("rb") as source:
        if source.read(6) != b"\x93NUMPY":
            raise ValueError(f"not an NPY file: {path}")
        version = source.read(2)
        if len(version) != 2:
            raise ValueError(f"truncated NPY version: {path}")
        if version[0] == 1:
            header_length_bytes = source.read(2)
            prefix_bytes = 10
            encoding = "latin1"
            header_length_format = "<H"
        elif version[0] in {2, 3}:
            header_length_bytes = source.read(4)
            prefix_bytes = 12
            encoding = "utf-8" if version[0] == 3 else "latin1"
            header_length_format = "<I"
        else:
            raise ValueError(f"unsupported NPY version {version!r}: {path}")
        if len(header_length_bytes) != prefix_bytes - 8:
            raise ValueError(f"truncated NPY header length: {path}")
        header_length = struct.unpack(header_length_format, header_length_bytes)[0]
        if header_length > 1_000_000:
            raise ValueError(f"unreasonably large NPY header: {header_length}")
        header_text = source.read(header_length).decode(encoding).strip()
        if len(header_text) == 0:
            raise ValueError(f"empty NPY header: {path}")
        header = ast.literal_eval(header_text)
        if not isinstance(header, dict):
            raise ValueError(f"invalid NPY header object: {path}")
        header_end = prefix_bytes + header_length
    file_bytes = path.stat().st_size
    if file_bytes < header_end:
        raise ValueError(f"NPY file is shorter than its header: {path}")
    shape = header.get("shape")
    if not isinstance(shape, tuple) or not all(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in shape
    ):
        raise ValueError(f"invalid NPY shape: {path}")
    return {
        "version": list(version),
        "descr": header.get("descr"),
        "shape": list(shape),
        "fortran_order": header.get("fortran_order"),
        "header_bytes": header_end,
        "file_bytes": file_bytes,
        "payload_bytes": file_bytes - header_end,
    }


def systemd_status(unit: str, expected_invocation: str, *, terminal: bool) -> dict[str, object]:
    """Capture one exact user-unit invocation and optionally require termination."""
    command = [
        "systemctl",
        "--user",
        "show",
        *(f"--property={name}" for name in SYSTEMD_PROPERTIES),
        unit,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"systemctl show failed for {unit}: {result.stderr.strip()}")
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if properties.get("InvocationID") != expected_invocation:
        raise RuntimeError(f"unexpected or missing invocation ID for {unit}")
    if terminal and properties.get("ActiveState") not in {"inactive", "failed"}:
        raise RuntimeError(f"refusing collection while {unit} is not terminal")
    return {
        "unit": unit,
        "expected_invocation_id": expected_invocation,
        "properties": properties,
        "terminal_required": terminal,
    }


def artifact_record(path: Path, *, hash_payload: bool) -> dict[str, object]:
    """Describe an artifact without reading NPY payload data."""
    record: dict[str, object] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return record
    record["file_bytes"] = path.stat().st_size
    if hash_payload:
        record["sha256"] = sha256_file(path)
    else:
        try:
            record["npy_header"] = read_npy_header(path)
        except (OSError, ValueError, SyntaxError, UnicodeDecodeError) as error:
            record["header_error"] = str(error)
    return record


def prediction_header_errors(header: dict[str, object]) -> list[str]:
    """Check the fixed little-endian float32 prediction storage contract."""
    errors: list[str] = []
    shape = cast(list[int], header["shape"])
    if header.get("descr") != "<f4":
        errors.append("prediction NPY dtype is not <f4")
    if header.get("fortran_order") is not False:
        errors.append("prediction NPY is not C-contiguous")
    expected_payload_bytes = math.prod(shape) * 4
    if header.get("payload_bytes") != expected_payload_bytes:
        errors.append(
            f"prediction NPY payload has {header.get('payload_bytes')} bytes; "
            f"expected {expected_payload_bytes}"
        )
    return errors


def collect_cell(
    scratch: Path, cell: dict[str, object], source_pins: dict[str, str]
) -> dict[str, object]:
    """Inventory one readout without scoring or hashing prediction payloads."""
    output = Path(cast(str, cell["output"])).resolve()
    if not output.is_relative_to(scratch.resolve()):
        raise ValueError(f"cell output escapes frozen scratch: {output}")
    paths = {name: output / filename for name, filename in ARTIFACT_NAMES.items()}
    for path in paths.values():
        if path.exists() and not path.resolve().is_relative_to(output):
            raise ValueError(f"artifact path escapes cell output: {path}")
    artifacts = {
        name: artifact_record(path, hash_payload=name != "prediction_npy")
        for name, path in paths.items()
    }
    metadata_errors: list[str] = []
    prediction: dict[str, object] | None = None
    provenance: dict[str, object] | None = None
    if paths["prediction_sidecar"].is_file():
        try:
            prediction = read_json(paths["prediction_sidecar"])
        except (OSError, json.JSONDecodeError, ValueError) as error:
            metadata_errors.append(f"prediction sidecar: {error}")
    if paths["provenance"].is_file():
        try:
            provenance = read_json(paths["provenance"])
        except (OSError, json.JSONDecodeError, ValueError) as error:
            metadata_errors.append(f"provenance: {error}")
    if prediction is not None:
        if prediction.get("run") != cell.get("run") or prediction.get("epoch") != 2:
            metadata_errors.append("prediction sidecar run/epoch differs from recipe")
        checkpoint_meta = prediction.get("checkpoint")
        if not isinstance(checkpoint_meta, dict):
            metadata_errors.append("prediction sidecar has no checkpoint identity")
        elif checkpoint_meta.get("sha256") != artifacts["checkpoint"].get("sha256"):
            metadata_errors.append("checkpoint bytes differ from sidecar digest")
        if prediction.get("provenance_sha256") != artifacts["provenance"].get("sha256"):
            metadata_errors.append("provenance bytes differ from sidecar digest")
        npy_header = artifacts["prediction_npy"].get("npy_header")
        if isinstance(npy_header, dict):
            metadata_errors.extend(prediction_header_errors(npy_header))
            if npy_header.get("shape") != prediction.get("shape"):
                metadata_errors.append("NPY header shape differs from prediction sidecar")
            if npy_header.get("descr") != prediction.get("dtype"):
                metadata_errors.append("NPY header dtype differs from prediction sidecar")
        elif artifacts["prediction_npy"].get("exists"):
            metadata_errors.append("prediction NPY header could not be read")
    if provenance is not None and provenance.get("source_sha256") != source_pins:
        metadata_errors.append("provenance source pins differ from panel admission")
    missing = [name for name, artifact in artifacts.items() if not artifact["exists"]]
    if missing:
        state = "partial" if len(missing) < len(artifacts) else "missing"
    elif metadata_errors:
        state = "present_with_metadata_errors"
    else:
        state = "finished_files_present"
    return {
        "run": cell.get("run"),
        "fold": cell.get("fold"),
        "seed": cell.get("seed"),
        "lag": cell.get("lag"),
        "arm": cell.get("arm"),
        "output_dir": str(output),
        "state": state,
        "missing_artifacts": missing,
        "artifacts": artifacts,
        "prediction_payload_sha256_independently_verified": False,
        "metadata_errors": metadata_errors,
    }


def compare_source_pins(root: Path, pins: dict[str, str]) -> dict[str, object]:
    """Compare admission hashes with current runtime or grader source files."""
    observed = {
        relative_path: sha256_file(root / relative_path)
        if (root / relative_path).is_file()
        else None
        for relative_path in pins
    }
    return {
        "expected_sha256": pins,
        "observed_sha256": observed,
        "mismatches": [path for path, expected in pins.items() if observed[path] != expected],
    }


def build_receipt(root: Path) -> dict[str, object]:
    """Build terminal-only metadata after the pinned panel unit has stopped."""
    packet = root / PACKET
    execution_path = packet / PANEL_RECEIPT
    admission_path = packet / ADMISSION_RECEIPT
    observer_path = packet / OBSERVER_RECEIPT
    execution = read_json(execution_path)
    admission = read_json(admission_path)
    recipe_reference = cast(dict[str, object], execution["recipe"])
    recipe_path = packet / cast(str, recipe_reference["path"])
    recipe_sha = sha256_file(recipe_path)
    if recipe_sha != recipe_reference["sha256"]:
        raise ValueError("execution recipe hash differs from frozen panel receipt")
    unit = cast(str, execution["unit"])
    invocation_id = cast(str, execution["invocation_id"])
    worker_status = systemd_status(unit, invocation_id, terminal=True)
    observer_status: dict[str, object] = {"available": False, "reason": "observer receipt absent"}
    if observer_path.is_file():
        try:
            observer = read_json(observer_path)
            observer_identity = cast(dict[str, object], observer["observer"])
            observer_status = {
                "available": True,
                "receipt_path": str(observer_path),
                "status": systemd_status(
                    cast(str, observer["observer_unit"]),
                    cast(str, observer_identity["InvocationID"]),
                    terminal=False,
                ),
            }
        except (KeyError, OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
            observer_status = {
                "available": False,
                "receipt_path": str(observer_path),
                "error": str(error),
            }
    recipe = read_json(recipe_path)
    scratch = Path(cast(str, cast(dict[str, object], recipe["resource_contract"])["scratch"]))
    runtime_pins = cast(dict[str, str], admission["source_sha256"])
    grader_pins = cast(dict[str, str], admission["grader_source_sha256"])
    cells = cast(list[dict[str, object]], recipe["cells"])
    expected = {(fold, lag, arm) for fold in (0, 1) for lag in (1, 16) for arm in ("A", "B", "C")}
    observed = {(cell.get("fold"), cell.get("lag"), cell.get("arm")) for cell in cells}
    if len(cells) != 12 or observed != expected:
        raise ValueError("execution recipe does not contain the frozen 12-cell panel")
    endpoints = [collect_cell(scratch, cell, runtime_pins) for cell in cells]
    return {
        "schema": "wnn.online-relationships.terminal-collection.v1",
        "captured_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": "artifact inventory only; no prediction payload hash, target, mask, grade, or quality metric read",
        "worker": worker_status,
        "observer_at_collection": observer_status,
        "execution_receipt": {
            "path": str(execution_path),
            "sha256": sha256_file(execution_path),
        },
        "recipe": {"path": str(recipe_path), "sha256": recipe_sha},
        "admission_receipt": {
            "path": str(admission_path),
            "sha256": sha256_file(admission_path),
        },
        "source_pins": {
            "admission_sha256": sha256_file(admission_path),
            "runtime": compare_source_pins(root, runtime_pins),
            "grader": compare_source_pins(root, grader_pins),
        },
        "endpoints": endpoints,
        "finished_files_present": sum(
            endpoint["state"] == "finished_files_present" for endpoint in endpoints
        ),
        "partial_or_metadata_error": sum(
            endpoint["state"] in {"partial", "present_with_metadata_errors"}
            for endpoint in endpoints
        ),
        "missing": sum(endpoint["state"] == "missing" for endpoint in endpoints),
        "independent_grader_owns_target_quality_validation": True,
    }


def main() -> int:
    """Write one create-only receipt after the pinned unit terminates."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    packet = root / PACKET
    output = args.output
    if not output.is_absolute():
        output = root / output
    output = output.resolve(strict=False)
    if not output.is_relative_to(packet.resolve()):
        raise SystemExit("receipt output must stay in the online experiment packet")
    receipt = build_receipt(root)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(receipt, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
