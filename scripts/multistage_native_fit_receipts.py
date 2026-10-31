from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wnn.multistage_search import file_sha256, read_json


def regular_file(path: Path, label: str, max_bytes: int = 64 * 1024 * 1024) -> Path:
    """Resolve one bounded regular file without following a final symlink."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"{label} must be a bounded regular file")
    return path.resolve(strict=True)


def read_jsonl(path: Path, label: str, max_bytes: int) -> list[dict[str, Any]]:
    """Read a bounded complete JSONL journal and reject partial or oversized lines."""
    regular_file(path, label, max_bytes)
    rows = []
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if len(raw) > 131_072 or not raw.endswith(b"\n"):
                raise ValueError(f"{label} line {number} is partial or oversized")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{label} line {number} is not an object")
            rows.append(value)
    return rows


def verify_completion_receipt(
    fit_dir: Path,
    *,
    child_receipt: Path | None,
    terminal_attestation: Path | None,
    terminal: dict[str, Any],
    index: dict[str, Any],
    model_sha256: str,
    index_file_sha256: str,
    terminal_file_sha256: str,
    stage: int,
    trial_number: int,
) -> dict[str, Any]:
    """Require an observed successful child or a bound legacy exit-unknown attestation."""
    if (child_receipt is None) == (terminal_attestation is None):
        raise ValueError(
            "provide exactly one observed child receipt or legacy terminal attestation"
        )
    if child_receipt is not None:
        path = regular_file(child_receipt, "child receipt", 4_000_000)
        receipt = read_json(path)
        command = receipt.get("command")
        if (
            receipt.get("schema") != "wnn.multistage-owned-process.v1"
            or receipt.get("status") != "completed"
            or receipt.get("exit_code") != 0
            or receipt.get("timed_out") is not False
            or receipt.get("stage") != stage
            or receipt.get("trial_number") != trial_number
            or Path(str(receipt.get("output_dir"))).resolve() != fit_dir.resolve()
            or not isinstance(command, list)
            or "train_sequence.py" not in " ".join(str(item) for item in command)
        ):
            raise ValueError("child receipt does not prove this fit completed successfully")
        return {
            "kind": "observed_child_exit",
            "path": str(path),
            "sha256": file_sha256(path),
            "exit_code": 0,
        }

    assert terminal_attestation is not None
    path = regular_file(terminal_attestation, "legacy terminal attestation", 4_000_000)
    attestation = read_json(path)
    if (
        attestation.get("schema") != "wnn.multistage-fit-terminal.v1"
        or attestation.get("status") != "artifacts_complete_exit_unknown"
        or attestation.get("exit_code") is not None
        or attestation.get("exit_observation") != "outer_wrapper_returncode_not_persisted"
        or attestation.get("failure_logs_empty") is not True
        or attestation.get("output_dir") != str(fit_dir)
        or attestation.get("model_sha256") != model_sha256
        or attestation.get("epoch_index_sha256") != index.get("index_sha256")
        or attestation.get("epoch_index_file_sha256") != index_file_sha256
        or attestation.get("terminal_status_sha256") != terminal_file_sha256
        or attestation.get("global_updates") != terminal.get("global_updates")
    ):
        raise ValueError("legacy terminal attestation does not bind completed fit artifacts")
    return {
        "kind": "explicit_legacy_exit_unknown_attestation",
        "path": str(path),
        "sha256": file_sha256(path),
        "exit_code": None,
        "owner_receipt_sha256": attestation.get("owner_receipt_sha256"),
    }
