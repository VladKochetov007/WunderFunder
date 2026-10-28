"""Produce the root-issued source identity after bounded chronological-source checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from wnn.chronological_source_audit_scan import SourceIdentity, audit_source
from wnn.chronological_source_index import (
    OFFICIAL_SOURCE_PINS,
    SEQUENCE_LENGTH,
    SOURCE_AUDIT_FIELDS,
    SOURCE_AUDIT_SCHEMA,
    SourcePin,
    SourceRowGroupReadError,
)

DETAILS_SCHEMA = "wnn.source-audit-details.v1"
DETAILS_PATH = Path("cache/source_audit_details.json")


def audit_sources(
    root: Path,
    output: Path,
) -> dict[str, object]:
    """Audit the pinned official sources and publish their root-issued receipt."""
    return _audit_sources(root, output, pins=OFFICIAL_SOURCE_PINS)


def _audit_sources(
    root: Path,
    output: Path,
    *,
    pins: Mapping[str, SourcePin],
    sequence_length: int = SEQUENCE_LENGTH,
    progress_interval: int = 64,
) -> dict[str, object]:
    """Hash and audit both official parquets, then publish the exact source-audit receipt.

    The full SHA-256 is computed from each file. The second pass reads only sequence
    identity, step, labels, and the official valid scoring mask; feature values are
    audited by the source-qualified graph pass.
    """
    root = root.resolve()
    output = output.resolve()
    details_path = root / DETAILS_PATH
    if output == details_path.resolve():
        raise ValueError("source audit and details outputs must be different files")
    if output.exists():
        raise FileExistsError(f"refusing to replace completed source audit: {output}")
    if set(pins) != {"train", "valid"}:
        raise ValueError("source audit requires exactly the official train and valid pins")
    if sequence_length <= 0 or progress_interval <= 0:
        raise ValueError("sequence length and progress interval must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details: dict[str, object] = {
        "schema": DETAILS_SCHEMA,
        "status": "running",
        "root": str(root),
        "source_audit_output": str(output),
        "sources": {},
    }
    source_details: dict[str, dict[str, object]] = {}
    details["sources"] = source_details
    _write_details(details_path, details)
    sources: dict[str, SourceIdentity] = {}

    for split in ("train", "valid"):
        source = root / "datasets" / f"{split}.parquet"
        current: dict[str, object] = {
            "source": str(source),
            "split": split,
            "status": "running",
            "phase": "starting",
            "expected_bytes": pins[split].byte_count,
            "expected_sha256": pins[split].sha256,
            "row_groups_expected": pins[split].row_group_count,
            "row_groups_scanned": 0,
        }
        source_details[split] = current
        _write_details(details_path, details)

        try:
            identity, summary = audit_source(
                source,
                split,
                pins[split],
                sequence_length,
                progress_interval,
                _progress_reporter(current, details_path, details),
            )
        except Exception as exc:
            current.update({"status": "failed", "phase": "failed"})
            failure: dict[str, object] = {
                "source": str(source),
                "split": split,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if isinstance(exc, SourceRowGroupReadError):
                failure["row_group"] = exc.row_group
            details.update({"status": "failed", "failure": failure})
            _write_details(details_path, details)
            raise
        current.update({"status": "complete", "phase": "complete", **identity, **summary})
        sources[f"datasets/{split}.parquet"] = identity
        _write_details(details_path, details)

    receipt: dict[str, object] = {"schema": SOURCE_AUDIT_SCHEMA, "sources": sources}
    if any(set(identity) != SOURCE_AUDIT_FIELDS for identity in sources.values()):
        raise RuntimeError("constructed source receipt does not match the canonical schema")
    try:
        _write_audit_once(output, receipt)
    except Exception as exc:
        details.update(
            {
                "status": "failed",
                "failure": {
                    "source": str(output),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }
        )
        _write_details(details_path, details)
        raise
    details.update({"status": "complete", "source_audit_sha256": _json_sha256(receipt)})
    _write_details(details_path, details)
    return receipt


def _progress_reporter(
    current: dict[str, object], details_path: Path, details: dict[str, object]
) -> Callable[[dict[str, int | str]], None]:
    """Bind one source's progress record to the appendable details receipt."""

    def report(progress: dict[str, int | str]) -> None:
        current.update(progress)
        _write_details(details_path, details)

    return report


def _write_details(path: Path, details: dict[str, object]) -> None:
    """Atomically replace the bounded progress/failure record."""
    _write_json_atomic(path, details, replace=True)


def _write_audit_once(path: Path, receipt: dict[str, object]) -> None:
    """Publish a complete receipt atomically without replacing an earlier audit."""
    _write_json_atomic(path, receipt, replace=False)


def _write_json_atomic(path: Path, value: dict[str, object], *, replace: bool) -> None:
    """Write canonical JSON through a same-directory temporary file and atomic link/replace."""
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_sha256(value: dict[str, object]) -> str:
    """Hash the final serialized receipt for details without changing its schema."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    encoded = (encoded + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()
