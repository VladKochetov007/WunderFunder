"""Small file and source-identity helpers for the R01 delivery recovery packet."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

PORTABLE_ONLY_FILE = "src/wnn/training/README.md"
CHECKPOINT_IDENTITY_SHA256 = "a1ed31086f6b9154e798a363d7b44daeaa6a6128f5cc4ea86741b1a4c161e537"


def sha256_file(path: Path) -> str:
    """Hash one file in bounded memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular nonsymlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read an object-valued UTF-8 JSON receipt."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular nonsymlink JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    """Create a durable JSON receipt without replacing prior recovery evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite recovery evidence: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".r01-recovery-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_under(path: Path, parent: Path, label: str) -> Path:
    """Resolve an existing path and require it to remain within its declared root."""
    resolved = path.resolve(strict=True)
    root = parent.resolve(strict=True)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes its declared root: {resolved}")
    return resolved


def producer_source_identity(portable_files: dict[str, str]) -> dict[str, Any]:
    """Derive the trainer's 60-file identity from the exact 61-file portable map."""
    if PORTABLE_ONLY_FILE not in portable_files or len(portable_files) != 61:
        raise ValueError(
            "portable a266 identity must contain its one training README and 60 code files"
        )
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for name, digest in portable_files.items()
    ):
        raise ValueError("portable a266 source map contains a malformed path or SHA-256")
    producer_files = dict(portable_files)
    producer_files.pop(PORTABLE_ONLY_FILE)
    encoded = json.dumps(producer_files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "wnn.training-source.v1",
        "source_identity_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": producer_files,
        "portable_source_identity_sha256": hashlib.sha256(
            json.dumps(portable_files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "portable_only_file": PORTABLE_ONLY_FILE,
    }


def validate_epoch_records(
    history: list[dict[str, Any]], provenance: dict[str, Any]
) -> dict[str, Any]:
    """Require six full, internally consistent fit-only epoch records and counters."""

    def require_finite(value: Any, label: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite recorded value")
        if isinstance(value, dict):
            for item in value.values():
                require_finite(item, label)
        elif isinstance(value, list):
            for item in value:
                require_finite(item, label)

    require_finite(history, "R01 fit history")
    require_finite(provenance, "R01 fit provenance")
    if len(history) != 6 or any(
        type(row.get("epoch")) is not int or row.get("epoch") != index
        for index, row in enumerate(history)
    ):
        raise ValueError("R01 recovery requires the exact six completed epoch indices 0..5")
    if any(row.get("evaluation") != "withheld" for row in history):
        raise ValueError("R01 history contains an outer evaluation during fitting")
    args = provenance.get("args", {})
    groups = provenance.get("fit_row_groups", {})
    if args.get("batch") != 8 or args.get("chunk") != 2000:
        raise ValueError("Full R01 budget requires its frozen batch8/chunk2000 recipe")
    if not isinstance(groups.get("train"), list) or not isinstance(groups.get("valid"), list):
        raise ValueError("Full R01 budget requires source-qualified fit membership")
    expected_batches = (len(groups["train"]) + len(groups["valid"])) // 8
    expected_updates = expected_batches * 10  # 20000 rows, ten chunk2000 optimizer windows.
    updates: list[int] = []
    provenance_updates = provenance.get("update_diagnostics")
    if not isinstance(provenance_updates, list) or len(provenance_updates) != 6:
        raise ValueError("provenance lacks six per-epoch update-count records")
    for index, row in enumerate(history):
        diagnostics = row.get("update_diagnostics", {})
        epoch_counts = diagnostics.get("epoch", {})
        expected = provenance_updates[index]
        if (
            not isinstance(expected, dict)
            or type(expected.get("epoch_index")) is not int
            or expected.get("epoch_index") != index
        ):
            raise ValueError("provenance update records are not in canonical epoch order")
        recorded_counts = expected.get("epoch", {})
        if not isinstance(recorded_counts, dict):
            raise ValueError("provenance epoch counter is malformed")
        keys = (
            "optimizer_attempts",
            "optimizer_completed_updates",
            "amp_skipped_updates",
            "completed_batches",
        )
        if any(type(epoch_counts.get(key)) is not int for key in keys):
            raise ValueError("history lacks exact integer optimizer/batch counters")
        if any(epoch_counts.get(key) != recorded_counts.get(key) for key in keys):
            raise ValueError("history and provenance disagree on epoch update counts")
        attempts = epoch_counts["optimizer_attempts"]
        if (
            attempts <= 0
            or attempts != expected_updates
            or epoch_counts["optimizer_completed_updates"] != attempts
            or epoch_counts["amp_skipped_updates"] != 0
            or epoch_counts["completed_batches"] != expected_batches
            or type(row.get("fit_summary", {}).get("optimizer_windows")) is not int
            or row["fit_summary"]["optimizer_windows"] != attempts
        ):
            raise ValueError(
                "an epoch is partial, skipped updates, or has inconsistent batch counts"
            )
        updates.append(attempts)
    if len(set(updates)) != 1:
        raise ValueError("full R01 epochs do not have the same canonical update budget")
    cumulative = provenance.get("cumulative_updates", {})
    cumulative_keys = (
        "optimizer_attempts",
        "optimizer_completed_updates",
        "amp_skipped_updates",
        "completed_batches",
    )
    if any(type(cumulative.get(key)) is not int for key in cumulative_keys) or any(
        cumulative.get(key) != sum(row["update_diagnostics"]["epoch"][key] for row in history)
        for key in cumulative_keys
    ):
        raise ValueError("provenance cumulative optimizer counts differ from the six epochs")
    return {"per_epoch": updates[0], "total": sum(updates), "epochs": 6}
