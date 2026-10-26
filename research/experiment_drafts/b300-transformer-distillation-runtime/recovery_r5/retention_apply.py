"""Apply or verify one explicit, hash-bound R5 post-selection retention plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import tempfile
from pathlib import Path
from typing import Any  # Validated manifests contain heterogeneous JSON fields.

from epoch_contract import file_sha256, safe_file
from retention_contract import _load_plan
from retention_receipts import (
    _complete,
    _intent,
    _verify_receipts,
    verify_pair_retention,
    verify_retention,
)
from retention_verify import _verify_plan_state, _verify_removed_absent

__all__ = ["apply_retention_plan", "verify_retention"]

_TEMP_RECEIPT = re.compile(r"\.retention_(?:intent|complete)\.json\.[A-Za-z0-9_-]+\.partial")


def _fsync_directory(path: Path) -> None:
    """Make a newly linked or unlinked receipt/checkpoint durable on Linux."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable(path: Path, value: dict[str, Any]) -> str:
    """Atomically write and directory-fsync one create-once authorization receipt."""
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if path.is_symlink():
        raise ValueError(f"existing retention receipt is a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing retention receipt differs: {path}")
        return file_sha256(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _remove_stale_receipt_temps(plan: dict[str, Any]) -> None:
    """Remove only atomic-writer partials after plan bytes and both intents verify."""
    for fold in plan["folds"]:
        root = Path(fold["fit_dir"])
        for path in root.iterdir():
            if _TEMP_RECEIPT.fullmatch(path.name) is None:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("stale retention receipt temp is not a regular file")
            path.unlink()
        _fsync_directory(root)


def _delete_checkpoint_files(plan: dict[str, Any]) -> None:
    """Unlink only the plan's explicit, rehashed nonselected checkpoint paths."""
    for fold in plan["folds"]:
        fit_dir = Path(fold["fit_dir"])
        checkpoint_dir = fit_dir / "epoch_checkpoints"
        for entry in fold["remove"]:
            candidate = fit_dir / entry["path"]
            if candidate.is_symlink():
                raise ValueError("deletion target became a symlink")
            if not candidate.exists():
                continue
            path = safe_file(fit_dir, entry["path"])
            if path.parent != checkpoint_dir or path.suffix != ".pt":
                raise ValueError("deletion target is outside the exact epoch checkpoint directory")
            if path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"deletion target changed after intent: {entry['path']}")
            path.unlink()
            _fsync_directory(checkpoint_dir)


def apply_retention_plan(plan_path: Path, plan_sha256: str) -> dict[str, Any]:
    """Persist paired intent, then idempotently prune only verified nonselected snapshots."""
    if platform.system() != "Linux":
        raise RuntimeError("R5 checkpoint retention apply is supported only on Linux")
    if plan_path.is_symlink():
        raise ValueError("retention plan must not be a symlink")
    plan_path = plan_path.expanduser().resolve(strict=True)
    plan, raw_sha256 = _load_plan(plan_path, plan_sha256)
    intents, _ = _verify_receipts(plan, raw_sha256, plan_path)
    allow_missing = intents == len(plan["folds"])
    if intents == 1 and any(
        not (Path(fold["fit_dir"]) / item["path"]).exists()
        for fold in plan["folds"]
        for item in fold["remove"]
    ):
        raise ValueError("checkpoint disappeared before both fit intents were durable")
    _verify_plan_state(plan, allow_missing=allow_missing)
    for fold in plan["folds"]:
        _publish_immutable(
            Path(fold["fit_dir"]) / "retention_intent.json",
            _intent(plan, raw_sha256, fold, plan_path),
        )
    _remove_stale_receipt_temps(plan)
    _delete_checkpoint_files(plan)
    _verify_plan_state(plan, allow_missing=True)
    _verify_removed_absent(plan)
    for fold in plan["folds"]:
        _publish_immutable(
            Path(fold["fit_dir"]) / "retention_complete.json",
            _complete(plan, raw_sha256, fold, plan_path),
        )
    _remove_stale_receipt_temps(plan)
    _verify_receipts(plan, raw_sha256, plan_path)
    return {
        "status": "completed",
        "plan_sha256": raw_sha256,
        "selected_completed_epochs": plan["selected_completed_epochs"],
    }


def main() -> None:
    """Apply or verify cleanup only when the exact reviewed plan hash is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    result = (
        apply_retention_plan(args.plan, args.plan_sha256)
        if args.apply
        else verify_pair_retention(args.plan, args.plan_sha256)
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
