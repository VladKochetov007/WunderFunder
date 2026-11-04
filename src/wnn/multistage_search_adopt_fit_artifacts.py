from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import (
    atomic_write_json,
    canonical_json_sha256,
    file_lock,
    file_sha256,
    read_json,
)


def link_fit_artifacts(source: Path, destination: Path) -> None:
    """Hard-link verified artifacts and resume only from identical existing hardlinks."""
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("fit artifact source and destination cannot be symlinks")
    if not source.is_dir():
        raise FileNotFoundError("completed fit artifact source is not a directory")
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise ValueError("fit artifact destination is not a directory")
    index = read_json(source / "epoch_index.json")
    from wnn.training.sequence_epoch_prediction import validate_index

    index_sha = index.get("index_sha256")
    if not isinstance(index_sha, str):
        raise ValueError("completed fit epoch index has no self-digest")
    validate_index(index, index_sha)
    required = [
        "model.pt",
        "terminal_status.json",
        "epoch_index.json",
        "monitor_events.jsonl",
        "monitor_evaluations.jsonl",
        "metrics.jsonl",
    ]
    for name in required:
        original = source / name
        if original.is_symlink() or not original.is_file():
            raise FileNotFoundError(f"completed fit is missing a safe {name}")
        _link_or_validate_hardlink(original, destination / name)
    for path in source.glob("rank*.jsonl"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("completed fit has an unsafe rank log")
        _link_or_validate_hardlink(path, destination / path.name)
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise ValueError("completed fit epoch index is malformed")
    checkpoint_dir = destination / "epoch_checkpoints"
    if checkpoint_dir.is_symlink():
        raise ValueError("adoption epoch checkpoint directory cannot be a symlink")
    checkpoint_dir.mkdir(exist_ok=True)
    if not checkpoint_dir.is_dir():
        raise ValueError("adoption epoch checkpoint path is not a directory")
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("checkpoint_path"), str):
            raise ValueError("completed fit epoch entry is malformed")
        relative = Path(str(entry["checkpoint_path"]))
        unresolved = source / relative
        if unresolved.is_symlink():
            raise ValueError("epoch checkpoint cannot be a symlink")
        original = unresolved.resolve(strict=True)
        if not original.is_relative_to(source.resolve()):
            raise ValueError("epoch checkpoint escapes the original fit output")
        _link_or_validate_hardlink(original, destination / relative)


def _link_or_validate_hardlink(source: Path, destination: Path) -> None:
    """Create a missing hardlink or require an existing destination be that same inode."""
    if destination.is_symlink():
        raise ValueError(f"adoption artifact cannot be a symlink: {destination}")
    if destination.exists():
        if not destination.is_file() or not os.path.samefile(source, destination):
            raise ValueError(
                f"partial adoption artifact differs from its verified source: {destination}"
            )
        return
    os.link(source, destination)


def existing_adoption(
    run_dir: Path, attempt_id: str, owner: Path, terminal: Path
) -> dict[str, object] | None:
    """Return an identical completed adoption without rewriting source artifacts."""
    path = run_dir / "stage1" / "preexisting" / attempt_id / "adoption.json"
    if not path.is_file():
        return None
    value = read_json(path)
    if (
        value.get("status") != "complete"
        or value.get("original_owner_receipt_sha256") != file_sha256(owner)
        or value.get("terminal_receipt_sha256") != file_sha256(terminal)
    ):
        raise ValueError("existing adoption differs; preserve it and use a new attempt id")
    return value


def complete_preexisting_attempt(
    run_dir: Path, attempt_id: str, adoption: Mapping[str, object]
) -> None:
    """Mark its already-counted fit attempt complete after durable adoption artifacts exist."""
    with file_lock(run_dir / "budget.lock"):
        budget = read_json(run_dir / "budget.json")
        attempts = budget.get("fit_attempts")
        matches = (
            [
                row
                for row in attempts
                if isinstance(row, dict) and row.get("attempt_id") == attempt_id
            ]
            if isinstance(attempts, list)
            else []
        )
        if len(matches) != 1:
            raise ValueError("budget ledger does not have one matching preexisting fit")
        adoption_sha = canonical_json_sha256(adoption)
        if matches[0].get("status") == "complete":
            if matches[0].get("adoption_sha256") == adoption_sha:
                return
            raise ValueError("budget attempt was already completed by another adoption")
        matches[0].update(
            {
                "status": "complete",
                "adoption_path": str(
                    (run_dir / "stage1" / "preexisting" / attempt_id / "adoption.json").resolve()
                ),
                "adoption_sha256": adoption_sha,
                "trial_result_path": adoption["trial_result_path"],
                "trial_result_sha256": adoption["trial_result_sha256"],
                "completed_at_utc": adoption["completed_at_utc"],
            }
        )
        atomic_write_json(run_dir / "budget.json", budget)


def config_params(config: Mapping[str, object]) -> dict[str, object]:
    """Expose the compact model dimensions needed to describe the imported config."""
    model = config.get("model")
    return {
        "batch_size": config.get("batch_size"),
        "epochs": config.get("epochs"),
        "hidden": model.get("hidden") if isinstance(model, Mapping) else None,
        "layers": model.get("layers") if isinstance(model, Mapping) else None,
    }


def prediction_batch_size(config: Mapping[str, object]) -> int:
    """Return the exact batch size used for selected-control prediction export."""
    monitoring = config.get("monitoring")
    value = monitoring.get("eval_batch_size", 64) if isinstance(monitoring, Mapping) else 64
    if type(value) is not int or value < 1:
        raise ValueError("adopted fit prediction batch size is invalid")
    return value
