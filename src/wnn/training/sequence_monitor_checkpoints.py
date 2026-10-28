from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EarlyStopDecision:
    """One full-validation checkpoint decision and its resumable patience state."""

    best_wp: float
    best_epoch: int
    best_improved: bool
    reference_wp: float
    delta_improved: bool
    bad_checks: int
    should_stop: bool


def advance_early_stop(
    history: list[dict[str, object]],
    completed_epochs: int,
    valid_wp: float,
    policy: Mapping[str, object],
) -> EarlyStopDecision:
    """Update strict-best and min-delta patience from one completed full-valid check."""
    min_epoch, patience, min_delta = _policy(policy)
    if completed_epochs != len(history) + 1 or completed_epochs < 1:
        raise ValueError("early-stop epoch must extend the complete validation history")
    if not math.isfinite(valid_wp):
        raise ValueError("early-stop full-valid WP must be finite")
    if history:
        previous = history[-1]
        previous_best = _history_number(previous, "early_stop_best_raw_wp")
        previous_best_epoch = _history_integer(previous, "early_stop_best_epoch")
        previous_reference = _history_number(previous, "early_stop_patience_reference_wp")
        previous_bad_checks = _history_integer(previous, "early_stop_bad_checks")
    else:
        previous_best = None
        previous_best_epoch = 0
        previous_reference = None
        previous_bad_checks = 0

    best_improved = previous_best is None or valid_wp > previous_best
    best_wp = valid_wp if best_improved else float(previous_best)
    best_epoch = completed_epochs if best_improved else previous_best_epoch
    delta_improved = previous_reference is None or valid_wp > previous_reference + min_delta
    reference_wp = valid_wp if delta_improved else float(previous_reference)
    bad_checks = 0 if delta_improved else previous_bad_checks + 1
    should_stop = completed_epochs >= min_epoch and bad_checks >= patience
    return EarlyStopDecision(
        best_wp=best_wp,
        best_epoch=best_epoch,
        best_improved=best_improved,
        reference_wp=reference_wp,
        delta_improved=delta_improved,
        bad_checks=bad_checks,
        should_stop=should_stop,
    )


def write_checkpoint_pointers(
    output: Path,
    run_identity_sha256: str,
    status: str,
    history: list[dict[str, object]],
    epoch_index: Mapping[str, object],
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Atomically publish best and last immutable epoch references for monitoring."""
    if not history:
        raise ValueError("checkpoint pointers require at least one full-valid epoch")
    entries = epoch_index.get("entries")
    if not isinstance(entries, list) or len(entries) != len(history):
        raise ValueError("epoch index must cover every evaluated checkpoint")
    last_history = history[-1]
    best_epoch = _history_integer(last_history, "early_stop_best_epoch")
    if not 1 <= best_epoch <= len(entries):
        raise ValueError("best checkpoint epoch is outside the retained prefix")
    best_entry, last_entry = entries[best_epoch - 1], entries[-1]
    if not isinstance(best_entry, Mapping) or not isinstance(last_entry, Mapping):
        raise ValueError("epoch index contains malformed checkpoint entries")
    best = _pointer_entry(
        best_entry, best_epoch, _history_number(last_history, "early_stop_best_raw_wp")
    )
    last = _pointer_entry(
        last_entry,
        len(history),
        _history_number(last_history, "valid_wp"),
    )
    min_epoch, patience, min_delta = _policy(policy)
    document: dict[str, object] = {
        "schema": "wnn.sequence-monitor-checkpoints.v1",
        "run_identity_sha256": run_identity_sha256,
        "status": status,
        "completed_epochs": len(history),
        "monitor_metric": "full official-valid pooled WP",
        "best_checkpoint_policy": "strict highest valid WP; report-only validation is exposed",
        "early_stopping": {
            "min_epoch": min_epoch,
            "patience": patience,
            "min_delta": min_delta,
            "bad_count": _history_integer(last_history, "early_stop_bad_checks"),
            "patience_reference_wp": _history_number(
                last_history, "early_stop_patience_reference_wp"
            ),
            "min_delta_applies_to": "patience reset only; best pointer uses strict raw WP",
            "patience_counting": (
                "consecutive full-valid checks from the first check; stopping is eligible "
                "after min_epoch"
            ),
        },
        "best_raw_wp": _history_number(last_history, "early_stop_best_raw_wp"),
        "best": best,
        "last": last,
    }
    _atomic_replace_json(output / "checkpoint_pointers.json", document)
    return document


def _policy(policy: Mapping[str, object]) -> tuple[int, int, float]:
    min_epoch = policy.get("min_epoch")
    patience = policy.get("patience")
    min_delta = policy.get("min_delta")
    if type(min_epoch) is not int or min_epoch < 1:
        raise ValueError("early-stop min_epoch must be a positive integer")
    if type(patience) is not int or patience < 1:
        raise ValueError("early-stop patience must be a positive integer")
    if (
        isinstance(min_delta, bool)
        or not isinstance(min_delta, (int, float))
        or not math.isfinite(min_delta)
        or min_delta < 0
    ):
        raise ValueError("early-stop min_delta must be finite and nonnegative")
    return min_epoch, patience, float(min_delta)


def _history_number(history: Mapping[str, object], key: str) -> float:
    value = history.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"resume history is missing finite {key}")
    return float(value)


def _history_integer(history: Mapping[str, object], key: str) -> int:
    value = history.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"resume history is missing nonnegative {key}")
    return value


def _pointer_entry(entry: Mapping[str, object], epoch: int, valid_wp: float) -> dict[str, object]:
    path, digest = entry.get("checkpoint_path"), entry.get("checkpoint_sha256")
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("epoch index checkpoint pointer is malformed")
    return {"epoch": epoch, "valid_wp": valid_wp, "path": path, "sha256": digest}


def _atomic_replace_json(path: Path, value: Mapping[str, object]) -> None:
    """Replace the small current pointer without exposing a partial JSON document."""
    if path.is_symlink() or (path.exists() and not stat.S_ISREG(path.lstat().st_mode)):
        raise ValueError("checkpoint pointer must be a regular non-symlink file")
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if len(encoded) > 16_384:
        raise ValueError("checkpoint pointer document exceeds its size bound")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
