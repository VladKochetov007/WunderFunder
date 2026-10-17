"""Small, dependency-free helpers for traceable research run records."""

from __future__ import annotations

import json
import math
import os
import stat
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


def _json_value(value: object, label: str) -> object:
    """Copy supported JSON values while rejecting non-finite or opaque values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} mapping keys must be strings")
            result[key] = _json_value(item, f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{label}[]") for item in value]
    raise TypeError(f"{label} contains unsupported JSON value {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, object], label: str) -> Mapping[str, object]:
    """Return a recursively immutable, JSON-compatible snapshot of a mapping."""
    copied = _json_value(value, label)
    if not isinstance(copied, dict):
        raise TypeError(f"{label} must be a mapping")
    frozen = {key: _freeze_value(item) for key, item in copied.items()}
    return MappingProxyType(frozen)


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _encode_json(value: object) -> bytes:
    normalized = _json_value(value, "record")
    return (
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class RunMetadata:
    """Immutable run identity, source, configuration, and ancestry information."""

    source: Mapping[str, object] = field(default_factory=dict)
    config: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    _encoded: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Freeze metadata and encode a canonical on-disk snapshot."""
        source = _freeze_mapping(self.source, "source")
        config = _freeze_mapping(self.config, "config")
        provenance = _freeze_mapping(self.provenance, "provenance")
        record = {
            "source": _thaw_value(source),
            "config": _thaw_value(config),
            "provenance": _thaw_value(provenance),
        }
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "_encoded", _encode_json(record))

    def to_dict(self) -> dict[str, object]:
        """Return a fresh mutable copy suitable for JSON serialization."""
        decoded = json.loads(self._encoded)
        if not isinstance(decoded, dict):
            raise ValueError("Encoded run metadata is not an object")
        return decoded


def _finite_number(value: int | float, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be an integer or float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _numeric_mapping(value: Mapping[str, int | float], label: str) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} mapping keys must be strings")
        result[key] = _finite_number(item, f"{label}.{key}")
    return result


def _scalar_or_mapping(
    value: int | float | Mapping[str, int | float], label: str
) -> int | float | dict[str, int | float]:
    if isinstance(value, Mapping):
        return _numeric_mapping(value, label)
    return _finite_number(value, label)


class RunRecorder:
    """Write immutable metadata and append-only JSONL training metrics."""

    def __init__(self, run_dir: str | Path, metadata: RunMetadata) -> None:
        """Create or reopen a run directory with matching immutable metadata.

        Args:
            run_dir: Directory for ``run_metadata.json`` and ``metrics.jsonl``.
            metadata: Source, configuration, and ancestry to freeze for this run.

        Raises:
            ValueError: If the directory or existing metadata is incompatible.
            OSError: If the run directory cannot be created or opened.
        """
        self._run_dir = Path(run_dir)
        self._metadata = metadata
        self._lock = threading.Lock()
        self._run_dir.mkdir(parents=True, exist_ok=True)
        if self._run_dir.is_symlink() or not self._run_dir.is_dir():
            raise ValueError(f"Run directory must be a regular directory: {self._run_dir}")
        self._initialize_metadata()

    @property
    def metadata(self) -> RunMetadata:
        """Return the immutable metadata snapshot used for this run."""
        return self._metadata

    def _initialize_metadata(self) -> None:
        path = self._run_dir / "run_metadata.json"
        expected = self._metadata._encoded
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, read_flags)
            with os.fdopen(descriptor, "rb") as existing:
                file_stat = os.fstat(existing.fileno())
                if not stat.S_ISREG(file_stat.st_mode):
                    raise ValueError(f"Run metadata must be a regular file: {path}") from None
                observed = existing.read(len(expected) + 1)
            if observed != expected:
                raise ValueError(f"Run metadata is immutable and does not match: {path}") from None
            return
        with os.fdopen(descriptor, "wb") as output:
            output.write(expected)
            output.flush()
            os.fsync(output.fileno())

    def record_metrics(
        self,
        *,
        step: int,
        epoch: int | float,
        loss: int | float,
        learning_rates: Mapping[str, int | float] | Sequence[int | float] | None = None,
        gradient_norm: int | float | None = None,
        val_metric: int | float | Mapping[str, int | float] | None = None,
        throughput: int | float | Mapping[str, int | float] | None = None,
        timing: int | float | Mapping[str, int | float] | None = None,
    ) -> None:
        """Append one finite-valued metric row without rewriting earlier rows.

        Learning rates may be keyed by optimizer group or supplied as a list.
        Validation metrics, throughput, and timing may each be a scalar or a
        named mapping. Convert tensor or array scalars to Python numbers first.

        Args:
            step: Non-negative optimizer or batch step.
            epoch: Current epoch, including a fractional epoch if useful.
            loss: Training loss for this row.
            learning_rates: Rates by optimizer group or in optimizer order.
            gradient_norm: Optional measured gradient norm.
            val_metric: Optional validation metric or named metric mapping.
            throughput: Optional examples or tokens per second, labeled as needed.
            timing: Optional elapsed time or timing values by phase.

        Raises:
            TypeError: If a value is not a supported JSON number or mapping.
            ValueError: If a required value is invalid or non-finite.
            OSError: If the metrics file cannot be appended.
        """
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        row: dict[str, object] = {
            "step": step,
            "epoch": _finite_number(epoch, "epoch"),
            "loss": _finite_number(loss, "loss"),
        }
        if learning_rates is not None:
            if isinstance(learning_rates, Mapping):
                row["learning_rates"] = _numeric_mapping(learning_rates, "learning_rates")
            elif isinstance(learning_rates, Sequence) and not isinstance(
                learning_rates, (str, bytes)
            ):
                row["learning_rates"] = [
                    _finite_number(rate, "learning_rates[]") for rate in learning_rates
                ]
            else:
                raise TypeError("learning_rates must be a mapping or numeric sequence")
        if gradient_norm is not None:
            row["gradient_norm"] = _finite_number(gradient_norm, "gradient_norm")
        for name, value in (
            ("val_metric", val_metric),
            ("throughput", throughput),
            ("timing", timing),
        ):
            if value is not None:
                row[name] = _scalar_or_mapping(value, name)
        line = _encode_json(row)
        metrics_path = self._run_dir / "metrics.jsonl"
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        with self._lock:
            descriptor = os.open(metrics_path, flags, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError(f"Metrics log must be a regular file: {metrics_path}")
                offset = 0
                while offset < len(line):
                    written = os.write(descriptor, line[offset:])
                    if written == 0:
                        raise OSError("Metrics log write made no progress")
                    offset += written
            finally:
                os.close(descriptor)
