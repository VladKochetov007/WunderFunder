"""Explicitly source-qualified fitting-only normalization for Kalman inputs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

RAW_COLUMNS = (0, 3, 52, 55)


@dataclass(frozen=True)
class InputNormalizationPrior:
    """Explicit fitting-only, inherited-checkpoint, or identity normalization."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    source_manifest_sha256: str
    fit_members_sha256: str | None
    fit_members: tuple[tuple[str, int], ...]
    mode: str = "fit-only"

    def __post_init__(self) -> None:
        if len(self.mean) != len(RAW_COLUMNS) or len(self.scale) != len(RAW_COLUMNS):
            raise ValueError("normalization prior must cover the four selected channels")
        if any(not math.isfinite(x) for x in (*self.mean, *self.scale)):
            raise ValueError("normalization prior must be finite")
        if any(x <= 0.0 for x in self.scale):
            raise ValueError("normalization scales must be positive")
        if self.mode not in {
            "fit-only",
            "inherited-checkpoint-buffer",
            "identity-unscaled-reference",
        }:
            raise ValueError("unknown normalization provenance mode")
        if self.mode == "fit-only" and self.fit_members_sha256 is None:
            raise ValueError("fit-only normalization requires exact fitting membership")
        if self.mode != "fit-only" and self.fit_members:
            raise ValueError("non-fit-only normalization cannot claim fitting membership")
        for digest in (self.source_manifest_sha256, self.fit_members_sha256):
            if digest is None:
                continue
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("normalization provenance requires lowercase SHA-256 values")


def identity_unscaled_prior() -> InputNormalizationPrior:
    """Return an explicit raw-unit reference; this is not fitted normalization."""
    return InputNormalizationPrior(
        mean=(0.0,) * 4,
        scale=(1.0,) * 4,
        source_manifest_sha256=hashlib.sha256(b"identity-unscaled-reference-v1").hexdigest(),
        fit_members_sha256=None,
        fit_members=(),
        mode="identity-unscaled-reference",
    )


def prior_from_parent_normalization(payload: dict[str, object]) -> InputNormalizationPrior:
    """Build a prior from trusted, extracted buffers without claiming membership."""
    if payload.get("schema") != "wnn.checkpoint-normalization-extract.v1":
        raise ValueError("unsupported parent normalization schema")
    if tuple(payload.get("selected_raw_columns", ())) != RAW_COLUMNS:
        raise ValueError("parent normalization uses different raw channels")
    mean = tuple(float(value) for value in payload.get("mean", ()))
    scale = tuple(float(value) for value in payload.get("std", ()))
    source_sha = str(payload.get("source_args_split_manifest_sha256", ""))
    if len(mean) != 4 or len(scale) != 4:
        raise ValueError("parent normalization must contain four means and scales")
    return InputNormalizationPrior(
        mean=mean,
        scale=scale,
        source_manifest_sha256=source_sha,
        fit_members_sha256=None,
        fit_members=(),
        mode="inherited-checkpoint-buffer",
    )


@dataclass(frozen=True)
class FitSequence:
    """One bounded fitting-only sequence identified by source and row group."""

    split: str
    row_group: int
    state: np.ndarray


def fit_normalization_prior(
    sequences: Iterable[FitSequence],
    fit_members: tuple[tuple[str, int], ...],
    source_manifest_sha256: str,
    max_rows: int = 1_280_000,
) -> InputNormalizationPrior:
    """Stream μ/σ from exactly the caller-declared fitting members only.

    The caller supplies a bounded sequence iterator and source-qualified
    membership from a frozen manifest. No evaluation rows are read.
    """
    expected = set(fit_members)
    if not expected or len(expected) != len(fit_members) or len(expected) > 64:
        raise ValueError("fit membership must be nonempty, unique, and at most 64 groups")
    if any(split not in {"train", "valid"} or group < 0 for split, group in expected):
        raise ValueError("invalid source-qualified fitting membership")
    if len(source_manifest_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in source_manifest_sha256
    ):
        raise ValueError("source manifest SHA-256 is required")
    seen: set[tuple[str, int]] = set()
    count = 0
    mean = np.zeros(4, dtype=np.float64)
    m2 = np.zeros(4, dtype=np.float64)
    for item in sequences:
        key = (item.split, int(item.row_group))
        if key not in expected or key in seen:
            raise ValueError("iterator yielded unknown or duplicate fitting membership")
        values = np.asarray(item.state)
        if values.ndim != 2 or values.shape[1] != 112 or not 0 < len(values) <= 20_000:
            raise ValueError("fit iterator must yield one bounded Nx112 sequence")
        if count + len(values) > max_rows:
            raise ValueError("fit iterator exceeded its explicit row bound")
        selected = np.asarray(values[:, RAW_COLUMNS], dtype=np.float64)
        if not np.isfinite(selected).all():
            raise ValueError("fit input channels contain nonfinite values")
        batch_count = len(selected)
        batch_mean = selected.mean(axis=0)
        batch_m2 = np.square(selected - batch_mean).sum(axis=0)
        delta = batch_mean - mean
        combined = count + batch_count
        m2 += batch_m2 + np.square(delta) * count * batch_count / combined
        mean += delta * batch_count / combined
        count = combined
        seen.add(key)
    if seen != expected or count == 0:
        raise ValueError("fit iterator did not exactly cover the declared membership")
    scale = np.maximum(np.sqrt(m2 / count), 1.0e-6)
    encoded = json.dumps(sorted(expected), separators=(",", ":")).encode()
    return InputNormalizationPrior(
        mean=tuple(map(float, mean)),
        scale=tuple(map(float, scale)),
        source_manifest_sha256=source_manifest_sha256,
        fit_members_sha256=hashlib.sha256(encoded).hexdigest(),
        fit_members=tuple(sorted(expected)),
    )
