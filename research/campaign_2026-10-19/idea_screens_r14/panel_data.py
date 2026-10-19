"""Stream the fixed R3 fit panel into one preallocated sufficient-statistic design."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from attrs import frozen

from .history_features import FEATURE_WIDTH, build_a_features, fit_raw_normalizer

FIT_ROWS = 31_808


@frozen
class FitPanel:
    """Fit-only rows, targets, and normalization used by both coordinates."""

    features_a: np.ndarray
    targets: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    components: tuple[int, ...]
    memory_guards: tuple[dict[str, int | str], ...]


def validate_sample(sequences: list[dict], manifest: dict) -> tuple[int, int]:
    """Check complete-panel metadata and mapped NPY headers without target use."""
    records = manifest.get("records")
    if (
        manifest.get("split_sha256")
        != "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
    ):
        raise ValueError("R14 sample split identity differs")
    if not isinstance(records, list) or len(records) != 96 or len(sequences) != 96:
        raise ValueError("R14 requires exactly 96 complete sample sequences")
    if not manifest.get("membership_file"):
        raise ValueError("R14 sample has no bound membership file")
    roles: dict[str, set[int]] = {"fit": set(), "eval": set()}
    role_counts = {"fit": 0, "eval": 0}
    mapped_files = 0
    for record, sequence in zip(records, sequences, strict=True):
        keys = ("role", "source", "group", "component", "seq_ix")
        if any(record.get(key) != sequence.get(key) for key in keys):
            raise ValueError("R14 loaded identity differs from its sample metadata")
        role, component = str(sequence["role"]), int(sequence["component"])
        if role not in roles or component in roles[role]:
            raise ValueError("R14 role membership has an unknown role or duplicate component")
        roles[role].add(component)
        role_counts[role] += 1
        bindings = record.get("files")
        if not isinstance(bindings, dict) or set(bindings) != {"x", "y", "scored"}:
            raise ValueError("R14 sequence must bind exactly x, y, and scored NPY files")
        mapped_files += len(bindings)
        for key, shape, dtype in (
            ("x", (20_000, 112), np.dtype("float32")),
            ("y", (20_000, 2), np.dtype("float32")),
            ("scored", (20_000,), np.dtype("bool")),
        ):
            array = sequence[key]
            if np.shape(array) != shape or np.dtype(array.dtype) != dtype:
                raise ValueError(f"R14 {key} array header differs from the fixed sample")
    if role_counts != {"fit": 64, "eval": 32} or roles["fit"] & roles["eval"]:
        raise ValueError("R14 fit/eval components are incomplete or overlapping")
    if mapped_files != 288:
        raise ValueError("R14 expected exactly 288 source-bound NPY mappings")
    return mapped_files, len(roles["eval"])


def _stride_rows(sequence: dict) -> np.ndarray:
    scored = np.asarray(sequence["scored"], dtype=bool)
    if scored.ndim != 1 or len(scored) != len(sequence["x"]):
        raise ValueError("R14 sample scored mask shape differs from its sequence")
    rows = np.arange(0, len(scored), 40, dtype=np.int64)
    return rows[scored[rows]]


def prepare_fit_panel(
    sequences: list[dict], memory_guard: Callable[[str], dict[str, int | str]]
) -> FitPanel:
    """Build full-history features per fit sequence, retaining only stride rows."""
    fit_sequences = [sequence for sequence in sequences if sequence.get("role") == "fit"]
    components = tuple(sorted(int(item["component"]) for item in fit_sequences))
    if len(fit_sequences) != 64 or len(set(components)) != 64:
        raise ValueError("R14 requires one complete fit sequence per 64 components")
    guards = [memory_guard("before_fit_normalizer")]
    mean, scale, normalizer_rows = fit_raw_normalizer(fit_sequences)
    guards.append(memory_guard("after_fit_normalizer"))
    if normalizer_rows != FIT_ROWS:
        raise ValueError("R14 fit normalizer row count differs from its fixed panel")

    guards.append(memory_guard("before_preallocated_fit_design"))
    features = np.empty((FIT_ROWS, FEATURE_WIDTH), dtype=np.float64)
    targets = np.empty((FIT_ROWS, 2), dtype=np.float64)
    guards.append(memory_guard("after_preallocated_fit_design"))
    offset = 0
    for index, sequence in enumerate(fit_sequences):
        rows = _stride_rows(sequence)
        if len(rows) != 497:
            raise ValueError("R14 requires 497 scored stride-40 rows per fit sequence")
        stop = offset + len(rows)
        if stop > FIT_ROWS:
            raise ValueError("R14 selected fit rows exceed the preallocated design")
        raw_targets = np.asarray(sequence["y"])[rows]
        if raw_targets.shape != (len(rows), 2) or not np.isfinite(raw_targets).all():
            raise ValueError("R14 selected fit labels are malformed or nonfinite")
        guards.append(memory_guard(f"before_fit_sequence_history_{index}"))
        full_features = build_a_features(sequence["x"], mean, scale)
        features[offset:stop] = full_features[rows]
        targets[offset:stop] = np.asarray(raw_targets, dtype=np.float64)
        del full_features, raw_targets, rows
        offset = stop
        guards.append(memory_guard(f"after_fit_sequence_history_{index}"))
    if offset != FIT_ROWS:
        raise ValueError(f"R14 fit design has {offset} rows, expected {FIT_ROWS}")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("R14 fit design or targets are nonfinite")
    return FitPanel(features, targets, mean, scale, components, tuple(guards))
