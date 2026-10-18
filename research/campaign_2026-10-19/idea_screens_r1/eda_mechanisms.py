"""Bounded input diagnostics and target-operator screens."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import FEATURE_WIDTH, MAX_ROWS, MAX_SEQUENCES
from .input_diagnostics import input_summary
from .target_operator import target_screens


def _label(value: object) -> str | int | float | bool | None:
    """Return a compact JSON-safe sequence identifier."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _validate(sequence: dict[str, object], index: int) -> dict[str, object]:
    """Validate shapes while leaving evaluation labels and masks unread."""
    try:
        x = np.asarray(sequence["x"], dtype=np.float32)
        raw_y, raw_scored = sequence["y"], sequence["scored"]
    except KeyError as error:
        raise ValueError(f"sequence {index} is missing {error.args[0]!r}") from error
    if x.ndim != 2 or x.shape[1] != FEATURE_WIDTH or not 0 < len(x) <= MAX_ROWS:
        raise ValueError(f"sequence {index} x must have shape (1..{MAX_ROWS},112)")
    role = sequence.get("role")
    if role not in ("fit", "eval"):
        raise ValueError(f"sequence {index} role must be 'fit' or 'eval'")
    if np.shape(raw_y) != (len(x), 2) or np.shape(raw_scored) != (len(x),):
        raise ValueError(f"sequence {index} y/scored must have shapes (T,2)/(T,)")
    metadata = {key: _label(sequence.get(key)) for key in ("source", "group", "component")}
    if any(metadata[key] is None for key in metadata):
        raise ValueError(f"sequence {index} requires source, group, and component")
    result: dict[str, object] = {"x": x, "role": role, **metadata}
    if role == "fit":
        result["y"] = np.asarray(raw_y, dtype=np.float32)
        result["scored"] = np.asarray(raw_scored, dtype=bool)
    return result


def _median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def _role_summary(records: list[dict[str, object]], role: str) -> dict[str, object]:
    """Aggregate input-only values by role with realized sample counts."""
    selected = [record for record in records if record["role"] == role]
    inputs = [record["input_only"] for record in selected]
    level_ratios = [
        item["trade_to_quote_magnitude_ratios"]["i0_dp_to_i0_p_level"] for item in inputs
    ]
    return {
        "sequences": len(selected),
        "input_rows": sum(item["rows"] for item in inputs),
        "median_fully_finite_row_fraction": _median(
            np.asarray([item["fully_finite_feature_row_fraction"] for item in inputs])
        ),
        "median_i0_volume_positive_row_fraction": _median(
            np.asarray(
                [
                    item["volume_blocks"]["i0_v"]["strictly_positive_complete_row_fraction"]
                    for item in inputs
                ]
            )
        ),
        "median_i1_volume_positive_row_fraction": _median(
            np.asarray(
                [
                    item["volume_blocks"]["i1_v"]["strictly_positive_complete_row_fraction"]
                    for item in inputs
                ]
            )
        ),
        "median_i0_trade_to_quote_level_ratio": _median(
            np.asarray([v for v in level_ratios if v is not None])
        ),
    }


def run(sequences: list[dict[str, object]], output: Path) -> dict[str, object]:
    """Run fixed, bounded screens and create a JSON result without overwriting.

    Args:
        sequences: At most48 input sequences with x, y, scored, and source metadata.
        output: New JSON file path. Existing paths are rejected.

    Returns:
        The JSON-serializable result also written to ``output``.

    Raises:
        ValueError: If a sequence has an invalid schema, role, or size.
        FileExistsError: If the output path already exists.
    """
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    if not sequences or len(sequences) > MAX_SEQUENCES:
        raise ValueError(f"sequences must contain 1..{MAX_SEQUENCES} items")
    validated = [_validate(sequence, index) for index, sequence in enumerate(sequences)]
    records, fit_sequences = [], []
    for sequence in validated:
        role = str(sequence["role"])
        item: dict[str, object] = {
            "role": role,
            "source": sequence["source"],
            "group": sequence["group"],
            "component": sequence["component"],
            "input_only": input_summary(sequence["x"]),
        }
        if role == "fit":
            y, scored = sequence["y"], sequence["scored"]
            item["fit_label_rows"] = int((scored & np.isfinite(y).all(axis=1)).sum())
            fit_sequences.append(sequence)
        else:
            item["fit_label_rows"] = None
        records.append(item)
    result = {
        "schema": "idea_screens_r1.eda_mechanisms.v1",
        "sequence_count": len(records),
        "input_only_by_role": {role: _role_summary(records, role) for role in ("fit", "eval")},
        "per_sequence": records,
        "target_operator": target_screens(fit_sequences),
        "screened_cards": {
            "1": {"status": "target_operator_fit_diagnostic"},
            "3": {"status": "descriptive_premise_only"},
            "4": {"status": "descriptive_premise_only"},
            "5": {"status": "unsupported"},
            "7": {"status": "descriptive_premise_only"},
            "11": {"status": "descriptive_premise_only"},
            "14": {"status": "descriptive_premise_only"},
            "15": {"status": "NOT_TESTED_hiddenstate_input_innovation_descriptive_only"},
        },
        "not_screened_cards": [2, 6, 8, 9, 10, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return result
