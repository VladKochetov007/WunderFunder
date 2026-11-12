from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.training.sequence_ensemble_config import (
    RAW_STUDENT_MODES,
    STANDALONE_LABELS_KD_MODE,
    STANDALONE_LABELS_MODE,
)


def validate_fit_reference_inputs(
    config: Mapping[str, object],
    teacher_cache: Path | None,
    reference_cache: Path | None,
    monitor_reference_root: Path | None,
) -> None:
    """Reject reference inputs that a recipe would ignore before runtime or data setup."""
    objective_mode = config.get("objective_mode")
    standalone_labels = objective_mode == STANDALONE_LABELS_MODE
    standalone_kd = objective_mode == STANDALONE_LABELS_KD_MODE
    raw_student_monitor = objective_mode in RAW_STUDENT_MODES
    single_target = config.get("supervised_target") is not None
    if raw_student_monitor and "monitoring" not in config:
        raise ValueError("standalone student objectives require report-only full-valid monitoring")
    if standalone_labels or single_target:
        recipe = "standalone labels" if standalone_labels else "single-target supervision"
        if reference_cache is not None:
            raise ValueError(
                f"{recipe} reject --reference-cache; training has no reference ancestry"
            )
        if teacher_cache is not None:
            raise ValueError(f"{recipe} reject --teacher-cache; KD and hint ancestry are off")
        if standalone_labels and monitor_reference_root is not None:
            raise ValueError(
                "standalone labels reject --monitor-reference-root; "
                "use an offline fixed readout for TL3 diagnostics"
            )
    elif standalone_kd:
        if reference_cache is not None:
            raise ValueError(
                "standalone label+KD rejects --reference-cache; training has no blend-reference ancestry"
            )
        if teacher_cache is None:
            raise ValueError("standalone label+KD requires a validated fit-only --teacher-cache")
        if monitor_reference_root is not None:
            raise ValueError(
                "standalone label+KD rejects --monitor-reference-root; "
                "monitoring reports raw student z"
            )
    if not raw_student_monitor and ("monitoring" in config) != (monitor_reference_root is not None):
        raise ValueError("monitoring requires --monitor-reference-root, omitted for old recipes")
